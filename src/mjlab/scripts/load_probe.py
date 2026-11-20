"""Measure peak foot impacts and joint loads of a C1 policy, for mechanical design.

Usage: uv run src/mjlab/scripts/load_probe.py <ckpt> [speed ...] [--envs N]
  [--steps N]

For each commanded forward speed (default 1.0 2.0 3.0 3.5), runs N parallel envs
(default 64) with the deterministic policy and records, at every physics substep:
  - foot ground reaction force (net force on each foot, in N and body weights)
  - foot speed (total and downward) just before touchdown, which does not
    depend on the contact model
  - base specific force, i.e. what an accelerometer on the base reads, in g
  - actuator torque, the unclipped PD demand, and the fraction of time the motor
    is saturated against the X8-120 speed-torque curve and torque cap
  - joint velocity and mechanical power per joint
  - positive mechanical work per meter and cost of transport
Recording per substep (not per policy step) catches touchdown spikes that last a
few milliseconds. Steps right after a reset are skipped. Terminations stay on, so
falls are counted; push disturbances are off (play config).

Writes a CSV summary next to the checkpoint.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.asset_zoo.robots.caltech_c1 import c1_constants as c1
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.c1 import FEET
from mjlab.tasks.velocity.config.c1.checkpoint_actuation import apply_trained_actuation
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK = "Mjlab-Velocity-Flat-C1"
SETTLE_S = 1.0
G = 9.81
DEVICE = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=Path)
parser.add_argument("speeds", type=float, nargs="*", default=[1.0, 2.0, 3.0, 3.5])
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
args = parser.parse_args()

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = args.envs
trained = apply_trained_actuation(env_cfg, args.ckpt)
twist_cfg = env_cfg.commands["twist"]
assert isinstance(twist_cfg, UniformVelocityCommandCfg)
twist_cfg.rel_forward_envs = 0.0  # keep fixed commands fixed (no |vx| flip)
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(
  ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE), clip_actions=agent_cfg.clip_actions
)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(str(args.ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(device=DEVICE)

base = env.unwrapped
robot = base.scene["robot"]
feet: ContactSensor = base.scene[FEET]
joint_names = robot.joint_names
mass = float(base.sim.mj_model.body_mass.sum())
weight = mass * G
sub_dt = base.physics_dt
settle_steps = int(SETTLE_S / base.step_dt)
# Both the contact sensor primaries and the robot sites are ordered (left, right).
assert robot.site_names == ("left_foot_contact", "right_foot_contact")

# Actuators are ordered by actuator group (hips, then knees), not by joint, so
# reorder every actuator-space quantity into joint order.
act_perm = [robot.actuator_names.index(name) for name in joint_names]
# The X8-120 model is a PD law clipped by the speed-torque curve and torque cap.
# Recompute the unclipped PD demand and the available torque at each speed.
kp_list, kd_list = trained.joint_gains(joint_names)
kp = torch.tensor(kp_list, device=DEVICE)
kd = torch.tensor(kd_list, device=DEVICE)


def available_torque(demand: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
  """Most torque the motor can give in the demanded direction at this speed:
  the speed-torque curve when driving the motion, the cap when braking it."""
  curve = c1.VOLTAGE_LINE_STALL_TORQUE * (1.0 - vel.abs() / c1.NO_LOAD_SPEED)
  driving = demand * vel > 0
  return torch.where(driving, curve.clamp(0.0, c1.TORQUE_CAP), c1.TORQUE_CAP)


class Recorder:
  """Collects per-substep samples from envs past the settle period."""

  def __init__(self) -> None:
    self.clear()

  def clear(self) -> None:
    self.tau: list[torch.Tensor] = []
    self.tau_demand: list[torch.Tensor] = []
    self.sat: list[torch.Tensor] = []
    self.vel: list[torch.Tensor] = []
    self.foot_force: list[torch.Tensor] = []
    self.touchdown_speed: list[torch.Tensor] = []
    self.touchdown_vz: list[torch.Tensor] = []
    self.base_accel: list[torch.Tensor] = []
    self.pos_work = 0.0  # J, summed over joints, envs and substeps
    self.distance = 0.0  # m, summed over envs
    self.prev_found: torch.Tensor | None = None
    self.prev_foot_vel: torch.Tensor | None = None
    self.prev_base_vel: torch.Tensor | None = None

  def record(self) -> None:
    settled = base.episode_length_buf >= settle_steps
    # Joint loads.
    tau = robot.data.actuator_force[:, act_perm]
    vel = robot.data.joint_vel
    demand = kp * (robot.data.joint_pos_target - robot.data.joint_pos) - kd * vel
    # Saturated: the motor gives all it can at this speed, in the demanded direction.
    sat = (tau.abs() >= 0.98 * available_torque(demand, vel)) & (
      demand.abs() > tau.abs() + 1e-3
    )
    power = tau * vel
    self.tau.append(tau[settled].clone())
    self.tau_demand.append(demand[settled].clone())
    self.sat.append(sat[settled].clone())
    self.vel.append(vel[settled].clone())
    self.pos_work += float(power[settled].clamp(min=0).sum()) * sub_dt
    base_vel = robot.data.root_link_lin_vel_w.clone()
    self.distance += float(base_vel[settled, 0].sum()) * sub_dt
    # Feet: force this substep, and the speed just before each new contact.
    assert feet.data.force is not None and feet.data.found is not None
    self.foot_force.append(feet.data.force[settled].norm(dim=-1).flatten())
    found = feet.data.found > 0
    foot_vel = robot.data.site_lin_vel_w.clone()
    if self.prev_found is not None and self.prev_foot_vel is not None:
      touchdown = found & ~self.prev_found & settled[:, None]
      self.touchdown_speed.append(self.prev_foot_vel.norm(dim=-1)[touchdown])
      self.touchdown_vz.append(-self.prev_foot_vel[..., 2][touchdown])
    # Base: an accelerometer reads acceleration minus gravity.
    if self.prev_base_vel is not None:
      accel = (base_vel - self.prev_base_vel) / sub_dt
      accel[:, 2] += G
      self.base_accel.append(accel[settled].norm(dim=-1) / G)
    self.prev_found, self.prev_foot_vel, self.prev_base_vel = (
      found,
      foot_vel,
      base_vel,
    )


rec = Recorder()
scene_update = base.scene.update


def recording_update(dt: float) -> None:
  # Runs once per physics substep, after sim.step, so sensors are fresh here.
  scene_update(dt)
  rec.record()


base.scene.update = recording_update  # type: ignore[method-assign]


def q(x: torch.Tensor, p: float) -> float:
  return float(torch.quantile(x.float(), p)) if x.numel() else float("nan")


def peak(x: torch.Tensor) -> float:
  return float(x.max()) if x.numel() else float("nan")


rows: list[dict] = []


def add_row(speed: float, actual: float, falls: int, name: str, joint: str, x):
  rows.append(
    {
      "speed_cmd": speed,
      "speed_actual": round(actual, 3),
      "falls": falls,
      "quantity": name,
      "joint": joint,
      "peak": round(peak(x), 3),
      "p999": round(q(x, 0.999), 3),
      "p99": round(q(x, 0.99), 3),
      "rms": round(float(x.float().square().mean().sqrt()), 3),
    }
  )


print(f"robot mass {mass:.1f} kg, body weight {weight:.0f} N")
for speed in args.speeds:
  twist_cfg.ranges.lin_vel_x = (speed, speed)
  with torch.inference_mode():
    obs = env.reset()
  obs = obs[0] if isinstance(obs, tuple) else obs
  rec.clear()
  falls = 0
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, dones, _ = env.step(policy(obs))
      falls += int(dones.sum())
  tau = torch.cat(rec.tau)
  demand = torch.cat(rec.tau_demand)
  saturated = torch.cat(rec.sat)
  vel = torch.cat(rec.vel)
  power = (tau * vel).abs()
  f = torch.cat(rec.foot_force)
  td = torch.cat(rec.touchdown_speed)
  tdz = torch.cat(rec.touchdown_vz)
  acc = torch.cat(rec.base_accel)
  sim_time = tau.shape[0] * sub_dt  # env-seconds of settled data
  actual = rec.distance / sim_time
  work_per_m = rec.pos_work / abs(rec.distance)
  cot = work_per_m / weight
  mean_power = rec.pos_work / sim_time

  print(f"\ncmd {speed:.2f} m/s  actual {actual:.2f}  falls {falls}")
  print(
    f"  foot force        peak {peak(f):6.0f} N ({peak(f) / weight:.1f} BW)  "
    f"p99.9 {q(f, 0.999):6.0f} N  p99 {q(f, 0.99):6.0f} N"
  )
  print(
    f"  touchdown speed   peak {peak(td):6.2f} m/s  p99 {q(td, 0.99):5.2f}  "
    f"median {q(td, 0.5):5.2f}  ({td.numel()} touchdowns)"
  )
  print(
    f"  touchdown vz down peak {peak(tdz):6.2f} m/s  p99 {q(tdz, 0.99):5.2f}  "
    f"median {q(tdz, 0.5):5.2f}"
  )
  print(
    f"  base accel        peak {peak(acc):6.1f} g  p99.9 {q(acc, 0.999):5.1f} g  "
    f"p99 {q(acc, 0.99):5.1f} g"
  )
  print(
    f"  energy            {mean_power:6.0f} W positive mech power, "
    f"{work_per_m:5.0f} J/m, CoT {cot:.2f}"
  )
  print(
    f"  {'joint':14s} {'tau pk':>6} {'rms':>5} {'%sat':>5} | "
    f"{'demand pk':>9} {'p99.9':>6} {'p99':>5} | {'vel pk':>6} {'p99':>5} | "
    f"{'pwr pk':>6} {'p99':>5}"
  )
  for j, name in enumerate(joint_names):
    t, dm, v, p = tau[:, j].abs(), demand[:, j].abs(), vel[:, j].abs(), power[:, j]
    sat = saturated[:, j].float().mean()
    print(
      f"  {name:14s} {peak(t):6.1f} {t.square().mean().sqrt():5.1f} {sat:5.1%} | "
      f"{peak(dm):9.0f} {q(dm, 0.999):6.0f} {q(dm, 0.99):5.0f} | "
      f"{peak(v):6.2f} {q(v, 0.99):5.2f} | {peak(p):6.0f} {q(p, 0.99):5.0f}"
    )
    for quantity, x in (
      ("torque_Nm", t),
      ("torque_demand_Nm", dm),
      ("joint_vel_rad_s", v),
      ("power_W", p),
    ):
      add_row(speed, actual, falls, quantity, name, x)
  add_row(speed, actual, falls, "foot_force_N", "feet", f)
  add_row(speed, actual, falls, "touchdown_speed_m_s", "feet", td)
  add_row(speed, actual, falls, "touchdown_vz_down_m_s", "feet", tdz)
  add_row(speed, actual, falls, "base_accel_g", "base", acc)
  for quantity, value in (
    ("mean_pos_mech_power_W", mean_power),
    ("pos_work_per_m_J", work_per_m),
    ("cost_of_transport", cot),
  ):
    add_row(speed, actual, falls, quantity, "all", torch.tensor([value]))

env.close()
out = args.ckpt.parent / f"loads_{args.ckpt.stem}.csv"
with out.open("w", newline="") as fh:
  writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
print(f"\nsaved {out}")
