"""Replay C1 gaits through inverse dynamics with modified leg mass distributions.

Usage:
  uv run src/mjlab/scripts/inertia_replay.py <ckpt> [speed ...] [--envs N]
    [--steps N] [--variant NAME=EDIT,EDIT ...]

Records joint trajectories of the deterministic policy on the GPU, then, on the CPU,
runs MuJoCo inverse dynamics on the same trajectories with a modified model. This
answers "same motion, different leg -> how much joint torque?" without retraining.

Each variant torque is the recorded actuator torque plus the change in M(q) qacc +
bias(q, qvel) between variant and baseline. Ground forces are held fixed, so no
contact inversion is needed. As a check, swing-leg joints (which feel no contact
force) must satisfy M qacc + bias = actuator + passive torque; the error prints.

Edits apply to both legs. Bodies: thigh, shin, foot.
  shin*0.5           scale the body's mass and rotational inertia by 0.5
  thigh+-1.0@-0.40   add a point mass of -1.0 kg at local z = -0.40 m (removal)
  thigh+1.0@-0.05    add 1.0 kg at local z = -0.05 m
Example, knee motor moved from the knee to the hip on the thigh:
  --variant knee_motor_up=thigh+-1.0@-0.42,thigh+1.0@-0.05

Caveats: stance phases are approximate. The motion is held fixed, so a heavier or
lighter leg would also need different ground forces; the "base residual" column is
the phantom force on the unactuated base needed to keep the recorded motion, a
measure of how far the variant is from dynamically consistent. Swing-phase torques
are exact. A retrained policy would also change its gait to exploit the new leg.
"""

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.asset_zoo.robots.caltech_c1 import c1_constants as c1
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.c1.checkpoint_actuation import apply_trained_actuation
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK = "Mjlab-Velocity-Flat-C1"
SETTLE_S = 1.0
DEVICE = "cuda:0"
JOINTS = ("L_hip_joint", "L_knee_joint", "R_hip_joint", "R_knee_joint")
FOOT_GEOMS = {"L": "L_foot_geom", "R": "R_foot_geom"}
DEFAULT_POSE = {"hip": -0.498, "knee": 0.80}
DEFAULT_VARIANTS = [
  "foot_half=foot*0.5",
  "shin_half=shin*0.5",
  "shin_foot_half=shin*0.5,foot*0.5",
  "thigh_half=thigh*0.5",
]

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=Path)
parser.add_argument("speeds", type=float, nargs="*", default=[1.0, 3.5])
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--variant", action="append", default=None)
args = parser.parse_args()
variant_specs = args.variant or DEFAULT_VARIANTS

# ---------------------------------------------------------------------------
# Record trajectories on the GPU.
# ---------------------------------------------------------------------------
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = args.envs
apply_trained_actuation(env_cfg, args.ckpt)
# Per-env randomization would make the CPU model differ from the simulated one.
for name in ("foot_friction", "base_com", "encoder_bias"):
  env_cfg.events.pop(name, None)
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
settle_steps = int(SETTLE_S / base.step_dt)
sim_step = base.sim.step
buf: dict[str, list[torch.Tensor]] = {}


def recording_step() -> None:
  # qacc and qfrc_actuator computed during a step belong to the pre-step state.
  data = base.sim.data
  settled = base.episode_length_buf >= settle_steps
  qpos, qvel = data.qpos[settled].clone(), data.qvel[settled].clone()
  sim_step()
  buf["qpos"].append(qpos)
  buf["qvel"].append(qvel)
  buf["qacc"].append(data.qacc[settled].clone())
  buf["qfrc_act"].append(data.qfrc_actuator[settled].clone())


base.sim.step = recording_step  # type: ignore[method-assign]

traj: dict[float, dict[str, np.ndarray]] = {}
for speed in args.speeds:
  twist_cfg.ranges.lin_vel_x = (speed, speed)
  buf = {k: [] for k in ("qpos", "qvel", "qacc", "qfrc_act")}
  with torch.inference_mode():
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    for _ in range(args.steps):
      obs, _, _, _ = env.step(policy(obs))
  traj[speed] = {k: torch.cat(v).cpu().double().numpy() for k, v in buf.items()}

model0 = copy.copy(base.sim.mj_model)
env.close()

# ---------------------------------------------------------------------------
# Model edits.
# ---------------------------------------------------------------------------


def body_ids(m: mujoco.MjModel, part: str) -> list[int]:
  return [m.body(f"robot/{side}_{part}_link").id for side in "LR"]


def inertia_tensor(m: mujoco.MjModel, b: int) -> np.ndarray:
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, m.body_iquat[b])
  rot = rot.reshape(3, 3)
  return rot @ np.diag(m.body_inertia[b]) @ rot.T


def set_inertia_tensor(m: mujoco.MjModel, b: int, tensor: np.ndarray) -> None:
  vals, vecs = np.linalg.eigh(tensor)
  if np.linalg.det(vecs) < 0:
    vecs[:, 0] *= -1
  assert (vals > 0).all(), "edit leaves a body with non-positive inertia"
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, vecs.flatten())
  m.body_inertia[b] = vals
  m.body_iquat[b] = quat


def add_point_mass(m: mujoco.MjModel, b: int, mass: float, z: float) -> None:
  """Add (or remove, if negative) a point mass at local (0, 0, z) on body b."""
  m0, c0, i0 = m.body_mass[b], m.body_ipos[b].copy(), inertia_tensor(m, b)
  p = np.array([0.0, 0.0, z])
  m1 = m0 + mass
  assert m1 > 0, "edit leaves a body with non-positive mass"
  c1 = (m0 * c0 + mass * p) / m1

  def shift(mm: float, r: np.ndarray) -> np.ndarray:
    return mm * (r @ r * np.eye(3) - np.outer(r, r))

  # Inertia about the new COM: old body shifted, plus the point mass.
  i1 = i0 + shift(m0, c0 - c1) + shift(mass, p - c1)
  m.body_mass[b] = m1
  m.body_ipos[b] = c1
  set_inertia_tensor(m, b, i1)


def make_variant(spec: str) -> tuple[str, mujoco.MjModel]:
  name, edits = spec.split("=", 1)
  m = copy.copy(model0)
  for edit in edits.split(","):
    if "*" in edit:
      part, factor = edit.split("*")
      for b in body_ids(m, part):
        m.body_mass[b] *= float(factor)
        m.body_inertia[b] *= float(factor)
    else:
      part, rest = edit.split("+", 1)
      mass, z = rest.split("@")
      for b in body_ids(m, part):
        add_point_mass(m, b, float(mass), float(z))
  return name, m


# ---------------------------------------------------------------------------
# Inverse dynamics.
# ---------------------------------------------------------------------------
dof = [model0.joint(f"robot/{j}").dofadr[0] for j in JOINTS]
qadr = [model0.joint(f"robot/{j}").qposadr[0] for j in JOINTS]
# Planar root: the first three dofs (slide x, slide z, pitch) are unactuated.
root_dofs = [0, 1, 2]


def leg_mass(m: mujoco.MjModel) -> float:
  return sum(m.body_mass[body_ids(m, p)[0]] for p in ("thigh", "shin", "foot"))


def hip_inertia(m: mujoco.MjModel) -> float:
  d = mujoco.MjData(m)
  for i, j in enumerate(JOINTS):
    d.qpos[qadr[i]] = DEFAULT_POSE["hip" if "hip" in j else "knee"]
  mujoco.mj_forward(m, d)
  full = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, d, full)
  return float(full[dof[0], dof[0]])


def dynamics(m: mujoco.MjModel, t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """M(q) qacc + bias(q, qvel) for each recorded sample, plus passive forces and
  foot contact flags. No contact inversion: ground forces are held at whatever
  produced the recorded motion, so only the mass-dependent terms change."""
  d = mujoco.MjData(m)
  n = t["qpos"].shape[0]
  rne = np.zeros((n, m.nv))
  passive = np.zeros((n, m.nv))
  contact = np.zeros((n, 2), dtype=bool)
  foot_geoms = [m.geom(f"robot/{FOOT_GEOMS[s]}").id for s in "LR"]
  out = np.zeros(m.nv)
  for i in range(n):
    d.qpos[:] = t["qpos"][i]
    d.qvel[:] = t["qvel"][i]
    mujoco.mj_fwdPosition(m, d)
    mujoco.mj_fwdVelocity(m, d)
    d.qacc[:] = t["qacc"][i]
    mujoco.mj_rne(m, d, 1, out)
    # mj_rne leaves out armature, which is part of the joint-space inertia.
    rne[i] = out + m.dof_armature * t["qacc"][i]
    passive[i] = d.qfrc_passive
    for c in d.contact[: d.ncon]:
      for k, g in enumerate(foot_geoms):
        if g in (c.geom1, c.geom2) and c.dist < 0:
          contact[i, k] = True
  return {"rne": rne, "passive": passive, "contact": contact}


def over_curve(tau: np.ndarray, vel: np.ndarray) -> np.ndarray:
  """Torque beyond what the X8-120 can give at this speed (curve when driving
  the motion, the torque cap when braking it)."""
  curve = c1.VOLTAGE_LINE_STALL_TORQUE * (1.0 - np.abs(vel) / c1.NO_LOAD_SPEED)
  avail = np.where(tau * vel > 0, np.clip(curve, 0.0, c1.TORQUE_CAP), c1.TORQUE_CAP)
  return np.abs(tau) > avail + 1e-6


def rms(x: np.ndarray) -> float:
  return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


variants = [("baseline", copy.copy(model0))] + [make_variant(s) for s in variant_specs]
base_leg, base_hip_i = leg_mass(model0), hip_inertia(model0)
print(f"\nbaseline leg mass {base_leg:.2f} kg, hip inertia {base_hip_i:.3f} kg m^2")
for name, m in variants[1:]:
  dm = leg_mass(m) - base_leg
  di = hip_inertia(m) - base_hip_i
  print(
    f"  {name:16s} leg mass {dm:+.2f} kg, hip inertia {di:+.3f} kg m^2 "
    f"({di / base_hip_i:+.0%})"
  )

for speed, t in traj.items():
  recorded = t["qfrc_act"][:, dof]
  joint_vel = t["qvel"][:, dof]
  ref = dynamics(copy.copy(model0), t)
  # Swing = this leg's foot not touching the ground.
  swing = {"L": ~ref["contact"][:, 0], "R": ~ref["contact"][:, 1]}
  # Check: a swing leg's joints feel no contact force, so M qacc + bias must
  # equal actuator plus passive torque there.
  resid = ref["rne"][:, dof] - recorded - ref["passive"][:, dof]
  print(f"\ncmd {speed:.2f} m/s, {len(recorded)} substep samples")
  print(
    "  check, swing-leg |M qacc + bias - actuator - passive| RMS (N m): "
    + "  ".join(
      f"{j.split('_joint')[0]} {rms(resid[swing[j[0]], k]):.1f}"
      for k, j in enumerate(JOINTS)
    )
  )
  print(
    f"  {'variant':16s} {'joint':8s} {'rms':>6} {'Δrms':>6} {'swing rms':>9} "
    f"{'p99':>6} {'%over':>6} | {'base Fx,Fz':>10} {'pitch':>6}"
  )
  for name, m in variants:
    out = ref if name == "baseline" else dynamics(m, t)
    delta = out["rne"] - ref["rne"]
    tau = recorded + delta[:, dof]
    # Phantom force/torque the unactuated base would need to keep the motion.
    base_f = rms(np.linalg.norm(delta[:, root_dofs[:2]], axis=1))
    base_t = rms(delta[:, root_dofs[2]])
    for k, j in enumerate(JOINTS):
      x = tau[:, k]
      sw = x[swing[j[0]]]
      base_rms = rms(recorded[:, k])
      print(
        f"  {name if k == 0 else '':16s} {j.split('_joint')[0]:8s} {rms(x):6.1f} "
        f"{rms(x) - base_rms:+6.1f} {rms(sw):9.1f} "
        f"{np.percentile(np.abs(x), 99):6.1f} "
        f"{np.mean(over_curve(x, joint_vel[:, k])):6.1%} |"
        + (f" {base_f:8.1f} N {base_t:5.1f}" if k == 0 else "")
      )
