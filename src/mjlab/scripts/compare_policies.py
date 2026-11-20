"""Compare C1 policies side by side on gait quality and hardware cost.

Usage:
  uv run src/mjlab/scripts/compare_policies.py <ckpt> [<ckpt> ...]
    [--cmds -1.0 -0.5 0.5 1.0 1.5 2.0] [--envs 64] [--steps 500]

Each checkpoint is evaluated with the actuator gains and action scale it was
trained with, read from its run's params/env.yaml, on the current X8-120
speed-torque model. All checkpoints see the same fixed commands, deterministic
policy, play config (no pushes; friction and COM randomization stay on). Stats
skip the first second after every reset and are sampled every physics substep.

Columns:
  vx, err     mean actual speed and mean |vx - cmd| (m/s)
  falls       episodes ended by a termination
  L/R         touchdowns per foot; balance = min / max (1.0 = even, ~0 = hopping)
  foot p99    foot ground force, 99th percentile, in body weights
  td vz       median downward foot speed just before touchdown (m/s)
  acc p99     base specific force, 99th percentile (g)
  hip/knee    RMS torque in N*m and as a multiple of the 43 N*m rated torque
  sat         time the motor is saturated (delivering all it can at that speed
              while the PD asks for more), mean over left and right
  vel pk      peak joint speed over all joints (rad/s; no-load is 16.55)
  CoT         positive mechanical work / (weight * distance)
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
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


def available_torque(demand: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
  """Most torque the motor can give in the demanded direction at this speed."""
  curve = c1.VOLTAGE_LINE_STALL_TORQUE * (1.0 - vel.abs() / c1.NO_LOAD_SPEED)
  driving = demand * vel > 0
  return torch.where(driving, curve.clamp(0.0, c1.TORQUE_CAP), c1.TORQUE_CAP)


def evaluate(ckpt: Path, cmds: list[float], envs: int, steps: int) -> dict:
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = envs
  trained = apply_trained_actuation(env_cfg, ckpt)
  twist_cfg = env_cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  twist_cfg.rel_forward_envs = 0.0  # keep fixed commands fixed (no |vx| flip)

  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE),
    clip_actions=agent_cfg.clip_actions,
  )
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
  policy = runner.get_inference_policy(device=DEVICE)

  base = env.unwrapped
  robot = base.scene["robot"]
  feet: ContactSensor = base.scene[FEET]
  names = robot.joint_names
  act_perm = [robot.actuator_names.index(n) for n in names]
  kp_list, kd_list = trained.joint_gains(names)
  kp = torch.tensor(kp_list, device=DEVICE)
  kd = torch.tensor(kd_list, device=DEVICE)
  weight = float(base.sim.mj_model.body_mass.sum()) * G
  sub_dt = base.physics_dt
  settle_steps = int(SETTLE_S / base.step_dt)
  hips = [i for i, n in enumerate(names) if "hip" in n]
  knees = [i for i, n in enumerate(names) if "knee" in n]

  rec: dict = {}

  def record() -> None:
    settled = base.episode_length_buf >= settle_steps
    tau = robot.data.actuator_force[:, act_perm]
    vel = robot.data.joint_vel
    demand = kp * (robot.data.joint_pos_target - robot.data.joint_pos) - kd * vel
    sat = (tau.abs() >= 0.98 * available_torque(demand, vel)) & (
      demand.abs() > tau.abs() + 1e-3
    )
    rec["tau_sq"] += (tau[settled] ** 2).sum(0)
    rec["samples"].append(torch.stack([tau[settled], vel[settled]], dim=-1))
    rec["sat"] += sat[settled].float().sum(0)
    if settled.any():
      rec["vel_pk"] = max(rec["vel_pk"], float(vel[settled].abs().max()))
    rec["work"] += float((tau * vel)[settled].clamp(min=0).sum()) * sub_dt
    base_vel = robot.data.root_link_lin_vel_w.clone()
    rec["dist"] += float(base_vel[settled, 0].abs().sum()) * sub_dt
    rec["err"] += float((base_vel[settled, 0] - rec["cmd"]).abs().sum())
    rec["vx"] += float(base_vel[settled, 0].sum())
    rec["n"] += int(settled.sum())
    assert feet.data.force is not None and feet.data.found is not None
    rec["foot_f"].append(feet.data.force[settled].norm(dim=-1).flatten())
    found = feet.data.found > 0
    foot_vel = robot.data.site_lin_vel_w.clone()
    if rec["prev"] is not None:
      prev_found, prev_foot_vel, prev_base_vel = rec["prev"]
      touchdown = found & ~prev_found & settled[:, None]
      rec["td"] += touchdown.sum(0)
      rec["td_vz"].append(-prev_foot_vel[..., 2][touchdown])
      accel = (base_vel - prev_base_vel) / sub_dt
      accel[:, 2] += G
      rec["acc"].append(accel[settled].norm(dim=-1) / G)
    rec["prev"] = (found, foot_vel, base_vel)

  scene_update = base.scene.update

  def recording_update(dt: float) -> None:
    scene_update(dt)
    record()

  base.scene.update = recording_update  # type: ignore[method-assign]

  results = {}
  samples = {}
  for cmd in cmds:
    twist_cfg.ranges.lin_vel_x = (cmd, cmd)
    rec.clear()
    rec.update(
      cmd=cmd,
      tau_sq=torch.zeros(4, device=DEVICE),
      sat=torch.zeros(4, device=DEVICE),
      td=torch.zeros(2, device=DEVICE),
      vel_pk=0.0,
      work=0.0,
      dist=0.0,
      err=0.0,
      vx=0.0,
      n=0,
      foot_f=[],
      samples=[],
      td_vz=[],
      acc=[],
      prev=None,
    )
    falls = 0
    with torch.inference_mode():
      obs = env.reset()
      obs = obs[0] if isinstance(obs, tuple) else obs
      for _ in range(steps):
        obs, _, dones, _ = env.step(policy(obs))
        falls += int(dones.sum())
    n = max(rec["n"], 1)
    rms = (rec["tau_sq"] / n).sqrt()
    sat = rec["sat"] / n
    foot_f = torch.cat(rec["foot_f"])
    td_vz = torch.cat(rec["td_vz"])
    acc = torch.cat(rec["acc"])
    td_l, td_r = (int(x) for x in rec["td"])
    results[cmd] = {
      "vx": rec["vx"] / n,
      "err": rec["err"] / n,
      "falls": falls,
      "td": (td_l, td_r),
      "balance": min(td_l, td_r) / max(td_l, td_r, 1),
      "foot_p99": float(torch.quantile(foot_f, 0.99)) / weight,
      "td_vz": float(td_vz.median()) if td_vz.numel() else float("nan"),
      "acc_p99": float(torch.quantile(acc, 0.99)),
      "hip_rms": float(rms[hips].mean()),
      "knee_rms": float(rms[knees].mean()),
      "hip_sat": float(sat[hips].mean()),
      "knee_sat": float(sat[knees].mean()),
      "vel_pk": rec["vel_pk"],
      # No distance at a standing command, so no cost of transport.
      "cot": rec["work"] / (weight * max(rec["dist"], 1e-6))
      if abs(cmd) > 0.1
      else float("nan"),
      "falls_per_min": falls / (envs * steps * base.step_dt) * 60.0,
    }
    # Up to 8000 random (torque, speed) samples per joint for envelope plots.
    smp = torch.cat(rec["samples"])
    keep = torch.randperm(smp.shape[0], device=smp.device)[:8000]
    samples[cmd] = smp[keep].cpu().numpy()
  env.close()
  label = f"{ckpt.parent.name}/{ckpt.stem}"
  info = trained.describe()
  return {
    "label": label,
    "info": info,
    "results": results,
    "samples": samples,
    "joint_names": list(names),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("ckpts", type=Path, nargs="+")
  parser.add_argument(
    "--cmds", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
  )
  parser.add_argument("--envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--csv", type=Path, help="write one row per run and command")
  parser.add_argument(
    "--samples", type=Path, help="write (torque, speed) samples as .npz"
  )
  args = parser.parse_args()

  runs = [evaluate(c, args.cmds, args.envs, args.steps) for c in args.ckpts]
  rated = c1.RATED_TORQUE
  if args.csv:
    fields = [
      "run",
      "label",
      "cmd",
      "vx",
      "err",
      "falls",
      "falls_per_min",
      "td_left",
      "td_right",
      "balance",
      "foot_p99_bw",
      "td_vz",
      "acc_p99_g",
      "hip_rms",
      "knee_rms",
      "hip_sat",
      "knee_sat",
      "vel_pk",
      "cot",
    ]
    with args.csv.open("w", newline="") as fh:
      writer = csv.DictWriter(fh, fieldnames=fields)
      writer.writeheader()
      for i, run in enumerate(runs):
        for cmd, r in run["results"].items():
          writer.writerow(
            {
              "run": i,
              "label": run["label"],
              "cmd": cmd,
              "vx": r["vx"],
              "err": r["err"],
              "falls": r["falls"],
              "falls_per_min": r["falls_per_min"],
              "td_left": r["td"][0],
              "td_right": r["td"][1],
              "balance": r["balance"],
              "foot_p99_bw": r["foot_p99"],
              "td_vz": r["td_vz"],
              "acc_p99_g": r["acc_p99"],
              "hip_rms": r["hip_rms"],
              "knee_rms": r["knee_rms"],
              "hip_sat": r["hip_sat"],
              "knee_sat": r["knee_sat"],
              "vel_pk": r["vel_pk"],
              "cot": r["cot"],
            }
          )
    print(f"saved {args.csv}")
  if args.samples:
    arrays = {
      f"run{i}_cmd{cmd:+.1f}": smp
      for i, run in enumerate(runs)
      for cmd, smp in run["samples"].items()
    }
    np.savez(args.samples, joint_names=np.array(runs[0]["joint_names"]), **arrays)
    print(f"saved {args.samples}")

  print()
  for i, run in enumerate(runs):
    print(f"[{i}] {run['label']}: {run['info']}")
  header = (
    f"{'cmd':>5} {'':3} {'vx':>5} {'err':>5} {'falls':>5} {'L/R':>11} {'bal':>4} "
    f"{'foot':>5} {'td vz':>5} {'acc':>5} {'hip rms':>11} {'knee rms':>11} "
    f"{'hip sat':>7} {'kn sat':>6} {'vel pk':>6} {'CoT':>5}"
  )
  units = (
    f"{'':>5} {'':3} {'m/s':>5} {'m/s':>5} {'':>5} {'touchdowns':>11} {'':>4} "
    f"{'BW':>5} {'m/s':>5} {'g':>5} {'Nm (xrat)':>11} {'Nm (xrat)':>11} "
    f"{'':>7} {'':>6} {'rad/s':>6} {'':>5}"
  )
  print("\n" + header + "\n" + units)
  for cmd in args.cmds:
    for i, run in enumerate(runs):
      r = run["results"][cmd]
      print(
        f"{cmd:+5.1f} [{i}] {r['vx']:+5.2f} {r['err']:5.2f} {r['falls']:5d} "
        f"{r['td'][0]:5d}/{r['td'][1]:<5d} {r['balance']:4.2f} "
        f"{r['foot_p99']:5.1f} {r['td_vz']:5.2f} {r['acc_p99']:5.1f} "
        f"{r['hip_rms']:5.1f} ({r['hip_rms'] / rated:3.1f}) "
        f"{r['knee_rms']:5.1f} ({r['knee_rms'] / rated:3.1f}) "
        f"{r['hip_sat']:7.1%} {r['knee_sat']:6.1%} {r['vel_pk']:6.2f} "
        f"{r['cot']:5.2f}"
      )
    if len(runs) > 1:
      print()

  print("summary over commands (mean, except falls = total, balance = worst):")
  for i, run in enumerate(runs):
    rs = list(run["results"].values())

    def mean(key: str, rs=rs) -> float:
      return sum(r[key] for r in rs) / len(rs)

    print(
      f"  [{i}] err {mean('err'):.2f} m/s, falls {sum(r['falls'] for r in rs)}, "
      f"balance {min(r['balance'] for r in rs):.2f}, foot p99 "
      f"{mean('foot_p99'):.1f} BW, td vz {mean('td_vz'):.2f} m/s, hip rms "
      f"{mean('hip_rms'):.1f} Nm, knee rms {mean('knee_rms'):.1f} Nm, hip sat "
      f"{mean('hip_sat'):.1%}, knee sat {mean('knee_sat'):.1%}, CoT "
      f"{mean('cot'):.2f}"
    )


if __name__ == "__main__":
  main()
