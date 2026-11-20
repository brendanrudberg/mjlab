"""C1 actuator response test: can a leg follow its position targets?

Holds the base fixed in the air (contacts off, gravity on), uses the same PD
gains, armature and torque limit as c1_constants.py, and commands one joint:
  1. a 0.2 rad step  -> rise time, overshoot, settling
  2. a 0.2 rad sine at 1.25 Hz (expected stride rate) -> amplitude ratio, lag
The other joints hold the default pose. Also prints the predicted natural
frequency and damping ratio from the real joint inertia (mass matrix).

Run from the repo root:
  uv run python src/mjlab/scripts/actuator_test.py
  uv run python src/mjlab/scripts/actuator_test.py --kd 32   # try other gains
"""

import argparse
import math

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.caltech_c1.c1_constants import (
  ARMATURE_RMDX8120,
  C1_XML,
  DAMPING_RMDX8120,
  STIFFNESS_RMDX8120,
)

LEG_JOINTS = ("L_hip_joint", "L_knee_joint", "R_hip_joint", "R_knee_joint")
BASE_JOINTS = ("base_px", "base_pz", "base_phi")
DEFAULT = {
  "L_hip_joint": -0.498,
  "L_knee_joint": 0.80,
  "R_hip_joint": -0.498,
  "R_knee_joint": 0.80,
}
EFFORT = 120.0
DT = 0.005


def build(kp: float, kd: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
  model = mujoco.MjModel.from_xml_path(str(C1_XML))
  model.opt.timestep = DT / 10
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  for name in LEG_JOINTS:
    model.dof_armature[model.joint(name).dofadr[0]] = ARMATURE_RMDX8120
  # Make the base effectively immovable so leg reaction torques can't push it
  # within a step (pin_base then removes any residual drift).
  for name in BASE_JOINTS:
    model.dof_armature[model.joint(name).dofadr[0]] = 1e6
  data = mujoco.MjData(model)
  return model, data


def pin_base(model, data, z_up: float = 1.0) -> None:
  """Weld the base in place by resetting its planar joints every step."""
  for name, value in zip(BASE_JOINTS, (0.0, z_up, 0.0)):
    j = model.joint(name)
    data.qpos[j.qposadr[0]] = value
    data.qvel[j.dofadr[0]] = 0.0


def set_pose(model, data) -> None:
  for name, q in DEFAULT.items():
    data.qpos[model.joint(name).qposadr[0]] = q
  pin_base(model, data)
  mujoco.mj_forward(model, data)


def pd_torque(model, data, targets: dict[str, float], kp: float, kd: float) -> None:
  data.qfrc_applied[:] = 0.0
  for name in LEG_JOINTS:
    j = model.joint(name)
    q, qd = data.qpos[j.qposadr[0]], data.qvel[j.dofadr[0]]
    tau = kp * (targets[name] - q) - kd * qd
    data.qfrc_applied[j.dofadr[0]] = np.clip(tau, -EFFORT, EFFORT)


def predicted(
  model, data, joint: str, kp: float, kd: float
) -> tuple[float, float, float]:
  d = model.joint(joint).dofadr[0]
  e = np.zeros(model.nv)
  e[d] = 1.0
  Me = np.zeros(model.nv)
  mujoco.mj_mulM(model, data, Me, e)  # column d of the mass matrix
  inertia = Me[d]  # base fixed, other joints held: leg rotates rigidly
  wn = math.sqrt(kp / inertia)
  zeta = kd / (2.0 * math.sqrt(kp * inertia))
  return inertia, wn, zeta


def run(model, data, joint: str, kp: float, kd: float, target_fn, duration: float):
  set_pose(model, data)
  ts, qs, refs = [], [], []
  q0 = DEFAULT[joint]
  for k in range(int(duration / DT)):
    t = k * DT
    targets = dict(DEFAULT)
    targets[joint] = q0 + target_fn(t)
    for _ in range(10):
      pd_torque(model, data, targets, kp, kd)
      mujoco.mj_step(model, data)
    pin_base(model, data)
    ts.append(t)
    qs.append(data.qpos[model.joint(joint).qposadr[0]] - q0)
    refs.append(target_fn(t))
  return np.array(ts), np.array(qs), np.array(refs)


def step_metrics(t, q, step: float, t0: float):
  m = t >= t0
  t, q = t[m] - t0, q[m]
  try:
    rise = t[np.argmax(q >= 0.9 * step)] if np.any(q >= 0.9 * step) else float("nan")
  except ValueError:
    rise = float("nan")
  overshoot = max(0.0, (q.max() - step) / step) * 100
  outside = np.where(np.abs(q - step) > 0.05 * step)[0]
  settle = t[outside[-1]] if len(outside) else 0.0
  final_err = q[-1] - step
  return rise, overshoot, settle, final_err


def sine_metrics(t, q, freq: float, amp: float, t_start: float):
  m = t >= t_start
  t, q = t[m], q[m]
  w = 2 * math.pi * freq
  a = 2 * np.mean(q * np.sin(w * t))
  b = 2 * np.mean(q * np.cos(w * t))
  ratio = math.hypot(a, b) / amp
  lag_deg = -math.degrees(math.atan2(b, a))
  return ratio, lag_deg, lag_deg / 360.0 / freq


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--kp", type=float, default=STIFFNESS_RMDX8120)
  p.add_argument("--kd", type=float, default=DAMPING_RMDX8120)
  p.add_argument("--step", type=float, default=0.2)
  p.add_argument("--freq", type=float, default=1.25)
  p.add_argument("--plot", action="store_true")
  args = p.parse_args()

  model, data = build(args.kp, args.kd)
  print(
    f"kp = {args.kp:.1f} N*m/rad   kd = {args.kd:.2f} N*m*s/rad   "
    f"armature = {ARMATURE_RMDX8120:.4f}"
  )

  results = {}
  for joint in ("L_hip_joint", "L_knee_joint"):
    set_pose(model, data)
    inertia, wn, zeta = predicted(model, data, joint, args.kp, args.kd)
    print(f"\n== {joint}")
    print(
      f"  inertia at default pose: {inertia:.3f} kg*m^2 "
      f"({inertia / ARMATURE_RMDX8120:.0f}x armature)"
    )
    print(
      f"  predicted: natural freq {wn / (2 * math.pi):.2f} Hz, damping ratio {zeta:.2f}"
    )

    t0 = 0.5
    t, q, r = run(
      model, data, joint, args.kp, args.kd, lambda t: args.step if t >= t0 else 0.0, 3.0
    )
    rise, over, settle, err = step_metrics(t, q, args.step, t0)
    print(
      f"  step {args.step} rad: rise(90%) {rise * 1000:.0f} ms, "
      f"overshoot {over:.0f}%, settle(5%) {settle * 1000:.0f} ms, "
      f"final error {err * 1000:.1f} mrad"
    )

    ts, qs, rs = run(
      model,
      data,
      joint,
      args.kp,
      args.kd,
      lambda t: args.step * math.sin(2 * math.pi * args.freq * t),
      6.0,
    )
    ratio, lag_deg, lag_s = sine_metrics(ts, qs, args.freq, args.step, 2.0)
    print(
      f"  sine {args.freq} Hz: amplitude {ratio * 100:.0f}% of target, "
      f"lag {lag_deg:.0f} deg ({lag_s * 1000:.0f} ms)"
    )
    results[joint] = (t, q, r, ts, qs, rs)

  if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for col, (joint, (t, q, r, ts, qs, rs)) in enumerate(results.items()):
      axes[0, col].plot(t, r, "k--", label="target")
      axes[0, col].plot(t, q, label="joint")
      axes[0, col].set_title(f"{joint}: step")
      axes[1, col].plot(ts, rs, "k--", label="target")
      axes[1, col].plot(ts, qs, label="joint")
      axes[1, col].set_title(f"{joint}: {args.freq} Hz sine")
      for ax in axes[:, col]:
        ax.set_xlabel("t (s)")
        ax.set_ylabel("offset from default (rad)")
        ax.legend()
    fig.tight_layout()
    fig.savefig("actuator_test.png", dpi=120)
    print("\nsaved actuator_test.png")


if __name__ == "__main__":
  main()
