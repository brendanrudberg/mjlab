# record_video.py
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.c1 import C1_CONSTRAINTS, FEET
from mjlab.utils.wrappers import VideoRecorder

TASK = "Mjlab-Velocity-Flat-C1"
ckpt = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SETTLE_S = 1.0  # skip the speed-up from rest when summarizing velocity
no_terminations = "--no-terminations" in sys.argv
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width = 1280
env_cfg.viewer.height = 720
if no_terminations:
  env_cfg.terminations = {}
agent_cfg = load_rl_cfg(TASK)

# Each checkpoint gets its own folder, so videos never overwrite each other.
out_dir = ckpt.parent / "videos" / ckpt.stem
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(
  env,
  video_folder=out_dir,
  step_trigger=lambda step: step == 0,
  video_length=steps,
  disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

base = env.unwrapped
robot = base.scene["robot"]
settle_steps = int(SETTLE_S / base.step_dt)

obs = env.get_observations()
if isinstance(obs, tuple):
  obs = obs[0]
cmds, vxs, falls = [], [], 0
# Per-step peaks for comparing against the CaT constraint limits.
act_rates, pitches, air_times = [], [], []
feet: ContactSensor = base.scene[FEET]
with torch.inference_mode():
  for _ in range(steps):
    obs, _, dones, _ = env.step(policy(obs))
    falls += int(dones.sum())
    # Skip steps right after a reset, while the robot speeds up from rest.
    if base.episode_length_buf[0] >= settle_steps:
      twist = base.command_manager.get_command("twist")
      assert twist is not None
      cmds.append(twist[0, 0].item())
      vxs.append(robot.data.root_link_lin_vel_w[0, 0].item())
      # Same quantities the constraints measure (see mdp/constraints.py).
      am = base.action_manager
      rate = (am.action - am.prev_action).abs() / base.step_dt
      act_rates.append(rate[0].max().item())
      pitches.append(robot.data.projected_gravity_b[0, 0].abs().item())
      touchdown = feet.compute_first_contact(dt=base.step_dt)[0]
      assert feet.data.last_air_time is not None
      air_times += feet.data.last_air_time[0][touchdown].tolist()

env.close()  # flushes and saves the video
print(f"saved to {out_dir}")

if vxs:
  cmd_t, vx_t = torch.tensor(cmds), torch.tensor(vxs)
  print(
    f"commanded vx: {cmd_t.mean():.2f} m/s (range {cmd_t.min():.2f} to "
    f"{cmd_t.max():.2f})"
  )
  print(
    f"actual vx:    {vx_t.mean():.2f} +/- {vx_t.std():.2f} m/s "
    f"(range {vx_t.min():.2f} to {vx_t.max():.2f})"
  )
  print(f"mean |cmd - vx|: {(cmd_t - vx_t).abs().mean():.3f} m/s")
else:
  print(f"no steps after the first {SETTLE_S:.0f} s of an episode to summarize")
print(f"falls: {falls}  ({len(vxs)} of {steps} steps summarized)")


def report(name: str, values: list[float], limit: float, kind: str) -> None:
  """Print the extreme and 1st/99th percentile next to the constraint limit."""
  if not values:
    print(f"  {name:22s} no samples")
    return
  t = torch.tensor(values)
  if kind == "max":  # violated above the limit
    peak, pct = t.max().item(), torch.quantile(t, 0.99).item()
    label, used = "peak", peak / limit
  else:  # violated below the limit
    peak, pct = t.min().item(), torch.quantile(t, 0.01).item()
    label, used = "min", limit / peak if peak > 0 else float("inf")
  print(
    f"  {name:22s} {label} {peak:7.3f}  p{99 if kind == 'max' else 1:<2d} {pct:7.3f}"
    f"  limit {limit:7.3f}  ({used:.0%})"
  )


print("\nconstraint headroom after settling (100% = touching the limit):")
report(
  "action_rate (1/s)", act_rates, C1_CONSTRAINTS["action_rate"].params["limit"], "max"
)
report("|sin(pitch)|", pitches, C1_CONSTRAINTS["base_pitch"].params["limit"], "max")
report("air time (s)", air_times, C1_CONSTRAINTS["air_time_min"].params["limit"], "min")
