# play_angle.py
# Evaluate a checkpoint over 64 envs: survival, falls, velocity error, and the
# joint range used while walking (2nd-98th percentile, envs up for > MIN_UP steps).
#
#   uv run python src/mjlab/scripts/play_angle.py <checkpoint.pt> [steps] [min_up_steps]
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-C1"
NUM_ENVS = 64
ckpt = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
MIN_UP = int(sys.argv[3]) if len(sys.argv) > 3 else 100  # 100 steps = 2 s
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(TASK)

env = RslRlVecEnvWrapper(
  ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

base = env.unwrapped
robot = base.scene["robot"]
obs = env.get_observations()
if isinstance(obs, tuple):
  obs = obs[0]

falls = torch.zeros(NUM_ENVS, device=device)
vel_err = torch.zeros(NUM_ENVS, device=device)
samples = []

with torch.inference_mode():
  for i in range(steps):
    obs, _, dones, _ = env.step(policy(obs))
    falls += dones.float()
    cmd = base.command_manager.get_command("twist")[:, 0]
    vx = robot.data.root_link_lin_vel_w[:, 0]
    vel_err += torch.abs(cmd - vx)
    # Joint angles only from envs that have stayed up a while (walking, not falling).
    long_lived = base.episode_length_buf > MIN_UP
    if long_lived.any():
      samples.append(robot.data.joint_pos[long_lived].clone())

survived = (falls == 0).float().mean().item()
mean_falls = falls.mean().item()
print(
  f"survived all {steps} steps: {survived:.0%}   mean falls per env: {mean_falls:.2f}"
  f"   (~{steps / max(mean_falls, 1e-6) * base.step_dt:.1f} s per episode)"
)
print(f"mean |cmd - vx|: {(vel_err / steps).mean().item():.3f} m/s")

if samples:
  allq = torch.cat(samples)
  lo = torch.quantile(allq, 0.02, dim=0)
  hi = torch.quantile(allq, 0.98, dim=0)
  print(
    f"\njoint range while up > {MIN_UP * base.step_dt:.1f} s "
    f"({allq.shape[0]} samples, 2nd-98th percentile):"
  )
  for name, a, b in zip(robot.joint_names, lo.tolist(), hi.tolist()):
    print(f"  {name:14s} {a:+.3f}  to  {b:+.3f}")
else:
  print(
    f"\nno env stayed up longer than {MIN_UP * base.step_dt:.1f} s; "
    f"rerun with a smaller min_up_steps (3rd argument)"
  )
