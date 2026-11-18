# play_angle.py
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-C1"
ckpt = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
device = "cuda:0"

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 64
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

falls = torch.zeros(64, device=device)
vel_err = torch.zeros(64, device=device)
with torch.inference_mode():
  for i in range(steps):
    obs, _, dones, _ = env.step(policy(obs))
    falls += dones.float()
    cmd = base.command_manager.get_command("twist")[:, 0]
    vx = robot.data.root_link_lin_vel_w[:, 0]
    vel_err += torch.abs(cmd - vx)

survived = (falls == 0).float().mean().item()
print(f"survived all {steps} steps: {survived:.0%}   mean falls per env: {falls.mean().item():.2f}")
print(f"mean |cmd - vx|: {(vel_err / steps).mean().item():.3f} m/s")
term = env.unwrapped.reward_manager._term_cfgs[env.unwrapped.reward_manager._term_names.index("forward_step")].func
print(f"paid steps: {term.count} over 64 envs, {term.count / 64:.1f} per env")