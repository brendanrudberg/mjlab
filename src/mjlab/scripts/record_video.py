# record_video.py
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

TASK = "Mjlab-Velocity-Flat-C1"
ckpt = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
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

obs = env.get_observations()
if isinstance(obs, tuple):
  obs = obs[0]
with torch.inference_mode():
  for _ in range(steps):
    obs, _, _, _ = env.step(policy(obs))

env.close()  # flushes and saves the video
print(f"saved to {out_dir}")
