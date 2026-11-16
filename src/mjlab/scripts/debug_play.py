# scripts/debug_play.py
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.c1.env_cfgs import caltech_c1_flat_env_cfg

cfg = caltech_c1_flat_env_cfg(play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
robot = env.scene["robot"]
action = torch.zeros((1, 4), device=env.device)

for i in range(300):
  _, _, terminated, truncated, _ = env.step(action)
  g = robot.data.projected_gravity_b[0]
  tilt = torch.rad2deg(torch.acos(torch.clamp(-g[2], -1.0, 1.0))).item()
  pos = robot.data.root_link_pos_w[0]
  print(f"step {i:3d}  tilt {tilt:6.1f} deg  x {pos[0].item():6.3f}  z {pos[2].item():6.3f}")
  if terminated.any() or truncated.any():
    print("ENDED: terminated", terminated.item(), "truncated", truncated.item())
    break