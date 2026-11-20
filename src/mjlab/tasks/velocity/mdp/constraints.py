"""CaT constraint functions. Each returns a violation: > 0 means violated.

Ported from gepetto/constraints-as-terminations (constraints.py), adapted to
mjlab data names and a planar biped. Shapes are [B] or [B, k].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

_ROBOT = SceneEntityCfg("robot")


def _moving(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (torch.abs(command[:, 0]) > threshold).float()


# Safety (soft) ----------------------------------------------------------------


def joint_torque(env, limit: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.actuator_force) - limit


def joint_velocity(
  env, limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.joint_vel) - limit


def joint_acceleration(
  env, limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if not getattr(joint_acceleration, "_printed", False):
    print(
      "joint_acc shape:",
      tuple(asset.data.joint_acc.shape),
      "joint_vel shape:",
      tuple(asset.data.joint_vel.shape),
      "joint names:",
      asset.joint_names,
    )
    joint_acceleration._printed = True
  return torch.abs(asset.data.joint_acc) - limit


def action_rate(env, limit: float) -> torch.Tensor:
  am = env.action_manager
  return torch.abs(am.action - am.prev_action) / env.step_dt - limit


def foot_slip(
  env,
  sensor_name: str,
  limit: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=(".*_foot_contact",)),
) -> torch.Tensor:
  """Horizontal speed of a foot that is on the ground, above `limit` m/s."""
  asset: Entity = env.scene[asset_cfg.name]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_contact = (contact.data.found > 0).float()  # [B, 2]
  foot_vx = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 0].abs()  # [B, 2]
  return foot_vx * in_contact - limit


def joint_range(env, limit: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """How far each selected joint is from the default pose, beyond `limit` rad."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.abs(q - q0) - limit


def joint_bounds(
  env, lower: float, upper: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """How far each selected joint is outside [lower, upper] rad."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.maximum(lower - q, q - upper)


def swing_clearance(
  env,
  sensor_name: str,
  min_height: float,
  swing_speed: float = 0.3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=(".*_foot_contact",)),
) -> torch.Tensor:
  """A foot that is in the air and moving forward faster than `swing_speed`
  (mid-swing) must be at least `min_height` above flat ground."""
  asset: Entity = env.scene[asset_cfg.name]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_air = (contact.data.found == 0).float()  # [B, 2]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, 2]
  foot_vx = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 0]  # [B, 2]
  swinging = (torch.abs(foot_vx) > swing_speed).float()
  return (min_height - foot_z) * in_air * swinging


# Safety (hard) ----------------------------------------------------------------


def upside_down(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.projected_gravity_b[:, 2] > 0.0).float()


# Style (soft) -----------------------------------------------------------------


def base_pitch(env, limit: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """|sin(pitch)| above limit. Planar version of CaT's base_orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.projected_gravity_b[:, 0]) - limit


# Gait (soft) ------------------------------------------------------------------


def n_foot_contact(
  env,
  sensor_name: str,
  number_of_desired_feet: int,
  command_name: str = "twist",
  min_command: float = 0.1,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """|feet in contact - desired|. A foot counts as down if its contact force
  exceeded force_threshold in any substep of the last policy step (debounced,
  as in Bolt), so contact chatter doesn't read as lift-off."""
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.force_history is not None  # needs history_length > 0
  force = torch.norm(contact.data.force_history, dim=-1)  # [B, 2, H]
  in_contact = (force > force_threshold).any(dim=-1)  # [B, 2]
  n = in_contact.sum(dim=1).float()
  return torch.abs(n - number_of_desired_feet) * _moving(env, command_name, min_command)


def air_time_min(
  env,
  sensor_name: str,
  limit: float,
  command_name: str = "twist",
  min_command: float = 0.1,
) -> torch.Tensor:
  """At touchdown: how far the swing fell short of `limit` seconds."""
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.last_air_time is not None
  touchdown = contact.compute_first_contact(dt=env.step_dt).float()  # [B, 2]
  short = limit - contact.data.last_air_time
  return short * touchdown * _moving(env, command_name, min_command).unsqueeze(1)


def air_time_max(env, sensor_name: str, limit: float) -> torch.Tensor:
  """A foot held in the air longer than `limit` seconds. Not in CaT; closes the
  raised-foot shuffle, which satisfies single contact and min air time."""
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.current_air_time is not None
  return contact.data.current_air_time - limit


class StanceTimeMax:
  """A foot that sits still on the ground longer than `limit` s while commanded
  to move. 'Still' is measured by position, not contact, so contact chatter
  can't reset the timer."""

  def __init__(self):
    self.anchor_x = None  # [B, 2] position where each foot last settled
    self.still_t = None  # [B, 2] seconds each foot has stayed within move_tol

  def __call__(
    self,
    env,
    limit: float,
    move_tol: float = 0.02,
    ground_tol: float = 0.02,
    command_name: str = "twist",
    min_command: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
      "robot", site_names=(".*_foot_contact",)
    ),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # [B, 2, 3]
    foot_x, foot_z = pos[..., 0], pos[..., 2]
    if self.anchor_x is None:
      self.anchor_x = foot_x.clone()
      self.still_t = torch.zeros_like(foot_x)

    # New episodes: restart from the current foot positions.
    new_ep = (env.episode_length_buf == 0).unsqueeze(1)
    self.anchor_x = torch.where(new_ep, foot_x, self.anchor_x)
    self.still_t = torch.where(new_ep, torch.zeros_like(self.still_t), self.still_t)

    on_ground = foot_z < ground_tol
    moved = torch.abs(foot_x - self.anchor_x) > move_tol
    restart = moved | ~on_ground
    self.anchor_x = torch.where(restart, foot_x, self.anchor_x)
    self.still_t = torch.where(
      restart, torch.zeros_like(self.still_t), self.still_t + env.step_dt
    )
    return (self.still_t - limit) * _moving(env, command_name, min_command).unsqueeze(1)
