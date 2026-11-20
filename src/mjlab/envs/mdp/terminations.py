"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return (
    torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > limit_angle
  )


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics state."""
  return NanGuard.detect_nans(env.sim.data)


def double_support_too_long(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_time: float = 0.3,
  grace_time: float = 0.5,
) -> torch.Tensor:
  """Both feet have been on the ground longer than max_time."""
  contact: ContactSensor = env.scene[sensor_name]
  t_contact = contact.data.current_contact_time  # [B, 2]
  assert t_contact is not None
  started = env.episode_length_buf * env.step_dt > grace_time
  return (t_contact.min(dim=1).values > max_time) & started


def foot_airborne_too_long(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_time: float = 0.6,
) -> torch.Tensor:
  """Either foot has been in the air longer than max_time."""
  contact: ContactSensor = env.scene[sensor_name]
  t_air = contact.data.current_air_time  # [B, 2]
  assert t_air is not None
  return (t_air > max_time).any(dim=1)


def air_time_too_short(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_time: float = 0.1,
  grace_time: float = 0.5,
) -> torch.Tensor:
  """A foot touched down after less than min_time in the air."""
  contact: ContactSensor = env.scene[sensor_name]
  first_contact = contact.compute_first_contact(dt=env.step_dt)  # [B, 2]
  last_air = contact.data.last_air_time  # [B, 2]
  assert last_air is not None
  started = env.episode_length_buf * env.step_dt > grace_time
  return ((first_contact & (last_air < min_time)).any(dim=1)) & started


def joint_vel_out_of_limit(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.joint_vel.abs() > max_vel).any(dim=1)
