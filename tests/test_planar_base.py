import math

import pytest
import torch

from mjlab.asset_zoo.robots.caltech_c1 import c1_constants
from mjlab.entity import Entity
from mjlab.sim.sim import Simulation, SimulationCfg
from conftest import get_test_device


def initialize_entity_with_sim(entity, device, num_envs=1):
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(njmax=75), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def _pitch_quat(theta: float) -> list[float]:
  return [math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]

@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()

@pytest.mark.parametrize("pitch", [0.3, 2.0, -2.5])
def test_planar_write_root_pose(device, pitch):
  entity, sim = initialize_entity_with_sim(Entity(c1_constants.get_c1_robot_cfg()), device)
  pose = torch.tensor([[1.0, 0.0, 0.8, *_pitch_quat(pitch)]], device=device)
  entity.write_root_link_pose_to_sim(pose)
  q = sim.data.qpos[0, entity.indexing.planar_joint_q_adr]
  torch.testing.assert_close(q, torch.tensor([1.0, 0.8, pitch], device=device), atol=1e-5, rtol=0)

@pytest.mark.parametrize("pitch", [0.0, 2.0])
def test_planar_write_root_velocity(device, pitch):
  entity, sim = initialize_entity_with_sim(Entity(c1_constants.get_c1_robot_cfg()), device)

  # Lean the base first: world-frame velocity must not depend on orientation.
  pose = torch.tensor([[0.0, 0.0, 0.8, *_pitch_quat(pitch)]], device=device)
  entity.write_root_link_pose_to_sim(pose)

  # y, roll-rate, and yaw-rate are nonzero on purpose: they must be dropped.
  vel_w = torch.tensor([[0.5, 0.3, -0.2, 0.1, 1.5, 0.4]], device=device)
  entity.write_root_link_velocity_to_sim(vel_w)

  qvel = sim.data.qvel[0, entity.indexing.planar_joint_v_adr]
  torch.testing.assert_close(
    qvel, torch.tensor([0.5, -0.2, 1.5], device=device), atol=1e-5, rtol=0
  )
  