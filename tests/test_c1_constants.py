"""Tests for c1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.caltech_c1 import c1_constants
from mjlab.entity import Entity, RootType
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def c1_entity() -> Entity:
  return Entity(c1_constants.get_c1_robot_cfg())


@pytest.fixture(scope="module")
def c1_model(c1_entity: Entity) -> mujoco.MjModel:
  return c1_entity.spec.compile()


def test_actuators_follow_gain_switch():
  gains = c1_constants._GAINS[c1_constants.C1_GAINS]
  for cfg, joint in (
    (c1_constants.C1_HIP_ACTUATOR, "hip"),
    (c1_constants.C1_KNEE_ACTUATOR, "knee"),
  ):
    assert (cfg.stiffness, cfg.damping) == gains[joint]
    assert cfg.effort_limit == c1_constants.TORQUE_CAP
    params = cfg.motor_params
    assert isinstance(params, c1_constants.DcMotorDatasheetParams)
    assert params.no_load_speed == c1_constants.NO_LOAD_SPEED


def test_torque_speed_curve_passes_datasheet_points():
  stall = c1_constants.VOLTAGE_LINE_STALL_TORQUE
  no_load = c1_constants.NO_LOAD_SPEED
  rated = stall * (1 - c1_constants.RATED_SPEED / no_load)
  assert rated == pytest.approx(c1_constants.RATED_TORQUE)
  assert stall == pytest.approx(219.0, abs=1.0)


def test_actuator_parameters(c1_model):
  # Native <dcmotor>s in position mode, force-limited at the torque cap.
  for i in range(c1_model.nu):
    actuator = c1_model.actuator(i)
    assert actuator.forcerange[0] == -c1_constants.TORQUE_CAP
    assert actuator.forcerange[1] == c1_constants.TORQUE_CAP


def initialize_entity_with_sim(entity, device, num_envs=1):
  """Initialize an entity with a simulation."""
  model = entity.compile()
  sim_cfg = SimulationCfg(njmax=75)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_keyframe_base_position(c1_model) -> None:
  data = mujoco.MjData(c1_model)
  mujoco.mj_resetDataKeyframe(c1_model, data, 0)
  mujoco.mj_forward(c1_model, data)
  kf = c1_constants.KNEES_BENT_KEYFRAME
  rest = c1_model.body("base_link").pos
  expected_pitch = 2.0 * np.arctan2(kf.rot[2], kf.rot[0])
  np.testing.assert_allclose(
    data.qpos[:3], [kf.pos[0] - rest[0], kf.pos[2] - rest[2], expected_pitch]
  )


def test_keyframe_feet_on_ground(c1_model) -> None:
  data = mujoco.MjData(c1_model)
  mujoco.mj_resetDataKeyframe(c1_model, data, c1_model.key("init_state").id)
  mujoco.mj_forward(c1_model, data)
  for site in ("left_foot_contact", "right_foot_contact"):
    assert abs(data.site_xpos[c1_model.site(site).id][2]) < 0.01


def test_keyframe_joint_positions(c1_entity, c1_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = c1_model.key("init_state")
  expected_joint_pos = c1_constants.KNEES_BENT_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, c1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    c1_entity.joint_names, expected_values, strict=True
  ):
    joint = c1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(c1_model) -> None:
  foot_pattern = r"^(L|R)_foot_geom$"
  for i in range(c1_model.ngeom):
    geom = c1_model.geom(i)
    geom_name = geom.name
    # Foot collision geoms should have condim=3, priority=1, and friction=0.6.
    if re.match(foot_pattern, geom_name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_non_foot_collision_geoms(c1_model) -> None:
  foot_pattern = r"^(L|R)_foot_geom$"
  for i in range(c1_model.ngeom):
    geom = c1_model.geom(i)
    geom_name = geom.name
    if "_geom" not in geom_name:
      continue
    # Non-foot collision geoms should have condim=1
    if not re.match(foot_pattern, geom_name):
      assert geom.condim == 1


def test_collision_geom_count(c1_model) -> None:
  # There should be 1 geoms (capsules) per foot.
  collision_geoms = [
    c1_model.geom(i).name
    for i in range(c1_model.ngeom)
    if "_geom" in c1_model.geom(i).name
  ]
  assert len(collision_geoms) > 0
  foot_geoms = [
    name for name in collision_geoms if re.match(r"^(L|R)_foot_geom$", name)
  ]
  assert len(foot_geoms) == 2


def test_c1_entity_creation(c1_entity) -> None:
  assert c1_entity.num_actuators == 4
  assert c1_entity.num_joints == 4
  assert c1_entity.is_actuated
  assert not c1_entity.is_fixed_base
  assert c1_entity.root_type == RootType.PLANAR


def test_c1_actuators_configured_correctly(c1_model):
  """Verify that all C1 actuators have correct control and force limiting.

  All 4 C1 actuators are position-mode <dcmotor>s: ctrl is a joint angle
  (unlimited, so setpoints may pass joint limits) and force is limited to the
  torque cap.
  """
  for i in range(c1_model.nu):
    actuator_name = c1_model.actuator(i).name
    assert c1_model.actuator_ctrllimited[i] == 0, (
      f"Actuator '{actuator_name}' has ctrllimited=True, expected False"
    )
    assert c1_model.actuator_forcelimited[i] == 1, (
      f"Actuator '{actuator_name}' has forcelimited=False, expected True"
    )
