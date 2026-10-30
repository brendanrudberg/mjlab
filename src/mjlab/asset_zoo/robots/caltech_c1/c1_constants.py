"""Caltech C1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

C1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "caltech_c1" / "xmls" / "mario_primitive.xml"
)
assert C1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(C1_XML))


##
# Actuator config.
##

# Motor specs (from MyActuator). No stage MOIs, so we assume a single-stage gearbox.
ROTOR_INERTIA_RMDX8120 = 1.262e-4 # Manufacturer email
GEAR_RMDX8120 = 19.61

ARMATURE_RMDX8120 = ROTOR_INERTIA_RMDX8120 * GEAR_RMDX8120**2


ACTUATOR_RMDX8120 = ElectricActuator(
  reflected_inertia=ARMATURE_RMDX8120,
  velocity_limit=16.5, # no-load speed in rad/s
  effort_limit=120.0, # peak torque in Nm
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0 # compensates for using armature alone rather than armature plus link inertia, per BeyondMimic S1.

STIFFNESS_RMDX8120 = ARMATURE_RMDX8120 * NATURAL_FREQ**2

DAMPING_RMDX8120 = 2.0 * DAMPING_RATIO * ARMATURE_RMDX8120 * NATURAL_FREQ

C1_ACTUATOR_RMDX8120 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_joint",
    ".*_knee_joint",
  ),
  stiffness=STIFFNESS_RMDX8120,
  damping=DAMPING_RMDX8120,
  effort_limit=ACTUATOR_RMDX8120.effort_limit,
  armature=ACTUATOR_RMDX8120.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.80),
  joint_pos={
    ".*_hip_joint": -0.1,
    ".*_knee_joint": 0.3,

  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_joint": -0.3,
    ".*_knee_joint": 0.7,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_geom",),
  contype=1,
  conaffinity=1,
  condim={r"^(L|R)_foot_geom$": 3, ".*_geom": 1},
  priority={r"^(L|R)_foot_geom$": 1, ".*": 0},
  friction={r"^(L|R)_foot_geom$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_geom",),
  contype=0,
  conaffinity=1,
  condim={r"^(L|R)_foot_geom$": 3, ".*_geom": 1},
  priority={r"^(L|R)_foot_geom$": 1, ".*": 0},
  friction={r"^(L|R)_foot_geom$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(L|R)_foot_geom$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

C1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    C1_ACTUATOR_RMDX8120,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_c1_robot_cfg() -> EntityCfg:
  """Get a fresh C1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=C1_ARTICULATION,
  )


C1_ACTION_SCALE: dict[str, float] = {}
for a in C1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    C1_ACTION_SCALE[n] = 0.25 * e / s # Look into this


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_c1_robot_cfg())

  viewer.launch(robot.spec.compile())
