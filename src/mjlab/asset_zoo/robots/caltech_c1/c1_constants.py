"""Caltech C1 constants."""

import math
from pathlib import Path
from typing import Literal

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

C1_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "caltech_c1"
  / "xmls"
  / "mario_primitive.xml"
)
assert C1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(C1_XML))


##
# Actuator config.
##

# Motor specs (from MyActuator). No stage MOIs, so we assume a single-stage gearbox.
ROTOR_INERTIA_RMDX8120 = 1.262e-4  # Manufacturer email
GEAR_RMDX8120 = 19.61

ARMATURE_RMDX8120 = ROTOR_INERTIA_RMDX8120 * GEAR_RMDX8120**2

# Datasheet speed-torque points, output side.
RPM = 2.0 * math.pi / 60.0
NO_LOAD_SPEED = 158.0 * RPM  # 16.55 rad/s
RATED_SPEED = 127.0 * RPM  # 13.30 rad/s
RATED_TORQUE = 43.0  # N*m, continuous (thermal) rating
PEAK_TORQUE = 120.0  # N*m, current limit
# The rated and no-load points both lie on the voltage-limit line. Extended to zero
# speed it gives ~219 N*m, so with the 120 N*m current limit on top, torque is flat
# up to ~7.5 rad/s and then falls linearly to zero at no-load speed.
VOLTAGE_LINE_STALL_TORQUE = RATED_TORQUE * NO_LOAD_SPEED / (NO_LOAD_SPEED - RATED_SPEED)
TORQUE_CAP = 110.0  # N*m, peak minus a safety margin

ACTUATOR_RMDX8120 = ElectricActuator(
  reflected_inertia=ARMATURE_RMDX8120,
  velocity_limit=NO_LOAD_SPEED,
  effort_limit=PEAK_TORQUE,
)

# Armature recipe, as for the G1: 10 Hz on reflected rotor inertia alone.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0  # compensates for using armature alone rather than armature plus link inertia, per BeyondMimic S1.

STIFFNESS_RMDX8120 = ARMATURE_RMDX8120 * NATURAL_FREQ**2

DAMPING_RMDX8120 = 2.0 * DAMPING_RATIO * ARMATURE_RMDX8120 * NATURAL_FREQ

# Hand-picked gains (2026-11-18, during the 1-env runs), sized against the real
# joint inertia at the default pose: hip 1.285 kg*m^2, knee 0.219 kg*m^2.
HIP_STIFFNESS = 1270.0  # 5 Hz
HIP_DAMPING = 81.0
KNEE_STIFFNESS = 550.0  # 8 Hz
KNEE_DAMPING = 22.0

# Which gain set the C1 uses: "tuned" (hand-picked) or "armature" (G1 recipe).
C1_GAINS: Literal["tuned", "armature"] = "armature"
_GAINS = {
  "tuned": {
    "hip": (HIP_STIFFNESS, HIP_DAMPING),
    "knee": (KNEE_STIFFNESS, KNEE_DAMPING),
  },
  "armature": {
    "hip": (STIFFNESS_RMDX8120, DAMPING_RMDX8120),
    "knee": (STIFFNESS_RMDX8120, DAMPING_RMDX8120),
  },
}


# The speed-torque line only depends on stall torque and no-load speed when the
# drive voltage limit equals the nominal voltage, so the voltage value itself is a
# placeholder.
NOMINAL_VOLTAGE = 48.0


def x8_120_actuator(
  joint: str, stiffness: float, damping: float
) -> BuiltinDcMotorActuatorCfg:
  """MuJoCo <dcmotor> in position mode for joint "hip" or "knee": PD on the
  joint, delivered through the X8-120 speed-torque line (drive voltage clamped at
  nominal) and capped at TORQUE_CAP. Back-EMF damping is integrated implicitly,
  so stiff gains stay stable at the 5 ms timestep, unlike an explicit torch-side
  PD."""
  return BuiltinDcMotorActuatorCfg(
    target_names_expr=(f".*_{joint}_joint",),
    motor_params=DcMotorDatasheetParams(
      nominal_voltage=NOMINAL_VOLTAGE,
      stall_torque=VOLTAGE_LINE_STALL_TORQUE,
      no_load_speed=NO_LOAD_SPEED,
    ),
    mode=DcMotorInputMode.POSITION,
    stiffness=stiffness,
    damping=damping,
    voltage_limit=NOMINAL_VOLTAGE,
    effort_limit=TORQUE_CAP,
    armature=ARMATURE_RMDX8120,
  )


C1_HIP_ACTUATOR = x8_120_actuator("hip", *_GAINS[C1_GAINS]["hip"])
C1_KNEE_ACTUATOR = x8_120_actuator("knee", *_GAINS[C1_GAINS]["knee"])

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
  pos=(0.0, 0.0, 1.0124),
  joint_pos={
    "L_hip_joint": -0.498,
    "L_knee_joint": 0.80,
    "R_hip_joint": -0.498,
    "R_knee_joint": 0.80,
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
  actuators=(C1_HIP_ACTUATOR, C1_KNEE_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_c1_robot_cfg() -> EntityCfg:
  """Get a fresh C1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=C1_ARTICULATION,
  )


# Tuned gains use hand-picked scales; the armature recipe uses the G1 rule, where an
# action of 1 commands 25% of peak torque at rest.
if C1_GAINS == "tuned":
  C1_ACTION_SCALE: dict[str, float] = {".*_hip_joint": 0.1, ".*_knee_joint": 0.15}
else:
  C1_ACTION_SCALE = {
    ".*_hip_joint": 0.25 * PEAK_TORQUE / STIFFNESS_RMDX8120,
    ".*_knee_joint": 0.25 * PEAK_TORQUE / STIFFNESS_RMDX8120,
  }


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_c1_robot_cfg())

  viewer.launch(robot.spec.compile())
