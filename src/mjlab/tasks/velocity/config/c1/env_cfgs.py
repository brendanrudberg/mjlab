"""Caltech C1 velocity environment configurations.

Starts from mjlab's shared velocity task and overrides C1-specific settings.
Training uses Constraints as Terminations (CaT): velocity tracking is the only
reward, and gait/safety behavior comes from the constraints registered in
c1/__init__.py. Earlier reward-shaping setups (gait clock, forward_step, pose,
upright) are in git history.
"""

import math

from mjlab.asset_zoo.robots import C1_ACTION_SCALE, get_c1_robot_cfg
from mjlab.asset_zoo.robots.caltech_c1.c1_constants import RATED_SPEED, RATED_TORQUE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

BASE_BODY = "base_link"
FOOT_SITES = ("left_foot_contact", "right_foot_contact")
FOOT_GEOMS = ("L_foot_geom", "R_foot_geom")


def caltech_c1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the C1 rough-terrain task with robot-specific overrides."""
  cfg = make_velocity_env_cfg()

  # ---------------------------------------------------------------------------
  # Scene and simulation.
  # ---------------------------------------------------------------------------
  cfg.scene.entities = {"robot": get_c1_robot_cfg()}
  cfg.viewer.body_name = BASE_BODY
  # PPO and CaT's batch-max normalization both need many parallel envs.
  # Play keeps the shared default of 1; override with --env.scene.num-envs.
  if not play:
    cfg.scene.num_envs = 4096

  cfg.sim.mujoco.iterations = 50
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # ---------------------------------------------------------------------------
  # Sensors: bind inherited terrain sensors to C1 frames, add foot contact.
  # ---------------------------------------------------------------------------
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site, entity="robot") for site in FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Read by the CaT gait constraints (contact count, air time).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(L_foot_link|R_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )

  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(base_link|L_thigh_link|R_thigh_link|L_shin_link|R_shin_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, body_ground_cfg)

  # ---------------------------------------------------------------------------
  # Actions.
  # ---------------------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = C1_ACTION_SCALE

  # ---------------------------------------------------------------------------
  # Observations: the C1 XML has no IMU sensors yet, so use ground-truth base
  # velocities in their place.
  # ---------------------------------------------------------------------------
  imu_replacements = {
    "robot/imu_lin_vel": envs_mdp.base_lin_vel,
    "robot/imu_ang_vel": envs_mdp.base_ang_vel,
  }
  for group in ("actor", "critic"):
    for term in cfg.observations[group].terms.values():
      sensor_name = (term.params or {}).get("sensor_name")
      if sensor_name in imu_replacements:
        term.func = imu_replacements[sensor_name]
        term.params = {}

  # ---------------------------------------------------------------------------
  # Commands: planar forward motion only.
  # ---------------------------------------------------------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.9
  twist_cmd.ranges.lin_vel_x = (0.3, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.0
  # The shared config forces 20% of commands forward (|vx|); the curriculum
  # already sets the backward/forward mix.
  twist_cmd.rel_forward_envs = 0.0

  # ---------------------------------------------------------------------------
  # Events: bind inherited randomization to C1 geometry; spawn with small +-x speed.
  # ---------------------------------------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.events["reset_base"].params["velocity_range"]["x"] = (-0.3, 0.3)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.25, 0.25)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)
  cfg.terminations["body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "body_ground_contact"},
  )
  cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)

  # ---------------------------------------------------------------------------
  # Rewards: CaT uses velocity tracking only. The reward is progress toward the
  # command (vx / cmd, clipped to [0, 1]), so standing still earns nothing.
  # alpha=1.0 uses instantaneous vx; lower it to track a moving average.
  # ---------------------------------------------------------------------------
  cfg.rewards = {
    "track_smoothed_linear_velocity": RewardTermCfg(
      func=mdp.track_smoothed_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 0.5, "alpha": 1.0},
    ),
  }

  # ---------------------------------------------------------------------------
  # Curriculum: alternately widen the backward and forward command range. Steps
  # count env steps, so N * 24 is PPO iteration N.
  # ---------------------------------------------------------------------------
  lin_vel_x_stages = [
    (0.3, 1.0),
    (0.0, 1.0),
    (-0.3, 1.0),
    (-0.3, 1.5),
    (-1.0, 1.5),
    (-1.0, 2.0),
    (-1.5, 2.0),
    (-1.5, 2.5),
    (-2.0, 2.5),
    (-2.0, 3.0),
    (-2.0, 3.5),
    (-2.0, 4.0),
    (-2.0, 4.5),
    (-2.0, 5.0),
  ]
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": i * 200 * 24, "lin_vel_x": lin_vel_x}
    for i, lin_vel_x in enumerate(lin_vel_x_stages)
  ]
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # ---------------------------------------------------------------------------
  # Play mode: long episodes, clean observations, randomized terrain resets.
  # ---------------------------------------------------------------------------
  if play:
    twist_cmd.ranges.lin_vel_x = (-1.0, -1.0)
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      terrain_gen = cfg.scene.terrain.terrain_generator
      terrain_gen.curriculum = False
      terrain_gen.num_cols = 5
      terrain_gen.num_rows = 5
      terrain_gen.border_width = 10.0

  return cfg


def caltech_c1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Derive the flat-terrain task from the C1 rough configuration."""
  cfg = caltech_c1_rough_env_cfg(play=play)

  # Simulation: a plane needs far fewer contacts.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Terrain: replace the generator with a plane and drop the terrain scan.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  for group in ("actor", "critic"):
    del cfg.observations[group].terms["height_scan"]

  # Terminations and curriculum.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(35.0)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


# Weights relative to the velocity-tracking reward (weight 1.0, value in [0, 1]).
# -0.1 (run 2026-11-20_14-22-40) cut hip RMS torque by only 2-4%. At iteration
# 3200 the heat term averaged ~1.55 at full speed, where tracking earned ~0.76,
# so -0.3 costs ~0.47 per step there and keeps the net reward positive (ending an
# episode early never pays); -0.5 would push it to zero or below.
MOTOR_HEAT_WEIGHT = -0.3
JOINT_SPEED_MARGIN_WEIGHT = -0.3


def caltech_c1_flat_hw_penalty_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat C1 task plus penalties for approaching the X8-120 ratings: motor heat
  ((torque / 43 N*m)^2, i.e. RMS torque against the continuous rating) and joint
  speed above 80% of the 13.3 rad/s rated speed."""
  cfg = caltech_c1_flat_env_cfg(play=play)
  cfg.rewards["motor_heat"] = RewardTermCfg(
    func=mdp.motor_heat,
    weight=MOTOR_HEAT_WEIGHT,
    params={"rated_torque": RATED_TORQUE},
  )
  cfg.rewards["joint_speed_margin"] = RewardTermCfg(
    func=mdp.joint_speed_margin,
    weight=JOINT_SPEED_MARGIN_WEIGHT,
    params={"rated_speed": RATED_SPEED, "start_fraction": 0.8},
  )
  return cfg
