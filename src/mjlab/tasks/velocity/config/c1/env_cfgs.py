"""Caltech C1 velocity environment configurations.

Start with mjlab's shared velocity task and override C1-specific settings.
Reward overrides are grouped by purpose; unspecified terms retain factory defaults.
"""

import math

from mjlab.asset_zoo.robots import C1_ACTION_SCALE, get_c1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def caltech_c1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the C1 rough-terrain task with robot-specific overrides."""
  cfg = make_velocity_env_cfg()

  base_body = "base_link"

  foot_sites = ("left_foot_contact", "right_foot_contact")
  foot_geoms = ("L_foot_geom", "R_foot_geom")

  # ---------------------------------------------------------------------------
  # Scene and simulation.
  # ---------------------------------------------------------------------------
  cfg.scene.entities = {"robot": get_c1_robot_cfg()}
  cfg.viewer.body_name = base_body

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # ---------------------------------------------------------------------------
  # Sensors: terrain observations, foot clearance/contact, and self-contact.
  # ---------------------------------------------------------------------------
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = base_body
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site, entity="robot") for site in foot_sites
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=base_body, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=base_body, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # ---------------------------------------------------------------------------
  # Agent interface: joint-position actions and ground-truth base observations.
  # ---------------------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = C1_ACTION_SCALE

  # The C1 XML has no IMU sensors yet.
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

  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase, params={"period": 0.8}
    )
  

  # ---------------------------------------------------------------------------
  # Commands and curricula: forward motion with increasing target speed.
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

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.3, 1.0)},
    {"step": 5000 * 24, "lin_vel_x": (0.3, 1.5)},
  ]
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # ---------------------------------------------------------------------------
  # Rewards: retain inherited terms and tune only the C1-specific overrides.
  # ---------------------------------------------------------------------------
  rewards = cfg.rewards

  # 1. Task objective: track the commanded linear velocity.
  # Keep the factory's track_linear_velocity term unchanged.
  # Preserve the existing choice to omit direct yaw-rate tracking.
  rewards.pop("track_angular_velocity", None)
  w = rewards["track_linear_velocity"].weight
  rewards.pop("track_linear_velocity")
  rewards["track_smoothed_linear_velocity"] = RewardTermCfg(
    func=mdp.track_smoothed_linear_velocity,
    weight=w,
    params={"command_name": "twist", "std": 0.5, "alpha": 0.025},
  )

  # 2. Posture and stability: stay upright and limit body rotation.
  rewards["upright"].weight = 0.3
  rewards["upright"].params["asset_cfg"].body_names = (base_body,)

  rewards["body_ang_vel"].weight = -0.05
  rewards["body_ang_vel"].params["asset_cfg"].body_names = (base_body,)

  # Allow more joint deviation as commanded speed increases.
  cfg.rewards["pose"].weight = 0.2
  rewards["pose"].params["std_standing"] = {".*": 0.05}
  rewards["pose"].params["std_walking"] = {
    r".*hip.*": 0.3,
    r".*knee.*": 0.35,
  }
  rewards["pose"].params["std_running"] = {
    r".*hip.*": 0.5,
    r".*knee.*": 0.6,
  }
  rewards.pop("angular_momentum", None)

  # 3. Gait and ground contact: air time, foot clearance, and slip.
  rewards.pop("air_time", None)

  for reward_name in ("foot_clearance", "foot_slip"):
    rewards[reward_name].params["asset_cfg"].site_names = foot_sites

  cfg.rewards["foot_slip"].weight = -1.0

  # 4. Regularization and collision avoidance.
  # Keep inherited dof_pos_limits and action_rate_l2 settings unchanged.
  rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # 5. Foots swapping position between steps.
  # Custom function in rewards.py
  rewards["forward_step"] = RewardTermCfg(
    func=mdp.forward_step,
    weight=50.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "min_step": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
    },
  )

  # 6. Reward for following
  rewards["gait_contact"] = RewardTermCfg(
    func=mdp.gait_contact,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "period": 0.8,
      "stance_frac": 0.6,
    },
  )
  # ---------------------------------------------------------------------------
  # Domain randomization: bind inherited events to C1 geometry and bodies.
  # ---------------------------------------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = (base_body,)
  cfg.events["reset_base"].params["velocity_range"]["x"] = (0.0, 0.6)

  # ---------------------------------------------------------------------------
  # Play mode: long episodes, clean observations, and randomized terrain resets.
  # ---------------------------------------------------------------------------
  if play:
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

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def caltech_c1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Derive the flat-terrain task from the shared C1 rough configuration."""
  cfg = caltech_c1_rough_env_cfg(play=play)

  # Simulation overrides for flat terrain.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Replace generated terrain with a plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove terrain_scan and its observations; retain foot_height_scan.
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Flat-terrain termination and curriculum overrides.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(35.0)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
