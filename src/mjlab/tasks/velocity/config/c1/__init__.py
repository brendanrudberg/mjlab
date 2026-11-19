from mjlab.rl.cat import ConstraintCfg, make_cat_runner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import constraints as cstr
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .env_cfgs import (
  caltech_c1_flat_env_cfg,
  caltech_c1_rough_env_cfg,
)
from .rl_cfg import caltech_c1_ppo_runner_cfg 

FEET = "feet_ground_contact"

# Bolt / CaT constraint set, adapted to the planar C1.
# max_p 1.0 = hard; soft ones ramp from p = 0.05 up to max_p over ~1000 iterations.
C1_CONSTRAINTS = {
  # Safety, soft. No torque constraint in phase 1 (load extraction): the actuator
  # effort limit is the only cap. Add cstr.joint_torque for phase 2.
  "joint_velocity": ConstraintCfg(cstr.joint_velocity, {"limit": 8}),
  "action_rate": ConstraintCfg(cstr.action_rate, {"limit": 80.0}),
  # Safety, hard.
  "upside_down": ConstraintCfg(cstr.upside_down, max_p=1.0),
  # Style, soft. 0.2 = |sin(pitch)| ~ 11.5 degrees.
  "base_pitch": ConstraintCfg(cstr.base_pitch, {"limit": 0.2}),
  # Gait, soft.
  "one_foot_contact": ConstraintCfg(
    cstr.n_foot_contact, {"sensor_name": FEET, "number_of_desired_feet": 1}
  ),
  "air_time_min": ConstraintCfg(cstr.air_time_min, {"sensor_name": FEET, "limit": 0.15}),
  "air_time_max": ConstraintCfg(cstr.air_time_max, {"sensor_name": FEET, "limit": 0.5}),
  "hip_range": ConstraintCfg(
    cstr.joint_range,
    {"limit": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint",))},
  ),
  "knee_range": ConstraintCfg(
    cstr.joint_range,
    {"limit": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
  ),
  "foot_slip": ConstraintCfg(
    cstr.foot_slip,
    {"sensor_name": FEET, "limit": 0.1,
     "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot_contact", "right_foot_contact"))},
  ),
}

C1CaTRunner = make_cat_runner(VelocityOnPolicyRunner, C1_CONSTRAINTS)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-C1",
  env_cfg=caltech_c1_rough_env_cfg(),
  play_env_cfg=caltech_c1_rough_env_cfg(play=True),
  rl_cfg=caltech_c1_ppo_runner_cfg(),
  runner_cls=C1CaTRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-C1",
  env_cfg=caltech_c1_flat_env_cfg(),
  play_env_cfg=caltech_c1_flat_env_cfg(play=True),
  rl_cfg=caltech_c1_ppo_runner_cfg(),
  runner_cls=C1CaTRunner,
)