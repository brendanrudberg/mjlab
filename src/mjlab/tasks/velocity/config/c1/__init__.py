from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.cat import ConstraintCfg, make_cat_runner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import constraints as cstr
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  MOTOR_HEAT_WEIGHT,
  caltech_c1_flat_env_cfg,
  caltech_c1_flat_hw_penalty_env_cfg,
  caltech_c1_rough_env_cfg,
)
from .rl_cfg import caltech_c1_ppo_runner_cfg

FEET = "feet_ground_contact"

# Bolt / CaT constraint set, adapted to the planar C1.
# max_p 1.0 = hard; soft ones ramp from p = 0.05 up to max_p over ~1000 iterations.
C1_CONSTRAINTS = {
  # Safety, soft. No torque constraint in phase 1 (load extraction): the actuator
  # effort limit is the only cap. Add cstr.joint_torque for phase 2.
  "joint_velocity": ConstraintCfg(cstr.joint_velocity, {"limit": 16}),
  "action_rate": ConstraintCfg(cstr.action_rate, {"limit": 120.0}),
  # Safety, hard.
  "upside_down": ConstraintCfg(cstr.upside_down, max_p=1.0),
  # Style, soft. 0.3 = |sin(pitch)| ~ 17.5 degrees.
  "base_pitch": ConstraintCfg(cstr.base_pitch, {"limit": 0.3}),
  # Gait, soft.
  # "one_foot_contact": ConstraintCfg(
  #  cstr.n_foot_contact, {"sensor_name": FEET, "number_of_desired_feet": 1}, max_p=0.05
  # ),
  "air_time_min": ConstraintCfg(
    cstr.air_time_min, {"sensor_name": FEET, "limit": 0.15}
  ),
  "air_time_max": ConstraintCfg(cstr.air_time_max, {"sensor_name": FEET, "limit": 0.5}),
  "hip_bounds": ConstraintCfg(
    cstr.joint_bounds,
    {
      "lower": -1.15,
      "upper": 0.35,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint",)),
    },
  ),
  # Capped so a skidding fall can't dilute every ordinary slip's penalty.
  "foot_slip": ConstraintCfg(
    cstr.foot_slip,
    {
      "sensor_name": FEET,
      "limit": 0.1,
      "max_slip": 0.5,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot_contact", "right_foot_contact")
      ),
    },
  ),
  "swing_clearance": ConstraintCfg(
    cstr.swing_clearance,
    {
      "sensor_name": FEET,
      "min_height": 0.03,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot_contact", "right_foot_contact")
      ),
    },
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


def _hw_penalty_rl_cfg():
  cfg = caltech_c1_ppo_runner_cfg()
  # Log folder <timestamp>_hw_penalty_<weight>, e.g. ..._hw_penalty_0.3.
  cfg.run_name = f"hw_penalty_{abs(MOTOR_HEAT_WEIGHT):g}"
  return cfg


# Same as Flat-C1 plus rewards penalizing motor heat and near-rated joint speed.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-C1-HwPenalty",
  env_cfg=caltech_c1_flat_hw_penalty_env_cfg(),
  play_env_cfg=caltech_c1_flat_hw_penalty_env_cfg(play=True),
  rl_cfg=_hw_penalty_rl_cfg(),
  runner_cls=C1CaTRunner,
)
