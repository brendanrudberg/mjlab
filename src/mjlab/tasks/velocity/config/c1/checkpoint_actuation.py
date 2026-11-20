"""Evaluate a C1 checkpoint with the actuation it was trained with.

The C1 gain set and action scale are switched in code (c1_constants.C1_GAINS),
so a script that builds the env from the current code can play a checkpoint with
the wrong gains. These helpers read the gains and action scale from the run's
params/env.yaml and apply them to an env config, on the current X8-120 actuator
model.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mjlab.asset_zoo.robots.caltech_c1 import c1_constants as c1
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg


class _RunCfgLoader(yaml.SafeLoader):
  """Reads params/env.yaml without importing or constructing python objects."""


def _plain(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node):
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  assert isinstance(node, yaml.ScalarNode)
  return loader.construct_scalar(node)


_RunCfgLoader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)


@dataclass(frozen=True)
class TrainedActuation:
  gains: dict[str, tuple[float, float]]
  """(stiffness, damping) by joint group, "hip" and "knee"."""
  action_scale: dict[str, float]
  dc_curve: bool
  """True if the run trained on the X8-120 speed-torque model."""

  def describe(self) -> str:
    hip, knee = self.gains["hip"], self.gains["knee"]
    text = f"hip {hip[0]:g}/{hip[1]:g}, knee {knee[0]:g}/{knee[1]:g}, "
    text += f"scale {self.action_scale}"
    if not self.dc_curve:
      text += " [trained without the X8-120 curve]"
    return text

  def joint_gains(self, joint_names: tuple[str, ...] | list[str]):
    """Per-joint (kp, kd) lists in the given joint order."""
    groups = ["hip" if "hip" in n else "knee" for n in joint_names]
    return [self.gains[g][0] for g in groups], [self.gains[g][1] for g in groups]


def read_trained_actuation(ckpt: Path) -> TrainedActuation:
  path = Path(ckpt).parent / "params" / "env.yaml"
  with path.open() as fh:
    cfg = yaml.load(fh, Loader=_RunCfgLoader)
  actuators = cfg["scene"]["entities"]["robot"]["articulation"]["actuators"]
  gains = {}
  for act in actuators:
    for expr in act["target_names_expr"]:
      group = "hip" if "hip" in expr else "knee"
      gains[group] = (float(act["stiffness"]), float(act["damping"]))
  scale = {k: float(v) for k, v in cfg["actions"]["joint_pos"]["scale"].items()}
  dc_curve = all("motor_params" in act for act in actuators)
  return TrainedActuation(gains=gains, action_scale=scale, dc_curve=dc_curve)


def apply_trained_actuation(
  env_cfg: ManagerBasedRlEnvCfg, ckpt: Path
) -> TrainedActuation:
  """Set the robot's actuators and action scale to those the checkpoint trained
  with, and print what was applied."""
  trained = read_trained_actuation(ckpt)
  robot_cfg = env_cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation.actuators = tuple(
    c1.x8_120_actuator(group, *trained.gains[group]) for group in ("hip", "knee")
  )
  action_cfg = env_cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  action_cfg.scale = dict(trained.action_scale)
  print(f"[checkpoint actuation] {trained.describe()}")
  return trained
