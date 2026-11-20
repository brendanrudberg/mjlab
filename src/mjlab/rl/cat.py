"""Constraints as Terminations (CaT) for rsl_rl PPO.

Chane-Sane et al., "CaT: Constraints as Terminations for Legged Locomotion
Reinforcement Learning", IROS 2024. Ported from
github.com/gepetto/constraints-as-terminations (Isaac Lab version).

Each constraint returns a violation c (> 0 means violated). Per step:
  delta = max_i  p_max_i * clip(c_i / c_max_i, 0, 1)    (0 where c_i <= 0)
  reward <- max(reward * (1 - delta), 0)
  done   <- max(done, delta)                            (only inside PPO's returns)
Soft constraints never reset the episode; real resets still come from the
termination manager. c_max_i is a moving average of the largest violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


@dataclass
class ConstraintCfg:
  func: Callable[..., torch.Tensor]
  """f(env, **params) -> violation, shape [B] or [B, k]. > 0 means violated."""
  params: dict[str, Any] = field(default_factory=dict)
  max_p: float = 0.25
  """Final maximum termination probability. 1.0 = hard constraint."""
  ramp: bool = True
  """Soft constraints: ramp max_p up from 1/start_horizon over ramp_steps
  (CaT's curriculum). Ignored when max_p == 1.0."""


class CaT:
  """Computes the per-env termination probability delta from constraints."""

  def __init__(
    self,
    constraints: dict[str, ConstraintCfg],
    device: str | torch.device,
    tau: float = 0.95,
    ramp_steps: int = 24 * 1000,
    start_horizon: float = 20.0,
  ):
    self.constraints = constraints
    self.device = device
    self.tau = tau
    self.ramp_steps = ramp_steps
    self.start_horizon = start_horizon
    self.running_max: dict[str, torch.Tensor] = {}
    self.last_probs: dict[str, torch.Tensor] = {}

  def current_max_p(self, cfg: ConstraintCfg, step: int) -> float:
    # CaT curriculum: the expected time until termination (1 / p) moves
    # linearly from start_horizon steps to 1 / max_p steps.
    if cfg.max_p >= 1.0 or not cfg.ramp:
      return cfg.max_p
    progress = min(step / self.ramp_steps, 1.0)
    horizon = self.start_horizon + progress * (1.0 / cfg.max_p - self.start_horizon)
    return 1.0 / horizon

  def compute(self, env: Any) -> torch.Tensor:
    step = int(env.common_step_counter)
    delta = torch.zeros(env.num_envs, device=self.device)
    for name, cfg in self.constraints.items():
      if name not in self.running_max:  # first call: resolve names to indices
        for value in cfg.params.values():
          if isinstance(value, SceneEntityCfg):
            value.resolve(env.scene)
      c = cfg.func(env, **cfg.params).float()
      if c.ndim == 1:
        c = c.unsqueeze(1)
      batch_max = c.max(dim=0, keepdim=True).values.clamp(min=1e-6)
      if name in self.running_max:
        self.running_max[name].mul_(self.tau).add_((1.0 - self.tau) * batch_max)
      else:
        self.running_max[name] = batch_max.clone()
      p_max = self.current_max_p(cfg, step)
      normalized = torch.clamp(c / self.running_max[name], 0.0, 1.0)
      probs = torch.where(c > 0.0, p_max * normalized, torch.zeros_like(c))
      probs = probs.max(dim=1).values  # [B]
      self.last_probs[name] = probs
      delta = torch.maximum(delta, probs)
    return delta


class CaTVecEnvWrapper:
  """Wraps an RslRlVecEnvWrapper and adds extras["cat_delta"] every step."""

  def __init__(self, env: RslRlVecEnvWrapper, cat: CaT):
    object.__setattr__(self, "_env", env)
    object.__setattr__(self, "_cat", cat)

  def step(self, actions: torch.Tensor):
    obs, rew, dones, extras = self._env.step(actions)
    delta = self._cat.compute(self._env.unwrapped)
    extras["cat_delta"] = delta
    log = extras.setdefault("log", {})
    for name, probs in self._cat.last_probs.items():
      log[f"Constraint_violation/{name}"] = (probs > 0).float().mean()
      log[f"Constraint_prob/{name}"] = probs.mean()
    log["Constraint_prob/delta"] = delta.mean()
    robot = self._env.unwrapped.scene["robot"]
    log["Debug/max_joint_vel"] = robot.data.joint_vel.abs().max()
    return obs, rew, dones, extras

  # Forward everything else to the wrapped env, including writes such as
  # runner.env.episode_length_buf = ... (init_at_random_ep_len).
  def __getattr__(self, name: str):
    return getattr(self._env, name)

  def __setattr__(self, name: str, value: Any) -> None:
    setattr(self._env, name, value)


class CaTPPO(PPO):
  """PPO that applies CaT's reward scaling and fractional dones."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # rsl_rl stores dones as bytes; fractional delta needs floats.
    self.storage.dones = self.storage.dones.float()

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    delta = extras.get("cat_delta")
    if delta is not None:
      delta = delta.to(rewards.device)
      rewards = torch.clamp(rewards * (1.0 - delta), min=0.0)
      dones = torch.maximum(dones.float(), delta)
    super().process_env_step(obs, rewards, dones, extras)


def make_cat_runner(
  base_runner_cls: type[OnPolicyRunner],
  constraints: dict[str, ConstraintCfg],
  **cat_kwargs,
):
  """Return a runner class that trains with CaT. Use as runner_cls when registering a task."""

  class CaTRunner(base_runner_cls):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
      train_cfg["algorithm"]["class_name"] = "mjlab.rl.cat:CaTPPO"
      env = CaTVecEnvWrapper(
        env, CaT(constraints, device=env.unwrapped.device, **cat_kwargs)
      )
      super().__init__(env, train_cfg, log_dir, device, **kwargs)

  CaTRunner.__name__ = f"CaT{base_runner_cls.__name__}"
  return CaTRunner
