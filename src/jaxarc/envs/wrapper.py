"""
Thin compatibility shim for the ARC environment wrapper.

This module replaces the previous heavy, stateful `ArcEnv` wrapper with a minimal
adapter that defers to the new, simple Environment interface implemented in
`environment.py` following the Xland-Minigrid pattern.

Key changes:
- The single source of truth is the functional API (`reset(params, key)`,
  `step(params, timestep, action)`).
- This shim constructs `EnvParams` once and delegates `reset`/`step` to a small
  `FunctionalArcEnvironment` instance (optionally wrapped with auto-reset).
- Batch usage is supported via JAX `vmap` without additional wrapper-level logic.

Note:
- The old `ArcEnv` interface that returned tuples (state, obs, reward, done, info)
  is removed. The new interface returns a single `TimeStep` object that embeds
  the state, reward, discount, step_type, and observation.
- No implicit PRNG management is provided here. Pass keys explicitly, including
  per-environment keys for batched usage.

Example:
    from jaxarc.envs.environment import Environment
    from jaxarc.envs.wrapper import GymAutoResetWrapper
    from jaxarc.types import EnvParams
    import jax

    # Build EnvParams from a project config and a pre-stacked task buffer (not shown)
    params = EnvParams.from_config(config, buffer=buffer, episode_mode=0)

    env = GymAutoResetWrapper(Environment())

    key = jax.random.PRNGKey(0)
    ts = env.reset(params, key)  # -> TimeStep
    ts = env.step(params, ts, action)  # -> TimeStep
"""

from __future__ import annotations

from typing import Any

import jax

from jaxarc.types import TimeStep


class Wrapper:
    """
    Base class for environment wrappers.

    Delegates all methods to the wrapped environment by default.
    """

    def __init__(self, env):
        self._env = env

    def default_params(self, **kwargs: Any):
        return self._env.default_params(**kwargs)

    def num_actions(self, params) -> int:
        return self._env.num_actions(params)

    def observation_shape(self, params) -> Any:
        return self._env.observation_shape(params)

    def reset(self, params, key) -> TimeStep:
        return self._env.reset(params, key)

    def step(self, params, timestep: TimeStep, action: Any) -> TimeStep:
        return self._env.step(params, timestep, action)

    def render(self, params, timestep: TimeStep) -> Any:
        return self._env.render(params, timestep)


class GymAutoResetWrapper(Wrapper):
    """
    Gym-style auto-reset: if a step returns LAST, automatically reset on the same call.

    The returned TimeStep retains the reward/discount/step_type from the terminal step,
    but state and observation are replaced with the freshly reset ones.
    """

    def _auto_reset(self, params, timestep: TimeStep) -> TimeStep:
        key, _ = jax.random.split(timestep.state.key)
        reset_ts = self._env.reset(params, key)
        # Preserve terminal reward/discount/step_type; replace state/observation for continued rollout
        return TimeStep(
            state=reset_ts.state,
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=reset_ts.observation,
        )

    def step(self, params, timestep: TimeStep, action: Any) -> TimeStep:
        ts = self._env.step(params, timestep, action)
        ts = jax.lax.cond(
            ts.last(),
            lambda: self._auto_reset(params, ts),
            lambda: ts,
        )
        return ts


class DmEnvAutoResetWrapper(Wrapper):
    """
    dm_env/envpool-style auto-reset: if previous step was LAST, call reset instead of step.
    """

    def step(self, params, timestep: TimeStep, action: Any) -> TimeStep:
        ts = jax.lax.cond(
            timestep.last(),
            lambda: self._env.reset(params, timestep.state.key),
            lambda: self._env.step(params, timestep, action),
        )
        return ts


__all__ = ["DmEnvAutoResetWrapper", "GymAutoResetWrapper", "Wrapper"]
