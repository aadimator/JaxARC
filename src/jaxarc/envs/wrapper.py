"""Clean wrapper system with proper delegation (from Stoa)."""

from __future__ import annotations

from typing import Any

import jax

from jaxarc.types import EnvParams, TimeStep


class Wrapper:
    """Base wrapper class with clean delegation."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped environment."""
        return getattr(self._env, name)

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        return self._env.reset(params, key)

    def step(self, params: EnvParams, timestep: TimeStep, action: Any) -> TimeStep:
        return self._env.step(params, timestep, action)


class GymAutoResetWrapper(Wrapper):
    """Auto-reset on terminal states (simplified)."""

    def _auto_reset(self, params: EnvParams, timestep: TimeStep) -> TimeStep:
        key, _ = jax.random.split(timestep.state.key)
        reset_ts = self._env.reset(params, key)
        # Preserve terminal info, replace state/observation
        return TimeStep(
            state=reset_ts.state,
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=reset_ts.observation,
            extras=timestep.extras,
        )

    def step(self, params: EnvParams, timestep: TimeStep, action: Any) -> TimeStep:
        ts = self._env.step(params, timestep, action)
        return jax.lax.cond(ts.last(), lambda: self._auto_reset(params, ts), lambda: ts)


__all__ = ["GymAutoResetWrapper", "Wrapper"]
