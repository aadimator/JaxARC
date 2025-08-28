"""
Minimal environment delegate for JaxARC following the Xland-Minigrid pattern.

This module intentionally keeps only a small, JAX-friendly environment delegate
(`Environment`) that forwards calls to the functional core:
- reset(params, key) -> TimeStep
- step(params, timestep, action) -> TimeStep

Design changes:
- Removed the abstract Environment base class to avoid unnecessary indirection.
- Moved auto-reset wrappers to the wrapper module (`jaxarc.envs.wrapper`).
- Kept only convenience methods that are useful for agents (num_actions, observation_shape).

Typical usage:
    from jaxarc.envs.environment import Environment
    from jaxarc.types import EnvParams
    from jaxarc.configs.main_config import JaxArcConfig
    from jaxarc.types import JaxArcTask
    import jax

    params = EnvParams.from_config(config, task_data, episode_mode=0, pair_idx=0)
    env = Environment()

    key = jax.random.PRNGKey(0)
    timestep = env.reset(params, key)
    timestep = env.step(params, timestep, action)

Notes:
- The functional API in `jaxarc.envs.functional` remains the single source of truth.
- Wrappers such as Gym/DmEnv auto-reset now live in `jaxarc.envs.wrapper`.
"""

from __future__ import annotations

from typing import Any, Dict

import jax

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.types import EnvParams, JaxArcTask, TimeStep
from jaxarc.utils.jax_types import NUM_OPERATIONS

from .actions import StructuredAction
from .functional import reset as functional_reset
from .functional import step as functional_step


class Environment:
    """
    Minimal environment that directly delegates to the functional core.

    - default_params: builds EnvParams from config + task_data
    - reset/step: directly call functional.reset/functional.step
    - num_actions/observation_shape: convenience helpers
    """

    # -------------------------------------------------------------------------
    # Convenience API
    # -------------------------------------------------------------------------

    def default_params(
        self,
        *,
        config: JaxArcConfig,
        task_data: JaxArcTask,
        episode_mode: int = 0,
        pair_idx: int = 0,
    ) -> EnvParams:
        return EnvParams.from_config(
            config=config,
            task_data=task_data,
            episode_mode=episode_mode,
            pair_idx=pair_idx,
        )

    def num_actions(self, params: EnvParams) -> int:
        """
        Return the number of discrete operations available.

        For ARC, the action set is a selection plus an operation id in [0, NUM_OPERATIONS).
        Returning NUM_OPERATIONS is a useful default for discrete policies over operations.
        """
        return int(NUM_OPERATIONS)

    def observation_shape(self, params: EnvParams) -> tuple[int, int]:
        """
        Return the shape of observations for convenience.

        For ARC, observations typically match the working grid shape. We fall back to the
        dataset's configured maximum HxW (static) shape.
        """
        return (int(params.dataset.max_grid_height), int(params.dataset.max_grid_width))

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        """
        Reset the environment. Must be JAX-compatible and return a single TimeStep.
        """
        return functional_reset(params, key)

    def step(
        self,
        params: EnvParams,
        timestep: TimeStep,
        action: StructuredAction | Dict[str, jax.Array] | Any,
    ) -> TimeStep:
        """
        Step the environment. Must be JAX-compatible and return a single TimeStep.
        """
        return functional_step(params, timestep, action)

    def render(self, params: EnvParams, timestep: TimeStep) -> Any:
        """
        Optional rendering hook. Not implemented by default.
        """
        raise NotImplementedError("render is not implemented for this environment.")


__all__ = [
    "Environment",
]
