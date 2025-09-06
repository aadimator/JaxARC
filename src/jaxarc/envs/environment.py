"""
JaxARC environment following Stoa API patterns.

Concrete implementation that delegates to functional.py with Stoa-compatible interface.
"""

from __future__ import annotations

from typing import Any

import jax

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.envs.spaces import ARCActionSpace, BoundedArraySpace, GridSpace
from jaxarc.types import EnvParams, TimeStep

from .functional import reset as functional_reset
from .functional import step as functional_step


class Environment:
    """
    JaxARC environment implementing Stoa API patterns.
    
    Delegates to functional API while providing clean object-oriented interface.
    """

    def default_params(self, *, config: JaxArcConfig, buffer: Any, 
                      episode_mode: int = 0, subset_indices: Any | None = None) -> EnvParams:
        """Build EnvParams from config."""
        return EnvParams.from_config(
            config=config, episode_mode=episode_mode,
            buffer=buffer, subset_indices=subset_indices,
        )

    def observation_shape(self, params: EnvParams) -> tuple[int, int]:
        """Get observation shape."""
        return (int(params.dataset.max_grid_height), int(params.dataset.max_grid_width))

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        """Reset using functional API."""
        return functional_reset(params, key)

    def step(self, params: EnvParams, timestep: TimeStep, action: Any) -> TimeStep:
        """Step using functional API."""
        return functional_step(params, timestep, action)

    def observation_space(self, params: EnvParams) -> GridSpace:
        """Get ARC observation space."""
        height, width = self.observation_shape(params)
        return GridSpace(max_height=height, max_width=width)

    def action_space(self, params: EnvParams) -> ARCActionSpace:
        """Get ARC action space."""
        height, width = self.observation_shape(params)
        return ARCActionSpace(max_height=height, max_width=width)

    def reward_space(self, params: EnvParams) -> BoundedArraySpace:
        """Get reward space."""
        del params  # Unused for Stoa compatibility
        return BoundedArraySpace(shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0)

    def discount_space(self, params: EnvParams) -> BoundedArraySpace:
        """Get discount space.""" 
        del params  # Unused for Stoa compatibility
        return BoundedArraySpace(shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0)

    @property
    def unwrapped(self) -> Environment:
        """Get the unwrapped environment."""
        return self

    def close(self) -> None:
        """Close the environment."""


__all__ = ["Environment"]
