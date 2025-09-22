"""
JaxARC environment following Stoa API patterns.

Concrete implementation that delegates to functional.py with Stoa-compatible interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import stoa.environment

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.envs.actions import Action
from jaxarc.envs.spaces import ARCActionSpace, BoundedArraySpace, DictSpace, GridSpace
from jaxarc.types import EnvParams

from .functional import reset as functional_reset
from .functional import step as functional_step

if TYPE_CHECKING:
    from jaxarc.types import State, TimeStep


class Environment(stoa.environment.Environment):
    """
    JaxARC environment implementing Stoa API patterns.
    
    Delegates to functional API while providing clean object-oriented interface.
    """

    def __init__(self, config: JaxArcConfig, buffer: Any, episode_mode: int = 0, subset_indices: Any | None = None):
        self.params = EnvParams.from_config(
            config=config,
            episode_mode=episode_mode,
            buffer=buffer,
            subset_indices=subset_indices,
        )

    def observation_shape(self) -> tuple[int, int]:
        """Get observation shape."""
        return (int(self.params.dataset.max_grid_height), int(self.params.dataset.max_grid_width))

    def reset(self, rng_key: jax.Array, env_params: EnvParams | None = None) -> tuple[State, TimeStep]:
        """Reset using functional API (supports optional per-call params override)."""
        p = self.params if env_params is None else env_params
        return functional_reset(p, rng_key)

    def step(self, state: State, action: Action, env_params: EnvParams | None = None) -> tuple[State, TimeStep]:
        """Step using functional API (supports optional per-call params override)."""
        p = self.params if env_params is None else env_params
        return functional_step(p, state, action)

    def state_space(self, _env_params: EnvParams | None = None) -> stoa.spaces.Space:
        """Return the state space of the environment."""
        height, width = self.observation_shape()
        return DictSpace(
            {
                "working_grid": GridSpace(max_height=height, max_width=width),
                "working_grid_mask": BoundedArraySpace(shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True),
                "input_grid": GridSpace(max_height=height, max_width=width),
                "input_grid_mask": BoundedArraySpace(shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True),
                "target_grid": GridSpace(max_height=height, max_width=width),
                "target_grid_mask": BoundedArraySpace(shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True),
                "selected": BoundedArraySpace(shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True),
                "clipboard": GridSpace(max_height=height, max_width=width),
                "step_count": BoundedArraySpace(shape=(), dtype=jnp.int32, minimum=0, maximum=self.params.max_episode_steps),
                "task_idx": BoundedArraySpace(shape=(), dtype=jnp.int32, minimum=0, maximum=int(jnp.iinfo(jnp.int32).max)),
                "pair_idx": BoundedArraySpace(shape=(), dtype=jnp.int32, minimum=0, maximum=int(jnp.iinfo(jnp.int32).max)),
                "allowed_operations_mask": BoundedArraySpace(shape=(35,), dtype=jnp.bool_, minimum=False, maximum=True),
                "similarity_score": BoundedArraySpace(shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0),
                "key": BoundedArraySpace(shape=(2,), dtype=jnp.uint32, minimum=0, maximum=int(jnp.iinfo(jnp.uint32).max)),
            }
        )

    def observation_space(self, _env_params: EnvParams | None = None) -> GridSpace:
        """Get ARC observation space."""
        height, width = self.observation_shape()
        return GridSpace(max_height=height, max_width=width)

    def action_space(self, _env_params: EnvParams | None = None) -> ARCActionSpace:
        """Get ARC action space."""
        height, width = self.observation_shape()
        return ARCActionSpace(max_height=height, max_width=width)

    def reward_space(self, _env_params: EnvParams | None = None) -> BoundedArraySpace:
        """Get reward space."""
        return BoundedArraySpace(shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0)

    def discount_space(self, _env_params: EnvParams | None = None) -> BoundedArraySpace:
        """Get discount space.""" 
        return BoundedArraySpace(shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0)

    @property
    def unwrapped(self) -> Environment:
        """Get the unwrapped environment."""
        return self

    def close(self) -> None:
        """Close the environment."""
        return

    def render(self, *_args: Any, **_kwargs: Any) -> None:
        """Basic render stub to satisfy abstract base; users use visualization utils externally."""
        return


__all__ = ["Environment"]
