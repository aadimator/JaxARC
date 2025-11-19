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
from jaxarc.envs.actions import Action, create_action
from jaxarc.envs.spaces import ARCActionSpace, BoundedArraySpace, DictSpace, GridSpace
from jaxarc.types import EnvParams, TimeStep
from jaxarc.utils.visualization.core import draw_grid_svg, render_rgb
from jaxarc.utils.visualization.display import render_ansi

from .functional import reset as functional_reset
from .functional import step as functional_step

if TYPE_CHECKING:
    from jaxarc.types import State


class Environment(stoa.environment.Environment):
    """
    JaxARC environment implementing Stoa API patterns.

    Delegates to functional API while providing clean object-oriented interface.
    """

    def __init__(
        self,
        config: JaxArcConfig,
        buffer: Any,
        episode_mode: int = 0,
        subset_indices: Any | None = None,
    ):
        self.params = EnvParams.from_config(
            config=config,
            episode_mode=episode_mode,
            buffer=buffer,
            subset_indices=subset_indices,
        )
        self.render_mode = config.environment.render_mode

    def observation_shape(self) -> tuple[int, int, int]:
        """Get observation shape."""
        return (
            int(self.params.dataset.max_grid_height),
            int(self.params.dataset.max_grid_width),
            1,
        )

    def reset(
        self, rng_key: jax.Array, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Reset using functional API (supports optional per-call params override)."""
        p = self.params if env_params is None else env_params
        state, timestep = functional_reset(p, rng_key)

        # Adding canconical action and operation_id to extras for logging/visualization
        # Ensure a stable, JAX-friendly extras schema for JIT/scan compatibility
        height = int(p.dataset.max_grid_height)
        width = int(p.dataset.max_grid_width)
        zero_sel = jnp.zeros((height, width), dtype=jnp.bool_)
        op_sentinel = jnp.array(-1, dtype=jnp.int32)

        base_extras = (
            timestep.extras
            if isinstance(getattr(timestep, "extras", None), dict)
            else {}
        )
        extras = dict(base_extras)
        # Canonical action present with static shapes; values are JAX arrays
        extras.setdefault(
            "canonical_action",
            {
                "operation": op_sentinel,
                "selection": zero_sel,
            },
        )
        # Convenience scalar mirrors canonical_action.operation
        extras.setdefault("operation_id", op_sentinel)

        # Rebuild timestep with enriched extras
        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=timestep.observation,
            extras=extras,
        )
        return state, new_timestep

    def step(
        self, state: State, action: Action | dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Step using functional API (supports optional per-call params override)."""
        p = self.params if env_params is None else env_params
        # Accept canonical dict-form mask actions from ARCActionSpace and convert to internal Action
        if (
            isinstance(action, dict)
            and ("operation" in action)
            and ("selection" in action)
        ):
            op = jnp.asarray(action["operation"], dtype=jnp.int32)
            sel = jnp.asarray(action["selection"], dtype=jnp.bool_)
            action = create_action(op, sel)

        next_state, timestep = functional_step(p, state, action)

        # Populate a stable, JAX-friendly extras schema based on the actual mask action
        height = int(p.dataset.max_grid_height)
        width = int(p.dataset.max_grid_width)
        zero_sel = jnp.zeros((height, width), dtype=jnp.bool_)
        op_sentinel = jnp.array(-1, dtype=jnp.int32)

        base_extras = (
            timestep.extras
            if isinstance(getattr(timestep, "extras", None), dict)
            else {}
        )
        extras = dict(base_extras)

        # Derive canonical mask-based action: operation and selection must be JAX arrays
        op_attr = getattr(action, "operation", None)
        sel_attr = getattr(action, "selection", None)
        op = op_sentinel if op_attr is None else jnp.asarray(op_attr, dtype=jnp.int32)
        sel = zero_sel if sel_attr is None else jnp.asarray(sel_attr, dtype=jnp.bool_)

        extras["canonical_action"] = {"operation": op, "selection": sel}
        extras["operation_id"] = op

        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=timestep.observation,
            extras=extras,
        )
        return next_state, new_timestep

    def state_space(self, _env_params: EnvParams | None = None) -> stoa.spaces.Space:
        """Return the state space of the environment."""
        height, width = self.observation_shape()
        return DictSpace(
            {
                "working_grid": GridSpace(max_height=height, max_width=width),
                "working_grid_mask": BoundedArraySpace(
                    shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "input_grid": GridSpace(max_height=height, max_width=width),
                "input_grid_mask": BoundedArraySpace(
                    shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "target_grid": GridSpace(max_height=height, max_width=width),
                "target_grid_mask": BoundedArraySpace(
                    shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "selected": BoundedArraySpace(
                    shape=(height, width), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "clipboard": GridSpace(max_height=height, max_width=width),
                "step_count": BoundedArraySpace(
                    shape=(),
                    dtype=jnp.int32,
                    minimum=0,
                    maximum=self.params.max_episode_steps,
                ),
                "task_idx": BoundedArraySpace(
                    shape=(),
                    dtype=jnp.int32,
                    minimum=0,
                    maximum=int(jnp.iinfo(jnp.int32).max),
                ),
                "pair_idx": BoundedArraySpace(
                    shape=(),
                    dtype=jnp.int32,
                    minimum=0,
                    maximum=int(jnp.iinfo(jnp.int32).max),
                ),
                "allowed_operations_mask": BoundedArraySpace(
                    shape=(35,), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "similarity_score": BoundedArraySpace(
                    shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0
                ),
                "key": BoundedArraySpace(
                    shape=(2,),
                    dtype=jnp.uint32,
                    minimum=0,
                    maximum=int(jnp.iinfo(jnp.uint32).max),
                ),
            }
        )

    def observation_space(
        self, _env_params: EnvParams | None = None
    ) -> BoundedArraySpace:
        """Get ARC observation space."""
        height, width, channels = self.observation_shape()
        return BoundedArraySpace(
            shape=(height, width, channels),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.params.dataset.max_colors,
            name="observation",
        )

    def action_space(self, _env_params: EnvParams | None = None) -> ARCActionSpace:
        """Get ARC action space."""
        height, width, _ = self.observation_shape()
        return ARCActionSpace(max_height=height, max_width=width)

    def reward_space(self, _env_params: EnvParams | None = None) -> BoundedArraySpace:
        """Get reward space."""
        return BoundedArraySpace(
            shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0
        )

    def discount_space(self, _env_params: EnvParams | None = None) -> BoundedArraySpace:
        """Get discount space."""
        return BoundedArraySpace(
            shape=(), dtype=jax.numpy.float32, minimum=0.0, maximum=1.0
        )

    @property
    def unwrapped(self) -> Environment:
        """Get the unwrapped environment."""
        return self

    def close(self) -> None:
        """Close the environment."""
        return

    def render(self, state: State, mode: str | None = None) -> Any:
        """
        Render the environment state.

        Args:
            state: The current environment state.
            mode: The rendering mode ("rgb_array", "ansi", "svg").
                  If None, uses the default mode from configuration.

        Returns:
            The rendered output (numpy array, string, or SVG string).
        """
        # Determine render mode
        render_mode = mode if mode is not None else self.render_mode

        # Dispatch to appropriate renderer
        if render_mode == "rgb_array":
            return render_rgb(state.working_grid)
        if render_mode == "ansi":
            return render_ansi(state.working_grid)
        if render_mode == "svg":
            drawing = draw_grid_svg(
                state.working_grid,
                state.working_grid_mask,
                label=f"Step {int(state.step_count)}",
                show_size=True,
            )
            # Ensure we have a Drawing object (default behavior of draw_grid_svg)
            if isinstance(drawing, tuple):
                drawing = drawing[0]
            return drawing.as_svg()
        raise ValueError(f"Unsupported render mode: {render_mode}")


__all__ = ["Environment"]
