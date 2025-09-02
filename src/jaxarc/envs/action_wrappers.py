"""
Action wrappers for JaxARC environments.

This module implements the separation of concerns principle for the JaxARC action system:
- **Core Environment**: Only knows about MaskAction objects (mask-based selections)
- **Action Wrappers**: Handle conversion from user-friendly formats to masks
- PointActionWrapper: Converts (operation, row, col) tuples to mask actions
- BboxActionWrapper: Converts (operation, r1, c1, r2, c2) tuples to mask actions

Usage:
    ```python
    from jaxarc.registration import make
    from jaxarc.envs.action_wrappers import BboxActionWrapper

    # Create base environment (handles MaskAction only)
    env, env_params = make("Mini")

    # Wrap with action wrapper (converts bbox to mask)
    env = BboxActionWrapper(env)

    # Use normal environment API with bbox actions
    timestep = env.reset(env_params, key)
    action = (15, 2, 3, 7, 8)  # (operation, r1, c1, r2, c2)
    timestep = env.step(env_params, timestep, action)

    # The wrapper handles the conversion:
    # bbox (15, 2, 3, 7, 8) -> MaskAction with rectangular mask -> core environment
    ```
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..types import EnvParams, TimeStep
from .actions import MaskAction, create_mask_action
from .environment import Environment


def _point_to_mask(point_action: tuple, grid_shape: tuple[int, int]) -> MaskAction:
    """Convert point action tuple to mask action.

    Args:
        point_action: Tuple of (operation, row, col)
        grid_shape: Shape of the grid (height, width)

    Returns:
        MaskAction with single point selected
    """
    operation, row, col = point_action
    height, width = grid_shape

    # Create mask with single point
    mask = jnp.zeros((height, width), dtype=jnp.bool_)

    # Clip coordinates to valid range
    valid_row = jnp.clip(row, 0, height - 1)
    valid_col = jnp.clip(col, 0, width - 1)

    # Set the point in the mask
    mask = mask.at[valid_row, valid_col].set(True)

    return create_mask_action(operation, mask)


def _bbox_to_mask(bbox_action: tuple, grid_shape: tuple[int, int]) -> MaskAction:
    """Convert bounding box action tuple to mask action.

    Args:
        bbox_action: Tuple of (operation, r1, c1, r2, c2)
        grid_shape: Shape of the grid (height, width)

    Returns:
        MaskAction with rectangular region selected
    """
    operation, r1, c1, r2, c2 = bbox_action
    height, width = grid_shape

    # Clip coordinates to valid range
    r1 = jnp.clip(r1, 0, height - 1)
    c1 = jnp.clip(c1, 0, width - 1)
    r2 = jnp.clip(r2, 0, height - 1)
    c2 = jnp.clip(c2, 0, width - 1)

    # Ensure proper ordering (min, max)
    min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
    min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)

    # Create coordinate meshes
    rows = jnp.arange(height)
    cols = jnp.arange(width)
    row_mesh, col_mesh = jnp.meshgrid(rows, cols, indexing="ij")

    # Create bbox mask (inclusive bounds)
    mask = (
        (row_mesh >= min_r)
        & (row_mesh <= max_r)
        & (col_mesh >= min_c)
        & (col_mesh <= max_c)
    )

    return create_mask_action(operation, mask)


# Generic JIT versions using static_argnums for better performance
_jit_point_to_mask = jax.jit(_point_to_mask, static_argnums=1)
_jit_bbox_to_mask = jax.jit(_bbox_to_mask, static_argnums=1)


class PointActionWrapper:
    """Wrapper that converts point actions to mask actions.

    This wrapper allows users to provide actions as simple (operation, row, col)
    tuples, which are automatically converted to the mask format expected by
    the core environment.

    Examples:
        ```python
        from jaxarc.registration import make
        from jaxarc.envs.action_wrappers import PointActionWrapper
        import jax

        # Create environment with wrapper
        env, env_params = make("Mini")
        env = PointActionWrapper(env)

        # Use point actions - much simpler than creating masks
        key = jax.random.PRNGKey(42)
        timestep = env.reset(env_params, key)

        action = (15, 5, 10)  # operation=15, row=5, col=10
        new_timestep = env.step(env_params, timestep, action)
        ```
    """

    def __init__(self, env: Environment):
        """Initialize the point action wrapper.

        Args:
            env: Base environment to wrap
        """
        self.env = env

    def default_params(
        self,
        *,
        config: Any,
        buffer: Any,
        episode_mode: int = 0,
        subset_indices: Any | None = None,
    ) -> EnvParams:
        """Create default environment parameters.

        Forwards to the wrapped environment.
        """
        return self.env.default_params(
            config=config,
            buffer=buffer,
            episode_mode=episode_mode,
            subset_indices=subset_indices,
        )

    def observation_shape(self, params: EnvParams) -> tuple[int, int]:
        """Return the shape of observations.

        Forwards to the wrapped environment.
        """
        return self.env.observation_shape(params)

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        """Reset the environment.

        Forwards to the wrapped environment.
        """
        return self.env.reset(params, key)

    def step(self, params: EnvParams, timestep: TimeStep, action: tuple) -> TimeStep:
        """Execute a point action step.

        Args:
            params: Environment parameters
            timestep: Current environment timestep
            action: Point action as (operation, row, col) tuple

        Returns:
            New environment timestep after action execution
        """
        # Get grid shape from current state
        state = timestep.state
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])

        # Convert point action to mask action using JIT-compiled version
        mask_action = _jit_point_to_mask(action, grid_shape)

        # Execute using wrapped environment
        return self.env.step(params, timestep, mask_action)

    def render(self, params: EnvParams, timestep: TimeStep) -> Any:
        """Render the environment.

        Forwards to the wrapped environment.
        """
        return self.env.render(params, timestep)


class BboxActionWrapper:
    """Wrapper that converts bounding box actions to mask actions.

    This wrapper allows users to provide actions as (operation, r1, c1, r2, c2)
    tuples, which are automatically converted to the mask format expected by
    the core environment.

    Examples:
        ```python
        from jaxarc.registration import make
        from jaxarc.envs.action_wrappers import BboxActionWrapper
        import jax

        # Create environment with wrapper
        env, env_params = make("Mini")
        env = BboxActionWrapper(env)

        # Use bbox actions - specify rectangular regions easily
        key = jax.random.PRNGKey(42)
        timestep = env.reset(env_params, key)

        action = (15, 2, 3, 7, 8)  # operation=15, bbox from (2,3) to (7,8)
        new_timestep = env.step(env_params, timestep, action)
        ```
    """

    def __init__(self, env: Environment):
        """Initialize the bbox action wrapper.

        Args:
            env: Base environment to wrap
        """
        self.env = env

    def default_params(
        self,
        *,
        config: Any,
        buffer: Any,
        episode_mode: int = 0,
        subset_indices: Any | None = None,
    ) -> EnvParams:
        """Create default environment parameters.

        Forwards to the wrapped environment.
        """
        return self.env.default_params(
            config=config,
            buffer=buffer,
            episode_mode=episode_mode,
            subset_indices=subset_indices,
        )

    def observation_shape(self, params: EnvParams) -> tuple[int, int]:
        """Return the shape of observations.

        Forwards to the wrapped environment.
        """
        return self.env.observation_shape(params)

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        """Reset the environment.

        Forwards to the wrapped environment.
        """
        return self.env.reset(params, key)

    def step(self, params: EnvParams, timestep: TimeStep, action: tuple) -> TimeStep:
        """Execute a bounding box action step.

        Args:
            params: Environment parameters
            timestep: Current environment timestep
            action: Bbox action as (operation, r1, c1, r2, c2) tuple

        Returns:
            New environment timestep after action execution
        """
        # Get grid shape from current state
        state = timestep.state
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])

        # Convert bbox action to mask action using JIT-compiled version
        mask_action = _jit_bbox_to_mask(action, grid_shape)

        # Execute using wrapped environment
        return self.env.step(params, timestep, mask_action)

    def render(self, params: EnvParams, timestep: TimeStep) -> Any:
        """Render the environment.

        Forwards to the wrapped environment.
        """
        return self.env.render(params, timestep)


__all__ = [
    "BboxActionWrapper",
    "PointActionWrapper",
]
