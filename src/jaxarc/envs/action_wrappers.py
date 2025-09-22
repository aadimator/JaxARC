"""
Action wrappers for JaxARC environments (simplified with clean delegation).

This module implements clean action wrappers following Stoa delegation patterns:
- **Core Environment**: Only knows about Action objects (mask-based selections)
- **Action Wrappers**: Handle conversion from user-friendly formats to masks, using
  clean delegation to remove boilerplate

- PointActionWrapper: Converts {"operation": op, "row": r, "col": c} dicts to mask actions
- BboxActionWrapper: Converts {"operation": op, "r1": r1, "c1": c1, "r2": r2, "c2": c2} dicts to mask actions

Usage:
    ```python
    from jaxarc.registration import make
    from jaxarc.envs.action_wrappers import BboxActionWrapper

    # Create base environment (handles Action only)
    env, env_params = make("Mini")

    # Wrap with action wrapper (converts bbox to mask)
    env = BboxActionWrapper(env)

    # Use normal environment API with bbox actions
    state, timestep = env.reset(key, env_params=env_params)
    action = {"operation": 15, "r1": 2, "c1": 3, "r2": 7, "c2": 8}
    state, timestep = env.step(state, action, env_params=env_params)

    # The wrapper handles the conversion:
    # bbox dict -> Action with rectangular mask -> core environment
    ```
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from stoa.core_wrappers.wrapper import Wrapper

from ..state import State
from ..types import EnvParams, TimeStep
from .actions import Action, create_action
from .spaces import DictSpace, DiscreteSpace


def _point_to_mask(point_action: dict, grid_shape: tuple[int, int]) -> Action:
    """Convert point action dict to mask action.

    Args:
        point_action: Dict with keys 'operation', 'row', 'col'
        grid_shape: Shape of the grid (height, width)

    Returns:
        Action with single point selected
    """
    operation = point_action["operation"]
    row = point_action["row"]
    col = point_action["col"]
    height, width = grid_shape

    # Create mask with single point
    mask = jnp.zeros((height, width), dtype=jnp.bool_)

    # Clip coordinates to valid range
    valid_row = jnp.clip(row, 0, height - 1)
    valid_col = jnp.clip(col, 0, width - 1)

    # Set the point in the mask
    mask = mask.at[valid_row, valid_col].set(True)

    return create_action(operation, mask)


def _bbox_to_mask(bbox_action: dict, grid_shape: tuple[int, int]) -> Action:
    """Convert bounding box action dict to mask action.

    Args:
        bbox_action: Dict with keys 'operation', 'r1', 'c1', 'r2', 'c2'
        grid_shape: Shape of the grid (height, width)

    Returns:
        Action with rectangular region selected
    """
    operation = bbox_action["operation"]
    r1, c1 = bbox_action["r1"], bbox_action["c1"]
    r2, c2 = bbox_action["r2"], bbox_action["c2"]
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

    return create_action(operation, mask)


# Generic JIT versions using static_argnums for better performance
_jit_point_to_mask = jax.jit(_point_to_mask, static_argnums=1)
_jit_bbox_to_mask = jax.jit(_bbox_to_mask, static_argnums=1)


class PointActionWrapper(Wrapper):
    """Point action wrapper with custom action space."""

    def action_space(self, env_params: EnvParams | None = None) -> DictSpace:
        """Custom action space for point actions: (operation, row, col)."""
        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(env_params)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = env_params.dataset.max_grid_height
        width = env_params.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "row": DiscreteSpace(height),
                "col": DiscreteSpace(width),
            },
            name="point_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert point to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_point_to_mask(action, grid_shape)
        return self._env.step(state, mask_action, env_params)


class BboxActionWrapper(Wrapper):
    """Bbox action wrapper with custom action space."""

    def action_space(self, env_params: EnvParams | None = None) -> DictSpace:
        """Custom action space for bbox actions: (operation, r1, c1, r2, c2)."""
        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(env_params)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = env_params.dataset.max_grid_height
        width = env_params.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "r1": DiscreteSpace(height),
                "c1": DiscreteSpace(width),
                "r2": DiscreteSpace(height),
                "c2": DiscreteSpace(width),
            },
            name="bbox_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert bbox to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_bbox_to_mask(action, grid_shape)
        return self._env.step(state, mask_action, env_params)


__all__ = [
    "BboxActionWrapper",
    "PointActionWrapper",
]
