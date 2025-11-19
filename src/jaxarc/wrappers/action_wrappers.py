"""
Wrappers for JaxARC environments (simplified with clean delegation).

This module implements clean wrappers following Stoa delegation patterns:
- Core Environment: Only knows about Action objects (mask-based selections)
- Wrappers: Convert user-friendly formats to masks or reshape observations

- PointActionWrapper: Converts {"operation": op, "row": r, "col": c} dicts to mask actions
- BboxActionWrapper: Converts {"operation": op, "r1": r1, "c1": c1, "r2": r2, "c2": c2} dicts to mask actions
- FlattenActionWrapper: Flattens a DictSpace of Discrete sub-spaces into a single Discrete action space

Usage:
    ```python
    from jaxarc.registration import make
    from jaxarc.wrappers import BboxActionWrapper

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

import operator
from functools import reduce
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from stoa import MultiDiscreteSpace, Space
from stoa.core_wrappers.wrapper import Wrapper

from jaxarc.envs.actions import Action, create_action
from jaxarc.envs.environment import Environment
from jaxarc.envs.spaces import DictSpace, DiscreteSpace
from jaxarc.state import State
from jaxarc.types import EnvParams, TimeStep


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
        # Use provided params or fall back to the environment's default params.
        p = self._env.params if env_params is None else env_params

        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(p)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = p.dataset.max_grid_height
        width = p.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "row": DiscreteSpace(height, dtype=jnp.int32),
                "col": DiscreteSpace(width, dtype=jnp.int32),
            },
            name="point_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert point to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_point_to_mask(action, grid_shape)

        # Delegate to underlying env using mask-based Action
        next_state, timestep = self._env.step(state, mask_action, env_params)

        # Core Environment now guarantees canonical_action/operation_id in extras.
        return next_state, timestep


class BboxActionWrapper(Wrapper):
    """Bbox action wrapper with custom action space."""

    def action_space(self, env_params: EnvParams | None = None) -> DictSpace:
        """Custom action space for bbox actions: (operation, r1, c1, r2, c2)."""
        # Use provided params or fall back to the environment's default params.
        p = self._env.params if env_params is None else env_params

        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(p)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = p.dataset.max_grid_height
        width = p.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "r1": DiscreteSpace(height, dtype=jnp.int32),
                "c1": DiscreteSpace(width, dtype=jnp.int32),
                "r2": DiscreteSpace(height, dtype=jnp.int32),
                "c2": DiscreteSpace(width, dtype=jnp.int32),
            },
            name="bbox_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert bbox to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_bbox_to_mask(action, grid_shape)

        # Delegate to underlying env using mask-based Action
        next_state, timestep = self._env.step(state, mask_action, env_params)

        # Core Environment now guarantees canonical_action/operation_id in extras.
        return next_state, timestep


class FlattenActionWrapper(Wrapper[State]):
    """
    A general-purpose wrapper to flatten any composite discrete action space.

    This wrapper can handle any combination of DictSpace, MultiDiscreteSpace,
    and DiscreteSpace, converting them into a single, unified DiscreteSpace.
    """

    def __init__(self, env: Environment):
        super().__init__(env)

        base_action_space = self._env.action_space()

        # 1. Recursively find all the fundamental discrete components and their sizes.
        #    Also, store the dictionary keys if the top-level space is a dict.
        self._components, self._component_sizes = self._get_components_and_sizes(
            base_action_space
        )
        self._action_keys = None
        if isinstance(base_action_space, DictSpace):
            # Preserve insertion order from the DictSpace to match wrapper-defined order
            self._action_keys = list(base_action_space.spaces.keys())

        # 2. Calculate the total number of discrete actions.
        if not self._component_sizes:
            self._total_actions = 0
        else:
            self._total_actions = reduce(operator.mul, self._component_sizes)

    def _get_components_and_sizes(self, space: Space) -> tuple[list[Space], list[int]]:
        """Recursively decomposes a space into its base discrete components."""
        if isinstance(space, DiscreteSpace):
            return [space], [space.num_values]

        if isinstance(space, MultiDiscreteSpace):
            return [space], [int(np.prod(space.num_values))]

        if isinstance(space, DictSpace):
            components, sizes = [], []
            # Preserve insertion order for deterministic mapping consistent with the env
            for key in space.spaces:
                sub_components, sub_sizes = self._get_components_and_sizes(
                    space.spaces[key]
                )
                components.extend(sub_components)
                sizes.extend(sub_sizes)
            return components, sizes

        msg = f"FlattenActionWrapper does not support space type: {type(space)}"
        raise TypeError(msg)

    def _unflatten_action(self, flat_action: Action) -> Action:
        """Converts a single integer action back into the original structured action."""
        unflattened_parts = []
        remainder = flat_action

        # Calculate the choice index for each component from the flat action
        for i, size in enumerate(self._component_sizes):
            divisor = reduce(operator.mul, self._component_sizes[i + 1 :], 1)
            choice_index = remainder // divisor
            remainder %= divisor
            unflattened_parts.append(choice_index)

        # Reconstruct each component action from its choice index
        reconstructed_actions = []
        for i, component in enumerate(self._components):
            part = unflattened_parts[i]
            if isinstance(component, DiscreteSpace):
                reconstructed_actions.append(part.astype(component.dtype))

            # *** THE FIX IS IN THIS BLOCK ***
            elif isinstance(component, MultiDiscreteSpace):
                num_values = component.num_values.flatten()

                # Define the JIT-safe scan function
                def scan_body(carry, n):
                    new_carry, choice = jnp.divmod(carry, n)
                    return new_carry, choice

                # Use lax.scan to perform the sequential division and modulus
                # We scan over the reversed num_values array
                _, choices = lax.scan(scan_body, init=part, xs=jnp.flip(num_values))

                # The choices are generated in reverse order, so we flip them back
                multi_discrete_action = (
                    jnp.flip(choices).astype(component.dtype).reshape(component.shape)
                )
                reconstructed_actions.append(multi_discrete_action)

        if self._action_keys:
            return {
                key: action
                for key, action in zip(self._action_keys, reconstructed_actions)
            }
        return reconstructed_actions[0]

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> tuple[State, TimeStep]:
        """Un-flattens the action and steps the underlying environment."""
        structured_action = self._unflatten_action(action)
        return self._env.step(state, structured_action, env_params)

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Returns the single, flattened DiscreteSpace."""
        return DiscreteSpace(num_values=self._total_actions, dtype=jnp.int32)


__all__ = [
    "BboxActionWrapper",
    "FlattenActionWrapper",
    "PointActionWrapper",
]
