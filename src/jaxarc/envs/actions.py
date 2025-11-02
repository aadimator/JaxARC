"""
Action system for JaxARC environments.

This module provides the complete action system following KISS principle:
- Action class for representing actions
- Action creation and processing utilities
- Operation validation and filtering utilities

Combined from simplified actions.py and action_filtering.py for better organization.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxarc.configs.action_config import ActionConfig
from jaxarc.state import State
from jaxarc.types import (
    NUM_OPERATIONS,
    MaskArray,
    OperationMask,
    SelectionArray,
)


class Action(eqx.Module):
    """Simple action representation.

    Attributes:
        operation: ARC operation ID (0-34)
        selection: Boolean mask indicating selected cells
    """

    operation: jnp.int32
    selection: SelectionArray

    def validate(self, grid_shape: tuple[int, int], max_operations: int = 35) -> Action:
        """Validate action parameters.

        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations

        Returns:
            Validated action with clipped operation
        """
        # Clip operation to valid range
        valid_operation = jnp.clip(self.operation, 0, max_operations - 1)

        # Return with validated operation (assume selection is already correct shape)
        return Action(
            operation=valid_operation,
            selection=self.selection,
        )


def create_action(operation, selection: SelectionArray) -> Action:
    """Create an action.

    Args:
        operation: ARC operation ID (0-34)
        selection: Boolean mask indicating selected cells

    Returns:
        Action instance
    """
    return Action(
        operation=jnp.array(operation, dtype=jnp.int32),
        selection=selection,
    )


@jax.jit
def action_handler(action: Action, working_grid_mask: MaskArray) -> SelectionArray:
    """Process action to create selection mask.

    Args:
        action: Action with operation and selection
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean selection mask constrained to working grid area
    """
    # Ensure selection is boolean and constrain to working grid
    selection = action.selection.astype(jnp.bool_)
    return selection & working_grid_mask


def get_allowed_operations(state: State, config: ActionConfig) -> OperationMask:
    """Get mask of allowed operations based on config and state.

    Args:
        state: Current environment state
        config: Action configuration

    Returns:
        Boolean mask indicating which operations are allowed
    """
    allowed_ops = getattr(config, "allowed_operations", None)
    if isinstance(allowed_ops, tuple) and len(allowed_ops) > 0:
        idx = jnp.asarray(allowed_ops, dtype=jnp.int32)
        idx = jnp.clip(idx, 0, NUM_OPERATIONS - 1)
        base = jnp.zeros((NUM_OPERATIONS,), dtype=jnp.bool_).at[idx].set(True)
    else:
        base = jnp.ones((NUM_OPERATIONS,), dtype=jnp.bool_)

    if hasattr(state, "allowed_operations_mask"):
        base = jnp.logical_and(base, state.allowed_operations_mask)

    return base


def validate_operation(
    operation_id: jnp.ndarray, state: State, config: ActionConfig
) -> jnp.ndarray:
    """Validate if operation ID is in range and allowed by current mask.

    Args:
        operation_id: Operation ID to validate
        state: Current environment state
        config: Action configuration

    Returns:
        Boolean indicating if operation is valid
    """
    mask = get_allowed_operations(state, config)
    in_range = (operation_id >= 0) & (operation_id < NUM_OPERATIONS)
    safe = jnp.clip(operation_id, 0, NUM_OPERATIONS - 1)
    return in_range & mask[safe]


def _find_nearest_valid_operation(
    op_id: jnp.ndarray, mask: OperationMask
) -> jnp.ndarray:
    """Find nearest valid operation ID based on allowed operations mask.

    Args:
        op_id: Target operation ID
        mask: Boolean mask of allowed operations

    Returns:
        Nearest valid operation ID, or 0 if none available
    """
    ids = jnp.arange(NUM_OPERATIONS)
    dists = jnp.abs(ids - op_id)
    dists = jnp.where(mask, dists, jnp.inf)
    idx = jnp.argmin(dists)
    return jnp.where(jnp.any(mask), idx, jnp.array(0, dtype=jnp.int32))


def filter_invalid_operation(
    operation_id: jnp.ndarray, state: State, config: ActionConfig
) -> jnp.ndarray:
    """Filter invalid operation according to configured policy.

    Args:
        operation_id: Operation ID to filter
        state: Current environment state
        config: Action configuration with invalid_operation_policy

    Returns:
        Filtered operation ID based on policy
    """
    arr = operation_id.astype(jnp.int32)
    mask = get_allowed_operations(state, config)
    valid = validate_operation(arr, state, config)
    policy = getattr(config, "invalid_operation_policy", "clip")
    clipped = jnp.clip(arr, 0, NUM_OPERATIONS - 1)

    if policy in ("clip", "penalize"):
        repl = _find_nearest_valid_operation(clipped, mask)
        out = jnp.where(valid, arr, repl)
    elif policy == "reject":
        out = jnp.where(valid, arr, jnp.array(-1, dtype=jnp.int32))
    elif policy == "passthrough":
        out = arr
    else:
        # Default to clip behavior for unknown policies
        repl = _find_nearest_valid_operation(clipped, mask)
        out = jnp.where(valid, arr, repl)

    return out
