"""
Action handlers for JaxARC environments.

This module provides mask-based actions for the core JaxARC environment.
The core environment exclusively handles mask-based actions, with other action
formats handled by wrapper classes that transform to mask format.

Key Features:
- Mask-only action format for core environment simplicity
- JIT-compiled mask handler for maximum performance
- Static shapes throughout (output masks match grid dimensions)
- Automatic coordinate clipping and validation
- Working grid mask constraint enforcement
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from ..utils.jax_types import (
    MaskArray,
    OperationId,
    SelectionArray,
)
from ..utils.validation import assert_shape_matches


class MaskAction(eqx.Module):
    """Mask-based action using arbitrary selection.

    This action type allows arbitrary selection patterns using a boolean
    mask that directly specifies which cells are selected. This is the
    canonical action format for the core JaxARC environment.

    Attributes:
        operation: ARC operation ID (0-34)
        action_type: Type identifier (always 2 for MaskAction)
        selection: Boolean mask indicating selected cells
    """

    operation: jnp.int32
    action_type: jnp.int32  # Always 2 for MaskAction
    selection: SelectionArray

    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Return the selection mask directly.

        Args:
            grid_shape: Shape of the grid (height, width) - used for validation

        Returns:
            The selection mask (assumed to already match grid_shape)
        """
        # For simplicity, assume the mask already has the correct shape
        # This is reasonable since masks are typically created with the correct shape
        return self.selection

    def validate(
        self, grid_shape: tuple[int, int], max_operations: int = 35
    ) -> MaskAction:
        """Validate mask action parameters.

        Args:
            grid_shape: Shape of the grid (height, width)
            max_operations: Maximum number of operations

        Returns:
            Validated mask action with clipped operation
        """
        # Clip operation to valid range
        valid_operation = jnp.clip(self.operation, 0, max_operations - 1)

        # Return with validated operation (assume selection is already correct shape)
        return MaskAction(
            operation=valid_operation,
            action_type=jnp.array(2, dtype=jnp.int32),
            selection=self.selection,
        )


def create_mask_action(operation, selection: SelectionArray) -> MaskAction:
    """Create a mask action with the given parameters.

    This function supports both single and batched inputs. When batched inputs
    are provided, all fields will have the same batch dimension.

    Args:
        operation: ARC operation ID (0-34). Can be int or JAX array.
        selection: Boolean mask indicating selected cells

    Returns:
        MaskAction instance with consistent batch dimensions
    """
    # Convert operation to JAX array
    operation_array = jnp.array(operation, dtype=jnp.int32)

    # Create action_type with the same shape as operation
    # For mask actions, action_type is always 2
    if operation_array.shape == ():
        # Scalar case
        action_type_array = jnp.array(2, dtype=jnp.int32)
    else:
        # Batched case - create array of 2s with same shape as operation
        action_type_array = jnp.full_like(operation_array, 2, dtype=jnp.int32)

    return MaskAction(
        operation=operation_array,
        action_type=action_type_array,
        selection=selection,
    )


@jax.jit
def mask_handler(action: MaskAction, working_grid_mask: MaskArray) -> SelectionArray:
    """Convert mask action to selection mask.

    This is the core action handler for the JaxARC environment. All actions
    are ultimately converted to this mask format for processing.

    Args:
        action: MaskAction with operation and selection fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with selection applied
    """
    expected_shape = working_grid_mask.shape
    validated_selection = assert_shape_matches(
        action.selection, expected_shape, "mask_selection"
    )

    # Get the selection mask from the action
    mask = validated_selection.astype(jnp.bool_)

    # Constrain to working grid area
    return mask & working_grid_mask
