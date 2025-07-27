"""
Action handlers for JaxARC environments.

This module provides specialized, JAX-compiled handlers for different action formats.
Each handler converts action data to a standardized boolean mask format that matches
the grid dimensions, ensuring JAX compatibility and optimal performance.

Key Features:
- JIT-compiled handlers for maximum performance
- Static shapes throughout (output masks match grid dimensions)
- Automatic coordinate clipping and validation
- Working grid mask constraint enforcement
- Configuration-driven handler selection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from ..utils.jax_types import (
    BboxActionData,
    MaskActionData,
    MaskArray,
    PointActionData,
    SelectionArray,
)


@jax.jit
def point_handler(
    action: 'PointAction', working_grid_mask: MaskArray
) -> SelectionArray:
    """Convert point action to selection mask.

    Args:
        action: PointAction with operation, row, and col fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with single point selected
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Extract and clip coordinates to valid range
    row = jnp.clip(action.row, 0, grid_height - 1)
    col = jnp.clip(action.col, 0, grid_width - 1)

    # Create mask with single point
    mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
    mask = mask.at[row, col].set(True)

    # Constrain to working grid area
    return mask & working_grid_mask


@jax.jit
def bbox_handler(
    action: 'BboxAction', working_grid_mask: MaskArray
) -> SelectionArray:
    """Convert bounding box action to selection mask.

    Args:
        action: BboxAction with operation, r1, c1, r2, c2 fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with rectangular region selected
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Extract and clip coordinates to valid range
    r1 = jnp.clip(action.r1, 0, grid_height - 1)
    c1 = jnp.clip(action.c1, 0, grid_width - 1)
    r2 = jnp.clip(action.r2, 0, grid_height - 1)
    c2 = jnp.clip(action.c2, 0, grid_width - 1)

    # Ensure proper ordering (min, max)
    min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
    min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)

    # Create coordinate meshes
    rows = jnp.arange(grid_height)
    cols = jnp.arange(grid_width)
    row_mesh, col_mesh = jnp.meshgrid(rows, cols, indexing="ij")

    # Create bbox mask (inclusive bounds)
    mask = (
        (row_mesh >= min_r)
        & (row_mesh <= max_r)
        & (col_mesh >= min_c)
        & (col_mesh <= max_c)
    )

    # Constrain to working grid area
    return mask & working_grid_mask


@jax.jit
def mask_handler(
    action: 'MaskAction', working_grid_mask: MaskArray
) -> SelectionArray:
    """Convert mask action to selection mask.

    Args:
        action: MaskAction with operation and selection fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with selection applied
    """
    # Get the selection mask from the action
    mask = action.selection.astype(jnp.bool_)

    # Constrain to working grid area
    return mask & working_grid_mask


def get_action_handler(action_type: str):
    """Factory function to get appropriate action handler for structured actions.

    Args:
        action_type: Action type string ("point", "bbox", "mask")

    Returns:
        JAX-compiled handler function that accepts structured actions

    Raises:
        ValueError: If action_type is not recognized
    """
    if action_type == "point":
        logger.debug("Using point action handler for structured actions")
        return point_handler
    if action_type == "bbox":
        logger.debug("Using bbox action handler for structured actions")
        return bbox_handler
    if action_type == "mask":
        logger.debug("Using mask action handler for structured actions")
        return mask_handler
    error_msg = f"Unknown action type: {action_type}"
    raise ValueError(error_msg)


# Utility functions for testing and debugging


def validate_structured_action(
    action: 'StructuredAction',
    grid_shape: tuple = None,
) -> None:
    """Validate structured action format and parameters.

    Args:
        action: Structured action (PointAction, BboxAction, or MaskAction)
        grid_shape: Expected grid shape for validation (height, width)

    Raises:
        ValueError: If action parameters are invalid
    """
    from .structured_actions import PointAction, BboxAction, MaskAction
    
    if isinstance(action, PointAction):
        if grid_shape is not None:
            grid_height, grid_width = grid_shape
            if action.row < 0 or action.row >= grid_height:
                raise ValueError(f"Point row {action.row} out of bounds for grid height {grid_height}")
            if action.col < 0 or action.col >= grid_width:
                raise ValueError(f"Point col {action.col} out of bounds for grid width {grid_width}")
    elif isinstance(action, BboxAction):
        if grid_shape is not None:
            grid_height, grid_width = grid_shape
            if action.r1 < 0 or action.r1 >= grid_height:
                raise ValueError(f"Bbox r1 {action.r1} out of bounds for grid height {grid_height}")
            if action.c1 < 0 or action.c1 >= grid_width:
                raise ValueError(f"Bbox c1 {action.c1} out of bounds for grid width {grid_width}")
            if action.r2 < 0 or action.r2 >= grid_height:
                raise ValueError(f"Bbox r2 {action.r2} out of bounds for grid height {grid_height}")
            if action.c2 < 0 or action.c2 >= grid_width:
                raise ValueError(f"Bbox c2 {action.c2} out of bounds for grid width {grid_width}")
    elif isinstance(action, MaskAction):
        if grid_shape is not None:
            expected_shape = grid_shape
            if action.selection.shape != expected_shape:
                raise ValueError(f"Mask selection shape {action.selection.shape} doesn't match expected {expected_shape}")
    else:
        raise ValueError(f"Unknown action type: {type(action)}")
    
    # Validate operation range
    if action.operation < 0 or action.operation >= 42:
        raise ValueError(f"Operation {action.operation} out of valid range [0, 41]")


def create_test_structured_action(
    action_type: str, grid_shape: tuple = (30, 30), **kwargs
) -> 'StructuredAction':
    """Create test structured action for given type.

    Args:
        action_type: Action type to create ("point", "bbox", "mask")
        grid_shape: Grid shape (height, width) for mask actions
        **kwargs: Action-specific parameters

    Returns:
        Structured action suitable for the type
    """
    from .structured_actions import PointAction, BboxAction, MaskAction, create_point_action, create_bbox_action, create_mask_action
    
    operation = kwargs.get("operation", 0)
    
    if action_type == "point":
        row = kwargs.get("row", 5)
        col = kwargs.get("col", 10)
        return create_point_action(operation, row, col)
    elif action_type == "bbox":
        r1 = kwargs.get("r1", 2)
        c1 = kwargs.get("c1", 3)
        r2 = kwargs.get("r2", 4)
        c2 = kwargs.get("c2", 5)
        return create_bbox_action(operation, r1, c1, r2, c2)
    elif action_type == "mask":
        # Create a simple test mask with some selected cells
        grid_height, grid_width = grid_shape
        mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
        start_row = kwargs.get("start_row", min(5, grid_height - 1))
        start_col = kwargs.get("start_col", min(5, grid_width - 1))
        size = kwargs.get(
            "size", min(3, grid_height - start_row, grid_width - start_col)
        )
        mask = mask.at[start_row : start_row + size, start_col : start_col + size].set(
            True
        )
        return create_mask_action(operation, mask)
    else:
        raise ValueError(f"Unknown action type: {action_type}")
