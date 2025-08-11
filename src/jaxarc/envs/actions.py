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

from jaxarc.envs.structured_actions import BboxAction, MaskAction, PointAction

from ..utils.error_handling import assert_in_range, assert_shape_matches
from ..utils.jax_types import (
    MaskArray,
    SelectionArray,
)


@jax.jit
def point_handler(action: PointAction, working_grid_mask: MaskArray) -> SelectionArray:
    """Convert point action to selection mask.

    Args:
        action: PointAction with operation, row, and col fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with single point selected
    """
    # Validate action coordinates using JAX-compatible error handling

    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Validate coordinates are within bounds (with error handling)
    validated_row = assert_in_range(action.row, 0, grid_height - 1, "point_row")
    validated_col = assert_in_range(action.col, 0, grid_width - 1, "point_col")

    # Extract and clip coordinates to valid range (fallback for graceful degradation)
    row = jnp.clip(validated_row, 0, grid_height - 1)
    col = jnp.clip(validated_col, 0, grid_width - 1)

    # Create mask with single point
    mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
    mask = mask.at[row, col].set(True)

    # Constrain to working grid area
    return mask & working_grid_mask


@jax.jit
def bbox_handler(action: BboxAction, working_grid_mask: MaskArray) -> SelectionArray:
    """Convert bounding box action to selection mask.

    Args:
        action: BboxAction with operation, r1, c1, r2, c2 fields
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with rectangular region selected
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Validate all coordinates are within bounds
    validated_r1 = assert_in_range(action.r1, 0, grid_height - 1, "bbox_r1")
    validated_c1 = assert_in_range(action.c1, 0, grid_width - 1, "bbox_c1")
    validated_r2 = assert_in_range(action.r2, 0, grid_height - 1, "bbox_r2")
    validated_c2 = assert_in_range(action.c2, 0, grid_width - 1, "bbox_c2")

    # Extract and clip coordinates to valid range (fallback for graceful degradation)
    r1 = jnp.clip(validated_r1, 0, grid_height - 1)
    c1 = jnp.clip(validated_c1, 0, grid_width - 1)
    r2 = jnp.clip(validated_r2, 0, grid_height - 1)
    c2 = jnp.clip(validated_c2, 0, grid_width - 1)

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
def mask_handler(action: MaskAction, working_grid_mask: MaskArray) -> SelectionArray:
    """Convert mask action to selection mask.

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
