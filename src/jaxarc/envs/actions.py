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
    action_data: PointActionData, working_grid_mask: MaskArray
) -> SelectionArray:
    """Convert point coordinates to selection mask.

    Args:
        action_data: Array with at least 2 elements [row, col]
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with single point selected
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Extract and clip coordinates to valid range
    row = jnp.clip(action_data[0].astype(jnp.int32), 0, grid_height - 1)
    col = jnp.clip(action_data[1].astype(jnp.int32), 0, grid_width - 1)

    # Create mask with single point
    mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
    mask = mask.at[row, col].set(True)

    # Constrain to working grid area
    return mask & working_grid_mask


@jax.jit
def bbox_handler(
    action_data: BboxActionData, working_grid_mask: MaskArray
) -> SelectionArray:
    """Convert bounding box coordinates to selection mask.

    Args:
        action_data: Array with at least 4 elements [r1, c1, r2, c2]
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with rectangular region selected
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape

    # Extract and clip coordinates to valid range
    r1 = jnp.clip(action_data[0].astype(jnp.int32), 0, grid_height - 1)
    c1 = jnp.clip(action_data[1].astype(jnp.int32), 0, grid_width - 1)
    r2 = jnp.clip(action_data[2].astype(jnp.int32), 0, grid_height - 1)
    c2 = jnp.clip(action_data[3].astype(jnp.int32), 0, grid_width - 1)

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
    action_data: MaskActionData, working_grid_mask: MaskArray
) -> SelectionArray:
    """Pass through mask data with validation.

    Args:
        action_data: Flattened array with grid_size elements (height*width mask)
        working_grid_mask: Boolean mask defining valid grid area

    Returns:
        Boolean mask with same shape as working_grid_mask with selection applied
    """
    # Get grid dimensions from working_grid_mask
    grid_height, grid_width = working_grid_mask.shape
    expected_size = grid_height * grid_width

    # Take first expected_size elements and reshape to grid
    mask = (
        action_data[:expected_size].reshape((grid_height, grid_width)).astype(jnp.bool_)
    )

    # Constrain to working grid area
    return mask & working_grid_mask


def get_action_handler(selection_format: str):
    """Factory function to get appropriate action handler.

    Args:
        selection_format: Format string ("point", "bbox", "mask")

    Returns:
        JAX-compiled handler function

    Raises:
        ValueError: If selection_format is not recognized
    """
    if selection_format == "point":
        logger.debug("Using point operation action handler")
        return point_handler
    if selection_format == "bbox":
        logger.debug("Using bbox operation action handler")
        return bbox_handler
    if selection_format == "mask":
        logger.debug(
            f"Using mask operation action handler for format: {selection_format}"
        )
        return mask_handler
    error_msg = f"Unknown selection format: {selection_format}"
    raise ValueError(error_msg)


# Utility functions for testing and debugging


def validate_action_data(
    action_data: PointActionData | BboxActionData | MaskActionData,
    selection_format: str,
    grid_shape: tuple = None,
) -> None:
    """Validate action data format and shape.

    Args:
        action_data: Action data array
        selection_format: Expected format
        grid_shape: Expected grid shape for mask validation (height, width)

    Raises:
        ValueError: If data doesn't match expected format
    """
    if selection_format == "point":
        if action_data.size < 2:
            error_msg = (
                f"Point action requires at least 2 elements, got {action_data.size}"
            )
            raise ValueError(error_msg)
    elif selection_format == "bbox":
        if action_data.size < 4:
            error_msg = (
                f"Bbox action requires at least 4 elements, got {action_data.size}"
            )
            raise ValueError(error_msg)
    elif selection_format == "mask":
        if grid_shape is not None:
            expected_size = grid_shape[0] * grid_shape[1]
            if action_data.size < expected_size:
                error_msg = f"Mask action requires at least {expected_size} elements for grid shape {grid_shape}, got {action_data.size}"
                raise ValueError(error_msg)
        # If no grid shape provided, just check for reasonable minimum
        elif action_data.size < 9:  # At least 3x3 grid
            error_msg = (
                f"Mask action requires at least 9 elements, got {action_data.size}"
            )
            raise ValueError(error_msg)
    else:
        error_msg = f"Unknown selection format: {selection_format}"
        raise ValueError(error_msg)


def create_test_action_data(
    selection_format: str, grid_shape: tuple = (30, 30), **kwargs
) -> PointActionData | BboxActionData | MaskActionData:
    """Create test action data for given format.

    Args:
        selection_format: Format to create data for
        grid_shape: Grid shape (height, width) for mask formats
        **kwargs: Format-specific parameters

    Returns:
        Action data array suitable for the format
    """
    if selection_format == "point":
        row = kwargs.get("row", 5)
        col = kwargs.get("col", 10)
        return jnp.array([row, col])
    if selection_format == "bbox":
        r1 = kwargs.get("r1", 2)
        c1 = kwargs.get("c1", 3)
        r2 = kwargs.get("r2", 4)
        c2 = kwargs.get("c2", 5)
        return jnp.array([r1, c1, r2, c2])
    if selection_format == "mask":
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
        return mask.flatten()
    raise ValueError(f"Unknown selection format: {selection_format}")
