"""
Grid utility functions for JAX-compatible grid operations.

This module provides utility functions for grid manipulation and processing
in the JaxARC environment.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

from .jax_types import (
    BoundingBox,
    ColorValue,
    GridArray,
    GridHeight,
    GridWidth,
    PaddingValue,
)


def pad_to_max_dims(
    grid: GridArray, max_height: GridHeight, max_width: GridWidth, fill_value: PaddingValue = 0
) -> GridArray:
    """
    Pad a grid to maximum dimensions.

    Args:
        grid: Input grid to pad
        max_height: Maximum height to pad to
        max_width: Maximum width to pad to
        fill_value: Value to use for padding (default: 0)

    Returns:
        Grid padded to max dimensions
    """
    current_h, current_w = grid.shape

    # If already max size, return as is
    if current_h == max_height and current_w == max_width:
        return grid

    # Calculate padding needed
    pad_h = max_height - current_h
    pad_w = max_width - current_w

    # Ensure non-negative padding
    pad_h = max(0, pad_h)
    pad_w = max(0, pad_w)

    # Pad with fill_value to max dimensions
    return jnp.pad(
        grid, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=fill_value
    )


def get_grid_bounds(
    grid: GridArray, background_value: ColorValue = 0
) -> BoundingBox:
    """
    Get the bounding box of non-background content in a grid.

    Args:
        grid: Input grid
        background_value: Value considered as background (default: 0)

    Returns:
        Tuple of (min_row, max_row, min_col, max_col) bounds
    """
    # Find non-background positions
    non_bg_mask = grid != background_value

    # Get row and column indices of non-background cells
    rows, cols = jnp.where(non_bg_mask)

    # If no non-background cells, return empty bounds
    if len(rows) == 0:
        return 0, 0, 0, 0

    # Calculate bounds
    min_row = int(jnp.min(rows))
    max_row = int(jnp.max(rows))
    min_col = int(jnp.min(cols))
    max_col = int(jnp.max(cols))

    return min_row, max_row, min_col, max_col


def crop_grid_to_content(grid: GridArray, background_value: ColorValue = 0) -> GridArray:
    """
    Crop a grid to its content bounds (remove background padding).

    Args:
        grid: Input grid to crop
        background_value: Value considered as background (default: 0)

    Returns:
        Cropped grid containing only the content area
    """
    min_row, max_row, min_col, max_col = get_grid_bounds(grid, background_value)

    # If no content, return a 1x1 grid with background value
    if min_row == max_row == min_col == max_col == 0:
        return jnp.array([[background_value]])

    # Crop to content bounds (inclusive)
    return grid[min_row : max_row + 1, min_col : max_col + 1]
