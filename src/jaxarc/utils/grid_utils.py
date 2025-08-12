"""
Grid utility functions for JAX-compatible grid operations.

This module provides utility functions for grid manipulation and processing
in the JaxARC environment.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .jax_types import (
    BoundingBox,
    ColorValue,
    GridArray,
    GridHeight,
    GridWidth,
    MaskArray,
    PaddingValue,
    SelectionArray,
    SimilarityScore,
)


def pad_to_max_dims(
    grid: GridArray,
    max_height: GridHeight,
    max_width: GridWidth,
    fill_value: PaddingValue = 0,
) -> GridArray:
    """
    Pad a grid to maximum dimensions.

    **WARNING**: This function is NOT fully JIT-compatible when max_height/max_width
    are dynamic values. For JIT compatibility, ensure max_height and max_width are
    concrete (compile-time known) values.

    Args:
        grid: Input grid to pad
        max_height: Maximum height to pad to (should be concrete for JIT)
        max_width: Maximum width to pad to (should be concrete for JIT)
        fill_value: Value to use for padding (default: 0)

    Returns:
        Grid padded to max dimensions
    """
    current_h, current_w = grid.shape

    # Calculate padding needed using JAX operations
    pad_h = jnp.maximum(0, max_height - current_h)
    pad_w = jnp.maximum(0, max_width - current_w)

    # Pad with fill_value to max dimensions
    # Note: This will fail with JIT if pad_h/pad_w are not concrete values
    return jnp.pad(
        grid, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=fill_value
    )


def get_grid_bounds(grid: GridArray, background_value: ColorValue = 0) -> BoundingBox:
    """
    Get the bounding box of non-background content in a grid.

    JAX-compatible function that works with JIT compilation.

    Args:
        grid: Input grid
        background_value: Value considered as background (default: 0)

    Returns:
        Tuple of (min_row, max_row, min_col, max_col) bounds
    """
    # Find non-background positions
    non_bg_mask = grid != background_value

    # Create index arrays
    row_indices = jnp.arange(grid.shape[0])
    col_indices = jnp.arange(grid.shape[1])

    # Find rows and columns with non-background content
    has_content_rows = jnp.any(non_bg_mask, axis=1)
    has_content_cols = jnp.any(non_bg_mask, axis=0)

    # Use JAX-compatible operations to find bounds
    min_row = jnp.where(
        jnp.any(has_content_rows),
        jnp.min(jnp.where(has_content_rows, row_indices, grid.shape[0])),
        0,
    )
    max_row = jnp.where(
        jnp.any(has_content_rows),
        jnp.max(jnp.where(has_content_rows, row_indices, -1)),
        0,
    )
    min_col = jnp.where(
        jnp.any(has_content_cols),
        jnp.min(jnp.where(has_content_cols, col_indices, grid.shape[1])),
        0,
    )
    max_col = jnp.where(
        jnp.any(has_content_cols),
        jnp.max(jnp.where(has_content_cols, col_indices, -1)),
        0,
    )

    return min_row, max_row, min_col, max_col


def crop_grid_to_content(
    grid: GridArray, background_value: ColorValue = 0
) -> GridArray:
    """
    Crop a grid to its content bounds (remove background padding).

    JAX-compatible function that works with JIT compilation.

    **Important**: This function returns a dynamically-sized array which may not
    be compatible with all JAX transformations that require static shapes.

    Args:
        grid: Input grid to crop
        background_value: Value considered as background (default: 0)

    Returns:
        Cropped grid containing only the content area
    """
    min_row, max_row, min_col, max_col = get_grid_bounds(grid, background_value)

    # Calculate crop dimensions using JAX-compatible operations
    crop_height = jnp.maximum(1, max_row - min_row + 1)
    crop_width = jnp.maximum(1, max_col - min_col + 1)

    # Use dynamic slice to extract the content area
    cropped_grid = jax.lax.dynamic_slice(
        grid, (min_row, min_col), (crop_height, crop_width)
    )

    return cropped_grid


def get_actual_grid_shape_from_mask(mask: MaskArray) -> tuple[int, int]:
    """
    Get the actual shape of a grid based on its validity mask.

    JAX-compatible function that works with JIT compilation.

    Since JAX requires static shapes, grids are often padded to max dimensions,
    but the actual meaningful grid size is determined by the mask.

    Args:
        mask: Boolean mask indicating valid cells in the grid

    Returns:
        Tuple of (height, width) representing the actual grid dimensions

    Examples:
        ```python
        # For a 5x5 actual grid with a 30x30 mask
        mask = jnp.zeros((30, 30), dtype=bool)
        mask = mask.at[:5, :5].set(True)
        actual_height, actual_width = get_actual_grid_shape_from_mask(mask)
        # Returns (5, 5) instead of (30, 30)
        ```
    """
    # Find which rows and columns have any valid cells
    valid_rows = jnp.any(mask, axis=1)
    valid_cols = jnp.any(mask, axis=0)

    # Create index arrays for rows and columns
    row_indices = jnp.arange(mask.shape[0])
    col_indices = jnp.arange(mask.shape[1])

    # Find the maximum valid indices using JAX-compatible operations
    # Use jnp.where to handle the case where no valid cells exist
    max_row_idx = jnp.where(
        jnp.any(valid_rows), jnp.max(jnp.where(valid_rows, row_indices, -1)), -1
    )
    max_col_idx = jnp.where(
        jnp.any(valid_cols), jnp.max(jnp.where(valid_cols, col_indices, -1)), -1
    )

    # The actual dimensions are the maximum valid indices + 1
    # Use jnp.maximum to ensure non-negative values
    actual_height = jnp.maximum(0, max_row_idx + 1)
    actual_width = jnp.maximum(0, max_col_idx + 1)

    return (actual_height, actual_width)


def crop_grid_to_mask(grid: GridArray, mask: chex.Array) -> GridArray:
    """
    Non-JIT version for visualization and debugging.

    **WARNING**: This function is NOT JIT-compatible because it returns
    dynamically-sized arrays. Use only for visualization, debugging, or
    non-JIT contexts.

    Args:
        grid: Input grid to crop (may be padded)
        mask: Boolean mask indicating valid cells

    Returns:
        GridArray containing only the actual grid content (variable size)

    Examples:
        ```python
        # For visualization/debugging (non-JIT):
        padded_grid = jnp.zeros((30, 30))
        mask = jnp.zeros((30, 30), dtype=bool).at[:5, :5].set(True)
        actual_grid = crop_grid_to_mask_non_jit(padded_grid, mask)
        # Returns actual 5x5 grid
        ```
    """
    actual_height, actual_width = get_actual_grid_shape_from_mask(mask)

    # Convert to Python ints for slicing (breaks JIT compatibility)
    h = int(actual_height)
    w = int(actual_width)

    # Handle empty case
    if h == 0 or w == 0:
        return jnp.array([[0]], dtype=grid.dtype)

    # Use regular Python slicing (not JIT compatible)
    return grid[:h, :w]


@jax.jit
def compute_grid_similarity(
    working_grid: GridArray,
    working_mask: SelectionArray,
    target_grid: GridArray,
    target_mask: SelectionArray,
) -> SimilarityScore:
    """
    Compute size-aware similarity between working and target grids.

    This function compares grids on a fixed canvas based on the target grid's
    actual dimensions. Size mismatches are properly penalized:
    - If working grid is smaller: missing areas count as mismatches
    - If working grid is larger: only the target-sized area is compared

    Args:
        working_grid: The current working grid
        working_mask: Mask indicating active area of working grid
        target_grid: The target grid to compare against
        target_mask: Mask indicating active area of target grid

    Returns:
        Similarity score from 0.0 to 1.0
    """

    # Check if target has any content
    target_has_content = jnp.sum(target_mask) > 0

    def compute_similarity_with_content():
        # Use utility function to get target dimensions
        target_height, target_width = get_actual_grid_shape_from_mask(target_mask)

        # Create comparison canvas based on target dimensions
        rows = jnp.arange(target_grid.shape[0])[:, None]
        cols = jnp.arange(target_grid.shape[1])[None, :]
        canvas_mask = (rows < target_height) & (cols < target_width)

        # Compare grids only where target_mask indicates content should exist
        # Working grid content is considered within canvas and working mask
        working_content = jnp.where(canvas_mask & working_mask, working_grid, -1)
        target_content = jnp.where(target_mask, target_grid, -1)

        # Count matches only in target mask positions
        matches = jnp.sum((working_content == target_content) & target_mask)
        total_target_positions = jnp.sum(target_mask)

        return matches.astype(jnp.float32) / total_target_positions.astype(jnp.float32)

    def empty_target_similarity():
        # If target is empty, similarity is 1.0 if working is also empty, 0.0 otherwise
        working_has_content = jnp.sum(working_mask) > 0
        return jnp.where(working_has_content, 0.0, 1.0)

    return jax.lax.cond(
        target_has_content, compute_similarity_with_content, empty_target_similarity
    )
