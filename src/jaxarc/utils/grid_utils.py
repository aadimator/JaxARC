"""
Consolidated grid utility functions for JAX-compatible grid operations.

This module provides all essential grid manipulation and processing functions
for the JaxARC environment, combining parsing utilities with grid operations
in a simplified, KISS-compliant design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from ..types import (
    GridArray,
    MaskArray,
    SelectionArray,
    SimilarityScore,
)

# =============================================================================
# Core Padding Functions (Consolidated)
# =============================================================================


def pad_grid_to_size(
    grid: jnp.ndarray, target_height: int, target_width: int, fill_value: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad a grid to target dimensions and create a validity mask.

    Args:
        grid: Input grid as JAX array of shape (height, width)
        target_height: Target height after padding
        target_width: Target width after padding
        fill_value: Value to use for padding (default: 0)

    Returns:
        Tuple of (padded_grid, mask) where:
        - padded_grid: Grid padded to (target_height, target_width)
        - mask: Boolean mask indicating valid (non-padded) regions

    Raises:
        ValueError: If grid dimensions exceed target dimensions
    """
    current_height, current_width = grid.shape

    if current_height > target_height or current_width > target_width:
        msg = (
            f"Grid dimensions ({current_height}x{current_width}) exceed "
            f"target dimensions ({target_height}x{target_width})"
        )
        raise ValueError(msg)

    # Calculate padding amounts
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # Pad the grid
    padded_grid = jnp.pad(
        grid,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=fill_value,
    )

    # Create validity mask (True for original data, False for padding)
    mask = jnp.pad(
        jnp.ones_like(grid, dtype=jnp.bool_),
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=False,
    )

    return padded_grid, mask


def pad_array_sequence(
    arrays: list[jnp.ndarray],
    target_length: int,
    target_height: int,
    target_width: int,
    fill_value: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad a sequence of grids to uniform dimensions.

    Args:
        arrays: List of JAX arrays, each of shape (height, width)
        target_length: Target number of arrays (will pad with empty grids)
        target_height: Target height for each grid
        target_width: Target width for each grid
        fill_value: Value to use for padding

    Returns:
        Tuple of (padded_arrays, masks) where:
        - padded_arrays: Array of shape (target_length, target_height, target_width)
        - masks: Boolean mask array of same shape indicating valid regions

    Raises:
        ValueError: If any array dimensions exceed target dimensions
        ValueError: If number of arrays exceeds target_length
    """
    if len(arrays) > target_length:
        msg = (
            f"Number of arrays ({len(arrays)}) exceeds target length ({target_length})"
        )
        raise ValueError(msg)

    padded_grids = []
    masks = []

    # Process existing arrays
    for i, array in enumerate(arrays):
        try:
            padded_grid, mask = pad_grid_to_size(
                array, target_height, target_width, fill_value
            )
            padded_grids.append(padded_grid)
            masks.append(mask)
        except ValueError as e:
            logger.error(f"Error padding array {i}: {e}")
            raise

    # Add empty arrays for remaining slots
    empty_slots = target_length - len(arrays)
    if empty_slots > 0:
        empty_grid = jnp.full(
            (target_height, target_width),
            fill_value,
            dtype=arrays[0].dtype if arrays else jnp.int32,
        )
        empty_mask = jnp.zeros((target_height, target_width), dtype=jnp.bool_)

        for _ in range(empty_slots):
            padded_grids.append(empty_grid)
            masks.append(empty_mask)

    # Stack into final arrays
    final_arrays = jnp.stack(padded_grids, axis=0)
    final_masks = jnp.stack(masks, axis=0)

    return final_arrays, final_masks


# =============================================================================
# Bounding Box Utilities
# =============================================================================


@jax.jit
def get_selection_bounding_box(
    selection: SelectionArray,
) -> tuple[jnp.int32, jnp.int32, jnp.int32, jnp.int32]:
    """Calculate minimal bounding box for selected pixels.

    JAX-compatible function that calculates the smallest rectangle containing
    all selected (True) pixels in the selection mask.

    Args:
        selection: Boolean selection mask indicating selected pixels

    Returns:
        Tuple of (min_row, max_row, min_col, max_col) representing the bounding box.
        Returns (-1, -1, -1, -1) if no pixels are selected (empty selection).

    Example:
        >>> selection = jnp.array([[False, True, False],
        ...                       [True, True, False],
        ...                       [False, False, False]])
        >>> get_selection_bounding_box(selection)
        (0, 1, 0, 1)  # min_row=0, max_row=1, min_col=0, max_col=1
    """
    has_selection = jnp.sum(selection) > 0

    def calculate_bbox():
        # Create coordinate grids
        height, width = selection.shape
        rows = jnp.arange(height, dtype=jnp.int32)[:, None]
        cols = jnp.arange(width, dtype=jnp.int32)[None, :]

        # Find selected coordinates using JAX-compatible operations
        # For min calculations, use max int value for non-selected pixels
        selected_rows = jnp.where(selection, rows, jnp.iinfo(jnp.int32).max)
        selected_cols = jnp.where(selection, cols, jnp.iinfo(jnp.int32).max)

        # For max calculations, use -1 for non-selected pixels
        max_selected_rows = jnp.where(selection, rows, -1)
        max_selected_cols = jnp.where(selection, cols, -1)

        # Calculate bounds
        min_row = jnp.min(selected_rows)
        max_row = jnp.max(max_selected_rows)
        min_col = jnp.min(selected_cols)
        max_col = jnp.max(max_selected_cols)

        return min_row, max_row, min_col, max_col

    def no_selection():
        return jnp.int32(-1), jnp.int32(-1), jnp.int32(-1), jnp.int32(-1)

    return jax.lax.cond(has_selection, calculate_bbox, no_selection)


# =============================================================================
# Rectangle Extraction Utilities
# =============================================================================


@jax.jit
def extract_object_rectangle(
    grid: GridArray, selection: SelectionArray
) -> tuple[GridArray, tuple[jnp.int32, jnp.int32, jnp.int32, jnp.int32]]:
    """Extract rectangular region containing all selected pixels (JIT-compatible).

    JAX-compatible function that creates a masked version of the full grid
    where only the bounding box region is preserved. This maintains static
    shapes for JIT compatibility while providing the bounding box coordinates
    for further processing.

    Args:
        grid: Source grid to extract from (fixed size)
        selection: Boolean selection mask indicating pixels to include in bounding box

    Returns:
        Tuple of (masked_grid, bounding_box_coords) where:
        - masked_grid: Full-size grid with only bounding box region preserved, rest zeroed
        - bounding_box_coords: Tuple of (min_row, max_row, min_col, max_col)

    Note:
        - If selection is empty, returns zeroed grid and (-1, -1, -1, -1) coords
        - The masked grid preserves spatial relationships within the bounding box
        - Non-selected pixels within the bounding box are preserved as-is (including zeros)
        - This function is JIT-compatible by maintaining static array shapes
        - Use the bounding box coordinates to determine the actual extraction region

    Example:
        >>> grid = jnp.array([[1, 2, 3, 0],
        ...                   [4, 5, 6, 0],
        ...                   [0, 0, 0, 0]])
        >>> selection = jnp.array([[True, False, True, False],
        ...                        [False, True, False, False],
        ...                        [False, False, False, False]])
        >>> masked_grid, bbox = extract_object_rectangle(grid, selection)
        >>> bbox
        (0, 1, 0, 2)  # min_row=0, max_row=1, min_col=0, max_col=2
        >>> # The actual rectangle can be extracted using: grid[0:2, 0:3]
    """
    # Get bounding box coordinates
    min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

    # Check if we have a valid bounding box
    has_valid_bbox = min_row >= 0

    # Create coordinate grids for masking
    height, width = grid.shape
    rows = jnp.arange(height)[:, None]
    cols = jnp.arange(width)[None, :]

    # Create bounding box mask - only preserve pixels within the bounding box
    bbox_mask = (
        (rows >= min_row) & (rows <= max_row) & (cols >= min_col) & (cols <= max_col)
    )

    # For invalid bounding box, create empty mask
    empty_mask = jnp.zeros_like(grid, dtype=jnp.bool_)

    # Select appropriate mask based on validity
    final_mask = jnp.where(has_valid_bbox, bbox_mask, empty_mask)

    # Apply mask to grid - preserve bounding box region, zero everything else
    masked_grid = jnp.where(final_mask, grid, 0)

    # Return the masked grid and bounding box coordinates
    bbox = (min_row, max_row, min_col, max_col)

    return masked_grid, bbox


@jax.jit
def extract_bounding_box_region(
    grid: GridArray,
    min_row: jnp.int32,
    max_row: jnp.int32,
    min_col: jnp.int32,
    max_col: jnp.int32,
) -> GridArray:
    """Extract a rectangular region using bounding box coordinates (JIT-compatible).

    This function creates a mask for the specified bounding box region and
    returns a grid where only that region is preserved (rest is zeroed).
    This maintains static shapes for JIT compatibility.

    Args:
        grid: Source grid to extract from
        min_row: Top edge of bounding box
        max_row: Bottom edge of bounding box
        min_col: Left edge of bounding box
        max_col: Right edge of bounding box

    Returns:
        Grid with only the bounding box region preserved, rest zeroed

    Note:
        - Invalid coordinates (e.g., -1 values) result in empty extraction
        - This function maintains the original grid shape for JIT compatibility
        - Use this with coordinates from get_selection_bounding_box()

    Example:
        >>> grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> extracted = extract_bounding_box_region(grid, 0, 1, 0, 1)
        >>> extracted
        Array([[1, 2, 0],
               [4, 5, 0],
               [0, 0, 0]], dtype=int32)
    """
    # Check if we have valid coordinates
    has_valid_coords = min_row >= 0

    # Create coordinate grids
    height, width = grid.shape
    rows = jnp.arange(height)[:, None]
    cols = jnp.arange(width)[None, :]

    # Create bounding box mask
    bbox_mask = (
        (rows >= min_row) & (rows <= max_row) & (cols >= min_col) & (cols <= max_col)
    )

    # For invalid coordinates, use empty mask
    empty_mask = jnp.zeros_like(grid, dtype=jnp.bool_)

    # Select appropriate mask
    final_mask = jnp.where(has_valid_coords, bbox_mask, empty_mask)

    # Apply mask to grid
    return jnp.where(final_mask, grid, 0)


# =============================================================================
# Selection Validation Utilities
# =============================================================================


@jax.jit
def validate_single_cell_selection(selection: SelectionArray) -> jnp.bool_:
    """Validate that exactly one cell is selected.

    JAX-compatible function that checks if the selection mask contains
    exactly one True value. Used for operations like flood fill that
    require a single starting point.

    Args:
        selection: Boolean selection mask indicating selected pixels

    Returns:
        True if exactly one cell is selected, False otherwise

    Example:
        >>> # Single cell selected - valid
        >>> selection = jnp.array([[False, True, False],
        ...                       [False, False, False],
        ...                       [False, False, False]])
        >>> validate_single_cell_selection(selection)
        True

        >>> # Multiple cells selected - invalid
        >>> selection = jnp.array([[True, True, False],
        ...                       [False, False, False],
        ...                       [False, False, False]])
        >>> validate_single_cell_selection(selection)
        False

        >>> # No cells selected - invalid
        >>> selection = jnp.array([[False, False, False],
        ...                       [False, False, False],
        ...                       [False, False, False]])
        >>> validate_single_cell_selection(selection)
        False
    """
    selection_count = jnp.sum(selection)
    return selection_count == 1


# =============================================================================
# Shape and Mask Utilities
# =============================================================================


def get_actual_grid_shape_from_mask(mask: MaskArray) -> tuple[int, int]:
    """Get the actual shape of a grid based on its validity mask.

    JAX-compatible function that works with JIT compilation.

    Since JAX requires static shapes, grids are often padded to max dimensions,
    but the actual meaningful grid size is determined by the mask.

    Args:
        mask: Boolean mask indicating valid cells in the grid

    Returns:
        Tuple of (height, width) representing the actual grid dimensions
    """
    # Find which rows and columns have any valid cells
    valid_rows = jnp.any(mask, axis=1)
    valid_cols = jnp.any(mask, axis=0)

    # Create index arrays for rows and columns
    row_indices = jnp.arange(mask.shape[0])
    col_indices = jnp.arange(mask.shape[1])

    # Find the maximum valid indices using JAX-compatible operations
    max_row_idx = jnp.where(
        jnp.any(valid_rows), jnp.max(jnp.where(valid_rows, row_indices, -1)), -1
    )
    max_col_idx = jnp.where(
        jnp.any(valid_cols), jnp.max(jnp.where(valid_cols, col_indices, -1)), -1
    )

    # The actual dimensions are the maximum valid indices + 1
    actual_height = jnp.maximum(0, max_row_idx + 1)
    actual_width = jnp.maximum(0, max_col_idx + 1)

    return (actual_height, actual_width)


@jax.jit
def compute_grid_similarity(
    working_grid: GridArray,
    working_mask: SelectionArray,
    target_grid: GridArray,
    target_mask: SelectionArray,
) -> SimilarityScore:
    """Compute size-aware similarity between working and target grids.

    This function compares grids on a fixed canvas based on the target grid's
    actual dimensions. Size mismatches are properly penalized.

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
