"""
Grid Operations Module - JAX-compatible operations for grid manipulation.

This module implements core grid operations that transform grids based on
selection masks and operation IDs. All operations are JAX-compiled for performance.

Key Features:
- Mask-Aware Auto-Selection: When no region is selected, operations automatically
  use the working_grid_mask to select only the active (non-padded) grid area
- Unified Operation Logic: Single code path handles both explicit selections and
  auto-selections using effective_selection pattern
- Boundary Respect: All operations respect grid boundaries defined by working_grid_mask
- Dynamic Resizing: Grid active area can be expanded/shrunk with proper content management

Operations:
- 0-9: Fill colors (fill selection with color 0-9)
- 10-19: Flood fill colors (flood fill from selection with color 0-9)
- 20-23: Move object (up, down, left, right) with edge wrapping within active area
- 24-25: Rotate object (90° clockwise, 90° counterclockwise)
- 26-27: Flip object (horizontal, vertical)
- 28-30: Clipboard operations (copy, paste, cut)
- 31-33: Grid operations (clear, copy input, dynamic resize)
- 34: Submit (mark as terminated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ..utils.jax_types import (
    ColorValue,
    GridArray,
    OperationId,
    SelectionArray,
    SimilarityScore,
)

if TYPE_CHECKING:
    from ..state import ArcEnvState


@jax.jit
def compute_grid_similarity(grid1: GridArray, grid2: GridArray) -> SimilarityScore:
    """Compute pixel-wise similarity between two grids."""
    # Only compare valid regions (non-negative values)
    valid_mask1 = grid1 >= 0
    valid_mask2 = grid2 >= 0
    valid_mask = valid_mask1 & valid_mask2

    # Count matches in valid regions
    matches = jnp.sum((grid1 == grid2) & valid_mask)
    total_valid = jnp.sum(valid_mask)

    # Avoid division by zero
    return jnp.where(
        total_valid > 0,
        matches.astype(jnp.float32) / total_valid.astype(jnp.float32),
        0.0,
    )


@jax.jit
def _copy_grid_to_target_shape(
    source_grid: GridArray, target_shape_grid: GridArray
) -> GridArray:
    """
    Copy source grid to a new grid with target shape, filling with zeros.

    Args:
        source_grid: Source grid to copy
        target_shape_grid: Grid with the desired target shape

    Returns:
        New grid with target shape containing source grid data
    """
    # Create new grid with target shape, filled with zeros
    new_grid = jnp.zeros_like(target_shape_grid)

    # Use dynamic_update_slice to copy source to top-left corner
    return jax.lax.dynamic_update_slice(new_grid, source_grid, (0, 0))


@jax.jit
def apply_within_bounds(
    grid: GridArray, selection: SelectionArray, new_values: ColorValue | GridArray
) -> GridArray:
    """Apply new values to grid only where selection is True."""
    return jnp.where(selection, new_values, grid)


# --- Color Fill Operations (0-9) ---


@jax.jit
def fill_color(
    state: ArcEnvState, selection: SelectionArray, color: ColorValue
) -> ArcEnvState:
    """Fill selected region with specified color."""
    new_grid = apply_within_bounds(state.working_grid, selection, color)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


# --- Flood Fill Operations (10-19) ---


@jax.jit
def simple_flood_fill(
    grid: GridArray,
    selection: SelectionArray,
    fill_color: ColorValue,
    max_iterations: int = 64,
) -> GridArray:
    """Simple flood fill with fixed iterations for JAX compatibility."""
    # Find the first selected pixel as starting point
    h, w = grid.shape
    flat_selection = selection.flatten()
    has_selection = jnp.sum(selection) > 0

    def get_start_pos():
        first_idx = jnp.argmax(flat_selection)
        start_y = first_idx // w
        start_x = first_idx % w
        return start_y, start_x

    def no_fill():
        return grid

    def do_flood_fill():
        start_y, start_x = get_start_pos()
        target_color = grid[start_y, start_x]

        # Initialize flood mask
        initial_flood_mask = jnp.zeros_like(grid, dtype=jnp.bool_)
        initial_flood_mask = initial_flood_mask.at[start_y, start_x].set(True)

        def flood_step(_i, flood_mask):
            # Expand in 4 directions
            up = jnp.roll(flood_mask, -1, axis=0)
            down = jnp.roll(flood_mask, 1, axis=0)
            left = jnp.roll(flood_mask, -1, axis=1)
            right = jnp.roll(flood_mask, 1, axis=1)

            # Combine expansions
            expanded = flood_mask | up | down | left | right

            # Only keep pixels with target color
            return expanded & (grid == target_color)

        # Run flood fill with fixed iterations using JAX loop
        final_flood_mask = jax.lax.fori_loop(
            0, max_iterations, flood_step, initial_flood_mask
        )

        # Apply fill color
        return jnp.where(final_flood_mask, fill_color, grid)

    return jax.lax.cond(has_selection, do_flood_fill, no_fill)


@jax.jit
def flood_fill_color(
    state: ArcEnvState, selection: SelectionArray, color: ColorValue
) -> ArcEnvState:
    """Flood fill from selected region with specified color."""
    new_grid = simple_flood_fill(state.working_grid, selection, color)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


# --- Object Movement Operations (20-23) ---


@jax.jit
def move_object(
    state: ArcEnvState, selection: SelectionArray, direction: int
) -> ArcEnvState:
    """Move selected object in specified direction (0=up, 1=down, 2=left, 3=right)."""
    # If no selection, auto-select the entire working grid
    has_selection = jnp.sum(selection) > 0
    effective_selection = jnp.where(
        has_selection,
        selection,
        state.working_grid_mask,  # Select entire grid if no selection
    )

    # Extract selected object
    object_pixels = jnp.where(effective_selection, state.working_grid, 0)
    # Clear original positions
    cleared_grid = jnp.where(effective_selection, 0, state.working_grid)

    # Move object based on direction (with wrapping)
    def move_up():
        return jnp.roll(object_pixels, -1, axis=0)

    def move_down():
        return jnp.roll(object_pixels, 1, axis=0)

    def move_left():
        return jnp.roll(object_pixels, -1, axis=1)

    def move_right():
        return jnp.roll(object_pixels, 1, axis=1)

    moved_object = jax.lax.switch(
        direction, [move_up, move_down, move_left, move_right]
    )
    # Combine with cleared grid
    new_grid = jnp.where(moved_object > 0, moved_object, cleared_grid)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


# --- Object Rotation Operations (24-25) ---


@jax.jit
def rotate_object(
    state: ArcEnvState, selection: SelectionArray, angle: int
) -> ArcEnvState:
    """Rotate selected region (0=90° clockwise, 1=90° counterclockwise)."""
    # If no selection, auto-select the entire working grid
    has_selection = jnp.sum(selection) > 0
    effective_selection = jnp.where(
        has_selection,
        selection,
        state.working_grid_mask,  # Select entire grid if no selection
    )

    # Extract selected region
    selected_region = jnp.where(effective_selection, state.working_grid, 0)
    # Clear original positions
    cleared_grid = jnp.where(effective_selection, 0, state.working_grid)

    # Get original grid shape
    orig_height, orig_width = state.working_grid.shape

    # Rotate selected region
    def rotate_clockwise():
        return jnp.rot90(selected_region, k=-1)  # k=-1 for clockwise

    def rotate_counterclockwise():
        return jnp.rot90(selected_region, k=1)  # k=1 for counterclockwise

    rotated_region = jax.lax.switch(angle, [rotate_clockwise, rotate_counterclockwise])

    # Handle dimension mismatch by ensuring rotated region matches original dimensions
    rotated_height, rotated_width = rotated_region.shape

    # Create a new array with the original dimensions, filled with zeros
    final_rotated_region = jnp.zeros(
        (orig_height, orig_width), dtype=rotated_region.dtype
    )

    # Calculate how much we can copy from the rotated region
    copy_height = min(rotated_height, orig_height)
    copy_width = min(rotated_width, orig_width)

    # Copy the overlapping part
    final_rotated_region = final_rotated_region.at[:copy_height, :copy_width].set(
        rotated_region[:copy_height, :copy_width]
    )

    # Combine with cleared grid
    new_grid = jnp.where(final_rotated_region != 0, final_rotated_region, cleared_grid)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


# --- Object Flip Operations (26-27) ---


@jax.jit
def flip_object(
    state: ArcEnvState, selection: SelectionArray, axis: int
) -> ArcEnvState:
    """Flip selected region (0=horizontal, 1=vertical)."""
    # If no selection, auto-select the entire working grid
    has_selection = jnp.sum(selection) > 0
    effective_selection = jnp.where(
        has_selection,
        selection,
        state.working_grid_mask,  # Select entire grid if no selection
    )

    # Extract selected region
    selected_region = jnp.where(effective_selection, state.working_grid, 0)
    # Clear original positions
    cleared_grid = jnp.where(effective_selection, 0, state.working_grid)

    # Flip selected region
    def flip_horizontal():
        return jnp.fliplr(selected_region)

    def flip_vertical():
        return jnp.flipud(selected_region)

    flipped_region = jax.lax.switch(axis, [flip_horizontal, flip_vertical])
    # Combine with cleared grid
    new_grid = jnp.where(flipped_region != 0, flipped_region, cleared_grid)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


# --- Clipboard Operations (28-30) ---


@jax.jit
def copy_to_clipboard(state: ArcEnvState, selection: SelectionArray) -> ArcEnvState:
    """Copy selected region to clipboard."""
    new_clipboard = jnp.where(selection, state.working_grid, 0)
    return eqx.tree_at(lambda s: s.clipboard, state, new_clipboard)


@jax.jit
def paste_from_clipboard(state: ArcEnvState, selection: SelectionArray) -> ArcEnvState:
    """Paste clipboard content to selected region."""
    # Find the bounding boxes of clipboard content and selection
    clipboard_mask = state.clipboard != 0

    # Check if we have clipboard content and selection
    has_clipboard = jnp.any(clipboard_mask)
    has_selection = jnp.any(selection)
    should_paste = has_clipboard & has_selection

    # Create coordinate grids with integer type
    rows = jnp.arange(state.working_grid.shape[0], dtype=jnp.int32)[:, None]
    cols = jnp.arange(state.working_grid.shape[1], dtype=jnp.int32)[None, :]

    # Find minimum coordinates using masking instead of jnp.where
    # Use a large integer instead of inf to keep integer types
    large_int = jnp.iinfo(jnp.int32).max

    # For clipboard
    clipboard_rows_masked = jnp.where(clipboard_mask, rows, large_int)
    clipboard_cols_masked = jnp.where(clipboard_mask, cols, large_int)
    clipboard_min_r = jnp.where(
        has_clipboard, jnp.min(clipboard_rows_masked), 0
    ).astype(jnp.int32)
    clipboard_min_c = jnp.where(
        has_clipboard, jnp.min(clipboard_cols_masked), 0
    ).astype(jnp.int32)

    # For selection
    selection_rows_masked = jnp.where(selection, rows, large_int)
    selection_cols_masked = jnp.where(selection, cols, large_int)
    selection_min_r = jnp.where(
        has_selection, jnp.min(selection_rows_masked), 0
    ).astype(jnp.int32)
    selection_min_c = jnp.where(
        has_selection, jnp.min(selection_cols_masked), 0
    ).astype(jnp.int32)

    # Calculate the offset to align clipboard with selection
    offset_r = (selection_min_r - clipboard_min_r).astype(jnp.int32)
    offset_c = (selection_min_c - clipboard_min_c).astype(jnp.int32)

    # Map each grid position back to clipboard position
    clipboard_r = (rows - offset_r).astype(jnp.int32)
    clipboard_c = (cols - offset_c).astype(jnp.int32)

    # Check bounds for clipboard access
    valid_r = (clipboard_r >= 0) & (clipboard_r < state.clipboard.shape[0])
    valid_c = (clipboard_c >= 0) & (clipboard_c < state.clipboard.shape[1])
    valid_bounds = valid_r & valid_c

    # Get clipboard values, using 0 for out-of-bounds
    clipboard_values = jnp.where(
        valid_bounds,
        state.clipboard[
            jnp.clip(clipboard_r, 0, state.clipboard.shape[0] - 1),
            jnp.clip(clipboard_c, 0, state.clipboard.shape[1] - 1),
        ],
        0,
    )

    # Only paste if we should paste and where selected
    paste_mask = should_paste & selection
    new_grid = jnp.where(paste_mask, clipboard_values, state.working_grid)

    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


@jax.jit
def cut_to_clipboard(state: ArcEnvState, selection: SelectionArray) -> ArcEnvState:
    """Cut selected region to clipboard (copy + clear)."""
    # Copy to clipboard
    new_clipboard = jnp.where(selection, state.working_grid, 0)

    # Clear selected region
    new_grid = jnp.where(selection, 0, state.working_grid)

    # Update both working_grid and clipboard using Equinox tree_at
    return eqx.tree_at(
        lambda s: (s.working_grid, s.clipboard), state, (new_grid, new_clipboard)
    )


# --- Grid Operations (31-33) ---


@jax.jit
def clear_grid(state: ArcEnvState, selection: SelectionArray) -> ArcEnvState:
    """Clear the entire grid or selected region."""
    has_selection = jnp.sum(selection) > 0

    def clear_selection():
        return jnp.where(selection, 0, state.working_grid)

    def clear_all():
        return jnp.zeros_like(state.working_grid)

    new_grid = jax.lax.cond(has_selection, clear_selection, clear_all)
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


@jax.jit
def copy_input_grid(state: ArcEnvState, _selection: SelectionArray) -> ArcEnvState:
    """Copy input grid to current grid."""
    input_grid = state.task_data.input_grids_examples[state.current_example_idx]
    # Copy input grid to working grid shape
    new_working_grid = _copy_grid_to_target_shape(input_grid, state.working_grid)
    return eqx.tree_at(lambda s: s.working_grid, state, new_working_grid)


@jax.jit
def resize_grid(state: ArcEnvState, selection: SelectionArray) -> ArcEnvState:
    """Resize grid active area based on selection."""
    has_selection = jnp.sum(selection) > 0

    def resize_to_selection():
        # Use selection as new working grid mask (defines new active area)
        new_mask = selection

        # Handle grid content changes:
        # - Areas that were active and remain active: keep existing values
        # - Areas that were inactive and become active: set to background (0)
        # - Areas that were active and become inactive: set to padding (-1)

        # Find areas becoming active (were padding, now selected)
        becoming_active = selection & ~state.working_grid_mask

        # Find areas becoming inactive (were active, now not selected)
        becoming_inactive = state.working_grid_mask & ~selection

        # Update grid values
        new_grid = state.working_grid
        new_grid = jnp.where(
            becoming_active, 0, new_grid
        )  # New active areas = background
        new_grid = jnp.where(
            becoming_inactive, -1, new_grid
        )  # New inactive areas = padding

        return eqx.tree_at(
            lambda s: (s.working_grid, s.working_grid_mask), state, (new_grid, new_mask)
        )

    def no_resize():
        return state

    return jax.lax.cond(has_selection, resize_to_selection, no_resize)


# --- Submit Operation (34) ---


@jax.jit
def submit_solution(state: ArcEnvState, _selection: SelectionArray) -> ArcEnvState:
    """Submit current grid as solution."""
    return eqx.tree_at(lambda s: s.episode_done, state, True)


# --- Main Operation Execution ---


@jax.jit
def execute_grid_operation(state: ArcEnvState, operation: OperationId) -> ArcEnvState:
    """Execute grid operation based on operation ID."""
    selection = state.selected

    # Define all operations
    def op_0():
        return fill_color(state, selection, 0)

    def op_1():
        return fill_color(state, selection, 1)

    def op_2():
        return fill_color(state, selection, 2)

    def op_3():
        return fill_color(state, selection, 3)

    def op_4():
        return fill_color(state, selection, 4)

    def op_5():
        return fill_color(state, selection, 5)

    def op_6():
        return fill_color(state, selection, 6)

    def op_7():
        return fill_color(state, selection, 7)

    def op_8():
        return fill_color(state, selection, 8)

    def op_9():
        return fill_color(state, selection, 9)

    def op_10():
        return flood_fill_color(state, selection, 0)

    def op_11():
        return flood_fill_color(state, selection, 1)

    def op_12():
        return flood_fill_color(state, selection, 2)

    def op_13():
        return flood_fill_color(state, selection, 3)

    def op_14():
        return flood_fill_color(state, selection, 4)

    def op_15():
        return flood_fill_color(state, selection, 5)

    def op_16():
        return flood_fill_color(state, selection, 6)

    def op_17():
        return flood_fill_color(state, selection, 7)

    def op_18():
        return flood_fill_color(state, selection, 8)

    def op_19():
        return flood_fill_color(state, selection, 9)

    def op_20():
        return move_object(state, selection, 0)  # up

    def op_21():
        return move_object(state, selection, 1)  # down

    def op_22():
        return move_object(state, selection, 2)  # left

    def op_23():
        return move_object(state, selection, 3)  # right

    def op_24():
        return rotate_object(state, selection, 0)  # 90° clockwise

    def op_25():
        return rotate_object(state, selection, 1)  # 90° counterclockwise

    def op_26():
        return flip_object(state, selection, 0)  # horizontal

    def op_27():
        return flip_object(state, selection, 1)  # vertical

    def op_28():
        return copy_to_clipboard(state, selection)

    def op_29():
        return paste_from_clipboard(state, selection)

    def op_30():
        return cut_to_clipboard(state, selection)

    def op_31():
        return clear_grid(state, selection)

    def op_32():
        return copy_input_grid(state, selection)

    def op_33():
        return resize_grid(state, selection)

    def op_34():
        return submit_solution(state, selection)

    # Operation dispatch using jax.lax.switch
    operations = [
        op_0,
        op_1,
        op_2,
        op_3,
        op_4,
        op_5,
        op_6,
        op_7,
        op_8,
        op_9,
        op_10,
        op_11,
        op_12,
        op_13,
        op_14,
        op_15,
        op_16,
        op_17,
        op_18,
        op_19,
        op_20,
        op_21,
        op_22,
        op_23,
        op_24,
        op_25,
        op_26,
        op_27,
        op_28,
        op_29,
        op_30,
        op_31,
        op_32,
        op_33,
        op_34,
    ]

    # Execute operation using JAX switch (handles traced arrays)
    new_state = jax.lax.switch(operation, operations)

    # Update similarity score if grid changed
    target_grid = new_state.task_data.output_grids_examples[
        new_state.current_example_idx
    ]
    similarity = compute_grid_similarity(new_state.working_grid, target_grid)

    return eqx.tree_at(lambda s: s.similarity_score, new_state, similarity)
