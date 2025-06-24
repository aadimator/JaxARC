"""
ARCLE Operations Module - JAX-compatible operations for grid manipulation.

This module implements the core ARCLE operations that transform grids based on
selection masks and operation IDs. All operations are JAX-compiled for performance.

Operations:
- 0-9: Fill colors (fill selection with color 0-9)
- 10-19: Flood fill colors (flood fill from selection with color 0-9)
- 20-23: Move object (up, down, left, right)
- 24-25: Rotate object (90°, 270°)
- 26-27: Flip object (horizontal, vertical)
- 28-30: Clipboard operations (copy, paste, cut)
- 31-33: Grid operations (clear, copy input, resize)
- 34: Submit (mark as terminated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .arcle_env import ARCLEState


@jax.jit
def compute_grid_similarity(grid1: jnp.ndarray, grid2: jnp.ndarray) -> jnp.ndarray:
    """Compute pixel-wise similarity between two grids."""
    # Only compare valid regions (non-negative values)
    valid_mask1 = grid1 >= 0
    valid_mask2 = grid2 >= 0
    valid_mask = valid_mask1 & valid_mask2

    # Count matches in valid regions
    matches = jnp.sum((grid1 == grid2) & valid_mask)
    total_valid = jnp.sum(valid_mask)

    # Avoid division by zero
    similarity = jnp.where(
        total_valid > 0,
        matches.astype(jnp.float32) / total_valid.astype(jnp.float32),
        0.0,
    )

    return similarity


@jax.jit
def apply_within_bounds(
    grid: jnp.ndarray, selection: jnp.ndarray, new_values: jnp.ndarray
) -> jnp.ndarray:
    """Apply new values to grid only where selection is True."""
    return jnp.where(selection, new_values, grid)


# --- Color Fill Operations (0-9) ---


@jax.jit
def fill_color(state: ARCLEState, selection: jnp.ndarray, color: int) -> ARCLEState:
    """Fill selected region with specified color."""
    new_grid = apply_within_bounds(state.working_grid, selection, color)
    return state.replace(working_grid=new_grid)


# --- Flood Fill Operations (10-19) ---


@jax.jit
def simple_flood_fill(
    grid: jnp.ndarray, selection: jnp.ndarray, fill_color: int, max_iterations: int = 64
) -> jnp.ndarray:
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

        def flood_step(i, flood_mask):
            # Expand in 4 directions
            up = jnp.roll(flood_mask, -1, axis=0)
            down = jnp.roll(flood_mask, 1, axis=0)
            left = jnp.roll(flood_mask, -1, axis=1)
            right = jnp.roll(flood_mask, 1, axis=1)

            # Combine expansions
            expanded = flood_mask | up | down | left | right

            # Only keep pixels with target color
            new_mask = expanded & (grid == target_color)

            return new_mask

        # Run flood fill with fixed iterations using JAX loop
        final_flood_mask = jax.lax.fori_loop(
            0, max_iterations, flood_step, initial_flood_mask
        )

        # Apply fill color
        return jnp.where(final_flood_mask, fill_color, grid)

    return jax.lax.cond(has_selection, do_flood_fill, no_fill)


@jax.jit
def flood_fill_color(
    state: ARCLEState, selection: jnp.ndarray, color: int
) -> ARCLEState:
    """Flood fill from selected region with specified color."""
    new_grid = simple_flood_fill(state.working_grid, selection, color)
    return state.replace(working_grid=new_grid)


# --- Object Movement Operations (20-23) ---


@jax.jit
def move_object(
    state: ARCLEState, selection: jnp.ndarray, direction: int
) -> ARCLEState:
    """Move selected object in specified direction (0=up, 1=down, 2=left, 3=right)."""
    # Extract selected object
    object_pixels = jnp.where(selection, state.working_grid, 0)

    # Clear original positions
    cleared_grid = jnp.where(selection, 0, state.working_grid)

    # Move object based on direction
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
    return state.replace(working_grid=new_grid)


# --- Object Rotation Operations (24-25) ---


@jax.jit
def rotate_object(state: ARCLEState, selection: jnp.ndarray, angle: int) -> ARCLEState:
    """Rotate selected object (0=90°, 1=270°)."""
    # Extract selected object
    object_pixels = jnp.where(selection, state.working_grid, 0)

    # Clear original positions
    cleared_grid = jnp.where(selection, 0, state.working_grid)

    # Rotate object
    def rotate_90():
        return jnp.rot90(object_pixels, k=1)

    def rotate_270():
        return jnp.rot90(object_pixels, k=3)

    rotated_object = jax.lax.switch(angle, [rotate_90, rotate_270])

    # Combine with cleared grid
    new_grid = jnp.where(rotated_object > 0, rotated_object, cleared_grid)
    return state.replace(working_grid=new_grid)


# --- Object Flip Operations (26-27) ---


@jax.jit
def flip_object(state: ARCLEState, selection: jnp.ndarray, axis: int) -> ARCLEState:
    """Flip selected object (0=horizontal, 1=vertical)."""
    # Extract selected object
    object_pixels = jnp.where(selection, state.working_grid, 0)

    # Clear original positions
    cleared_grid = jnp.where(selection, 0, state.working_grid)

    # Flip object
    def flip_horizontal():
        return jnp.fliplr(object_pixels)

    def flip_vertical():
        return jnp.flipud(object_pixels)

    flipped_object = jax.lax.switch(axis, [flip_horizontal, flip_vertical])

    # Combine with cleared grid
    new_grid = jnp.where(flipped_object > 0, flipped_object, cleared_grid)
    return state.replace(working_grid=new_grid)


# --- Clipboard Operations (28-30) ---


@jax.jit
def copy_to_clipboard(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Copy selected region to clipboard."""
    new_clipboard = jnp.where(selection, state.working_grid, 0)
    return state.replace(clipboard=new_clipboard)


@jax.jit
def paste_from_clipboard(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Paste clipboard content to selected region."""
    # Only paste where clipboard has content and selection is active
    paste_mask = selection & (state.clipboard != 0)
    new_grid = jnp.where(paste_mask, state.clipboard, state.working_grid)
    return state.replace(working_grid=new_grid)


@jax.jit
def cut_to_clipboard(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Cut selected region to clipboard (copy + clear)."""
    # Copy to clipboard
    new_clipboard = jnp.where(selection, state.working_grid, 0)

    # Clear selected region
    new_grid = jnp.where(selection, 0, state.working_grid)

    return state.replace(working_grid=new_grid, clipboard=new_clipboard)


# --- Grid Operations (31-33) ---


@jax.jit
def clear_grid(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Clear the entire grid or selected region."""
    has_selection = jnp.sum(selection) > 0

    def clear_selection():
        return jnp.where(selection, 0, state.working_grid)

    def clear_all():
        return jnp.zeros_like(state.working_grid)

    new_grid = jax.lax.cond(has_selection, clear_selection, clear_all)
    return state.replace(working_grid=new_grid)


@jax.jit
def copy_input_grid(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Copy input grid to current grid."""
    input_grid = state.task_data.input_grids_examples[state.active_train_pair_idx]
    return state.replace(working_grid=input_grid)


@jax.jit
def resize_grid(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Resize grid (no-op for fixed-size implementation)."""
    # In fixed-size grid implementation, this is a no-op
    return state


# --- Submit Operation (34) ---


@jax.jit
def submit_solution(state: ARCLEState, selection: jnp.ndarray) -> ARCLEState:
    """Submit current grid as solution."""
    return state.replace(terminated=jnp.array(True, dtype=jnp.bool_))


# --- Main Operation Execution ---


@jax.jit
def execute_arcle_operation(state: ARCLEState, operation: jnp.ndarray) -> ARCLEState:
    """Execute ARCLE operation based on operation ID."""
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
        return rotate_object(state, selection, 0)  # 90°

    def op_25():
        return rotate_object(state, selection, 1)  # 270°

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
        new_state.active_train_pair_idx
    ]
    similarity = compute_grid_similarity(new_state.working_grid, target_grid)

    return new_state.replace(similarity_score=similarity)
