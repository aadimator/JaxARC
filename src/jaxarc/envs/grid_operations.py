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

from jaxarc.utils.grid_utils import (
    compute_grid_similarity,
    extract_bounding_box_region,
    get_selection_bounding_box,
    validate_single_cell_selection,
)
from jaxarc.utils.state_utils import (
    update_multiple_fields,
    update_similarity_score,
    update_working_grid,
)

from ..types import (
    ColorValue,
    GridArray,
    OperationId,
    SelectionArray,
)
from ..utils.state_utils import validate_state_consistency

if TYPE_CHECKING:
    from ..state import State

# Central operation names mapping
OPERATION_NAMES = {
    # Fill operations (0-9)
    0: "Fill 0",
    1: "Fill 1",
    2: "Fill 2",
    3: "Fill 3",
    4: "Fill 4",
    5: "Fill 5",
    6: "Fill 6",
    7: "Fill 7",
    8: "Fill 8",
    9: "Fill 9",
    # Flood fill operations (10-19)
    10: "Flood Fill 0",
    11: "Flood Fill 1",
    12: "Flood Fill 2",
    13: "Flood Fill 3",
    14: "Flood Fill 4",
    15: "Flood Fill 5",
    16: "Flood Fill 6",
    17: "Flood Fill 7",
    18: "Flood Fill 8",
    19: "Flood Fill 9",
    # Movement operations (20-23)
    20: "Move Up",
    21: "Move Down",
    22: "Move Left",
    23: "Move Right",
    # Transformation operations (24-27)
    24: "Rotate CW",
    25: "Rotate CCW",
    26: "Flip H",
    27: "Flip V",
    # Editing operations (28-31)
    28: "Copy",
    29: "Paste",
    30: "Cut",
    31: "Clear",
    # Special operations (32-34)
    32: "Copy Input",
    33: "Resize",
    34: "Submit",
}


def get_operation_name(operation_id: int) -> str:
    """Get the human-readable name for an operation ID.

    Args:
        operation_id: Integer operation ID

    Returns:
        Human-readable operation name

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.grid_operations import get_operation_name

        name = get_operation_name(0)  # "Fill 0"
        name = get_operation_name(24)  # "Rotate CW"
        ```
    """
    if operation_id not in OPERATION_NAMES:
        raise ValueError(f"Unknown operation ID: {operation_id}")

    return OPERATION_NAMES[operation_id]


def get_operation_display_text(operation_id: int) -> str:
    """Get display text for visualization (includes ID and name).

    Args:
        operation_id: Integer operation ID

    Returns:
        Display text in format "Op {id}: {name}"

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.grid_operations import get_operation_display_text

        text = get_operation_display_text(0)  # "Op 0: Fill 0"
        text = get_operation_display_text(34)  # "Op 34: Submit"
        ```
    """
    name = get_operation_name(operation_id)
    return f"Op {operation_id}: {name}"


def is_valid_operation_id(operation_id: int) -> bool:
    """Check if an operation ID is valid.

    Args:
        operation_id: Integer operation ID to check

    Returns:
        True if operation ID is valid, False otherwise

    Example:
        ```python
        from jaxarc.envs.grid_operations import is_valid_operation_id

        assert is_valid_operation_id(0) == True
        assert is_valid_operation_id(34) == True
        assert is_valid_operation_id(41) == False
        assert is_valid_operation_id(35) == False
        ```
    """
    return operation_id in OPERATION_NAMES


def get_all_operation_ids() -> list[int]:
    """Get all valid operation IDs.

    Returns:
        List of all valid operation IDs sorted in ascending order

    Example:
        ```python
        from jaxarc.envs.grid_operations import get_all_operation_ids

        ids = get_all_operation_ids()  # [0, 1, 2, ..., 34]
        ```
    """
    return sorted(OPERATION_NAMES.keys())


def get_operations_by_category() -> dict[str, list[int]]:
    """Get operations grouped by category.

    Returns:
        Dictionary mapping category names to lists of operation IDs

    Example:
        ```python
        from jaxarc.envs.grid_operations import get_operations_by_category

        categories = get_operations_by_category()
        fill_ops = categories["fill"]  # [0, 1, 2, ..., 9]
        movement_ops = categories["movement"]  # [20, 21, 22, 23]
        # No control ops (removed)
        ```
    """
    return {
        "fill": list(range(10)),
        "flood_fill": list(range(10, 20)),
        "movement": list(range(20, 24)),
        "transformation": list(range(24, 28)),
        "editing": list(range(28, 32)),
        "special": list(range(32, 35)),
        # Control operations removed
    }


def get_operation_category(operation_id: int) -> str:
    """Get the category name for an operation ID.

    Args:
        operation_id: Integer operation ID

    Returns:
        Category name for the operation

    Raises:
        ValueError: If operation_id is not recognized

    Example:
        ```python
        from jaxarc.envs.grid_operations import get_operation_category

        category = get_operation_category(5)  # "fill"
        category = get_operation_category(24)  # "transformation"
        ```
    """
    if not is_valid_operation_id(operation_id):
        raise ValueError(f"Unknown operation ID: {operation_id}")

    categories = get_operations_by_category()
    for category_name, op_ids in categories.items():
        if operation_id in op_ids:
            return category_name

    # This should never happen if is_valid_operation_id works correctly
    raise ValueError(f"Operation ID {operation_id} not found in any category")


@eqx.filter_jit
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


@eqx.filter_jit
def apply_within_bounds(
    grid: GridArray, selection: SelectionArray, new_values: ColorValue | GridArray
) -> GridArray:
    """Apply new values to grid only where selection is True."""
    return jnp.where(selection, new_values, grid)


@eqx.filter_jit
def get_effective_selection(
    selection: SelectionArray, working_grid_mask: SelectionArray
) -> SelectionArray:
    """Get effective selection for object operations.

    If no selection is provided (all False), auto-selects the entire working grid area.
    This provides consistent behavior across all object operations.

    Args:
        selection: Original selection mask
        working_grid_mask: Mask indicating active grid area

    Returns:
        Effective selection mask to use for the operation
    """
    has_selection = jnp.sum(selection) > 0
    return jnp.where(has_selection, selection, working_grid_mask)


@eqx.filter_jit
def validate_bounding_box_for_operation(
    min_row: jnp.int32,
    max_row: jnp.int32,
    min_col: jnp.int32,
    max_col: jnp.int32,
    require_square: bool = False,
) -> jnp.bool_:
    """Validate bounding box for object operations.

    Provides consistent validation logic across all object operations.

    Args:
        min_row: Minimum row coordinate
        max_row: Maximum row coordinate
        min_col: Minimum column coordinate
        max_col: Maximum column coordinate
        require_square: Whether to require square bounding box (for rotation)

    Returns:
        True if bounding box is valid for the operation, False otherwise
    """
    # Check if we have a valid bounding box
    has_valid_bbox = min_row >= 0

    if require_square:
        # For rotation, also check if bounding box is square
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        is_square = height == width
        return has_valid_bbox & is_square

    return has_valid_bbox


# --- Color Fill Operations (0-9) ---


@eqx.filter_jit
def fill_color(state: State, selection: SelectionArray, color: ColorValue) -> State:
    """Fill selected region with specified color."""

    new_grid = apply_within_bounds(state.working_grid, selection, color)
    return update_working_grid(state, new_grid)


# --- Flood Fill Operations (10-19) ---


@eqx.filter_jit
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
            # Expand in 4 directions without wrapping at boundaries
            h, w = flood_mask.shape

            # Create shifted versions with proper boundary handling
            # Up: shift down and zero out bottom row
            up = jnp.concatenate(
                [flood_mask[1:], jnp.zeros((1, w), dtype=jnp.bool_)], axis=0
            )

            # Down: shift up and zero out top row
            down = jnp.concatenate(
                [jnp.zeros((1, w), dtype=jnp.bool_), flood_mask[:-1]], axis=0
            )

            # Left: shift right and zero out rightmost column
            left = jnp.concatenate(
                [flood_mask[:, 1:], jnp.zeros((h, 1), dtype=jnp.bool_)], axis=1
            )

            # Right: shift left and zero out leftmost column
            right = jnp.concatenate(
                [jnp.zeros((h, 1), dtype=jnp.bool_), flood_mask[:, :-1]], axis=1
            )

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


@eqx.filter_jit
def flood_fill_color(
    state: State, selection: SelectionArray, color: ColorValue
) -> State:
    """Flood fill from selected region with specified color.

    Only performs flood fill if exactly one cell is selected. Returns original
    state unchanged if multiple cells or no cells are selected.
    """
    # Validate that exactly one cell is selected
    is_valid_selection = validate_single_cell_selection(selection)

    def do_flood_fill():
        return simple_flood_fill(state.working_grid, selection, color)

    def no_flood_fill():
        return state.working_grid

    # Only perform flood fill if selection is valid (single cell)
    new_grid = jax.lax.cond(is_valid_selection, do_flood_fill, no_flood_fill)
    return update_working_grid(state, new_grid)


# --- Object Movement Operations (20-23) ---


@eqx.filter_jit
def move_object(state: State, selection: SelectionArray, direction: int) -> State:
    """Move selected object in specified direction (0=up, 1=down, 2=left, 3=right).

    Uses bounding box extraction to move only the rectangular region containing
    all selected pixels. Movement wraps within the bounding box boundaries.

    Args:
        state: Current environment state
        selection: Boolean mask indicating selected pixels
        direction: Movement direction (0=up, 1=down, 2=left, 3=right)

    Returns:
        Updated state with moved object, or original state if invalid bounding box

    Note:
        - If no selection provided, auto-selects entire working grid
        - Movement wraps within bounding box (pixels moving out one side appear on opposite)
        - Returns original state unchanged if bounding box is invalid
    """
    # Get effective selection using standardized utility
    effective_selection = get_effective_selection(selection, state.working_grid_mask)

    # Get bounding box coordinates using consistent utility function
    min_row, max_row, min_col, max_col = get_selection_bounding_box(effective_selection)

    # Validate bounding box using standardized utility
    has_valid_bbox = validate_bounding_box_for_operation(
        min_row, max_row, min_col, max_col
    )

    def apply_move():
        # Create coordinate grids
        grid_height, grid_width = state.working_grid.shape
        rows = jnp.arange(grid_height)[:, None]
        cols = jnp.arange(grid_width)[None, :]

        # Create bounding box mask
        bbox_mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        # Calculate bounding box dimensions
        bbox_height = max_row - min_row + 1
        bbox_width = max_col - min_col + 1

        # For each direction, calculate the new position within the bounding box
        def move_up():
            # Map each position to its new position after moving up with wrapping
            new_row = jnp.where(
                rows == min_row,  # Top row wraps to bottom
                max_row,
                rows - 1,  # Other rows move up by 1
            )
            return jnp.where(bbox_mask, state.working_grid[new_row, cols], 0)

        def move_down():
            # Map each position to its new position after moving down with wrapping
            new_row = jnp.where(
                rows == max_row,  # Bottom row wraps to top
                min_row,
                rows + 1,  # Other rows move down by 1
            )
            return jnp.where(bbox_mask, state.working_grid[new_row, cols], 0)

        def move_left():
            # Map each position to its new position after moving left with wrapping
            new_col = jnp.where(
                cols == min_col,  # Left column wraps to right
                max_col,
                cols - 1,  # Other columns move left by 1
            )
            return jnp.where(bbox_mask, state.working_grid[rows, new_col], 0)

        def move_right():
            # Map each position to its new position after moving right with wrapping
            new_col = jnp.where(
                cols == max_col,  # Right column wraps to left
                min_col,
                cols + 1,  # Other columns move right by 1
            )
            return jnp.where(bbox_mask, state.working_grid[rows, new_col], 0)

        moved_region = jax.lax.switch(
            direction, [move_up, move_down, move_left, move_right]
        )

        # Clear the original bounding box region and place the moved region
        cleared_grid = jnp.where(bbox_mask, 0, state.working_grid)
        result_grid = jnp.where(moved_region > 0, moved_region, cleared_grid)

        return result_grid

    def no_move():
        return state.working_grid

    # Apply move only if we have a valid bounding box
    new_grid = jax.lax.cond(has_valid_bbox, apply_move, no_move)
    return update_working_grid(state, new_grid)


# --- Object Rotation Operations (24-25) ---


@eqx.filter_jit
def rotate_object(state: State, selection: SelectionArray, angle: int) -> State:
    """Rotate selected region (0=90° clockwise, 1=90° counterclockwise).

    Uses bounding box extraction to rotate only the rectangular region containing
    all selected pixels. Only works if the bounding box is square.

    Args:
        state: Current environment state
        selection: Boolean mask indicating selected pixels
        angle: Rotation angle (0=90° clockwise, 1=90° counterclockwise)

    Returns:
        Updated state with rotated object, or original state if bounding box invalid/non-square

    Note:
        - If no selection provided, auto-selects entire working grid
        - Only rotates if bounding box is square (height == width)
        - Returns original state unchanged if bounding box is invalid or non-square
    """
    # Get effective selection using standardized utility
    effective_selection = get_effective_selection(selection, state.working_grid_mask)

    # Get bounding box coordinates using consistent utility function
    min_row, max_row, min_col, max_col = get_selection_bounding_box(effective_selection)

    # Validate bounding box for rotation (requires square) using standardized utility
    can_rotate = validate_bounding_box_for_operation(
        min_row, max_row, min_col, max_col, require_square=True
    )

    def apply_rotation():
        # Extract the bounding box region
        bbox_region = extract_bounding_box_region(
            state.working_grid, min_row, max_row, min_col, max_col
        )

        # Apply rotation
        def rotate_clockwise():
            return jnp.rot90(bbox_region, k=-1)  # Clockwise

        def rotate_counterclockwise():
            return jnp.rot90(bbox_region, k=1)  # Counterclockwise

        rotated_region = jax.lax.switch(
            angle, [rotate_clockwise, rotate_counterclockwise]
        )

        # Create a mask for the bounding box area
        grid_height, grid_width = state.working_grid.shape
        rows = jnp.arange(grid_height)[:, None]
        cols = jnp.arange(grid_width)[None, :]

        bbox_mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        # Clear the original bounding box region and place the rotated region
        cleared_grid = jnp.where(bbox_mask, 0, state.working_grid)
        result_grid = jnp.where(rotated_region > 0, rotated_region, cleared_grid)

        return result_grid

    def no_rotation():
        return state.working_grid

    # Apply rotation only if we have a valid square bounding box
    new_grid = jax.lax.cond(can_rotate, apply_rotation, no_rotation)
    return update_working_grid(state, new_grid)


# --- Object Flip Operations (26-27) ---


@eqx.filter_jit
def flip_object(state: State, selection: SelectionArray, axis: int) -> State:
    """Flip selected region (0=horizontal, 1=vertical).

    Uses bounding box extraction to flip only the rectangular region containing
    all selected pixels.

    Args:
        state: Current environment state
        selection: Boolean mask indicating selected pixels
        axis: Flip axis (0=horizontal, 1=vertical)

    Returns:
        Updated state with flipped object, or original state if invalid bounding box

    Note:
        - If no selection provided, auto-selects entire working grid
        - Flips within the bounding box containing all selected pixels
        - Returns original state unchanged if bounding box is invalid
    """
    # Get effective selection using standardized utility
    effective_selection = get_effective_selection(selection, state.working_grid_mask)

    # Get bounding box coordinates using consistent utility function
    min_row, max_row, min_col, max_col = get_selection_bounding_box(effective_selection)

    # Validate bounding box using standardized utility
    has_valid_bbox = validate_bounding_box_for_operation(
        min_row, max_row, min_col, max_col
    )

    def apply_flip():
        # Create coordinate grids
        grid_height, grid_width = state.working_grid.shape
        rows = jnp.arange(grid_height)[:, None]
        cols = jnp.arange(grid_width)[None, :]

        # Create bounding box mask
        bbox_mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        # For each position in the bounding box, calculate where it should come from after flip
        def flip_horizontal():
            # For horizontal flip, map column positions within bounding box
            # col -> (max_col + min_col) - col
            new_col = jnp.where(bbox_mask, (max_col + min_col) - cols, cols)
            return jnp.where(bbox_mask, state.working_grid[rows, new_col], 0)

        def flip_vertical():
            # For vertical flip, map row positions within bounding box
            # row -> (max_row + min_row) - row
            new_row = jnp.where(bbox_mask, (max_row + min_row) - rows, rows)
            return jnp.where(bbox_mask, state.working_grid[new_row, cols], 0)

        flipped_region = jax.lax.switch(axis, [flip_horizontal, flip_vertical])

        # Clear the original bounding box region and place the flipped region
        cleared_grid = jnp.where(bbox_mask, 0, state.working_grid)
        result_grid = jnp.where(flipped_region > 0, flipped_region, cleared_grid)

        return result_grid

    def no_flip():
        return state.working_grid

    # Apply flip only if we have a valid bounding box
    new_grid = jax.lax.cond(has_valid_bbox, apply_flip, no_flip)
    return update_working_grid(state, new_grid)


# --- Clipboard Operations (28-30) ---


@eqx.filter_jit
def copy_to_clipboard(state: State, selection: SelectionArray) -> State:
    """Copy selected region to clipboard."""
    new_clipboard = jnp.where(selection, state.working_grid, 0)
    return update_multiple_fields(state, clipboard=new_clipboard)


@eqx.filter_jit
def paste_from_clipboard(state: State, selection: SelectionArray) -> State:
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
    return update_working_grid(state, new_grid)


@eqx.filter_jit
def cut_to_clipboard(state: State, selection: SelectionArray) -> State:
    """Cut selected region to clipboard (copy + clear)."""
    # Copy to clipboard
    new_clipboard = jnp.where(selection, state.working_grid, 0)

    # Clear selected region
    new_grid = jnp.where(selection, 0, state.working_grid)

    # Update both working_grid and clipboard using PyTree utilities
    return update_multiple_fields(state, working_grid=new_grid, clipboard=new_clipboard)


# --- Grid Operations (31-33) ---


@eqx.filter_jit
def clear_grid(state: State, selection: SelectionArray) -> State:
    """Clear the entire grid or selected region."""
    has_selection = jnp.sum(selection) > 0

    def clear_selection():
        return jnp.where(selection, 0, state.working_grid)

    def clear_all():
        return jnp.zeros_like(state.working_grid)

    new_grid = jax.lax.cond(has_selection, clear_selection, clear_all)
    return update_working_grid(state, new_grid)


@eqx.filter_jit
def copy_input_grid(state: State, _selection: SelectionArray) -> State:
    """Copy original input_grid into working_grid, respecting current canvas shape."""
    new_working_grid = _copy_grid_to_target_shape(state.input_grid, state.working_grid)
    return update_working_grid(state, new_working_grid)


@eqx.filter_jit
def _get_bottom_right_from_selection(selection: SelectionArray) -> tuple[int, int]:
    """Find the bottom-rightmost selected cell coordinates.

    Args:
        selection: Boolean selection mask

    Returns:
        Tuple of (bottom_right_row, bottom_right_col) or (-1, -1) if no selection
    """
    # Check if selection is empty
    has_selection = jnp.sum(selection) > 0

    def get_coordinates():
        # Create coordinate grids
        height, width = selection.shape
        rows = jnp.arange(height)[:, None]
        cols = jnp.arange(width)[None, :]

        # Find all selected positions
        selected_rows = jnp.where(selection, rows, -1)
        selected_cols = jnp.where(selection, cols, -1)

        # Find maximum row among selected cells
        max_row = jnp.max(selected_rows)

        # Among cells in the maximum row, find the maximum column
        # Create mask for cells in the maximum row
        max_row_mask = selection & (rows == max_row)
        max_row_cols = jnp.where(max_row_mask, cols, -1)
        max_col = jnp.max(max_row_cols)

        return max_row, max_col

    def no_selection():
        return -1, -1

    return jax.lax.cond(has_selection, get_coordinates, no_selection)


@eqx.filter_jit
def resize_grid(state: State, selection: SelectionArray) -> State:
    """Resize grid active area using bottom-rightmost selected cell to define new dimensions.

    The resizing always originates from the top-left (0,0) corner:
    - When expanding: preserves existing content and fills new area with black (0)
    - When shrinking: preserves top-left content that fits in new dimensions
    """
    # Get the bottom-right coordinate that defines new dimensions
    bottom_right_row, bottom_right_col = _get_bottom_right_from_selection(selection)

    # Check if we have a valid selection
    has_valid_selection = (bottom_right_row >= 0) & (bottom_right_col >= 0)

    def resize_to_bottom_right():
        # Calculate new dimensions
        new_height = bottom_right_row + 1
        new_width = bottom_right_col + 1

        # Get current grid dimensions from mask
        from jaxarc.utils.grid_utils import get_actual_grid_shape_from_mask

        current_height, current_width = get_actual_grid_shape_from_mask(
            state.working_grid_mask
        )

        # Create new mask for the resized area
        max_height, max_width = state.working_grid.shape
        rows = jnp.arange(max_height)[:, None]
        cols = jnp.arange(max_width)[None, :]
        new_mask = (rows < new_height) & (cols < new_width)

        # Create new grid starting with padding (-1)
        new_grid = jnp.full_like(state.working_grid, -1)

        # Determine how much content to copy from original grid
        copy_height = jnp.minimum(current_height, new_height)
        copy_width = jnp.minimum(current_width, new_width)

        # Copy the overlapping content from original grid
        copy_mask = (rows < copy_height) & (cols < copy_width)
        new_grid = jnp.where(copy_mask, state.working_grid, new_grid)

        # Fill newly active areas (that weren't copied) with black (0)
        newly_active = new_mask & ~copy_mask
        new_grid = jnp.where(newly_active, 0, new_grid)
        return update_multiple_fields(
            state, working_grid=new_grid, working_grid_mask=new_mask
        )

    def no_resize():
        return state

    return jax.lax.cond(has_valid_selection, resize_to_bottom_right, no_resize)


# --- Submit Operation (34) ---


@eqx.filter_jit
def submit_solution(state: State, _selection: SelectionArray) -> State:
    """Submit current grid as solution."""

    return state


# --- Main Operation Execution ---


@eqx.filter_jit
def execute_grid_operation(state: State, operation: OperationId) -> State:
    """Execute grid operation based on operation ID."""
    # Validate grid operation before execution
    validated_state = validate_state_consistency(state)

    selection = validated_state.selected

    # This refactored structure is much more efficient for the JAX JIT compiler.
    # It uses conditional branching instead of a large switch over Python functions.

    def is_in_range(op, start, end):
        return (op >= start) & (op <= end)

    # We create a nested conditional tree. JAX can prune branches it knows won't be taken.
    def body_fn():
        # --- Category: Fill & Flood Fill ---
        is_fill = is_in_range(operation, 0, 9)
        is_flood_fill = is_in_range(operation, 10, 19)

        # --- Category: Movement & Transformation ---
        is_move = is_in_range(operation, 20, 23)
        is_rotate = is_in_range(operation, 24, 25)
        is_flip = is_in_range(operation, 26, 27)

        # --- Category: Editing & Special ---
        is_clipboard = is_in_range(operation, 28, 30)
        is_grid_op = is_in_range(operation, 31, 33)
        is_submit = operation == 34

        # Nested conditionals for efficient compilation
        s1 = jax.lax.cond(
            is_fill,
            lambda: fill_color(state, selection, operation),
            lambda: jax.lax.cond(
                is_flood_fill,
                lambda: flood_fill_color(state, selection, operation - 10),
                lambda: state,
            ),
        )
        s2 = jax.lax.cond(
            is_move,
            lambda: move_object(state, selection, operation - 20),
            lambda: jax.lax.cond(
                is_rotate,
                lambda: rotate_object(state, selection, operation - 24),
                lambda: jax.lax.cond(
                    is_flip,
                    lambda: flip_object(state, selection, operation - 26),
                    lambda: state,
                ),
            ),
        )
        s3 = jax.lax.cond(
            operation == 28,
            lambda: copy_to_clipboard(state, selection),
            lambda: jax.lax.cond(
                operation == 29,
                lambda: paste_from_clipboard(state, selection),
                lambda: jax.lax.cond(
                    operation == 30,
                    lambda: cut_to_clipboard(state, selection),
                    lambda: state,
                ),
            ),
        )
        s4 = jax.lax.cond(
            operation == 31,
            lambda: clear_grid(state, selection),
            lambda: jax.lax.cond(
                operation == 32,
                lambda: copy_input_grid(state, selection),
                lambda: jax.lax.cond(
                    operation == 33,
                    lambda: resize_grid(state, selection),
                    lambda: state,
                ),
            ),
        )
        s5 = jax.lax.cond(
            is_submit, lambda: submit_solution(state, selection), lambda: state
        )

        # Combine results. Only one of these will have changed from the original state.
        # This looks complex, but JAX understands that only one branch is taken.
        # We merge the changes from the single taken branch back into the state.
        # A simple way is to check which operation range was active.

        new_state = jax.lax.cond(is_fill | is_flood_fill, lambda: s1, lambda: state)
        new_state = jax.lax.cond(
            is_move | is_rotate | is_flip, lambda: s2, lambda: new_state
        )
        new_state = jax.lax.cond(is_clipboard, lambda: s3, lambda: new_state)
        new_state = jax.lax.cond(is_grid_op, lambda: s4, lambda: new_state)
        new_state = jax.lax.cond(is_submit, lambda: s5, lambda: new_state)

        return new_state

    # The final returned state from the one chosen path.
    new_state = body_fn()

    # Update similarity score if grid changed
    # Use target_grid from state (which handles train/test mode properly)
    similarity = compute_grid_similarity(
        new_state.working_grid,
        new_state.working_grid_mask,
        new_state.target_grid,
        new_state.target_grid_mask,
    )
    return update_similarity_score(new_state, similarity)
