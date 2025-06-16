"""Utility functions for ARC data parsing and preprocessing."""

from __future__ import annotations

import jax.numpy as jnp
from loguru import logger


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


def validate_arc_grid_data(grid_data: list[list[int]]) -> None:
    """Validate that grid data is in the correct ARC format.

    Args:
        grid_data: Grid as list of lists of integers

    Raises:
        ValueError: If grid format is invalid
    """
    if not grid_data:
        msg = "Grid data cannot be empty"
        raise ValueError(msg)

    if not isinstance(grid_data, list):
        msg = "Grid data must be a list"
        raise ValueError(msg)

    if not all(isinstance(row, list) for row in grid_data):
        msg = "Grid data must be a list of lists"
        raise ValueError(msg)

    # Check consistent row lengths (only if grid is not empty)
    if grid_data:
        row_length = len(grid_data[0])
        if not all(len(row) == row_length for row in grid_data):
            msg = "All rows in grid must have the same length"
            raise ValueError(msg)

    # Check that all cells are integers
    for i, row in enumerate(grid_data):
        for j, cell in enumerate(row):
            if not isinstance(cell, int):
                msg = f"Grid cell at ({i}, {j}) must be an integer, got {type(cell)}"
                raise ValueError(msg)

            # ARC colors are typically 0-9
            if not (0 <= cell <= 9):
                logger.warning(
                    f"Grid cell at ({i}, {j}) has value {cell}, "
                    f"which is outside typical ARC color range (0-9)"
                )


def convert_grid_to_jax(grid_data: list[list[int]]) -> jnp.ndarray:
    """Convert grid data from list format to JAX array.

    Args:
        grid_data: Grid as list of lists of integers

    Returns:
        JAX array of shape (height, width) with int32 dtype

    Raises:
        ValueError: If grid format is invalid
    """
    validate_arc_grid_data(grid_data)
    return jnp.array(grid_data, dtype=jnp.int32)


def log_parsing_stats(
    num_train_pairs: int,
    num_test_pairs: int,
    max_grid_dims: tuple[int, int],
    task_id: str | None = None,
) -> None:
    """Log statistics about parsed task data.

    Args:
        num_train_pairs: Number of training pairs in the task
        num_test_pairs: Number of test pairs in the task
        max_grid_dims: Maximum grid dimensions (height, width) in the task
        task_id: Optional task identifier for logging
    """
    task_info = f"Task {task_id}" if task_id else "Task"
    logger.debug(
        f"{task_info}: {num_train_pairs} train pairs, {num_test_pairs} test pairs, "
        f"max grid size: {max_grid_dims[0]}x{max_grid_dims[1]}"
    )
