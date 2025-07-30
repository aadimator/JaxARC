"""Utility functions for JaxARC visualization.

This module contains helper functions used throughout the visualization system,
including data extraction, grid processing, and common utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .constants import ARC_COLOR_PALETTE

if TYPE_CHECKING:
    from jaxarc.types import Grid
    from jaxarc.utils.jax_types import GridArray


def _extract_grid_data(
    grid_input: GridArray | np.ndarray | Grid,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract numpy array and mask from various grid input types.

    Args:
        grid_input: Grid data in various formats

    Returns:
        Tuple of (grid_data as numpy array, mask as numpy array or None)

    Raises:
        ValueError: If input type is not supported
    """
    # Check for Grid type by duck typing (more robust than isinstance)
    if hasattr(grid_input, 'data') and hasattr(grid_input, 'mask'):
        return np.asarray(grid_input.data), np.asarray(grid_input.mask)
    if isinstance(grid_input, (jnp.ndarray, np.ndarray)):
        return np.asarray(grid_input), None

    msg = f"Unsupported grid input type: {type(grid_input)}"
    raise ValueError(msg)


def _extract_valid_region(
    grid: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Extract the valid (non-padded) region from a grid.

    Args:
        grid: The grid array
        mask: Optional boolean mask indicating valid cells

    Returns:
        Tuple of (valid_grid, (start_row, start_col), (height, width))
    """
    if mask is None:
        # Assume all cells are valid if no mask provided
        return grid, (0, 0), (grid.shape[0], grid.shape[1])

    if not np.any(mask):
        # No valid cells
        return np.array([[]], dtype=grid.dtype), (0, 0), (0, 0)

    # Find bounding box of valid region
    valid_rows = np.where(np.any(mask, axis=1))[0]
    valid_cols = np.where(np.any(mask, axis=0))[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return np.array([[]], dtype=grid.dtype), (0, 0), (0, 0)

    start_row, end_row = valid_rows[0], valid_rows[-1] + 1
    start_col, end_col = valid_cols[0], valid_cols[-1] + 1

    valid_grid = grid[start_row:end_row, start_col:end_col]

    return (
        valid_grid,
        (start_row, start_col),
        (end_row - start_row, end_col - start_col),
    )


def get_color_name(color_id: int) -> str:
    """Get human-readable color name from color ID.

    Args:
        color_id: Integer color ID

    Returns:
        Human-readable color name
    """
    color_names = {
        0: "Black (0)",
        1: "Blue (1)",
        2: "Red (2)",
        3: "Green (3)",
        4: "Yellow (4)",
        5: "Grey (5)",
        6: "Pink (6)",
        7: "Orange (7)",
        8: "Light Blue (8)",
        9: "Brown (9)",
    }

    return color_names.get(color_id, f"Color {color_id}")


def detect_changed_cells(
    before_grid: Grid,
    after_grid: Grid,
) -> jnp.ndarray:
    """Detect which cells changed between before and after grids.

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action

    Returns:
        Boolean mask indicating which cells changed
    """
    before_data = np.asarray(before_grid.data)
    after_data = np.asarray(after_grid.data)

    # Handle different shapes by padding to match
    max_height = max(before_data.shape[0], after_data.shape[0])
    max_width = max(before_data.shape[1], after_data.shape[1])

    # Pad both grids to same size
    before_padded = np.zeros((max_height, max_width), dtype=before_data.dtype)
    after_padded = np.zeros((max_height, max_width), dtype=after_data.dtype)

    before_padded[: before_data.shape[0], : before_data.shape[1]] = before_data
    after_padded[: after_data.shape[0], : after_data.shape[1]] = after_data

    # Find changed cells
    changed = before_padded != after_padded

    return jnp.array(changed)


def infer_fill_color_from_grids(
    before_grid: Grid, after_grid: Grid, selection_mask: np.ndarray
) -> int:
    """Infer what color was used to fill selected cells by comparing grids.

    Args:
        before_grid: Grid state before the action
        after_grid: Grid state after the action
        selection_mask: Boolean mask of selected cells

    Returns:
        Color ID that was used for filling, or -1 if can't determine
    """
    try:
        before_data = np.asarray(before_grid.data)
        after_data = np.asarray(after_grid.data)

        # Find cells that were selected and changed
        for i in range(min(before_data.shape[0], after_data.shape[0])):
            for j in range(min(before_data.shape[1], after_data.shape[1])):
                if (
                    i < selection_mask.shape[0]
                    and j < selection_mask.shape[1]
                    and selection_mask[i, j]
                    and before_data[i, j] != after_data[i, j]
                ):
                    # This cell was selected and changed, return the new color
                    return int(after_data[i, j])

        return -1  # Couldn't determine
    except Exception:
        return -1


def _clear_output_directory(output_dir: str) -> None:
    """Clear output directory for new episode."""
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)