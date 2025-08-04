"""Simple JAX-compatible callbacks for visualization.

This module provides basic JAX debug callback wrappers for visualization
functions. Keeps things simple for research use.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
from loguru import logger

from jaxarc.types import Grid
from jaxarc.utils.jax_types import GridArray, MaskArray
from jaxarc.utils.serialization_utils import serialize_jax_array, serialize_arc_state, serialize_action


def safe_callback_wrapper(callback_func: Callable[..., Any]) -> Callable[..., None]:
    """Wrap a callback function with basic error handling.

    Args:
        callback_func: Function to wrap

    Returns:
        Wrapped callback function that's safe for JAX debug callbacks
    """
    @functools.wraps(callback_func)
    def wrapped_callback(*args: Any, **kwargs: Any) -> None:
        try:
            callback_func(*args, **kwargs)
        except Exception as e:
            # Log error but don't re-raise to avoid breaking JAX
            logger.error(f"Error in callback: {e}")

    return wrapped_callback


def jax_debug_callback(
    callback_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Create a JAX debug callback with basic error handling.

    Args:
        callback_func: Function to call
        *args: Arguments to pass to the callback
        **kwargs: Keyword arguments to pass to the callback
    """
    wrapped_func = safe_callback_wrapper(callback_func)
    jax.debug.callback(wrapped_func, *args, **kwargs)


def create_grid_from_arrays(data: GridArray, mask: MaskArray | None = None) -> Grid:
    """Create a Grid object from JAX arrays with proper serialization.

    Args:
        data: Grid data array
        mask: Optional mask array

    Returns:
        Grid object with serialized NumPy arrays
    """
    serialized_data = serialize_jax_array(data)
    serialized_mask = serialize_jax_array(mask) if mask is not None else None

    return Grid(data=serialized_data, mask=serialized_mask)


def log_grid_callback(
    grid_data: GridArray,
    mask: MaskArray | None = None,
    title: str = "Grid",
    show_coordinates: bool = False,
    show_numbers: bool = False,
) -> None:
    """JAX callback for logging grid to console.

    Args:
        grid_data: Grid data array
        mask: Optional mask array
        title: Title for the grid
        show_coordinates: Whether to show coordinates
        show_numbers: Whether to show numbers instead of blocks
    """
    from jaxarc.utils.visualization.rich_display import log_grid_to_console

    grid = create_grid_from_arrays(grid_data, mask)
    log_grid_to_console(
        grid, title=title, show_coordinates=show_coordinates, show_numbers=show_numbers
    )


def jax_log_grid(
    grid_data: GridArray,
    mask: MaskArray | None = None,
    title: str = "Grid",
    **kwargs: Any,
) -> None:
    """Convenience function for logging grids from JAX functions.

    Args:
        grid_data: Grid data array
        mask: Optional mask array
        title: Title for the grid
        **kwargs: Additional arguments for log_grid_callback
    """
    jax_debug_callback(log_grid_callback, grid_data, mask, title, **kwargs)
