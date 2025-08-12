"""JAX-compatible callbacks for ExperimentLogger integration.

This module provides JAX debug callback wrappers that integrate with the new
ExperimentLogger architecture. It maintains JAX compatibility while providing
a clean interface for logging step data and episode summaries through the
centralized logging system.

Key Features:
- JAX-compatible callbacks using jax.debug.callback
- Integration with ExperimentLogger handler architecture
- Safe error handling that doesn't break JAX transformations
- Support for step logging and episode summary logging
- Proper serialization of JAX data structures
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

import jax
from loguru import logger

from jaxarc.types import Grid
from jaxarc.utils.jax_types import GridArray, MaskArray
from jaxarc.utils.serialization_utils import (
    serialize_jax_array,
)


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


# =============================================================================
# ExperimentLogger Integration Callbacks
# =============================================================================


def jax_save_step_visualization(
    step_data: Dict[str, Any], logger_instance: Optional[Any] = None
) -> None:
    """JAX-compatible callback for logging step data through ExperimentLogger.

    This function is designed to be called from within JAX transformations
    via jax.debug.callback. It safely passes step data to the ExperimentLogger
    while maintaining JAX compatibility.

    Args:
        step_data: Dictionary containing step information with keys:
            - step_num: Step number within episode
            - before_state: State before action (serialized)
            - after_state: State after action (serialized)
            - action: Action taken (serialized)
            - reward: Reward received
            - info: Additional information including metrics
        logger_instance: Optional ExperimentLogger instance. If None, will try
                        to get logger from environment context.

    Note:
        This function is called from within JAX transformations and must be
        careful about what operations it performs. All complex operations
        should be delegated to the ExperimentLogger handlers.
    """
    if logger_instance is not None:
        try:
            logger_instance.log_step(step_data)
        except Exception as e:
            # Log error but don't re-raise to avoid breaking JAX
            logger.warning(f"Failed to log step data via ExperimentLogger: {e}")
    else:
        # Fallback: try to get logger from global context or environment
        # This is a temporary solution - ideally logger should be passed explicitly
        logger.debug("No logger instance provided to jax_save_step_visualization")


def jax_save_episode_summary(
    summary_data: Dict[str, Any], logger_instance: Optional[Any] = None
) -> None:
    """JAX-compatible callback for logging episode summary through ExperimentLogger.

    This function is designed to be called from within JAX transformations
    via jax.debug.callback. It safely passes episode summary data to the
    ExperimentLogger while maintaining JAX compatibility.

    Args:
        summary_data: Dictionary containing episode summary with keys:
            - episode_num: Episode number
            - total_steps: Total steps in episode
            - total_reward: Total reward accumulated
            - final_similarity: Final similarity score
            - success: Whether episode was successful
            - task_id: Task identifier
        logger_instance: Optional ExperimentLogger instance. If None, will try
                        to get logger from environment context.

    Note:
        This function is called from within JAX transformations and must be
        careful about what operations it performs. All complex operations
        should be delegated to the ExperimentLogger handlers.
    """
    if logger_instance is not None:
        try:
            logger_instance.log_episode_summary(summary_data)
        except Exception as e:
            # Log error but don't re-raise to avoid breaking JAX
            logger.warning(f"Failed to log episode summary via ExperimentLogger: {e}")
    else:
        # Fallback: try to get logger from global context or environment
        logger.debug("No logger instance provided to jax_save_episode_summary")


def create_step_logging_callback(
    logger_instance: Any,
) -> Callable[[Dict[str, Any]], None]:
    """Create a step logging callback bound to a specific ExperimentLogger instance.

    This factory function creates a callback that's bound to a specific logger
    instance, avoiding the need to pass the logger through JAX transformations.

    Args:
        logger_instance: ExperimentLogger instance to use for logging

    Returns:
        Callback function that can be used with jax.debug.callback

    Example:
        ```python
        from jaxarc.utils.logging import ExperimentLogger
        from jaxarc.utils.visualization.jax_callbacks import create_step_logging_callback

        # Create logger and callback
        logger = ExperimentLogger(config)
        step_callback = create_step_logging_callback(logger)


        # Use in JAX function
        def jax_step_function(state, action):
            # ... step logic ...
            step_data = {...}
            jax.debug.callback(step_callback, step_data)
            return new_state, reward, done, info
        ```
    """

    def step_callback(step_data: Dict[str, Any]) -> None:
        """Bound step logging callback."""
        jax_save_step_visualization(step_data, logger_instance)

    return step_callback


def create_episode_logging_callback(
    logger_instance: Any,
) -> Callable[[Dict[str, Any]], None]:
    """Create an episode summary logging callback bound to a specific ExperimentLogger instance.

    This factory function creates a callback that's bound to a specific logger
    instance, avoiding the need to pass the logger through JAX transformations.

    Args:
        logger_instance: ExperimentLogger instance to use for logging

    Returns:
        Callback function that can be used with jax.debug.callback

    Example:
        ```python
        from jaxarc.utils.logging import ExperimentLogger
        from jaxarc.utils.visualization.jax_callbacks import create_episode_logging_callback

        # Create logger and callback
        logger = ExperimentLogger(config)
        episode_callback = create_episode_logging_callback(logger)


        # Use in JAX function
        def jax_episode_end(state, summary):
            # ... episode end logic ...
            summary_data = {...}
            jax.debug.callback(episode_callback, summary_data)
            return final_state
        ```
    """

    def episode_callback(summary_data: Dict[str, Any]) -> None:
        """Bound episode summary logging callback."""
        jax_save_episode_summary(summary_data, logger_instance)

    return episode_callback


# =============================================================================
# JAX Transformation Compatibility Utilities
# =============================================================================


def ensure_jax_callback_compatibility(
    callback_func: Callable[..., Any],
) -> Callable[..., None]:
    """Ensure a callback function is compatible with JAX transformations.

    This function wraps callbacks to ensure they work correctly with JAX
    transformations like jit, vmap, and pmap. It handles serialization
    of JAX arrays and provides error isolation.

    Args:
        callback_func: Function to make JAX-compatible

    Returns:
        JAX-compatible callback function

    Note:
        This is an enhanced version of safe_callback_wrapper that specifically
        handles JAX transformation compatibility.
    """

    @functools.wraps(callback_func)
    def jax_compatible_callback(*args: Any, **kwargs: Any) -> None:
        try:
            # Ensure all JAX arrays are properly serialized before callback
            serialized_args = []
            for arg in args:
                if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                    # This is likely a JAX array - serialize it
                    serialized_args.append(serialize_jax_array(arg))
                else:
                    serialized_args.append(arg)

            # Call the original function with serialized data
            callback_func(*serialized_args, **kwargs)

        except Exception as e:
            # Log error but don't re-raise to avoid breaking JAX transformations
            logger.error(f"Error in JAX callback: {e}")

    return jax_compatible_callback


def validate_jax_callback_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and prepare data for JAX callback usage.

    This function ensures that data passed to JAX callbacks is properly
    serialized and doesn't contain any JAX-incompatible objects.

    Args:
        data: Dictionary of data to validate

    Returns:
        Validated and serialized data dictionary

    Raises:
        ValueError: If data contains unsupported types
    """
    validated_data = {}

    for key, value in data.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            # JAX array - serialize it
            validated_data[key] = serialize_jax_array(value)
        elif isinstance(value, dict):
            # Nested dictionary - recursively validate
            validated_data[key] = validate_jax_callback_data(value)
        elif isinstance(value, (list, tuple)):
            # Sequence - validate each element
            validated_data[key] = [
                serialize_jax_array(item)
                if hasattr(item, "shape") and hasattr(item, "dtype")
                else item
                for item in value
            ]
        else:
            # Basic type - should be safe
            validated_data[key] = value

    return validated_data
