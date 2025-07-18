"""JAX-compatible callback system for visualization.

This module provides JAX debug callback wrappers that ensure visualization functions
work correctly with JAX transformations while maintaining performance and error handling.

Key Features:
- JAX debug callback wrappers for all visualization functions
- Proper array serialization for JAX arrays
- Error handling that doesn't break JAX transformations
- Performance monitoring and adaptive logging
- Memory-efficient callback management
"""

from __future__ import annotations

import functools
import time
import traceback
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.types import Grid
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import GridArray, MaskArray, PRNGKey

# Type variables for generic callback wrapping
F = TypeVar("F", bound=Callable[..., Any])


class JAXCallbackError(Exception):
    """Exception raised when JAX callback encounters an error."""

    pass


class CallbackPerformanceMonitor:
    """Monitor performance impact of visualization callbacks."""

    def __init__(self) -> None:
        self.callback_times: dict[str, list[float]] = {}
        self.error_counts: dict[str, int] = {}
        self.total_calls: dict[str, int] = {}

    def record_call(self, callback_name: str, duration: float, had_error: bool = False) -> None:
        """Record a callback execution."""
        if callback_name not in self.callback_times:
            self.callback_times[callback_name] = []
            self.error_counts[callback_name] = 0
            self.total_calls[callback_name] = 0

        self.callback_times[callback_name].append(duration)
        self.total_calls[callback_name] += 1
        if had_error:
            self.error_counts[callback_name] += 1

    def get_stats(self, callback_name: str) -> dict[str, Any]:
        """Get performance statistics for a callback."""
        if callback_name not in self.callback_times:
            return {}

        times = self.callback_times[callback_name]
        return {
            "total_calls": self.total_calls[callback_name],
            "error_count": self.error_counts[callback_name],
            "avg_time_ms": np.mean(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "total_time_ms": np.sum(times) * 1000,
        }

    def should_reduce_logging(self, callback_name: str, max_avg_time_ms: float = 10.0) -> bool:
        """Determine if logging should be reduced due to performance impact."""
        stats = self.get_stats(callback_name)
        if not stats:
            return False

        return stats.get("avg_time_ms", 0) > max_avg_time_ms


# Global performance monitor
_performance_monitor = CallbackPerformanceMonitor()


def serialize_jax_array(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
    """Safely serialize JAX array to numpy array.

    Args:
        arr: JAX or numpy array to serialize

    Returns:
        Numpy array copy of the input
    """
    try:
        if isinstance(arr, jnp.ndarray):
            return np.asarray(arr)
        elif isinstance(arr, np.ndarray):
            return arr.copy()
        else:
            return np.asarray(arr)
    except Exception as e:
        logger.warning(f"Failed to serialize array: {e}")
        return np.array([])


def serialize_arc_state(state: ArcEnvState) -> dict[str, Any]:
    """Serialize ArcEnvState for safe callback usage.

    Args:
        state: Environment state to serialize

    Returns:
        Dictionary with serialized state data
    """
    try:
        return {
            "working_grid": serialize_jax_array(state.working_grid),
            "working_grid_mask": serialize_jax_array(state.working_grid_mask),
            "target_grid": serialize_jax_array(state.target_grid),
            "target_grid_mask": serialize_jax_array(state.target_grid_mask),
            "step_count": int(state.step_count),
            "episode_index": int(state.episode_index),
            "task_index": int(state.task_index),
            "done": bool(state.done),
            "similarity": float(state.similarity),
        }
    except Exception as e:
        logger.warning(f"Failed to serialize ArcEnvState: {e}")
        return {}


def serialize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Serialize action dictionary for safe callback usage.

    Args:
        action: Action dictionary to serialize

    Returns:
        Dictionary with serialized action data
    """
    try:
        serialized = {}
        for key, value in action.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                serialized[key] = serialize_jax_array(value)
            elif isinstance(value, (int, float, bool, str)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    except Exception as e:
        logger.warning(f"Failed to serialize action: {e}")
        return {}


def safe_callback_wrapper(
    callback_func: Callable[..., Any],
    callback_name: str,
    enable_performance_monitoring: bool = True,
) -> Callable[..., None]:
    """Wrap a callback function with error handling and performance monitoring.

    Args:
        callback_func: Function to wrap
        callback_name: Name for monitoring and logging
        enable_performance_monitoring: Whether to monitor performance

    Returns:
        Wrapped callback function that's safe for JAX debug callbacks
    """

    @functools.wraps(callback_func)
    def wrapped_callback(*args: Any, **kwargs: Any) -> None:
        start_time = time.time()
        had_error = False

        try:
            callback_func(*args, **kwargs)
        except Exception as e:
            had_error = True
            # Log error but don't re-raise to avoid breaking JAX
            logger.error(f"Error in callback '{callback_name}': {e}")
            logger.debug(f"Callback traceback:\n{traceback.format_exc()}")

        if enable_performance_monitoring:
            duration = time.time() - start_time
            _performance_monitor.record_call(callback_name, duration, had_error)

    return wrapped_callback


def jax_debug_callback(
    callback_func: Callable[..., Any],
    *args: Any,
    callback_name: str | None = None,
    enable_performance_monitoring: bool = True,
    **kwargs: Any,
) -> None:
    """Create a JAX debug callback with proper error handling.

    Args:
        callback_func: Function to call
        *args: Arguments to pass to the callback
        callback_name: Name for monitoring (defaults to function name)
        enable_performance_monitoring: Whether to monitor performance
        **kwargs: Keyword arguments to pass to the callback
    """
    if callback_name is None:
        callback_name = getattr(callback_func, "__name__", "unknown_callback")

    wrapped_func = safe_callback_wrapper(
        callback_func, callback_name, enable_performance_monitoring
    )

    jax.debug.callback(wrapped_func, *args, **kwargs)


def create_grid_from_arrays(
    data: GridArray, mask: MaskArray | None = None
) -> Grid:
    """Create a Grid object from JAX arrays with proper serialization.

    Args:
        data: Grid data array
        mask: Optional mask array

    Returns:
        Grid object with serialized numpy arrays
    """
    serialized_data = serialize_jax_array(data)
    serialized_mask = serialize_jax_array(mask) if mask is not None else None

    return Grid(data=serialized_data, mask=serialized_mask)


# Specialized callback functions for common visualization tasks


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
    from jaxarc.utils.visualization.core import log_grid_to_console

    grid = create_grid_from_arrays(grid_data, mask)
    log_grid_to_console(
        grid, title=title, show_coordinates=show_coordinates, show_numbers=show_numbers
    )


def save_step_visualization_callback(
    before_state: ArcEnvState,
    action: dict[str, Any],
    after_state: ArcEnvState,
    output_dir: str,
    step_label: str = "Step",
) -> None:
    """JAX callback for saving step visualization.

    Args:
        before_state: State before action
        action: Action taken
        after_state: State after action
        output_dir: Output directory
        step_label: Label for the step
    """
    from pathlib import Path

    from jaxarc.utils.visualization.core import draw_rl_step_svg

    try:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Serialize state and action data
        before_data = serialize_arc_state(before_state)
        after_data = serialize_arc_state(after_state)
        action_data = serialize_action(action)

        # Create Grid objects
        before_grid = Grid(
            data=before_data["working_grid"],
            mask=before_data["working_grid_mask"],
        )
        after_grid = Grid(
            data=after_data["working_grid"],
            mask=after_data["working_grid_mask"],
        )

        # Extract action components
        selection_mask = action_data.get("selection", np.array([]))
        operation_id = action_data.get("operation", 0)
        step_number = before_data.get("step_count", 0)

        # Generate visualization
        svg_content = draw_rl_step_svg(
            before_grid=before_grid,
            after_grid=after_grid,
            selection_mask=selection_mask,
            operation_id=operation_id,
            step_number=step_number,
            label=step_label,
        )

        # Save to file
        filename = f"step_{step_number:03d}.svg"
        filepath = Path(output_dir) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_content)

    except Exception as e:
        logger.error(f"Failed to save step visualization: {e}")


def log_episode_summary_callback(
    episode_num: int,
    total_steps: int,
    total_reward: float,
    final_similarity: float,
    success: bool,
) -> None:
    """JAX callback for logging episode summary.

    Args:
        episode_num: Episode number
        total_steps: Total steps taken
        total_reward: Total reward earned
        final_similarity: Final similarity score
        success: Whether episode was successful
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"Episode {episode_num} {status}: "
        f"steps={total_steps}, reward={total_reward:.3f}, "
        f"similarity={final_similarity:.3f}"
    )


def adaptive_visualization_callback(
    callback_func: Callable[..., Any],
    *args: Any,
    callback_name: str | None = None,
    max_avg_time_ms: float = 10.0,
    **kwargs: Any,
) -> None:
    """Adaptive callback that reduces frequency based on performance impact.

    Args:
        callback_func: Function to call
        *args: Arguments to pass to the callback
        callback_name: Name for monitoring
        max_avg_time_ms: Maximum average time before reducing frequency
        **kwargs: Keyword arguments to pass to the callback
    """
    if callback_name is None:
        callback_name = getattr(callback_func, "__name__", "unknown_callback")

    # Check if we should reduce logging frequency
    if _performance_monitor.should_reduce_logging(callback_name, max_avg_time_ms):
        # Only log every 10th call if performance is poor
        stats = _performance_monitor.get_stats(callback_name)
        if stats["total_calls"] % 10 != 0:
            return

    jax_debug_callback(
        callback_func, *args, callback_name=callback_name, **kwargs
    )


# Convenience functions for common patterns


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
    jax_debug_callback(
        log_grid_callback,
        grid_data,
        mask,
        title,
        callback_name=f"log_grid_{title}",
        **kwargs,
    )


def jax_save_step_visualization(
    before_state: ArcEnvState,
    action: dict[str, Any],
    after_state: ArcEnvState,
    output_dir: str,
    **kwargs: Any,
) -> None:
    """Convenience function for saving step visualizations from JAX functions.

    Args:
        before_state: State before action
        action: Action taken
        after_state: State after action
        output_dir: Output directory
        **kwargs: Additional arguments for save_step_visualization_callback
    """
    adaptive_visualization_callback(
        save_step_visualization_callback,
        before_state,
        action,
        after_state,
        output_dir,
        callback_name="save_step_visualization",
        **kwargs,
    )


def jax_log_episode_summary(
    episode_num: int,
    total_steps: int,
    total_reward: float,
    final_similarity: float,
    success: bool,
) -> None:
    """Convenience function for logging episode summaries from JAX functions.

    Args:
        episode_num: Episode number
        total_steps: Total steps taken
        total_reward: Total reward earned
        final_similarity: Final similarity score
        success: Whether episode was successful
    """
    jax_debug_callback(
        log_episode_summary_callback,
        episode_num,
        total_steps,
        total_reward,
        final_similarity,
        success,
        callback_name="log_episode_summary",
    )


# Performance monitoring utilities


def get_callback_performance_stats() -> dict[str, dict[str, Any]]:
    """Get performance statistics for all callbacks.

    Returns:
        Dictionary mapping callback names to their performance stats
    """
    stats = {}
    for callback_name in _performance_monitor.callback_times:
        stats[callback_name] = _performance_monitor.get_stats(callback_name)
    return stats


def reset_callback_performance_stats() -> None:
    """Reset all callback performance statistics."""
    global _performance_monitor
    _performance_monitor = CallbackPerformanceMonitor()


def print_callback_performance_report() -> None:
    """Print a performance report for all callbacks."""
    stats = get_callback_performance_stats()
    if not stats:
        logger.info("No callback performance data available")
        return

    logger.info("Callback Performance Report:")
    logger.info("-" * 50)

    for callback_name, callback_stats in stats.items():
        logger.info(f"Callback: {callback_name}")
        logger.info(f"  Total calls: {callback_stats['total_calls']}")
        logger.info(f"  Errors: {callback_stats['error_count']}")
        logger.info(f"  Avg time: {callback_stats['avg_time_ms']:.2f}ms")
        logger.info(f"  Max time: {callback_stats['max_time_ms']:.2f}ms")
        logger.info(f"  Total time: {callback_stats['total_time_ms']:.2f}ms")
        logger.info("")