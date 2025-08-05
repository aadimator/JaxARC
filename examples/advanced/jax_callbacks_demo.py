#!/usr/bin/env python3
"""
Demonstration of JAX-compatible callback system for visualization.

This example shows how to use the enhanced visualization system with JAX transformations
while maintaining performance and proper error handling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.utils.visualization import (
    get_callback_performance_stats,
    jax_debug_callback,
    jax_log_episode_summary,
    jax_log_grid,
    print_callback_performance_report,
    reset_callback_performance_stats,
)


def demo_basic_jax_callbacks():
    """Demonstrate basic JAX callback functionality."""
    logger.info("=== Basic JAX Callbacks Demo ===")

    # Create sample grid data
    grid_data = jnp.array([[1, 2, 0], [3, 4, 1], [0, 2, 3]])
    mask_data = jnp.array(
        [[True, True, False], [True, True, True], [False, True, True]]
    )

    @jax.jit
    def process_grid(grid, mask):
        """Process grid with JAX transformations and logging."""
        # Log the input grid
        jax_log_grid(grid, mask, "Input Grid")

        # Perform some computation
        processed = jnp.where(mask, grid + 1, 0)

        # Log the processed grid
        jax_log_grid(processed, mask, "Processed Grid")

        return processed

    # Process the grid
    result = process_grid(grid_data, mask_data)
    logger.info(f"Processing result shape: {result.shape}")

    return result


def demo_performance_monitoring():
    """Demonstrate performance monitoring of callbacks."""
    logger.info("=== Performance Monitoring Demo ===")

    # Reset stats for clean demo
    reset_callback_performance_stats()

    def slow_callback(x):
        """Simulate a slow callback."""
        import time

        time.sleep(0.01)  # 10ms delay
        logger.info(f"Slow callback processed: {x}")

    def fast_callback(x):
        """Simulate a fast callback."""
        logger.info(f"Fast callback processed: {x}")

    @jax.jit
    def test_function(x):
        """Function that uses both fast and slow callbacks."""
        jax_debug_callback(slow_callback, x, callback_name="slow_callback")
        jax_debug_callback(fast_callback, x * 2, callback_name="fast_callback")
        return x + 1

    # Run multiple times to collect stats
    for i in range(5):
        test_function(jnp.array(float(i)))

    # Print performance report
    print_callback_performance_report()

    # Get specific stats
    stats = get_callback_performance_stats()
    if "slow_callback" in stats:
        slow_stats = stats["slow_callback"]
        logger.info(f"Slow callback average time: {slow_stats['avg_time_ms']:.2f}ms")

    return stats


def demo_batch_processing():
    """Demonstrate callbacks with batch processing using vmap."""
    logger.info("=== Batch Processing Demo ===")

    def process_single_grid(grid):
        """Process a single grid with logging."""
        # Log each grid in the batch
        jax_log_grid(grid, title="Batch Item")

        # Simple processing
        return jnp.sum(grid)

    # Create batch of grids
    batch_grids = jnp.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 0], [1, 2]],
        ]
    )

    # Process batch with vmap
    batch_process = jax.vmap(process_single_grid)
    results = batch_process(batch_grids)

    logger.info(f"Batch processing results: {results}")
    return results


def demo_episode_logging():
    """Demonstrate episode summary logging."""
    logger.info("=== Episode Logging Demo ===")

    @jax.jit
    def simulate_episode(episode_num, steps, base_reward):
        """Simulate an episode with logging."""
        total_reward = base_reward * steps
        final_similarity = jnp.tanh(total_reward / 10.0)  # Normalize to [0,1]
        success = final_similarity > 0.8

        # Log episode summary
        jax_log_episode_summary(
            episode_num, steps, total_reward, final_similarity, success
        )

        return total_reward, final_similarity, success

    # Simulate multiple episodes
    episodes_data = []
    for ep in range(3):
        steps = 10 + ep * 5
        base_reward = 1.0 + ep * 0.5
        total_reward, similarity, success = simulate_episode(ep, steps, base_reward)
        episodes_data.append((total_reward, similarity, success))

    logger.info(f"Simulated {len(episodes_data)} episodes")
    return episodes_data


def demo_error_handling():
    """Demonstrate error handling in callbacks."""
    logger.info("=== Error Handling Demo ===")

    def error_callback(x):
        """Callback that raises an error."""
        if x > 5:
            error_msg = f"Value too large: {x}"
            raise ValueError(error_msg)
        logger.info(f"Error callback processed: {x}")

    @jax.jit
    def test_with_errors(x):
        """Function that may trigger callback errors."""
        jax_debug_callback(error_callback, x, callback_name="error_callback")
        return x * 2

    # Test with values that will and won't cause errors
    test_values = [3.0, 7.0, 2.0, 10.0]

    results = []
    for val in test_values:
        try:
            result = test_with_errors(jnp.array(val))
            results.append(result)
            logger.info(f"Processed {val} -> {result}")
        except Exception as e:
            logger.error(f"Error processing {val}: {e}")
            results.append(None)

    logger.info(
        "Error handling demo completed - JAX execution continued despite callback errors"
    )
    return results


def demo_memory_optimization():
    """Memory optimization features removed - use standard Python memory management."""
    logger.info("=== Memory Optimization Demo (REMOVED) ===")
    logger.info("Memory management functionality has been removed from the logging system.")
    logger.info("Use standard Python memory profiling tools like memory_profiler or tracemalloc instead.")
    
    # Create some large arrays to demonstrate basic logging
    large_arrays = []
    for i in range(5):
        arr = jnp.ones((100, 100)) * i
        large_arrays.append(arr)

        # Log the array (this will use basic logging)
        jax_log_grid(arr, title=f"Large Array {i}")

    # Memory usage monitoring removed
    memory_manager.check_memory_usage()

    # Get memory report
    report = memory_manager.get_memory_report()
    logger.info(f"Memory report: {report}")

    # Force cleanup
    cleanup_stats = memory_manager.cleanup_memory()
    logger.info(f"Cleanup stats: {cleanup_stats}")

    return report


def main():
    """Run all demonstrations."""
    logger.info("Starting JAX Callbacks Demonstration")
    logger.info("=" * 50)

    try:
        # Run all demos
        demo_basic_jax_callbacks()
        demo_performance_monitoring()
        demo_batch_processing()
        demo_episode_logging()
        demo_error_handling()
        demo_memory_optimization()

        logger.info("=" * 50)
        logger.info("All demonstrations completed successfully!")

        # Final performance report
        logger.info("\n=== Final Performance Report ===")
        print_callback_performance_report()

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
