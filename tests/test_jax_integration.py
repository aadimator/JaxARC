"""Integration tests for JAX callback system with existing visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxarc.utils.visualization import (
    get_callback_performance_stats,
    jax_log_grid,
    reset_callback_performance_stats,
)


class TestJAXVisualizationIntegration:
    """Test integration between JAX callbacks and visualization system."""

    def test_jax_grid_logging_integration(self):
        """Test JAX grid logging with actual visualization functions."""
        reset_callback_performance_stats()

        @jax.jit
        def process_and_log_grid(grid_data, mask_data):
            """Process grid and log it using JAX callbacks."""
            # Log input
            jax_log_grid(grid_data, mask_data, "Input Grid")

            # Process the grid
            processed = jnp.where(mask_data, grid_data + 1, 0)

            # Log output
            jax_log_grid(processed, mask_data, "Processed Grid")

            return processed

        # Create test data
        grid_data = jnp.array([[1, 2, 0], [3, 4, 1], [0, 2, 3]])
        mask_data = jnp.array(
            [[True, True, False], [True, True, True], [False, True, True]]
        )

        # Process with JAX
        result = process_and_log_grid(grid_data, mask_data)

        # Verify result
        expected = jnp.where(mask_data, grid_data + 1, 0)
        np.testing.assert_array_equal(result, expected)

        # Check that callbacks were recorded
        stats = get_callback_performance_stats()
        assert len(stats) > 0
        assert any("log_grid" in name for name in stats.keys())

    def test_jax_step_visualization_integration(self):
        """Test JAX step visualization with mock environment states."""
        with tempfile.TemporaryDirectory() as temp_dir:

            @jax.jit
            def simulate_step_with_visualization(grid_before, grid_after, step_num):
                """Simulate a step and save visualization."""
                # Create mock states (simplified for JAX compatibility)
                action_data = {
                    "selection": jnp.array([[True, False], [False, True]]),
                    "operation": jnp.array(5),
                }

                # Mock the save function to avoid complex state serialization in JIT
                def mock_save_callback(
                    before_grid, after_grid, action, output_dir, step_label
                ):
                    # This would normally save the visualization
                    pass

                jax.debug.callback(
                    mock_save_callback,
                    grid_before,
                    grid_after,
                    action_data,
                    temp_dir,
                    f"Step {step_num}",
                )

                return grid_after

            # Test data
            before_grid = jnp.array([[1, 2], [3, 4]])
            after_grid = jnp.array([[2, 1], [4, 3]])

            # Run simulation
            result = simulate_step_with_visualization(before_grid, after_grid, 1)
            np.testing.assert_array_equal(result, after_grid)

    def test_batch_processing_with_callbacks(self):
        """Test JAX callbacks work correctly with batch processing."""
        reset_callback_performance_stats()

        @jax.jit
        def process_batch_with_logging(batch_grids):
            """Process a batch of grids with logging."""

            def process_single(grid):
                # Log each grid (simplified for batch processing)
                jax_log_grid(grid, title="Batch Item")
                return jnp.sum(grid)

            # Use vmap for batch processing
            return jax.vmap(process_single)(batch_grids)

        # Create batch data
        batch_grids = jnp.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 0], [1, 2]],
            ]
        )

        # Process batch
        results = process_batch_with_logging(batch_grids)

        # Verify results
        expected = jnp.array([10, 26, 12])
        np.testing.assert_array_equal(results, expected)

        # Check callback statistics
        stats = get_callback_performance_stats()
        batch_callbacks = [name for name in stats.keys() if "Batch Item" in name]
        assert len(batch_callbacks) > 0

    def test_error_resilience_in_jax_context(self):
        """Test that callback errors don't break JAX execution."""

        def error_prone_callback(x):
            """Callback that sometimes fails."""
            if x > 5:
                raise ValueError(f"Value too large: {x}")

        @jax.jit
        def robust_computation(x):
            """Computation that continues despite callback errors."""
            # This callback might fail, but shouldn't break JAX
            jax.debug.callback(error_prone_callback, x)
            return x * 2

        # Test with values that will and won't cause errors
        test_values = [3.0, 7.0, 2.0, 10.0]
        results = []

        for val in test_values:
            result = robust_computation(jnp.array(val))
            results.append(float(result))

        # All computations should complete despite callback errors
        expected = [6.0, 14.0, 4.0, 20.0]
        assert results == expected

    def test_performance_monitoring_integration(self):
        """Test performance monitoring works with real JAX operations."""
        reset_callback_performance_stats()

        @jax.jit
        def monitored_computation(x):
            """Computation with monitored callbacks."""
            # Fast callback
            jax.debug.callback(lambda val: None, x, callback_name="fast_op")

            # Slower callback (simulated)
            def slow_callback(val):
                # Simulate some work
                for _ in range(100):
                    pass

            jax.debug.callback(slow_callback, x, callback_name="slow_op")

            return x + 1

        # Run multiple times to collect stats
        for i in range(5):
            result = monitored_computation(jnp.array(float(i)))

        # Check performance stats
        stats = get_callback_performance_stats()
        assert "fast_op" in stats
        assert "slow_op" in stats

        fast_stats = stats["fast_op"]
        slow_stats = stats["slow_op"]

        assert fast_stats["total_calls"] == 5
        assert slow_stats["total_calls"] == 5
        assert fast_stats["error_count"] == 0
        assert slow_stats["error_count"] == 0

        # Slow operation should take more time on average
        assert slow_stats["avg_time_ms"] >= fast_stats["avg_time_ms"]

    def test_memory_optimization_with_jax(self):
        """Test memory optimization features work with JAX arrays."""
        from jaxarc.utils.visualization import optimize_array_memory

        # Create JAX arrays of different types
        jax_int64 = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int64)
        jax_float64 = jnp.array([1.1, 2.2, 3.3], dtype=jnp.float64)

        # Convert to numpy and optimize
        np_int64 = np.asarray(jax_int64)
        np_float64 = np.asarray(jax_float64)

        optimized_int = optimize_array_memory(np_int64)
        optimized_float = optimize_array_memory(np_float64)

        # Check optimization worked
        assert optimized_int.dtype == np.int8  # Should be optimized to int8
        assert optimized_float.dtype == np.float32  # Should be optimized to float32

        # Verify data integrity
        np.testing.assert_array_equal(optimized_int, np_int64)
        np.testing.assert_allclose(optimized_float, np_float64, rtol=1e-6)

    def test_lazy_loading_with_jax_data(self):
        """Test lazy loading works with JAX-generated data."""
        from jaxarc.utils.visualization import LazyLoader

        load_count = 0

        def jax_data_loader():
            """Loader that generates JAX data."""
            nonlocal load_count
            load_count += 1

            # Generate data using JAX
            key = jax.random.PRNGKey(load_count)
            data = jax.random.normal(key, (100, 100))

            return {
                "jax_data": data,
                "numpy_data": np.asarray(data),
                "load_count": load_count,
            }

        lazy_loader = LazyLoader(jax_data_loader)

        # First access
        data1 = lazy_loader.get()
        assert data1["load_count"] == 1
        assert isinstance(data1["jax_data"], jnp.ndarray)
        assert isinstance(data1["numpy_data"], np.ndarray)

        # Second access (should be cached)
        data2 = lazy_loader.get()
        assert data2["load_count"] == 1  # Same data
        assert load_count == 1  # No additional load

        # Verify data shapes
        assert data1["jax_data"].shape == (100, 100)
        assert data1["numpy_data"].shape == (100, 100)

    def test_compressed_storage_with_jax_arrays(self):
        """Test compressed storage works with JAX arrays."""
        from jaxarc.utils.visualization import CompressedStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = CompressedStorage(Path(temp_dir))

            # Create data with JAX arrays
            key = jax.random.PRNGKey(42)
            jax_data = {
                "grid": jax.random.randint(key, (30, 30), 0, 10),
                "mask": jax.random.bernoulli(key, 0.8, (30, 30)),
                "metadata": {"created_with": "jax", "shape": (30, 30)},
            }

            # Convert to numpy for storage
            numpy_data = {
                "grid": np.asarray(jax_data["grid"]),
                "mask": np.asarray(jax_data["mask"]),
                "metadata": jax_data["metadata"],
            }

            # Save and load
            storage.save(numpy_data, "jax_test_data")
            loaded_data = storage.load("jax_test_data")

            # Verify data integrity
            np.testing.assert_array_equal(loaded_data["grid"], numpy_data["grid"])
            np.testing.assert_array_equal(loaded_data["mask"], numpy_data["mask"])
            assert loaded_data["metadata"]["created_with"] == "jax"

            # Convert back to JAX arrays
            restored_jax_data = {
                "grid": jnp.array(loaded_data["grid"]),
                "mask": jnp.array(loaded_data["mask"]),
                "metadata": loaded_data["metadata"],
            }

            # Verify JAX arrays work correctly
            assert restored_jax_data["grid"].shape == (30, 30)
            assert restored_jax_data["mask"].shape == (30, 30)
            np.testing.assert_array_equal(
                np.asarray(restored_jax_data["grid"]), np.asarray(jax_data["grid"])
            )


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_training_loop_simulation(self):
        """Simulate a training loop with visualization callbacks."""
        reset_callback_performance_stats()

        @jax.jit
        def training_step(state, action):
            """Simulate a training step with logging."""
            # Log current state (without step number to avoid tracing issues)
            jax_log_grid(state, title="Current State")

            # Simulate state update
            new_state = state + action

            # Log new state
            jax_log_grid(new_state, title="Updated State")

            return new_state

        # Simulate training loop
        initial_state = jnp.array([[1, 2], [3, 4]])
        state = initial_state

        for step in range(5):
            action = jnp.ones((2, 2)) * 0.1
            state = training_step(state, action)

        # Verify final state
        expected_final = initial_state + 5 * 0.1
        np.testing.assert_allclose(state, expected_final, rtol=1e-6)

        # Check callback statistics
        stats = get_callback_performance_stats()
        assert (
            len(stats) >= 2
        )  # Should have at least current and updated state callbacks

        # Check that we have the expected callback types
        callback_names = list(stats.keys())
        assert any("Current State" in name for name in callback_names)
        assert any("Updated State" in name for name in callback_names)

    def test_episode_management_integration(self):
        """Test episode management with JAX callbacks."""
        from jaxarc.utils.visualization import jax_log_episode_summary

        @jax.jit
        def simulate_episode(episode_num, num_steps):
            """Simulate an episode with summary logging."""
            total_reward = num_steps * 1.5
            final_similarity = jnp.tanh(total_reward / 10.0)
            success = final_similarity > 0.8

            jax_log_episode_summary(
                episode_num, num_steps, total_reward, final_similarity, success
            )

            return total_reward, final_similarity, success

        # Simulate multiple episodes
        episodes = []
        for ep in range(3):
            steps = 10 + ep * 5
            reward, similarity, success = simulate_episode(ep, steps)
            episodes.append((float(reward), float(similarity), bool(success)))

        # Verify episode data
        assert len(episodes) == 3
        assert all(len(ep) == 3 for ep in episodes)

        # Check that rewards increase with more steps
        rewards = [ep[0] for ep in episodes]
        assert rewards[1] > rewards[0]
        assert rewards[2] > rewards[1]
