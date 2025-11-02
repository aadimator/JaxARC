"""
Performance and robustness tests for JaxARC.

This module tests memory usage, performance characteristics, and thread safety
to ensure robust behavior under various conditions.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.types import Grid, JaxArcTask


class TestMemoryUsage:
    """Test memory usage for typical use cases and scenarios."""

    def test_grid_memory_efficiency(self):
        """Test Grid memory usage for various sizes."""
        # Test small grids
        small_data = jnp.zeros((5, 5), dtype=jnp.int32)
        small_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        small_grid = Grid(data=small_data, mask=small_mask)

        # Verify grid is created successfully
        assert small_grid.shape == (5, 5)

        # Test medium grids
        medium_data = jnp.zeros((15, 15), dtype=jnp.int32)
        medium_mask = jnp.ones((15, 15), dtype=jnp.bool_)
        medium_grid = Grid(data=medium_data, mask=medium_mask)

        assert medium_grid.shape == (15, 15)

        # Test large grids (typical ARC maximum)
        large_data = jnp.zeros((30, 30), dtype=jnp.int32)
        large_mask = jnp.ones((30, 30), dtype=jnp.bool_)
        large_grid = Grid(data=large_data, mask=large_mask)

        assert large_grid.shape == (30, 30)

        # Test that grids can be garbage collected
        del small_grid, medium_grid, large_grid
        gc.collect()

    def test_task_memory_usage(self):
        """Test JaxArcTask memory usage with multiple pairs."""
        max_pairs, max_height, max_width = 5, 20, 20

        # Create arrays for multiple pairs
        input_arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        output_arrays = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=input_arrays,
            input_masks_examples=masks,
            output_grids_examples=output_arrays,
            output_masks_examples=masks,
            num_train_pairs=3,
            test_input_grids=input_arrays,
            test_input_masks=masks,
            true_test_output_grids=output_arrays,
            true_test_output_masks=masks,
            num_test_pairs=2,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Verify task is created successfully
        assert task.num_train_pairs == 3
        assert task.num_test_pairs == 2

        # Test accessing multiple pairs doesn't cause memory issues
        for i in range(task.num_train_pairs):
            train_pair = task.get_train_pair(i)
            assert train_pair.input_grid.shape == (max_height, max_width)

        for i in range(task.num_test_pairs):
            test_pair = task.get_test_pair(i)
            assert test_pair.input_grid.shape == (max_height, max_width)

        # Clean up
        del task
        gc.collect()

    def test_batch_operations_memory(self):
        """Test memory usage with batch operations."""
        batch_size = 10
        grid_size = 10

        # Create batch of grids
        batch_data = jnp.zeros((batch_size, grid_size, grid_size), dtype=jnp.int32)
        batch_masks = jnp.ones((batch_size, grid_size, grid_size), dtype=jnp.bool_)

        # Test that batch operations don't cause memory issues
        grids = []
        for i in range(batch_size):
            grid = Grid(data=batch_data[i], mask=batch_masks[i])
            grids.append(grid)

        # Verify all grids are created
        assert len(grids) == batch_size
        for grid in grids:
            assert grid.shape == (grid_size, grid_size)

        # Clean up
        del grids, batch_data, batch_masks
        gc.collect()

    @pytest.mark.slow
    def test_large_scale_memory_usage(self):
        """Test memory usage with large-scale operations."""
        # Test with many small grids
        num_grids = 100
        grid_size = 5

        grids = []
        for i in range(num_grids):
            data = jnp.full((grid_size, grid_size), i % 10, dtype=jnp.int32)
            mask = jnp.ones((grid_size, grid_size), dtype=jnp.bool_)
            grid = Grid(data=data, mask=mask)
            grids.append(grid)

        # Verify all grids are created successfully
        assert len(grids) == num_grids

        # Test accessing all grids
        for i, grid in enumerate(grids):
            assert grid.shape == (grid_size, grid_size)
            assert jnp.all(grid.data == (i % 10))

        # Clean up
        del grids
        gc.collect()


class TestPerformanceCharacteristics:
    """Test performance characteristics of key operations."""

    def test_grid_creation_performance(self):
        """Test Grid creation performance for various sizes."""
        sizes = [5, 10, 20, 30]

        for size in sizes:
            data = jnp.zeros((size, size), dtype=jnp.int32)
            mask = jnp.ones((size, size), dtype=jnp.bool_)

            # Time grid creation
            start_time = time.time()
            grid = Grid(data=data, mask=mask)
            creation_time = time.time() - start_time

            # Grid creation should be fast (< 1 second even for large grids)
            assert creation_time < 1.0, (
                f"Grid creation took {creation_time:.3f}s for size {size}x{size}"
            )
            assert grid.shape == (size, size)

    def test_task_access_performance(self):
        """Test JaxArcTask pair access performance."""
        max_pairs, max_height, max_width = 10, 15, 15

        # Create task with multiple pairs
        arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=arrays,
            input_masks_examples=masks,
            output_grids_examples=arrays,
            output_masks_examples=masks,
            num_train_pairs=max_pairs,
            test_input_grids=arrays,
            test_input_masks=masks,
            true_test_output_grids=arrays,
            true_test_output_masks=masks,
            num_test_pairs=max_pairs,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Time pair access
        start_time = time.time()
        for i in range(task.num_train_pairs):
            pair = task.get_train_pair(i)
            assert pair.input_grid.shape == (max_height, max_width)
        access_time = time.time() - start_time

        # Pair access should be fast
        assert access_time < 1.0, (
            f"Pair access took {access_time:.3f}s for {max_pairs} pairs"
        )

    def test_jax_transformation_performance(self):
        """Test performance of JAX transformations on core operations."""

        # Test JIT compilation performance
        def create_grid_fn(data, mask):
            return Grid(data=data, mask=mask)

        data = jnp.zeros((10, 10), dtype=jnp.int32)
        mask = jnp.ones((10, 10), dtype=jnp.bool_)

        # Time JIT compilation
        start_time = time.time()
        jitted_fn = jax.jit(create_grid_fn)
        grid = jitted_fn(data, mask)
        jit_time = time.time() - start_time

        # JIT compilation should complete reasonably quickly
        assert jit_time < 5.0, f"JIT compilation took {jit_time:.3f}s"
        assert grid.shape == (10, 10)

        # Test subsequent calls are fast
        start_time = time.time()
        grid2 = jitted_fn(data, mask)
        subsequent_time = time.time() - start_time

        # Subsequent JIT calls should be very fast
        assert subsequent_time < 0.1, f"Subsequent JIT call took {subsequent_time:.3f}s"

    def test_batch_processing_performance(self):
        """Test performance of batch processing operations."""
        batch_size = 20
        grid_size = 8

        # Create batch data
        batch_data = jnp.zeros((batch_size, grid_size, grid_size), dtype=jnp.int32)
        batch_masks = jnp.ones((batch_size, grid_size, grid_size), dtype=jnp.bool_)

        # Time batch grid creation
        start_time = time.time()
        grids = []
        for i in range(batch_size):
            grid = Grid(data=batch_data[i], mask=batch_masks[i])
            grids.append(grid)
        batch_time = time.time() - start_time

        # Batch processing should be reasonably fast
        assert batch_time < 2.0, (
            f"Batch processing took {batch_time:.3f}s for {batch_size} grids"
        )
        assert len(grids) == batch_size

    @pytest.mark.slow
    def test_stress_test_performance(self):
        """Stress test with many operations to check for performance degradation."""
        num_iterations = 50
        grid_size = 10

        times = []
        for i in range(num_iterations):
            data = jnp.full((grid_size, grid_size), i % 10, dtype=jnp.int32)
            mask = jnp.ones((grid_size, grid_size), dtype=jnp.bool_)

            start_time = time.time()
            grid = Grid(data=data, mask=mask)
            _ = grid.shape  # Access property
            iteration_time = time.time() - start_time
            times.append(iteration_time)

        # Check that performance doesn't degrade significantly
        early_avg = sum(times[:10]) / 10
        late_avg = sum(times[-10:]) / 10

        # Later iterations shouldn't be more than 2x slower than early ones
        assert late_avg < early_avg * 2, (
            f"Performance degraded: early={early_avg:.4f}s, late={late_avg:.4f}s"
        )


class TestThreadSafety:
    """Test thread safety for concurrent access where applicable."""

    def test_grid_concurrent_read_access(self):
        """Test that multiple threads can safely read from the same Grid."""
        data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        mask = jnp.ones((3, 3), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)

        results = []
        errors = []

        def read_grid():
            try:
                # Multiple read operations
                shape = grid.shape
                data_sum = jnp.sum(grid.data)
                mask_count = jnp.sum(grid.mask)
                return (shape, data_sum, mask_count)
            except Exception as e:
                errors.append(e)
                return None

        # Run concurrent reads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_grid) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # Check that no errors occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        # Check that all results are consistent
        expected_shape = (3, 3)
        expected_sum = jnp.sum(data)
        expected_mask_count = jnp.sum(mask)

        for result in results:
            if result is not None:
                shape, data_sum, mask_count = result
                assert shape == expected_shape
                chex.assert_trees_all_equal(data_sum, expected_sum)
                chex.assert_trees_all_equal(mask_count, expected_mask_count)

    def test_task_concurrent_access(self):
        """Test that multiple threads can safely access JaxArcTask."""
        max_pairs, max_height, max_width = 3, 5, 5
        arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=arrays,
            input_masks_examples=masks,
            output_grids_examples=arrays,
            output_masks_examples=masks,
            num_train_pairs=2,
            test_input_grids=arrays,
            test_input_masks=masks,
            true_test_output_grids=arrays,
            true_test_output_masks=masks,
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),
        )

        results = []
        errors = []

        def access_task():
            try:
                # Multiple access operations
                summary = task.get_task_summary()
                train_pair = task.get_train_pair(0)
                test_pair = task.get_test_pair(0)
                return (
                    summary,
                    train_pair.input_grid.shape,
                    test_pair.input_grid.shape,
                )
            except Exception as e:
                errors.append(e)
                return None

        # Run concurrent access
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_task) for _ in range(8)]
            results = [future.result() for future in as_completed(futures)]

        # Check that no errors occurred
        assert len(errors) == 0, f"Errors during concurrent task access: {errors}"

        # Check that all results are consistent
        for result in results:
            if result is not None:
                summary, train_shape, test_shape = result
                assert summary["task_index"] == 42
                assert summary["num_train_pairs"] == 2
                assert summary["num_test_pairs"] == 1
                assert train_shape == (max_height, max_width)
                assert test_shape == (max_height, max_width)

    def test_jax_operations_thread_safety(self):
        """Test that JAX operations are thread-safe for read access."""
        data = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)

        results = []
        errors = []

        def jax_operations():
            try:
                # Various JAX operations
                sum_result = jnp.sum(data)
                mean_result = jnp.mean(data)
                max_result = jnp.max(data)
                shape_result = data.shape
                return (sum_result, mean_result, max_result, shape_result)
            except Exception as e:
                errors.append(e)
                return None

        # Run concurrent JAX operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(jax_operations) for _ in range(6)]
            results = [future.result() for future in as_completed(futures)]

        # Check that no errors occurred
        assert len(errors) == 0, f"Errors during concurrent JAX operations: {errors}"

        # Check that all results are consistent
        expected_sum = jnp.sum(data)
        expected_mean = jnp.mean(data)
        expected_max = jnp.max(data)
        expected_shape = data.shape

        for result in results:
            if result is not None:
                sum_result, mean_result, max_result, shape_result = result
                chex.assert_trees_all_equal(sum_result, expected_sum)
                chex.assert_trees_all_equal(mean_result, expected_mean)
                chex.assert_trees_all_equal(max_result, expected_max)
                assert shape_result == expected_shape

    def test_configuration_thread_safety(self):
        """Test that configuration objects are thread-safe for read access."""
        config = JaxArcConfig()

        results = []
        errors = []

        def access_config():
            try:
                # Access various config properties
                env_config = config.environment
                action_config = config.action
                dataset_config = config.dataset
                return (env_config.max_episode_steps, action_config, dataset_config)
            except Exception as e:
                errors.append(e)
                return None

        # Run concurrent config access
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(access_config) for _ in range(6)]
            results = [future.result() for future in as_completed(futures)]

        # Check that no errors occurred
        assert len(errors) == 0, f"Errors during concurrent config access: {errors}"

        # Check that all results are consistent
        for result in results:
            if result is not None:
                max_episode_steps, action_config, dataset_config = result
                assert max_episode_steps > 0  # Should be a positive value
                assert action_config is not None
                assert dataset_config is not None


class TestRobustnessUnderStress:
    """Test robustness under various stress conditions."""

    def test_rapid_creation_destruction(self):
        """Test rapid creation and destruction of objects."""
        num_cycles = 100

        for i in range(num_cycles):
            # Create objects
            data = jnp.full((5, 5), i % 10, dtype=jnp.int32)
            mask = jnp.ones((5, 5), dtype=jnp.bool_)
            grid = Grid(data=data, mask=mask)

            # Use objects
            shape = grid.shape
            assert shape == (5, 5)

            # Objects will be destroyed at end of loop iteration
            del grid, data, mask

        # Force garbage collection
        gc.collect()

    def test_exception_recovery(self):
        """Test that the system recovers gracefully from exceptions."""
        valid_data = jnp.array([[1, 2]], dtype=jnp.int32)
        valid_mask = jnp.ones((1, 2), dtype=jnp.bool_)

        # Cause some exceptions and verify recovery
        for i in range(5):
            try:
                # Try to create invalid grid
                invalid_data = jnp.array([[10 + i]], dtype=jnp.int32)  # Invalid color
                Grid(data=invalid_data, mask=valid_mask[:, :1])
            except ValueError:
                pass  # Expected

            # Create valid grid after exception
            valid_grid = Grid(data=valid_data, mask=valid_mask)
            assert valid_grid.shape == (1, 2)

    @pytest.mark.slow
    def test_long_running_operations(self):
        """Test behavior during long-running operations."""
        # Simulate long-running operation with many small tasks
        num_operations = 200
        results = []

        for i in range(num_operations):
            data = jnp.full((3, 3), i % 10, dtype=jnp.int32)
            mask = jnp.ones((3, 3), dtype=jnp.bool_)
            grid = Grid(data=data, mask=mask)

            # Perform some operations
            shape = grid.shape
            data_sum = jnp.sum(grid.data)

            results.append((shape, data_sum))

            # Occasional cleanup
            if i % 50 == 0:
                gc.collect()

        # Verify all operations completed successfully
        assert len(results) == num_operations
        for i, (shape, data_sum) in enumerate(results):
            assert shape == (3, 3)
            expected_sum = (i % 10) * 9  # 3x3 grid filled with i%10
            chex.assert_trees_all_equal(data_sum, expected_sum)

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        # Create many objects and verify they can be cleaned up
        objects = []

        for i in range(50):
            data = jnp.zeros((4, 4), dtype=jnp.int32)
            mask = jnp.ones((4, 4), dtype=jnp.bool_)
            grid = Grid(data=data, mask=mask)
            objects.append(grid)

        # Verify objects are created
        assert len(objects) == 50

        # Clear references
        objects.clear()

        # Force garbage collection
        gc.collect()

        # Create new objects to verify no resource leaks
        for i in range(10):
            data = jnp.ones((6, 6), dtype=jnp.int32)
            mask = jnp.ones((6, 6), dtype=jnp.bool_)
            grid = Grid(data=data, mask=mask)
            assert grid.shape == (6, 6)


class TestErrorRecovery:
    """Test error recovery and system stability."""

    def test_recovery_after_validation_errors(self):
        """Test that system recovers properly after validation errors."""
        # Cause validation errors and verify recovery
        for i in range(10):
            try:
                # Invalid shape
                data = jnp.ones((2, 3), dtype=jnp.int32)
                mask = jnp.ones((3, 2), dtype=jnp.bool_)
                Grid(data=data, mask=mask)
            except AssertionError:
                pass  # Expected

            try:
                # Invalid color values
                data = jnp.array([[10 + i]], dtype=jnp.int32)
                mask = jnp.ones((1, 1), dtype=jnp.bool_)
                Grid(data=data, mask=mask)
            except ValueError:
                pass  # Expected

            # Create valid grid after errors
            valid_data = jnp.array([[i % 10]], dtype=jnp.int32)
            valid_mask = jnp.ones((1, 1), dtype=jnp.bool_)
            valid_grid = Grid(data=valid_data, mask=valid_mask)
            assert valid_grid.shape == (1, 1)

    def test_partial_failure_handling(self):
        """Test handling of partial failures in batch operations."""
        batch_size = 10
        successful_grids = []

        for i in range(batch_size):
            try:
                if i % 3 == 0:
                    # Intentionally create invalid grid
                    data = jnp.array([[10]], dtype=jnp.int32)  # Invalid color
                    mask = jnp.ones((1, 1), dtype=jnp.bool_)
                    grid = Grid(data=data, mask=mask)
                else:
                    # Create valid grid
                    data = jnp.array([[i % 10]], dtype=jnp.int32)
                    mask = jnp.ones((1, 1), dtype=jnp.bool_)
                    grid = Grid(data=data, mask=mask)
                    successful_grids.append(grid)
            except ValueError:
                pass  # Expected for invalid grids

        # Verify that valid grids were created successfully
        # Every 3rd item (i % 3 == 0) should fail, so we expect 2/3 to succeed
        expected_successful = (
            batch_size - (batch_size + 2) // 3
        )  # More accurate calculation
        assert len(successful_grids) == expected_successful

        for grid in successful_grids:
            assert grid.shape == (1, 1)

    def test_state_consistency_after_errors(self):
        """Test that system state remains consistent after errors."""
        # Create initial valid state
        config = JaxArcConfig()
        assert config is not None

        # Cause some errors
        for i in range(5):
            try:
                # Try invalid configuration
                invalid_config = JaxArcConfig(environment={"max_grid_size": -1})
            except (ValueError, TypeError):
                pass  # Expected

        # Verify original config is still valid
        assert config.environment.max_episode_steps > 0

        # Create new valid objects
        data = jnp.array([[1, 2]], dtype=jnp.int32)
        mask = jnp.ones((1, 2), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)
        assert grid.shape == (1, 2)
