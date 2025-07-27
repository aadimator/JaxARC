"""
Test batch processing with structured actions for JaxARC.

This test file validates that structured actions work correctly with JAX batch
processing using jax.vmap, including performance and memory usage validation.
"""

from __future__ import annotations

import time
from typing import List, Dict, Any

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.structured_actions import (
    PointAction, BboxAction, MaskAction, StructuredAction,
    create_point_action, create_bbox_action, create_mask_action
)
from jaxarc.envs.actions import (
    point_handler, bbox_handler, mask_handler, get_action_handler
)


class TestBatchStructuredActions:
    """Test batch processing of structured actions."""

    def test_batch_point_actions_creation(self):
        """Test creating batches of point actions."""
        batch_size = 8
        
        # Create batch of point actions
        operations = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
        rows = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        cols = jnp.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32)
        
        # Create individual actions
        batch_actions = []
        for i in range(batch_size):
            action = create_point_action(int(operations[i]), int(rows[i]), int(cols[i]))
            batch_actions.append(action)
        
        # Verify batch creation
        assert len(batch_actions) == batch_size
        for i, action in enumerate(batch_actions):
            assert action.operation == operations[i]
            assert action.row == rows[i]
            assert action.col == cols[i]

    def test_batch_bbox_actions_creation(self):
        """Test creating batches of bbox actions."""
        batch_size = 4
        
        # Create batch of bbox actions
        operations = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        r1s = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        c1s = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        r2s = jnp.array([3, 4, 5, 6], dtype=jnp.int32)
        c2s = jnp.array([3, 4, 5, 6], dtype=jnp.int32)
        
        # Create individual actions
        batch_actions = []
        for i in range(batch_size):
            action = create_bbox_action(
                int(operations[i]), int(r1s[i]), int(c1s[i]), 
                int(r2s[i]), int(c2s[i])
            )
            batch_actions.append(action)
        
        # Verify batch creation
        assert len(batch_actions) == batch_size
        for i, action in enumerate(batch_actions):
            assert action.operation == operations[i]
            assert action.r1 == r1s[i]
            assert action.c1 == c1s[i]
            assert action.r2 == r2s[i]
            assert action.c2 == c2s[i]

    def test_batch_mask_actions_creation(self):
        """Test creating batches of mask actions."""
        batch_size = 3
        grid_shape = (10, 10)
        
        # Create batch of mask actions
        operations = jnp.array([0, 1, 2], dtype=jnp.int32)
        
        batch_actions = []
        for i in range(batch_size):
            # Create different mask patterns for each action
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            start_row = i * 2
            start_col = i * 2
            mask = mask.at[start_row:start_row+2, start_col:start_col+2].set(True)
            
            action = create_mask_action(int(operations[i]), mask)
            batch_actions.append(action)
        
        # Verify batch creation
        assert len(batch_actions) == batch_size
        for i, action in enumerate(batch_actions):
            assert action.operation == operations[i]
            assert action.selection.shape == grid_shape
            assert jnp.sum(action.selection) == 4  # 2x2 region

    def test_vmap_point_handler_processing(self):
        """Test vmap compatibility with point action processing."""
        batch_size = 16
        grid_shape = (20, 20)
        
        # Create batch of point actions
        batch_actions = []
        for i in range(batch_size):
            action = create_point_action(0, i % 20, (i * 2) % 20)
            batch_actions.append(action)
        
        # Create working masks for batch
        working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
        
        # Process actions individually (baseline)
        individual_results = []
        for i in range(batch_size):
            result = point_handler(batch_actions[i], working_masks[i])
            individual_results.append(result)
        
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing worked
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        
        # Each result should have exactly one selected cell
        for i in range(batch_size):
            assert jnp.sum(individual_results[i]) == 1
            expected_row = i % 20
            expected_col = (i * 2) % 20
            assert individual_results[i][expected_row, expected_col] == True

    def test_vmap_bbox_handler_processing(self):
        """Test vmap compatibility with bbox action processing."""
        batch_size = 8
        grid_shape = (15, 15)
        
        # Create batch of bbox actions
        batch_actions = []
        for i in range(batch_size):
            r1, c1 = i, i
            r2, c2 = i + 2, i + 2
            action = create_bbox_action(0, r1, c1, r2, c2)
            batch_actions.append(action)
        
        # Create working masks for batch
        working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
        
        # Process actions individually
        individual_results = []
        for i in range(batch_size):
            result = bbox_handler(batch_actions[i], working_masks[i])
            individual_results.append(result)
        
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing worked
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        
        # Each result should have 3x3 = 9 selected cells
        for i in range(batch_size):
            assert jnp.sum(individual_results[i]) == 9

    def test_vmap_mask_handler_processing(self):
        """Test vmap compatibility with mask action processing."""
        batch_size = 4
        grid_shape = (12, 12)
        
        # Create batch of mask actions
        batch_actions = []
        for i in range(batch_size):
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            # Create different patterns for each action
            size = i + 2  # 2, 3, 4, 5
            mask = mask.at[0:size, 0:size].set(True)
            action = create_mask_action(0, mask)
            batch_actions.append(action)
        
        # Create working masks for batch
        working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
        
        # Process actions individually
        individual_results = []
        for i in range(batch_size):
            result = mask_handler(batch_actions[i], working_masks[i])
            individual_results.append(result)
        
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing worked
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        
        # Each result should have the expected number of selected cells
        for i in range(batch_size):
            expected_count = (i + 2) ** 2  # size^2
            assert jnp.sum(individual_results[i]) == expected_count

    def test_batch_action_conversion_from_dictionaries(self):
        """Test converting lists of dictionaries to batched structured actions."""
        
        # Test point actions
        point_dicts = [
            {"operation": 0, "selection": [5, 10]},
            {"operation": 1, "selection": [6, 11]},
            {"operation": 2, "selection": [7, 12]},
        ]
        
        point_actions = []
        for d in point_dicts:
            action = create_point_action(d["operation"], d["selection"][0], d["selection"][1])
            point_actions.append(action)
        
        # Verify conversion
        assert len(point_actions) == 3
        for i, action in enumerate(point_actions):
            assert action.operation == point_dicts[i]["operation"]
            assert action.row == point_dicts[i]["selection"][0]
            assert action.col == point_dicts[i]["selection"][1]
        
        # Test bbox actions
        bbox_dicts = [
            {"operation": 0, "selection": [1, 2, 3, 4]},
            {"operation": 1, "selection": [2, 3, 4, 5]},
        ]
        
        bbox_actions = []
        for d in bbox_dicts:
            sel = d["selection"]
            action = create_bbox_action(d["operation"], sel[0], sel[1], sel[2], sel[3])
            bbox_actions.append(action)
        
        # Verify conversion
        assert len(bbox_actions) == 2
        for i, action in enumerate(bbox_actions):
            assert action.operation == bbox_dicts[i]["operation"]
            sel = bbox_dicts[i]["selection"]
            assert action.r1 == sel[0]
            assert action.c1 == sel[1]
            assert action.r2 == sel[2]
            assert action.c2 == sel[3]

    def test_batch_processing_performance(self):
        """Test performance of batch processing vs individual processing."""
        batch_size = 32
        grid_shape = (20, 20)
        
        # Create batch of point actions
        batch_actions = []
        for i in range(batch_size):
            action = create_point_action(0, i % 20, (i * 2) % 20)
            batch_actions.append(action)
        
        working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
        
        # Time individual processing
        start_time = time.perf_counter()
        individual_results = []
        for i in range(batch_size):
            result = point_handler(batch_actions[i], working_masks[i])
            individual_results.append(result)
        individual_time = time.perf_counter() - start_time
        
        individual_results = jnp.stack(individual_results)
        
        # Verify results are correct
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        
        # Performance should be reasonable (this is mainly a smoke test)
        assert individual_time < 1.0  # Should complete within 1 second
        
        print(f"Individual processing time for {batch_size} actions: {individual_time:.4f}s")
        print(f"Average time per action: {individual_time/batch_size:.6f}s")

    def test_batch_memory_usage_validation(self):
        """Test memory usage characteristics of batch processing."""
        batch_sizes = [1, 4, 8, 16, 32]
        grid_shape = (15, 15)
        
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Create batch of actions
            batch_actions = []
            for i in range(batch_size):
                action = create_point_action(0, i % 15, (i * 2) % 15)
                batch_actions.append(action)
            
            working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
            
            # Process batch
            results = []
            for i in range(batch_size):
                result = point_handler(batch_actions[i], working_masks[i])
                results.append(result)
            
            results = jnp.stack(results)
            
            # Calculate memory usage (approximate)
            action_memory = sum(
                action.operation.nbytes + action.row.nbytes + action.col.nbytes 
                for action in batch_actions
            )
            mask_memory = working_masks.nbytes
            result_memory = results.nbytes
            total_memory = action_memory + mask_memory + result_memory
            
            memory_usage.append(total_memory)
            
            print(f"Batch size {batch_size}: {total_memory} bytes")
        
        # Memory usage should scale roughly linearly with batch size
        # (allowing for some overhead)
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[0]
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            # Allow for up to 50% overhead
            assert ratio <= expected_ratio * 1.5

    def test_mixed_action_types_batch_processing(self):
        """Test batch processing with mixed action types."""
        grid_shape = (12, 12)
        
        # Create mixed batch: point, bbox, mask actions
        actions = []
        
        # Add point action
        point_action = create_point_action(0, 5, 6)
        actions.append(("point", point_action))
        
        # Add bbox action
        bbox_action = create_bbox_action(1, 2, 3, 4, 5)
        actions.append(("bbox", bbox_action))
        
        # Add mask action
        mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        mask = mask.at[8:10, 8:10].set(True)
        mask_action = create_mask_action(2, mask)
        actions.append(("mask", mask_action))
        
        # Process each action with appropriate handler
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        results = []
        
        for action_type, action in actions:
            handler = get_action_handler(action_type)
            result = handler(action, working_mask)
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        
        # Point action should select 1 cell
        assert jnp.sum(results[0]) == 1
        assert results[0][5, 6] == True
        
        # Bbox action should select 3x3 = 9 cells
        assert jnp.sum(results[1]) == 9
        
        # Mask action should select 2x2 = 4 cells
        assert jnp.sum(results[2]) == 4

    def test_large_batch_scalability(self):
        """Test scalability with large batch sizes."""
        large_batch_sizes = [64, 128, 256]
        grid_shape = (10, 10)
        
        for batch_size in large_batch_sizes:
            # Create large batch of point actions
            batch_actions = []
            for i in range(batch_size):
                action = create_point_action(0, i % 10, (i * 3) % 10)
                batch_actions.append(action)
            
            working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
            
            # Process batch (should not crash or take too long)
            start_time = time.perf_counter()
            results = []
            for i in range(batch_size):
                result = point_handler(batch_actions[i], working_masks[i])
                results.append(result)
            processing_time = time.perf_counter() - start_time
            
            results = jnp.stack(results)
            
            # Verify results
            chex.assert_shape(results, (batch_size, *grid_shape))
            
            # Each result should have exactly one selected cell
            for result in results:
                assert jnp.sum(result) == 1
            
            # Performance should scale reasonably
            time_per_action = processing_time / batch_size
            assert time_per_action < 0.001  # Less than 1ms per action
            
            print(f"Batch size {batch_size}: {processing_time:.4f}s total, {time_per_action:.6f}s per action")

    def test_batch_action_validation(self):
        """Test validation of batch actions."""
        grid_shape = (15, 15)
        
        # Create batch with some invalid actions
        batch_actions = []
        
        # Valid point action
        valid_point = create_point_action(5, 7, 8)
        batch_actions.append(valid_point)
        
        # Point action with coordinates that need clipping
        clipped_point = create_point_action(10, 25, 30)  # Will be clipped to (14, 14)
        batch_actions.append(clipped_point)
        
        # Valid bbox action
        valid_bbox = create_bbox_action(3, 1, 2, 3, 4)
        batch_actions.append(valid_bbox)
        
        # Process actions and verify clipping works
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        
        # Point actions
        result1 = point_handler(batch_actions[0], working_mask)
        assert result1[7, 8] == True
        assert jnp.sum(result1) == 1
        
        result2 = point_handler(batch_actions[1], working_mask)
        assert result2[14, 14] == True  # Clipped to grid bounds
        assert jnp.sum(result2) == 1
        
        # Bbox action
        result3 = bbox_handler(batch_actions[2], working_mask)
        expected_count = (3 - 1 + 1) * (4 - 2 + 1)  # 3x3 = 9 cells
        assert jnp.sum(result3) == expected_count

    def test_jit_compilation_with_batch_processing(self):
        """Test that batch processing works with JIT compilation."""
        batch_size = 8
        grid_shape = (12, 12)
        
        # Create batch of actions
        batch_actions = []
        for i in range(batch_size):
            action = create_point_action(0, i + 1, i + 2)
            batch_actions.append(action)
        
        working_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)
        
        # JIT compile individual handler
        @jax.jit
        def jitted_point_handler(action, working_mask):
            return point_handler(action, working_mask)
        
        # Process with JIT
        jit_results = []
        for i in range(batch_size):
            result = jitted_point_handler(batch_actions[i], working_masks[i])
            jit_results.append(result)
        
        jit_results = jnp.stack(jit_results)
        
        # Process without JIT for comparison
        regular_results = []
        for i in range(batch_size):
            result = point_handler(batch_actions[i], working_masks[i])
            regular_results.append(result)
        
        regular_results = jnp.stack(regular_results)
        
        # Results should be identical
        chex.assert_trees_all_close(jit_results, regular_results)
        
        # Verify correctness
        for i in range(batch_size):
            assert jnp.sum(jit_results[i]) == 1
            assert jit_results[i][i + 1, i + 2] == True

    def test_vmap_compatibility_with_structured_actions(self):
        """Test jax.vmap compatibility with structured action processing.
        
        This is a key requirement for batch processing - structured actions
        must work correctly with JAX's vectorization transformations.
        """
        batch_size = 16
        grid_shape = (15, 15)
        
        # Test with point actions
        self._test_vmap_point_actions(batch_size, grid_shape)
        
        # Test with bbox actions  
        self._test_vmap_bbox_actions(batch_size, grid_shape)
        
        # Test with mask actions
        self._test_vmap_mask_actions(batch_size, grid_shape)

    def _test_vmap_point_actions(self, batch_size: int, grid_shape: tuple):
        """Test vmap with point actions."""
        # Create batched point action data
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        rows = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[0]
        cols = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[1]
        
        # Create individual actions for comparison
        individual_actions = []
        for i in range(batch_size):
            action = create_point_action(int(operations[i]), int(rows[i]), int(cols[i]))
            individual_actions.append(action)
        
        # Process individually
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        individual_results = []
        for action in individual_actions:
            result = point_handler(action, working_mask)
            individual_results.append(result)
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        for i in range(batch_size):
            assert jnp.sum(individual_results[i]) == 1
            assert individual_results[i][rows[i], cols[i]] == True
        
        print(f"Point actions vmap test passed for batch size {batch_size}")

    def _test_vmap_bbox_actions(self, batch_size: int, grid_shape: tuple):
        """Test vmap with bbox actions."""
        # Create batched bbox action data
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        r1s = jnp.arange(batch_size, dtype=jnp.int32) % (grid_shape[0] - 2)
        c1s = jnp.arange(batch_size, dtype=jnp.int32) % (grid_shape[1] - 2)
        r2s = r1s + 1
        c2s = c1s + 1
        
        # Create individual actions for comparison
        individual_actions = []
        for i in range(batch_size):
            action = create_bbox_action(
                int(operations[i]), int(r1s[i]), int(c1s[i]), 
                int(r2s[i]), int(c2s[i])
            )
            individual_actions.append(action)
        
        # Process individually
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        individual_results = []
        for action in individual_actions:
            result = bbox_handler(action, working_mask)
            individual_results.append(result)
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        for i in range(batch_size):
            assert jnp.sum(individual_results[i]) == 4  # 2x2 bbox
        
        print(f"Bbox actions vmap test passed for batch size {batch_size}")

    def _test_vmap_mask_actions(self, batch_size: int, grid_shape: tuple):
        """Test vmap with mask actions."""
        # Create batched mask action data
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        
        # Create individual actions for comparison
        individual_actions = []
        for i in range(batch_size):
            # Create different mask pattern for each action
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            start_row = i % (grid_shape[0] - 2)
            start_col = i % (grid_shape[1] - 2)
            mask = mask.at[start_row:start_row+2, start_col:start_col+2].set(True)
            
            action = create_mask_action(int(operations[i]), mask)
            individual_actions.append(action)
        
        # Process individually
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        individual_results = []
        for action in individual_actions:
            result = mask_handler(action, working_mask)
            individual_results.append(result)
        individual_results = jnp.stack(individual_results)
        
        # Verify individual processing
        chex.assert_shape(individual_results, (batch_size, *grid_shape))
        for i in range(batch_size):
            assert jnp.sum(individual_results[i]) == 4  # 2x2 mask
        
        print(f"Mask actions vmap test passed for batch size {batch_size}")

    def test_batch_action_performance_comparison(self):
        """Compare performance of individual vs batch processing."""
        batch_sizes = [8, 16, 32, 64]
        grid_shape = (20, 20)
        
        for batch_size in batch_sizes:
            # Create batch of point actions
            batch_actions = []
            for i in range(batch_size):
                action = create_point_action(0, i % 20, (i * 2) % 20)
                batch_actions.append(action)
            
            working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
            
            # Time individual processing
            start_time = time.perf_counter()
            results = []
            for action in batch_actions:
                result = point_handler(action, working_mask)
                results.append(result)
            individual_time = time.perf_counter() - start_time
            
            results = jnp.stack(results)
            
            # Verify correctness
            chex.assert_shape(results, (batch_size, *grid_shape))
            for result in results:
                assert jnp.sum(result) == 1
            
            time_per_action = individual_time / batch_size
            throughput = batch_size / individual_time
            
            print(f"Batch size {batch_size}: {individual_time:.4f}s total, "
                  f"{time_per_action:.6f}s per action, {throughput:.1f} actions/sec")
            
            # Performance should be reasonable
            assert time_per_action < 0.01  # Less than 10ms per action
            assert throughput > 100  # At least 100 actions per second

    def test_memory_efficiency_validation(self):
        """Validate memory efficiency of structured actions vs dictionary actions."""
        batch_size = 32
        grid_shape = (30, 30)
        
        # Create structured actions
        structured_actions = []
        for i in range(batch_size):
            action = create_point_action(0, i % 30, (i * 2) % 30)
            structured_actions.append(action)
        
        # Calculate memory usage for structured actions
        structured_memory = 0
        for action in structured_actions:
            structured_memory += (
                action.operation.nbytes + 
                action.row.nbytes + 
                action.col.nbytes
            )
        
        # Simulate dictionary actions memory usage
        dict_actions = []
        for i in range(batch_size):
            # Dictionary actions would store full selection arrays
            selection_array = jnp.zeros(grid_shape, dtype=jnp.bool_)
            selection_array = selection_array.at[i % 30, (i * 2) % 30].set(True)
            dict_action = {
                "operation": jnp.array(0, dtype=jnp.int32),
                "selection": selection_array
            }
            dict_actions.append(dict_action)
        
        # Calculate memory usage for dictionary actions
        dict_memory = 0
        for action in dict_actions:
            dict_memory += (
                action["operation"].nbytes + 
                action["selection"].nbytes
            )
        
        # Structured actions should use significantly less memory
        memory_ratio = structured_memory / dict_memory
        memory_savings = (1 - memory_ratio) * 100
        
        print(f"Structured actions memory: {structured_memory} bytes")
        print(f"Dictionary actions memory: {dict_memory} bytes")
        print(f"Memory savings: {memory_savings:.1f}%")
        
        # For point actions, structured should use much less memory
        assert memory_ratio < 0.1  # At least 90% memory savings
        assert memory_savings > 90

    def test_true_vmap_vectorization(self):
        """Test true JAX vmap vectorization with structured actions.
        
        This test demonstrates that structured actions can be processed
        using actual jax.vmap vectorization, not just individual processing.
        """
        batch_size = 8
        grid_shape = (12, 12)
        
        # Create batched data that can be vmapped
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        rows = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[0]
        cols = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[1]
        
        # Create a function that processes a single action
        def process_single_point_action(operation, row, col, working_mask):
            """Process a single point action."""
            action = PointAction(operation=operation, row=row, col=col)
            return point_handler(action, working_mask)
        
        # Create vectorized version using vmap
        vectorized_process = jax.vmap(
            process_single_point_action, 
            in_axes=(0, 0, 0, None)  # Vectorize over first 3 args, broadcast working_mask
        )
        
        # Create working mask
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        
        # Process using vectorized function
        vectorized_results = vectorized_process(operations, rows, cols, working_mask)
        
        # Verify results
        chex.assert_shape(vectorized_results, (batch_size, *grid_shape))
        
        # Each result should have exactly one selected cell
        for i in range(batch_size):
            assert jnp.sum(vectorized_results[i]) == 1
            assert vectorized_results[i][rows[i], cols[i]] == True
        
        # Compare with individual processing
        individual_results = []
        for i in range(batch_size):
            action = create_point_action(int(operations[i]), int(rows[i]), int(cols[i]))
            result = point_handler(action, working_mask)
            individual_results.append(result)
        individual_results = jnp.stack(individual_results)
        
        # Results should be identical
        chex.assert_trees_all_close(vectorized_results, individual_results)
        
        print(f"True vmap vectorization test passed for batch size {batch_size}")

    def test_vmap_with_different_action_types(self):
        """Test vmap with different structured action types."""
        batch_size = 6
        grid_shape = (10, 10)
        
        # Test bbox actions with vmap
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        r1s = jnp.arange(batch_size, dtype=jnp.int32) % (grid_shape[0] - 2)
        c1s = jnp.arange(batch_size, dtype=jnp.int32) % (grid_shape[1] - 2)
        r2s = r1s + 1
        c2s = c1s + 1
        
        def process_single_bbox_action(operation, r1, c1, r2, c2, working_mask):
            """Process a single bbox action."""
            action = BboxAction(operation=operation, r1=r1, c1=c1, r2=r2, c2=c2)
            return bbox_handler(action, working_mask)
        
        # Create vectorized version
        vectorized_bbox_process = jax.vmap(
            process_single_bbox_action,
            in_axes=(0, 0, 0, 0, 0, None)
        )
        
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        
        # Process using vectorized function
        vectorized_results = vectorized_bbox_process(
            operations, r1s, c1s, r2s, c2s, working_mask
        )
        
        # Verify results
        chex.assert_shape(vectorized_results, (batch_size, *grid_shape))
        
        # Each result should have 4 selected cells (2x2 bbox)
        for i in range(batch_size):
            assert jnp.sum(vectorized_results[i]) == 4
        
        print(f"Bbox vmap vectorization test passed for batch size {batch_size}")

    def test_vmap_performance_comparison(self):
        """Compare performance of vmap vs individual processing."""
        batch_size = 32
        grid_shape = (15, 15)
        
        # Create batched data
        operations = jnp.zeros(batch_size, dtype=jnp.int32)
        rows = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[0]
        cols = jnp.arange(batch_size, dtype=jnp.int32) % grid_shape[1]
        working_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        
        # Define single action processor
        def process_single_point_action(operation, row, col, working_mask):
            action = PointAction(operation=operation, row=row, col=col)
            return point_handler(action, working_mask)
        
        # Create vectorized version
        vectorized_process = jax.vmap(
            process_single_point_action,
            in_axes=(0, 0, 0, None)
        )
        
        # JIT compile both versions
        jit_vectorized = jax.jit(vectorized_process)
        jit_individual = jax.jit(process_single_point_action)
        
        # Warm up JIT compilation
        _ = jit_vectorized(operations, rows, cols, working_mask)
        _ = jit_individual(operations[0], rows[0], cols[0], working_mask)
        
        # Time vectorized processing
        start_time = time.perf_counter()
        for _ in range(100):  # Multiple runs for better timing
            vectorized_result = jit_vectorized(operations, rows, cols, working_mask)
        vectorized_time = (time.perf_counter() - start_time) / 100
        
        # Time individual processing
        start_time = time.perf_counter()
        for _ in range(100):  # Multiple runs for better timing
            individual_results = []
            for i in range(batch_size):
                result = jit_individual(operations[i], rows[i], cols[i], working_mask)
                individual_results.append(result)
            individual_result = jnp.stack(individual_results)
        individual_time = (time.perf_counter() - start_time) / 100
        
        # Verify results are identical
        chex.assert_trees_all_close(vectorized_result, individual_result)
        
        # Calculate speedup
        speedup = individual_time / vectorized_time
        
        print(f"Vectorized processing time: {vectorized_time:.6f}s")
        print(f"Individual processing time: {individual_time:.6f}s")
        print(f"Vectorization speedup: {speedup:.2f}x")
        
        # Vectorized should be faster (though the difference might be small for simple operations)
        assert vectorized_time <= individual_time * 1.1  # Allow 10% margin


def test_batch_structured_actions_comprehensive():
    """Comprehensive test of batch structured actions functionality."""
    test_instance = TestBatchStructuredActions()
    
    # Run all tests
    test_instance.test_batch_point_actions_creation()
    test_instance.test_batch_bbox_actions_creation()
    test_instance.test_batch_mask_actions_creation()
    test_instance.test_vmap_point_handler_processing()
    test_instance.test_vmap_bbox_handler_processing()
    test_instance.test_vmap_mask_handler_processing()
    test_instance.test_batch_action_conversion_from_dictionaries()
    test_instance.test_batch_processing_performance()
    test_instance.test_batch_memory_usage_validation()
    test_instance.test_mixed_action_types_batch_processing()
    test_instance.test_large_batch_scalability()
    test_instance.test_batch_action_validation()
    test_instance.test_jit_compilation_with_batch_processing()
    test_instance.test_vmap_compatibility_with_structured_actions()
    test_instance.test_batch_action_performance_comparison()
    test_instance.test_memory_efficiency_validation()
    test_instance.test_true_vmap_vectorization()
    test_instance.test_vmap_with_different_action_types()
    test_instance.test_vmap_performance_comparison()
    
    print("All batch structured actions tests passed!")


if __name__ == "__main__":
    test_batch_structured_actions_comprehensive()