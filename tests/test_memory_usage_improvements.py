"""
Test memory usage improvements for action history system.

This test validates the memory-efficient action history implementation
that provides format-specific storage optimization.

Requirements tested:
- 3.6: Memory usage scales appropriately with history length
- 3.7: 99%+ memory reduction for point and bbox actions (vs mask format)
"""

import pytest
import jax.numpy as jnp
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.jax_types import (
    get_action_record_fields,
    get_selection_data_size,
)
from jaxarc.envs.action_history import HistoryConfig


class TestMemoryUsageImprovements:
    """Test suite for memory usage improvements in action history."""
    
    def create_test_state(
        self, 
        selection_format: str, 
        max_grid_height: int = 30, 
        max_grid_width: int = 30,
        history_length: int = 1000
    ) -> ArcEnvState:
        """Create a test state with format-specific action history sizing."""
        
        # Calculate action record fields for this format
        record_fields = get_action_record_fields(
            selection_format, max_grid_height, max_grid_width
        )
        
        # Create action history with format-specific sizing
        action_history = jnp.zeros((history_length, record_fields), dtype=jnp.float32)
        
        # Create minimal state for testing
        test_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        test_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        
        # Create dummy task data
        dummy_task = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            output_grids_examples=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            output_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            test_input_grids=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            true_test_output_grids=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            true_test_output_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_train_pairs=1,
            num_test_pairs=1,
            task_index=jnp.array(0),
        )
        
        state = ArcEnvState(
            task_data=dummy_task,
            working_grid=test_grid,
            working_grid_mask=test_mask,
            target_grid=test_grid,
            target_grid_mask=test_mask,
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros((5, 5), dtype=jnp.bool_),
            clipboard=test_grid,
            similarity_score=jnp.array(0.0),
            episode_mode=jnp.array(0),
            available_demo_pairs=jnp.ones(1, dtype=jnp.bool_),
            available_test_pairs=jnp.ones(1, dtype=jnp.bool_),
            demo_completion_status=jnp.zeros(1, dtype=jnp.bool_),
            test_completion_status=jnp.zeros(1, dtype=jnp.bool_),
            action_history=action_history,
            action_history_length=jnp.array(0),
            action_history_write_pos=jnp.array(0),
            allowed_operations_mask=jnp.ones(42, dtype=jnp.bool_),
        )
        
        return state
    
    def test_format_specific_field_counts(self):
        """Test that different formats use the expected number of fields."""
        
        # Test field counts for 30x30 grid (standard ARC size)
        point_fields = get_action_record_fields("point", 30, 30)
        bbox_fields = get_action_record_fields("bbox", 30, 30)
        mask_fields = get_action_record_fields("mask", 30, 30)
        
        # Point: 2 coordinates + 4 metadata = 6 fields
        assert point_fields == 6, f"Point format should use 6 fields, got {point_fields}"
        
        # Bbox: 4 coordinates + 4 metadata = 8 fields
        assert bbox_fields == 8, f"Bbox format should use 8 fields, got {bbox_fields}"
        
        # Mask: 900 mask values + 4 metadata = 904 fields
        assert mask_fields == 904, f"Mask format should use 904 fields for 30x30, got {mask_fields}"
        
        # Verify ordering: point < bbox < mask
        assert point_fields < bbox_fields < mask_fields
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales linearly with history length."""
        
        history_lengths = [100, 200, 500, 1000]
        format_name = "point"
        grid_size = 10
        
        memory_usages = []
        
        for hist_len in history_lengths:
            state = self.create_test_state(format_name, grid_size, grid_size, hist_len)
            memory_bytes = state.action_history.nbytes
            memory_usages.append(memory_bytes)
        
        # Check that memory scales linearly
        base_memory = memory_usages[0]
        base_length = history_lengths[0]
        
        for i, (hist_len, memory) in enumerate(zip(history_lengths[1:], memory_usages[1:]), 1):
            expected_memory = base_memory * (hist_len / base_length)
            scaling_factor = memory / expected_memory
            
            # Should be very close to 1.0 (perfect linear scaling)
            assert abs(scaling_factor - 1.0) < 0.01, (
                f"Memory scaling not linear: expected {expected_memory}, got {memory} "
                f"(scaling factor: {scaling_factor})"
            )
    
    def test_99_percent_memory_savings(self):
        """Test that point and bbox formats achieve 99%+ memory savings vs mask format."""
        
        # Test with large grid size where savings should be maximized
        grid_height, grid_width = 30, 30
        history_length = 1000
        
        # Create states for each format
        point_state = self.create_test_state("point", grid_height, grid_width, history_length)
        bbox_state = self.create_test_state("bbox", grid_height, grid_width, history_length)
        mask_state = self.create_test_state("mask", grid_height, grid_width, history_length)
        
        # Get memory usage
        point_memory = point_state.action_history.nbytes
        bbox_memory = bbox_state.action_history.nbytes
        mask_memory = mask_state.action_history.nbytes
        
        # Calculate savings
        point_savings = ((mask_memory - point_memory) / mask_memory) * 100
        bbox_savings = ((mask_memory - bbox_memory) / mask_memory) * 100
        
        # Verify 99%+ savings
        assert point_savings >= 99.0, (
            f"Point format should achieve 99%+ memory savings, got {point_savings:.1f}%"
        )
        assert bbox_savings >= 99.0, (
            f"Bbox format should achieve 99%+ memory savings, got {bbox_savings:.1f}%"
        )
        
        # Verify actual expected values
        # Point: 6 fields vs Mask: 904 fields = 99.34% savings
        expected_point_savings = ((904 - 6) / 904) * 100
        assert abs(point_savings - expected_point_savings) < 0.1
        
        # Bbox: 8 fields vs Mask: 904 fields = 99.11% savings  
        expected_bbox_savings = ((904 - 8) / 904) * 100
        assert abs(bbox_savings - expected_bbox_savings) < 0.1
    
    def test_action_history_functionality(self):
        """Test that action history functionality works correctly with different formats."""
        
        for format_name in ["point", "bbox", "mask"]:
            state = self.create_test_state(format_name, 5, 5, 100)
            
            # Create format-specific selection data
            if format_name == "point":
                selection_data = jnp.array([2.0, 3.0])  # [row, col]
            elif format_name == "bbox":
                selection_data = jnp.array([1.0, 1.0, 3.0, 4.0])  # [r1, c1, r2, c2]
            else:  # mask
                selection_data = jnp.zeros(25)  # 5x5 flattened
                selection_data = selection_data.at[7].set(1.0)  # Select one cell
            
            # Add action to history
            new_state = state.add_action_to_history(
                operation_id=5,
                selection_data=selection_data,
                timestamp=10.0,
                pair_index=0
            )
            
            # Verify action was added
            assert int(new_state.action_history_length) == 1
            
            # Retrieve action
            retrieved_action = new_state.get_action_from_history(0)
            assert retrieved_action['operation_id'] == 5
            assert retrieved_action['timestamp'] == 10.0
            assert retrieved_action['pair_index'] == 0
            assert retrieved_action['valid'] == True
            
            # Test history summary
            summary = new_state.get_action_history_summary()
            assert summary['length'] == 1
            assert summary['record_fields'] == state.action_history.shape[1]
    
    def test_memory_configuration_system(self):
        """Test the HistoryConfig memory estimation system."""
        
        # Test memory estimation for different configurations
        config = HistoryConfig(max_history_length=1000, store_selection_data=True)
        
        # Test different formats
        point_usage = config.estimate_memory_usage("point", 30, 30)
        bbox_usage = config.estimate_memory_usage("bbox", 30, 30)
        mask_usage = config.estimate_memory_usage("mask", 30, 30)
        
        # Verify memory ordering
        assert point_usage['total_bytes'] < bbox_usage['total_bytes'] < mask_usage['total_bytes']
        
        # Test disabled configuration
        disabled_config = HistoryConfig(enabled=False)
        disabled_usage = disabled_config.estimate_memory_usage("mask", 30, 30)
        assert disabled_usage['total_bytes'] == 0
        
        # Test configuration without selection data
        no_selection_config = HistoryConfig(store_selection_data=False)
        point_no_sel = no_selection_config.estimate_memory_usage("point", 30, 30)
        bbox_no_sel = no_selection_config.estimate_memory_usage("bbox", 30, 30)
        
        # Without selection data, point and bbox should use same memory (just metadata)
        assert point_no_sel['total_bytes'] == bbox_no_sel['total_bytes']
    
    def test_edge_cases(self):
        """Test edge cases for memory calculations."""
        
        # Test very small grids
        point_1x1 = get_action_record_fields("point", 1, 1)
        mask_1x1 = get_action_record_fields("mask", 1, 1)
        
        # For 1x1 grids, mask (5 fields) can be more efficient than point (6 fields)
        # This is expected behavior
        assert point_1x1 == 6  # 2 + 4 metadata
        assert mask_1x1 == 5   # 1 + 4 metadata
        
        # Test very large grids
        point_100x100 = get_action_record_fields("point", 100, 100)
        bbox_100x100 = get_action_record_fields("bbox", 100, 100)
        mask_100x100 = get_action_record_fields("mask", 100, 100)
        
        # Point and bbox should remain constant regardless of grid size
        assert point_100x100 == 6
        assert bbox_100x100 == 8
        assert mask_100x100 == 10004  # 100*100 + 4
        
        # Calculate savings for very large grid
        point_savings = ((mask_100x100 - point_100x100) / mask_100x100) * 100
        bbox_savings = ((mask_100x100 - bbox_100x100) / mask_100x100) * 100
        
        # For very large grids, savings should be even higher
        assert point_savings > 99.9
        assert bbox_savings > 99.9
    
    def test_grid_size_invariance(self):
        """Test that point and bbox formats are invariant to grid size."""
        
        grid_sizes = [1, 5, 10, 20, 30, 50, 100]
        
        for format_name in ["point", "bbox"]:
            expected_fields = 6 if format_name == "point" else 8
            
            for grid_size in grid_sizes:
                actual_fields = get_action_record_fields(format_name, grid_size, grid_size)
                assert actual_fields == expected_fields, (
                    f"{format_name} fields should be invariant to grid size: "
                    f"expected {expected_fields}, got {actual_fields} for {grid_size}x{grid_size}"
                )
    
    def test_selection_data_size_calculations(self):
        """Test that selection data size calculations are correct."""
        
        test_cases = [
            ("point", 5, 5, 2),      # Point always uses 2 fields
            ("point", 30, 30, 2),    # Point always uses 2 fields
            ("bbox", 5, 5, 4),       # Bbox always uses 4 fields
            ("bbox", 30, 30, 4),     # Bbox always uses 4 fields
            ("mask", 5, 5, 25),      # Mask uses height * width fields
            ("mask", 10, 10, 100),   # Mask uses height * width fields
            ("mask", 30, 30, 900),   # Mask uses height * width fields
        ]
        
        for format_name, height, width, expected_size in test_cases:
            actual_size = get_selection_data_size(format_name, height, width)
            assert actual_size == expected_size, (
                f"Selection data size mismatch for {format_name} {height}x{width}: "
                f"expected {expected_size}, got {actual_size}"
            )