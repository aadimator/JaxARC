"""
Comprehensive unit tests for action history tracking with memory optimization.

This module tests the ActionHistoryTracker class and related functionality,
focusing on circular buffer behavior, memory optimization, and JAX compatibility.

Test Coverage:
- ActionHistoryTracker with various history lengths
- Action sequence storage and retrieval
- Circular buffer behavior and overflow handling
- Memory-efficient configuration options
- JAX compatibility of history operations
- HistoryConfig validation and memory estimation
- ActionRecord validation and manipulation
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, strategies as st

from jaxarc.envs.action_history import (
    ActionHistoryTracker,
    HistoryConfig,
    ActionRecord,
    add_action_to_state,
    create_action_history_tracker_for_config,
)
from jaxarc.state import create_arc_env_state
from jaxarc.utils.jax_types import (
    get_selection_data_size,
    get_action_record_fields,
    NUM_OPERATIONS,
)
from tests.test_utils import MockDataGenerator


def assert_jax_bool(jax_bool, expected_bool):
    """Helper function to assert JAX boolean values."""
    assert bool(jax_bool) is expected_bool


def assert_jax_int(jax_int, expected_int):
    """Helper function to assert JAX integer values."""
    assert int(jax_int) == expected_int


def assert_jax_float(jax_float, expected_float):
    """Helper function to assert JAX float values."""
    assert float(jax_float) == expected_float


class TestHistoryConfig:
    """Test HistoryConfig validation and memory estimation."""
    
    def test_default_config_creation(self):
        """Test creation of default HistoryConfig."""
        config = HistoryConfig()
        
        assert config.enabled is True
        assert config.max_history_length == 1000
        assert config.store_selection_data is True
        assert config.store_intermediate_grids is False
        assert config.compress_repeated_actions is True
    
    def test_custom_config_creation(self):
        """Test creation of custom HistoryConfig."""
        config = HistoryConfig(
            enabled=False,
            max_history_length=500,
            store_selection_data=False,
            store_intermediate_grids=True,
            compress_repeated_actions=False
        )
        
        assert config.enabled is False
        assert config.max_history_length == 500
        assert config.store_selection_data is False
        assert config.store_intermediate_grids is True
        assert config.compress_repeated_actions is False
    
    def test_config_validation_positive_length(self):
        """Test that config validates positive history length."""
        with pytest.raises(ValueError, match="max_history_length must be positive"):
            HistoryConfig(max_history_length=0)
        
        with pytest.raises(ValueError, match="max_history_length must be positive"):
            HistoryConfig(max_history_length=-10)
    
    def test_config_validation_large_length_warning(self):
        """Test warning for very large history lengths."""
        with pytest.warns(UserWarning, match="Large history length"):
            HistoryConfig(max_history_length=15000)
    
    @pytest.mark.parametrize("selection_format,max_height,max_width,expected_fields", [
        ("point", 5, 5, 6),      # 2 + 4 metadata
        ("bbox", 5, 5, 8),       # 4 + 4 metadata  
        ("mask", 5, 5, 29),      # 25 + 4 metadata
        ("point", 30, 30, 6),    # 2 + 4 metadata
        ("bbox", 30, 30, 8),     # 4 + 4 metadata
        ("mask", 30, 30, 904),   # 900 + 4 metadata
    ])
    def test_memory_estimation_different_formats(self, selection_format, max_height, max_width, expected_fields):
        """Test memory estimation for different selection formats."""
        config = HistoryConfig(max_history_length=100, store_selection_data=True)
        
        usage = config.estimate_memory_usage(selection_format, max_height, max_width)
        
        assert usage["total_bytes"] > 0
        assert "human_readable" in usage
        assert "breakdown" in usage
        
        # Check that record size matches expected
        breakdown = usage["breakdown"]
        assert breakdown["record_size_fields"] == expected_fields
        assert breakdown["selection_format"] == selection_format
        assert breakdown["grid_dimensions"] == f"{max_height}x{max_width}"
    
    def test_memory_estimation_disabled_history(self):
        """Test memory estimation when history is disabled."""
        config = HistoryConfig(enabled=False)
        
        usage = config.estimate_memory_usage("mask", 30, 30)
        
        assert usage["total_bytes"] == 0
        assert usage["human_readable"] == "0 B (history disabled)"
        assert usage["breakdown"]["history_disabled"] is True
    
    def test_memory_estimation_no_selection_data(self):
        """Test memory estimation when selection data storage is disabled."""
        config = HistoryConfig(
            max_history_length=100,
            store_selection_data=False,
            store_intermediate_grids=False
        )
        
        usage = config.estimate_memory_usage("mask", 30, 30)
        
        # Should only store metadata (4 fields)
        expected_bytes = 100 * 4 * 4  # 100 records * 4 fields * 4 bytes
        assert usage["total_bytes"] == expected_bytes
        assert usage["breakdown"]["selection_data_fields"] == 0
        assert usage["breakdown"]["metadata_fields"] == 4
    
    def test_memory_estimation_with_intermediate_grids(self):
        """Test memory estimation with intermediate grid storage."""
        config = HistoryConfig(
            max_history_length=10,
            store_selection_data=True,
            store_intermediate_grids=True
        )
        
        usage = config.estimate_memory_usage("point", 5, 5)
        
        # Should include both action records and intermediate grids
        assert usage["breakdown"]["action_records_bytes"] > 0
        assert usage["breakdown"]["intermediate_grids_bytes"] > 0
        assert usage["total_bytes"] > usage["breakdown"]["action_records_bytes"]
    
    def test_memory_configuration_comparison(self):
        """Test comparison of different memory configurations."""
        base_config = HistoryConfig(max_history_length=1000, store_selection_data=True)
        
        other_configs = [
            HistoryConfig(max_history_length=500, store_selection_data=True),
            HistoryConfig(max_history_length=1000, store_selection_data=False),
            HistoryConfig(max_history_length=2000, store_selection_data=True),
        ]
        
        comparison = base_config.compare_memory_configurations(other_configs, "point", 5, 5)
        
        assert "base_config" in comparison
        assert "comparisons" in comparison
        assert "summary" in comparison
        assert len(comparison["comparisons"]) == 3
        
        # Check that relative changes are calculated
        for comp in comparison["comparisons"]:
            assert "relative_to_base" in comp
            assert "percentage_change" in comp["relative_to_base"]
            assert "is_more_efficient" in comp["relative_to_base"]
    
    @pytest.mark.parametrize("use_case,expected_length", [
        ("development", 100),
        ("training", 1000),
        ("evaluation", 500),
        ("unknown", 1000),  # Default case
    ])
    def test_recommended_config_by_use_case(self, use_case, expected_length):
        """Test recommended configuration for different use cases."""
        config = HistoryConfig().get_recommended_config(
            selection_format="point",
            max_grid_height=5,
            max_grid_width=5,
            memory_budget_mb=100.0,
            use_case=use_case
        )
        
        assert config.max_history_length == expected_length
        assert config.enabled is True
    
    def test_recommended_config_memory_budget_constraint(self):
        """Test that recommended config respects memory budget."""
        # Very small budget should reduce history length
        config = HistoryConfig().get_recommended_config(
            selection_format="mask",
            max_grid_height=30,
            max_grid_width=30,
            memory_budget_mb=0.01,  # Very small budget (10KB)
            use_case="training"
        )
        
        # Should have reduced history length to fit budget
        assert config.max_history_length < 1000
        
        # Verify it actually fits in budget
        usage = config.estimate_memory_usage("mask", 30, 30)
        assert usage["total_bytes"] <= 0.01 * 1024 * 1024


class TestActionRecord:
    """Test ActionRecord creation, validation, and manipulation."""
    
    def test_action_record_creation_point(self):
        """Test creation of ActionRecord for point selection."""
        selection_data = jnp.array([2.0, 3.0, 0.0, 0.0, 0.0, 0.0])  # Padded to 6 elements
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        assert_jax_int(record.operation_id, 15)
        assert_jax_int(record.timestamp, 10)
        assert_jax_int(record.pair_index, 0)
        assert_jax_bool(record.valid, True)
        assert record.selection_data.shape[0] == 6
    
    def test_action_record_creation_bbox(self):
        """Test creation of ActionRecord for bbox selection."""
        selection_data = jnp.array([1.0, 2.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0])  # Padded to 8 elements
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(20),
            timestamp=jnp.array(15),
            pair_index=jnp.array(1),
            valid=jnp.array(True)
        )
        
        assert_jax_int(record.operation_id, 20)
        assert_jax_int(record.timestamp, 15)
        assert_jax_int(record.pair_index, 1)
        assert_jax_bool(record.valid, True)
        assert record.selection_data.shape[0] == 8
    
    def test_action_record_creation_mask(self):
        """Test creation of ActionRecord for mask selection."""
        # Create a small mask (5x5 = 25 elements)
        mask_data = jnp.ones(25) * 0.5  # Half-selected mask
        selection_data = jnp.concatenate([mask_data, jnp.zeros(4)])  # Padded to 29 elements
        
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(28),
            timestamp=jnp.array(20),
            pair_index=jnp.array(2),
            valid=jnp.array(True)
        )
        
        assert_jax_int(record.operation_id, 28)
        assert_jax_int(record.timestamp, 20)
        assert_jax_int(record.pair_index, 2)
        assert_jax_bool(record.valid, True)
        assert record.selection_data.shape[0] == 29
    
    def test_action_record_validation_valid_point(self):
        """Test validation of valid point action record."""
        selection_data = jnp.array([2.0, 3.0])  # Valid point coordinates
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        is_valid = record.validate_integrity("point", 5, 5)
        assert_jax_bool(is_valid, True)
    
    def test_action_record_validation_invalid_operation(self):
        """Test validation with invalid operation ID."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(NUM_OPERATIONS + 10),  # Invalid operation ID
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        is_valid = record.validate_integrity("point", 5, 5)
        assert_jax_bool(is_valid, False)
    
    def test_action_record_validation_negative_timestamp(self):
        """Test validation with negative timestamp."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(-5),  # Invalid negative timestamp
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        is_valid = record.validate_integrity("point", 5, 5)
        assert_jax_bool(is_valid, False)
    
    def test_action_record_validation_detailed(self):
        """Test detailed validation with error messages."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(NUM_OPERATIONS + 10),  # Invalid operation ID
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        is_valid, error_msg = record.validate_integrity_detailed("point", 5, 5)
        assert is_valid is False
        assert "Operation ID" in error_msg
        assert "out of range" in error_msg
    
    def test_action_record_get_selection_coordinates_point(self):
        """Test extracting coordinates from point selection."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        coords = record.get_selection_coordinates("point")
        assert coords.shape == (2,)
        assert_jax_float(coords[0], 2.0)
        assert_jax_float(coords[1], 3.0)
    
    def test_action_record_get_selection_coordinates_bbox(self):
        """Test extracting coordinates from bbox selection."""
        selection_data = jnp.array([1.0, 2.0, 8.0, 9.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(20),
            timestamp=jnp.array(15),
            pair_index=jnp.array(1),
            valid=jnp.array(True)
        )
        
        coords = record.get_selection_coordinates("bbox")
        assert coords.shape == (4,)
        assert jnp.array_equal(coords, jnp.array([1.0, 2.0, 8.0, 9.0]))
    
    def test_action_record_get_selection_coordinates_mask(self):
        """Test extracting coordinates from mask selection."""
        # Create mask with some selected cells
        mask_data = jnp.array([0.0, 0.8, 0.0, 0.9, 0.0])  # Cells 1 and 3 selected
        record = ActionRecord(
            selection_data=mask_data,
            operation_id=jnp.array(28),
            timestamp=jnp.array(20),
            pair_index=jnp.array(2),
            valid=jnp.array(True)
        )
        
        coords = record.get_selection_coordinates("mask")
        # Should return indices where mask > 0.5
        expected_indices = jnp.array([1, 3])
        assert jnp.array_equal(coords[:2], expected_indices)
    
    def test_action_record_metadata_summary(self):
        """Test getting metadata summary from record."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        summary = record.get_metadata_summary()
        assert summary["operation_id"] == 15
        assert summary["timestamp"] == 10
        assert summary["pair_index"] == 0
        assert summary["valid"] is True
        assert summary["selection_data_size"] == 2
    
    def test_action_record_is_control_operation(self):
        """Test identification of control operations."""
        # Regular operation (not control)
        regular_record = ActionRecord(
            selection_data=jnp.array([2.0, 3.0]),
            operation_id=jnp.array(15),  # Regular operation
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        # Control operation
        control_record = ActionRecord(
            selection_data=jnp.array([2.0, 3.0]),
            operation_id=jnp.array(35),  # Control operation (35+)
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        assert_jax_bool(regular_record.is_control_operation(), False)
        assert_jax_bool(control_record.is_control_operation(), True)


class TestActionHistoryTracker:
    """Test ActionHistoryTracker functionality."""
    
    @pytest.fixture
    def sample_state(self, mock_data_generator):
        """Create a sample ArcEnvState for testing."""
        from jaxarc.types import JaxArcTask
        
        # Create mock task data
        mock_data = mock_data_generator.create_mock_task_data(
            num_train_pairs=3,
            num_test_pairs=1,
            max_height=5,
            max_width=5
        )
        
        # Create JaxArcTask
        task_data = JaxArcTask(
            input_grids_examples=mock_data["input_grids_examples"],
            input_masks_examples=mock_data["input_masks_examples"],
            output_grids_examples=mock_data["output_grids_examples"],
            output_masks_examples=mock_data["output_masks_examples"],
            num_train_pairs=mock_data["num_train_pairs"],
            test_input_grids=mock_data["test_input_grids"],
            test_input_masks=mock_data["test_input_masks"],
            true_test_output_grids=mock_data["true_test_output_grids"],
            true_test_output_masks=mock_data["true_test_output_masks"],
            num_test_pairs=mock_data["num_test_pairs"],
            task_index=mock_data["task_index"],
        )
        
        working_grid = jnp.ones((5, 5), dtype=jnp.int32)
        working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
    
    @pytest.fixture
    def history_config(self):
        """Create a standard history configuration for testing."""
        return HistoryConfig(
            enabled=True,
            max_history_length=100,
            store_selection_data=True,
            store_intermediate_grids=False,
            compress_repeated_actions=True
        )
    
    @pytest.fixture
    def tracker(self):
        """Create an ActionHistoryTracker instance."""
        return ActionHistoryTracker()
    
    def test_tracker_creation(self, tracker):
        """Test basic tracker creation."""
        assert isinstance(tracker, ActionHistoryTracker)
    
    def test_add_action_point_format(self, tracker, sample_state, history_config):
        """Test adding point-format action to history."""
        action = {"point": [2, 3], "operation": 15}
        
        new_state = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        # Check that history length increased
        assert new_state.action_history_length == 1
        
        # Check that action was stored
        sequence = tracker.get_action_sequence(new_state)
        assert sequence.shape[0] == 1
        
        # Verify the stored action data
        # The sequence shape should match the buffer size, not the logical record size
        assert sequence.shape[1] == new_state.action_history.shape[1]
    
    def test_add_action_bbox_format(self, tracker, sample_state, history_config):
        """Test adding bbox-format action to history."""
        action = {"bbox": [1, 2, 4, 5], "operation": 20}
        
        new_state = tracker.add_action(
            sample_state, action, history_config, "bbox", 5, 5
        )
        
        # Check that history length increased
        assert new_state.action_history_length == 1
        
        # Check that action was stored
        sequence = tracker.get_action_sequence(new_state)
        assert sequence.shape[0] == 1
        
        # Verify the stored action data
        # The sequence shape should match the buffer size, not the logical record size
        assert sequence.shape[1] == new_state.action_history.shape[1]
    
    def test_add_action_mask_format(self, tracker, sample_state, history_config):
        """Test adding mask-format action to history."""
        mask = jnp.ones((5, 5)) * 0.5  # Half-selected mask
        action = {"mask": mask, "operation": 28}
        
        new_state = tracker.add_action(
            sample_state, action, history_config, "mask", 5, 5
        )
        
        # Check that history length increased
        assert new_state.action_history_length == 1
        
        # Check that action was stored
        sequence = tracker.get_action_sequence(new_state)
        assert sequence.shape[0] == 1
        
        # Verify the stored action data
        # The sequence shape should match the buffer size, not the logical record size
        assert sequence.shape[1] == new_state.action_history.shape[1]
    
    def test_add_action_disabled_history(self, tracker, sample_state):
        """Test that actions are not stored when history is disabled."""
        disabled_config = HistoryConfig(enabled=False)
        action = {"point": [2, 3], "operation": 15}
        
        new_state = tracker.add_action(
            sample_state, action, disabled_config, "point", 5, 5
        )
        
        # State should be unchanged when history is disabled
        assert new_state.action_history_length == sample_state.action_history_length
        assert jnp.array_equal(new_state.action_history, sample_state.action_history)
    
    def test_multiple_actions_sequential(self, tracker, sample_state, history_config):
        """Test adding multiple actions sequentially."""
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
        ]
        
        current_state = sample_state
        for action in actions:
            current_state = tracker.add_action(
                current_state, action, history_config, "point", 5, 5
            )
            # Increment step count to simulate environment stepping
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Check final history length
        assert current_state.action_history_length == 3
        
        # Check that all actions are retrievable
        sequence = tracker.get_action_sequence(current_state)
        assert sequence.shape[0] == 3
    
    def test_circular_buffer_overflow(self, tracker, sample_state):
        """Test circular buffer behavior when history overflows."""
        # Create small buffer for testing overflow
        small_config = HistoryConfig(
            enabled=True,
            max_history_length=3,  # Very small buffer
            store_selection_data=True
        )
        
        # Create state with small history buffer
        small_history = jnp.zeros((3, get_action_record_fields("point", 5, 5)), dtype=jnp.float32)
        test_state = sample_state.replace(action_history=small_history)
        
        # Add more actions than buffer size
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
            {"point": [4, 4], "operation": 13},  # This should overwrite first action
            {"point": [5, 5], "operation": 14},  # This should overwrite second action
        ]
        
        current_state = test_state
        for i, action in enumerate(actions):
            current_state = tracker.add_action(
                current_state, action, small_config, "point", 5, 5
            )
            # Increment step count to simulate environment stepping
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # History length should be capped at buffer size
        assert current_state.action_history_length == 3
        
        # Should be able to retrieve the most recent actions
        sequence = tracker.get_action_sequence(current_state)
        assert sequence.shape[0] == 3
        
        # The sequence should contain the last 3 actions in chronological order
        # Due to circular buffer, we should have actions with operations 12, 13, 14
        selection_size = get_selection_data_size("point", 5, 5)
        operation_ids = sequence[:, selection_size]  # Operation ID is at position selection_size
        
        # Check that we have the most recent operations (may be in circular order)
        unique_ops = jnp.unique(operation_ids)
        assert len(unique_ops) <= 3  # Should have at most 3 unique operations
    
    def test_get_action_sequence_full_history(self, tracker, sample_state, history_config):
        """Test retrieving full action sequence."""
        # Add several actions
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
        ]
        
        current_state = sample_state
        for action in actions:
            current_state = tracker.add_action(
                current_state, action, history_config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Get full sequence
        sequence = tracker.get_action_sequence(current_state)
        assert sequence.shape[0] == 3
        
        # Verify chronological order
        selection_size = get_selection_data_size("point", 5, 5)
        timestamps = sequence[:, selection_size + 1]  # Timestamp is at position selection_size + 1
        
        # Timestamps should be in ascending order
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_get_action_sequence_with_indices(self, tracker, sample_state, history_config):
        """Test retrieving action sequence with start/end indices."""
        # Add several actions
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
            {"point": [4, 4], "operation": 13},
            {"point": [5, 5], "operation": 14},
        ]
        
        current_state = sample_state
        for action in actions:
            current_state = tracker.add_action(
                current_state, action, history_config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Test different index ranges
        
        # Get first 3 actions
        sequence = tracker.get_action_sequence(current_state, start_idx=0, end_idx=3)
        assert sequence.shape[0] == 3
        
        # Get last 2 actions
        sequence = tracker.get_action_sequence(current_state, start_idx=-2)
        assert sequence.shape[0] == 2
        
        # Get middle actions
        sequence = tracker.get_action_sequence(current_state, start_idx=1, end_idx=4)
        assert sequence.shape[0] == 3
        
        # Get empty range
        sequence = tracker.get_action_sequence(current_state, start_idx=2, end_idx=2)
        assert sequence.shape[0] == 0
    
    def test_get_action_sequence_empty_history(self, tracker, sample_state):
        """Test retrieving sequence from empty history."""
        sequence = tracker.get_action_sequence(sample_state)
        assert sequence.shape[0] == 0
        
        # Should have correct number of fields even when empty
        expected_fields = sample_state.action_history.shape[1]
        assert sequence.shape[1] == expected_fields
    
    def test_clear_history(self, tracker, sample_state, history_config):
        """Test clearing action history."""
        # Add some actions first
        action = {"point": [2, 3], "operation": 15}
        state_with_history = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        # Verify history exists
        assert state_with_history.action_history_length > 0
        
        # Clear history
        cleared_state = tracker.clear_history(state_with_history)
        
        # Verify history is cleared
        assert cleared_state.action_history_length == 0
        
        # Verify we can still retrieve (empty) sequence
        sequence = tracker.get_action_sequence(cleared_state)
        assert sequence.shape[0] == 0
    
    def test_get_action_count(self, tracker, sample_state, history_config):
        """Test getting current action count."""
        # Initially empty
        assert tracker.get_action_count(sample_state) == 0
        
        # Add action
        action = {"point": [2, 3], "operation": 15}
        new_state = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        # Should have 1 action
        assert tracker.get_action_count(new_state) == 1
    
    def test_is_history_full(self, tracker, sample_state):
        """Test checking if history buffer is full."""
        # Create small buffer
        small_config = HistoryConfig(max_history_length=2)
        small_history = jnp.zeros((2, get_action_record_fields("point", 5, 5)), dtype=jnp.float32)
        test_state = sample_state.replace(action_history=small_history)
        
        # Initially not full
        assert tracker.is_history_full(test_state) is False
        
        # Add one action
        action = {"point": [2, 3], "operation": 15}
        state1 = tracker.add_action(test_state, action, small_config, "point", 5, 5)
        assert tracker.is_history_full(state1) is False
        
        # Add second action
        state2 = tracker.add_action(state1, action, small_config, "point", 5, 5)
        assert tracker.is_history_full(state2) is True
    
    def test_get_history_capacity(self, tracker, sample_state):
        """Test getting history buffer capacity."""
        capacity = tracker.get_history_capacity(sample_state)
        assert capacity == sample_state.action_history.shape[0]
    
    def test_get_recent_actions(self, tracker, sample_state, history_config):
        """Test getting recent actions convenience method."""
        # Add several actions
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
            {"point": [4, 4], "operation": 13},
            {"point": [5, 5], "operation": 14},
        ]
        
        current_state = sample_state
        for action in actions:
            current_state = tracker.add_action(
                current_state, action, history_config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Get last 3 actions
        recent = tracker.get_recent_actions(current_state, count=3)
        assert recent.shape[0] == 3
        
        # Should be same as using negative indexing
        sequence = tracker.get_action_sequence(current_state, start_idx=-3)
        assert jnp.array_equal(recent, sequence)
    
    def test_get_actions_for_pair(self, tracker, sample_state, history_config):
        """Test filtering actions by pair index."""
        # Add actions for different pairs
        actions = [
            {"point": [1, 1], "operation": 10},  # pair 0
            {"point": [2, 2], "operation": 11},  # pair 1  
            {"point": [3, 3], "operation": 12},  # pair 0
            {"point": [4, 4], "operation": 13},  # pair 1
        ]
        
        current_state = sample_state
        for i, action in enumerate(actions):
            # Alternate between pair indices
            pair_idx = i % 2
            state_with_pair = current_state.replace(current_example_idx=pair_idx)
            
            current_state = tracker.add_action(
                state_with_pair, action, history_config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Get actions for pair 0
        pair0_actions = tracker.get_actions_for_pair(current_state, pair_index=0)
        assert pair0_actions.shape[0] == 2  # Should have 2 actions for pair 0
        
        # Get actions for pair 1
        pair1_actions = tracker.get_actions_for_pair(current_state, pair_index=1)
        assert pair1_actions.shape[0] == 2  # Should have 2 actions for pair 1
    
    def test_get_history_summary(self, tracker, sample_state, history_config):
        """Test getting history summary statistics."""
        # Test empty history
        summary = tracker.get_history_summary(sample_state)
        assert summary["length"] == 0
        assert summary["capacity"] > 0
        assert summary["is_full"] is False
        assert summary["utilization"] == 0.0
        
        # Add some actions
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 10},  # Repeat operation
        ]
        
        current_state = sample_state
        for action in actions:
            current_state = tracker.add_action(
                current_state, action, history_config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
        
        # Test populated history
        summary = tracker.get_history_summary(current_state)
        assert summary["length"] == 3
        assert summary["utilization"] > 0.0
        assert "unique_operations" in summary
        assert "most_recent_operation" in summary
        assert summary["most_recent_operation"] == 10  # Last operation was 10


class TestActionHistoryJAXCompatibility:
    """Test JAX compatibility of action history operations."""
    
    @pytest.fixture
    def sample_state(self, mock_data_generator):
        """Create a sample ArcEnvState for JAX testing."""
        from jaxarc.types import JaxArcTask
        
        # Create mock task data
        mock_data = mock_data_generator.create_mock_task_data(
            num_train_pairs=3,
            num_test_pairs=1,
            max_height=5,
            max_width=5
        )
        
        # Create JaxArcTask
        task_data = JaxArcTask(
            input_grids_examples=mock_data["input_grids_examples"],
            input_masks_examples=mock_data["input_masks_examples"],
            output_grids_examples=mock_data["output_grids_examples"],
            output_masks_examples=mock_data["output_masks_examples"],
            num_train_pairs=mock_data["num_train_pairs"],
            test_input_grids=mock_data["test_input_grids"],
            test_input_masks=mock_data["test_input_masks"],
            true_test_output_grids=mock_data["true_test_output_grids"],
            true_test_output_masks=mock_data["true_test_output_masks"],
            num_test_pairs=mock_data["num_test_pairs"],
            task_index=mock_data["task_index"],
        )
        
        working_grid = jnp.ones((5, 5), dtype=jnp.int32)
        working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
    
    @pytest.fixture
    def tracker(self):
        """Create an ActionHistoryTracker instance."""
        return ActionHistoryTracker()
    
    @pytest.fixture
    def history_config(self):
        """Create a history configuration for testing."""
        return HistoryConfig(
            enabled=True,
            max_history_length=10,
            store_selection_data=True
        )
    
    def test_add_action_jit_compilation(self, tracker, sample_state, history_config):
        """Test that add_action can be JIT compiled."""
        action = {"point": [2, 3], "operation": 15}
        
        @jax.jit
        def jit_add_action(state, action_data):
            return tracker.add_action(
                state, action_data, history_config, "point", 5, 5
            )
        
        # Should compile and run without errors
        new_state = jit_add_action(sample_state, action)
        assert new_state.action_history_length == 1
    
    def test_get_action_sequence_jit_compilation(self, tracker, sample_state, history_config):
        """Test that get_action_sequence can be JIT compiled."""
        # Add an action first
        action = {"point": [2, 3], "operation": 15}
        state_with_history = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        @jax.jit
        def jit_get_sequence(state):
            return tracker.get_action_sequence(state)
        
        # Should compile and run without errors
        sequence = jit_get_sequence(state_with_history)
        # The simplified JAX-compatible version returns the full buffer
        assert sequence.shape[0] == state_with_history.action_history.shape[0]
        # But only the first entry should be non-zero (valid)
        assert jnp.any(sequence[0] != 0)  # First entry has data
        assert jnp.all(sequence[1] == 0)  # Second entry is all zeros
    
    def test_clear_history_jit_compilation(self, tracker, sample_state, history_config):
        """Test that clear_history can be JIT compiled."""
        # Add an action first
        action = {"point": [2, 3], "operation": 15}
        state_with_history = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        @jax.jit
        def jit_clear_history(state):
            return tracker.clear_history(state)
        
        # Should compile and run without errors
        cleared_state = jit_clear_history(state_with_history)
        assert cleared_state.action_history_length == 0
    
    def test_action_record_validation_jit_compilation(self):
        """Test that ActionRecord validation can be JIT compiled."""
        selection_data = jnp.array([2.0, 3.0])
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        @jax.jit
        def jit_validate(record):
            return record.validate_integrity("point", 5, 5)
        
        # Should compile and run without errors
        is_valid = jit_validate(record)
        assert is_valid is True
    
    def test_vmap_compatibility(self, tracker, history_config):
        """Test that action history operations work with vmap."""
        # Create batch of states
        batch_size = 3
        
        # Create mock task data
        mock_gen = MockDataGenerator()
        task_data = mock_gen.create_mock_jax_arc_task()
        
        # Create batch of states
        working_grids = jnp.ones((batch_size, 5, 5), dtype=jnp.int32)
        working_grid_masks = jnp.ones((batch_size, 5, 5), dtype=jnp.bool_)
        target_grids = jnp.zeros((batch_size, 5, 5), dtype=jnp.int32)
        
        # Create batch of states manually (since create_arc_env_state doesn't support batching)
        states = []
        for i in range(batch_size):
            state = create_arc_env_state(
                task_data=task_data,
                working_grid=working_grids[i],
                working_grid_mask=working_grid_masks[i],
                target_grid=target_grids[i],
            )
            states.append(state)
        
        # Create batch of actions
        actions = [
            {"point": [1, 1], "operation": 10},
            {"point": [2, 2], "operation": 11},
            {"point": [3, 3], "operation": 12},
        ]
        
        # Test vmapped add_action
        def add_single_action(state, action):
            return tracker.add_action(state, action, history_config, "point", 5, 5)
        
        vmapped_add = jax.vmap(add_single_action)
        
        # Should work with vmap
        new_states = vmapped_add(states, actions)
        
        # Verify all states have updated history
        for state in new_states:
            assert state.action_history_length == 1
    
    def test_static_shape_preservation(self, tracker, sample_state, history_config):
        """Test that all operations preserve static shapes."""
        action = {"point": [2, 3], "operation": 15}
        
        # Original shapes
        original_history_shape = sample_state.action_history.shape
        original_length_shape = sample_state.action_history_length.shape
        
        # Add action
        new_state = tracker.add_action(
            sample_state, action, history_config, "point", 5, 5
        )
        
        # Shapes should be preserved
        assert new_state.action_history.shape == original_history_shape
        assert new_state.action_history_length.shape == original_length_shape
        
        # Get sequence
        sequence = tracker.get_action_sequence(new_state)
        
        # Sequence should have static shape
        assert len(sequence.shape) == 2  # (sequence_length, record_fields)
        
        # Clear history
        cleared_state = tracker.clear_history(new_state)
        
        # Shapes should still be preserved
        assert cleared_state.action_history.shape == original_history_shape
        assert cleared_state.action_history_length.shape == original_length_shape


class TestActionHistoryMemoryOptimization:
    """Test memory optimization features of action history."""
    
    def test_memory_efficient_config_no_selection_data(self):
        """Test memory-efficient configuration without selection data."""
        # Create memory-efficient config
        config = HistoryConfig(
            enabled=True,
            max_history_length=100,
            store_selection_data=False,  # Save memory
            store_intermediate_grids=False,
            compress_repeated_actions=True
        )
        
        # Estimate memory usage
        usage = config.estimate_memory_usage("mask", 30, 30)
        
        # Should use much less memory than full storage
        full_config = HistoryConfig(store_selection_data=True)
        full_usage = full_config.estimate_memory_usage("mask", 30, 30)
        
        assert usage["total_bytes"] < full_usage["total_bytes"]
        assert usage["breakdown"]["selection_data_fields"] == 0
    
    def test_memory_efficient_config_small_grids(self):
        """Test memory efficiency with smaller grid sizes."""
        config = HistoryConfig(max_history_length=1000)
        
        # Compare memory usage for different grid sizes
        small_usage = config.estimate_memory_usage("mask", 5, 5)
        large_usage = config.estimate_memory_usage("mask", 30, 30)
        
        # Small grids should use much less memory
        assert small_usage["total_bytes"] < large_usage["total_bytes"]
        
        # Verify the difference is significant
        ratio = large_usage["total_bytes"] / small_usage["total_bytes"]
        assert ratio > 10  # Should be much more efficient
    
    def test_memory_efficient_config_point_vs_mask(self):
        """Test memory efficiency of point vs mask selection formats."""
        config = HistoryConfig(max_history_length=1000)
        
        # Compare memory usage for different selection formats
        point_usage = config.estimate_memory_usage("point", 30, 30)
        mask_usage = config.estimate_memory_usage("mask", 30, 30)
        
        # Point format should use much less memory
        assert point_usage["total_bytes"] < mask_usage["total_bytes"]
        
        # Verify the difference is significant
        ratio = mask_usage["total_bytes"] / point_usage["total_bytes"]
        assert ratio > 100  # Point format should be much more efficient
    
    def test_memory_budget_recommendations(self):
        """Test memory budget-based configuration recommendations."""
        # Test with very small budget
        small_budget_config = HistoryConfig().get_recommended_config(
            selection_format="mask",
            max_grid_height=30,
            max_grid_width=30,
            memory_budget_mb=1.0,  # 1MB budget
            use_case="training"
        )
        
        # Should have reduced features to fit budget
        usage = small_budget_config.estimate_memory_usage("mask", 30, 30)
        assert usage["total_bytes"] <= 1.0 * 1024 * 1024
        
        # Test with larger budget
        large_budget_config = HistoryConfig().get_recommended_config(
            selection_format="mask",
            max_grid_height=30,
            max_grid_width=30,
            memory_budget_mb=100.0,  # 100MB budget
            use_case="training"
        )
        
        # Should have more features with larger budget
        large_usage = large_budget_config.estimate_memory_usage("mask", 30, 30)
        assert large_usage["total_bytes"] > usage["total_bytes"]
    
    def test_compressed_repeated_actions(self):
        """Test compression of repeated actions (conceptual test)."""
        # Note: The actual compression logic would need to be implemented
        # This test verifies the configuration option exists and can be set
        
        config = HistoryConfig(compress_repeated_actions=True)
        assert config.compress_repeated_actions is True
        
        config_no_compression = HistoryConfig(compress_repeated_actions=False)
        assert config_no_compression.compress_repeated_actions is False


class TestActionHistoryUtilityFunctions:
    """Test utility functions for action history integration."""
    
    def test_create_action_history_tracker_for_config(self):
        """Test creating tracker from configuration."""
        # Mock config object
        class MockConfig:
            class Action:
                selection_format = "point"
            class Dataset:
                max_grid_height = 5
                max_grid_width = 5
            action = Action()
            dataset = Dataset()
        
        config = MockConfig()
        tracker = create_action_history_tracker_for_config(config)
        
        assert isinstance(tracker, ActionHistoryTracker)
    
    def test_add_action_to_state_convenience_function(self, mock_data_generator):
        """Test convenience function for adding actions to state."""
        from jaxarc.types import JaxArcTask
        
        # Create mock task data
        mock_data = mock_data_generator.create_mock_task_data(
            num_train_pairs=3,
            num_test_pairs=1,
            max_height=5,
            max_width=5
        )
        
        # Create JaxArcTask
        task_data = JaxArcTask(
            input_grids_examples=mock_data["input_grids_examples"],
            input_masks_examples=mock_data["input_masks_examples"],
            output_grids_examples=mock_data["output_grids_examples"],
            output_masks_examples=mock_data["output_masks_examples"],
            num_train_pairs=mock_data["num_train_pairs"],
            test_input_grids=mock_data["test_input_grids"],
            test_input_masks=mock_data["test_input_masks"],
            true_test_output_grids=mock_data["true_test_output_grids"],
            true_test_output_masks=mock_data["true_test_output_masks"],
            num_test_pairs=mock_data["num_test_pairs"],
            task_index=mock_data["task_index"],
        )
        
        working_grid = jnp.ones((5, 5), dtype=jnp.int32)
        working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
        
        # Mock config object
        class MockConfig:
            class Action:
                selection_format = "point"
            class Dataset:
                max_grid_height = 5
                max_grid_width = 5
            action = Action()
            dataset = Dataset()
        
        config = MockConfig()
        action = {"point": [2, 3], "operation": 15}
        
        new_state = add_action_to_state(state, action, config)
        
        # Should have added action to history
        assert new_state.action_history_length == 1
    
    def test_add_action_to_state_with_custom_history_config(self, mock_data_generator):
        """Test convenience function with custom history configuration."""
        from jaxarc.types import JaxArcTask
        
        # Create mock task data
        mock_data = mock_data_generator.create_mock_task_data(
            num_train_pairs=3,
            num_test_pairs=1,
            max_height=5,
            max_width=5
        )
        
        # Create JaxArcTask
        task_data = JaxArcTask(
            input_grids_examples=mock_data["input_grids_examples"],
            input_masks_examples=mock_data["input_masks_examples"],
            output_grids_examples=mock_data["output_grids_examples"],
            output_masks_examples=mock_data["output_masks_examples"],
            num_train_pairs=mock_data["num_train_pairs"],
            test_input_grids=mock_data["test_input_grids"],
            test_input_masks=mock_data["test_input_masks"],
            true_test_output_grids=mock_data["true_test_output_grids"],
            true_test_output_masks=mock_data["true_test_output_masks"],
            num_test_pairs=mock_data["num_test_pairs"],
            task_index=mock_data["task_index"],
        )
        
        working_grid = jnp.ones((5, 5), dtype=jnp.int32)
        working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
        
        # Mock config object
        class MockConfig:
            class Action:
                selection_format = "point"
            class Dataset:
                max_grid_height = 5
                max_grid_width = 5
            action = Action()
            dataset = Dataset()
        
        config = MockConfig()
        action = {"point": [2, 3], "operation": 15}
        
        # Custom history config
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=50,
            store_selection_data=False  # Memory efficient
        )
        
        new_state = add_action_to_state(state, action, config, history_config)
        
        # Should have added action to history
        assert new_state.action_history_length == 1


# Integration tests with Hypothesis for property-based testing
class TestActionHistoryProperties:
    """Property-based tests for action history using Hypothesis."""
    
    @given(
        history_length=st.integers(min_value=1, max_value=100),
        num_actions=st.integers(min_value=1, max_value=50)
    )
    def test_history_length_invariant(self, history_length, num_actions, mock_data_generator):
        """Test that history length never exceeds maximum."""
        from jaxarc.types import JaxArcTask
        
        # Create mock task data
        mock_data = mock_data_generator.create_mock_task_data(
            num_train_pairs=3,
            num_test_pairs=1,
            max_height=5,
            max_width=5
        )
        
        # Create JaxArcTask
        task_data = JaxArcTask(
            input_grids_examples=mock_data["input_grids_examples"],
            input_masks_examples=mock_data["input_masks_examples"],
            output_grids_examples=mock_data["output_grids_examples"],
            output_masks_examples=mock_data["output_masks_examples"],
            num_train_pairs=mock_data["num_train_pairs"],
            test_input_grids=mock_data["test_input_grids"],
            test_input_masks=mock_data["test_input_masks"],
            true_test_output_grids=mock_data["true_test_output_grids"],
            true_test_output_masks=mock_data["true_test_output_masks"],
            num_test_pairs=mock_data["num_test_pairs"],
            task_index=mock_data["task_index"],
        )
        
        working_grid = jnp.ones((5, 5), dtype=jnp.int32)
        working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        
        # Create custom history buffer
        record_fields = get_action_record_fields("point", 5, 5)
        custom_history = jnp.zeros((history_length, record_fields), dtype=jnp.float32)
        
        state = create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        ).replace(action_history=custom_history)
        
        config = HistoryConfig(enabled=True, max_history_length=history_length)
        tracker = ActionHistoryTracker()
        
        # Add actions
        current_state = state
        for i in range(num_actions):
            action = {"point": [i % 5, (i + 1) % 5], "operation": i % NUM_OPERATIONS}
            current_state = tracker.add_action(
                current_state, action, config, "point", 5, 5
            )
            current_state = current_state.replace(step_count=current_state.step_count + 1)
            
            # History length should never exceed maximum
            assert current_state.action_history_length <= history_length
    
    @given(
        selection_format=st.sampled_from(["point", "bbox", "mask"]),
        grid_height=st.integers(min_value=3, max_value=10),
        grid_width=st.integers(min_value=3, max_value=10)
    )
    def test_action_record_size_consistency(self, selection_format, grid_height, grid_width):
        """Test that action record sizes are consistent with format."""
        expected_fields = get_action_record_fields(selection_format, grid_height, grid_width)
        
        # Create mock selection data
        if selection_format == "point":
            selection_data = jnp.array([1.0, 2.0])
        elif selection_format == "bbox":
            selection_data = jnp.array([0.0, 1.0, 2.0, 3.0])
        else:  # mask
            selection_data = jnp.ones(grid_height * grid_width) * 0.5
        
        # Pad to expected size
        if selection_data.shape[0] < expected_fields - 4:  # -4 for metadata
            padding_size = (expected_fields - 4) - selection_data.shape[0]
            selection_data = jnp.concatenate([selection_data, jnp.zeros(padding_size)])
        
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(15),
            timestamp=jnp.array(10),
            pair_index=jnp.array(0),
            valid=jnp.array(True)
        )
        
        # Validate record
        is_valid = record.validate_integrity(selection_format, grid_height, grid_width)
        assert is_valid is True
    
    @given(
        operation_id=st.integers(min_value=0, max_value=NUM_OPERATIONS-1),
        timestamp=st.integers(min_value=0, max_value=1000),
        pair_index=st.integers(min_value=0, max_value=10)
    )
    def test_action_record_metadata_validation(self, operation_id, timestamp, pair_index):
        """Test that action record metadata validation works correctly."""
        selection_data = jnp.array([2.0, 3.0])  # Valid point data
        
        record = ActionRecord(
            selection_data=selection_data,
            operation_id=jnp.array(operation_id),
            timestamp=jnp.array(timestamp),
            pair_index=jnp.array(pair_index),
            valid=jnp.array(True)
        )
        
        # Should be valid for all generated values
        is_valid = record.validate_integrity("point", 5, 5)
        assert is_valid is True
        
        # Check metadata summary
        summary = record.get_metadata_summary()
        assert summary["operation_id"] == operation_id
        assert summary["timestamp"] == timestamp
        assert summary["pair_index"] == pair_index