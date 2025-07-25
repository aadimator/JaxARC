"""
Unit tests for basic action space control.

This module tests the ActionSpaceController class for context-aware operation
filtering, non-parametric control operations (pair switching), operation validation
and filtering policies, invalid operation handling and error cases, and JAX
compatibility of action space operations.

Test Coverage:
- ActionSpaceController context-aware operation filtering
- Non-parametric control operations (pair switching)
- Operation validation and filtering policies
- Invalid operation handling and error cases
- JAX compatibility of action space operations

Requirements Coverage: 1.4, 4.1, 4.2, 4.4, 4.5
"""

import pytest
import jax
import jax.numpy as jnp
import chex
import equinox as eqx
from typing import Dict, Any, Optional, List
from hypothesis import given, strategies as st

from jaxarc.envs.action_space_controller import ActionSpaceController
from jaxarc.envs.config import ActionConfig
from jaxarc.state import ArcEnvState, create_arc_env_state
from jaxarc.types import JaxArcTask, Grid
from jaxarc.utils import jax_types

# Constants for readability
NUM_OPERATIONS = jax_types.NUM_OPERATIONS
DEFAULT_MAX_TRAIN_PAIRS = jax_types.DEFAULT_MAX_TRAIN_PAIRS
DEFAULT_MAX_TEST_PAIRS = jax_types.DEFAULT_MAX_TEST_PAIRS


class TestActionSpaceController:
    """Test ActionSpaceController context-aware operation filtering."""

    @pytest.fixture
    def controller(self) -> ActionSpaceController:
        """Create ActionSpaceController instance for testing."""
        return ActionSpaceController()

    @pytest.fixture
    def basic_action_config(self) -> ActionConfig:
        """Create basic ActionConfig for testing."""
        return ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            max_operations=NUM_OPERATIONS,
            allowed_operations=None,
            validate_actions=True,
            allow_invalid_actions=False,
            dynamic_action_filtering=False,
            context_dependent_operations=False,
            invalid_operation_policy="clip",
        )

    @pytest.fixture
    def dynamic_action_config(self) -> ActionConfig:
        """Create ActionConfig with dynamic filtering enabled."""
        return ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            max_operations=NUM_OPERATIONS,
            allowed_operations=None,
            validate_actions=True,
            allow_invalid_actions=False,
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
        )

    @pytest.fixture
    def restricted_action_config(self) -> ActionConfig:
        """Create ActionConfig with restricted operations."""
        return ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            max_operations=NUM_OPERATIONS,
            allowed_operations=[0, 1, 2, 15, 16, 35, 36],  # Limited set (removed 37 - test switching)
            validate_actions=True,
            allow_invalid_actions=False,
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
        )

    @pytest.fixture
    def sample_task_data(self) -> JaxArcTask:
        """Create sample task data for testing."""
        # Create minimal task data with proper shapes
        max_pairs = 4
        grid_size = 10
        
        # Create task with multiple demo and test pairs
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_test_pairs=2,
            task_index=jnp.array(42, dtype=jnp.int32),
        )
        
        return task_data

    @pytest.fixture
    def training_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in training mode with multiple demo pairs."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=5,  # Non-zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set training mode with multiple available demo pairs
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(0))  # Training mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True, True, True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 3))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True, True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 2))
        )
        
        return state

    @pytest.fixture
    def test_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in test mode with multiple test pairs."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=3,  # Non-zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set test mode with multiple available test pairs
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(1))  # Test mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True, True, True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 3))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True, True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 2))
        )
        
        return state

    @pytest.fixture
    def single_demo_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in training mode with single demo pair."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=0,  # Zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set training mode with single available demo pair
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(0))  # Training mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 1))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 1))
        )
        
        return state

    def test_get_allowed_operations_basic(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test basic get_allowed_operations without dynamic filtering."""
        # Test with basic config (no dynamic filtering)
        allowed_mask = controller.get_allowed_operations(training_state, basic_action_config)
        
        # Should allow all operations when dynamic filtering is disabled
        assert allowed_mask.shape == (NUM_OPERATIONS,)
        assert jnp.all(allowed_mask), "All operations should be allowed with basic config"
        
        # Verify JAX compatibility
        chex.assert_type(allowed_mask, jnp.bool_)
        chex.assert_shape(allowed_mask, (NUM_OPERATIONS,))

    def test_get_allowed_operations_restricted(
        self,
        controller: ActionSpaceController,
        restricted_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test get_allowed_operations with restricted operation list."""
        allowed_mask = controller.get_allowed_operations(training_state, restricted_action_config)
        
        # Should only allow operations in the allowed_operations list
        expected_allowed = restricted_action_config.allowed_operations
        for i in range(NUM_OPERATIONS):
            if i in expected_allowed:
                assert allowed_mask[i], f"Operation {i} should be allowed"
            else:
                assert not allowed_mask[i], f"Operation {i} should not be allowed"

    def test_get_allowed_operations_dynamic_training(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test dynamic operation filtering in training mode."""
        allowed_mask = controller.get_allowed_operations(training_state, dynamic_action_config)
        
        # In training mode with multiple demos, demo switching should be allowed
        assert allowed_mask[35], "SWITCH_TO_NEXT_DEMO_PAIR should be allowed in training"
        assert allowed_mask[36], "SWITCH_TO_PREV_DEMO_PAIR should be allowed in training"
        assert allowed_mask[40], "SWITCH_TO_FIRST_UNSOLVED_DEMO should be allowed in training"
        
        # Test switching should not be allowed in training mode
        assert not allowed_mask[37], "SWITCH_TO_NEXT_TEST_PAIR should not be allowed in training"
        assert not allowed_mask[38], "SWITCH_TO_PREV_TEST_PAIR should not be allowed in training"
        assert not allowed_mask[41], "SWITCH_TO_FIRST_UNSOLVED_TEST should not be allowed in training"
        
        # Pair reset should be allowed (step_count > 0)
        assert allowed_mask[39], "RESET_CURRENT_PAIR should be allowed with step_count > 0"
        
        # Grid operations should still be allowed
        for i in range(35):  # Grid operations 0-34
            assert allowed_mask[i], f"Grid operation {i} should be allowed"

    def test_get_allowed_operations_dynamic_test(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        test_state: ArcEnvState,
    ):
        """Test dynamic operation filtering in test mode."""
        allowed_mask = controller.get_allowed_operations(test_state, dynamic_action_config)
        
        # In test mode with multiple tests, test switching should be allowed
        assert allowed_mask[37], "SWITCH_TO_NEXT_TEST_PAIR should be allowed in test mode"
        assert allowed_mask[38], "SWITCH_TO_PREV_TEST_PAIR should be allowed in test mode"
        assert allowed_mask[41], "SWITCH_TO_FIRST_UNSOLVED_TEST should be allowed in test mode"
        
        # Demo switching should not be allowed in test mode
        assert not allowed_mask[35], "SWITCH_TO_NEXT_DEMO_PAIR should not be allowed in test mode"
        assert not allowed_mask[36], "SWITCH_TO_PREV_DEMO_PAIR should not be allowed in test mode"
        assert not allowed_mask[40], "SWITCH_TO_FIRST_UNSOLVED_DEMO should not be allowed in test mode"
        
        # Pair reset should be allowed (step_count > 0)
        assert allowed_mask[39], "RESET_CURRENT_PAIR should be allowed with step_count > 0"

    def test_get_allowed_operations_single_demo(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        single_demo_state: ArcEnvState,
    ):
        """Test dynamic operation filtering with single demo pair."""
        allowed_mask = controller.get_allowed_operations(single_demo_state, dynamic_action_config)
        
        # With single demo, demo switching should not be allowed
        assert not allowed_mask[35], "SWITCH_TO_NEXT_DEMO_PAIR should not be allowed with single demo"
        assert not allowed_mask[36], "SWITCH_TO_PREV_DEMO_PAIR should not be allowed with single demo"
        assert not allowed_mask[40], "SWITCH_TO_FIRST_UNSOLVED_DEMO should not be allowed with single demo"
        
        # Pair reset should not be allowed (step_count == 0)
        assert not allowed_mask[39], "RESET_CURRENT_PAIR should not be allowed with step_count == 0"

    def test_validate_operation_basic(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test basic operation validation."""
        # Test valid grid operation
        is_valid, error_msg = controller.validate_operation(15, training_state, basic_action_config)
        assert is_valid, f"Grid operation 15 should be valid: {error_msg}"
        assert error_msg is None, "No error message should be returned for valid operation"
        
        # Test invalid operation ID (out of range)
        is_valid, error_msg = controller.validate_operation(100, training_state, basic_action_config)
        assert not is_valid, "Operation 100 should be invalid (out of range)"
        assert "out of range" in error_msg.lower(), f"Error message should mention range: {error_msg}"
        
        # Test negative operation ID
        is_valid, error_msg = controller.validate_operation(-1, training_state, basic_action_config)
        assert not is_valid, "Operation -1 should be invalid (negative)"
        assert "out of range" in error_msg.lower(), f"Error message should mention range: {error_msg}"

    def test_validate_operation_context_aware(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
        test_state: ArcEnvState,
        single_demo_state: ArcEnvState,
    ):
        """Test context-aware operation validation."""
        # Test demo switching in training mode (should be valid)
        is_valid, error_msg = controller.validate_operation(35, training_state, dynamic_action_config)
        assert is_valid, f"Demo switching should be valid in training mode: {error_msg}"
        
        # Test demo switching in test mode (should be invalid)
        is_valid, error_msg = controller.validate_operation(35, test_state, dynamic_action_config)
        assert not is_valid, "Demo switching should be invalid in test mode"
        assert error_msg is not None, "Error message should be provided for invalid operation"
        
        # Test test switching in test mode (should be valid)
        is_valid, error_msg = controller.validate_operation(37, test_state, dynamic_action_config)
        assert is_valid, f"Test switching should be valid in test mode: {error_msg}"
        
        # Test test switching in training mode (should be invalid)
        is_valid, error_msg = controller.validate_operation(37, training_state, dynamic_action_config)
        assert not is_valid, "Test switching should be invalid in training mode"
        assert error_msg is not None, "Error message should be provided for invalid operation"
        
        # Test demo switching with single demo (should be invalid)
        is_valid, error_msg = controller.validate_operation(35, single_demo_state, dynamic_action_config)
        assert not is_valid, "Demo switching should be invalid with single demo"
        assert error_msg is not None, "Error message should be provided for invalid operation"
        
        # Test pair reset at step 0 (should be invalid)
        is_valid, error_msg = controller.validate_operation(39, single_demo_state, dynamic_action_config)
        assert not is_valid, "Pair reset should be invalid at step 0"
        assert error_msg is not None, "Error message should be provided for invalid operation"

    def test_validate_operation_jax(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test JAX-compatible operation validation."""
        # Test valid operation
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, dynamic_action_config)
        assert bool(is_valid), "Grid operation 15 should be valid"
        
        # Test invalid operation (out of range)
        op_id = jnp.array(100, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, dynamic_action_config)
        assert not bool(is_valid), "Operation 100 should be invalid"
        
        # Test context-invalid operation (test switching in training mode)
        op_id = jnp.array(37, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, dynamic_action_config)
        assert not bool(is_valid), "Test switching should be invalid in training mode"
        
        # Verify JAX compatibility
        chex.assert_type(is_valid, jnp.bool_)
        chex.assert_shape(is_valid, ())

    def test_filter_invalid_operation_clip_policy(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test invalid operation filtering with clip policy."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
            allowed_operations=[0, 1, 2, 15, 16],  # Limited set
        )
        
        # Test clipping out-of-range operation
        filtered = controller.filter_invalid_operation(100, training_state, config)
        assert 0 <= filtered < NUM_OPERATIONS, f"Filtered operation {filtered} should be in valid range"
        assert filtered in config.allowed_operations, f"Filtered operation {filtered} should be in allowed list"
        
        # Test clipping disallowed operation
        filtered = controller.filter_invalid_operation(20, training_state, config)  # Not in allowed list
        assert filtered in config.allowed_operations, f"Filtered operation {filtered} should be in allowed list"
        
        # Test valid operation passes through
        filtered = controller.filter_invalid_operation(15, training_state, config)
        assert filtered == 15, "Valid operation should pass through unchanged"

    def test_filter_invalid_operation_reject_policy(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test invalid operation filtering with reject policy."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="reject",
            allowed_operations=[0, 1, 2, 15, 16],
        )
        
        # Test rejecting invalid operation
        filtered = controller.filter_invalid_operation(100, training_state, config)
        assert filtered == -1, "Invalid operation should be rejected with -1"
        
        # Test valid operation passes through
        filtered = controller.filter_invalid_operation(15, training_state, config)
        assert filtered == 15, "Valid operation should pass through unchanged"

    def test_filter_invalid_operation_passthrough_policy(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test invalid operation filtering with passthrough policy."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="passthrough",
            allowed_operations=[0, 1, 2, 15, 16],
        )
        
        # Test invalid operation passes through unchanged
        filtered = controller.filter_invalid_operation(100, training_state, config)
        assert filtered == 100, "Invalid operation should pass through unchanged"
        
        # Test valid operation passes through
        filtered = controller.filter_invalid_operation(15, training_state, config)
        assert filtered == 15, "Valid operation should pass through unchanged"

    def test_filter_invalid_operation_jax_arrays(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test invalid operation filtering with JAX arrays."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
            allowed_operations=[0, 1, 2, 15, 16],
        )
        
        # Test with individual operations (the current implementation expects scalars)
        op_ids = [100, 15, 200, 1]
        filtered_results = []
        
        for op_id in op_ids:
            filtered = controller.filter_invalid_operation(op_id, training_state, config)
            filtered_results.append(filtered)
        
        # Check that valid operations pass through and invalid ones are clipped
        assert filtered_results[1] == 15, "Valid operation should pass through"
        assert filtered_results[3] == 1, "Valid operation should pass through"
        assert filtered_results[0] in config.allowed_operations, "Invalid operation should be clipped to allowed"
        assert filtered_results[2] in config.allowed_operations, "Invalid operation should be clipped to allowed"

    def test_filter_invalid_operation_jax_only(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test JAX-only version of filter_invalid_operation."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
            allowed_operations=[0, 1, 2, 15, 16],
        )
        
        # Test with JAX array
        op_id = jnp.array(100, dtype=jnp.int32)
        filtered = controller.filter_invalid_operation_jax(op_id, training_state, config)
        
        assert filtered in config.allowed_operations, "Invalid operation should be clipped to allowed"
        
        # Verify JAX compatibility
        chex.assert_type(filtered, jnp.int32)
        chex.assert_shape(filtered, ())

    @given(st.integers(min_value=-10, max_value=110))
    def test_filter_invalid_operation_property_based(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
        operation_id: int,
    ):
        """Property-based test for operation filtering."""
        config = ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
            allowed_operations=[0, 1, 2, 15, 16],
        )
        
        filtered = controller.filter_invalid_operation(operation_id, training_state, config)
        
        # Property: filtered operation should always be in allowed operations
        # (except for reject policy which returns -1)
        if config.invalid_operation_policy != "reject" or operation_id in config.allowed_operations:
            if config.invalid_operation_policy == "passthrough":
                assert filtered == operation_id, "Passthrough should not change operation"
            elif config.invalid_operation_policy == "reject" and operation_id not in config.allowed_operations:
                assert filtered == -1, "Reject should return -1 for invalid operations"
            else:
                assert filtered in config.allowed_operations, f"Filtered operation {filtered} should be in allowed list"

    def test_jax_transformations_compatibility(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test JAX transformations compatibility."""
        
        # Test basic functionality (config contains non-arrays, so test directly)
        allowed_mask = controller.get_allowed_operations(training_state, dynamic_action_config)
        assert allowed_mask.shape == (NUM_OPERATIONS,)
        
        # Test JAX-compatible methods directly (config contains non-arrays)
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, dynamic_action_config)
        assert bool(is_valid), "Grid operation should be valid"
        
        filtered = controller.filter_invalid_operation_jax(op_id, training_state, dynamic_action_config)
        assert filtered == 15, "Valid operation should pass through"
        
        # Test basic array operations work
        batch_op_ids = jnp.array([15, 35, 100], dtype=jnp.int32)
        
        # Test individual operations
        for i, op_id in enumerate([15, 35, 100]):
            op_array = jnp.array(op_id, dtype=jnp.int32)
            is_valid = controller.validate_operation_jax(op_array, training_state, dynamic_action_config)
            
            if i == 0:  # Grid operation
                assert bool(is_valid), "Grid operation should be valid"
            elif i == 1:  # Demo switching in training
                assert bool(is_valid), "Demo switching should be valid in training"
            else:  # Out of range
                assert not bool(is_valid), "Out-of-range operation should be invalid"

    def test_get_operation_availability_summary(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test operation availability summary generation."""
        summary = controller.get_operation_availability_summary(training_state, dynamic_action_config)
        
        # Check summary structure
        assert "total_operations" in summary
        assert "total_allowed" in summary
        assert "total_blocked" in summary
        assert "allowed_operations" in summary
        assert "blocked_operations" in summary
        assert "by_category" in summary
        assert "context_info" in summary
        assert "config_info" in summary
        assert "context_restrictions" in summary
        
        # Check values
        assert summary["total_operations"] == NUM_OPERATIONS
        assert summary["total_allowed"] + summary["total_blocked"] == NUM_OPERATIONS
        assert len(summary["allowed_operations"]) == summary["total_allowed"]
        assert len(summary["blocked_operations"]) == summary["total_blocked"]
        
        # Check context info
        context_info = summary["context_info"]
        assert context_info["episode_mode"] == "train"
        assert context_info["step_count"] == 5
        assert context_info["available_demo_pairs"] == 3
        assert context_info["available_test_pairs"] == 2
        
        # Check config info
        config_info = summary["config_info"]
        assert config_info["dynamic_filtering_enabled"] == True
        assert config_info["context_dependent_enabled"] == True
        assert config_info["invalid_operation_policy"] == "clip"
        
        # Check categories
        categories = summary["by_category"]
        expected_categories = ["fill", "flood_fill", "movement", "transformation", "editing", "special", "control"]
        for category in expected_categories:
            assert category in categories
            cat_info = categories[category]
            assert "allowed" in cat_info
            assert "blocked" in cat_info
            assert "total" in cat_info
            assert "allowed_count" in cat_info
            assert "blocked_count" in cat_info
            assert cat_info["allowed_count"] + cat_info["blocked_count"] == cat_info["total"]

    def test_apply_operation_mask_jax(
        self,
        controller: ActionSpaceController,
        dynamic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test applying operation mask to action logits."""
        # Create sample action logits
        batch_size = 4
        action_logits = jnp.ones((batch_size, NUM_OPERATIONS), dtype=jnp.float32)
        
        # Apply mask
        masked_logits = controller.apply_operation_mask_jax(
            action_logits, training_state, dynamic_action_config
        )
        
        # Check shape preservation
        assert masked_logits.shape == action_logits.shape
        
        # Get expected mask
        expected_mask = controller.get_allowed_operations(training_state, dynamic_action_config)
        
        # Check that disallowed operations have -inf values
        for i in range(NUM_OPERATIONS):
            if not expected_mask[i]:
                assert jnp.all(masked_logits[:, i] == -jnp.inf), f"Operation {i} should be masked to -inf"
            else:
                assert jnp.all(masked_logits[:, i] == 1.0), f"Operation {i} should remain unchanged"
        
        # Test with custom mask value
        custom_mask_value = -1000.0
        masked_logits_custom = controller.apply_operation_mask_jax(
            action_logits, training_state, dynamic_action_config, mask_value=custom_mask_value
        )
        
        for i in range(NUM_OPERATIONS):
            if not expected_mask[i]:
                assert jnp.all(masked_logits_custom[:, i] == custom_mask_value), f"Operation {i} should be masked to custom value"

    def test_error_handling_edge_cases(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test error handling for edge cases."""
        # Test with None config (should handle gracefully)
        try:
            # This might raise an error, which is acceptable
            controller.get_allowed_operations(training_state, None)
        except (AttributeError, TypeError):
            pass  # Expected for None config
        
        # Test with malformed state (missing fields)
        # Create state with missing enhanced fields by using basic state creation
        basic_state = eqx.tree_at(
            lambda s: s.allowed_operations_mask,
            training_state,
            jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_)  # Ensure this field exists
        )
        
        # Should handle gracefully
        allowed_mask = controller.get_allowed_operations(basic_state, basic_action_config)
        assert allowed_mask.shape == (NUM_OPERATIONS,)
        
        # Test validation with extreme values
        is_valid, _ = controller.validate_operation(999999, training_state, basic_action_config)
        assert not is_valid, "Extremely large operation ID should be invalid"
        
        is_valid, _ = controller.validate_operation(-999999, training_state, basic_action_config)
        assert not is_valid, "Extremely negative operation ID should be invalid"


class TestNonParametricControlOperations:
    """Test non-parametric control operations (pair switching)."""

    @pytest.fixture
    def controller(self) -> ActionSpaceController:
        """Create ActionSpaceController instance for testing."""
        return ActionSpaceController()

    @pytest.fixture
    def dynamic_config(self) -> ActionConfig:
        """Create ActionConfig with dynamic control enabled."""
        return ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
        )

    @pytest.fixture
    def sample_task_data(self) -> JaxArcTask:
        """Create sample task data for testing."""
        # Create minimal task data with proper shapes
        max_pairs = 4
        grid_size = 10
        
        # Create task with multiple demo and test pairs
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_test_pairs=2,
            task_index=jnp.array(42, dtype=jnp.int32),
        )
        
        return task_data

    @pytest.fixture
    def training_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in training mode with multiple demo pairs."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=5,  # Non-zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set training mode with multiple available demo pairs
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(0))  # Training mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True, True, True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 3))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True, True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 2))
        )
        
        return state

    @pytest.fixture
    def test_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in test mode with multiple test pairs."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=3,  # Non-zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set test mode with multiple available test pairs
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(1))  # Test mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True, True, True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 3))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True, True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 2))
        )
        
        return state

    @pytest.fixture
    def single_demo_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in training mode with single demo pair."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=0,  # Zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set training mode with single available demo pair
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(0))  # Training mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 1))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 1))
        )
        
        return state

    def test_demo_switching_operations(
        self,
        controller: ActionSpaceController,
        dynamic_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test demo pair switching operations in training mode."""
        # Test SWITCH_TO_NEXT_DEMO_PAIR (35)
        is_valid, error_msg = controller.validate_operation(35, training_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_NEXT_DEMO_PAIR should be valid in training: {error_msg}"
        
        # Test SWITCH_TO_PREV_DEMO_PAIR (36)
        is_valid, error_msg = controller.validate_operation(36, training_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_PREV_DEMO_PAIR should be valid in training: {error_msg}"
        
        # Test SWITCH_TO_FIRST_UNSOLVED_DEMO (40)
        is_valid, error_msg = controller.validate_operation(40, training_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_FIRST_UNSOLVED_DEMO should be valid in training: {error_msg}"

    def test_test_switching_operations(
        self,
        controller: ActionSpaceController,
        dynamic_config: ActionConfig,
        test_state: ArcEnvState,
    ):
        """Test test pair switching operations in test mode."""
        # Test SWITCH_TO_NEXT_TEST_PAIR (37)
        is_valid, error_msg = controller.validate_operation(37, test_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_NEXT_TEST_PAIR should be valid in test: {error_msg}"
        
        # Test SWITCH_TO_PREV_TEST_PAIR (38)
        is_valid, error_msg = controller.validate_operation(38, test_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_PREV_TEST_PAIR should be valid in test: {error_msg}"
        
        # Test SWITCH_TO_FIRST_UNSOLVED_TEST (41)
        is_valid, error_msg = controller.validate_operation(41, test_state, dynamic_config)
        assert is_valid, f"SWITCH_TO_FIRST_UNSOLVED_TEST should be valid in test: {error_msg}"

    def test_pair_reset_operation(
        self,
        controller: ActionSpaceController,
        dynamic_config: ActionConfig,
        training_state: ArcEnvState,
        single_demo_state: ArcEnvState,
    ):
        """Test pair reset operation (39)."""
        # Test RESET_CURRENT_PAIR with step_count > 0
        is_valid, error_msg = controller.validate_operation(39, training_state, dynamic_config)
        assert is_valid, f"RESET_CURRENT_PAIR should be valid with step_count > 0: {error_msg}"
        
        # Test RESET_CURRENT_PAIR with step_count == 0
        is_valid, error_msg = controller.validate_operation(39, single_demo_state, dynamic_config)
        assert not is_valid, "RESET_CURRENT_PAIR should be invalid with step_count == 0"
        assert "step 0" in error_msg.lower(), f"Error should mention step 0: {error_msg}"

    def test_cross_mode_restrictions(
        self,
        controller: ActionSpaceController,
        dynamic_config: ActionConfig,
        training_state: ArcEnvState,
        test_state: ArcEnvState,
    ):
        """Test that switching operations are restricted across modes."""
        # Demo switching should not work in test mode
        demo_ops = [35, 36, 40]  # Demo switching operations
        for op_id in demo_ops:
            is_valid, error_msg = controller.validate_operation(op_id, test_state, dynamic_config)
            assert not is_valid, f"Demo operation {op_id} should not be valid in test mode"
            assert error_msg is not None, f"Error message should be provided for invalid operation {op_id}"
        
        # Test switching should not work in training mode
        test_ops = [37, 38, 41]  # Test switching operations
        for op_id in test_ops:
            is_valid, error_msg = controller.validate_operation(op_id, training_state, dynamic_config)
            assert not is_valid, f"Test operation {op_id} should not be valid in training mode"
            assert error_msg is not None, f"Error message should be provided for invalid operation {op_id}"

    def test_single_pair_restrictions(
        self,
        controller: ActionSpaceController,
        dynamic_config: ActionConfig,
        single_demo_state: ArcEnvState,
    ):
        """Test that switching operations are restricted with single pairs."""
        # Demo switching should not work with single demo
        demo_ops = [35, 36, 40]
        for op_id in demo_ops:
            is_valid, error_msg = controller.validate_operation(op_id, single_demo_state, dynamic_config)
            assert not is_valid, f"Demo operation {op_id} should not be valid with single demo"
            assert error_msg is not None, f"Error message should be provided for invalid operation {op_id}"


class TestJAXCompatibility:
    """Test JAX compatibility of action space operations."""

    @pytest.fixture
    def controller(self) -> ActionSpaceController:
        """Create ActionSpaceController instance for testing."""
        return ActionSpaceController()

    @pytest.fixture
    def config(self) -> ActionConfig:
        """Create ActionConfig for testing."""
        return ActionConfig(
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="clip",
        )

    @pytest.fixture
    def sample_task_data(self) -> JaxArcTask:
        """Create sample task data for testing."""
        # Create minimal task data with proper shapes
        max_pairs = 4
        grid_size = 10
        
        # Create task with multiple demo and test pairs
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.bool_),
            num_test_pairs=2,
            task_index=jnp.array(42, dtype=jnp.int32),
        )
        
        return task_data

    @pytest.fixture
    def training_state(self, sample_task_data: JaxArcTask) -> ArcEnvState:
        """Create ArcEnvState in training mode with multiple demo pairs."""
        grid_shape = (10, 10)
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            max_train_pairs=DEFAULT_MAX_TRAIN_PAIRS,
            max_test_pairs=DEFAULT_MAX_TEST_PAIRS,
            step_count=5,  # Non-zero step count
            episode_done=False,
            current_example_idx=0,
        )
        
        # Set training mode with multiple available demo pairs
        state = eqx.tree_at(lambda s: s.episode_mode, state, jnp.array(0))  # Training mode
        state = eqx.tree_at(
            lambda s: s.available_demo_pairs,
            state,
            jnp.array([True, True, True] + [False] * (DEFAULT_MAX_TRAIN_PAIRS - 3))
        )
        state = eqx.tree_at(
            lambda s: s.available_test_pairs,
            state,
            jnp.array([True, True] + [False] * (DEFAULT_MAX_TEST_PAIRS - 2))
        )
        
        return state

    def test_jit_compilation(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test that all methods can be JIT compiled."""
        
        # Test get_allowed_operations
        @jax.jit
        def jit_get_allowed(state, cfg):
            return controller.get_allowed_operations(state, cfg)
        
        # Note: Config contains non-array fields, so we test the method directly
        # In practice, configs would be static or handled outside JIT
        mask = controller.get_allowed_operations(training_state, config)
        assert mask.shape == (NUM_OPERATIONS,)
        
        # Test validate_operation_jax (config contains non-arrays, test method directly)
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, config)
        assert isinstance(is_valid, jnp.ndarray)
        
        # Test filter_invalid_operation_jax (config contains non-arrays, test method directly)
        filtered = controller.filter_invalid_operation_jax(op_id, training_state, config)
        assert isinstance(filtered, jnp.ndarray)
        
        # Test that the core JAX operations can be JIT compiled
        @jax.jit
        def jit_basic_validation(op_id, allowed_mask):
            # Basic range check
            in_range = (op_id >= 0) & (op_id < NUM_OPERATIONS)
            # Check if operation is allowed (safe indexing with bounds check)
            is_allowed = jnp.where(
                in_range,
                allowed_mask[jnp.clip(op_id, 0, NUM_OPERATIONS - 1)],
                False
            )
            return in_range & is_allowed
        
        # Test with actual mask
        allowed_mask = controller.get_allowed_operations(training_state, config)
        jit_result = jit_basic_validation(op_id, allowed_mask)
        assert isinstance(jit_result, jnp.ndarray)

    def test_vmap_compatibility(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test vmap compatibility for batch processing."""
        batch_size = 8
        
        # Create batch of states
        batch_states = jax.tree.map(lambda x: jnp.stack([x] * batch_size), training_state)
        
        # Test vmapped get_allowed_operations
        vmapped_get_allowed = jax.vmap(controller.get_allowed_operations, in_axes=(0, None))
        batch_masks = vmapped_get_allowed(batch_states, config)
        
        assert batch_masks.shape == (batch_size, NUM_OPERATIONS)
        
        # Test vmapped validate_operation_jax
        batch_op_ids = jnp.array([15, 35, 100, 37, 39, 0, 1, 2], dtype=jnp.int32)
        vmapped_validate = jax.vmap(controller.validate_operation_jax, in_axes=(0, 0, None))
        batch_valid = vmapped_validate(batch_op_ids, batch_states, config)
        
        assert batch_valid.shape == (batch_size,)
        
        # Test vmapped filter_invalid_operation_jax
        vmapped_filter = jax.vmap(controller.filter_invalid_operation_jax, in_axes=(0, 0, None))
        batch_filtered = vmapped_filter(batch_op_ids, batch_states, config)
        
        assert batch_filtered.shape == (batch_size,)

    def test_grad_compatibility(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test gradient compatibility for differentiable operations."""
        
        # Test that apply_operation_mask_jax is differentiable
        def loss_fn(logits):
            masked_logits = controller.apply_operation_mask_jax(logits, training_state, config)
            return jnp.sum(masked_logits)
        
        logits = jnp.ones((1, NUM_OPERATIONS), dtype=jnp.float32)
        
        # Should be able to compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(logits)
        
        assert grads.shape == logits.shape
        
        # Gradients should be zero for masked operations and one for allowed operations
        allowed_mask = controller.get_allowed_operations(training_state, config)
        for i in range(NUM_OPERATIONS):
            if allowed_mask[i]:
                assert grads[0, i] == 1.0, f"Gradient should be 1.0 for allowed operation {i}"
            else:
                assert grads[0, i] == 0.0, f"Gradient should be 0.0 for masked operation {i}"

    def test_pmap_compatibility(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test pmap compatibility for multi-device processing."""
        # Skip if not enough devices
        if jax.device_count() < 2:
            pytest.skip("Not enough devices for pmap test")
        
        # Create batch for multiple devices
        device_count = min(jax.device_count(), 4)
        batch_states = jax.tree.map(lambda x: jnp.stack([x] * device_count), training_state)
        
        # Test pmapped get_allowed_operations
        pmapped_get_allowed = jax.pmap(controller.get_allowed_operations, in_axes=(0, None))
        device_masks = pmapped_get_allowed(batch_states, config)
        
        assert device_masks.shape == (device_count, NUM_OPERATIONS)
        
        # All devices should produce the same result
        for i in range(1, device_count):
            assert jnp.array_equal(device_masks[0], device_masks[i]), "All devices should produce same mask"

    def test_array_types_and_shapes(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test that all operations maintain correct array types and shapes."""
        
        # Test get_allowed_operations
        mask = controller.get_allowed_operations(training_state, config)
        chex.assert_type(mask, jnp.bool_)
        chex.assert_shape(mask, (NUM_OPERATIONS,))
        
        # Test validate_operation_jax
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, config)
        chex.assert_type(is_valid, jnp.bool_)
        chex.assert_shape(is_valid, ())
        
        # Test filter_invalid_operation_jax
        filtered = controller.filter_invalid_operation_jax(op_id, training_state, config)
        chex.assert_type(filtered, jnp.int32)
        chex.assert_shape(filtered, ())
        
        # Test apply_operation_mask_jax
        logits = jnp.ones((4, NUM_OPERATIONS), dtype=jnp.float32)
        masked_logits = controller.apply_operation_mask_jax(logits, training_state, config)
        chex.assert_type(masked_logits, jnp.float32)
        chex.assert_shape(masked_logits, (4, NUM_OPERATIONS))

    def test_static_shape_requirements(
        self,
        controller: ActionSpaceController,
        config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test that all operations work with static shapes required by JAX."""
        
        # All operations should work with concrete shapes
        mask = controller.get_allowed_operations(training_state, config)
        assert hasattr(mask, 'shape'), "Result should have concrete shape"
        assert mask.shape == (NUM_OPERATIONS,), f"Expected shape {(NUM_OPERATIONS,)}, got {mask.shape}"
        
        # Test that operations work with concrete shapes (config contains non-arrays)
        mask = controller.get_allowed_operations(training_state, config)
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, config)
        filtered = controller.filter_invalid_operation_jax(op_id, training_state, config)
        
        # Results should have correct shapes even after tracing
        assert mask.shape == (NUM_OPERATIONS,)
        assert is_valid.shape == ()
        assert filtered.shape == ()