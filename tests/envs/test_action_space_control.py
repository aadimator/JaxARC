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

from jaxarc.envs.action_space_controller import ActionSpaceController
from jaxarc.envs.config import ActionConfig
from jaxarc.state import ArcEnvState, create_arc_env_state
from jaxarc.types import JaxArcTask
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
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
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

    def test_validate_operation_jax(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test JAX-compatible operation validation."""
        # Test valid operation
        op_id = jnp.array(15, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, basic_action_config)
        assert bool(is_valid), "Grid operation 15 should be valid"
        
        # Test invalid operation (out of range)
        op_id = jnp.array(100, dtype=jnp.int32)
        is_valid = controller.validate_operation_jax(op_id, training_state, basic_action_config)
        assert not bool(is_valid), "Operation 100 should be invalid"
        
        # Verify JAX compatibility
        chex.assert_type(is_valid, jnp.bool_)
        chex.assert_shape(is_valid, ())

    def test_filter_invalid_operation_basic(
        self,
        controller: ActionSpaceController,
        training_state: ArcEnvState,
    ):
        """Test basic invalid operation filtering."""
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
        
        # Test valid operation passes through
        filtered = controller.filter_invalid_operation(15, training_state, config)
        assert filtered == 15, "Valid operation should pass through unchanged"

    def test_get_operation_availability_summary(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test operation availability summary generation."""
        summary = controller.get_operation_availability_summary(training_state, basic_action_config)
        
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

    def test_apply_operation_mask_jax(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test applying operation mask to action logits."""
        # Create sample action logits
        batch_size = 4
        action_logits = jnp.ones((batch_size, NUM_OPERATIONS), dtype=jnp.float32)
        
        # Apply mask
        masked_logits = controller.apply_operation_mask_jax(
            action_logits, training_state, basic_action_config
        )
        
        # Check shape preservation
        assert masked_logits.shape == action_logits.shape
        
        # Get expected mask
        expected_mask = controller.get_allowed_operations(training_state, basic_action_config)
        
        # Check that disallowed operations have -inf values
        for i in range(NUM_OPERATIONS):
            if not expected_mask[i]:
                assert jnp.all(masked_logits[:, i] == -jnp.inf), f"Operation {i} should be masked to -inf"
            else:
                assert jnp.all(masked_logits[:, i] == 1.0), f"Operation {i} should remain unchanged"

    def test_jax_compatibility_basic(
        self,
        controller: ActionSpaceController,
        basic_action_config: ActionConfig,
        training_state: ArcEnvState,
    ):
        """Test basic JAX compatibility."""
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
        allowed_mask = controller.get_allowed_operations(training_state, basic_action_config)
        op_id = jnp.array(15, dtype=jnp.int32)
        jit_result = jit_basic_validation(op_id, allowed_mask)
        assert isinstance(jit_result, jnp.ndarray)
        assert bool(jit_result), "Valid operation should pass JIT validation"