"""
Unit tests for enhanced state management and observations.

This module tests the enhanced ArcEnvState fields, ArcObservation construction,
create_observation function, ObservationConfig options, utility methods for pair
access and status tracking, JAX compatibility, and state transitions.

Test Coverage:
- Enhanced ArcEnvState fields and validation
- ArcObservation construction and context information
- create_observation function with different configurations
- ObservationConfig options and observation formats
- Utility methods for pair access and status tracking
- JAX compatibility of all new state and observation operations
- State transitions with new fields

Requirements Coverage: 1.1, 1.2, 1.3, 6.1, 6.2, 6.4
"""

import pytest
import jax
import jax.numpy as jnp
import chex
import equinox as eqx

from jaxarc.state import ArcEnvState, create_arc_env_state
from jaxarc.envs.observations import (
    ObservationConfig,
    create_observation,
    create_minimal_observation,
    create_evaluation_observation,
)
from jaxarc.types import JaxArcTask
from jaxarc.utils import jax_types


class TestEnhancedArcEnvState:
    """Test enhanced ArcEnvState fields and validation."""

    @pytest.fixture
    def sample_task_data(self) -> JaxArcTask:
        """Create sample task data for testing."""
        # Create minimal task data with proper shapes
        max_pairs = 4
        grid_size = 10
        
        return JaxArcTask(
            input_grids_examples=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,
            task_index=jnp.array(42, dtype=jnp.int32),
        )

    @pytest.fixture
    def sample_grids(self):
        """Create sample grids for testing."""
        grid_size = 10
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        return working_grid, working_grid_mask, target_grid

    def test_enhanced_state_creation(self, sample_task_data: JaxArcTask, sample_grids):
        """Test creation of enhanced ArcEnvState with new fields."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        # Test creation with default parameters
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
        
        # Verify core fields exist and have correct types
        assert hasattr(state, 'task_data')
        assert hasattr(state, 'working_grid')
        assert hasattr(state, 'working_grid_mask')
        assert hasattr(state, 'target_grid')
        assert hasattr(state, 'step_count')
        assert hasattr(state, 'episode_done')
        assert hasattr(state, 'current_example_idx')
        assert hasattr(state, 'selected')
        assert hasattr(state, 'clipboard')
        assert hasattr(state, 'similarity_score')
        
        # Verify enhanced fields exist and have correct types
        assert hasattr(state, 'episode_mode')
        assert hasattr(state, 'available_demo_pairs')
        assert hasattr(state, 'available_test_pairs')
        assert hasattr(state, 'demo_completion_status')
        assert hasattr(state, 'test_completion_status')
        assert hasattr(state, 'action_history')
        assert hasattr(state, 'action_history_length')
        assert hasattr(state, 'allowed_operations_mask')

    def test_enhanced_state_field_types(self, sample_task_data: JaxArcTask, sample_grids):
        """Test that enhanced state fields have correct JAX types."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=5,
            max_test_pairs=3,
        )
        
        # Test enhanced field types
        chex.assert_type(state.episode_mode, jnp.integer)
        chex.assert_type(state.available_demo_pairs, jnp.bool_)
        chex.assert_type(state.available_test_pairs, jnp.bool_)
        chex.assert_type(state.demo_completion_status, jnp.bool_)
        chex.assert_type(state.test_completion_status, jnp.bool_)
        chex.assert_type(state.action_history, jnp.floating)
        chex.assert_type(state.action_history_length, jnp.integer)
        chex.assert_type(state.allowed_operations_mask, jnp.bool_)

    def test_enhanced_state_field_shapes(self, sample_task_data: JaxArcTask, sample_grids):
        """Test that enhanced state fields have correct shapes."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        max_train_pairs = 5
        max_test_pairs = 3
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
        )
        
        # Test enhanced field shapes
        chex.assert_shape(state.episode_mode, ())
        chex.assert_shape(state.available_demo_pairs, (max_train_pairs,))
        chex.assert_shape(state.available_test_pairs, (max_test_pairs,))
        chex.assert_shape(state.demo_completion_status, (max_train_pairs,))
        chex.assert_shape(state.test_completion_status, (max_test_pairs,))
        chex.assert_shape(state.action_history, (jax_types.MAX_HISTORY_LENGTH, jax_types.ACTION_RECORD_FIELDS))
        chex.assert_shape(state.action_history_length, ())
        chex.assert_shape(state.allowed_operations_mask, (jax_types.NUM_OPERATIONS,))

    def test_enhanced_state_validation(self, sample_task_data: JaxArcTask, sample_grids):
        """Test validation of enhanced state fields."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        # Test valid state creation
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
        )
        
        # Validation should pass without errors
        assert state.episode_mode == jax_types.EPISODE_MODE_TRAIN
        
        # Test with test mode
        state_test = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            episode_mode=jax_types.EPISODE_MODE_TEST,
        )
        
        assert state_test.episode_mode == jax_types.EPISODE_MODE_TEST

    def test_enhanced_state_default_values(self, sample_task_data: JaxArcTask, sample_grids):
        """Test default values for enhanced state fields."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
        
        # Test default values
        assert state.episode_mode == jax_types.EPISODE_MODE_TRAIN
        assert jnp.all(state.available_demo_pairs)  # All demo pairs available by default
        assert jnp.all(state.available_test_pairs)  # All test pairs available by default
        assert not jnp.any(state.demo_completion_status)  # No pairs completed initially
        assert not jnp.any(state.test_completion_status)  # No pairs completed initially
        assert state.action_history_length == 0  # No actions initially
        assert jnp.all(state.allowed_operations_mask)  # All operations allowed by default

    def test_enhanced_state_utility_methods(self, sample_task_data: JaxArcTask, sample_grids):
        """Test utility methods for enhanced state functionality."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
        )
        
        # Test mode checking methods
        assert state.is_training_mode()
        assert not state.is_test_mode()
        
        # Test pair counting methods
        demo_count = state.get_available_demo_count()
        test_count = state.get_available_test_count()
        assert demo_count > 0
        assert test_count > 0
        
        # Test completion counting methods
        completed_demos = state.get_completed_demo_count()
        completed_tests = state.get_completed_test_count()
        assert completed_demos == 0  # Initially no pairs completed
        assert completed_tests == 0
        
        # Test operation checking methods
        allowed_ops = state.get_allowed_operations_count()
        assert allowed_ops == jax_types.NUM_OPERATIONS  # All operations allowed initially
        
        # Test specific operation checking
        assert state.is_operation_allowed(0)  # First operation should be allowed
        assert state.is_operation_allowed(jax_types.NUM_OPERATIONS - 1)  # Last operation should be allowed
        
        # Test action history methods
        assert state.get_action_history_length() == 0
        assert not state.has_action_history()

    def test_enhanced_state_jax_compatibility(self, sample_task_data: JaxArcTask, sample_grids):
        """Test JAX compatibility of enhanced state operations."""
        working_grid, working_grid_mask, target_grid = sample_grids
        
        state = create_arc_env_state(
            task_data=sample_task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
        )
        
        # Test JIT compilation of utility methods
        @jax.jit
        def test_jit_methods(state):
            return {
                'is_training': state.is_training_mode(),
                'demo_count': state.get_available_demo_count(),
                'test_count': state.get_available_test_count(),
                'completed_demos': state.get_completed_demo_count(),
                'allowed_ops': state.get_allowed_operations_count(),
                'history_length': state.get_action_history_length(),
                'has_history': state.has_action_history(),
            }
        
        # Should compile and run without errors
        result = test_jit_methods(state)
        assert isinstance(result, dict)
        assert result['is_training']
        assert result['demo_count'] > 0
        assert result['history_length'] == 0
        assert not result['has_history']


class TestObservationConfig:
    """Test ObservationConfig options and validation."""

    def test_observation_config_defaults(self):
        """Test default ObservationConfig values."""
        config = ObservationConfig()
        
        # Test default values
        assert config.include_target_grid
        assert config.include_completion_status
        assert config.include_action_space_info
        assert not config.include_recent_actions
        assert config.recent_action_count == 10
        assert config.include_step_count
        assert config.observation_format == "standard"
        assert config.mask_internal_state

    def test_observation_config_formats(self):
        """Test different observation formats."""
        # Test minimal format
        minimal_config = ObservationConfig(observation_format="minimal")
        assert minimal_config.observation_format == "minimal"
        assert not minimal_config.include_recent_actions
        assert not minimal_config.include_completion_status
        
        # Test standard format
        standard_config = ObservationConfig(observation_format="standard")
        assert standard_config.observation_format == "standard"
        
        # Test rich format
        rich_config = ObservationConfig(observation_format="rich")
        assert rich_config.observation_format == "rich"
        assert rich_config.include_completion_status
        assert rich_config.include_action_space_info

    def test_observation_config_validation(self):
        """Test ObservationConfig validation."""
        # Test valid configurations
        config = ObservationConfig(recent_action_count=5)
        assert config.recent_action_count == 5
        
        # Test invalid recent_action_count
        with pytest.raises(ValueError, match="recent_action_count must be non-negative"):
            ObservationConfig(recent_action_count=-1)
        
        # Test invalid observation_format
        with pytest.raises(ValueError, match="observation_format must be"):
            ObservationConfig(observation_format="invalid")


class TestArcObservation:
    """Test ArcObservation construction and context information."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for observation testing."""
        grid_size = 8
        max_train_pairs = 4
        max_test_pairs = 2
        
        # Create task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=3,
            test_input_grids=jnp.zeros((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=2,
            task_index=jnp.array(123, dtype=jnp.int32),
        )
        
        # Create grids
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
            step_count=3,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
        )

    def test_arc_observation_creation(self, sample_state: ArcEnvState):
        """Test basic ArcObservation creation."""
        config = ObservationConfig()
        observation = create_observation(sample_state, config)
        
        # Test core fields exist
        assert hasattr(observation, 'working_grid')
        assert hasattr(observation, 'working_grid_mask')
        assert hasattr(observation, 'episode_mode')
        assert hasattr(observation, 'current_pair_idx')
        assert hasattr(observation, 'step_count')
        assert hasattr(observation, 'demo_completion_status')
        assert hasattr(observation, 'test_completion_status')
        assert hasattr(observation, 'allowed_operations_mask')
        
        # Test optional fields
        assert hasattr(observation, 'target_grid')
        assert hasattr(observation, 'recent_actions')

    def test_arc_observation_context_information(self, sample_state: ArcEnvState):
        """Test ArcObservation context information."""
        config = ObservationConfig()
        observation = create_observation(sample_state, config)
        
        # Test episode context
        assert observation.episode_mode == jax_types.EPISODE_MODE_TRAIN
        assert observation.is_training_mode()
        assert not observation.is_test_mode()
        assert observation.current_pair_idx == 0  # Default initial pair
        assert observation.step_count == 3  # From sample state
        
        # Test progress information
        assert observation.get_completed_demo_count() == 0  # Initially no completions
        assert observation.get_completed_test_count() == 0
        
        # Test action space information
        assert observation.get_allowed_operations_count() == jax_types.NUM_OPERATIONS
        assert observation.is_operation_allowed(0)
        
        # Test target access
        assert observation.has_target_access()
        assert observation.target_grid is not None


class TestCreateObservationFunction:
    """Test create_observation function with different configurations."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for testing."""
        grid_size = 6
        max_train_pairs = 3
        max_test_pairs = 2
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((max_train_pairs, grid_size, grid_size), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((max_test_pairs, grid_size, grid_size), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(456, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=max_train_pairs,
            max_test_pairs=max_test_pairs,
            step_count=7,
            episode_mode=jax_types.EPISODE_MODE_TRAIN,
            action_history_length=3,
        )

    def test_create_observation_target_masking(self, sample_state: ArcEnvState):
        """Test target grid masking in different modes."""
        # Test training mode - target should be included
        train_config = ObservationConfig(include_target_grid=True)
        train_obs = create_observation(sample_state, train_config)
        assert train_obs.has_target_access()  # Should have meaningful target data
        
        # Test with target disabled
        no_target_config = ObservationConfig(include_target_grid=False)
        no_target_obs = create_observation(sample_state, no_target_config)
        assert not no_target_obs.has_target_access()  # Should be masked (zeros)
        
        # Test test mode - target should be masked even if config includes it
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        test_obs = create_observation(test_state, train_config)
        assert not test_obs.has_target_access()  # Masked in test mode


class TestObservationConvenienceFunctions:
    """Test convenience functions for creating observations."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for testing."""
        grid_size = 5
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((3, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((3, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((3, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((3, grid_size, grid_size), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(789, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=3,
            max_test_pairs=2,
        )

    def test_create_minimal_observation(self, sample_state: ArcEnvState):
        """Test create_minimal_observation convenience function."""
        observation = create_minimal_observation(sample_state)
        
        # Should be minimal format - recent actions should be masked (all zeros)
        assert jnp.all(observation.recent_actions == 0)  # Masked when disabled
        assert observation.working_grid is not None
        assert observation.episode_mode == jax_types.EPISODE_MODE_TRAIN

    def test_create_evaluation_observation(self, sample_state: ArcEnvState):
        """Test create_evaluation_observation convenience function."""
        # Test with training state (target should still be hidden)
        train_state = sample_state
        eval_obs = create_evaluation_observation(train_state)
        
        # Should hide target even in training state
        assert not eval_obs.has_target_access()  # Target should be masked
        assert eval_obs.working_grid is not None
        
        # Test with test state
        test_state = eqx.tree_at(
            lambda s: s.episode_mode,
            sample_state,
            jnp.array(jax_types.EPISODE_MODE_TEST)
        )
        test_eval_obs = create_evaluation_observation(test_state)
        assert not test_eval_obs.has_target_access()  # Target should be masked


class TestJAXCompatibility:
    """Test JAX compatibility of all new state and observation operations."""

    @pytest.fixture
    def sample_state(self) -> ArcEnvState:
        """Create sample state for JAX testing."""
        grid_size = 4
        
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((2, grid_size, grid_size), dtype=jnp.int32),
            input_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            output_grids_examples=jnp.ones((2, grid_size, grid_size), dtype=jnp.int32),
            output_masks_examples=jnp.ones((2, grid_size, grid_size), dtype=bool),
            num_train_pairs=2,
            test_input_grids=jnp.zeros((1, grid_size, grid_size), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, grid_size, grid_size), dtype=bool),
            true_test_output_grids=jnp.ones((1, grid_size, grid_size), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, grid_size, grid_size), dtype=bool),
            num_test_pairs=1,
            task_index=jnp.array(999, dtype=jnp.int32),
        )
        
        working_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        working_grid_mask = jnp.ones((grid_size, grid_size), dtype=bool)
        target_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        return create_arc_env_state(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            max_train_pairs=2,
            max_test_pairs=1,
        )

    def test_jit_compilation_state_methods(self, sample_state: ArcEnvState):
        """Test JIT compilation of state utility methods."""
        @jax.jit
        def test_state_methods(state):
            return {
                'is_training': state.is_training_mode(),
                'is_test': state.is_test_mode(),
                'demo_count': state.get_available_demo_count(),
                'test_count': state.get_available_test_count(),
                'completed_demos': state.get_completed_demo_count(),
                'completed_tests': state.get_completed_test_count(),
                'allowed_ops': state.get_allowed_operations_count(),
                'history_length': state.get_action_history_length(),
                'has_history': state.has_action_history(),
                'current_completed': state.is_current_pair_completed(),
            }
        
        # Should compile and execute without errors
        result = test_state_methods(sample_state)
        
        # Verify results
        assert result['is_training']
        assert not result['is_test']
        assert result['demo_count'] > 0
        assert result['history_length'] == 0
        assert not result['has_history']

    def test_jit_compilation_observation_creation(self, sample_state: ArcEnvState):
        """Test JIT compilation of observation creation."""
        config = ObservationConfig()
        
        @jax.jit
        def create_obs_jit(state):
            return create_observation(state, config)
        
        # Should compile and execute without errors
        observation = create_obs_jit(sample_state)
        
        # Verify observation properties
        assert observation.working_grid is not None
        assert observation.episode_mode == jax_types.EPISODE_MODE_TRAIN
        assert observation.is_training_mode()


if __name__ == "__main__":
    pytest.main([__file__])