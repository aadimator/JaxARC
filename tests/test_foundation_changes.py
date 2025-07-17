"""
Tests for foundation changes in the codebase refactoring.

This module tests the centralized state definition, JAXTyping type validation,
simplified arc_step function, and backward compatibility with existing code.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc.envs import (
    ArcEnvConfig,
    ActionConfig,
    GridConfig,
    RewardConfig,
    arc_reset,
    arc_step,
    create_standard_config,
)
from jaxarc.envs.functional import (
    _calculate_reward,
    _ensure_config,
    _get_observation,
    _is_episode_done,
    _validate_operation,
)
from jaxarc.state import ArcEnvState
from jaxarc.types import ARCLEAction, JaxArcTask
from jaxarc.utils.jax_types import (
    GridArray,
    MaskArray,
    SelectionArray,
    SimilarityScore,
    StepCount,
    EpisodeDone,
    EpisodeIndex,
    OperationId,
    ColorValue,
    PointCoords,
    BboxCoords,
    ContinuousSelectionArray,
    TaskIndex,
)


class TestCentralizedStateDefinition:
    """Test the centralized ArcEnvState definition."""

    def test_arc_env_state_creation(self):
        """Test ArcEnvState creation with proper types."""
        # Create test task data
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 5, 5), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 5, 5), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 5, 5), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 5, 5), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Create state with JAXTyping annotations
        working_grid: GridArray = jnp.zeros((5, 5), dtype=jnp.int32)
        working_grid_mask: MaskArray = jnp.ones((5, 5), dtype=jnp.bool_)
        target_grid: GridArray = jnp.ones((5, 5), dtype=jnp.int32)
        selected: SelectionArray = jnp.zeros((5, 5), dtype=jnp.bool_)
        clipboard: GridArray = jnp.zeros((5, 5), dtype=jnp.int32)
        
        state = ArcEnvState(
            task_data=task_data,
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            target_grid=target_grid,
            step_count=jnp.array(0, dtype=jnp.int32),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0, dtype=jnp.int32),
            selected=selected,
            clipboard=clipboard,
            similarity_score=jnp.array(0.5, dtype=jnp.float32),
        )

        # Verify state structure
        assert isinstance(state, ArcEnvState)
        assert state.task_data is task_data
        chex.assert_shape(state.working_grid, (5, 5))
        chex.assert_shape(state.target_grid, (5, 5))
        chex.assert_shape(state.selected, (5, 5))
        chex.assert_shape(state.clipboard, (5, 5))
        
        # Verify scalar fields
        chex.assert_shape(state.step_count, ())
        chex.assert_shape(state.episode_done, ())
        chex.assert_shape(state.current_example_idx, ())
        chex.assert_shape(state.similarity_score, ())

    def test_arc_env_state_validation(self):
        """Test ArcEnvState validation in __post_init__."""
        # Create valid state
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Valid state should not raise
        state = ArcEnvState(
            task_data=task_data,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=jnp.bool_),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0, dtype=jnp.int32),
            selected=jnp.zeros((3, 3), dtype=jnp.bool_),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
        )
        
        # Should validate successfully
        assert isinstance(state, ArcEnvState)

    def test_arc_env_state_immutability(self):
        """Test that ArcEnvState is immutable and uses replace() for updates."""
        # Create initial state
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        original_state = ArcEnvState(
            task_data=task_data,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=jnp.bool_),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0, dtype=jnp.int32),
            selected=jnp.zeros((3, 3), dtype=jnp.bool_),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
        )

        # Test replace() method
        new_state = original_state.replace(step_count=jnp.array(1, dtype=jnp.int32))
        
        # Original state should be unchanged
        assert original_state.step_count == 0
        assert new_state.step_count == 1
        assert original_state is not new_state

    def test_arc_env_state_jax_compatibility(self):
        """Test that ArcEnvState works with JAX transformations."""
        def increment_step(state: ArcEnvState) -> ArcEnvState:
            return state.replace(step_count=state.step_count + 1)

        # Create test state
        task_data = JaxArcTask(
            input_grids_examples=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            input_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            output_grids_examples=jnp.ones((1, 3, 3), dtype=jnp.int32),
            output_masks_examples=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_train_pairs=1,
            test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
            test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
            true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        state = ArcEnvState(
            task_data=task_data,
            working_grid=jnp.zeros((3, 3), dtype=jnp.int32),
            working_grid_mask=jnp.ones((3, 3), dtype=jnp.bool_),
            target_grid=jnp.ones((3, 3), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0, dtype=jnp.int32),
            selected=jnp.zeros((3, 3), dtype=jnp.bool_),
            clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
        )

        # Test JIT compilation
        jitted_increment = jax.jit(increment_step)
        new_state = jitted_increment(state)
        
        assert new_state.step_count == 1
        assert isinstance(new_state, ArcEnvState)


class TestJAXTypingValidation:
    """Test JAXTyping type validation and compatibility."""

    def test_grid_array_type_validation(self):
        """Test GridArray type validation with different shapes."""
        # Single grid
        single_grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        assert single_grid.shape == (2, 2)
        assert single_grid.dtype == jnp.int32

        # Batched grids
        batch_grids: GridArray = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32)
        assert batch_grids.shape == (2, 2, 2)
        assert batch_grids.dtype == jnp.int32

        # Both should work with JAX operations
        single_sum = jnp.sum(single_grid)
        batch_sum = jnp.sum(batch_grids, axis=(1, 2))
        
        assert isinstance(single_sum, jnp.ndarray)
        assert batch_sum.shape == (2,)

    def test_mask_array_type_validation(self):
        """Test MaskArray type validation."""
        # Single mask
        single_mask: MaskArray = jnp.array([[True, False], [False, True]])
        assert single_mask.shape == (2, 2)
        assert single_mask.dtype == jnp.bool_

        # Batched masks
        batch_masks: MaskArray = jnp.array([[[True, False], [False, True]], [[False, True], [True, False]]])
        assert batch_masks.shape == (2, 2, 2)
        assert batch_masks.dtype == jnp.bool_

    def test_selection_array_type_validation(self):
        """Test SelectionArray type validation."""
        selection: SelectionArray = jnp.array([[True, False], [False, True]])
        assert selection.shape == (2, 2)
        assert selection.dtype == jnp.bool_

        # Test with grid operations
        grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        masked_grid = jnp.where(selection, grid, 0)
        assert masked_grid.shape == (2, 2)

    def test_continuous_selection_array_validation(self):
        """Test ContinuousSelectionArray type validation."""
        continuous_selection: ContinuousSelectionArray = jnp.array([[0.5, 0.8], [0.2, 1.0]], dtype=jnp.float32)
        assert continuous_selection.shape == (2, 2)
        assert continuous_selection.dtype == jnp.float32
        
        # Values should be in [0, 1] range
        assert jnp.min(continuous_selection) >= 0.0
        assert jnp.max(continuous_selection) <= 1.0

    def test_scalar_types_validation(self):
        """Test scalar type validation."""
        # Step count
        step_count: StepCount = jnp.array(42, dtype=jnp.int32)
        assert step_count.shape == ()
        assert step_count.dtype == jnp.int32

        # Episode done
        episode_done: EpisodeDone = jnp.array(True)
        assert episode_done.shape == ()
        assert episode_done.dtype == jnp.bool_

        # Episode index
        episode_idx: EpisodeIndex = jnp.array(5, dtype=jnp.int32)
        assert episode_idx.shape == ()
        assert episode_idx.dtype == jnp.int32

        # Similarity score
        similarity: SimilarityScore = jnp.array(0.85, dtype=jnp.float32)
        assert similarity.shape == ()
        assert similarity.dtype in (jnp.float32, jnp.float64)

    def test_action_types_validation(self):
        """Test action-related type validation."""
        # Point coordinates
        point: PointCoords = jnp.array([2, 3], dtype=jnp.int32)
        assert point.shape == (2,)
        assert point.dtype == jnp.int32

        # Bounding box coordinates
        bbox: BboxCoords = jnp.array([1, 1, 3, 3], dtype=jnp.int32)
        assert bbox.shape == (4,)
        assert bbox.dtype == jnp.int32

        # Operation ID
        operation: OperationId = jnp.array(15, dtype=jnp.int32)
        assert operation.shape == ()
        assert operation.dtype == jnp.int32

        # Color value
        color: ColorValue = jnp.array(7, dtype=jnp.int32)
        assert color.shape == ()
        assert color.dtype == jnp.int32

    def test_type_compatibility_with_jax_operations(self):
        """Test that typed arrays work seamlessly with JAX operations."""
        # Create typed arrays
        grid1: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2: GridArray = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
        mask: MaskArray = jnp.array([[True, False], [True, True]])

        # Test arithmetic operations
        result = grid1 + grid2
        assert result.shape == (2, 2)
        assert result.dtype == jnp.int32

        # Test conditional operations
        conditional_result = jnp.where(mask, grid1, grid2)
        assert conditional_result.shape == (2, 2)

        # Test reduction operations
        sum_result = jnp.sum(grid1)
        assert isinstance(sum_result, jnp.ndarray)

        # Test broadcasting
        broadcast_result = grid1 * mask
        assert broadcast_result.shape == (2, 2)

    def test_batch_operations_with_typed_arrays(self):
        """Test batch operations with typed arrays."""
        # Create batch arrays
        batch_grids: GridArray = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32)
        batch_masks: MaskArray = jnp.array([[[True, False], [False, True]], [[False, True], [True, False]]])

        # Test vmap-style operations
        def process_single_grid(grid, mask):
            return jnp.sum(grid * mask)

        # Apply to batch
        batch_results = jax.vmap(process_single_grid)(batch_grids, batch_masks)
        assert batch_results.shape == (2,)
        assert batch_results.dtype == jnp.int32


class TestSimplifiedArcStepFunction:
    """Test the simplified arc_step function and its components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.config = create_standard_config(max_episode_steps=10)

    def test_ensure_config_function(self):
        """Test _ensure_config function with different config types."""
        # Test with typed config
        typed_config = create_standard_config()
        result = _ensure_config(typed_config)
        assert result is typed_config
        assert isinstance(result, ArcEnvConfig)

        # Test with Hydra DictConfig
        hydra_config = OmegaConf.create({
            "max_episode_steps": 50,
            "reward": {"success_bonus": 10.0},
            "grid": {"max_grid_height": 20},
        })
        result = _ensure_config(hydra_config)
        assert isinstance(result, ArcEnvConfig)
        assert result.max_episode_steps == 50

    def test_validate_operation_function(self):
        """Test _validate_operation function."""
        config = create_standard_config()

        # Test with integer
        result = _validate_operation(5, config)
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.int32
        assert result == 5

        # Test with jnp.array
        op_array = jnp.array(10, dtype=jnp.int32)
        result = _validate_operation(op_array, config)
        assert isinstance(result, jnp.ndarray)
        assert result == 10

        # Test with invalid type
        with pytest.raises(ValueError, match="Operation must be int or jnp.ndarray"):
            _validate_operation("invalid", config)

    def test_validate_operation_clipping(self):
        """Test operation clipping functionality."""
        # Create config with clipping enabled
        action_config = ActionConfig(
            selection_format="mask",
            validate_actions=True,
            clip_invalid_actions=True,
            num_operations=35,
        )
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=action_config,
        )

        # Test clipping of invalid operation
        result = _validate_operation(999, config)
        assert result <= 34  # Should be clipped to max valid operation

        result = _validate_operation(-5, config)
        assert result >= 0  # Should be clipped to min valid operation

    def test_get_observation_function(self):
        """Test _get_observation function."""
        # Create test state
        state, _ = arc_reset(self.key, self.config)
        
        # Get observation
        obs = _get_observation(state, self.config)
        
        # Should return working grid
        assert isinstance(obs, jnp.ndarray)
        chex.assert_rank(obs, 2)
        assert jnp.array_equal(obs, state.working_grid)

    def test_calculate_reward_function(self):
        """Test _calculate_reward function."""
        # Create test states
        old_state, _ = arc_reset(self.key, self.config)
        new_state = old_state.replace(similarity_score=jnp.array(0.8, dtype=jnp.float32))

        # Calculate reward
        reward = _calculate_reward(old_state, new_state, self.config)
        
        assert isinstance(reward, jnp.ndarray)
        chex.assert_rank(reward, 0)  # Should be scalar
        assert reward.dtype in (jnp.float32, jnp.float64)

    def test_calculate_reward_components(self):
        """Test individual reward components."""
        # Create config with reward_on_submit_only=False to test all reward components
        reward_config = RewardConfig(
            reward_on_submit_only=False,
            step_penalty=-0.01,
            success_bonus=10.0,
            similarity_weight=1.0,
            progress_bonus=0.5,
        )
        config = create_standard_config().replace(reward=reward_config)
        
        # Create states with different similarity scores
        old_state, _ = arc_reset(self.key, config)
        old_state = old_state.replace(similarity_score=jnp.array(0.5, dtype=jnp.float32))
        
        # Improved similarity
        improved_state = old_state.replace(similarity_score=jnp.array(0.8, dtype=jnp.float32))
        reward_improved = _calculate_reward(old_state, improved_state, config)
        
        # Decreased similarity
        decreased_state = old_state.replace(similarity_score=jnp.array(0.3, dtype=jnp.float32))
        reward_decreased = _calculate_reward(old_state, decreased_state, config)
        
        # Perfect similarity (success)
        perfect_state = old_state.replace(similarity_score=jnp.array(1.0, dtype=jnp.float32))
        reward_perfect = _calculate_reward(old_state, perfect_state, config)
        
        # Improved should be better than decreased
        assert reward_improved > reward_decreased
        # Perfect should include success bonus
        assert reward_perfect > reward_improved

    def test_is_episode_done_function(self):
        """Test _is_episode_done function."""
        config = create_standard_config(max_episode_steps=5)
        
        # Create test state
        state, _ = arc_reset(self.key, config)
        
        # Episode should not be done initially
        done = _is_episode_done(state, config)
        assert done == False

        # Test max steps reached
        max_steps_state = state.replace(step_count=jnp.array(5, dtype=jnp.int32))
        done = _is_episode_done(max_steps_state, config)
        assert done == True

        # Test task solved
        solved_state = state.replace(similarity_score=jnp.array(1.0, dtype=jnp.float32))
        done = _is_episode_done(solved_state, config)
        assert done == True

        # Test submitted
        submitted_state = state.replace(episode_done=jnp.array(True))
        done = _is_episode_done(submitted_state, config)
        assert done == True

    def test_arc_step_function_integration(self):
        """Test the complete arc_step function."""
        # Reset environment
        state, obs = arc_reset(self.key, self.config)
        
        # Create action
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        
        # Execute step
        new_state, new_obs, reward, done, info = arc_step(state, action, self.config)
        
        # Verify return types and structure
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(new_obs, jnp.ndarray)
        assert isinstance(reward, jnp.ndarray)
        assert isinstance(done, jnp.ndarray)
        assert isinstance(info, dict)
        
        # Verify state progression
        assert new_state.step_count == state.step_count + 1
        
        # Verify info dict contents
        required_keys = ["success", "similarity", "step_count", "similarity_improvement"]
        for key in required_keys:
            assert key in info

    def test_arc_step_with_different_action_formats(self):
        """Test arc_step with different action formats."""
        # Test with mask format
        mask_config = create_standard_config()
        mask_config = mask_config.replace(
            action=mask_config.action.replace(selection_format="mask")
        )
        
        state, _ = arc_reset(self.key, mask_config)
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        
        new_state, _, _, _, _ = arc_step(state, action, mask_config)
        assert new_state.step_count == 1

        # Test with point format
        from jaxarc.envs import create_point_config
        point_config = create_point_config(max_episode_steps=10)
        
        state, _ = arc_reset(self.key, point_config)
        action = {
            "point": jnp.array([1, 1], dtype=jnp.int32),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        
        new_state, _, _, _, _ = arc_step(state, action, point_config)
        assert new_state.step_count == 1

        # Test with bbox format
        from jaxarc.envs import create_bbox_config
        bbox_config = create_bbox_config(max_episode_steps=10)
        
        state, _ = arc_reset(self.key, bbox_config)
        action = {
            "bbox": jnp.array([0, 0, 2, 2], dtype=jnp.int32),
            "operation": jnp.array(2, dtype=jnp.int32),
        }
        
        new_state, _, _, _, _ = arc_step(state, action, bbox_config)
        assert new_state.step_count == 1

    def test_arc_step_error_handling(self):
        """Test arc_step error handling."""
        state, _ = arc_reset(self.key, self.config)
        
        # Test missing operation field
        with pytest.raises(ValueError, match="must contain 'operation'"):
            arc_step(state, {"selection": jnp.ones((3, 3), dtype=jnp.bool_)}, self.config)
        
        # Test missing selection field
        with pytest.raises(ValueError, match="must contain 'selection'"):
            arc_step(state, {"operation": 0}, self.config)
        
        # Test invalid action type
        with pytest.raises(ValueError, match="Action must be a dictionary"):
            arc_step(state, "invalid_action", self.config)

    def test_arc_step_with_arcle_action(self):
        """Test arc_step with ARCLEAction input."""
        state, _ = arc_reset(self.key, self.config)
        
        # Create ARCLEAction
        arcle_action = ARCLEAction(
            selection=jnp.ones_like(state.working_grid, dtype=jnp.float32),
            operation=jnp.array(5, dtype=jnp.int32),
            agent_id=1,
            timestamp=100,
        )
        
        # Should convert to dict format internally
        new_state, _, _, _, _ = arc_step(state, arcle_action, self.config)
        assert new_state.step_count == 1


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_state_import_compatibility(self):
        """Test that ArcEnvState can be imported from expected locations."""
        # Should be importable from state module
        from jaxarc.state import ArcEnvState as StateArcEnvState
        
        # Should be the same class
        assert StateArcEnvState is ArcEnvState

    def test_jax_types_import_compatibility(self):
        """Test that JAX types can be imported from expected locations."""
        # Should be importable from utils
        from jaxarc.utils import GridArray as UtilsGridArray
        from jaxarc.utils.jax_types import GridArray as JaxTypesGridArray
        
        # Should be the same type
        assert UtilsGridArray is JaxTypesGridArray

    def test_functional_api_compatibility(self):
        """Test that functional API maintains compatibility."""
        key = jax.random.PRNGKey(42)
        config = create_standard_config()
        
        # Test arc_reset
        state, obs = arc_reset(key, config)
        assert isinstance(state, ArcEnvState)
        assert isinstance(obs, jnp.ndarray)
        
        # Test arc_step
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        
        new_state, new_obs, reward, done, info = arc_step(state, action, config)
        
        # Should return expected types
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(new_obs, jnp.ndarray)
        assert isinstance(reward, jnp.ndarray)
        assert isinstance(done, jnp.ndarray)
        assert isinstance(info, dict)

    def test_config_compatibility(self):
        """Test that config system maintains compatibility."""
        # Test factory functions still work
        from jaxarc.envs import (
            create_standard_config,
            create_point_config,
            create_bbox_config,
        )
        
        configs = [
            create_standard_config(),
            create_point_config(),
            create_bbox_config(),
        ]
        
        for config in configs:
            assert isinstance(config, ArcEnvConfig)
            # Should work with functional API
            key = jax.random.PRNGKey(42)
            state, obs = arc_reset(key, config)
            assert isinstance(state, ArcEnvState)

    def test_hydra_config_compatibility(self):
        """Test that Hydra config integration still works."""
        hydra_config = OmegaConf.create({
            "max_episode_steps": 25,
            "reward": {"success_bonus": 15.0},
            "grid": {"max_grid_height": 15},
            "action": {"selection_format": "mask"},
        })
        
        # Should work with functional API
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, hydra_config)
        assert isinstance(state, ArcEnvState)
        
        # Should work with arc_step
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        
        new_state, _, _, _, _ = arc_step(state, action, hydra_config)
        assert new_state.step_count == 1

    def test_jax_transformations_compatibility(self):
        """Test that JAX transformations still work with foundation changes."""
        key = jax.random.PRNGKey(42)
        config = create_standard_config()
        
        # Test JIT compilation
        jitted_reset = jax.jit(arc_reset, static_argnums=(1,))
        state, obs = jitted_reset(key, config)
        assert isinstance(state, ArcEnvState)
        
        # Test with arc_step
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        
        jitted_step = jax.jit(arc_step, static_argnums=(2,))
        new_state, _, _, _, _ = jitted_step(state, action, config)
        assert isinstance(new_state, ArcEnvState)
        assert new_state.step_count == 1

    def test_existing_test_compatibility(self):
        """Test that existing tests should still pass with foundation changes."""
        # This test verifies that the foundation changes don't break existing functionality
        # by running some key operations that existing tests depend on
        
        key = jax.random.PRNGKey(42)
        config = create_standard_config(max_episode_steps=5)
        
        # Reset and step multiple times
        state, obs = arc_reset(key, config)
        
        for i in range(3):
            action = {
                "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
                "operation": jnp.array(i % 3, dtype=jnp.int32),
            }
            
            state, obs, reward, done, info = arc_step(state, action, config)
            
            # Verify expected behavior
            assert state.step_count == i + 1
            assert isinstance(reward, jnp.ndarray)
            assert isinstance(done, jnp.ndarray)
            assert "success" in info
            assert "similarity" in info


if __name__ == "__main__":
    pytest.main([__file__])