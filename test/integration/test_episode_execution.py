"""
Integration tests for episode execution workflows.

This module tests complete reset-step-done episode cycles, action creation
to grid modification pipeline, and episode termination conditions.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import create_action
from jaxarc.registration import available_task_ids, make
from jaxarc.types import StepType


class TestEpisodeExecution:
    """Test complete reset-step-done episode cycles."""

    @pytest.fixture
    def sample_environment(self):
        """Create a sample environment for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        if not task_ids:
            pytest.skip("No Mini tasks available")
        
        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_basic_episode_cycle(self, sample_environment):
        """Test basic reset-step-done episode cycle."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(42)
        
        # Reset environment
        timestep = env.reset(env_params, key)
        
        # Verify initial timestep
        assert timestep.step_type == StepType.FIRST
        assert timestep.reward == 0.0
        assert timestep.discount == 1.0
        assert timestep.observation is not None
        assert timestep.state is not None
        
        # Verify observation shape
        expected_shape = (
            env_params.dataset.max_grid_height,
            env_params.dataset.max_grid_width
        )
        chex.assert_shape(timestep.observation, expected_shape)
        
        # Verify state initialization
        assert timestep.state.step_count == 0
        assert timestep.state.task_idx >= 0
        assert timestep.state.pair_idx >= 0
        
        # Take a step
        grid_shape = timestep.observation.shape
        action = create_action(
            operation=jnp.array(1, dtype=jnp.int32),  # Fill with color 1
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
        )
        
        next_timestep = env.step(env_params, timestep, action)
        
        # Verify step progression
        assert next_timestep.step_type in [StepType.MID, StepType.TERMINATED, StepType.TRUNCATED]
        assert next_timestep.state.step_count == 1
        assert next_timestep.observation is not None
        
        # Verify grid modification occurred
        # The working grid should have been modified at position [0, 0]
        if next_timestep.state.working_grid_mask[0, 0]:  # If position is valid
            assert next_timestep.state.working_grid[0, 0] == 1

    def test_multi_step_episode(self, sample_environment):
        """Test episode with multiple steps."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(123)
        
        # Reset environment
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Take multiple steps
        num_steps = 5
        for i in range(num_steps):
            # Create action for different positions
            row = i % grid_shape[0]
            col = i % grid_shape[1]
            color = (i % 9) + 1  # Colors 1-9
            
            action = create_action(
                operation=jnp.array(color, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[row, col].set(True)
            )
            
            prev_step_count = timestep.state.step_count
            timestep = env.step(env_params, timestep, action)
            
            # Verify step progression
            assert timestep.state.step_count == prev_step_count + 1
            assert timestep.step_type in [StepType.MID, StepType.TERMINATED, StepType.TRUNCATED]
            
            # If episode ended, break
            if timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]:
                break
        
        # Verify final state
        assert timestep.state.step_count <= num_steps

    def test_episode_termination_conditions(self, sample_environment):
        """Test various episode termination conditions."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(456)
        
        # Test max episode steps termination
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        max_steps = env_params.max_episode_steps
        
        # Take steps until max episode length
        for i in range(max_steps + 1):  # +1 to potentially exceed limit
            if timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]:
                break
            
            # Simple action
            action = create_action(
                operation=jnp.array(1, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
            )
            
            timestep = env.step(env_params, timestep, action)
        
        # Should terminate by max steps
        assert timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]
        assert timestep.state.step_count <= max_steps

    def test_episode_reproducibility(self, sample_environment):
        """Test that episodes are reproducible with same PRNG key."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(789)
        
        # Run first episode
        timestep1 = env.reset(env_params, key)
        grid_shape = timestep1.observation.shape
        
        action = create_action(
            operation=jnp.array(2, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[1, 1].set(True)
        )
        
        timestep1 = env.step(env_params, timestep1, action)
        
        # Run second episode with same key
        timestep2 = env.reset(env_params, key)
        timestep2 = env.step(env_params, timestep2, action)
        
        # Results should be identical
        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(timestep1.state.working_grid, timestep2.state.working_grid)
        assert timestep1.state.task_idx == timestep2.state.task_idx
        assert timestep1.state.pair_idx == timestep2.state.pair_idx

    def test_episode_state_consistency(self, sample_environment):
        """Test that episode state remains consistent throughout execution."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(999)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Track state consistency
        initial_task_idx = timestep.state.task_idx
        initial_pair_idx = timestep.state.pair_idx
        
        # Take several steps
        for i in range(3):
            action = create_action(
                operation=jnp.array(i + 1, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[i, i].set(True)
            )
            
            timestep = env.step(env_params, timestep, action)
            
            # Task and pair indices should remain constant
            assert timestep.state.task_idx == initial_task_idx
            assert timestep.state.pair_idx == initial_pair_idx
            
            # Step count should increment
            assert timestep.state.step_count == i + 1
            
            # Observation should match working grid
            chex.assert_trees_all_close(
                timestep.observation,
                timestep.state.working_grid
            )


class TestActionGridModificationPipeline:
    """Test action creation to grid modification pipeline."""

    @pytest.fixture
    def sample_environment(self):
        """Create a sample environment for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        if not task_ids:
            pytest.skip("No Mini tasks available")
        
        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_fill_operation_pipeline(self, sample_environment):
        """Test fill operation from action creation to grid modification."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(42)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Create fill action
        target_color = 5
        target_position = (2, 3)
        
        action = create_action(
            operation=jnp.array(target_color, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[target_position].set(True)
        )
        
        # Execute action
        next_timestep = env.step(env_params, timestep, action)
        
        # Verify grid modification
        if next_timestep.state.working_grid_mask[target_position]:
            assert next_timestep.state.working_grid[target_position] == target_color

    def test_multiple_cell_selection(self, sample_environment):
        """Test action with multiple cell selection."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(123)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Create selection covering multiple cells
        selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)  # 2x2 square
        
        target_color = 7
        action = create_action(
            operation=jnp.array(target_color, dtype=jnp.int32),
            selection=selection
        )
        
        # Execute action
        next_timestep = env.step(env_params, timestep, action)
        
        # Verify all selected cells were modified
        for i in range(1, 3):
            for j in range(1, 3):
                if next_timestep.state.working_grid_mask[i, j]:
                    assert next_timestep.state.working_grid[i, j] == target_color

    def test_invalid_action_handling(self, sample_environment):
        """Test handling of invalid actions in the pipeline."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(456)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Create action with invalid operation (out of range)
        invalid_action = create_action(
            operation=jnp.array(15, dtype=jnp.int32),  # Invalid color
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
        )
        
        # Execute action - should handle gracefully
        next_timestep = env.step(env_params, timestep, invalid_action)
        
        # Episode should continue (not crash)
        assert next_timestep.step_type in [StepType.MID, StepType.TERMINATED, StepType.TRUNCATED]
        assert next_timestep.state.step_count == 1

    def test_empty_selection_handling(self, sample_environment):
        """Test handling of actions with empty selection."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(789)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Create action with empty selection
        empty_action = create_action(
            operation=jnp.array(3, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_)  # No cells selected
        )
        
        # Execute action
        next_timestep = env.step(env_params, timestep, empty_action)
        
        # Episode should continue
        assert next_timestep.step_type in [StepType.MID, StepType.TERMINATED, StepType.TRUNCATED]
        assert next_timestep.state.step_count == 1
        
        # Grid should remain unchanged
        chex.assert_trees_all_close(
            timestep.state.working_grid,
            next_timestep.state.working_grid
        )

    def test_action_validation_pipeline(self, sample_environment):
        """Test action validation in the pipeline."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(111)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Test various action types
        actions = [
            # Valid fill action
            create_action(
                operation=jnp.array(1, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
            ),
            # Valid multi-cell action
            create_action(
                operation=jnp.array(2, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0:2, 0:2].set(True)
            ),
            # Edge case: selection at boundary
            create_action(
                operation=jnp.array(3, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[-1, -1].set(True)
            ),
        ]
        
        for i, action in enumerate(actions):
            if timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]:
                break
            
            prev_step_count = timestep.state.step_count
            timestep = env.step(env_params, timestep, action)
            
            # Verify step progression
            assert timestep.state.step_count == prev_step_count + 1
            assert timestep.step_type in [StepType.MID, StepType.TERMINATED, StepType.TRUNCATED]

    def test_grid_mask_interaction(self, sample_environment):
        """Test interaction between actions and grid masks."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(222)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Get initial mask
        initial_mask = timestep.state.working_grid_mask
        
        # Create action targeting both masked and unmasked areas
        selection = jnp.ones(grid_shape, dtype=jnp.bool_)  # Select all cells
        
        action = create_action(
            operation=jnp.array(4, dtype=jnp.int32),
            selection=selection
        )
        
        # Execute action
        next_timestep = env.step(env_params, timestep, action)
        
        # Verify only valid (masked) cells were modified
        modified_cells = (
            next_timestep.state.working_grid != timestep.state.working_grid
        )
        
        # Modified cells should be subset of valid cells
        assert jnp.all(modified_cells <= initial_mask)


class TestEpisodeTerminationConditions:
    """Test episode termination and truncation conditions."""

    @pytest.fixture
    def sample_environment(self):
        """Create a sample environment for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        if not task_ids:
            pytest.skip("No Mini tasks available")
        
        task_id = task_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_max_episode_steps_termination(self, sample_environment):
        """Test termination due to maximum episode steps."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(42)
        
        # Modify environment to have short episodes for testing
        import equinox as eqx
        
        short_env_params = eqx.tree_at(
            lambda p: p.max_episode_steps, env_params, 5
        )
        
        timestep = env.reset(short_env_params, key)
        grid_shape = timestep.observation.shape
        
        # Take steps until termination
        step_count = 0
        while timestep.step_type not in [StepType.TERMINATED, StepType.TRUNCATED] and step_count < 10:
            action = create_action(
                operation=jnp.array(1, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
            )
            
            timestep = env.step(short_env_params, timestep, action)
            step_count += 1
        
        # Should terminate due to max steps
        assert timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]
        assert timestep.state.step_count <= 5

    def test_success_termination(self, sample_environment):
        """Test termination due to successful task completion."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(123)
        
        timestep = env.reset(env_params, key)
        
        # Check if task can be solved trivially (target matches input)
        if jnp.array_equal(
            timestep.state.working_grid * timestep.state.working_grid_mask,
            timestep.state.target_grid * timestep.state.target_grid_mask
        ):
            # Task is already solved, should have high similarity
            assert timestep.state.similarity_score >= 0.99
        
        # Note: Testing actual success termination would require knowing
        # the specific task solution, which varies by task

    def test_episode_discount_and_reward(self, sample_environment):
        """Test episode discount and reward behavior during termination."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(456)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Initial timestep should have discount = 1.0, reward = 0.0
        assert timestep.discount == 1.0
        assert timestep.reward == 0.0
        
        # Take a step
        action = create_action(
            operation=jnp.array(2, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[1, 1].set(True)
        )
        
        next_timestep = env.step(env_params, timestep, action)
        
        # Should have some reward (positive or negative)
        assert isinstance(next_timestep.reward, (float, jnp.ndarray))
        
        # Discount should be appropriate for step type
        if next_timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]:
            assert next_timestep.discount == 0.0
        else:
            assert next_timestep.discount == 1.0

    def test_episode_info_tracking(self, sample_environment):
        """Test episode information tracking through termination."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(789)
        
        timestep = env.reset(env_params, key)
        grid_shape = timestep.observation.shape
        
        # Track episode progression
        initial_similarity = timestep.state.similarity_score
        
        # Take a step that might change similarity
        action = create_action(
            operation=jnp.array(3, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True)
        )
        
        next_timestep = env.step(env_params, timestep, action)
        
        # Similarity might have changed
        final_similarity = next_timestep.state.similarity_score
        
        # Verify similarity is in valid range
        assert 0.0 <= final_similarity <= 1.0
        assert 0.0 <= initial_similarity <= 1.0

    def test_episode_state_at_termination(self, sample_environment):
        """Test episode state properties at termination."""
        env, env_params = sample_environment
        key = jax.random.PRNGKey(111)
        
        # Force short episode for testing
        import equinox as eqx
        
        short_env_params = eqx.tree_at(
            lambda p: p.max_episode_steps, env_params, 3
        )
        
        timestep = env.reset(short_env_params, key)
        grid_shape = timestep.observation.shape
        
        # Run until termination
        while timestep.step_type not in [StepType.TERMINATED, StepType.TRUNCATED]:
            action = create_action(
                operation=jnp.array(1, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True)
            )
            
            timestep = env.step(short_env_params, timestep, action)
        
        # Verify final state properties
        assert timestep.step_type in [StepType.TERMINATED, StepType.TRUNCATED]
        assert timestep.discount == 0.0
        assert timestep.state.step_count > 0
        assert timestep.state.step_count <= 3
        
        # State should still be valid
        assert timestep.state.task_idx >= 0
        assert timestep.state.pair_idx >= 0
        assert 0.0 <= timestep.state.similarity_score <= 1.0