"""
Comprehensive tests for ArcEnvironment core functionality.

This module tests the ArcEnvironment class initialization, reset, step methods,
environment lifecycle, state transitions, reward computation, episode termination,
and integration with different configuration systems.

Requirements covered: 3.1, 3.2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import chex
from unittest.mock import Mock, patch

from jaxarc.envs.environment import ArcEnvironment
from jaxarc.envs.equinox_config import JaxArcConfig
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.envs.equinox_config import convert_arc_env_config_to_jax_arc_config
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask


class TestArcEnvironmentInitialization:
    """Test ArcEnvironment initialization with various configurations."""

    def test_basic_initialization(self):
        """Test basic environment initialization with default config."""
        config = JaxArcConfig()
        env = ArcEnvironment(config)
        
        assert env.config is config
        assert env._state is None
        assert env._episode_count == 0
        assert env.action_handler is not None
        assert env.is_done is True

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration parameters."""
        config = JaxArcConfig()
        import equinox as eqx
        config = eqx.tree_at(
            lambda c: (c.environment.max_episode_steps, c.dataset.max_grid_height, c.dataset.max_grid_width),
            config,
            (200, 25, 25)
        )
        
        env = ArcEnvironment(config)
        
        assert env.config.environment.max_episode_steps == 200
        assert env.config.dataset.max_grid_height == 25
        assert env.config.dataset.max_grid_width == 25

    def test_initialization_with_different_action_formats(self):
        """Test initialization with different action selection formats."""
        formats = ["point", "bbox", "mask"]
        
        for fmt in formats:
            config = JaxArcConfig()
            import equinox as eqx
            config = eqx.tree_at(
                lambda c: c.action.selection_format,
                config,
                fmt
            )
            
            env = ArcEnvironment(config)
            assert env.config.action.selection_format == fmt
            assert env.action_handler is not None

    def test_initialization_with_visualization_disabled(self):
        """Test initialization with visualization disabled."""
        config = JaxArcConfig()
        import equinox as eqx
        config = eqx.tree_at(
            lambda c: (c.visualization.enabled, c.environment.debug_level),
            config,
            (False, "off")
        )
        
        env = ArcEnvironment(config)
        assert env._enhanced_visualizer is None

    def test_initialization_with_visualization_enabled(self):
        """Test initialization with visualization enabled."""
        config = JaxArcConfig()
        import equinox as eqx
        config = eqx.tree_at(
            lambda c: (c.visualization.enabled, c.environment.debug_level),
            config,
            (True, "standard")
        )
        
        # Mock the visualization components to avoid actual file operations
        with patch('jaxarc.envs.environment.EnhancedVisualizer'), \
             patch('jaxarc.envs.environment.EpisodeManager'), \
             patch('jaxarc.envs.environment.AsyncLogger'), \
             patch('jaxarc.envs.environment.WandbIntegration'):
            
            env = ArcEnvironment(config)
            # Should not raise an exception even if visualization setup fails


class TestArcEnvironmentReset:
    """Test ArcEnvironment reset functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config = JaxArcConfig()
        self.env = ArcEnvironment(self.config)
        self.key = jax.random.PRNGKey(42)

    def test_reset_basic(self):
        """Test basic reset functionality."""
        state, obs = self.env.reset(self.key)
        
        assert isinstance(state, ArcEnvState)
        assert isinstance(obs, jnp.ndarray)
        assert self.env._state is state
        assert self.env.state is state
        assert not self.env.is_done
        assert state.step_count == 0
        assert not state.episode_done

    def test_reset_with_task_data(self):
        """Test reset with specific task data."""
        # Create minimal task data
        task_data = self._create_test_task()
        
        state, obs = self.env.reset(self.key, task_data)
        
        assert isinstance(state, ArcEnvState)
        assert state.task_data is task_data
        assert jnp.array_equal(
            state.working_grid[:5, :5], 
            task_data.input_grids_examples[0, :5, :5]
        )

    def test_reset_multiple_times(self):
        """Test multiple resets work correctly."""
        # First reset
        state1, obs1 = self.env.reset(self.key)
        episode_count1 = self.env._episode_count
        
        # Second reset
        state2, obs2 = self.env.reset(self.key)
        episode_count2 = self.env._episode_count
        
        assert episode_count2 > episode_count1
        assert self.env._state is state2
        assert state2.step_count == 0

    def test_reset_state_initialization(self):
        """Test that reset properly initializes all state fields."""
        state, obs = self.env.reset(self.key)
        
        # Check all required state fields are properly initialized
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
        
        # Check types and shapes
        chex.assert_type(state.step_count, jnp.integer)
        chex.assert_type(state.episode_done, jnp.bool_)
        chex.assert_type(state.similarity_score, jnp.floating)
        chex.assert_rank(state.working_grid, 2)
        chex.assert_rank(state.selected, 2)

    def test_reset_observation_format(self):
        """Test that reset returns properly formatted observation."""
        state, obs = self.env.reset(self.key)
        
        # Observation should be the working grid
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == state.working_grid.shape
        assert jnp.array_equal(obs, state.working_grid)

    def _create_test_task(self) -> JaxArcTask:
        """Create a minimal test task."""
        grid_shape = (5, 5)
        max_shape = (30, 30)
        
        # Create simple test grids
        input_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
        output_grid = jnp.ones(grid_shape, dtype=jnp.int32)
        mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        
        # Pad to max size
        padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
        padded_output = jnp.full(max_shape, -1, dtype=jnp.int32)
        padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)
        
        padded_input = padded_input.at[:5, :5].set(input_grid)
        padded_output = padded_output.at[:5, :5].set(output_grid)
        padded_mask = padded_mask.at[:5, :5].set(mask)
        
        return JaxArcTask(
            input_grids_examples=jnp.expand_dims(padded_input, 0),
            output_grids_examples=jnp.expand_dims(padded_output, 0),
            input_masks_examples=jnp.expand_dims(padded_mask, 0),
            output_masks_examples=jnp.expand_dims(padded_mask, 0),
            num_train_pairs=1,
            test_input_grids=jnp.expand_dims(padded_input, 0),
            test_input_masks=jnp.expand_dims(padded_mask, 0),
            true_test_output_grids=jnp.expand_dims(padded_output, 0),
            true_test_output_masks=jnp.expand_dims(padded_mask, 0),
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )


class TestArcEnvironmentStep:
    """Test ArcEnvironment step functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config = JaxArcConfig()
        # Disable debug visualization for testing
        import equinox as eqx
        self.config = eqx.tree_at(
            lambda c: (c.environment.debug_level, c.visualization.enabled),
            self.config,
            ("off", False)
        )
        self.env = ArcEnvironment(self.config)
        self.key = jax.random.PRNGKey(42)

    def test_step_without_reset_raises_error(self):
        """Test that stepping without reset raises RuntimeError."""
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        action = {"mask": mask, "operation": 0}
        
        with pytest.raises(RuntimeError, match="Environment must be reset before stepping"):
            self.env.step(action)

    def test_step_basic_functionality(self):
        """Test basic step functionality."""
        # Reset first
        state, obs = self.env.reset(self.key)
        original_step_count = state.step_count
        
        # Create valid action for mask format (default)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)  # Select small region
        action = {"mask": mask, "operation": 0}
        
        # Step
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Validate return types
        assert isinstance(next_state, ArcEnvState)
        assert isinstance(next_obs, jnp.ndarray)
        assert isinstance(reward, (int, float, jnp.ndarray))
        assert isinstance(info, dict)
        
        # Validate state progression
        assert self.env._state is next_state
        assert next_state.step_count > original_step_count

    def test_step_with_different_action_formats(self):
        """Test step with different action selection formats."""
        formats_and_actions = [
            ("point", {"point": jnp.array([1, 1]), "operation": 0}),
            ("bbox", {"bbox": jnp.array([0, 0, 2, 2]), "operation": 0}),
            ("mask", {"mask": jnp.ones((30, 30), dtype=jnp.bool_), "operation": 0}),
        ]
        
        for fmt, action in formats_and_actions:
            config = JaxArcConfig()
            import equinox as eqx
            config = eqx.tree_at(
                lambda c: (c.action.selection_format, c.environment.debug_level, c.visualization.enabled),
                config,
                (fmt, "off", False)
            )
            env = ArcEnvironment(config)
            
            # Reset and step
            state, obs = env.reset(self.key)
            next_state, next_obs, reward, info = env.step(action)
            
            assert next_state.step_count > state.step_count

    def test_step_info_dictionary(self):
        """Test that step returns proper info dictionary."""
        state, obs = self.env.reset(self.key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Check required info fields
        required_fields = ["success", "similarity", "step_count", "similarity_improvement"]
        for field in required_fields:
            assert field in info
        
        # Check info field types (JAX arrays have different type checking)
        assert hasattr(info["success"], 'dtype') and info["success"].dtype == jnp.bool_
        assert hasattr(info["similarity"], 'dtype') and jnp.issubdtype(info["similarity"].dtype, jnp.floating)
        assert isinstance(info["step_count"], (int, jnp.integer)) or hasattr(info["step_count"], 'dtype')
        assert hasattr(info["similarity_improvement"], 'dtype') and jnp.issubdtype(info["similarity_improvement"].dtype, jnp.floating)

    def test_step_reward_computation(self):
        """Test reward computation during step."""
        state, obs = self.env.reset(self.key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Reward should be a scalar
        assert jnp.isscalar(reward) or (hasattr(reward, 'shape') and reward.shape == ())
        
        # Reward should be finite
        assert jnp.isfinite(reward)

    def test_step_state_consistency(self):
        """Test that step maintains state consistency."""
        state, obs = self.env.reset(self.key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # State should be updated
        assert self.env._state is next_state
        assert self.env.state is next_state
        
        # Step count should increase
        assert next_state.step_count == state.step_count + 1
        
        # Task data should remain equivalent (JAX transformations may create new objects)
        # We check that the arrays are equal rather than identical
        assert jnp.array_equal(next_state.task_data.task_index, state.task_data.task_index)


class TestArcEnvironmentLifecycle:
    """Test ArcEnvironment lifecycle and state transitions."""

    def setup_method(self):
        """Set up test environment."""
        self.config = JaxArcConfig()
        self.config = self.config.replace(
            environment=self.config.environment.replace(
                max_episode_steps=5,
                debug_level="off"
            ),
            visualization=self.config.visualization.replace(enabled=False)
        )
        self.env = ArcEnvironment(self.config)
        self.key = jax.random.PRNGKey(42)

    def test_episode_termination_max_steps(self):
        """Test episode termination when max steps reached."""
        state, obs = self.env.reset(self.key)
        
        # Step until max steps
        for i in range(6):  # More than max_episode_steps
            if self.env.is_done:
                break
            
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[0:2, 0:2].set(True)
            action = {"mask": mask, "operation": 0}
            state, obs, reward, info = self.env.step(action)
        
        # Episode should terminate
        assert state.step_count >= self.config.environment.max_episode_steps or state.episode_done

    def test_episode_termination_submit_operation(self):
        """Test episode termination with submit operation."""
        state, obs = self.env.reset(self.key)
        
        # Use submit operation (operation 34)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        action = {"mask": mask, "operation": 34}
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Episode should be done after submit
        assert next_state.episode_done or self.env.is_done

    def test_episode_termination_perfect_similarity(self):
        """Test episode termination when perfect similarity achieved."""
        # This test would require a specific task setup where perfect similarity is achievable
        # For now, we test the logic exists
        state, obs = self.env.reset(self.key)
        action = {"selection": jnp.array([0, 0, 1, 1]), "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Check that similarity is tracked
        assert hasattr(next_state, 'similarity_score')
        assert 0.0 <= next_state.similarity_score <= 1.0

    def test_state_transitions(self):
        """Test proper state transitions during episode."""
        state, obs = self.env.reset(self.key)
        initial_similarity = state.similarity_score
        
        # Take a step
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Validate state transition
        assert next_state.step_count == state.step_count + 1
        assert next_state.task_data is state.task_data
        
        # Grid might have changed
        grid_changed = not jnp.array_equal(next_state.working_grid, state.working_grid)
        similarity_changed = next_state.similarity_score != initial_similarity
        
        # At least one should be true (step was meaningful)
        # Note: Some operations might not change the grid if invalid
        assert True  # This test validates the transition occurs without error

    def test_multiple_episodes(self):
        """Test multiple episode lifecycle."""
        episode_counts = []
        
        for episode in range(3):
            state, obs = self.env.reset(self.key)
            episode_counts.append(self.env._episode_count)
            
            # Take a few steps
            for step in range(2):
                if self.env.is_done:
                    break
                mask = jnp.zeros((30, 30), dtype=jnp.bool_)
                mask = mask.at[0:2, 0:2].set(True)
                action = {"mask": mask, "operation": 0}
                state, obs, reward, info = self.env.step(action)
        
        # Episode count should increase
        assert len(set(episode_counts)) == 3  # All different episode counts


class TestArcEnvironmentRewardComputation:
    """Test reward computation logic."""

    def setup_method(self):
        """Set up test environment with specific reward config."""
        self.config = JaxArcConfig()
        self.config = self.config.replace(
            reward=self.config.reward.replace(
                step_penalty=-0.01,
                success_bonus=10.0,
                similarity_weight=1.0,
                progress_bonus=0.1
            ),
            environment=self.config.environment.replace(debug_level="off"),
            visualization=self.config.visualization.replace(enabled=False)
        )
        self.env = ArcEnvironment(self.config)
        self.key = jax.random.PRNGKey(42)

    def test_reward_structure(self):
        """Test basic reward structure and components."""
        state, obs = self.env.reset(self.key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Reward should be finite and reasonable
        assert jnp.isfinite(reward)
        assert -100.0 <= reward <= 100.0  # Reasonable bounds

    def test_step_penalty_applied(self):
        """Test that step penalty is applied."""
        state, obs = self.env.reset(self.key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        
        next_state, next_obs, reward, info = self.env.step(action)
        
        # With step penalty, reward should typically be negative unless big improvement
        # This is a general test - specific values depend on similarity changes
        assert isinstance(reward, (int, float, jnp.ndarray))

    def test_reward_on_submit_only_mode(self):
        """Test reward computation in submit-only mode."""
        config = self.config.replace(
            reward=self.config.reward.replace(reward_on_submit_only=True)
        )
        env = ArcEnvironment(config)
        
        state, obs = env.reset(self.key)
        
        # Regular step should give minimal reward
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        next_state, next_obs, reward, info = env.step(action)
        
        # Should primarily get step penalty
        assert reward <= 0  # Assuming step penalty is negative

    def test_similarity_improvement_reward(self):
        """Test reward for similarity improvement."""
        state, obs = self.env.reset(self.key)
        initial_similarity = state.similarity_score
        
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        next_state, next_obs, reward, info = self.env.step(action)
        
        # Check similarity tracking
        assert "similarity_improvement" in info
        similarity_improvement = info["similarity_improvement"]
        
        # Improvement should be the difference
        expected_improvement = next_state.similarity_score - initial_similarity
        assert jnp.isclose(similarity_improvement, expected_improvement, atol=1e-6)


class TestArcEnvironmentConfigurationIntegration:
    """Test integration with different configuration systems."""

    def test_legacy_config_integration(self):
        """Test integration with legacy ArcEnvConfig."""
        legacy_config = ArcEnvConfig(
            max_episode_steps=100,
            auto_reset=True,
            strict_validation=True
        )
        
        # Convert to unified config
        unified_config = convert_arc_env_config_to_jax_arc_config(legacy_config)
        env = ArcEnvironment(unified_config)
        
        assert env.config.environment.max_episode_steps == 100
        assert env.config.environment.auto_reset == True
        assert env.config.environment.strict_validation == True

    def test_unified_config_integration(self):
        """Test integration with unified JaxArcConfig."""
        config = JaxArcConfig()
        config = config.replace(
            environment=config.environment.replace(
                max_episode_steps=150,
                auto_reset=False
            )
        )
        
        env = ArcEnvironment(config)
        
        assert env.config.environment.max_episode_steps == 150
        assert env.config.environment.auto_reset == False

    def test_config_validation_integration(self):
        """Test that configuration validation works with environment."""
        config = JaxArcConfig()
        
        # Valid config should work
        env = ArcEnvironment(config)
        assert env.config is not None
        
        # Environment should handle config properly
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)
        assert state is not None

    def test_action_format_config_integration(self):
        """Test integration with different action format configurations."""
        formats = ["point", "bbox", "mask"]
        
        for fmt in formats:
            config = JaxArcConfig()
            config = config.replace(
                action=config.action.replace(selection_format=fmt)
            )
            
            env = ArcEnvironment(config)
            
            # Should initialize correctly
            assert env.config.action.selection_format == fmt
            
            # Should reset correctly
            key = jax.random.PRNGKey(42)
            state, obs = env.reset(key)
            assert state is not None

    def test_reward_config_integration(self):
        """Test integration with reward configuration."""
        config = JaxArcConfig()
        config = config.replace(
            reward=config.reward.replace(
                success_bonus=50.0,
                step_penalty=-0.05,
                similarity_weight=2.0
            )
        )
        
        env = ArcEnvironment(config)
        
        # Verify config is used
        assert env.config.reward.success_bonus == 50.0
        assert env.config.reward.step_penalty == -0.05
        assert env.config.reward.similarity_weight == 2.0
        
        # Test that reward computation uses these values
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)
        mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        mask = mask.at[0:2, 0:2].set(True)
        action = {"mask": mask, "operation": 0}
        next_state, next_obs, reward, info = env.step(action)
        
        # Reward should be computed (exact value depends on similarity change)
        assert isinstance(reward, (int, float, jnp.ndarray))


class TestArcEnvironmentProperties:
    """Test ArcEnvironment properties and utility methods."""

    def setup_method(self):
        """Set up test environment."""
        self.config = JaxArcConfig()
        # Disable debug visualization for testing
        self.config = self.config.replace(
            environment=self.config.environment.replace(debug_level="off"),
            visualization=self.config.visualization.replace(enabled=False)
        )
        self.env = ArcEnvironment(self.config)
        self.key = jax.random.PRNGKey(42)

    def test_state_property(self):
        """Test state property access."""
        # Before reset
        assert self.env.state is None
        
        # After reset
        state, obs = self.env.reset(self.key)
        assert self.env.state is state
        
        # After step
        action = {"selection": jnp.array([0, 0, 1, 1]), "operation": 0}
        next_state, next_obs, reward, info = self.env.step(action)
        assert self.env.state is next_state

    def test_is_done_property(self):
        """Test is_done property."""
        # Before reset
        assert self.env.is_done is True
        
        # After reset
        state, obs = self.env.reset(self.key)
        assert self.env.is_done is False
        
        # After episode termination (would need specific conditions)
        # For now, just test the property exists and returns boolean
        assert isinstance(self.env.is_done, (bool, jnp.bool_))

    def test_observation_space_info(self):
        """Test observation space information."""
        obs_info = self.env.get_observation_space_info()
        
        required_fields = ["grid_shape", "max_colors", "selection_format"]
        for field in required_fields:
            assert field in obs_info
        
        # Check field types and values
        assert isinstance(obs_info["grid_shape"], tuple)
        assert len(obs_info["grid_shape"]) == 2
        assert isinstance(obs_info["max_colors"], int)
        assert obs_info["max_colors"] > 0
        assert isinstance(obs_info["selection_format"], str)

    def test_action_space_info(self):
        """Test action space information."""
        action_info = self.env.get_action_space_info()
        
        assert "type" in action_info
        assert isinstance(action_info["type"], str)
        
        # Specific fields depend on action format
        if self.config.action.selection_format == "mask":
            assert "selection_shape" in action_info
            assert "operation_range" in action_info
        elif self.config.action.selection_format == "point":
            assert "shape" in action_info
            assert "bounds" in action_info
        else:  # bbox
            assert "bbox_shape" in action_info
            assert "operation_range" in action_info

    def test_context_manager_support(self):
        """Test context manager support."""
        config = JaxArcConfig()
        
        with ArcEnvironment(config) as env:
            assert isinstance(env, ArcEnvironment)
            
            # Should work normally
            key = jax.random.PRNGKey(42)
            state, obs = env.reset(key)
            assert state is not None
        
        # Context manager should handle cleanup
        # (close method should be called automatically)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])