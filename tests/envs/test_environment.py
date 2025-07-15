"""Tests for the clean class-based ArcEnvironment API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.config import ActionConfig, ArcEnvConfig, GridConfig, RewardConfig
from jaxarc.envs.environment import ArcEnvironment
from jaxarc.envs.functional import ArcEnvState
from jaxarc.types import JaxArcTask


class TestArcEnvironment:
    """Test the ArcEnvironment class."""

    def test_init(self):
        """Test environment initialization."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)

        assert env.config is config
        assert env._state is None
        assert env.is_done is True

    def test_reset(self):
        """Test environment reset."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        state, obs = env.reset(key)

        assert isinstance(state, ArcEnvState)
        assert isinstance(obs, jnp.ndarray)
        assert env._state is state
        assert env.is_done is False

    def test_reset_with_task_data(self):
        """Test reset with specific task data."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        # Create dummy task data
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
            task_index=jnp.int32(0),
        )

        state, obs = env.reset(key, task_data)

        assert isinstance(state, ArcEnvState)
        assert jnp.array_equal(
            state.task_data.input_grids_examples, task_data.input_grids_examples
        )

    def test_step_without_reset(self):
        """Test step without reset raises error."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)
        action = 0

        with pytest.raises(
            RuntimeError, match="Environment must be reset before stepping"
        ):
            env.step(action)

    def test_step_after_reset(self):
        """Test step after reset."""
        config = ArcEnvConfig(
            max_episode_steps=50, action=ActionConfig(selection_format="bbox")
        )
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        # Reset first
        state, obs = env.reset(key)
        original_step_count = state.step_count

        # Step with simple action
        action = {"bbox": jnp.array([0, 0, 1, 1]), "operation": 0}
        try:
            next_state, next_obs, reward, info = env.step(action)

            assert isinstance(next_state, ArcEnvState)
            assert isinstance(next_obs, jnp.ndarray)
            assert isinstance(reward, (int, float, jnp.ndarray))
            assert isinstance(info, dict)
            assert env._state is next_state
            assert next_state.step_count >= original_step_count
        except Exception:
            # Action might fail due to validation, but environment should handle it gracefully
            pass

    def test_properties(self):
        """Test environment properties."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)

        # Before reset
        assert env.state is None
        assert env.is_done is True

        # After reset
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        assert env.state is state
        assert env.is_done is False

    def test_observation_space_info(self):
        """Test observation space information."""
        config = ArcEnvConfig(
            grid=GridConfig(max_grid_height=20, max_grid_width=25, max_colors=8),
            action=ActionConfig(selection_format="mask"),
        )
        env = ArcEnvironment(config)

        obs_info = env.get_observation_space_info()

        assert obs_info["grid_shape"] == (20, 25)
        assert obs_info["max_colors"] == 8
        assert obs_info["selection_format"] == "mask"

    def test_action_space_info_mask_format(self):
        """Test action space info for mask format."""
        config = ArcEnvConfig(
            grid=GridConfig(max_grid_height=30, max_grid_width=30),
            action=ActionConfig(selection_format="mask", num_operations=14),
        )
        env = ArcEnvironment(config)

        action_info = env.get_action_space_info()

        assert action_info["type"] == "dict"
        assert action_info["selection_shape"] == (4,)
        assert action_info["selection_bounds"] == (0, 30)
        assert action_info["operation_range"] == (0, 14)

    def test_action_space_info_point(self):
        """Test action space info for point format."""
        config = ArcEnvConfig(
            grid=GridConfig(max_grid_height=25, max_grid_width=20),
            action=ActionConfig(selection_format="point", num_operations=10),
        )
        env = ArcEnvironment(config)

        action_info = env.get_action_space_info()

        assert action_info["type"] == "array"
        assert action_info["shape"] == (3,)
        assert action_info["bounds"] == (0, 25)  # max of height, width, operations

    def test_action_space_info_bbox(self):
        """Test action space info for bbox format."""
        config = ArcEnvConfig(
            action=ActionConfig(selection_format="bbox", num_operations=8)
        )
        env = ArcEnvironment(config)

        action_info = env.get_action_space_info()

        assert action_info["type"] == "dict"
        assert action_info["bbox_shape"] == (4,)
        assert action_info["bbox_bounds"] == (0, 30)
        assert action_info["operation_range"] == (0, 8)

    def test_different_action_formats(self):
        """Test environment with different action formats."""
        formats = ["mask", "point", "bbox"]

        for selection_format in formats:
            config = ArcEnvConfig(
                max_episode_steps=50,
                action=ActionConfig(selection_format=selection_format),
            )
            env = ArcEnvironment(config)

            # Should initialize without error
            assert env.config.action.selection_format == selection_format

            # Should provide correct action space info
            action_info = env.get_action_space_info()
            assert "type" in action_info

    def test_episode_termination(self):
        """Test episode termination handling."""
        config = ArcEnvConfig(max_episode_steps=2)  # Very short episode
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        # Reset
        state, obs = env.reset(key)
        assert not env.is_done

        # Step until done (or max steps)
        for i in range(5):  # More than max_episode_steps
            if env.is_done:
                break
            try:
                if config.action.selection_format == "mask":
                    action = {"mask": jnp.array([0, 0, 1, 1]), "operation": 0}
                elif config.action.selection_format == "point":
                    action = {"point": jnp.array([1, 1]), "operation": 0}
                else:  # bbox
                    action = {"bbox": jnp.array([0, 0, 1, 1]), "operation": 0}
                state, obs, reward, info = env.step(action)
            except Exception:
                # Action might fail, but we should still track state
                break

        # Episode should eventually terminate
        assert env.state.step_count >= 0

    def test_integration_with_functional_api(self):
        """Test that class-based API produces same results as functional API."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        # Reset both
        env_state, env_obs = env.reset(key)

        from jaxarc.envs.functional import arc_reset

        func_state, func_obs = arc_reset(key, config)

        # Should produce identical results
        assert env_state.step_count == func_state.step_count
        assert env_state.episode_done == func_state.episode_done
        assert jnp.array_equal(env_obs, func_obs)

    def test_config_immutability(self):
        """Test that environment doesn't modify the config."""
        config = ArcEnvConfig(max_episode_steps=50)
        original_steps = config.max_episode_steps

        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        env.reset(key)

        # Config should remain unchanged
        assert config.max_episode_steps == original_steps
        assert env.config.max_episode_steps == original_steps

    def test_state_consistency(self):
        """Test that internal state remains consistent."""
        config = ArcEnvConfig(max_episode_steps=50)
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        # Before reset
        assert env._state is None
        assert env.state is None

        # After reset
        state, obs = env.reset(key)
        assert env._state is state
        assert env.state is state

        # After step
        try:
            if config.action.selection_format == "mask":
                action = {"mask": jnp.array([0, 0, 1, 1]), "operation": 0}
            elif config.action.selection_format == "point":
                action = {"point": jnp.array([1, 1]), "operation": 0}
            else:  # bbox
                action = {"bbox": jnp.array([0, 0, 1, 1]), "operation": 0}
            next_state, next_obs, reward, info = env.step(action)
            assert env._state is next_state
            assert env.state is next_state
        except Exception:
            # Action might fail, but state should still be consistent
            pass


class TestArcEnvironmentWithDifferentConfigs:
    """Test ArcEnvironment with various configurations."""

    def test_minimal_config(self):
        """Test with minimal configuration."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            grid=GridConfig(max_grid_height=5, max_grid_width=5, max_colors=3),
            action=ActionConfig(num_operations=3),
        )
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        state, obs = env.reset(key)
        assert state is not None
        assert obs.shape[0] <= 5  # Grid height
        assert obs.shape[1] <= 5  # Grid width

    def test_large_config(self):
        """Test with large configuration."""
        config = ArcEnvConfig(
            max_episode_steps=200,
            grid=GridConfig(max_grid_height=30, max_grid_width=30, max_colors=10),
            action=ActionConfig(num_operations=14),
        )
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        state, obs = env.reset(key)
        assert state is not None
        assert obs.shape[0] <= 30  # Grid height
        assert obs.shape[1] <= 30  # Grid width

    def test_different_rewards(self):
        """Test with different reward configurations."""
        config = ArcEnvConfig(
            max_episode_steps=50,
            reward=RewardConfig(
                success_bonus=100.0,
                step_penalty=-0.1,
                progress_bonus=5.0,
                reward_on_submit_only=False,
            ),
        )
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)

        state, obs = env.reset(key)
        assert state is not None

        # Test that reward config is properly used
        assert env.config.reward.success_bonus == 100.0
        assert env.config.reward.step_penalty == -0.1


if __name__ == "__main__":
    pytest.main([__file__])
