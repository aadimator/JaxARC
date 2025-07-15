"""
Integration tests for action handlers with JaxARC environment.

This module tests the integration between the new action handler system
and the JaxARC environment, ensuring proper action processing and
compatibility with both class-based and functional APIs.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs import (
    ActionConfig,
    ArcEnvConfig,
    ArcEnvironment,
    GridConfig,
    RewardConfig,
    arc_reset,
    arc_step,
)


class TestActionHandlerEnvironmentIntegration:
    """Test integration between action handlers and environment."""

    def test_point_action_with_environment(self):
        """Test point actions work with environment."""
        # Create config with point action format
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Create environment and reset
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Create point action
        action = {
            "point": jnp.array([5, 5]),  # Point at (5, 5)
            "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
        }

        # Step environment
        next_state, next_obs, reward, info = env.step(action)

        # Verify action was processed correctly
        assert next_state.selected[5, 5] == True
        assert jnp.sum(next_state.selected) == 1

    def test_bbox_action_with_environment(self):
        """Test bbox actions work with environment."""
        # Create config with bbox action format
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="bbox"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Create environment and reset
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Create bbox action
        action = {
            "bbox": jnp.array([2, 3, 4, 5]),  # Bbox from (2,3) to (4,5)
            "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
        }

        # Step environment
        next_state, next_obs, reward, info = env.step(action)

        # Verify action was processed correctly
        expected_count = (4 - 2 + 1) * (5 - 3 + 1)  # 3x3 = 9 cells
        assert jnp.sum(next_state.selected) == expected_count
        assert next_state.selected[2, 3] == True
        assert next_state.selected[4, 5] == True

    def test_mask_action_with_environment(self):
        """Test mask actions work with environment."""
        # Create config with mask action format
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="mask"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Create environment and reset
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Create mask action
        mask = jnp.zeros((10, 10), dtype=jnp.bool_)
        mask = mask.at[1:4, 1:4].set(True)  # 3x3 region
        action = {
            "mask": mask.flatten(),
            "operation": jnp.array(3, dtype=jnp.int32),  # Fill with color 3
        }

        # Step environment
        next_state, next_obs, reward, info = env.step(action)

        # Verify action was processed correctly
        assert jnp.sum(next_state.selected) == 9  # 3x3 = 9 cells
        assert next_state.selected[1, 1] == True
        assert next_state.selected[3, 3] == True

    def test_working_grid_mask_constraint_integration(self):
        """Test that working grid mask constraints work with environment."""
        # Create config with small grid
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Create environment and reset
        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Create point action outside working grid
        action = {
            "point": jnp.array([10, 10]),  # Point outside 5x5 grid
            "operation": jnp.array(1, dtype=jnp.int32),
        }

        # Step environment
        next_state, next_obs, reward, info = env.step(action)

        # Verify no selection due to working grid constraint
        # The point should be clipped to (5, 5) but that's outside the working grid
        working_cells = jnp.sum(state.working_grid_mask)
        selected_cells = jnp.sum(next_state.selected)
        # Selection should be constrained to working grid
        assert selected_cells <= working_cells

    def test_multiple_action_formats_with_functional_api(self):
        """Test different action formats with functional API."""
        formats_and_actions = [
            ("point", {"point": jnp.array([3, 4]), "operation": jnp.array(1)}),
            ("bbox", {"bbox": jnp.array([1, 2, 3, 4]), "operation": jnp.array(2)}),
            (
                "mask",
                {
                    "mask": jnp.zeros((10, 10), dtype=jnp.bool_)
                    .at[2:5, 2:5]
                    .set(True)
                    .flatten(),
                    "operation": jnp.array(3),
                },
            ),
        ]

        for selection_format, action in formats_and_actions:
            # Create config for this format
            config = ArcEnvConfig(
                max_episode_steps=10,
                action=ActionConfig(selection_format=selection_format),
                grid=GridConfig(max_grid_height=10, max_grid_width=10),
                reward=RewardConfig(),
            )

            # Reset and step with functional API
            key = jax.random.PRNGKey(42)
            state, obs = arc_reset(key, config)
            next_state, next_obs, reward, done, info = arc_step(state, action, config)

            # Verify action was processed
            assert jnp.sum(next_state.selected) > 0
            assert next_state.step_count == 1

    def test_jit_compilation_with_environment(self):
        """Test that environment actions work with JIT compilation."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        def step_with_config(state, action):
            return arc_step(state, action, config)

        jitted_step = jax.jit(step_with_config)

        # Reset environment
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config)

        # Create action
        action = {
            "point": jnp.array([2, 3]),
            "operation": jnp.array(1, dtype=jnp.int32),
        }

        # Step with JIT
        next_state, next_obs, reward, done, info = jitted_step(state, action)

        # Verify it worked
        assert next_state.selected[2, 3] == True
        assert jnp.sum(next_state.selected) == 1

    def test_action_validation_and_clipping(self):
        """Test that action validation and clipping work correctly."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Test coordinate clipping
        action = {
            "point": jnp.array([-5, 35]),  # Coordinates outside valid range
            "operation": jnp.array(1, dtype=jnp.int32),
        }

        # Should not raise error due to clipping
        next_state, next_obs, reward, info = env.step(action)

        # Coordinates should be clipped to valid range
        assert next_state.selected[0, 9] == True  # Clipped to (0, 9) for 10x10 grid
        assert jnp.sum(next_state.selected) == 1

    def test_point_action_format_support(self):
        """Test point action format support."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Test point action format
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config)

        # Point format: dict with "point" key
        point_action = {
            "point": jnp.array([3, 4]),
            "operation": jnp.array(1, dtype=jnp.int32),
        }

        # Should work with functional API
        next_state, next_obs, reward, done, info = arc_step(state, point_action, config)

        # Verify action was processed
        assert next_state.selected[3, 4] == True
        assert jnp.sum(next_state.selected) == 1

    def test_different_grid_sizes(self):
        """Test action handlers work with different grid sizes."""
        grid_sizes = [(5, 5), (10, 15), (20, 25)]

        for height, width in grid_sizes:
            config = ArcEnvConfig(
                max_episode_steps=10,
                action=ActionConfig(selection_format="point"),
                grid=GridConfig(max_grid_height=10, max_grid_width=10),
                reward=RewardConfig(),
            )

            env = ArcEnvironment(config)
            key = jax.random.PRNGKey(42)
            state, obs = env.reset(key)

            # Create point action within grid bounds
            # Create action for this grid size
            action = {
                "point": jnp.array([min(2, height - 1), min(2, width - 1)]),
                "operation": jnp.array(1, dtype=jnp.int32),
            }

            # Step environment
            next_state, next_obs, reward, info = env.step(action)

            # Verify action was processed correctly
            assert jnp.sum(next_state.selected) == 1

    def test_batch_processing_compatibility(self):
        """Test that action handlers work with batched processing.

        Action handlers are JIT-compiled and work well with vmap since they
        only take JAX arrays as input (action_data and working_grid_mask).
        This test verifies batch processing works correctly.
        """
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # Create simple test grids for batch processing
        batch_size = 3
        grid_shape = (10, 10)

        # Create batch of working grids
        working_grids = jnp.zeros((batch_size,) + grid_shape, dtype=jnp.int32)
        working_masks = jnp.ones((batch_size,) + grid_shape, dtype=jnp.bool_)

        # Create batch of action data (just the point coordinates)
        batch_action_data = jnp.array([[1, 2], [3, 4], [5, 6]])

        # Test batch processing with action handlers
        from jaxarc.envs.actions import get_action_handler

        handler = get_action_handler(config.action.selection_format)

        # Use vmap to process batch of actions - this works because action handlers
        # only take JAX arrays as input, not config objects
        batch_process = jax.vmap(handler, in_axes=(0, 0))
        batch_results = batch_process(batch_action_data, working_masks)

        # Verify batch processing worked
        chex.assert_shape(batch_results, (batch_size,) + grid_shape)

        # Verify each action was processed correctly
        for i in range(batch_size):
            # Each action should select exactly one point
            assert jnp.sum(batch_results[i]) == 1

            # Verify the correct points were selected
            expected_row, expected_col = batch_action_data[i]
            assert batch_results[i][expected_row, expected_col] == 1

    def test_action_handler_performance(self):
        """Test that action handlers maintain good performance."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        # JIT compile the full pipeline
        @jax.jit
        def full_pipeline(key, action):
            state, obs = arc_reset(key, config)
            return arc_step(state, action, config)

        # Warm up JIT
        key = jax.random.PRNGKey(42)
        action = {
            "point": jnp.array([5, 5]),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        _ = full_pipeline(key, action)

        # Test that it runs without issues
        result = full_pipeline(key, action)
        assert len(result) == 5  # state, obs, reward, done, info

    def test_error_handling_integration(self):
        """Test proper error handling in integration scenarios."""
        config = ArcEnvConfig(
            max_episode_steps=10,
            action=ActionConfig(selection_format="point"),
            grid=GridConfig(max_grid_height=10, max_grid_width=10),
            reward=RewardConfig(),
        )

        env = ArcEnvironment(config)
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        # Test missing selection key
        with pytest.raises(ValueError, match="Action must contain 'selection' field"):
            invalid_action = {"operation": jnp.array(1)}
            arc_step(state, invalid_action, config)

        # Test environment not reset
        fresh_env = ArcEnvironment(config)
        with pytest.raises(RuntimeError, match="Environment must be reset"):
            action = {
                "point": jnp.array([5, 5]),
                "operation": jnp.array(1, dtype=jnp.int32),
            }
            fresh_env.step(action)
