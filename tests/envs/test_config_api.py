"""
Tests for config-based ARC environment API.

This module tests the new config-based architecture including typed configs,
Hydra integration, functional API, and factory functions.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc.envs import (
    ActionConfig,
    ArcEnvConfig,
    GridConfig,
    RewardConfig,
    arc_reset,
    arc_step,
    create_bbox_config,
    create_config_from_hydra,
    create_evaluation_config,
    create_point_config,
    create_raw_config,
    create_restricted_config,
    create_standard_config,
    create_training_config,
    get_preset_config,
)
from jaxarc.state import ArcEnvState


class TestConfigClasses:
    """Test configuration dataclasses."""

    def test_reward_config_creation(self):
        """Test RewardConfig creation and validation."""
        config = RewardConfig(
            reward_on_submit_only=True,
            step_penalty=-0.01,
            success_bonus=10.0,
        )

        assert config.reward_on_submit_only is True
        assert config.step_penalty == -0.01
        assert config.success_bonus == 10.0

    def test_reward_config_from_hydra(self):
        """Test RewardConfig creation from Hydra config."""
        hydra_config = OmegaConf.create(
            {
                "reward_on_submit_only": False,
                "step_penalty": -0.02,
                "success_bonus": 15.0,
            }
        )

        config = RewardConfig.from_hydra(hydra_config)
        assert config.reward_on_submit_only is False
        assert config.step_penalty == -0.02
        assert config.success_bonus == 15.0

    def test_grid_config_validation(self):
        """Test GridConfig validation."""
        # Valid config
        config = GridConfig(
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=3,
            min_grid_width=3,
        )
        assert config.max_grid_size == (30, 30)

        # Invalid config - max < min
        with pytest.raises(ValueError, match="max_grid_height.*min_grid_height"):
            GridConfig(max_grid_height=2, min_grid_height=3)

    def test_action_config_validation(self):
        """Test ActionConfig validation."""
        # Valid config
        config = ActionConfig(selection_format="mask")
        assert config.selection_format == "mask"

        # Invalid selection format
        with pytest.raises(ValueError, match="Invalid selection_format"):
            ActionConfig(selection_format="invalid_format")

        # Invalid selection threshold
        with pytest.raises(ValueError, match="selection_threshold must be in"):
            ActionConfig(selection_threshold=1.5)

    def test_arc_env_config_creation(self):
        """Test complete ArcEnvConfig creation."""
        config = ArcEnvConfig(
            max_episode_steps=100,
            reward=RewardConfig(success_bonus=5.0),
            grid=GridConfig(max_grid_height=20),
            action=ActionConfig(num_operations=30),
        )

        assert config.max_episode_steps == 100
        assert config.reward.success_bonus == 5.0
        assert config.grid.max_grid_height == 20
        assert config.action.num_operations == 30

    def test_arc_env_config_from_hydra(self):
        """Test ArcEnvConfig creation from Hydra config."""
        hydra_config = OmegaConf.create(
            {
                "max_episode_steps": 150,
                "log_operations": True,
                "reward": {
                    "success_bonus": 20.0,
                    "step_penalty": -0.05,
                },
                "grid": {
                    "max_grid_height": 25,
                    "max_colors": 8,
                },
                "action": {
                    "selection_format": "point",
                    "num_operations": 20,
                },
            }
        )

        config = ArcEnvConfig.from_hydra(hydra_config)

        assert config.max_episode_steps == 150
        assert config.log_operations is True
        assert config.reward.success_bonus == 20.0
        assert config.reward.step_penalty == -0.05
        assert config.grid.max_grid_height == 25
        assert config.grid.max_colors == 8
        assert config.action.selection_format == "point"
        assert config.action.num_operations == 20

    def test_config_serialization(self):
        """Test config to_dict and from_dict."""
        original_config = ArcEnvConfig(
            max_episode_steps=80,
            reward=RewardConfig(success_bonus=7.5),
        )

        config_dict = original_config.to_dict()
        assert config_dict["max_episode_steps"] == 80
        assert config_dict["reward"]["success_bonus"] == 7.5

        # Test round-trip conversion
        from jaxarc.envs.config import config_from_dict

        restored_config = config_from_dict(config_dict)
        assert restored_config.max_episode_steps == 80
        assert restored_config.reward.success_bonus == 7.5


class TestFactoryFunctions:
    """Test factory functions for creating configs."""

    def test_create_raw_config(self):
        """Test raw config creation."""
        config = create_raw_config(
            max_episode_steps=50,
            step_penalty=-0.02,
            success_bonus=5.0,
        )

        assert config.max_episode_steps == 50
        assert config.reward.step_penalty == -0.02
        assert config.reward.success_bonus == 5.0
        assert config.reward.reward_on_submit_only is False
        assert config.strict_validation is False

    def test_create_standard_config(self):
        """Test standard config creation."""
        config = create_standard_config(
            max_episode_steps=120,
            reward_on_submit_only=False,
        )

        assert config.max_episode_steps == 120
        assert config.reward.reward_on_submit_only is False
        assert config.strict_validation is True
        assert config.action.validate_actions is True

    def test_create_point_config(self):
        """Test point-based config creation."""
        config = create_point_config(max_episode_steps=80)

        assert config.max_episode_steps == 80
        assert config.action.selection_format == "point"
        assert config.action.allow_partial_selection is False

    def test_create_bbox_config(self):
        """Test bbox-based config creation."""
        config = create_bbox_config(max_episode_steps=90)

        assert config.max_episode_steps == 90
        assert config.action.selection_format == "bbox"
        assert config.action.allow_partial_selection is False

    def test_create_restricted_config(self):
        """Test restricted config creation."""
        allowed_ops = [0, 1, 2, 34]
        config = create_restricted_config(
            max_episode_steps=60,
            allowed_operations=allowed_ops,
        )

        assert config.max_episode_steps == 60
        assert config.action.selection_threshold == 0.7
        assert config.reward.step_penalty == -0.02
        assert config.reward.success_bonus == 15.0

    def test_create_training_config(self):
        """Test training config creation."""
        basic_config = create_training_config("basic")
        assert basic_config.max_episode_steps == 50

        standard_config = create_training_config("standard")
        assert standard_config.max_episode_steps == 100

        advanced_config = create_training_config("advanced")
        assert advanced_config.max_episode_steps == 150

        expert_config = create_training_config("expert")
        assert expert_config.max_episode_steps == 200

        with pytest.raises(ValueError, match="Unknown curriculum level"):
            create_training_config("invalid")

    def test_create_evaluation_config(self):
        """Test evaluation config creation."""
        config = create_evaluation_config(strict_mode=True)

        assert config.reward.step_penalty == 0.0
        assert config.reward.success_bonus == 1.0
        assert config.strict_validation is True
        assert config.allow_invalid_actions is False

    def test_get_preset_config(self):
        """Test preset config retrieval."""
        config = get_preset_config("standard", max_episode_steps=99)
        assert config.max_episode_steps == 99

        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent")

    def test_create_config_from_hydra(self):
        """Test config creation from Hydra with base config."""
        base_config = create_standard_config(max_episode_steps=100)

        hydra_override = OmegaConf.create(
            {
                "max_episode_steps": 150,
                "reward": {"success_bonus": 20.0},
            }
        )

        merged_config = create_config_from_hydra(hydra_override, base_config)

        assert merged_config.max_episode_steps == 150
        assert merged_config.reward.success_bonus == 20.0
        # Base config values should be preserved
        assert merged_config.reward.step_penalty == base_config.reward.step_penalty


class TestFunctionalAPI:
    """Test functional API for ARC environment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.config = create_standard_config(max_episode_steps=10)

    def test_arc_reset_with_typed_config(self):
        """Test arc_reset with typed config."""
        state, obs = arc_reset(self.key, self.config)

        # Check state structure
        assert isinstance(state, ArcEnvState)
        chex.assert_rank(state.working_grid, 2)
        chex.assert_rank(state.target_grid, 2)
        chex.assert_rank(obs, 2)

        # Check initial values
        assert state.step_count == 0
        assert state.episode_done is False
        assert state.current_example_idx == 0

        # Check JAX compatibility
        chex.assert_type(state.working_grid, jnp.integer)
        chex.assert_type(state.similarity_score, jnp.floating)

    def test_arc_reset_with_hydra_config(self):
        """Test arc_reset with Hydra DictConfig."""
        hydra_config = OmegaConf.create(
            {
                "max_episode_steps": 15,
                "reward": {"success_bonus": 12.0},
                "grid": {"max_grid_height": 20},
                "action": {"num_operations": 30},
            }
        )

        state, obs = arc_reset(self.key, hydra_config)

        assert isinstance(state, ArcEnvState)
        chex.assert_rank(obs, 2)
        assert state.step_count == 0

    def test_arc_step_with_mask_format(self):
        """Test arc_step with mask format action."""
        state, obs = arc_reset(self.key, self.config)

        # Create a simple action
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),  # Fill with color 0
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, self.config)

        # Check return types
        assert isinstance(new_state, ArcEnvState)
        chex.assert_rank(new_obs, 2)
        chex.assert_rank(reward, 0)  # Scalar
        chex.assert_rank(done, 0)  # Scalar
        assert isinstance(info, dict)

        # Check state progression
        assert new_state.step_count == 1
        assert "success" in info
        assert "similarity" in info
        assert "step_count" in info

    def test_arc_step_with_point_action(self):
        """Test arc_step with point-based action."""
        point_config = create_point_config(max_episode_steps=10)
        state, obs = arc_reset(self.key, point_config)

        # Create a point action
        action = {
            "point": (2, 3),
            "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, point_config)

        # Check that step executed successfully
        assert new_state.step_count == 1
        assert isinstance(info, dict)

    def test_arc_step_with_bbox_action(self):
        """Test arc_step with bbox-based action."""
        bbox_config = create_bbox_config(max_episode_steps=10)
        state, obs = arc_reset(self.key, bbox_config)

        # Create a bbox action
        action = {
            "bbox": (1, 1, 3, 3),  # (row1, col1, row2, col2)
            "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, bbox_config)

        # Check that step executed successfully
        assert new_state.step_count == 1
        assert isinstance(info, dict)

    def test_arc_step_invalid_action_validation(self):
        """Test action validation and error handling."""
        state, obs = arc_reset(self.key, self.config)

        # Missing selection field
        with pytest.raises(ValueError, match="must contain 'selection'"):
            arc_step(state, {"operation": 0}, self.config)

        # Missing operation field
        with pytest.raises(ValueError, match="must contain 'operation'"):
            arc_step(
                state,
                {"selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_)},
                self.config,
            )

        # Invalid selection shape
        with pytest.raises(ValueError, match="doesn't match grid shape"):
            arc_step(
                state,
                {"selection": jnp.ones((5, 5), dtype=jnp.bool_), "operation": 0},
                self.config,
            )

    def test_arc_step_operation_clipping(self):
        """Test operation clipping for invalid operations."""
        # Create config with clipping enabled
        config = create_standard_config(max_episode_steps=10)
        # Create new config with clipping enabled (frozen dataclass)
        action_config = ActionConfig(
            selection_format=config.action.selection_format,
            selection_threshold=config.action.selection_threshold,
            allow_partial_selection=config.action.allow_partial_selection,
            num_operations=config.action.num_operations,
            validate_actions=config.action.validate_actions,
            clip_invalid_actions=True,
        )
        config = ArcEnvConfig(
            max_episode_steps=config.max_episode_steps,
            auto_reset=config.auto_reset,
            log_operations=config.log_operations,
            log_grid_changes=config.log_grid_changes,
            log_rewards=config.log_rewards,
            strict_validation=config.strict_validation,
            allow_invalid_actions=config.allow_invalid_actions,
            reward=config.reward,
            grid=config.grid,
            action=action_config,
        )

        state, obs = arc_reset(self.key, config)

        # Create action with invalid operation
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(999, dtype=jnp.int32),  # Invalid operation
        }

        # Should not raise error due to clipping
        new_state, new_obs, reward, done, info = arc_step(state, action, config)
        assert new_state.step_count == 1

    def test_reward_calculation(self):
        """Test reward calculation logic."""
        # Test with reward on submit only
        base_config = create_standard_config(max_episode_steps=10)
        # Create new config with reward on submit only (frozen dataclass)
        reward_config = RewardConfig(
            reward_on_submit_only=True,
            step_penalty=-0.1,
            success_bonus=base_config.reward.success_bonus,
            similarity_weight=base_config.reward.similarity_weight,
            progress_bonus=base_config.reward.progress_bonus,
            invalid_action_penalty=base_config.reward.invalid_action_penalty,
        )
        submit_config = ArcEnvConfig(
            max_episode_steps=base_config.max_episode_steps,
            auto_reset=base_config.auto_reset,
            log_operations=base_config.log_operations,
            log_grid_changes=base_config.log_grid_changes,
            log_rewards=base_config.log_rewards,
            strict_validation=base_config.strict_validation,
            allow_invalid_actions=base_config.allow_invalid_actions,
            reward=reward_config,
            grid=base_config.grid,
            action=base_config.action,
        )

        state, obs = arc_reset(self.key, submit_config)

        # Non-submit action should only get step penalty
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),  # Fill operation
        }

        new_state, new_obs, reward, done, info = arc_step(state, action, submit_config)

        # Should get step penalty but not other rewards
        assert float(reward) == pytest.approx(-0.1, abs=1e-6)
        assert not done

    def test_episode_termination(self):
        """Test episode termination conditions."""
        # Test max steps termination
        short_config = create_standard_config(max_episode_steps=2)
        state, obs = arc_reset(self.key, short_config)

        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }

        # Step 1
        state, obs, reward, done, info = arc_step(state, action, short_config)
        assert not done
        assert state.step_count == 1

        # Step 2 - should terminate due to max steps
        state, obs, reward, done, info = arc_step(state, action, short_config)
        assert done
        assert state.step_count == 2

    def test_jax_compatibility(self):
        """Test JAX transformations on functional API."""
        # Test with jit compilation - mark config as static since it contains non-JAX types
        jitted_reset_static = jax.jit(arc_reset, static_argnums=(1,))
        state, obs = jitted_reset_static(self.key, self.config)
        assert isinstance(state, ArcEnvState)

        # Test that step function also works with JIT
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }

        jitted_step_static = jax.jit(arc_step, static_argnums=(2,))
        new_state, new_obs, reward, done, info = jitted_step_static(
            state, action, self.config
        )
        assert isinstance(new_state, ArcEnvState)
        assert new_state.step_count == 1


class TestHydraIntegration:
    """Test Hydra integration features."""

    def test_hydra_config_override(self):
        """Test Hydra config overrides."""
        base_hydra = OmegaConf.create(
            {
                "max_episode_steps": 100,
                "reward": {"success_bonus": 10.0},
            }
        )

        override_hydra = OmegaConf.create(
            {
                "max_episode_steps": 200,
                "reward": {"step_penalty": -0.02},
            }
        )

        # Test merge functionality
        merged = OmegaConf.merge(base_hydra, override_hydra)
        config = ArcEnvConfig.from_hydra(merged)

        assert config.max_episode_steps == 200
        assert config.reward.success_bonus == 10.0  # From base
        assert config.reward.step_penalty == -0.02  # From override

    def test_nested_hydra_config(self):
        """Test deeply nested Hydra configuration."""
        complex_hydra = OmegaConf.create(
            {
                "max_episode_steps": 150,
                "log_operations": True,
                "reward": {
                    "reward_on_submit_only": False,
                    "step_penalty": -0.01,
                    "success_bonus": 15.0,
                    "progress_bonus": 0.5,
                },
                "grid": {
                    "max_grid_height": 25,
                    "max_grid_width": 25,
                    "max_colors": 8,
                },
                "action": {
                    "selection_format": "bbox",
                    "selection_threshold": 0.7,
                    "num_operations": 30,
                },
            }
        )

        config = ArcEnvConfig.from_hydra(complex_hydra)

        assert config.max_episode_steps == 150
        assert config.log_operations is True
        assert config.reward.reward_on_submit_only is False
        assert config.reward.progress_bonus == 0.5
        assert config.grid.max_grid_height == 25
        assert config.action.selection_format == "bbox"
        assert config.action.selection_threshold == 0.7


class TestConfigValidation:
    """Test configuration validation and error handling."""

    def test_config_consistency_validation(self):
        """Test configuration consistency checks."""
        from jaxarc.envs.config import validate_config

        # Test valid config
        valid_config = create_standard_config()
        validate_config(valid_config)  # Should not raise

        # Test invalid config - background color >= max colors
        base_config = create_standard_config()
        # Create new config with invalid background color (frozen dataclass)
        invalid_grid_config = GridConfig(
            max_grid_height=base_config.grid.max_grid_height,
            max_grid_width=base_config.grid.max_grid_width,
            min_grid_height=base_config.grid.min_grid_height,
            min_grid_width=base_config.grid.min_grid_width,
            max_colors=10,
            background_color=15,
        )
        invalid_config = ArcEnvConfig(
            max_episode_steps=base_config.max_episode_steps,
            auto_reset=base_config.auto_reset,
            log_operations=base_config.log_operations,
            log_grid_changes=base_config.log_grid_changes,
            log_rewards=base_config.log_rewards,
            strict_validation=base_config.strict_validation,
            allow_invalid_actions=base_config.allow_invalid_actions,
            reward=base_config.reward,
            grid=invalid_grid_config,
            action=base_config.action,
        )

        with pytest.raises(ValueError, match="background_color.*must be.*max_colors"):
            validate_config(invalid_config)

    def test_config_summary(self):
        """Test config summary generation."""
        from jaxarc.envs.config import get_config_summary

        config = create_standard_config(max_episode_steps=123)
        summary = get_config_summary(config)

        assert "max_steps=123" in summary
        assert "ARC Environment Configuration" in summary
        assert isinstance(summary, str)


class TestParserIntegration:
    """Test parser integration with Hydra configuration."""

    def test_create_config_with_parser(self):
        """Test creating config with parser instance."""
        from jaxarc.envs.factory import create_config_with_parser

        # Create a mock parser
        class MockParser:
            def get_random_task(self, key):
                from jaxarc.envs.config import ArcEnvConfig
                from jaxarc.envs.functional import _create_demo_task

                config = ArcEnvConfig()
                return _create_demo_task(config)

        base_config = create_standard_config()
        parser = MockParser()

        config_with_parser = create_config_with_parser(base_config, parser)

        assert config_with_parser.parser is parser
        assert config_with_parser.max_episode_steps == base_config.max_episode_steps
        assert config_with_parser.reward == base_config.reward

    def test_create_config_from_hydra_with_parser(self):
        """Test creating config from Hydra with parser."""
        from jaxarc.envs.factory import create_config_from_hydra

        # Create a mock parser
        class MockParser:
            def get_random_task(self, key):
                from jaxarc.envs.config import ArcEnvConfig
                from jaxarc.envs.functional import _create_demo_task

                config = ArcEnvConfig()
                return _create_demo_task(config)

        hydra_config = OmegaConf.create(
            {
                "max_episode_steps": 50,
                "reward": {"success_bonus": 15.0},
                "grid": {"max_grid_height": 25},
            }
        )

        parser = MockParser()
        config = create_config_from_hydra(hydra_config, parser=parser)

        assert config.parser is parser
        assert config.max_episode_steps == 50

    def test_functional_api_with_parser(self):
        """Test functional API with parser-enabled config."""
        from jaxarc.envs.factory import create_config_with_parser

        # Create a mock parser that returns a specific task
        class MockParser:
            def get_random_task(self, key):
                from jaxarc.envs.config import ArcEnvConfig
                from jaxarc.envs.functional import _create_demo_task

                config = ArcEnvConfig()
                return _create_demo_task(config)

        base_config = create_standard_config()
        parser = MockParser()
        config = create_config_with_parser(base_config, parser)

        # Test that reset uses the parser
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config)

        assert isinstance(state, ArcEnvState)
        assert state.step_count == 0
        chex.assert_rank(obs, 2)

    def test_backward_compatibility_task_sampler(self):
        """Test backward compatibility with task sampler."""
        from jaxarc.envs.factory import create_config_with_task_sampler

        def mock_task_sampler(key, dataset_config):
            from jaxarc.envs.config import ArcEnvConfig
            from jaxarc.envs.functional import _create_demo_task

            config = ArcEnvConfig()
            return _create_demo_task(config)

        base_config = create_standard_config()

        # This should work but show deprecation warning
        with pytest.warns(match="create_config_with_task_sampler is deprecated"):
            config = create_config_with_task_sampler(base_config, mock_task_sampler)

        assert config.parser is not None

        # Test that it still works with functional API
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config)
        assert isinstance(state, ArcEnvState)


if __name__ == "__main__":
    pytest.main([__file__])
