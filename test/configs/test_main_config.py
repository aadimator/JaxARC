"""
Tests for JaxArcConfig unified configuration.

This module tests the main configuration class that unifies all configuration
aspects, including validation, Equinox compliance, and cross-config consistency.
"""

from __future__ import annotations

import jax
import pytest
from omegaconf import DictConfig

from jaxarc.configs import JaxArcConfig
from jaxarc.configs.action_config import ActionConfig
from jaxarc.configs.dataset_config import DatasetConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.reward_config import RewardConfig


class TestJaxArcConfigCreation:
    """Test JaxArcConfig creation and initialization."""

    def test_default_initialization(self):
        """Test that JaxArcConfig can be created with default values."""
        config = JaxArcConfig()

        # Verify all components are initialized
        assert config.environment is not None
        assert config.dataset is not None
        assert config.action is not None
        assert config.reward is not None
        assert config.grid_initialization is not None
        assert config.visualization is not None
        assert config.storage is not None
        assert config.logging is not None
        assert config.wandb is not None

    def test_custom_component_initialization(self):
        """Test JaxArcConfig with custom component configurations."""
        custom_env = EnvironmentConfig(max_episode_steps=200, debug_level="verbose")
        custom_dataset = DatasetConfig(max_grid_height=50, max_colors=15)

        config = JaxArcConfig(environment=custom_env, dataset=custom_dataset)

        assert config.environment.max_episode_steps == 200
        assert config.environment.debug_level == "verbose"
        assert config.dataset.max_grid_height == 50
        assert config.dataset.max_colors == 15

    def test_partial_component_initialization(self):
        """Test that unspecified components use defaults."""
        custom_action = ActionConfig(max_operations=50)

        config = JaxArcConfig(action=custom_action)

        # Custom component should have custom values
        assert config.action.max_operations == 50

        # Other components should have defaults
        assert config.environment.max_episode_steps == 100  # default
        assert config.dataset.max_grid_height == 30  # default


class TestJaxArcConfigEquinoxCompliance:
    """Test Equinox Module compliance and JAX compatibility."""

    def test_hashability(self):
        """Test that JaxArcConfig is hashable for JAX compatibility."""
        config = JaxArcConfig()

        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_check_init_validation(self):
        """Test that __check_init__ validates hashability."""
        # This should not raise an exception
        config = JaxArcConfig()
        config.__check_init__()

    def test_jax_tree_compatibility(self):
        """Test that JaxArcConfig works as a JAX PyTree."""
        config = JaxArcConfig()

        # Should be able to use in JAX transformations
        def dummy_func(cfg):
            return cfg.environment.max_episode_steps

        # This should not raise an error
        result = jax.jit(dummy_func)(config)
        assert result == config.environment.max_episode_steps

    def test_immutability(self):
        """Test that JaxArcConfig is immutable (Equinox Module behavior)."""
        config = JaxArcConfig()

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.environment = EnvironmentConfig(max_episode_steps=999)


class TestJaxArcConfigValidation:
    """Test JaxArcConfig validation functionality."""

    def test_validate_returns_tuple(self):
        """Test that validate() returns a tuple of error strings."""
        config = JaxArcConfig()
        errors = config.validate()

        assert isinstance(errors, tuple)
        # All strings in the tuple should be strings
        for error in errors:
            assert isinstance(error, str)

    def test_valid_config_has_no_errors(self):
        """Test that a valid configuration returns no errors."""
        config = JaxArcConfig()
        errors = config.validate()

        # Should have no validation errors
        assert len(errors) == 0

    def test_invalid_component_config_detected(self):
        """Test that invalid component configurations are detected."""
        # Create config with invalid action config
        invalid_action = ActionConfig(max_operations=-5)  # Invalid: negative
        config = JaxArcConfig(action=invalid_action)

        errors = config.validate()
        assert len(errors) > 0

        # Should contain error about max_operations
        error_text = " ".join(errors)
        assert "max_operations" in error_text

    def test_cross_config_validation(self):
        """Test cross-configuration consistency validation."""
        # Create configuration that should trigger cross-validation warnings
        env_config = EnvironmentConfig(max_episode_steps=5)  # Very short episodes
        reward_config = RewardConfig(step_penalty=-0.1)  # Higher step penalty

        config = JaxArcConfig(environment=env_config, reward=reward_config)

        # This should trigger warnings (which are logged, not returned as errors)
        # The validate method should still complete without errors
        errors = config.validate()

        # Cross-validation issues are warnings, not errors, so errors should be empty
        # unless there are actual validation errors
        assert isinstance(errors, tuple)


class TestJaxArcConfigHydraIntegration:
    """Test Hydra integration and configuration loading."""

    def test_from_hydra_empty_config(self):
        """Test creating JaxArcConfig from empty Hydra config."""
        hydra_config = DictConfig({})
        config = JaxArcConfig.from_hydra(hydra_config)

        # Should create valid config with defaults
        assert isinstance(config, JaxArcConfig)
        assert config.environment.max_episode_steps == 100  # default

    def test_from_hydra_partial_config(self):
        """Test creating JaxArcConfig from partial Hydra config."""
        hydra_config = DictConfig(
            {
                "environment": {"max_episode_steps": 150, "debug_level": "verbose"},
                "action": {
                    "num_operations": 40
                },  # ActionConfig uses num_operations in Hydra
            }
        )

        config = JaxArcConfig.from_hydra(hydra_config)

        # Should use provided values
        assert config.environment.max_episode_steps == 150
        assert config.environment.debug_level == "verbose"
        assert config.action.max_operations == 40

        # Should use defaults for unspecified components
        assert config.dataset.max_grid_height == 30  # default

    def test_from_hydra_complete_config(self):
        """Test creating JaxArcConfig from complete Hydra config."""
        hydra_config = DictConfig(
            {
                "environment": {"max_episode_steps": 200, "auto_reset": False},
                "dataset": {"max_grid_height": 40, "max_colors": 12},
                "action": {
                    "num_operations": 50,
                    "validate_actions": False,
                },  # Use num_operations
                "reward": {"success_bonus": 20.0, "step_penalty": -0.02},
                "grid_initialization": {},
                "visualization": {},
                "storage": {},
                "logging": {},
                "wandb": {},
            }
        )

        config = JaxArcConfig.from_hydra(hydra_config)

        # Verify all specified values are used
        assert config.environment.max_episode_steps == 200
        assert config.environment.auto_reset is False
        assert config.dataset.max_grid_height == 40
        assert config.dataset.max_colors == 12
        assert config.action.max_operations == 50
        assert config.action.validate_actions is False
        assert config.reward.success_bonus == 20.0
        assert config.reward.step_penalty == -0.02

    def test_from_hydra_invalid_config_validation_errors(self):
        """Test that invalid Hydra config creates config with validation errors."""
        # Create config with invalid values
        hydra_config = DictConfig(
            {
                "environment": {"max_episode_steps": "invalid"}  # Should be int
            }
        )

        # Config creation should succeed but validation should fail
        config = JaxArcConfig.from_hydra(hydra_config)
        errors = config.validate()

        # Should have validation errors
        assert len(errors) > 0
        assert any("max_episode_steps" in error for error in errors)


class TestJaxArcConfigSerialization:
    """Test configuration serialization and export functionality."""

    def test_to_yaml_basic(self):
        """Test basic YAML export functionality."""
        config = JaxArcConfig()
        yaml_output = config.to_yaml()

        assert isinstance(yaml_output, str)
        assert len(yaml_output) > 0

        # Should contain main configuration sections
        assert "environment:" in yaml_output
        assert "dataset:" in yaml_output
        assert "action:" in yaml_output
        assert "reward:" in yaml_output

    def test_to_yaml_with_custom_values(self):
        """Test YAML export with custom configuration values."""
        custom_config = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=250),
            action=ActionConfig(max_operations=60),
        )

        yaml_output = custom_config.to_yaml()

        # Should contain custom values
        assert "max_episode_steps: 250" in yaml_output
        assert "max_operations: 60" in yaml_output

    def test_to_yaml_file(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = JaxArcConfig()
        yaml_file = tmp_path / "test_config.yaml"

        # Should not raise an error
        config.to_yaml_file(yaml_file)

        # File should exist and contain content
        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert len(content) > 0
        assert "environment:" in content

    def test_to_yaml_file_creates_directories(self, tmp_path):
        """Test that to_yaml_file creates necessary directories."""
        config = JaxArcConfig()
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"

        # Should create directories and file
        config.to_yaml_file(nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestJaxArcConfigComposition:
    """Test configuration composition and defaults."""

    def test_configuration_composition(self):
        """Test that configuration properly composes all components."""
        config = JaxArcConfig()

        # All components should be present and properly typed
        assert isinstance(config.environment, EnvironmentConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.action, ActionConfig)
        assert isinstance(config.reward, RewardConfig)

        # Components should have their default values
        assert config.environment.max_episode_steps > 0
        assert config.dataset.max_grid_height > 0
        assert config.action.max_operations > 0
        assert config.reward.success_bonus != 0

    def test_configuration_defaults_are_reasonable(self):
        """Test that default configuration values are reasonable."""
        config = JaxArcConfig()

        # Environment defaults
        assert 10 <= config.environment.max_episode_steps <= 1000
        assert config.environment.debug_level in ["off", "minimal", "verbose"]

        # Dataset defaults
        assert 1 <= config.dataset.max_grid_height <= 100
        assert 1 <= config.dataset.max_grid_width <= 100
        assert 2 <= config.dataset.max_colors <= 50

        # Action defaults
        assert 1 <= config.action.max_operations <= 200

        # Reward defaults
        assert config.reward.success_bonus > 0
        assert config.reward.step_penalty <= 0

    def test_component_independence(self):
        """Test that components can be configured independently."""
        # Create configs with different settings
        config1 = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=100))
        config2 = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=200))

        # Configs should be different
        assert (
            config1.environment.max_episode_steps
            != config2.environment.max_episode_steps
        )

        # Other components should be the same (defaults)
        assert config1.dataset.max_grid_height == config2.dataset.max_grid_height
        assert config1.action.max_operations == config2.action.max_operations
