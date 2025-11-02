"""
Tests for EnvironmentConfig environment behavior configurations.

This module tests environment configuration including behavior settings,
episode parameters, and debug level configurations.
"""

from __future__ import annotations

import jax
import pytest
from omegaconf import DictConfig

from jaxarc.configs.environment_config import EnvironmentConfig


class TestEnvironmentConfigCreation:
    """Test EnvironmentConfig creation and initialization."""

    def test_default_initialization(self):
        """Test EnvironmentConfig creation with default values."""
        config = EnvironmentConfig()

        # Verify default values
        assert config.max_episode_steps == 100
        assert config.auto_reset is True
        assert config.debug_level == "minimal"

    def test_custom_initialization(self):
        """Test EnvironmentConfig with custom parameters."""
        config = EnvironmentConfig(
            max_episode_steps=200, auto_reset=False, debug_level="verbose"
        )

        assert config.max_episode_steps == 200
        assert config.auto_reset is False
        assert config.debug_level == "verbose"


class TestEnvironmentConfigEquinoxCompliance:
    """Test Equinox Module compliance and JAX compatibility."""

    def test_hashability(self):
        """Test that EnvironmentConfig is hashable for JAX compatibility."""
        config = EnvironmentConfig()

        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_check_init_validation(self):
        """Test that __check_init__ validates hashability."""
        config = EnvironmentConfig()
        config.__check_init__()  # Should not raise

    def test_jax_tree_compatibility(self):
        """Test that EnvironmentConfig works as a JAX PyTree."""
        config = EnvironmentConfig(max_episode_steps=42)

        def dummy_func(cfg):
            return cfg.max_episode_steps

        # Should work with JAX transformations
        result = jax.jit(dummy_func)(config)
        assert result == 42

    def test_immutability(self):
        """Test that EnvironmentConfig is immutable."""
        config = EnvironmentConfig()

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.max_episode_steps = 999


class TestEnvironmentConfigValidation:
    """Test EnvironmentConfig validation functionality."""

    def test_validate_returns_tuple(self):
        """Test that validate() returns a tuple of error strings."""
        config = EnvironmentConfig()
        errors = config.validate()

        assert isinstance(errors, tuple)
        for error in errors:
            assert isinstance(error, str)

    def test_valid_config_has_no_errors(self):
        """Test that valid configuration returns no errors."""
        config = EnvironmentConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_invalid_max_episode_steps(self):
        """Test validation of invalid max_episode_steps values."""
        # Negative max_episode_steps
        config = EnvironmentConfig(max_episode_steps=-5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_episode_steps" in error for error in errors)

        # Zero max_episode_steps
        config = EnvironmentConfig(max_episode_steps=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_episode_steps" in error for error in errors)

    def test_invalid_debug_level(self):
        """Test validation of invalid debug level."""
        config = EnvironmentConfig(debug_level="invalid_level")
        errors = config.validate()
        assert len(errors) > 0
        assert any("debug_level" in error for error in errors)

    def test_valid_debug_levels(self):
        """Test that all valid debug levels are accepted."""
        valid_levels = ["off", "minimal", "verbose"]

        for level in valid_levels:
            config = EnvironmentConfig(debug_level=level)
            errors = config.validate()
            # Should not have errors related to debug_level
            level_errors = [e for e in errors if "debug_level" in e]
            assert len(level_errors) == 0


class TestEnvironmentConfigHydraIntegration:
    """Test Hydra integration and configuration loading."""

    def test_from_hydra_empty_config(self):
        """Test creating EnvironmentConfig from empty Hydra config."""
        hydra_config = DictConfig({})
        config = EnvironmentConfig.from_hydra(hydra_config)

        # Should use defaults
        assert config.max_episode_steps == 100
        assert config.auto_reset is True
        assert config.debug_level == "minimal"

    def test_from_hydra_with_values(self):
        """Test creating EnvironmentConfig from Hydra config with values."""
        hydra_config = DictConfig(
            {"max_episode_steps": 150, "auto_reset": False, "debug_level": "verbose"}
        )

        config = EnvironmentConfig.from_hydra(hydra_config)

        assert config.max_episode_steps == 150
        assert config.auto_reset is False
        assert config.debug_level == "verbose"


class TestEnvironmentConfigBehaviorSettings:
    """Test environment behavior setting configurations."""

    def test_episode_configuration(self):
        """Test episode-related configuration."""
        config = EnvironmentConfig(max_episode_steps=500, auto_reset=True)

        errors = config.validate()
        assert len(errors) == 0

        assert config.max_episode_steps == 500
        assert config.auto_reset is True

    def test_debug_configuration(self):
        """Test debug level configuration."""
        # Test each debug level
        for level in ["off", "minimal", "verbose"]:
            config = EnvironmentConfig(debug_level=level)
            errors = config.validate()
            assert len(errors) == 0
            assert config.debug_level == level

    def test_short_episode_configuration(self):
        """Test configuration for short episodes."""
        config = EnvironmentConfig(max_episode_steps=10, debug_level="off")

        errors = config.validate()
        assert len(errors) == 0

        assert config.max_episode_steps == 10
        assert config.debug_level == "off"

    def test_long_episode_configuration(self):
        """Test configuration for long episodes."""
        config = EnvironmentConfig(max_episode_steps=1000, debug_level="verbose")

        errors = config.validate()
        assert len(errors) == 0

        assert config.max_episode_steps == 1000
        assert config.debug_level == "verbose"

    def test_no_auto_reset_configuration(self):
        """Test configuration without auto reset."""
        config = EnvironmentConfig(auto_reset=False, max_episode_steps=50)

        errors = config.validate()
        assert len(errors) == 0

        assert config.auto_reset is False
        assert config.max_episode_steps == 50
