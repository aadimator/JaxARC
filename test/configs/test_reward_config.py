"""
Tests for RewardConfig reward function configurations.

This module tests reward configuration including validation, reward function
parameters, and Equinox compliance.
"""

from __future__ import annotations

import jax
import pytest
from omegaconf import DictConfig

from jaxarc.configs.reward_config import RewardConfig


class TestRewardConfigCreation:
    """Test RewardConfig creation and initialization."""

    def test_default_initialization(self):
        """Test RewardConfig creation with default values."""
        config = RewardConfig()

        # Verify default values
        assert config.step_penalty == -0.01
        assert config.success_bonus == 10.0
        assert config.similarity_weight == 1.0
        assert config.unsolved_submission_penalty == 0.0

    def test_custom_initialization(self):
        """Test RewardConfig with custom parameters."""
        config = RewardConfig(
            step_penalty=-0.05,
            success_bonus=20.0,
            similarity_weight=2.0,
            unsolved_submission_penalty=-5.0,
        )

        assert config.step_penalty == -0.05
        assert config.success_bonus == 20.0
        assert config.similarity_weight == 2.0
        assert config.unsolved_submission_penalty == -5.0


class TestRewardConfigEquinoxCompliance:
    """Test Equinox Module compliance and JAX compatibility."""

    def test_hashability(self):
        """Test that RewardConfig is hashable for JAX compatibility."""
        config = RewardConfig()

        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_check_init_validation(self):
        """Test that __check_init__ validates hashability."""
        config = RewardConfig()
        config.__check_init__()  # Should not raise

    def test_jax_tree_compatibility(self):
        """Test that RewardConfig works as a JAX PyTree."""
        config = RewardConfig(success_bonus=42.0)

        def dummy_func(cfg):
            return cfg.success_bonus

        # Should work with JAX transformations
        result = jax.jit(dummy_func)(config)
        assert result == 42.0

    def test_immutability(self):
        """Test that RewardConfig is immutable."""
        config = RewardConfig()

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.success_bonus = 999.0


class TestRewardConfigValidation:
    """Test RewardConfig validation functionality."""

    def test_validate_returns_tuple(self):
        """Test that validate() returns a tuple of error strings."""
        config = RewardConfig()
        errors = config.validate()

        assert isinstance(errors, tuple)
        for error in errors:
            assert isinstance(error, str)

    def test_valid_config_has_no_errors(self):
        """Test that valid configuration returns no errors."""
        config = RewardConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_invalid_step_penalty_range(self):
        """Test validation of step_penalty out of range."""
        # Below minimum
        config = RewardConfig(step_penalty=-15.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("step_penalty" in error for error in errors)

        # Above maximum
        config = RewardConfig(step_penalty=2.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("step_penalty" in error for error in errors)

    def test_invalid_success_bonus_range(self):
        """Test validation of success_bonus out of range."""
        # Below minimum
        config = RewardConfig(success_bonus=-150.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("success_bonus" in error for error in errors)

        # Above maximum
        config = RewardConfig(success_bonus=1500.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("success_bonus" in error for error in errors)

    def test_invalid_similarity_weight_range(self):
        """Test validation of similarity_weight out of range."""
        # Below minimum
        config = RewardConfig(similarity_weight=-1.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("similarity_weight" in error for error in errors)

        # Above maximum
        config = RewardConfig(similarity_weight=150.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("similarity_weight" in error for error in errors)

    def test_invalid_unsolved_submission_penalty_range(self):
        """Test validation of unsolved_submission_penalty out of range."""
        # Below minimum
        config = RewardConfig(unsolved_submission_penalty=-1500.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("unsolved_submission_penalty" in error for error in errors)

        # Above maximum (should be <= 0)
        config = RewardConfig(unsolved_submission_penalty=5.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("unsolved_submission_penalty" in error for error in errors)


class TestRewardConfigHydraIntegration:
    """Test Hydra integration and configuration loading."""

    def test_from_hydra_empty_config(self):
        """Test creating RewardConfig from empty Hydra config."""
        hydra_config = DictConfig({})
        config = RewardConfig.from_hydra(hydra_config)

        # Should use defaults
        assert config.step_penalty == -0.01
        assert config.success_bonus == 10.0

    def test_from_hydra_with_values(self):
        """Test creating RewardConfig from Hydra config with values."""
        hydra_config = DictConfig(
            {
                "step_penalty": -0.02,
                "success_bonus": 15.0,
                "similarity_weight": 1.5,
                "unsolved_submission_penalty": -2.0,
            }
        )

        config = RewardConfig.from_hydra(hydra_config)

        assert config.step_penalty == -0.02
        assert config.success_bonus == 15.0
        assert config.similarity_weight == 1.5
        assert config.unsolved_submission_penalty == -2.0


class TestRewardConfigRewardFunctionParameters:
    """Test reward function parameter configurations."""

    def test_step_based_reward_configuration(self):
        """Test step-based reward configuration."""
        config = RewardConfig(
            step_penalty=-0.1, success_bonus=10.0, similarity_weight=2.0
        )

        errors = config.validate()
        assert len(errors) == 0

        assert config.step_penalty == -0.1
        assert config.similarity_weight == 2.0

    def test_penalty_configuration(self):
        """Test penalty configuration."""
        config = RewardConfig(step_penalty=-0.05, unsolved_submission_penalty=-10.0)

        errors = config.validate()
        assert len(errors) == 0

        assert config.step_penalty == -0.05
        assert config.unsolved_submission_penalty == -10.0

    def test_similarity_weight_configuration(self):
        """Test similarity weight configuration."""
        config = RewardConfig(similarity_weight=3.0)

        errors = config.validate()
        assert len(errors) == 0

        assert config.similarity_weight == 3.0

    def test_balanced_reward_configuration(self):
        """Test balanced reward configuration with all parameters."""
        config = RewardConfig(
            step_penalty=-0.02,
            success_bonus=25.0,
            similarity_weight=1.5,
            unsolved_submission_penalty=-5.0,
        )

        errors = config.validate()
        assert len(errors) == 0

        # Verify all parameters are set correctly
        assert config.step_penalty == -0.02
        assert config.success_bonus == 25.0
        assert config.similarity_weight == 1.5
        assert config.unsolved_submission_penalty == -5.0
