"""
Tests for ActionConfig action format configurations.

This module tests action configuration including operation masks, validation,
action space parameters, and operation filtering functionality.
"""

from __future__ import annotations

import jax
import pytest
from omegaconf import DictConfig

from jaxarc.configs.action_config import ActionConfig


class TestActionConfigCreation:
    """Test ActionConfig creation and initialization."""

    def test_default_initialization(self):
        """Test ActionConfig creation with default values."""
        config = ActionConfig()

        # Verify default values
        assert config.max_operations == 35
        assert config.allowed_operations is None
        assert config.validate_actions is True
        assert config.allow_invalid_actions is False
        assert config.dynamic_action_filtering is False
        assert config.context_dependent_operations is False
        assert config.invalid_operation_policy == "clip"
        assert config.selection_threshold == 1.0

    def test_custom_initialization(self):
        """Test ActionConfig with custom parameters."""
        config = ActionConfig(
            max_operations=50,
            allowed_operations=(0, 1, 2, 10, 15),
            validate_actions=False,
            allow_invalid_actions=True,
            dynamic_action_filtering=True,
            context_dependent_operations=True,
            invalid_operation_policy="reject",
            selection_threshold=0.8,
        )

        assert config.max_operations == 50
        assert config.allowed_operations == (0, 1, 2, 10, 15)
        assert config.validate_actions is False
        assert config.allow_invalid_actions is True
        assert config.dynamic_action_filtering is True
        assert config.context_dependent_operations is True
        assert config.invalid_operation_policy == "reject"
        assert config.selection_threshold == 0.8

    def test_allowed_operations_tuple_conversion(self):
        """Test that allowed_operations is converted to tuple."""
        # Test with list input
        config = ActionConfig(allowed_operations=[0, 1, 2, 3])
        assert isinstance(config.allowed_operations, tuple)
        assert config.allowed_operations == (0, 1, 2, 3)

        # Test with None input
        config = ActionConfig(allowed_operations=None)
        assert config.allowed_operations is None

        # Test with empty list
        config = ActionConfig(allowed_operations=[])
        assert config.allowed_operations is None


class TestActionConfigEquinoxCompliance:
    """Test Equinox Module compliance and JAX compatibility."""

    def test_hashability(self):
        """Test that ActionConfig is hashable for JAX compatibility."""
        config = ActionConfig()

        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_check_init_validation(self):
        """Test that __check_init__ validates hashability."""
        config = ActionConfig()
        config.__check_init__()  # Should not raise

    def test_jax_tree_compatibility(self):
        """Test that ActionConfig works as a JAX PyTree."""
        config = ActionConfig(max_operations=42)

        def dummy_func(cfg):
            return cfg.max_operations

        # Should work with JAX transformations
        result = jax.jit(dummy_func)(config)
        assert result == 42

    def test_immutability(self):
        """Test that ActionConfig is immutable."""
        config = ActionConfig()

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.max_operations = 999


class TestActionConfigValidation:
    """Test ActionConfig validation functionality."""

    def test_validate_returns_tuple(self):
        """Test that validate() returns a tuple of error strings."""
        config = ActionConfig()
        errors = config.validate()

        assert isinstance(errors, tuple)
        for error in errors:
            assert isinstance(error, str)

    def test_valid_config_has_no_errors(self):
        """Test that valid configuration returns no errors."""
        config = ActionConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_invalid_max_operations(self):
        """Test validation of invalid max_operations values."""
        # Negative max_operations
        config = ActionConfig(max_operations=-5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_operations" in error for error in errors)

        # Zero max_operations
        config = ActionConfig(max_operations=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_operations" in error for error in errors)

    def test_invalid_selection_threshold(self):
        """Test validation of invalid selection_threshold values."""
        # Below range
        config = ActionConfig(selection_threshold=-0.1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("selection_threshold" in error for error in errors)

        # Above range
        config = ActionConfig(selection_threshold=1.5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("selection_threshold" in error for error in errors)

    def test_invalid_operation_policy_validation(self):
        """Test validation of invalid operation policy."""
        config = ActionConfig(invalid_operation_policy="invalid_policy")
        errors = config.validate()
        assert len(errors) > 0
        assert any("invalid_operation_policy" in error for error in errors)

    def test_valid_operation_policies(self):
        """Test that all valid operation policies are accepted."""
        valid_policies = ["clip", "reject", "passthrough", "penalize"]

        for policy in valid_policies:
            config = ActionConfig(invalid_operation_policy=policy)
            errors = config.validate()
            # Should not have errors related to policy
            policy_errors = [e for e in errors if "invalid_operation_policy" in e]
            assert len(policy_errors) == 0


class TestActionConfigOperationMasks:
    """Test operation mask and filtering functionality."""

    def test_allowed_operations_validation(self):
        """Test validation of allowed_operations parameter."""
        # Valid allowed_operations
        config = ActionConfig(max_operations=10, allowed_operations=(0, 1, 2, 5, 9))
        errors = config.validate()
        assert len(errors) == 0

    def test_allowed_operations_out_of_range(self):
        """Test validation when allowed_operations contains out-of-range values."""
        config = ActionConfig(
            max_operations=10,
            allowed_operations=(0, 1, 15),  # 15 is >= max_operations
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "allowed_operations" in error and "range" in error for error in errors
        )

    def test_allowed_operations_negative_values(self):
        """Test validation when allowed_operations contains negative values."""
        config = ActionConfig(
            max_operations=10,
            allowed_operations=(-1, 0, 1),  # -1 is invalid
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "allowed_operations" in error and "range" in error for error in errors
        )

    def test_allowed_operations_duplicates(self):
        """Test validation when allowed_operations contains duplicates."""
        config = ActionConfig(
            max_operations=10,
            allowed_operations=(0, 1, 2, 1, 3),  # 1 is duplicated
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("duplicate" in error for error in errors)

    def test_allowed_operations_empty_tuple(self):
        """Test validation when allowed_operations is empty."""
        config = ActionConfig(max_operations=10, allowed_operations=())
        errors = config.validate()
        assert len(errors) > 0
        assert any("empty" in error for error in errors)

    def test_allowed_operations_non_integer_values(self):
        """Test validation when allowed_operations contains non-integers."""
        # This test checks the validation logic, though tuple conversion happens in __init__
        config = ActionConfig(max_operations=10)
        # Manually set to test validation (normally prevented by __init__)
        object.__setattr__(config, "allowed_operations", (0, 1.5, 2))

        errors = config.validate()
        assert len(errors) > 0
        assert any("integer" in error for error in errors)


class TestActionConfigDynamicFiltering:
    """Test dynamic action filtering and context-dependent operations."""

    def test_dynamic_filtering_configuration(self):
        """Test dynamic action filtering configuration options."""
        config = ActionConfig(
            dynamic_action_filtering=True, context_dependent_operations=True
        )

        assert config.dynamic_action_filtering is True
        assert config.context_dependent_operations is True

        # Should validate without errors
        errors = config.validate()
        assert len(errors) == 0

    def test_context_dependent_without_dynamic_filtering_warning(self):
        """Test that context_dependent_operations without dynamic_filtering logs warning."""
        # This configuration should trigger a warning (logged, not returned as error)
        config = ActionConfig(
            dynamic_action_filtering=False, context_dependent_operations=True
        )

        # Validation should still pass (warnings are logged, not returned as errors)
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_settings_interaction(self):
        """Test interaction between validation settings."""
        # This configuration should trigger a warning (logged, not returned as error)
        config = ActionConfig(
            validate_actions=False,
            allow_invalid_actions=False,  # Has no effect when validate_actions=False
        )

        # Should still validate (warnings are logged, not errors)
        errors = config.validate()
        assert len(errors) == 0


class TestActionConfigHydraIntegration:
    """Test Hydra integration and configuration loading."""

    def test_from_hydra_empty_config(self):
        """Test creating ActionConfig from empty Hydra config."""
        hydra_config = DictConfig({})
        config = ActionConfig.from_hydra(hydra_config)

        # Should use defaults
        assert config.max_operations == 35  # Default from num_operations
        assert config.validate_actions is True
        assert (
            config.allow_invalid_actions is False
        )  # Inverted from clip_invalid_actions

    def test_from_hydra_with_values(self):
        """Test creating ActionConfig from Hydra config with values."""
        hydra_config = DictConfig(
            {
                "num_operations": 50,
                "allowed_operations": [0, 1, 2, 10],
                "validate_actions": False,
                "clip_invalid_actions": False,  # Should invert to allow_invalid_actions=True
                "dynamic_action_filtering": True,
                "context_dependent_operations": True,
                "invalid_operation_policy": "reject",
                "selection_threshold": 0.9,
            }
        )

        config = ActionConfig.from_hydra(hydra_config)

        assert config.max_operations == 50
        assert config.allowed_operations == (0, 1, 2, 10)
        assert config.validate_actions is False
        assert (
            config.allow_invalid_actions is True
        )  # Inverted from clip_invalid_actions=False
        assert config.dynamic_action_filtering is True
        assert config.context_dependent_operations is True
        assert config.invalid_operation_policy == "reject"
        assert config.selection_threshold == 0.9

    def test_from_hydra_allowed_operations_conversion(self):
        """Test that Hydra config properly converts allowed_operations to tuple."""
        hydra_config = DictConfig({"allowed_operations": [5, 10, 15, 20]})

        config = ActionConfig.from_hydra(hydra_config)

        assert isinstance(config.allowed_operations, tuple)
        assert config.allowed_operations == (5, 10, 15, 20)

    def test_from_hydra_none_allowed_operations(self):
        """Test Hydra config with None allowed_operations."""
        hydra_config = DictConfig({"allowed_operations": None})

        config = ActionConfig.from_hydra(hydra_config)
        assert config.allowed_operations is None

    def test_from_hydra_empty_allowed_operations(self):
        """Test Hydra config with empty allowed_operations."""
        hydra_config = DictConfig({"allowed_operations": []})

        config = ActionConfig.from_hydra(hydra_config)
        assert config.allowed_operations is None


class TestActionConfigOperationFiltering:
    """Test operation filtering and mask generation functionality."""

    def test_operation_mask_generation_concept(self):
        """Test the concept of operation mask generation."""
        # Test different configurations that would affect mask generation

        # Configuration with limited operations
        limited_config = ActionConfig(
            max_operations=20, allowed_operations=(0, 1, 2, 5, 10)
        )

        # Should validate successfully
        errors = limited_config.validate()
        assert len(errors) == 0

        # Configuration with all operations allowed
        full_config = ActionConfig(
            max_operations=35,
            allowed_operations=None,  # All operations allowed
        )

        errors = full_config.validate()
        assert len(errors) == 0

    def test_operation_validation_policies(self):
        """Test different operation validation policies."""
        policies = ["clip", "reject", "passthrough", "penalize"]

        for policy in policies:
            config = ActionConfig(
                invalid_operation_policy=policy, validate_actions=True
            )

            errors = config.validate()
            assert len(errors) == 0
            assert config.invalid_operation_policy == policy

    def test_action_space_parameters(self):
        """Test action space parameter validation."""
        config = ActionConfig(
            max_operations=100,
            selection_threshold=0.5,
            validate_actions=True,
            dynamic_action_filtering=True,
        )

        errors = config.validate()
        assert len(errors) == 0

        # Verify all parameters are set correctly
        assert config.max_operations == 100
        assert config.selection_threshold == 0.5
        assert config.validate_actions is True
        assert config.dynamic_action_filtering is True
