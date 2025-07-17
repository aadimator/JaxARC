"""
Tests for enhanced configuration validation.

This module tests the comprehensive validation system for JaxARC configuration
dataclasses, including field validators, cross-field validation, and error handling.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from jaxarc.envs.config import (
    ActionConfig,
    ArcEnvConfig,
    ConfigValidationError,
    DatasetConfig,
    DebugConfig,
    GridConfig,
    RewardConfig,
    validate_config,
    validate_dataset_name,
    validate_float_range,
    validate_non_negative_int,
    validate_operation_list,
    validate_path_string,
    validate_positive_int,
    validate_string_choice,
)


class TestValidationUtilities:
    """Test the validation utility functions."""

    def test_validate_positive_int(self):
        """Test positive integer validation."""
        # Valid cases
        validate_positive_int(1, "test_field")
        validate_positive_int(100, "test_field")
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be an integer"):
            validate_positive_int("not_int", "test_field")
        
        with pytest.raises(ConfigValidationError, match="must be positive"):
            validate_positive_int(0, "test_field")
        
        with pytest.raises(ConfigValidationError, match="must be positive"):
            validate_positive_int(-1, "test_field")

    def test_validate_non_negative_int(self):
        """Test non-negative integer validation."""
        # Valid cases
        validate_non_negative_int(0, "test_field")
        validate_non_negative_int(1, "test_field")
        validate_non_negative_int(100, "test_field")
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be an integer"):
            validate_non_negative_int("not_int", "test_field")
        
        with pytest.raises(ConfigValidationError, match="must be non-negative"):
            validate_non_negative_int(-1, "test_field")

    def test_validate_float_range(self):
        """Test float range validation."""
        # Valid cases
        validate_float_range(0.5, "test_field", 0.0, 1.0)
        validate_float_range(0.0, "test_field", 0.0, 1.0)
        validate_float_range(1.0, "test_field", 0.0, 1.0)
        validate_float_range(5, "test_field", 0.0, 10.0)  # int should work too
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be a number"):
            validate_float_range("not_number", "test_field", 0.0, 1.0)
        
        with pytest.raises(ConfigValidationError, match="must be in range"):
            validate_float_range(-0.1, "test_field", 0.0, 1.0)
        
        with pytest.raises(ConfigValidationError, match="must be in range"):
            validate_float_range(1.1, "test_field", 0.0, 1.0)

    def test_validate_string_choice(self):
        """Test string choice validation."""
        choices = ["option1", "option2", "option3"]
        
        # Valid cases
        validate_string_choice("option1", "test_field", choices)
        validate_string_choice("option2", "test_field", choices)
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be a string"):
            validate_string_choice(123, "test_field", choices)
        
        with pytest.raises(ConfigValidationError, match="must be one of"):
            validate_string_choice("invalid_option", "test_field", choices)

    def test_validate_path_string(self):
        """Test path string validation."""
        # Valid cases
        validate_path_string("valid/path", "test_field")
        validate_path_string("", "test_field")  # Empty path should be valid
        validate_path_string("output/logs", "test_field")
        
        # Invalid cases - invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            with pytest.raises(ConfigValidationError, match="contains invalid path characters"):
                validate_path_string(f"invalid{char}path", "test_field")
        
        # Invalid type
        with pytest.raises(ConfigValidationError, match="must be a string"):
            validate_path_string(123, "test_field")

    def test_validate_operation_list(self):
        """Test operation list validation."""
        # Valid cases
        validate_operation_list(None, "test_field", 35)  # None should be valid
        validate_operation_list([0, 1, 2], "test_field", 35)
        validate_operation_list([34], "test_field", 35)  # Max valid operation
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be a list or None"):
            validate_operation_list("not_list", "test_field", 35)
        
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            validate_operation_list([], "test_field", 35)
        
        with pytest.raises(ConfigValidationError, match="must be an integer"):
            validate_operation_list([0, "not_int"], "test_field", 35)
        
        with pytest.raises(ConfigValidationError, match="must be in range"):
            validate_operation_list([35], "test_field", 35)  # Out of range
        
        with pytest.raises(ConfigValidationError, match="must be in range"):
            validate_operation_list([-1], "test_field", 35)  # Negative
        
        with pytest.raises(ConfigValidationError, match="contains duplicate operations"):
            validate_operation_list([0, 1, 0], "test_field", 35)

    def test_validate_dataset_name(self):
        """Test dataset name validation."""
        # Valid cases
        validate_dataset_name("arc-agi-1", "test_field")
        validate_dataset_name("concept_arc", "test_field")
        validate_dataset_name("MiniARC123", "test_field")
        
        # Invalid cases
        with pytest.raises(ConfigValidationError, match="must be a string"):
            validate_dataset_name(123, "test_field")
        
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            validate_dataset_name("", "test_field")
        
        with pytest.raises(ConfigValidationError, match="must contain only alphanumeric"):
            validate_dataset_name("invalid@name", "test_field")
        
        with pytest.raises(ConfigValidationError, match="must contain only alphanumeric"):
            validate_dataset_name("invalid name", "test_field")  # Space not allowed


class TestDebugConfigValidation:
    """Test DebugConfig validation."""

    def test_valid_debug_config(self):
        """Test valid debug configuration."""
        config = DebugConfig(
            log_rl_steps=True,
            rl_steps_output_dir="output/debug",
            clear_output_dir=False,
        )
        assert config.log_rl_steps is True
        assert config.rl_steps_output_dir == "output/debug"
        assert config.clear_output_dir is False

    def test_invalid_debug_config_types(self):
        """Test invalid debug configuration types."""
        # Invalid log_rl_steps type
        with pytest.raises(ConfigValidationError, match="log_rl_steps must be a boolean"):
            DebugConfig(log_rl_steps="not_bool")
        
        # Invalid clear_output_dir type
        with pytest.raises(ConfigValidationError, match="clear_output_dir must be a boolean"):
            DebugConfig(clear_output_dir="not_bool")

    def test_debug_config_from_hydra(self):
        """Test DebugConfig creation from Hydra config."""
        hydra_cfg = OmegaConf.create({
            "log_rl_steps": True,
            "rl_steps_output_dir": "custom/output",
            "clear_output_dir": False,
        })
        
        config = DebugConfig.from_hydra(hydra_cfg)
        assert config.log_rl_steps is True
        assert config.rl_steps_output_dir == "custom/output"
        assert config.clear_output_dir is False


class TestRewardConfigValidation:
    """Test RewardConfig validation."""

    def test_valid_reward_config(self):
        """Test valid reward configuration."""
        config = RewardConfig(
            reward_on_submit_only=True,
            step_penalty=-0.01,
            success_bonus=10.0,
            similarity_weight=1.0,
            progress_bonus=0.1,
            invalid_action_penalty=-0.1,
        )
        assert config.reward_on_submit_only is True
        assert config.step_penalty == -0.01

    def test_invalid_reward_config_types(self):
        """Test invalid reward configuration types."""
        # Invalid boolean type
        with pytest.raises(ConfigValidationError, match="reward_on_submit_only must be a boolean"):
            RewardConfig(reward_on_submit_only="not_bool")
        
        # Invalid numeric types
        with pytest.raises(ConfigValidationError, match="step_penalty must be a number"):
            RewardConfig(step_penalty="not_number")
        
        with pytest.raises(ConfigValidationError, match="success_bonus must be a number"):
            RewardConfig(success_bonus="not_number")

    def test_reward_config_range_validation(self):
        """Test reward configuration range validation."""
        # Values outside reasonable ranges should raise errors
        with pytest.raises(ConfigValidationError, match="step_penalty must be in range"):
            RewardConfig(step_penalty=-15.0)  # Too negative
        
        with pytest.raises(ConfigValidationError, match="success_bonus must be in range"):
            RewardConfig(success_bonus=-200.0)  # Too negative
        
        with pytest.raises(ConfigValidationError, match="similarity_weight must be in range"):
            RewardConfig(similarity_weight=-1.0)  # Negative weight


class TestDatasetConfigValidation:
    """Test DatasetConfig validation."""

    def test_valid_dataset_config(self):
        """Test valid dataset configuration."""
        config = DatasetConfig(
            dataset_name="arc-agi-1",
            dataset_path="data/arc",
            task_split="train",
            max_tasks=100,
            shuffle_tasks=True,
        )
        assert config.dataset_name == "arc-agi-1"
        assert config.task_split == "train"

    def test_invalid_dataset_name(self):
        """Test invalid dataset name validation."""
        with pytest.raises(ConfigValidationError, match="must contain only alphanumeric"):
            DatasetConfig(dataset_name="invalid@name")

    def test_invalid_task_split(self):
        """Test invalid task split validation."""
        with pytest.raises(ConfigValidationError, match="task_split must be one of"):
            DatasetConfig(dataset_name="arc-agi-1", task_split="invalid_split")

    def test_dataset_specific_splits(self):
        """Test dataset-specific split validation."""
        # ConceptARC should accept "corpus"
        config = DatasetConfig(dataset_name="concept-arc", task_split="corpus")
        assert config.task_split == "corpus"
        
        # MiniARC should accept "training" and "evaluation"
        config = DatasetConfig(dataset_name="mini-arc", task_split="training")
        assert config.task_split == "training"

    def test_grid_constraint_validation(self):
        """Test dataset grid constraint validation."""
        # Valid constraints
        config = DatasetConfig(
            dataset_max_grid_height=20,
            dataset_max_grid_width=20,
            dataset_min_grid_height=3,
            dataset_min_grid_width=3,
            dataset_max_colors=10,
        )
        assert config.dataset_max_grid_height == 20
        
        # Invalid constraints - min > max
        with pytest.raises(ConfigValidationError, match="dataset_min_grid_height.*dataset_max_grid_height"):
            DatasetConfig(
                dataset_min_grid_height=25,
                dataset_max_grid_height=20,
            )

    def test_max_tasks_validation(self):
        """Test max_tasks validation."""
        # Valid max_tasks
        config = DatasetConfig(max_tasks=100)
        assert config.max_tasks == 100
        
        # Invalid max_tasks
        with pytest.raises(ConfigValidationError, match="max_tasks must be positive"):
            DatasetConfig(max_tasks=0)


class TestGridConfigValidation:
    """Test GridConfig validation."""

    def test_valid_grid_config(self):
        """Test valid grid configuration."""
        config = GridConfig(
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=3,
            min_grid_width=3,
            max_colors=10,
            background_color=0,
        )
        assert config.max_grid_height == 30
        assert config.background_color == 0

    def test_invalid_grid_dimensions(self):
        """Test invalid grid dimension validation."""
        # max < min
        with pytest.raises(ConfigValidationError, match="max_grid_height.*min_grid_height"):
            GridConfig(max_grid_height=2, min_grid_height=3)
        
        with pytest.raises(ConfigValidationError, match="max_grid_width.*min_grid_width"):
            GridConfig(max_grid_width=2, min_grid_width=3)

    def test_invalid_color_constraints(self):
        """Test invalid color constraint validation."""
        # max_colors too small
        with pytest.raises(ConfigValidationError, match="max_colors must be at least 2"):
            GridConfig(max_colors=1)
        
        # background_color >= max_colors
        with pytest.raises(ConfigValidationError, match="background_color.*must be.*max_colors"):
            GridConfig(max_colors=5, background_color=5)

    def test_grid_dimension_types(self):
        """Test grid dimension type validation."""
        # Non-integer dimensions
        with pytest.raises(ConfigValidationError, match="max_grid_height must be an integer"):
            GridConfig(max_grid_height="not_int")
        
        # Negative dimensions
        with pytest.raises(ConfigValidationError, match="min_grid_height must be positive"):
            GridConfig(min_grid_height=0)


class TestActionConfigValidation:
    """Test ActionConfig validation."""

    def test_valid_action_config(self):
        """Test valid action configuration."""
        config = ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            num_operations=35,
            allowed_operations=[0, 1, 2, 34],
            validate_actions=True,
            clip_invalid_actions=True,
        )
        assert config.selection_format == "mask"
        assert config.num_operations == 35

    def test_invalid_selection_format(self):
        """Test invalid selection format validation."""
        with pytest.raises(ConfigValidationError, match="selection_format must be one of"):
            ActionConfig(selection_format="invalid_format")

    def test_invalid_selection_threshold(self):
        """Test invalid selection threshold validation."""
        with pytest.raises(ConfigValidationError, match="selection_threshold must be in range"):
            ActionConfig(selection_threshold=-0.1)
        
        with pytest.raises(ConfigValidationError, match="selection_threshold must be in range"):
            ActionConfig(selection_threshold=1.1)

    def test_invalid_num_operations(self):
        """Test invalid num_operations validation."""
        with pytest.raises(ConfigValidationError, match="num_operations must be positive"):
            ActionConfig(num_operations=0)

    def test_invalid_allowed_operations(self):
        """Test invalid allowed_operations validation."""
        # Operations out of range
        with pytest.raises(ConfigValidationError, match="allowed_operations.*must be in range"):
            ActionConfig(num_operations=35, allowed_operations=[35])  # Out of range
        
        # Duplicate operations
        with pytest.raises(ConfigValidationError, match="contains duplicate operations"):
            ActionConfig(allowed_operations=[0, 1, 0])

    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        with pytest.raises(ConfigValidationError, match="allow_partial_selection must be a boolean"):
            ActionConfig(allow_partial_selection="not_bool")
        
        with pytest.raises(ConfigValidationError, match="validate_actions must be a boolean"):
            ActionConfig(validate_actions="not_bool")


class TestArcEnvConfigValidation:
    """Test ArcEnvConfig validation."""

    def test_valid_arc_env_config(self):
        """Test valid ARC environment configuration."""
        config = ArcEnvConfig(
            max_episode_steps=100,
            auto_reset=True,
            log_operations=False,
            strict_validation=True,
            allow_invalid_actions=False,
        )
        assert config.max_episode_steps == 100
        assert config.auto_reset is True

    def test_invalid_episode_steps(self):
        """Test invalid episode steps validation."""
        with pytest.raises(ConfigValidationError, match="max_episode_steps must be positive"):
            ArcEnvConfig(max_episode_steps=0)

    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        with pytest.raises(ConfigValidationError, match="auto_reset must be a boolean"):
            ArcEnvConfig(auto_reset="not_bool")
        
        with pytest.raises(ConfigValidationError, match="strict_validation must be a boolean"):
            ArcEnvConfig(strict_validation="not_bool")

    def test_sub_config_type_validation(self):
        """Test sub-configuration type validation."""
        # Invalid reward config type
        with pytest.raises(ConfigValidationError, match="reward must be RewardConfig"):
            ArcEnvConfig(reward="not_reward_config")
        
        # Invalid grid config type
        with pytest.raises(ConfigValidationError, match="grid must be GridConfig"):
            ArcEnvConfig(grid="not_grid_config")

    def test_cross_config_validation(self):
        """Test cross-configuration validation."""
        # This should not raise an error but may issue warnings
        config = ArcEnvConfig(
            strict_validation=True,
            allow_invalid_actions=True,  # Conflicting settings
        )
        assert config.strict_validation is True
        assert config.allow_invalid_actions is True

    def test_from_hydra_creation(self):
        """Test ArcEnvConfig creation from Hydra config."""
        hydra_cfg = OmegaConf.create({
            "max_episode_steps": 150,
            "auto_reset": False,
            "reward": {
                "success_bonus": 15.0,
                "step_penalty": -0.02,
            },
            "grid": {
                "max_grid_height": 25,
                "max_colors": 12,
            },
            "action": {
                "selection_format": "point",
                "num_operations": 30,
            },
        })
        
        config = ArcEnvConfig.from_hydra(hydra_cfg)
        assert config.max_episode_steps == 150
        assert config.auto_reset is False
        assert config.reward.success_bonus == 15.0
        assert config.grid.max_grid_height == 25
        assert config.action.selection_format == "point"


class TestConfigValidationFunction:
    """Test the validate_config function."""

    def test_valid_config_validation(self):
        """Test validation of a valid configuration."""
        config = ArcEnvConfig()  # Use defaults
        # Should not raise any errors
        validate_config(config)

    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        # The validation now happens during object creation, so we test that
        # invalid configurations are caught during construction
        with pytest.raises(ConfigValidationError, match="background_color.*must be.*max_colors"):
            GridConfig(max_colors=5, background_color=5)  # Invalid: background >= max_colors

    def test_parser_compatibility_validation(self):
        """Test parser compatibility validation."""
        # Mock parser for testing
        class MockParser:
            pass
        
        config = ArcEnvConfig(
            dataset=DatasetConfig(dataset_name="arc-agi-1"),
            parser=MockParser(),
        )
        
        # Should issue a warning but not raise an error
        validate_config(config)


class TestValidationErrorMessages:
    """Test that validation error messages are clear and helpful."""

    def test_clear_error_messages(self):
        """Test that error messages provide clear information."""
        # Test field name inclusion
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_positive_int(-1, "test_field")
        assert "test_field" in str(exc_info.value)
        assert "must be positive" in str(exc_info.value)
        
        # Test value inclusion in error message
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_float_range(2.0, "threshold", 0.0, 1.0)
        assert "threshold" in str(exc_info.value)
        assert "2.0" in str(exc_info.value)
        assert "[0.0, 1.0]" in str(exc_info.value)

    def test_nested_validation_errors(self):
        """Test that nested validation errors provide context."""
        with pytest.raises(ConfigValidationError) as exc_info:
            RewardConfig(step_penalty="not_number")
        
        error_msg = str(exc_info.value)
        assert "RewardConfig validation failed" in error_msg
        assert "step_penalty must be a number" in error_msg

    def test_operation_list_error_details(self):
        """Test that operation list errors provide specific details."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_operation_list([0, 1, 0], "operations", 35)
        
        error_msg = str(exc_info.value)
        assert "operations" in error_msg
        assert "duplicate operations" in error_msg
        assert "[0]" in error_msg  # Should show which operations are duplicated


if __name__ == "__main__":
    pytest.main([__file__])