"""
Tests for DatasetConfig dataset-specific configurations.

This module tests dataset configuration including parameter validation,
dataset selection, configuration options, and dataset path/loading parameters.
"""

from __future__ import annotations

import jax
import pytest
from omegaconf import DictConfig

from jaxarc.configs.dataset_config import DatasetConfig


class TestDatasetConfigCreation:
    """Test DatasetConfig creation and initialization."""

    def test_default_initialization(self):
        """Test DatasetConfig creation with default values."""
        config = DatasetConfig()

        # Verify default values
        assert config.dataset_name == "arc-agi-1"
        assert config.dataset_path == ""
        assert config.dataset_repo == ""
        assert config.parser_entry_point == "jaxarc.parsers:ArcAgiParser"
        assert config.expected_subdirs == ("data",)
        assert config.max_grid_height == 30
        assert config.max_grid_width == 30
        assert config.min_grid_height == 3
        assert config.min_grid_width == 3
        assert config.max_colors == 10
        assert config.background_color == -1
        assert config.max_train_pairs == 10
        assert config.max_test_pairs == 3
        assert config.task_split == "train"
        assert config.shuffle_tasks is True

    def test_custom_initialization(self):
        """Test DatasetConfig with custom parameters."""
        config = DatasetConfig(
            dataset_name="custom-dataset",
            dataset_path="/path/to/dataset",
            dataset_repo="https://github.com/user/dataset",
            parser_entry_point="custom.parser:CustomParser",
            expected_subdirs=("training", "evaluation"),
            max_grid_height=50,
            max_grid_width=40,
            min_grid_height=5,
            min_grid_width=4,
            max_colors=15,
            background_color=0,
            max_train_pairs=20,
            max_test_pairs=5,
            task_split="eval",
            shuffle_tasks=False,
        )

        assert config.dataset_name == "custom-dataset"
        assert config.dataset_path == "/path/to/dataset"
        assert config.dataset_repo == "https://github.com/user/dataset"
        assert config.parser_entry_point == "custom.parser:CustomParser"
        assert config.expected_subdirs == ("training", "evaluation")
        assert config.max_grid_height == 50
        assert config.max_grid_width == 40
        assert config.min_grid_height == 5
        assert config.min_grid_width == 4
        assert config.max_colors == 15
        assert config.background_color == 0
        assert config.max_train_pairs == 20
        assert config.max_test_pairs == 5
        assert config.task_split == "eval"
        assert config.shuffle_tasks is False


class TestDatasetConfigEquinoxCompliance:
    """Test Equinox Module compliance and JAX compatibility."""

    def test_hashability(self):
        """Test that DatasetConfig is hashable for JAX compatibility."""
        config = DatasetConfig()

        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_check_init_validation(self):
        """Test that __check_init__ validates hashability."""
        config = DatasetConfig()
        config.__check_init__()  # Should not raise

    def test_jax_tree_compatibility(self):
        """Test that DatasetConfig works as a JAX PyTree."""
        config = DatasetConfig(max_grid_height=42)

        def dummy_func(cfg):
            return cfg.max_grid_height

        # Should work with JAX transformations
        result = jax.jit(dummy_func)(config)
        assert result == 42

    def test_immutability(self):
        """Test that DatasetConfig is immutable."""
        config = DatasetConfig()

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.max_grid_height = 999


class TestDatasetConfigValidation:
    """Test DatasetConfig validation functionality."""

    def test_validate_returns_tuple(self):
        """Test that validate() returns a tuple of error strings."""
        config = DatasetConfig()
        errors = config.validate()

        assert isinstance(errors, tuple)
        for error in errors:
            assert isinstance(error, str)

    def test_valid_config_has_no_errors(self):
        """Test that valid configuration returns no errors."""
        config = DatasetConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_empty_dataset_name_validation(self):
        """Test validation of empty dataset name."""
        config = DatasetConfig(dataset_name="")
        errors = config.validate()
        assert len(errors) > 0
        assert any("dataset_name" in error and "empty" in error for error in errors)

        # Test whitespace-only name
        config = DatasetConfig(dataset_name="   ")
        errors = config.validate()
        assert len(errors) > 0
        assert any("dataset_name" in error and "empty" in error for error in errors)

    def test_invalid_grid_dimensions(self):
        """Test validation of invalid grid dimensions."""
        # Negative max_grid_height
        config = DatasetConfig(max_grid_height=-5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_grid_height" in error for error in errors)

        # Zero max_grid_width
        config = DatasetConfig(max_grid_width=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_grid_width" in error for error in errors)

        # Negative min dimensions
        config = DatasetConfig(min_grid_height=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("min_grid_height" in error for error in errors)

    def test_invalid_task_pair_counts(self):
        """Test validation of invalid task pair counts."""
        # Negative max_train_pairs
        config = DatasetConfig(max_train_pairs=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_train_pairs" in error for error in errors)

        # Zero max_test_pairs
        config = DatasetConfig(max_test_pairs=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_test_pairs" in error for error in errors)

    def test_invalid_color_constraints(self):
        """Test validation of invalid color constraints."""
        # Negative max_colors
        config = DatasetConfig(max_colors=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_colors" in error for error in errors)

        # Too few colors
        config = DatasetConfig(max_colors=1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_colors" in error and "at least 2" in error for error in errors)

        # Invalid background_color (too negative)
        config = DatasetConfig(background_color=-5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("background_color" in error for error in errors)

    def test_invalid_task_split(self):
        """Test validation of invalid task split."""
        config = DatasetConfig(task_split="invalid_split")
        errors = config.validate()
        assert len(errors) > 0
        assert any("task_split" in error for error in errors)

    def test_valid_task_splits(self):
        """Test that all valid task splits are accepted."""
        valid_splits = [
            "train",
            "eval",
            "test",
            "all",
            "training",
            "evaluation",
            "corpus",
        ]

        for split in valid_splits:
            config = DatasetConfig(task_split=split)
            errors = config.validate()
            # Should not have errors related to task_split
            split_errors = [e for e in errors if "task_split" in e]
            assert len(split_errors) == 0


class TestDatasetConfigCrossFieldValidation:
    """Test cross-field validation in DatasetConfig."""

    def test_grid_dimension_consistency(self):
        """Test validation of grid dimension consistency."""
        # max_grid_height < min_grid_height
        config = DatasetConfig(max_grid_height=5, min_grid_height=10)
        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "max_grid_height" in error and "min_grid_height" in error
            for error in errors
        )

        # max_grid_width < min_grid_width
        config = DatasetConfig(max_grid_width=8, min_grid_width=12)
        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "max_grid_width" in error and "min_grid_width" in error for error in errors
        )

    def test_background_color_consistency(self):
        """Test validation of background_color against max_colors."""
        # background_color >= max_colors (when background_color >= 0)
        config = DatasetConfig(background_color=10, max_colors=10)
        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "background_color" in error and "max_colors" in error for error in errors
        )

        # Valid: background_color = -1 (padding)
        config = DatasetConfig(background_color=-1, max_colors=10)
        errors = config.validate()
        background_errors = [e for e in errors if "background_color" in e]
        assert len(background_errors) == 0

        # Valid: background_color < max_colors
        config = DatasetConfig(background_color=5, max_colors=10)
        errors = config.validate()
        background_errors = [e for e in errors if "background_color" in e]
        assert len(background_errors) == 0

    def test_valid_cross_field_configuration(self):
        """Test that valid cross-field configurations pass validation."""
        config = DatasetConfig(
            max_grid_height=20,
            min_grid_height=5,
            max_grid_width=25,
            min_grid_width=3,
            max_colors=12,
            background_color=0,
        )

        errors = config.validate()
        assert len(errors) == 0


class TestDatasetConfigPathValidation:
    """Test dataset path and loading parameter validation."""

    def test_dataset_path_validation(self):
        """Test validation of dataset path strings."""
        # Valid empty path
        config = DatasetConfig(dataset_path="")
        errors = config.validate()
        path_errors = [e for e in errors if "dataset_path" in e]
        assert len(path_errors) == 0

        # Valid relative path
        config = DatasetConfig(dataset_path="data/arc-agi")
        errors = config.validate()
        path_errors = [e for e in errors if "dataset_path" in e]
        assert len(path_errors) == 0

        # Valid absolute path
        config = DatasetConfig(dataset_path="/home/user/datasets/arc")
        errors = config.validate()
        path_errors = [e for e in errors if "dataset_path" in e]
        assert len(path_errors) == 0

    def test_invalid_path_characters(self):
        """Test validation of paths with invalid characters."""
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]

        for char in invalid_chars:
            config = DatasetConfig(dataset_path=f"invalid{char}path")
            errors = config.validate()
            assert len(errors) > 0
            assert any(
                "dataset_path" in error and "invalid" in error for error in errors
            )

    def test_parser_entry_point_configuration(self):
        """Test parser entry point configuration."""
        config = DatasetConfig(parser_entry_point="custom.module:CustomParser")

        # Should not cause validation errors
        errors = config.validate()
        parser_errors = [e for e in errors if "parser_entry_point" in e]
        assert len(parser_errors) == 0

    def test_expected_subdirs_configuration(self):
        """Test expected subdirs configuration."""
        config = DatasetConfig(expected_subdirs=("training", "evaluation", "test"))

        # Should not cause validation errors
        errors = config.validate()
        subdir_errors = [e for e in errors if "expected_subdirs" in e]
        assert len(subdir_errors) == 0


class TestDatasetConfigHydraIntegration:
    """Test Hydra integration and configuration loading."""

    def test_from_hydra_empty_config(self):
        """Test creating DatasetConfig from empty Hydra config."""
        hydra_config = DictConfig({})
        config = DatasetConfig.from_hydra(hydra_config)

        # Should use defaults
        assert config.dataset_name == "arc-agi-1"
        assert config.max_grid_height == 30
        assert config.max_colors == 10

    def test_from_hydra_with_values(self):
        """Test creating DatasetConfig from Hydra config with values."""
        hydra_config = DictConfig(
            {
                "dataset_name": "concept-arc",
                "dataset_path": "/data/concept-arc",
                "dataset_repo": "https://github.com/conceptarc/data",
                "parser_entry_point": "jaxarc.parsers:ConceptArcParser",
                "expected_subdirs": ["corpus", "minimal"],
                "max_grid_height": 40,
                "max_grid_width": 35,
                "min_grid_height": 2,
                "min_grid_width": 2,
                "max_colors": 12,
                "background_color": -1,
                "max_train_pairs": 15,
                "max_test_pairs": 4,
                "task_split": "corpus",
                "shuffle_tasks": False,
            }
        )

        config = DatasetConfig.from_hydra(hydra_config)

        assert config.dataset_name == "concept-arc"
        assert config.dataset_path == "/data/concept-arc"
        assert config.dataset_repo == "https://github.com/conceptarc/data"
        assert config.parser_entry_point == "jaxarc.parsers:ConceptArcParser"
        assert config.expected_subdirs == ("corpus", "minimal")
        assert config.max_grid_height == 40
        assert config.max_grid_width == 35
        assert config.min_grid_height == 2
        assert config.min_grid_width == 2
        assert config.max_colors == 12
        assert config.background_color == -1
        assert config.max_train_pairs == 15
        assert config.max_test_pairs == 4
        assert config.task_split == "corpus"
        assert config.shuffle_tasks is False

    def test_from_hydra_expected_subdirs_conversion(self):
        """Test that Hydra config properly converts expected_subdirs to tuple."""
        hydra_config = DictConfig(
            {"expected_subdirs": ["data", "training", "evaluation"]}
        )

        config = DatasetConfig.from_hydra(hydra_config)

        assert isinstance(config.expected_subdirs, tuple)
        assert config.expected_subdirs == ("data", "training", "evaluation")


class TestDatasetConfigDatasetSelection:
    """Test dataset selection and configuration options."""

    def test_arc_agi_dataset_configuration(self):
        """Test configuration for ARC-AGI datasets."""
        config = DatasetConfig(
            dataset_name="arc-agi-1",
            parser_entry_point="jaxarc.parsers:ArcAgiParser",
            expected_subdirs=("data",),
            task_split="train",
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_concept_arc_dataset_configuration(self):
        """Test configuration for ConceptARC dataset."""
        config = DatasetConfig(
            dataset_name="concept-arc",
            parser_entry_point="jaxarc.parsers:ConceptArcParser",
            expected_subdirs=("corpus", "MinimalTasks"),
            task_split="corpus",
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_mini_arc_dataset_configuration(self):
        """Test configuration for Mini-ARC dataset."""
        config = DatasetConfig(
            dataset_name="mini-arc",
            parser_entry_point="jaxarc.parsers:MiniArcParser",
            expected_subdirs=("data",),
            task_split="test",
            max_grid_height=10,
            max_grid_width=10,
            max_train_pairs=3,
            max_test_pairs=1,
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_custom_dataset_configuration(self):
        """Test configuration for custom datasets."""
        config = DatasetConfig(
            dataset_name="custom-reasoning-tasks",
            dataset_path="/custom/path",
            parser_entry_point="custom.parsers:CustomParser",
            expected_subdirs=("tasks", "solutions"),
            max_grid_height=100,
            max_grid_width=100,
            max_colors=20,
            background_color=19,
            max_train_pairs=50,
            max_test_pairs=10,
        )

        errors = config.validate()
        assert len(errors) == 0
