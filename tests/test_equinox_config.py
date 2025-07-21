"""
Tests for Equinox-based configuration classes.

This module tests the new unified configuration system using Equinox and JaxTyping.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.jaxarc.envs.equinox_config import (
    ConfigValidationError,
    DatasetConfig,
    EnvironmentConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)


class TestEnvironmentConfig:
    """Test EnvironmentConfig class."""

    def test_default_creation(self):
        """Test creating EnvironmentConfig with defaults."""
        config = EnvironmentConfig()

        assert config.max_episode_steps == 100
        assert config.auto_reset is True
        assert config.strict_validation is True
        assert config.allow_invalid_actions is False
        assert config.debug_level == "standard"

    def test_custom_creation(self):
        """Test creating EnvironmentConfig with custom values."""
        config = EnvironmentConfig(
            max_episode_steps=200,
            auto_reset=False,
            strict_validation=False,
            allow_invalid_actions=True,
            debug_level="verbose",
        )

        assert config.max_episode_steps == 200
        assert config.auto_reset is False
        assert config.strict_validation is False
        assert config.allow_invalid_actions is True
        assert config.debug_level == "verbose"

    def test_validation_success(self):
        """Test successful validation."""
        config = EnvironmentConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_errors(self):
        """Test validation errors."""
        config = EnvironmentConfig(
            debug_level="invalid",  # Invalid debug level
        )

        errors = config.validate()
        assert len(errors) >= 1
        assert any("debug_level must be one of" in error for error in errors)

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "max_episode_steps": 150,
                "auto_reset": False,
                "debug_level": "verbose",
            }
        )

        config = EnvironmentConfig.from_hydra(hydra_cfg)

        assert config.max_episode_steps == 150
        assert config.auto_reset is False
        assert config.debug_level == "verbose"
        # Check defaults are preserved
        assert config.strict_validation is True
        assert config.allow_invalid_actions is False


class TestDatasetConfig:
    """Test DatasetConfig class."""

    def test_default_creation(self):
        """Test creating DatasetConfig with defaults."""
        config = DatasetConfig()

        assert config.dataset_name == "arc-agi-1"
        assert config.dataset_path == ""
        assert config.max_grid_height == 30
        assert config.max_grid_width == 30
        assert config.min_grid_height == 3
        assert config.min_grid_width == 3
        assert config.max_colors == 10
        assert config.background_color == 0
        assert config.task_split == "train"
        assert config.max_tasks is None
        assert config.shuffle_tasks is True

    def test_custom_creation(self):
        """Test creating DatasetConfig with custom values."""
        config = DatasetConfig(
            dataset_name="custom-dataset",
            max_grid_height=50,
            max_grid_width=40,
            min_grid_height=5,
            min_grid_width=4,
            max_colors=15,
            background_color=1,
            task_split="eval",
            max_tasks=100,
            shuffle_tasks=False,
        )

        assert config.dataset_name == "custom-dataset"
        assert config.max_grid_height == 50
        assert config.max_grid_width == 40
        assert config.min_grid_height == 5
        assert config.min_grid_width == 4
        assert config.max_colors == 15
        assert config.background_color == 1
        assert config.task_split == "eval"
        assert config.max_tasks == 100
        assert config.shuffle_tasks is False

    def test_validation_success(self):
        """Test successful validation."""
        config = DatasetConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_errors(self):
        """Test validation errors."""
        config = DatasetConfig(
            dataset_name="",  # Empty name
            max_colors=1,  # Too few colors
            background_color=10,  # >= max_colors
            max_grid_height=2,  # < min_grid_height
            max_grid_width=2,  # < min_grid_width
        )

        errors = config.validate()
        assert len(errors) >= 4
        assert any("dataset_name cannot be empty" in error for error in errors)
        assert any("max_colors must be at least 2" in error for error in errors)
        assert any(
            "background_color" in error and "max_colors" in error for error in errors
        )
        assert any(
            "max_grid_height" in error and "min_grid_height" in error
            for error in errors
        )
        assert any(
            "max_grid_width" in error and "min_grid_width" in error for error in errors
        )

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "dataset_name": "test-dataset",
                "max_grid_height": 25,
                "max_colors": 8,
                "task_split": "eval",
                "max_tasks": 50,
            }
        )

        config = DatasetConfig.from_hydra(hydra_cfg)

        assert config.dataset_name == "test-dataset"
        assert config.max_grid_height == 25
        assert config.max_colors == 8
        assert config.task_split == "eval"
        assert config.max_tasks == 50
        # Check defaults are preserved
        assert config.max_grid_width == 30
        assert config.min_grid_height == 3


class TestVisualizationConfig:
    """Test VisualizationConfig class."""

    def test_default_creation(self):
        """Test creating VisualizationConfig with defaults."""
        config = VisualizationConfig()

        assert config.enabled is True
        assert config.level == "standard"
        assert config.output_formats == ["svg"]
        assert config.image_quality == "high"
        assert config.show_coordinates is False
        assert config.show_operation_names is True
        assert config.highlight_changes is True
        assert config.include_metrics is True
        assert config.color_scheme == "default"
        assert config.visualize_episodes is True
        assert config.episode_summaries is True
        assert config.step_visualizations is True
        assert config.enable_comparisons is True
        assert config.save_intermediate_states is False
        assert config.lazy_loading is True
        assert config.memory_limit_mb == 500

    def test_validation_success(self):
        """Test successful validation."""
        config = VisualizationConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_invalid_formats(self):
        """Test validation with invalid output formats."""
        config = VisualizationConfig(output_formats=["invalid_format"])
        errors = config.validate()
        assert len(errors) >= 1
        assert any("Invalid output format" in error for error in errors)

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "enabled": False,
                "level": "verbose",
                "output_formats": ["svg", "png"],
                "image_quality": "medium",
                "show_coordinates": True,
                "memory_limit_mb": 1000,
            }
        )

        config = VisualizationConfig.from_hydra(hydra_cfg)

        assert config.enabled is False
        assert config.level == "verbose"
        assert config.output_formats == ["svg", "png"]
        assert config.image_quality == "medium"
        assert config.show_coordinates is True
        assert config.memory_limit_mb == 1000


class TestStorageConfig:
    """Test StorageConfig class."""

    def test_default_creation(self):
        """Test creating StorageConfig with defaults."""
        config = StorageConfig()

        assert config.policy == "standard"
        assert config.base_output_dir == "outputs"
        assert config.run_name is None
        assert config.episodes_dir == "episodes"
        assert config.debug_dir == "debug"
        assert config.visualization_dir == "visualizations"
        assert config.logs_dir == "logs"
        assert config.max_episodes_per_run == 100
        assert config.max_storage_gb == 5.0
        assert config.cleanup_policy == "size_based"
        assert config.cleanup_frequency == "after_run"
        assert config.keep_recent_episodes == 10
        assert config.create_run_subdirs is True
        assert config.timestamp_format == "%Y%m%d_%H%M%S"
        assert config.compress_old_files is True
        assert config.clear_output_on_start is True
        assert config.auto_cleanup is True
        assert config.warn_on_storage_limit is True
        assert config.fail_on_storage_full is False

    def test_validation_success(self):
        """Test successful validation."""
        config = StorageConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_invalid_policy(self):
        """Test validation with invalid policy."""
        config = StorageConfig(policy="invalid")
        errors = config.validate()
        assert len(errors) >= 1
        assert any("policy must be one of" in error for error in errors)

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "policy": "research",
                "base_output_dir": "custom",
                "episodes_dir": "custom_episodes",
                "debug_dir": "custom_debug",
                "max_episodes_per_run": 500,
                "max_storage_gb": 20.0,
                "cleanup_policy": "manual",
                "auto_cleanup": False,
            }
        )

        config = StorageConfig.from_hydra(hydra_cfg)

        assert config.policy == "research"
        assert config.base_output_dir == "custom"
        assert config.episodes_dir == "custom_episodes"
        assert config.debug_dir == "custom_debug"
        assert config.max_episodes_per_run == 500
        assert config.max_storage_gb == 20.0
        assert config.cleanup_policy == "manual"
        assert config.auto_cleanup is False


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_default_creation(self):
        """Test creating LoggingConfig with defaults."""
        config = LoggingConfig()

        assert config.structured_logging is True
        assert config.log_format == "json"
        assert config.log_level == "INFO"
        assert config.compression is True
        assert config.include_full_states is False
        assert config.log_operations is False
        assert config.log_grid_changes is False
        assert config.log_rewards is False
        assert config.log_episode_start is True
        assert config.log_episode_end is True
        assert config.log_key_moments is True
        assert config.log_frequency == 10
        assert config.queue_size == 500
        assert config.worker_threads == 1
        assert config.batch_size == 5
        assert config.flush_interval == 10.0
        assert config.enable_compression is True

    def test_validation_success(self):
        """Test successful validation."""
        config = LoggingConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_invalid_format(self):
        """Test validation with invalid log format."""
        config = LoggingConfig(log_format="invalid")
        errors = config.validate()
        assert len(errors) >= 1
        assert any("log_format must be one of" in error for error in errors)

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "structured_logging": False,
                "log_format": "text",
                "log_level": "DEBUG",
                "include_full_states": True,
                "log_operations": True,
                "log_rewards": True,
                "log_frequency": 5,
                "queue_size": 1000,
                "worker_threads": 2,
                "flush_interval": 5.0,
            }
        )

        config = LoggingConfig.from_hydra(hydra_cfg)

        assert config.structured_logging is False
        assert config.log_format == "text"
        assert config.log_level == "DEBUG"
        assert config.include_full_states is True
        assert config.log_operations is True
        assert config.log_rewards is True
        assert config.log_frequency == 5
        assert config.queue_size == 1000
        assert config.worker_threads == 2
        assert config.flush_interval == 5.0


class TestWandbConfig:
    """Test WandbConfig class."""

    def test_default_creation(self):
        """Test creating WandbConfig with defaults."""
        config = WandbConfig()

        assert config.enabled is False
        assert config.project_name == "jaxarc-experiments"
        assert config.entity is None
        assert config.tags == ["jaxarc"]
        assert config.notes == "JaxARC experiment"
        assert config.group is None
        assert config.job_type == "training"
        assert config.log_frequency == 10
        assert config.image_format == "png"
        assert config.max_image_size == (800, 600)
        assert config.log_gradients is False
        assert config.log_model_topology is False
        assert config.log_system_metrics is True
        assert config.offline_mode is False
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.save_code is True
        assert config.save_config is True

    def test_validation_success(self):
        """Test successful validation."""
        config = WandbConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_empty_project_name(self):
        """Test validation with empty project name."""
        config = WandbConfig(project_name="")
        errors = config.validate()
        assert len(errors) >= 1
        assert any("project_name cannot be empty" in error for error in errors)

    def test_validation_invalid_image_format(self):
        """Test validation with invalid image format."""
        config = WandbConfig(image_format="invalid")
        errors = config.validate()
        assert len(errors) >= 1
        assert any("image_format must be one of" in error for error in errors)

    def test_from_hydra(self):
        """Test creating from Hydra DictConfig."""
        hydra_cfg = OmegaConf.create(
            {
                "enabled": True,
                "project_name": "custom-project",
                "entity": "my-team",
                "tags": ["custom", "experiment"],
                "notes": "Custom experiment notes",
                "group": "experiment-group",
                "job_type": "research",
                "log_frequency": 5,
                "image_format": "svg",
                "max_image_size": [1024, 768],
                "log_gradients": True,
                "offline_mode": True,
                "retry_attempts": 5,
            }
        )

        config = WandbConfig.from_hydra(hydra_cfg)

        assert config.enabled is True
        assert config.project_name == "custom-project"
        assert config.entity == "my-team"
        assert config.tags == ["custom", "experiment"]
        assert config.notes == "Custom experiment notes"
        assert config.group == "experiment-group"
        assert config.job_type == "research"
        assert config.log_frequency == 5
        assert config.image_format == "svg"
        assert config.max_image_size == (1024, 768)
        assert config.log_gradients is True
        assert config.offline_mode is True
        assert config.retry_attempts == 5

    def test_from_hydra_string_tags(self):
        """Test creating from Hydra with string tags (should convert to list)."""
        hydra_cfg = OmegaConf.create({"tags": "single-tag"})

        config = WandbConfig.from_hydra(hydra_cfg)
        assert config.tags == ["single-tag"]


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error(self):
        """Test ConfigValidationError can be raised and caught."""
        with pytest.raises(ConfigValidationError) as exc_info:
            raise ConfigValidationError("Test error message")

        assert str(exc_info.value) == "Test error message"
        assert isinstance(exc_info.value, ValueError)


if __name__ == "__main__":
    pytest.main([__file__])
