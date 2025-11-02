"""
Tests for remaining configuration modules.

This module tests GridInitializationConfig, LoggingConfig, StorageConfig,
VisualizationConfig, and WandbConfig for Equinox compliance and validation.
"""

from __future__ import annotations

import jax
from omegaconf import DictConfig

from jaxarc.configs.grid_initialization_config import GridInitializationConfig
from jaxarc.configs.logging_config import LoggingConfig
from jaxarc.configs.storage_config import StorageConfig
from jaxarc.configs.visualization_config import VisualizationConfig
from jaxarc.configs.wandb_config import WandbConfig


class TestGridInitializationConfig:
    """Test GridInitializationConfig functionality."""

    def test_default_initialization(self):
        """Test GridInitializationConfig creation with default values."""
        config = GridInitializationConfig()

        # Verify default values
        assert config.demo_weight == 0.4
        assert config.permutation_weight == 0.3
        assert config.empty_weight == 0.2
        assert config.random_weight == 0.1
        assert config.permutation_types == ("rotate", "reflect", "color_remap")
        assert config.random_density == 0.3
        assert config.random_pattern_type == "sparse"

    def test_custom_initialization(self):
        """Test GridInitializationConfig with custom parameters."""
        config = GridInitializationConfig(
            demo_weight=0.5,
            permutation_weight=0.2,
            empty_weight=0.2,
            random_weight=0.1,
            permutation_types=("rotate", "reflect"),
            random_density=0.5,
            random_pattern_type="dense",
        )

        assert config.demo_weight == 0.5
        assert config.permutation_weight == 0.2
        assert config.empty_weight == 0.2
        assert config.random_weight == 0.1
        assert config.permutation_types == ("rotate", "reflect")
        assert config.random_density == 0.5
        assert config.random_pattern_type == "dense"

    def test_equinox_compliance(self):
        """Test Equinox Module compliance."""
        config = GridInitializationConfig()

        # Test hashability
        hash_value = hash(config)
        assert isinstance(hash_value, int)

        # Test JAX compatibility
        def dummy_func(cfg):
            return cfg.demo_weight

        result = jax.jit(dummy_func)(config)
        assert result == config.demo_weight

    def test_validation(self):
        """Test GridInitializationConfig validation."""
        # Valid config
        config = GridInitializationConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Invalid: all weights zero
        config = GridInitializationConfig(
            demo_weight=0.0, permutation_weight=0.0, empty_weight=0.0, random_weight=0.0
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("weight must be positive" in error for error in errors)

        # Invalid: negative weight
        config = GridInitializationConfig(demo_weight=-0.1)
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: random_density out of range
        config = GridInitializationConfig(random_density=1.5)
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: random_pattern_type
        config = GridInitializationConfig(random_pattern_type="invalid")
        errors = config.validate()
        assert len(errors) > 0

    def test_from_hydra(self):
        """Test Hydra integration."""
        hydra_config = DictConfig(
            {
                "demo_weight": 0.6,
                "permutation_types": ["rotate", "color_remap"],
                "random_density": 0.4,
            }
        )

        config = GridInitializationConfig.from_hydra(hydra_config)
        assert config.demo_weight == 0.6
        assert config.permutation_types == ("rotate", "color_remap")
        assert config.random_density == 0.4


class TestLoggingConfig:
    """Test LoggingConfig functionality."""

    def test_default_initialization(self):
        """Test LoggingConfig creation with default values."""
        config = LoggingConfig()

        # Verify default values
        assert config.log_operations is False
        assert config.log_rewards is False
        assert config.log_frequency == 10
        assert config.batched_logging_enabled is False
        assert config.log_format == "text"
        assert config.log_level == "INFO"
        assert config.structured_logging is False

    def test_custom_initialization(self):
        """Test LoggingConfig with custom parameters."""
        config = LoggingConfig(
            log_operations=True,
            log_rewards=True,
            log_frequency=5,
            batched_logging_enabled=True,
            log_format="json",
            log_level="DEBUG",
            structured_logging=True,
        )

        assert config.log_operations is True
        assert config.log_rewards is True
        assert config.log_frequency == 5
        assert config.batched_logging_enabled is True
        assert config.log_format == "json"
        assert config.log_level == "DEBUG"
        assert config.structured_logging is True

    def test_equinox_compliance(self):
        """Test Equinox Module compliance."""
        config = LoggingConfig()

        # Test hashability
        hash_value = hash(config)
        assert isinstance(hash_value, int)

        # Test JAX compatibility
        def dummy_func(cfg):
            return cfg.log_frequency

        result = jax.jit(dummy_func)(config)
        assert result == config.log_frequency

    def test_validation(self):
        """Test LoggingConfig validation."""
        # Valid config
        config = LoggingConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Invalid: log_format
        config = LoggingConfig(log_format="invalid")
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: log_level
        config = LoggingConfig(log_level="INVALID")
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: log_frequency
        config = LoggingConfig(log_frequency=0)
        errors = config.validate()
        assert len(errors) > 0

    def test_from_hydra(self):
        """Test Hydra integration."""
        hydra_config = DictConfig(
            {"log_operations": True, "log_format": "structured", "log_level": "WARNING"}
        )

        config = LoggingConfig.from_hydra(hydra_config)
        assert config.log_operations is True
        assert config.log_format == "structured"
        assert config.log_level == "WARNING"


class TestStorageConfig:
    """Test StorageConfig functionality."""

    def test_default_initialization(self):
        """Test StorageConfig creation with default values."""
        config = StorageConfig()

        # Verify default values
        assert config.base_output_dir == "outputs"
        assert config.run_name is None
        assert config.episodes_dir == "episodes"
        assert config.debug_dir == "debug"
        assert config.visualization_dir == "visualizations"
        assert config.logs_dir == "logs"
        assert config.max_episodes_per_run == 100
        assert config.max_storage_gb == 5.0
        assert config.cleanup_policy == "size_based"
        assert config.create_run_subdirs is True
        assert config.clear_output_on_start is True

    def test_custom_initialization(self):
        """Test StorageConfig with custom parameters."""
        config = StorageConfig(
            base_output_dir="/custom/output",
            run_name="test_run",
            max_episodes_per_run=500,
            max_storage_gb=10.0,
            cleanup_policy="oldest_first",
            create_run_subdirs=False,
            clear_output_on_start=False,
        )

        assert config.base_output_dir == "/custom/output"
        assert config.run_name == "test_run"
        assert config.max_episodes_per_run == 500
        assert config.max_storage_gb == 10.0
        assert config.cleanup_policy == "oldest_first"
        assert config.create_run_subdirs is False
        assert config.clear_output_on_start is False

    def test_equinox_compliance(self):
        """Test Equinox Module compliance."""
        config = StorageConfig()

        # Test hashability
        hash_value = hash(config)
        assert isinstance(hash_value, int)

        # Test JAX compatibility
        def dummy_func(cfg):
            return cfg.max_episodes_per_run

        result = jax.jit(dummy_func)(config)
        assert result == config.max_episodes_per_run

    def test_validation(self):
        """Test StorageConfig validation."""
        # Valid config
        config = StorageConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Invalid: cleanup_policy
        config = StorageConfig(cleanup_policy="invalid")
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: max_episodes_per_run
        config = StorageConfig(max_episodes_per_run=0)
        errors = config.validate()
        assert len(errors) > 0

        # Invalid: max_storage_gb
        config = StorageConfig(max_storage_gb=-1.0)
        errors = config.validate()
        assert len(errors) > 0

    def test_from_hydra(self):
        """Test Hydra integration."""
        hydra_config = DictConfig(
            {
                "base_output_dir": "/tmp/outputs",
                "cleanup_policy": "manual",
                "max_storage_gb": 20.0,
            }
        )

        config = StorageConfig.from_hydra(hydra_config)
        assert config.base_output_dir == "/tmp/outputs"
        assert config.cleanup_policy == "manual"
        assert config.max_storage_gb == 20.0


class TestVisualizationConfig:
    """Test VisualizationConfig functionality."""

    def test_default_initialization(self):
        """Test VisualizationConfig creation with default values."""
        config = VisualizationConfig()

        # Verify default values
        assert config.enabled is True
        assert config.episode_summaries is True
        assert config.step_visualizations is True

    def test_custom_initialization(self):
        """Test VisualizationConfig with custom parameters."""
        config = VisualizationConfig(
            enabled=False, episode_summaries=False, step_visualizations=True
        )

        assert config.enabled is False
        assert config.episode_summaries is False
        assert config.step_visualizations is True

    def test_equinox_compliance(self):
        """Test Equinox Module compliance."""
        config = VisualizationConfig()

        # Test hashability
        hash_value = hash(config)
        assert isinstance(hash_value, int)

        # Test JAX compatibility
        def dummy_func(cfg):
            return cfg.enabled

        result = jax.jit(dummy_func)(config)
        assert result == config.enabled

    def test_validation(self):
        """Test VisualizationConfig validation."""
        # Valid config
        config = VisualizationConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Test with all boolean combinations
        for enabled in [True, False]:
            for summaries in [True, False]:
                for step_viz in [True, False]:
                    config = VisualizationConfig(
                        enabled=enabled,
                        episode_summaries=summaries,
                        step_visualizations=step_viz,
                    )
                    errors = config.validate()
                    assert len(errors) == 0

    def test_from_hydra(self):
        """Test Hydra integration."""
        hydra_config = DictConfig(
            {"enabled": False, "episode_summaries": True, "step_visualizations": False}
        )

        config = VisualizationConfig.from_hydra(hydra_config)
        assert config.enabled is False
        assert config.episode_summaries is True
        assert config.step_visualizations is False


class TestWandbConfig:
    """Test WandbConfig functionality."""

    def test_default_initialization(self):
        """Test WandbConfig creation with default values."""
        config = WandbConfig()

        # Verify default values
        assert config.enabled is False
        assert config.project_name == "jaxarc-experiments"
        assert config.entity is None
        assert config.tags == ("jaxarc",)
        assert config.notes == "JaxARC experiment"
        assert config.group is None
        assert config.job_type == "training"
        assert config.offline_mode is False
        assert config.save_code is True

    def test_custom_initialization(self):
        """Test WandbConfig with custom parameters."""
        config = WandbConfig(
            enabled=True,
            project_name="custom-project",
            entity="my-team",
            tags=("experiment", "arc", "rl"),
            notes="Custom experiment",
            group="test-group",
            job_type="evaluation",
            offline_mode=True,
            save_code=False,
        )

        assert config.enabled is True
        assert config.project_name == "custom-project"
        assert config.entity == "my-team"
        assert config.tags == ("experiment", "arc", "rl")
        assert config.notes == "Custom experiment"
        assert config.group == "test-group"
        assert config.job_type == "evaluation"
        assert config.offline_mode is True
        assert config.save_code is False

    def test_tags_conversion(self):
        """Test tags conversion to tuple."""
        # Test with list
        config = WandbConfig(tags=["tag1", "tag2", "tag3"])
        assert isinstance(config.tags, tuple)
        assert config.tags == ("tag1", "tag2", "tag3")

        # Test with string
        config = WandbConfig(tags="single-tag")
        assert config.tags == ("single-tag",)

        # Test with empty list - should convert to empty tuple, not default
        config = WandbConfig(tags=[])
        assert config.tags == ()  # Empty list converts to empty tuple

    def test_equinox_compliance(self):
        """Test Equinox Module compliance."""
        config = WandbConfig()

        # Test hashability
        hash_value = hash(config)
        assert isinstance(hash_value, int)

        # Test JAX compatibility
        def dummy_func(cfg):
            return cfg.enabled

        result = jax.jit(dummy_func)(config)
        assert result == config.enabled

    def test_validation(self):
        """Test WandbConfig validation."""
        # Valid config
        config = WandbConfig()
        errors = config.validate()
        assert len(errors) == 0

        # Invalid: empty project_name
        config = WandbConfig(project_name="")
        errors = config.validate()
        assert len(errors) > 0
        assert any("project_name" in error for error in errors)

        # Invalid: whitespace-only project_name
        config = WandbConfig(project_name="   ")
        errors = config.validate()
        assert len(errors) > 0

    def test_from_hydra(self):
        """Test Hydra integration."""
        hydra_config = DictConfig(
            {
                "enabled": True,
                "project_name": "hydra-test",
                "tags": ["hydra", "test"],
                "entity": "test-entity",
            }
        )

        config = WandbConfig.from_hydra(hydra_config)
        assert config.enabled is True
        assert config.project_name == "hydra-test"
        assert config.tags == ("hydra", "test")
        assert config.entity == "test-entity"

    def test_from_hydra_tags_conversion(self):
        """Test Hydra tags conversion."""
        # Test with list
        hydra_config = DictConfig({"tags": ["a", "b", "c"]})
        config = WandbConfig.from_hydra(hydra_config)
        assert config.tags == ("a", "b", "c")

        # Test with string
        hydra_config = DictConfig({"tags": "single"})
        config = WandbConfig.from_hydra(hydra_config)
        assert config.tags == ("single",)
