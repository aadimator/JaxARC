"""
Tests for configuration factory functions.

This module tests the ConfigFactory class and its methods for creating
different configuration presets and handling overrides.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from jaxarc.envs.config_factory import ConfigFactory
from jaxarc.envs.config import (
    ConfigValidationError,
    JaxArcConfig,
)


class TestConfigFactory:
    """Test ConfigFactory class and its methods."""

    def test_create_development_config(self):
        """Test development configuration creation."""
        config = ConfigFactory.create_development_config()

        # Test that it's a valid JaxArcConfig
        assert isinstance(config, JaxArcConfig)

        # Test development-specific values
        assert config.environment.max_episode_steps == 50
        assert config.environment.debug_level == "standard"
        assert config.dataset.max_train_pairs == 5
        assert config.dataset.max_test_pairs == 2
        assert config.visualization.enabled is True
        assert config.visualization.level == "standard"
        assert config.storage.policy == "standard"
        assert config.storage.base_output_dir == "outputs/dev"
        assert config.logging.log_level == "INFO"
        assert config.wandb.enabled is False
        assert config.wandb.project_name == "jaxarc-dev"

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_create_research_config(self):
        """Test research configuration creation."""
        config = ConfigFactory.create_research_config()

        # Test that it's a valid JaxArcConfig
        assert isinstance(config, JaxArcConfig)

        # Test research-specific values
        assert config.environment.max_episode_steps == 200
        assert config.environment.debug_level == "research"
        assert config.dataset.max_train_pairs == 10
        assert config.dataset.max_test_pairs == 3
        assert config.reward.reward_on_submit_only is False
        assert config.reward.progress_bonus == 0.2
        assert config.visualization.enabled is True
        assert config.visualization.level == "full"
        assert config.visualization.save_intermediate_states is True
        assert config.storage.policy == "research"
        assert config.storage.base_output_dir == "outputs/research"
        assert config.storage.max_episodes_per_run == 500
        assert config.storage.max_storage_gb == 20.0
        assert config.logging.log_level == "DEBUG"
        assert config.logging.include_full_states is True
        assert config.logging.log_operations is True
        assert config.wandb.enabled is True
        assert config.wandb.project_name == "jaxarc-research"

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_create_production_config(self):
        """Test production configuration creation."""
        config = ConfigFactory.create_production_config()

        # Test that it's a valid JaxArcConfig
        assert isinstance(config, JaxArcConfig)

        # Test production-specific values
        assert config.environment.max_episode_steps == 100
        assert config.environment.debug_level == "off"
        assert config.dataset.task_split == "eval"
        assert config.dataset.shuffle_tasks is False
        assert config.reward.step_penalty == 0.0
        assert config.reward.progress_bonus == 0.0
        assert config.visualization.enabled is False
        assert config.visualization.level == "off"
        assert config.storage.policy == "minimal"
        assert config.storage.base_output_dir == "outputs/prod"
        assert config.logging.log_level == "ERROR"
        assert config.logging.log_operations is False
        assert config.wandb.enabled is False
        assert config.wandb.offline_mode is True

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_config_factory_overrides(self):
        """Test configuration overrides in factory methods."""
        # Test development config with overrides
        dev_config = ConfigFactory.create_development_config(
            max_episode_steps=75,
            dataset_name="custom-dataset",
            visualization_enabled=False,
            log_level="DEBUG",
        )

        # Test that overrides are applied
        assert dev_config.environment.max_episode_steps == 75
        assert dev_config.dataset.dataset_name == "custom-dataset"
        assert dev_config.visualization.enabled is False
        assert dev_config.logging.log_level == "DEBUG"

        # Test research config with overrides
        research_config = ConfigFactory.create_research_config(
            max_episode_steps=300,
            wandb_enabled=False,
            max_storage_gb=10.0,
        )

        # Test that overrides are applied
        assert research_config.environment.max_episode_steps == 300
        assert research_config.wandb.enabled is False
        assert research_config.storage.max_storage_gb == 10.0

        # Test production config with overrides
        prod_config = ConfigFactory.create_production_config(
            max_episode_steps=200,
            dataset_name="arc-agi-2",
            log_level="INFO",
        )

        # Test that overrides are applied
        assert prod_config.environment.max_episode_steps == 200
        assert prod_config.dataset.dataset_name == "arc-agi-2"
        assert prod_config.logging.log_level == "INFO"

    def test_nested_overrides(self):
        """Test nested configuration overrides in factory methods."""
        # Test nested overrides
        config = ConfigFactory.create_development_config(
            environment={"max_episode_steps": 80, "debug_level": "verbose"},
            dataset={"dataset_name": "mini-arc", "max_grid_height": 5},
            reward={"success_bonus": 15.0, "step_penalty": -0.02},
            visualization={"level": "minimal", "output_formats": ["png"]},
        )

        # Test that nested overrides are applied
        assert config.environment.max_episode_steps == 80
        assert config.environment.debug_level == "verbose"
        assert config.dataset.dataset_name == "mini-arc"
        assert config.dataset.max_grid_height == 5
        assert config.reward.success_bonus == 15.0
        assert config.reward.step_penalty == -0.02
        assert config.visualization.level == "minimal"
        assert config.visualization.output_formats == ["png"]

    def test_from_hydra(self):
        """Test configuration creation from Hydra DictConfig."""
        # Create a Hydra-style config
        hydra_config = OmegaConf.create(
            {
                "environment": {
                    "max_episode_steps": 150,
                    "debug_level": "verbose",
                },
                "dataset": {
                    "dataset_name": "arc-agi-2",
                    "max_grid_height": 25,
                },
                "action": {
                    "selection_format": "bbox",
                    "selection_threshold": 0.7,
                },
                "reward": {
                    "success_bonus": 15.0,
                    "step_penalty": -0.02,
                },
            }
        )

        # Convert to JaxArcConfig
        config = ConfigFactory.from_hydra(hydra_config)

        # Test that it's a valid JaxArcConfig
        assert isinstance(config, JaxArcConfig)

        # Test that values from Hydra config are applied
        assert config.environment.max_episode_steps == 150
        assert config.environment.debug_level == "verbose"
        assert config.dataset.dataset_name == "arc-agi-2"
        assert config.dataset.max_grid_height == 25
        assert config.action.selection_format == "bbox"
        assert config.action.selection_threshold == 0.7
        assert config.reward.success_bonus == 15.0
        assert config.reward.step_penalty == -0.02

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_errors(self):
        """Test validation errors in factory methods."""
        # Test with invalid parameters
        with pytest.raises(ConfigValidationError):
            ConfigFactory.create_development_config(max_episode_steps=-10)

        with pytest.raises(ConfigValidationError):
            ConfigFactory.create_research_config(
                dataset={"max_grid_height": 5, "min_grid_height": 10}
            )

        with pytest.raises(ConfigValidationError):
            ConfigFactory.create_production_config(
                action={"selection_format": "invalid_format"}
            )

    def test_preset_methods(self):
        """Test preset configuration methods."""
        # Test testing preset
        testing_config = ConfigFactory._create_testing_preset()
        assert testing_config.environment.max_episode_steps == 20
        assert testing_config.environment.debug_level == "minimal"
        assert testing_config.visualization.enabled is False
        assert testing_config.storage.max_storage_gb == 0.5

        # Test minimal preset
        minimal_config = ConfigFactory._create_minimal_preset()
        assert minimal_config.environment.max_episode_steps == 50
        assert minimal_config.environment.debug_level == "off"
        assert minimal_config.visualization.enabled is False
        assert minimal_config.logging.log_level == "ERROR"

        # Test debug preset
        debug_config = ConfigFactory._create_debug_preset()
        assert debug_config.environment.max_episode_steps == 30
        assert debug_config.environment.debug_level == "verbose"
        assert debug_config.logging.log_level == "DEBUG"
        assert debug_config.logging.log_operations is True
        assert debug_config.logging.log_grid_changes is True

    def test_curriculum_presets(self):
        """Test curriculum preset configuration methods."""
        # Test basic curriculum preset
        basic_config = ConfigFactory._create_curriculum_basic_preset()
        assert basic_config.environment.max_episode_steps == 30
        assert len(basic_config.action.allowed_operations) == 11  # Limited operations
        assert basic_config.reward.reward_on_submit_only is True
        assert basic_config.reward.success_bonus == 15.0

        # Test advanced curriculum preset
        advanced_config = ConfigFactory._create_curriculum_advanced_preset()
        assert advanced_config.environment.max_episode_steps == 150
        assert advanced_config.reward.reward_on_submit_only is False
        assert advanced_config.reward.progress_bonus > 0.0  # Just check it's positive

    def test_specialized_presets(self):
        """Test specialized preset configuration methods."""
        # Test evaluation preset
        eval_config = ConfigFactory._create_evaluation_preset()
        assert eval_config.dataset.task_split == "eval"
        assert eval_config.reward.reward_on_submit_only is True
        assert eval_config.reward.step_penalty == 0.0
        assert eval_config.reward.success_bonus == 1.0
        assert eval_config.dataset.shuffle_tasks is False

        # Test point actions preset
        point_config = ConfigFactory._create_point_actions_preset()
        assert point_config.action.selection_format == "point"
        # Note: allow_partial_selection might be ignored for point format
        assert point_config.environment.max_episode_steps == 80

        # Test bbox actions preset
        bbox_config = ConfigFactory._create_bbox_actions_preset()
        assert bbox_config.action.selection_format == "bbox"
        # Note: allow_partial_selection might be ignored for bbox format
        assert bbox_config.environment.max_episode_steps == 90

    def test_dataset_presets(self):
        """Test dataset-specific preset configuration methods."""
        # Test mini arc preset
        mini_config = ConfigFactory._create_mini_arc_preset()
        assert mini_config.dataset.dataset_name == "mini-arc"
        assert mini_config.dataset.max_grid_height == 5
        assert mini_config.dataset.max_grid_width == 5
        assert mini_config.environment.max_episode_steps == 40
        assert mini_config.action.selection_format == "point"

        # Test concept arc preset
        concept_config = ConfigFactory._create_concept_arc_preset()
        assert concept_config.dataset.dataset_name == "concept-arc"
        assert concept_config.dataset.max_grid_height == 15
        assert concept_config.dataset.max_grid_width == 15
        assert concept_config.dataset.task_split == "corpus"
        assert concept_config.environment.max_episode_steps == 120

    def test_specialized_feature_presets(self):
        """Test specialized feature preset configuration methods."""
        # Test wandb research preset
        wandb_config = ConfigFactory._create_wandb_research_preset()
        assert wandb_config.wandb.enabled is True
        assert wandb_config.wandb.project_name == "jaxarc-research"
        assert wandb_config.wandb.log_frequency == 5

        # Test memory efficient preset
        memory_config = ConfigFactory._create_memory_efficient_preset()
        # Just check that memory settings are reasonable
        assert memory_config.visualization.max_memory_mb < 1000
        assert memory_config.storage.max_storage_gb == 0.5
        assert memory_config.logging.queue_size == 50
        assert memory_config.logging.batch_size == 1

        # Test high performance preset
        perf_config = ConfigFactory._create_high_performance_preset()
        assert perf_config.environment.debug_level == "off"
        assert perf_config.visualization.enabled is False
        assert perf_config.logging.log_level == "ERROR"
        assert perf_config.logging.structured_logging is False
        assert perf_config.logging.compression is False


if __name__ == "__main__":
    pytest.main([__file__])
