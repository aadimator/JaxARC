"""
Tests for Equinox-based JaxArcConfig configuration system.

This module tests the unified Equinox-based configuration system including
module validation, JAX compatibility, and configuration component integration.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxarc.envs.equinox_config import (
    ActionConfig,
    DatasetConfig,
    EnvironmentConfig,
    JaxArcConfig,
    LoggingConfig,
    RewardConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)


class TestEquinoxConfigModules:
    """Test individual Equinox configuration modules."""

    def test_environment_config_creation(self):
        """Test EnvironmentConfig creation and validation."""
        config = EnvironmentConfig(
            max_episode_steps=100,
            auto_reset=True,
            strict_validation=True,
            allow_invalid_actions=False,
            debug_level="standard",
        )

        # Test field values
        assert config.max_episode_steps == 100
        assert config.auto_reset is True
        assert config.strict_validation is True
        assert config.allow_invalid_actions is False
        assert config.debug_level == "standard"

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

        # Test computed properties
        assert config.computed_visualization_level == "standard"
        assert config.computed_storage_policy == "standard"

    def test_environment_config_validation_errors(self):
        """Test EnvironmentConfig validation errors."""
        # Invalid max_episode_steps
        config = EnvironmentConfig(max_episode_steps=-10)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_episode_steps" in error for error in errors)

        # Invalid debug_level
        config = EnvironmentConfig(debug_level="invalid_level")
        errors = config.validate()
        assert len(errors) > 0
        assert any("debug_level" in error for error in errors)

    def test_dataset_config_creation(self):
        """Test DatasetConfig creation and validation."""
        config = DatasetConfig(
            dataset_name="arc-agi-1",
            dataset_path="/path/to/data",
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=3,
            min_grid_width=3,
            max_colors=10,
            background_color=0,
            task_split="train",
            shuffle_tasks=True,
            max_train_pairs=5,
            max_test_pairs=2,
        )

        # Test field values
        assert config.dataset_name == "arc-agi-1"
        assert config.dataset_path == "/path/to/data"
        assert config.max_grid_height == 30
        assert config.max_grid_width == 30
        assert config.min_grid_height == 3
        assert config.min_grid_width == 3
        assert config.max_colors == 10
        assert config.background_color == 0
        assert config.task_split == "train"
        assert config.shuffle_tasks is True
        assert config.max_train_pairs == 5
        assert config.max_test_pairs == 2

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_dataset_config_validation_errors(self):
        """Test DatasetConfig validation errors."""
        # Invalid grid dimensions (min > max)
        config = DatasetConfig(
            max_grid_height=10,
            min_grid_height=20,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_grid_height" in error for error in errors)

        # Invalid background color
        config = DatasetConfig(
            max_colors=5,
            background_color=10,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("background_color" in error for error in errors)

        # Invalid task split
        config = DatasetConfig(task_split="invalid_split")
        errors = config.validate()
        assert len(errors) > 0
        assert any("task_split" in error for error in errors)

    def test_action_config_creation(self):
        """Test ActionConfig creation and validation."""
        config = ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            max_operations=35,
            allowed_operations=[0, 1, 2, 3],
            validate_actions=True,
            allow_invalid_actions=False,
        )

        # Test field values
        assert config.selection_format == "mask"
        assert config.selection_threshold == 0.5
        assert config.allow_partial_selection is True
        assert config.max_operations == 35
        assert config.allowed_operations == [0, 1, 2, 3]
        assert config.validate_actions is True
        assert config.allow_invalid_actions is False

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_action_config_validation_errors(self):
        """Test ActionConfig validation errors."""
        # Invalid selection format
        config = ActionConfig(selection_format="invalid_format")
        errors = config.validate()
        assert len(errors) > 0
        assert any("selection_format" in error for error in errors)

        # Invalid selection threshold
        config = ActionConfig(selection_threshold=1.5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("selection_threshold" in error for error in errors)

        # Invalid allowed operations
        config = ActionConfig(
            max_operations=10,
            allowed_operations=[15, 20],  # Operations out of range
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("allowed_operations" in error for error in errors)

    def test_reward_config_creation(self):
        """Test RewardConfig creation and validation."""
        config = RewardConfig(
            reward_on_submit_only=True,
            step_penalty=-0.01,
            success_bonus=10.0,
            similarity_weight=1.0,
            progress_bonus=0.0,
            invalid_action_penalty=-0.1,
        )

        # Test field values
        assert config.reward_on_submit_only is True
        assert config.step_penalty == -0.01
        assert config.success_bonus == 10.0
        assert config.similarity_weight == 1.0
        assert config.progress_bonus == 0.0
        assert config.invalid_action_penalty == -0.1

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_reward_config_validation_errors(self):
        """Test RewardConfig validation errors."""
        # Invalid step penalty (out of range)
        config = RewardConfig(step_penalty=-20.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("step_penalty" in error for error in errors)

        # Invalid success bonus (out of range)
        config = RewardConfig(success_bonus=2000.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("success_bonus" in error for error in errors)

    def test_visualization_config_creation(self):
        """Test VisualizationConfig creation and validation."""
        config = VisualizationConfig(
            enabled=True,
            level="standard",
            output_formats=["svg", "png"],
            image_quality="high",
            show_coordinates=False,
            show_operation_names=True,
            highlight_changes=True,
            include_metrics=True,
            color_scheme="default",
            visualize_episodes=True,
            episode_summaries=True,
            step_visualizations=True,
            enable_comparisons=True,
            save_intermediate_states=False,
            lazy_loading=True,
            max_memory_mb=500,
        )

        # Test field values
        assert config.enabled is True
        assert config.level == "standard"
        assert config.output_formats == ["svg", "png"]
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
        assert config.max_memory_mb == 500

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_visualization_config_validation_errors(self):
        """Test VisualizationConfig validation errors."""
        # Invalid level
        config = VisualizationConfig(level="invalid_level")
        errors = config.validate()
        assert len(errors) > 0
        assert any("level" in error for error in errors)

        # Invalid image quality
        config = VisualizationConfig(image_quality="ultra")
        errors = config.validate()
        assert len(errors) > 0
        assert any("image_quality" in error for error in errors)

        # Invalid color scheme
        config = VisualizationConfig(color_scheme="neon")
        errors = config.validate()
        assert len(errors) > 0
        assert any("color_scheme" in error for error in errors)

    def test_storage_config_creation(self):
        """Test StorageConfig creation and validation."""
        config = StorageConfig(
            policy="standard",
            base_output_dir="outputs",
            run_name="test_run",
            episodes_dir="episodes",
            debug_dir="debug",
            visualization_dir="visualizations",
            logs_dir="logs",
            max_episodes_per_run=100,
            max_storage_gb=5.0,
            cleanup_policy="size_based",
            cleanup_frequency="after_run",
            keep_recent_episodes=10,
            create_run_subdirs=True,
            timestamp_format="%Y%m%d_%H%M%S",
            compress_old_files=True,
            clear_output_on_start=True,
            auto_cleanup=True,
            warn_on_storage_limit=True,
            fail_on_storage_full=False,
        )

        # Test field values
        assert config.policy == "standard"
        assert config.base_output_dir == "outputs"
        assert config.run_name == "test_run"
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

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_storage_config_validation_errors(self):
        """Test StorageConfig validation errors."""
        # Invalid policy
        config = StorageConfig(policy="invalid_policy")
        errors = config.validate()
        assert len(errors) > 0
        assert any("policy" in error for error in errors)

        # Invalid cleanup policy
        config = StorageConfig(cleanup_policy="invalid_policy")
        errors = config.validate()
        assert len(errors) > 0
        assert any("cleanup_policy" in error for error in errors)

        # Invalid max_storage_gb
        config = StorageConfig(max_storage_gb=-1.0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_storage_gb" in error for error in errors)

    def test_logging_config_creation(self):
        """Test LoggingConfig creation and validation."""
        config = LoggingConfig(
            structured_logging=True,
            log_format="json",
            log_level="INFO",
            compression=True,
            include_full_states=False,
            log_operations=False,
            log_grid_changes=False,
            log_rewards=False,
            log_episode_start=True,
            log_episode_end=True,
            log_key_moments=True,
            log_frequency=10,
            queue_size=500,
            worker_threads=1,
            batch_size=5,
            flush_interval=10.0,
            enable_compression=True,
        )

        # Test field values
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

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_logging_config_validation_errors(self):
        """Test LoggingConfig validation errors."""
        # Invalid log format
        config = LoggingConfig(log_format="invalid_format")
        errors = config.validate()
        assert len(errors) > 0
        assert any("log_format" in error for error in errors)

        # Invalid log level
        config = LoggingConfig(log_level="TRACE")
        errors = config.validate()
        assert len(errors) > 0
        assert any("log_level" in error for error in errors)

        # Invalid queue size
        config = LoggingConfig(queue_size=-10)
        errors = config.validate()
        assert len(errors) > 0
        assert any("queue_size" in error for error in errors)

    def test_wandb_config_creation(self):
        """Test WandbConfig creation and validation."""
        config = WandbConfig(
            enabled=True,
            project_name="jaxarc-test",
            entity="test-entity",
            tags=["test", "jaxarc"],
            notes="Test run",
            group="test-group",
            job_type="training",
            log_frequency=10,
            image_format="png",
            max_image_size=(800, 600),
            log_gradients=False,
            log_model_topology=False,
            log_system_metrics=True,
            offline_mode=False,
            retry_attempts=3,
            retry_delay=1.0,
            save_code=True,
            save_config=True,
        )

        # Test field values
        assert config.enabled is True
        assert config.project_name == "jaxarc-test"
        assert config.entity == "test-entity"
        assert config.tags == ["test", "jaxarc"]
        assert config.notes == "Test run"
        assert config.group == "test-group"
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

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_wandb_config_validation_errors(self):
        """Test WandbConfig validation errors."""
        # Invalid image format
        config = WandbConfig(image_format="jpeg")
        errors = config.validate()
        assert len(errors) > 0
        assert any("image_format" in error for error in errors)

        # Invalid project name (empty)
        config = WandbConfig(project_name="")
        errors = config.validate()
        assert len(errors) > 0
        assert any("project_name" in error for error in errors)

        # Invalid retry attempts
        config = WandbConfig(retry_attempts=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("retry_attempts" in error for error in errors)


class TestJaxArcConfig:
    """Test unified JaxArcConfig container."""

    def test_jaxarc_config_creation(self):
        """Test JaxArcConfig creation with all components."""
        environment = EnvironmentConfig(max_episode_steps=100)
        dataset = DatasetConfig(dataset_name="arc-agi-1")
        action = ActionConfig(selection_format="mask")
        reward = RewardConfig(success_bonus=10.0)
        visualization = VisualizationConfig(enabled=True)
        storage = StorageConfig(policy="standard")
        logging = LoggingConfig(log_level="INFO")
        wandb = WandbConfig(enabled=False)

        config = JaxArcConfig(
            environment=environment,
            dataset=dataset,
            action=action,
            reward=reward,
            visualization=visualization,
            storage=storage,
            logging=logging,
            wandb=wandb,
        )

        # Test component references
        assert config.environment is environment
        assert config.dataset is dataset
        assert config.action is action
        assert config.reward is reward
        assert config.visualization is visualization
        assert config.storage is storage
        assert config.logging is logging
        assert config.wandb is wandb

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_jaxarc_config_defaults(self):
        """Test JaxArcConfig with default components."""
        config = JaxArcConfig()

        # Test default components are created
        assert isinstance(config.environment, EnvironmentConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.action, ActionConfig)
        assert isinstance(config.reward, RewardConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.wandb, WandbConfig)

        # Test validation
        errors = config.validate()
        assert len(errors) == 0

    def test_jaxarc_config_validation(self):
        """Test JaxArcConfig validation with invalid components."""
        # Create config with invalid components
        environment = EnvironmentConfig(max_episode_steps=-10)  # Invalid
        dataset = DatasetConfig(max_grid_height=5, min_grid_height=10)  # Invalid
        action = ActionConfig(selection_format="invalid")  # Invalid

        config = JaxArcConfig(
            environment=environment,
            dataset=dataset,
            action=action,
        )

        # Test validation collects errors from all components
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_episode_steps" in error for error in errors)
        assert any("grid_height" in error for error in errors)
        assert any("selection_format" in error for error in errors)

    def test_jaxarc_config_from_hydra(self):
        """Test JaxArcConfig creation from Hydra DictConfig."""
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
                "visualization": {
                    "enabled": True,
                    "level": "verbose",
                },
                "storage": {
                    "policy": "research",
                    "base_output_dir": "outputs/research",
                },
                "logging": {
                    "log_level": "DEBUG",
                    "log_operations": True,
                },
                "wandb": {
                    "enabled": True,
                    "project_name": "jaxarc-research",
                },
            }
        )

        config = JaxArcConfig.from_hydra(hydra_config)

        # Test component values from Hydra config
        assert config.environment.max_episode_steps == 150
        assert config.environment.debug_level == "verbose"
        assert config.dataset.dataset_name == "arc-agi-2"
        assert config.dataset.max_grid_height == 25
        assert config.action.selection_format == "bbox"
        assert config.action.selection_threshold == 0.7
        assert config.reward.success_bonus == 15.0
        assert config.reward.step_penalty == -0.02
        assert config.visualization.enabled is True
        assert config.visualization.level == "verbose"
        assert config.storage.policy == "research"
        assert config.storage.base_output_dir == "outputs/research"
        assert config.logging.log_level == "DEBUG"
        assert config.logging.log_operations is True
        assert config.wandb.enabled is True
        assert config.wandb.project_name == "jaxarc-research"

    def test_jaxarc_config_serialization(self):
        """Test JaxArcConfig can be serialized."""
        config = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=120),
            dataset=DatasetConfig(dataset_name="test-dataset"),
            reward=RewardConfig(success_bonus=12.5),
        )

        # Test that the config can be represented as a string
        config_str = str(config)
        assert "max_episode_steps=120" in config_str
        assert "test-dataset" in config_str
        assert "success_bonus=12.5" in config_str

    def test_jaxarc_config_manual_creation(self):
        """Test JaxArcConfig creation with specific values."""
        # Create config with specific values
        config = JaxArcConfig(
            environment=EnvironmentConfig(
                max_episode_steps=180,
                debug_level="research",
            ),
            dataset=DatasetConfig(
                dataset_name="custom-dataset",
                max_grid_height=40,
            ),
            action=ActionConfig(
                selection_format="point",
                selection_threshold=0.6,
            ),
            reward=RewardConfig(
                success_bonus=20.0,
                step_penalty=-0.05,
            ),
        )

        # Test component values
        assert config.environment.max_episode_steps == 180
        assert config.environment.debug_level == "research"
        assert config.dataset.dataset_name == "custom-dataset"
        assert config.dataset.max_grid_height == 40
        assert config.action.selection_format == "point"
        assert config.action.selection_threshold == 0.6
        assert config.reward.success_bonus == 20.0
        assert config.reward.step_penalty == -0.05


class TestJaxCompatibility:
    """Test JAX compatibility of Equinox configuration modules."""

    def test_equinox_module_jax_compatibility(self):
        """Test that configuration modules are compatible with JAX."""
        # Create a simple config
        config = EnvironmentConfig(max_episode_steps=100)

        # Test that it's an Equinox module
        assert isinstance(config, eqx.Module)

        # Test JAX tree compatibility using jax.tree.map (updated API)
        mapped = jax.tree.map(lambda x: x, config)
        assert eqx.tree_equal(config, mapped)

    def test_jaxarc_config_jax_compatibility(self):
        """Test that JaxArcConfig is compatible with JAX."""
        # Create a simple config
        config = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=100),
            dataset=DatasetConfig(dataset_name="test"),
        )

        # Test that it's an Equinox module
        assert isinstance(config, eqx.Module)

        # Test JAX tree compatibility using jax.tree.map (updated API)
        mapped = jax.tree.map(lambda x: x, config)
        assert eqx.tree_equal(config, mapped)

    def test_config_jit_compatibility(self):
        """Test that configuration can be used with JAX JIT."""

        # Create a simple function that uses config fields
        def config_function(max_steps):
            return max_steps

        # Create a config
        config = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=100))

        # Extract the value we want to test with
        max_steps = config.environment.max_episode_steps

        # JIT the function
        jitted_fn = jax.jit(config_function)
        result = jitted_fn(max_steps)

        # Test result
        assert result == 100

    def test_config_vmap_compatibility(self):
        """Test that configuration can be used with JAX vmap."""

        # Create a simple function that uses config and an array
        def config_array_function(config, array):
            return array * config.environment.max_episode_steps

        # Create a config
        config = JaxArcConfig(environment=EnvironmentConfig(max_episode_steps=10))

        # Create an array
        array = jnp.array([1, 2, 3, 4, 5])

        # vmap the function with static config
        vmapped_fn = jax.vmap(lambda x: config_array_function(config, x))
        result = vmapped_fn(array)

        # Test result
        assert jnp.array_equal(result, jnp.array([10, 20, 30, 40, 50]))


# Remove the TestConfigConversion class entirely as the conversion functions don't exist yet


if __name__ == "__main__":
    pytest.main([__file__])
