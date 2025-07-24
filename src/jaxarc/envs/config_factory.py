"""
Configuration factory system for creating common JaxARC configuration patterns.

This module provides the ConfigFactory class with preset creation methods
for common configuration patterns (development, research, production) and
utilities for converting from Hydra configurations.

This replaces the dual configuration pattern with a unified approach using
the JaxArcConfig system.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from omegaconf import DictConfig

from .config import (
    ActionConfig,
    ConfigValidationError,
    DatasetConfig,
    EnvironmentConfig,
    JaxArcConfig,
    LoggingConfig,
    RewardConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)
from .episode_manager import ArcEpisodeConfig
from .action_history import HistoryConfig


class ConfigFactory:
    """Factory for creating common configuration patterns.

    This class provides static methods for creating JaxArcConfig instances
    with sensible defaults for different use cases. It eliminates the dual
    configuration pattern by providing a single, unified configuration system.

    All factory methods return fully validated JaxArcConfig instances that
    can be used directly with ArcEnvironment.
    """

    @staticmethod
    def create_development_config(**overrides: Any) -> JaxArcConfig:
        """Create development configuration with sensible defaults.

        Optimized for development and debugging with moderate logging,
        visualization enabled, and reasonable episode lengths.

        Args:
            **overrides: Configuration overrides to apply

        Returns:
            JaxArcConfig configured for development use

        Example:
            ```python
            config = ConfigFactory.create_development_config(
                max_episode_steps=75, dataset_name="mini-arc"
            )
            env = ArcEnvironment(config)
            ```
        """
        try:
            # Environment settings optimized for development
            environment = EnvironmentConfig(
                max_episode_steps=50,
                auto_reset=True,
                strict_validation=True,
                allow_invalid_actions=False,
                debug_level="standard",
            )

            # Dataset settings for development
            dataset = DatasetConfig(
                dataset_name="arc-agi-1",
                dataset_path="",
                max_grid_height=30,
                max_grid_width=30,
                min_grid_height=3,
                min_grid_width=3,
                max_colors=10,
                background_color=-1,
                task_split="train",
                shuffle_tasks=True,
                max_train_pairs=5,  # Smaller for faster development
                max_test_pairs=2,
            )

            # Action settings for development
            action = ActionConfig(
                selection_format="mask",
                selection_threshold=0.5,
                allow_partial_selection=True,
                max_operations=35,  # Standardized naming
                allowed_operations=None,  # All operations available
                validate_actions=True,
                allow_invalid_actions=False,  # Standardized naming
            )

            # Reward settings for development
            reward = RewardConfig(
                reward_on_submit_only=True,
                step_penalty=-0.01,
                invalid_action_penalty=-0.1,
                success_bonus=10.0,
                similarity_weight=1.0,
                progress_bonus=0.1,
            )

            # Visualization enabled for development
            visualization = VisualizationConfig(
                enabled=True,
                level="standard",
                output_formats=["svg"],
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
                max_memory_mb=300,  # Standardized naming
            )

            # Storage settings for development
            storage = StorageConfig(
                policy="standard",
                base_output_dir="outputs/dev",
                run_name=None,
                episodes_dir="episodes",
                debug_dir="debug",
                visualization_dir="visualizations",
                logs_dir="logs",
                max_episodes_per_run=50,
                max_storage_gb=2.0,
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

            # Logging settings for development
            logging = LoggingConfig(
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
                queue_size=200,
                worker_threads=1,
                batch_size=5,
                flush_interval=5.0,
                enable_compression=True,
            )

            # WandB disabled by default for development
            wandb = WandbConfig(
                enabled=False,
                project_name="jaxarc-dev",
                entity=None,
                tags=["development"],
                notes="Development configuration",
                group=None,
                job_type="development",
                log_frequency=20,
                image_format="png",
                max_image_size=(600, 400),
                log_gradients=False,
                log_model_topology=False,
                log_system_metrics=False,
                offline_mode=False,
                retry_attempts=2,
                retry_delay=1.0,
                save_code=False,
                save_config=True,
            )

            # Episode management for development
            episode = ArcEpisodeConfig(
                episode_mode="train",
                demo_selection_strategy="random",
                allow_demo_switching=True,
                require_all_demos_solved=False,
                test_selection_strategy="sequential",
                allow_test_switching=False,
                require_all_tests_solved=True,
                terminate_on_first_success=False,
                max_pairs_per_episode=3,  # Smaller for development
                success_threshold=1.0,
                training_reward_frequency="step",
                evaluation_reward_frequency="submit",
            )

            # Action history for development
            history = HistoryConfig(
                enabled=True,
                max_history_length=500,  # Moderate for development
                store_selection_data=True,
                store_intermediate_grids=False,  # Save memory
                compress_repeated_actions=True,
            )

            config = JaxArcConfig(
                environment=environment,
                dataset=dataset,
                action=action,
                reward=reward,
                visualization=visualization,
                storage=storage,
                logging=logging,
                wandb=wandb,
                episode=episode,
                history=history,
            )

            # Apply overrides
            if overrides:
                config = ConfigFactory._apply_overrides(config, overrides)

            # Validate final configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "Development config validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to create development config: {e}"
            ) from e

    @staticmethod
    def create_research_config(**overrides: Any) -> JaxArcConfig:
        """Create research configuration with full logging and analysis features.

        Optimized for research with comprehensive logging, full visualization,
        extended episode lengths, and detailed analysis capabilities.

        Args:
            **overrides: Configuration overrides to apply

        Returns:
            JaxArcConfig configured for research use

        Example:
            ```python
            config = ConfigFactory.create_research_config(
                max_episode_steps=300, wandb_enabled=True, wandb_project="my-research"
            )
            env = ArcEnvironment(config)
            ```
        """
        try:
            # Environment settings optimized for research
            environment = EnvironmentConfig(
                max_episode_steps=200,
                auto_reset=True,
                strict_validation=True,
                allow_invalid_actions=False,
                debug_level="research",
            )

            # Dataset settings for research
            dataset = DatasetConfig(
                dataset_name="arc-agi-1",
                dataset_path="",
                max_grid_height=30,
                max_grid_width=30,
                min_grid_height=3,
                min_grid_width=3,
                max_colors=10,
                background_color=-1,
                task_split="train",
                shuffle_tasks=True,
                max_train_pairs=10,  # Full dataset for research
                max_test_pairs=3,
            )

            # Action settings for research
            action = ActionConfig(
                selection_format="mask",
                selection_threshold=0.5,
                allow_partial_selection=True,
                max_operations=35,  # Standardized naming
                allowed_operations=None,  # All operations available
                validate_actions=True,
                allow_invalid_actions=False,  # Standardized naming
            )

            # Reward settings for research
            reward = RewardConfig(
                reward_on_submit_only=False,  # Intermediate rewards for analysis
                step_penalty=-0.005,  # Lower penalty for longer episodes
                invalid_action_penalty=-0.1,
                success_bonus=20.0,  # Higher bonus for success
                similarity_weight=1.0,
                progress_bonus=0.2,  # Reward progress for analysis
            )

            # Full visualization for research
            visualization = VisualizationConfig(
                enabled=True,
                level="full",
                output_formats=["svg", "png"],
                show_coordinates=True,
                show_operation_names=True,
                highlight_changes=True,
                include_metrics=True,
                color_scheme="default",
                visualize_episodes=True,
                episode_summaries=True,
                step_visualizations=True,
                enable_comparisons=True,
                save_intermediate_states=True,  # Save for analysis
                lazy_loading=True,
                max_memory_mb=1000,  # Higher limit for research, standardized naming
            )

            # Storage settings for research
            storage = StorageConfig(
                policy="research",
                base_output_dir="outputs/research",
                run_name=None,
                episodes_dir="episodes",
                debug_dir="debug",
                visualization_dir="visualizations",
                logs_dir="logs",
                max_episodes_per_run=500,  # More episodes for research
                max_storage_gb=20.0,  # Higher storage limit
                cleanup_policy="oldest_first",
                cleanup_frequency="manual",  # Manual cleanup for research
                keep_recent_episodes=50,
                create_run_subdirs=True,
                timestamp_format="%Y%m%d_%H%M%S",
                compress_old_files=True,
                clear_output_on_start=False,  # Keep previous runs
                auto_cleanup=False,  # Manual cleanup for research
                warn_on_storage_limit=True,
                fail_on_storage_full=False,
            )

            # Comprehensive logging for research
            logging = LoggingConfig(
                structured_logging=True,
                log_format="json",
                log_level="DEBUG",  # Detailed logging
                compression=True,
                include_full_states=True,  # Full state logging for analysis
                log_operations=True,
                log_grid_changes=True,
                log_rewards=True,
                log_episode_start=True,
                log_episode_end=True,
                log_key_moments=True,
                log_frequency=1,  # Log every step
                queue_size=1000,  # Larger queue for research
                worker_threads=2,
                batch_size=10,
                flush_interval=30.0,
                enable_compression=True,
            )

            # WandB enabled for research tracking
            wandb = WandbConfig(
                enabled=True,
                project_name="jaxarc-research",
                entity=None,
                tags=["research", "full-logging"],
                notes="Research configuration with full logging",
                group=None,
                job_type="research",
                log_frequency=5,  # Frequent logging to WandB
                image_format="both",
                max_image_size=(1200, 800),
                log_gradients=True,
                log_model_topology=True,
                log_system_metrics=True,
                offline_mode=False,
                retry_attempts=5,
                retry_delay=2.0,
                save_code=True,
                save_config=True,
            )

            # Episode management for research
            episode = ArcEpisodeConfig(
                episode_mode="train",
                demo_selection_strategy="random",
                allow_demo_switching=True,
                require_all_demos_solved=False,
                test_selection_strategy="sequential",
                allow_test_switching=True,  # Allow switching for research
                require_all_tests_solved=False,  # More flexible for research
                terminate_on_first_success=False,
                max_pairs_per_episode=10,  # More pairs for research
                success_threshold=1.0,
                training_reward_frequency="step",
                evaluation_reward_frequency="submit",
            )

            # Full action history for research
            history = HistoryConfig(
                enabled=True,
                max_history_length=2000,  # Large history for research
                store_selection_data=True,
                store_intermediate_grids=True,  # Full data for research
                compress_repeated_actions=False,  # No compression for analysis
            )

            config = JaxArcConfig(
                environment=environment,
                dataset=dataset,
                action=action,
                reward=reward,
                visualization=visualization,
                storage=storage,
                logging=logging,
                wandb=wandb,
                episode=episode,
                history=history,
            )

            # Apply overrides
            if overrides:
                config = ConfigFactory._apply_overrides(config, overrides)

            # Validate final configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "Research config validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(f"Failed to create research config: {e}") from e

    @staticmethod
    def create_production_config(**overrides: Any) -> JaxArcConfig:
        """Create production configuration with minimal overhead.

        Optimized for production deployment with minimal logging,
        no visualization, and efficient resource usage.

        Args:
            **overrides: Configuration overrides to apply

        Returns:
            JaxArcConfig configured for production use

        Example:
            ```python
            config = ConfigFactory.create_production_config(
                max_episode_steps=100, dataset_name="arc-agi-2"
            )
            env = ArcEnvironment(config)
            ```
        """
        try:
            # Environment settings optimized for production
            environment = EnvironmentConfig(
                max_episode_steps=100,
                auto_reset=True,
                strict_validation=True,
                allow_invalid_actions=False,
                debug_level="off",  # No debugging in production
            )

            # Dataset settings for production
            dataset = DatasetConfig(
                dataset_name="arc-agi-1",
                dataset_path="",
                max_grid_height=30,
                max_grid_width=30,
                min_grid_height=3,
                min_grid_width=3,
                max_colors=10,
                background_color=-1,
                task_split="eval",  # Use evaluation split for production
                shuffle_tasks=False,  # Deterministic for production
                max_train_pairs=10,
                max_test_pairs=3,
            )

            # Action settings for production
            action = ActionConfig(
                selection_format="mask",
                selection_threshold=0.5,
                allow_partial_selection=True,
                max_operations=35,  # Standardized naming
                allowed_operations=None,
                validate_actions=True,
                allow_invalid_actions=False,  # Standardized naming
            )

            # Reward settings for production
            reward = RewardConfig(
                reward_on_submit_only=True,
                step_penalty=0.0,  # No step penalty in production
                invalid_action_penalty=-0.1,
                success_bonus=1.0,  # Simple binary reward
                similarity_weight=1.0,
                progress_bonus=0.0,  # No intermediate rewards
            )

            # Visualization disabled for production
            visualization = VisualizationConfig(
                enabled=False,
                level="off",
                output_formats=[],
                show_coordinates=False,
                show_operation_names=False,
                highlight_changes=False,
                include_metrics=False,
                color_scheme="default",
                visualize_episodes=False,
                episode_summaries=False,
                step_visualizations=False,
                enable_comparisons=False,
                save_intermediate_states=False,
                lazy_loading=True,
                max_memory_mb=100,  # Minimal memory usage, standardized naming
            )

            # Minimal storage for production
            storage = StorageConfig(
                policy="minimal",
                base_output_dir="outputs/prod",
                run_name=None,
                episodes_dir="episodes",
                debug_dir="debug",
                visualization_dir="visualizations",
                logs_dir="logs",
                max_episodes_per_run=100,
                max_storage_gb=1.0,  # Minimal storage
                cleanup_policy="size_based",
                cleanup_frequency="after_run",
                keep_recent_episodes=5,  # Keep only recent episodes
                create_run_subdirs=False,  # Simpler structure
                timestamp_format="%Y%m%d_%H%M%S",
                compress_old_files=True,
                clear_output_on_start=True,
                auto_cleanup=True,
                warn_on_storage_limit=True,
                fail_on_storage_full=True,  # Fail fast in production
            )

            # Minimal logging for production
            logging = LoggingConfig(
                structured_logging=False,
                log_format="text",
                log_level="ERROR",  # Only errors in production
                compression=False,
                include_full_states=False,
                log_operations=False,
                log_grid_changes=False,
                log_rewards=False,
                log_episode_start=False,
                log_episode_end=False,
                log_key_moments=False,
                log_frequency=100,  # Minimal logging
                queue_size=50,
                worker_threads=1,
                batch_size=1,
                flush_interval=60.0,
                enable_compression=False,
            )

            # WandB disabled for production
            wandb = WandbConfig(
                enabled=False,
                project_name="jaxarc-prod",
                entity=None,
                tags=["production"],
                notes="Production configuration",
                group=None,
                job_type="production",
                log_frequency=100,
                image_format="png",
                max_image_size=(400, 300),
                log_gradients=False,
                log_model_topology=False,
                log_system_metrics=False,
                offline_mode=True,  # Offline mode for production
                retry_attempts=1,
                retry_delay=0.5,
                save_code=False,
                save_config=False,
            )

            # Episode management for production
            episode = ArcEpisodeConfig(
                episode_mode="test",  # Production uses test mode
                demo_selection_strategy="sequential",
                allow_demo_switching=False,  # No switching in production
                require_all_demos_solved=True,
                test_selection_strategy="sequential",
                allow_test_switching=False,
                require_all_tests_solved=True,
                terminate_on_first_success=True,  # Terminate early for efficiency
                max_pairs_per_episode=1,  # Single pair for production
                success_threshold=1.0,
                training_reward_frequency="submit",
                evaluation_reward_frequency="submit",
            )

            # Minimal action history for production
            history = HistoryConfig(
                enabled=False,  # Disabled for production efficiency
                max_history_length=100,  # Minimal if enabled
                store_selection_data=False,
                store_intermediate_grids=False,
                compress_repeated_actions=True,
            )

            config = JaxArcConfig(
                environment=environment,
                dataset=dataset,
                action=action,
                reward=reward,
                visualization=visualization,
                storage=storage,
                logging=logging,
                wandb=wandb,
                episode=episode,
                history=history,
            )

            # Apply overrides
            if overrides:
                config = ConfigFactory._apply_overrides(config, overrides)

            # Validate final configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "Production config validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to create production config: {e}"
            ) from e

    @staticmethod
    def from_hydra(hydra_config: DictConfig) -> JaxArcConfig:
        """Convert Hydra DictConfig to JaxArcConfig, eliminating dual config pattern.

        This method replaces the confusing dual configuration pattern by converting
        Hydra configurations directly to typed JaxArcConfig objects.

        Args:
            hydra_config: Hydra DictConfig object

        Returns:
            JaxArcConfig instance created from Hydra config

        Raises:
            ConfigValidationError: If conversion or validation fails

        Example:
            ```python
            @hydra.main(config_path="conf", config_name="config")
            def main(cfg: DictConfig):
                config = ConfigFactory.from_hydra(cfg)
                env = ArcEnvironment(config)  # Single config pattern
            ```
        """
        try:
            # Use the existing from_hydra method in JaxArcConfig
            config = JaxArcConfig.from_hydra(hydra_config)

            # Validate the converted configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "Hydra config conversion validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            logger.info("Successfully converted Hydra config to JaxArcConfig")
            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(f"Failed to convert Hydra config: {e}") from e

    @staticmethod
    def _create_testing_preset(**overrides: Any) -> JaxArcConfig:
        """Create testing preset with proper override handling."""
        testing_defaults = {
            "max_episode_steps": 20,
            "debug_level": "minimal",
            "visualization_enabled": False,
            "max_storage_gb": 0.5,
        }
        final_overrides = {**testing_defaults, **overrides}
        return ConfigFactory.create_development_config(**final_overrides)

    @staticmethod
    def _create_minimal_preset(**overrides: Any) -> JaxArcConfig:
        """Create minimal preset with proper override handling."""
        minimal_defaults = {
            "max_episode_steps": 50,
            "debug_level": "off",
            "visualization_enabled": False,
            "log_level": "ERROR",
        }
        final_overrides = {**minimal_defaults, **overrides}
        return ConfigFactory.create_production_config(**final_overrides)

    @staticmethod
    def _create_debug_preset(**overrides: Any) -> JaxArcConfig:
        """Create debug preset with proper override handling."""
        debug_defaults = {
            "max_episode_steps": 30,
            "debug_level": "verbose",
            "log_level": "DEBUG",
            "log_operations": True,
            "log_grid_changes": True,
        }
        final_overrides = {**debug_defaults, **overrides}
        return ConfigFactory.create_research_config(**final_overrides)

    @staticmethod
    def _create_curriculum_basic_preset(**overrides: Any) -> JaxArcConfig:
        """Create curriculum basic preset with proper override handling."""
        curriculum_defaults = {
            "max_episode_steps": 30,
            "allowed_operations": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                34,
            ],  # Fill colors + submit
            "reward_on_submit_only": True,
            "success_bonus": 15.0,
        }
        final_overrides = {**curriculum_defaults, **overrides}
        return ConfigFactory.create_development_config(**final_overrides)

    @staticmethod
    def _create_curriculum_advanced_preset(**overrides: Any) -> JaxArcConfig:
        """Create curriculum advanced preset with proper override handling."""
        curriculum_defaults = {
            "max_episode_steps": 150,
            "reward_on_submit_only": False,
            "progress_bonus": 0.3,
        }
        final_overrides = {**curriculum_defaults, **overrides}
        return ConfigFactory.create_research_config(**final_overrides)

    @staticmethod
    def _create_evaluation_preset(**overrides: Any) -> JaxArcConfig:
        """Create evaluation preset with proper override handling."""
        eval_defaults = {
            "max_episode_steps": 100,
            "task_split": "eval",
            "reward_on_submit_only": True,
            "step_penalty": 0.0,
            "success_bonus": 1.0,
            "shuffle_tasks": False,
        }
        final_overrides = {**eval_defaults, **overrides}
        return ConfigFactory.create_production_config(**final_overrides)

    @staticmethod
    def _create_point_actions_preset(**overrides: Any) -> JaxArcConfig:
        """Create point actions preset with proper override handling."""
        point_defaults = {
            "selection_format": "point",
            "allow_partial_selection": False,
            "max_episode_steps": 80,
        }
        final_overrides = {**point_defaults, **overrides}
        return ConfigFactory.create_development_config(**final_overrides)

    @staticmethod
    def _create_bbox_actions_preset(**overrides: Any) -> JaxArcConfig:
        """Create bbox actions preset with proper override handling."""
        bbox_defaults = {
            "selection_format": "bbox",
            "allow_partial_selection": False,
            "max_episode_steps": 90,
        }
        final_overrides = {**bbox_defaults, **overrides}
        return ConfigFactory.create_development_config(**final_overrides)

    @staticmethod
    def _create_mini_arc_preset(**overrides: Any) -> JaxArcConfig:
        """Create mini arc preset with proper override handling."""
        mini_defaults = {
            "dataset_name": "mini-arc",
            "max_grid_height": 5,
            "max_grid_width": 5,
            "max_episode_steps": 40,
            "selection_format": "point",
        }
        final_overrides = {**mini_defaults, **overrides}
        return ConfigFactory.create_development_config(**final_overrides)

    @staticmethod
    def _create_concept_arc_preset(**overrides: Any) -> JaxArcConfig:
        """Create concept arc preset with proper override handling."""
        concept_defaults = {
            "dataset_name": "concept-arc",
            "max_grid_height": 15,
            "max_grid_width": 15,
            "task_split": "corpus",
            "max_episode_steps": 120,
        }
        final_overrides = {**concept_defaults, **overrides}
        return ConfigFactory.create_research_config(**final_overrides)

    @staticmethod
    def _create_wandb_research_preset(**overrides: Any) -> JaxArcConfig:
        """Create wandb research preset with proper override handling."""
        wandb_defaults = {
            "wandb_enabled": True,
            "wandb_project": "jaxarc-research",
            "log_frequency": 5,
        }
        final_overrides = {**wandb_defaults, **overrides}
        return ConfigFactory.create_research_config(**final_overrides)

    @staticmethod
    def _create_memory_efficient_preset(**overrides: Any) -> JaxArcConfig:
        """Create memory efficient preset with proper override handling."""
        memory_defaults = {
            "memory_limit_mb": 50,
            "max_storage_gb": 0.5,
            "queue_size": 50,
            "batch_size": 1,
        }
        final_overrides = {**memory_defaults, **overrides}
        return ConfigFactory.create_production_config(**final_overrides)

    @staticmethod
    def _create_high_performance_preset(**overrides: Any) -> JaxArcConfig:
        """Create high performance preset with proper override handling."""
        perf_defaults = {
            "debug_level": "off",
            "visualization_enabled": False,
            "log_level": "ERROR",
            "structured_logging": False,
            "compression": False,
        }
        final_overrides = {**perf_defaults, **overrides}
        return ConfigFactory.create_production_config(**final_overrides)

    @staticmethod
    def _apply_overrides(
        config: JaxArcConfig, overrides: Dict[str, Any]
    ) -> JaxArcConfig:
        """Apply configuration overrides to a JaxArcConfig instance.

        This method handles nested overrides by updating the appropriate
        configuration components.

        Args:
            config: Base JaxArcConfig instance
            overrides: Dictionary of configuration overrides

        Returns:
            JaxArcConfig with overrides applied
        """
        try:
            # Convert config to dictionary for easier manipulation
            config_dict = {
                "environment": config._config_to_dict(config.environment),
                "dataset": config._config_to_dict(config.dataset),
                "action": config._config_to_dict(config.action),
                "reward": config._config_to_dict(config.reward),
                "visualization": config._config_to_dict(config.visualization),
                "storage": config._config_to_dict(config.storage),
                "logging": config._config_to_dict(config.logging),
                "wandb": config._config_to_dict(config.wandb),
            }

            # Apply overrides
            for key, value in overrides.items():
                if key in config_dict:
                    # Direct section override
                    if isinstance(value, dict):
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value
                else:
                    # Try to map to appropriate section
                    ConfigFactory._map_override_to_section(config_dict, key, value)

            # Create new config from updated dictionary
            return JaxArcConfig._from_dict(config_dict)

        except Exception as e:
            raise ConfigValidationError(f"Failed to apply overrides: {e}") from e

    @staticmethod
    def _map_override_to_section(
        config_dict: Dict[str, Any], key: str, value: Any
    ) -> None:
        """Map a configuration override to the appropriate section."""
        # Environment overrides
        if key in [
            "max_episode_steps",
            "auto_reset",
            "strict_validation",
            "allow_invalid_actions",
            "debug_level",
        ]:
            config_dict["environment"][key] = value
        # Dataset overrides
        elif key in [
            "dataset_name",
            "dataset_path",
            "max_grid_height",
            "max_grid_width",
            "task_split",
            "shuffle_tasks",
        ]:
            config_dict["dataset"][key] = value
        # Action overrides
        elif key in [
            "selection_format",
            "num_operations",
            "validate_actions",
            "allowed_operations",
        ]:
            config_dict["action"][key] = value
        # Reward overrides
        elif key in [
            "reward_on_submit_only",
            "step_penalty",
            "success_bonus",
            "similarity_weight",
        ]:
            config_dict["reward"][key] = value
        # Visualization overrides
        elif key in [
            "visualization_enabled",
            "visualization_level",
            "show_coordinates",
        ]:
            # Map common visualization overrides
            if key == "visualization_enabled":
                config_dict["visualization"]["enabled"] = value
            elif key == "visualization_level":
                config_dict["visualization"]["level"] = value
            else:
                config_dict["visualization"][key] = value
        # Storage overrides
        elif key in ["base_output_dir", "max_storage_gb", "cleanup_policy"]:
            config_dict["storage"][key] = value
        # Logging overrides
        elif key in ["log_operations", "log_grid_changes", "log_rewards", "log_level"]:
            config_dict["logging"][key] = value
        # WandB overrides
        elif key in ["wandb_enabled", "wandb_project", "wandb_entity"]:
            # Map common wandb overrides
            if key == "wandb_enabled":
                config_dict["wandb"]["enabled"] = value
            elif key == "wandb_project":
                config_dict["wandb"]["project_name"] = value
            elif key == "wandb_entity":
                config_dict["wandb"]["entity"] = value
        else:
            logger.warning(f"Unknown configuration override: {key} = {value}")


# Convenience functions for backward compatibility
def create_development_config(**overrides: Any) -> JaxArcConfig:
    """Convenience function for creating development configuration."""
    return ConfigFactory.create_development_config(**overrides)


def create_research_config(**overrides: Any) -> JaxArcConfig:
    """Convenience function for creating research configuration."""
    return ConfigFactory.create_research_config(**overrides)


def create_production_config(**overrides: Any) -> JaxArcConfig:
    """Convenience function for creating production configuration."""
    return ConfigFactory.create_production_config(**overrides)


def from_hydra(hydra_config: DictConfig) -> JaxArcConfig:
    """Convenience function for converting Hydra config."""
    return ConfigFactory.from_hydra(hydra_config)


# Preset system for task 4.2
class ConfigPresets:
    """Predefined configuration presets for different use cases.

    This class provides a registry of named presets that can be loaded
    with optional overrides. Each preset is internally consistent and
    validated.
    """

    # Registry of available presets
    _PRESETS = {
        "development": ConfigFactory.create_development_config,
        "research": ConfigFactory.create_research_config,
        "production": ConfigFactory.create_production_config,
        "testing": lambda **overrides: ConfigFactory._create_testing_preset(
            **overrides
        ),
        "minimal": lambda **overrides: ConfigFactory._create_minimal_preset(
            **overrides
        ),
        "debug": lambda **overrides: ConfigFactory._create_debug_preset(**overrides),
        "curriculum_basic": lambda **overrides: ConfigFactory._create_curriculum_basic_preset(
            **overrides
        ),
        "curriculum_advanced": lambda **overrides: ConfigFactory._create_curriculum_advanced_preset(
            **overrides
        ),
        "evaluation": lambda **overrides: ConfigFactory._create_evaluation_preset(
            **overrides
        ),
        "point_actions": lambda **overrides: ConfigFactory._create_point_actions_preset(
            **overrides
        ),
        "bbox_actions": lambda **overrides: ConfigFactory._create_bbox_actions_preset(
            **overrides
        ),
        "mini_arc": lambda **overrides: ConfigFactory._create_mini_arc_preset(
            **overrides
        ),
        "concept_arc": lambda **overrides: ConfigFactory._create_concept_arc_preset(
            **overrides
        ),
        "wandb_research": lambda **overrides: ConfigFactory._create_wandb_research_preset(
            **overrides
        ),
        "memory_efficient": lambda **overrides: ConfigFactory._create_memory_efficient_preset(
            **overrides
        ),
        "high_performance": lambda **overrides: ConfigFactory._create_high_performance_preset(
            **overrides
        ),
    }

    @classmethod
    def get_available_presets(cls) -> list[str]:
        """Get list of available preset names.

        Returns:
            List of available preset names
        """
        return list(cls._PRESETS.keys())

    @classmethod
    def from_preset(cls, preset_name: str, **overrides: Any) -> JaxArcConfig:
        """Load a named preset with optional overrides.

        Args:
            preset_name: Name of the preset to load
            **overrides: Configuration overrides to apply

        Returns:
            JaxArcConfig instance loaded from preset

        Raises:
            ConfigValidationError: If preset is unknown or validation fails

        Example:
            ```python
            # Load development preset with custom episode length
            config = ConfigPresets.from_preset("development", max_episode_steps=75)

            # Load research preset with WandB enabled
            config = ConfigPresets.from_preset("wandb_research", wandb_entity="my-team")
            ```
        """
        if preset_name not in cls._PRESETS:
            available = ", ".join(cls.get_available_presets())
            raise ConfigValidationError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        try:
            # Load the preset with overrides
            preset_func = cls._PRESETS[preset_name]
            config = preset_func(**overrides)

            # Validate the preset configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = f"Preset '{preset_name}' validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            logger.info(
                f"Successfully loaded preset '{preset_name}' with {len(overrides)} overrides"
            )
            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to load preset '{preset_name}': {e}"
            ) from e

    @classmethod
    def validate_preset(cls, preset_name: str) -> list[str]:
        """Validate a preset configuration without creating it.

        Args:
            preset_name: Name of the preset to validate

        Returns:
            List of validation errors (empty if valid)

        Raises:
            ConfigValidationError: If preset is unknown
        """
        if preset_name not in cls._PRESETS:
            available = ", ".join(cls.get_available_presets())
            raise ConfigValidationError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        try:
            # Create the preset configuration
            preset_func = cls._PRESETS[preset_name]
            config = preset_func()

            # Return validation results
            return config.validate()

        except Exception as e:
            return [f"Preset validation error: {e}"]

    @classmethod
    def get_preset_info(cls, preset_name: str) -> dict[str, Any]:
        """Get information about a preset configuration.

        Args:
            preset_name: Name of the preset

        Returns:
            Dictionary with preset information

        Raises:
            ConfigValidationError: If preset is unknown
        """
        if preset_name not in cls._PRESETS:
            available = ", ".join(cls.get_available_presets())
            raise ConfigValidationError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        try:
            # Create the preset to analyze it
            preset_func = cls._PRESETS[preset_name]
            config = preset_func()

            return {
                "name": preset_name,
                "debug_level": config.environment.debug_level,
                "max_episode_steps": config.environment.max_episode_steps,
                "dataset_name": config.dataset.dataset_name,
                "selection_format": config.action.selection_format,
                "visualization_enabled": config.visualization.enabled,
                "visualization_level": config.visualization.level,
                "storage_policy": config.storage.policy,
                "log_level": config.logging.log_level,
                "wandb_enabled": config.wandb.enabled,
                "validation_errors": config.validate(),
            }

        except Exception as e:
            return {"name": preset_name, "error": str(e)}

    @classmethod
    def add_custom_preset(cls, name: str, preset_func: callable) -> None:
        """Add a custom preset to the registry.

        Args:
            name: Name for the custom preset
            preset_func: Function that returns a JaxArcConfig

        Raises:
            ConfigValidationError: If preset name already exists
        """
        if name in cls._PRESETS:
            raise ConfigValidationError(f"Preset '{name}' already exists")

        # Validate that the function returns a valid config
        try:
            test_config = preset_func()
            if not isinstance(test_config, JaxArcConfig):
                raise ConfigValidationError(
                    f"Preset function must return JaxArcConfig, got {type(test_config).__name__}"
                )

            validation_errors = test_config.validate()
            if validation_errors:
                error_msg = f"Custom preset '{name}' validation failed:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            cls._PRESETS[name] = preset_func
            logger.info(f"Added custom preset '{name}'")

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to add custom preset '{name}': {e}"
            ) from e


# Add ConfigFactory methods for preset system
def _add_preset_methods_to_factory():
    """Add preset methods to ConfigFactory class."""

    @staticmethod
    def from_preset(preset_name: str, **overrides: Any) -> JaxArcConfig:
        """Load a named preset with optional overrides.

        This method provides access to the preset system through ConfigFactory
        for consistency with other factory methods.

        Args:
            preset_name: Name of the preset to load
            **overrides: Configuration overrides to apply

        Returns:
            JaxArcConfig instance loaded from preset

        Example:
            ```python
            config = ConfigFactory.from_preset("development", max_episode_steps=75)
            ```
        """
        return ConfigPresets.from_preset(preset_name, **overrides)

    @staticmethod
    def get_available_presets() -> list[str]:
        """Get list of available preset names."""
        return ConfigPresets.get_available_presets()

    @staticmethod
    def validate_preset(preset_name: str) -> list[str]:
        """Validate a preset configuration."""
        return ConfigPresets.validate_preset(preset_name)

    # Add methods to ConfigFactory class
    ConfigFactory.from_preset = from_preset
    ConfigFactory.get_available_presets = get_available_presets
    ConfigFactory.validate_preset = validate_preset


# Apply the preset methods to ConfigFactory
_add_preset_methods_to_factory()


# Convenience functions for preset system
def from_preset(preset_name: str, **overrides: Any) -> JaxArcConfig:
    """Convenience function for loading presets."""
    return ConfigPresets.from_preset(preset_name, **overrides)


def get_available_presets() -> list[str]:
    """Convenience function for getting available presets."""
    return ConfigPresets.get_available_presets()
