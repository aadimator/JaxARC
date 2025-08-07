"""
Unified configuration system using Equinox and JaxTyping.

This module provides typed configuration dataclasses using equinox.Module
with jaxtyping annotations for JAX compatibility and type safety.

Each configuration class has a single, clear responsibility:
- EnvironmentConfig: Core environment behavior and constraints
- DatasetConfig: Dataset-specific settings and constraints
- ActionConfig: Action space and validation settings
- RewardConfig: Reward calculation settings
- VisualizationConfig: All visualization and rendering settings
- StorageConfig: All storage, output, and file management
- LoggingConfig: All logging behavior and formats
- WandbConfig: Weights & Biases integration
- JaxArcConfig: Unified configuration container
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import equinox as eqx
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Import episode configuration
from .episode_manager import ArcEpisodeConfig

# Import action history configuration
from .action_history import HistoryConfig


# Validation utilities
class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        msg = f"{field_name} must be an integer, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value <= 0:
        msg = f"{field_name} must be positive, got {value}"
        raise ConfigValidationError(msg)


def validate_non_negative_int(value: int, field_name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        msg = f"{field_name} must be an integer, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value < 0:
        msg = f"{field_name} must be non-negative, got {value}"
        raise ConfigValidationError(msg)


def validate_float_range(
    value: float, field_name: str, min_val: float, max_val: float
) -> None:
    """Validate that a float value is within a specified range."""
    if not isinstance(value, (int, float)):
        msg = f"{field_name} must be a number, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if not min_val <= value <= max_val:
        msg = f"{field_name} must be in range [{min_val}, {max_val}], got {value}"
        raise ConfigValidationError(msg)


def validate_string_choice(
    value: str, field_name: str, choices: tuple[str, ...]
) -> None:
    """Validate that a string value is one of the allowed choices."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a string, got {type(value).__name__}"
        raise ConfigValidationError(msg)
    if value not in choices:
        msg = f"{field_name} must be one of {choices}, got '{value}'"
        raise ConfigValidationError(msg)


def validate_path_string(value: str, field_name: str, must_exist: bool = False) -> None:
    """Validate that a value is a valid path string."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a string, got {type(value).__name__}"
        raise ConfigValidationError(msg)

    # Check for invalid path characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    if any(char in value for char in invalid_chars):
        msg = f"{field_name} contains invalid path characters: {value}"
        raise ConfigValidationError(msg)

    if must_exist and value and not Path(value).exists():
        msg = f"{field_name} path does not exist: {value}"
        raise ConfigValidationError(msg)


class EnvironmentConfig(eqx.Module):
    """Core environment behavior and runtime settings.

    This config only contains settings that directly affect environment behavior,
    not dataset constraints, logging, visualization, or storage settings.
    """

    # Episode settings
    max_episode_steps: int = 100
    auto_reset: bool = True

    # Environment behavior
    strict_validation: bool = True
    allow_invalid_actions: bool = False

    # Debug level (moved from separate DebugConfig)
    debug_level: Literal["off", "minimal", "standard", "verbose", "research"] = (
        "standard"
    )

    def validate(self) -> tuple[str, ...]:
        """Validate environment configuration and return tuple of errors."""
        errors = []

        try:
            # Validate episode settings
            validate_positive_int(self.max_episode_steps, "max_episode_steps")
            if self.max_episode_steps > 10000:
                logger.warning(
                    f"max_episode_steps is very large: {self.max_episode_steps}"
                )

            # Validate debug level
            valid_levels = ("off", "minimal", "standard", "verbose", "research")
            validate_string_choice(self.debug_level, "debug_level", valid_levels)

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    @property
    def computed_visualization_level(self) -> str:
        """Get computed visualization level based on debug level."""
        level_mapping = {
            "off": "off",
            "minimal": "minimal",
            "standard": "standard",
            "verbose": "verbose",
            "research": "full",
        }
        return level_mapping.get(self.debug_level, "standard")

    @property
    def computed_storage_policy(self) -> str:
        """Get computed storage policy based on debug level."""
        level_mapping = {
            "off": "none",
            "minimal": "minimal",
            "standard": "standard",
            "verbose": "research",
            "research": "research",
        }
        return level_mapping.get(self.debug_level, "standard")

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"EnvironmentConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> EnvironmentConfig:
        """Create environment config from Hydra DictConfig."""
        return cls(
            max_episode_steps=cfg.get("max_episode_steps", 100),
            auto_reset=cfg.get("auto_reset", True),
            strict_validation=cfg.get("strict_validation", True),
            allow_invalid_actions=cfg.get("allow_invalid_actions", False),
            debug_level=cfg.get("debug_level", "standard"),
        )


class DatasetConfig(eqx.Module):
    """Dataset-specific settings and constraints.

    This config contains all dataset-related settings including grid constraints,
    color limits, task sampling, and dataset identification.
    """

    # Dataset identification
    dataset_name: str = "arc-agi-1"
    dataset_path: str = ""
    dataset_repo: str = ""

    # Dataset-specific grid constraints
    max_grid_height: int = 30
    max_grid_width: int = 30
    min_grid_height: int = 3
    min_grid_width: int = 3

    # Color constraints
    max_colors: int = 10
    background_color: int = -1

    # Task Configuration
    max_train_pairs: int = 10
    max_test_pairs: int = 3

    # Task sampling parameters
    task_split: str = "train"
    shuffle_tasks: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate dataset configuration and return tuple of errors."""
        errors = []

        try:
            # Validate dataset name
            if not self.dataset_name.strip():
                errors.append("dataset_name cannot be empty")

            # Validate dataset path
            validate_path_string(self.dataset_path, "dataset_path")

            # Validate grid dimensions
            validate_positive_int(self.max_grid_height, "max_grid_height")
            validate_positive_int(self.max_grid_width, "max_grid_width")
            validate_positive_int(self.min_grid_height, "min_grid_height")
            validate_positive_int(self.min_grid_width, "min_grid_width")

            # Validate task pair counts
            validate_positive_int(self.max_train_pairs, "max_train_pairs")
            validate_positive_int(self.max_test_pairs, "max_test_pairs")

            if self.max_train_pairs > 20:
                logger.warning(f"max_train_pairs is very large: {self.max_train_pairs}")
            if self.max_test_pairs > 5:
                logger.warning(f"max_test_pairs is very large: {self.max_test_pairs}")

            # Validate reasonable bounds
            if self.max_grid_height > 200:
                logger.warning(f"max_grid_height is very large: {self.max_grid_height}")
            if self.max_grid_width > 200:
                logger.warning(f"max_grid_width is very large: {self.max_grid_width}")

            # Validate color constraints
            validate_positive_int(self.max_colors, "max_colors")

            # Validate background_color: -1 is valid for padding, 0-9 are valid ARC colors
            if not isinstance(self.background_color, int):
                errors.append(
                    f"background_color must be an integer, got {type(self.background_color).__name__}"
                )
            elif self.background_color < -1:
                errors.append(
                    f"background_color must be >= -1 (for padding) or a valid color index, got {self.background_color}"
                )

            if self.max_colors < 2:
                errors.append("max_colors must be at least 2")
            if self.max_colors > 50:
                logger.warning(f"max_colors is very large: {self.max_colors}")

            # Validate task split
            valid_splits = [
                "train",
                "eval",
                "test",
                "all",
                "training",
                "evaluation",
                "corpus",
            ]
            validate_string_choice(self.task_split, "task_split", valid_splits)

            # Cross-field validation
            if self.max_grid_height < self.min_grid_height:
                errors.append(
                    f"max_grid_height ({self.max_grid_height}) < min_grid_height ({self.min_grid_height})"
                )

            if self.max_grid_width < self.min_grid_width:
                errors.append(
                    f"max_grid_width ({self.max_grid_width}) < min_grid_width ({self.min_grid_width})"
                )

            # Validate background_color against max_colors (but allow -1 for padding)
            if self.background_color >= 0 and self.background_color >= self.max_colors:
                errors.append(
                    f"background_color ({self.background_color}) must be < max_colors ({self.max_colors}) when >= 0"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"DatasetConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DatasetConfig:
        """Create dataset config from Hydra DictConfig."""
        return cls(
            dataset_name=cfg.get("dataset_name", "arc-agi-1"),
            dataset_path=cfg.get("dataset_path", ""),
            dataset_repo=cfg.get("dataset_repo", ""),
            max_grid_height=cfg.get("max_grid_height", 30),
            max_grid_width=cfg.get("max_grid_width", 30),
            min_grid_height=cfg.get("min_grid_height", 3),
            min_grid_width=cfg.get("min_grid_width", 3),
            max_colors=cfg.get("max_colors", 10),
            background_color=cfg.get("background_color", -1),
            task_split=cfg.get("task_split", "train"),
            max_train_pairs=cfg.get("max_train_pairs", 10),
            max_test_pairs=cfg.get("max_test_pairs", 3),
            shuffle_tasks=cfg.get("shuffle_tasks", True),
        )


class VisualizationConfig(eqx.Module):
    """All visualization and rendering settings.

    This config contains everything related to visual output, rendering,
    and visualization behavior. No logging or storage settings here.
    """

    # Core settings
    enabled: bool = True
    level: Literal["off", "minimal", "standard", "verbose", "full"] = "standard"

    # Output formats and quality - handled in from_hydra
    output_formats: tuple[str, ...] = ("svg",)  # Changed from List[str] to tuple
    image_quality: Literal["low", "medium", "high"] = "high"

    def __init__(self, **kwargs):
        """Initialize with automatic list-to-tuple conversion."""
        output_formats = kwargs.get("output_formats", ("svg",))
        if isinstance(output_formats, str):
            output_formats = (output_formats,)
        elif hasattr(output_formats, '__iter__') and not isinstance(output_formats, (str, tuple)):
            # Handle ListConfig and other iterable types
            output_formats = tuple(output_formats) if output_formats else ("svg",)
        elif not isinstance(output_formats, tuple):
            output_formats = ("svg",)

        # Set all fields
        self.enabled = kwargs.get("enabled", True)
        self.level = kwargs.get("level", "standard")
        self.output_formats = output_formats
        self.image_quality = kwargs.get("image_quality", "high")
        self.show_coordinates = kwargs.get("show_coordinates", False)
        self.show_operation_names = kwargs.get("show_operation_names", True)
        self.highlight_changes = kwargs.get("highlight_changes", True)
        self.include_metrics = kwargs.get("include_metrics", True)
        self.color_scheme = kwargs.get("color_scheme", "default")
        self.visualize_episodes = kwargs.get("visualize_episodes", True)
        self.episode_summaries = kwargs.get("episode_summaries", True)
        self.step_visualizations = kwargs.get("step_visualizations", True)
        self.enable_comparisons = kwargs.get("enable_comparisons", True)
        self.save_intermediate_states = kwargs.get("save_intermediate_states", False)
        self.lazy_loading = kwargs.get("lazy_loading", True)
        self.max_memory_mb = kwargs.get("max_memory_mb", 500)

    # Display settings
    show_coordinates: bool = False
    show_operation_names: bool = True
    highlight_changes: bool = True
    include_metrics: bool = True
    color_scheme: Literal["default", "high_contrast", "colorblind"] = "default"

    # Episode visualization
    visualize_episodes: bool = True
    episode_summaries: bool = True
    step_visualizations: bool = True

    # Comparison and analysis
    enable_comparisons: bool = True
    save_intermediate_states: bool = False

    # Memory and performance
    lazy_loading: bool = True
    max_memory_mb: int = 500  # Standardized naming: max_* pattern

    def validate(self) -> tuple[str, ...]:
        """Validate visualization configuration and return tuple of errors."""
        errors = []

        try:
            # Validate level choices
            valid_levels = ("off", "minimal", "standard", "verbose", "full")
            validate_string_choice(self.level, "level", valid_levels)

            valid_quality = ("low", "medium", "high")
            validate_string_choice(self.image_quality, "image_quality", valid_quality)

            valid_schemes = ("default", "high_contrast", "colorblind")
            validate_string_choice(self.color_scheme, "color_scheme", valid_schemes)

            # Validate numeric fields
            validate_positive_int(self.max_memory_mb, "max_memory_mb")

            if self.max_memory_mb > 5000:
                logger.warning(f"max_memory_mb is very large: {self.max_memory_mb}")

            # Validate output formats
            valid_formats = ("svg", "png", "html", "console")
            if hasattr(self.output_formats, "__iter__"):
                for fmt in self.output_formats:
                    if fmt not in valid_formats:
                        errors.append(
                            f"Invalid output format: {fmt}. Valid formats: {valid_formats}"
                        )

            # Cross-field validation warnings
            if not self.enabled and self.level != "off":
                logger.warning("Visualization disabled but level is not 'off'")

            if self.level == "off" and self.enabled:
                logger.warning("Visualization level is 'off' but enabled=True")

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"VisualizationConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> VisualizationConfig:
        """Create visualization config from Hydra DictConfig."""
        output_formats = cfg.get("output_formats", ["svg"])
        if isinstance(output_formats, str):
            output_formats = (output_formats,)
        elif hasattr(output_formats, '__iter__') and not isinstance(output_formats, (str, tuple)):
            # Handle ListConfig and other iterable types
            output_formats = tuple(output_formats) if output_formats else ("svg",)
        elif not isinstance(output_formats, tuple):
            output_formats = ("svg",)

        return cls(
            output_formats=output_formats,  # Pass as keyword argument
            enabled=cfg.get("enabled", True),
            level=cfg.get("level", "standard"),
            image_quality=cfg.get("image_quality", "high"),
            show_coordinates=cfg.get("show_coordinates", False),
            show_operation_names=cfg.get("show_operation_names", True),
            highlight_changes=cfg.get("highlight_changes", True),
            include_metrics=cfg.get("include_metrics", True),
            color_scheme=cfg.get("color_scheme", "default"),
            visualize_episodes=cfg.get("visualize_episodes", True),
            episode_summaries=cfg.get("episode_summaries", True),
            step_visualizations=cfg.get("step_visualizations", True),
            enable_comparisons=cfg.get("enable_comparisons", True),
            save_intermediate_states=cfg.get("save_intermediate_states", False),
            lazy_loading=cfg.get("lazy_loading", True),
            max_memory_mb=cfg.get("memory_limit_mb", 500),  # Support legacy name
        )


class StorageConfig(eqx.Module):
    """All storage, output, and file management settings.

    This config contains everything related to file storage, output directories,
    cleanup policies, and file organization. All output paths are managed here.
    """

    # Base storage configuration
    policy: Literal["none", "minimal", "standard", "research"] = "standard"
    base_output_dir: str = "outputs"
    run_name: str | None = None

    # Output directories for different types of content
    episodes_dir: str = "episodes"
    debug_dir: str = "debug"
    visualization_dir: str = "visualizations"
    logs_dir: str = "logs"

    # Storage limits
    max_episodes_per_run: int = 100
    max_storage_gb: float = 5.0

    # Cleanup settings
    cleanup_policy: Literal["none", "size_based", "oldest_first", "manual"] = (
        "size_based"
    )
    cleanup_frequency: Literal["never", "after_run", "daily", "manual"] = "after_run"
    keep_recent_episodes: int = 10

    # File organization
    create_run_subdirs: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    compress_old_files: bool = True
    clear_output_on_start: bool = True

    # Safety settings
    auto_cleanup: bool = True
    warn_on_storage_limit: bool = True
    fail_on_storage_full: bool = False

    def validate(self) -> tuple[str, ...]:
        """Validate storage configuration and return tuple of errors."""
        errors = []

        try:
            # Validate policy choices
            valid_policies = ("none", "minimal", "standard", "research")
            validate_string_choice(self.policy, "policy", valid_policies)

            valid_cleanup_policies = ("none", "size_based", "oldest_first", "manual")
            validate_string_choice(
                self.cleanup_policy, "cleanup_policy", valid_cleanup_policies
            )

            valid_cleanup_freq = ("never", "after_run", "daily", "manual")
            validate_string_choice(
                self.cleanup_frequency, "cleanup_frequency", valid_cleanup_freq
            )

            # Validate output directory paths
            validate_path_string(self.base_output_dir, "base_output_dir")
            validate_path_string(self.episodes_dir, "episodes_dir")
            validate_path_string(self.debug_dir, "debug_dir")
            validate_path_string(self.visualization_dir, "visualization_dir")
            validate_path_string(self.logs_dir, "logs_dir")

            # Validate numeric fields
            validate_positive_int(self.max_episodes_per_run, "max_episodes_per_run")
            validate_positive_int(self.keep_recent_episodes, "keep_recent_episodes")

            if not isinstance(self.max_storage_gb, (int, float)):
                msg = f"max_storage_gb must be a number, got {type(self.max_storage_gb).__name__}"
                errors.append(msg)
            elif self.max_storage_gb <= 0:
                errors.append("max_storage_gb must be positive")

            # Validate reasonable bounds
            if self.max_episodes_per_run > 10000:
                logger.warning(
                    f"max_episodes_per_run is very large: {self.max_episodes_per_run}"
                )

            if self.max_storage_gb > 100:
                logger.warning(f"max_storage_gb is very large: {self.max_storage_gb}")

            # Cross-field validation
            if self.keep_recent_episodes > self.max_episodes_per_run:
                logger.warning(
                    f"keep_recent_episodes ({self.keep_recent_episodes}) > max_episodes_per_run ({self.max_episodes_per_run})"
                )

            if self.cleanup_policy == "none" and self.auto_cleanup:
                logger.warning("auto_cleanup=True but cleanup_policy='none'")

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"StorageConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> StorageConfig:
        """Create storage config from Hydra DictConfig."""
        return cls(
            policy=cfg.get("policy", "standard"),
            base_output_dir=cfg.get("base_output_dir", "outputs"),
            run_name=cfg.get("run_name"),
            episodes_dir=cfg.get("episodes_dir", "episodes"),
            debug_dir=cfg.get("debug_dir", "debug"),
            visualization_dir=cfg.get("visualization_dir", "visualizations"),
            logs_dir=cfg.get("logs_dir", "logs"),
            max_episodes_per_run=cfg.get("max_episodes_per_run", 100),
            max_storage_gb=cfg.get("max_storage_gb", 5.0),
            cleanup_policy=cfg.get("cleanup_policy", "size_based"),
            cleanup_frequency=cfg.get("cleanup_frequency", "after_run"),
            keep_recent_episodes=cfg.get("keep_recent_episodes", 10),
            create_run_subdirs=cfg.get("create_run_subdirs", True),
            timestamp_format=cfg.get("timestamp_format", "%Y%m%d_%H%M%S"),
            compress_old_files=cfg.get("compress_old_files", True),
            clear_output_on_start=cfg.get("clear_output_on_start", True),
            auto_cleanup=cfg.get("auto_cleanup", True),
            warn_on_storage_limit=cfg.get("warn_on_storage_limit", True),
            fail_on_storage_full=cfg.get("fail_on_storage_full", False),
        )


class LoggingConfig(eqx.Module):
    """All logging behavior and formats.

    This config contains everything related to logging: what to log,
    how to format it, where to write it, and performance settings.
    """

    # Core logging settings
    structured_logging: bool = True
    log_format: Literal["json", "text", "structured"] = "json"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    compression: bool = True
    include_full_states: bool = False

    # What to log (specific content flags)
    log_operations: bool = False
    log_grid_changes: bool = False
    log_rewards: bool = False
    log_episode_start: bool = True
    log_episode_end: bool = True
    log_key_moments: bool = True

    # Logging frequency and timing
    log_frequency: int = 10  # Log every N steps

    # Batched logging settings
    batched_logging_enabled: bool = False
    sampling_enabled: bool = True
    num_samples: int = 3
    sample_frequency: int = 50
    
    # Aggregated metrics selection
    log_aggregated_rewards: bool = True
    log_aggregated_similarity: bool = True
    log_loss_metrics: bool = True
    log_gradient_norms: bool = True
    log_episode_lengths: bool = True
    log_success_rates: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate logging configuration and return tuple of errors."""
        errors = []

        try:
            # Validate format choices
            valid_formats = ("json", "text", "structured")
            validate_string_choice(self.log_format, "log_format", valid_formats)

            valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
            validate_string_choice(self.log_level, "log_level", valid_levels)

            # Validate numeric fields
            validate_positive_int(self.log_frequency, "log_frequency")

            if self.log_frequency > 1000:
                logger.warning(f"log_frequency is very high: {self.log_frequency}")

            # Validate batched logging parameters
            if self.batched_logging_enabled:
                validate_positive_int(self.num_samples, "num_samples")
                validate_positive_int(self.sample_frequency, "sample_frequency")
                
                if self.num_samples > 1000:
                    logger.warning(f"num_samples is very large: {self.num_samples}")
                
                if self.sample_frequency <= 0:
                    errors.append("sample_frequency must be positive")
                
                if self.num_samples <= 0:
                    errors.append("num_samples must be positive")

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"LoggingConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> LoggingConfig:
        """Create logging config from Hydra DictConfig."""
        return cls(
            structured_logging=cfg.get("structured_logging", True),
            log_format=cfg.get("log_format", "json"),
            log_level=cfg.get("log_level", "INFO"),
            compression=cfg.get("compression", True),
            include_full_states=cfg.get("include_full_states", False),
            log_operations=cfg.get("log_operations", False),
            log_grid_changes=cfg.get("log_grid_changes", False),
            log_rewards=cfg.get("log_rewards", False),
            log_episode_start=cfg.get("log_episode_start", True),
            log_episode_end=cfg.get("log_episode_end", True),
            log_key_moments=cfg.get("log_key_moments", True),
            log_frequency=cfg.get("log_frequency", 10),
            # Batched logging settings
            batched_logging_enabled=cfg.get("batched_logging_enabled", False),
            sampling_enabled=cfg.get("sampling_enabled", True),
            num_samples=cfg.get("num_samples", 3),
            sample_frequency=cfg.get("sample_frequency", 50),
            # Aggregated metrics selection
            log_aggregated_rewards=cfg.get("log_aggregated_rewards", True),
            log_aggregated_similarity=cfg.get("log_aggregated_similarity", True),
            log_loss_metrics=cfg.get("log_loss_metrics", True),
            log_gradient_norms=cfg.get("log_gradient_norms", True),
            log_episode_lengths=cfg.get("log_episode_lengths", True),
            log_success_rates=cfg.get("log_success_rates", True),
        )


class WandbConfig(eqx.Module):
    """Weights & Biases integration settings.

    This config contains everything related to W&B logging and tracking.
    No local logging or storage settings here.
    """

    # Core wandb settings
    enabled: bool = False
    project_name: str = "jaxarc-experiments"
    entity: str | None = None
    tags: tuple[str, ...] = ("jaxarc",)  # Changed from List[str] to tuple
    notes: str = "JaxARC experiment"
    group: str | None = None
    job_type: str = "training"

    # Logging settings
    log_frequency: int = 10
    image_format: Literal["png", "svg", "both"] = "png"
    max_image_size: tuple[int, int] = (800, 600)  # Changed Int to int

    # Advanced options
    log_gradients: bool = False
    log_model_topology: bool = False
    log_system_metrics: bool = True

    # Error handling
    offline_mode: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Storage
    save_code: bool = True
    save_config: bool = True

    def __init__(self, **kwargs):
        """Initialize with automatic list-to-tuple conversion."""
        tags = kwargs.get("tags", ("jaxarc",))
        if isinstance(tags, str):
            tags = (tags,)
        elif hasattr(tags, '__iter__') and not isinstance(tags, (str, tuple)):
            # Handle ListConfig and other iterable types
            tags = tuple(tags)
        elif not isinstance(tags, tuple):
            tags = ("jaxarc",)

        max_image_size = kwargs.get("max_image_size", (800, 600))
        if hasattr(max_image_size, '__iter__') and not isinstance(max_image_size, (str, tuple)):
            # Handle ListConfig and other iterable types
            max_image_size = tuple(max_image_size)
        elif not isinstance(max_image_size, tuple):
            max_image_size = (800, 600)

        # Set all fields
        self.enabled = kwargs.get("enabled", False)
        self.project_name = kwargs.get("project_name", "jaxarc-experiments")
        self.entity = kwargs.get("entity", None)
        self.tags = tags
        self.notes = kwargs.get("notes", "JaxARC experiment")
        self.group = kwargs.get("group", None)
        self.job_type = kwargs.get("job_type", "training")
        self.log_frequency = kwargs.get("log_frequency", 10)
        self.image_format = kwargs.get("image_format", "png")
        self.max_image_size = max_image_size
        self.log_gradients = kwargs.get("log_gradients", False)
        self.log_model_topology = kwargs.get("log_model_topology", False)
        self.log_system_metrics = kwargs.get("log_system_metrics", True)
        self.offline_mode = kwargs.get("offline_mode", False)
        self.retry_attempts = kwargs.get("retry_attempts", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
        self.save_code = kwargs.get("save_code", True)
        self.save_config = kwargs.get("save_config", True)

    def validate(self) -> tuple[str, ...]:
        """Validate wandb configuration and return tuple of errors."""
        errors = []

        try:
            # Validate format choices
            valid_formats = ("png", "svg", "both")
            validate_string_choice(self.image_format, "image_format", valid_formats)

            # Validate project name
            if not self.project_name.strip():
                errors.append("project_name cannot be empty")

            # Validate numeric fields
            validate_positive_int(self.log_frequency, "log_frequency")
            validate_positive_int(self.retry_attempts, "retry_attempts")

            if not isinstance(self.retry_delay, (int, float)):
                msg = f"retry_delay must be a number, got {type(self.retry_delay).__name__}"
                errors.append(msg)
            elif self.retry_delay < 0:
                errors.append("retry_delay must be non-negative")

            # Validate image size
            if len(self.max_image_size) != 2:
                errors.append("max_image_size must be a tuple of (width, height)")
            else:
                width, height = self.max_image_size
                if width <= 0 or height <= 0:
                    errors.append("max_image_size dimensions must be positive")

            # Validate reasonable bounds
            if self.log_frequency > 1000:
                logger.warning(f"log_frequency is very high: {self.log_frequency}")

            if self.retry_attempts > 10:
                logger.warning(f"retry_attempts is very high: {self.retry_attempts}")

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(f"WandbConfig must be hashable for JAX compatibility: {e}")

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> WandbConfig:
        """Create wandb config from Hydra DictConfig."""
        tags = cfg.get("tags", ["jaxarc"])
        if isinstance(tags, str):
            tags = (tags,)
        elif hasattr(tags, '__iter__') and not isinstance(tags, (str, tuple)):
            # Handle ListConfig and other iterable types
            tags = tuple(tags)
        elif not isinstance(tags, tuple):
            tags = ("jaxarc",)

        max_image_size = cfg.get("max_image_size", [800, 600])
        if hasattr(max_image_size, '__iter__') and not isinstance(max_image_size, (str, tuple)):
            # Handle ListConfig and other iterable types
            max_image_size = tuple(max_image_size)
        elif not isinstance(max_image_size, tuple):
            max_image_size = (800, 600)

        return cls(
            tags=tags,  # Pass as keyword argument
            enabled=cfg.get("enabled", False),
            project_name=cfg.get("project_name", "jaxarc-experiments"),
            entity=cfg.get("entity"),
            notes=cfg.get("notes", "JaxARC experiment"),
            group=cfg.get("group"),
            job_type=cfg.get("job_type", "training"),
            log_frequency=cfg.get("log_frequency", 10),
            image_format=cfg.get("image_format", "png"),
            max_image_size=max_image_size,
            log_gradients=cfg.get("log_gradients", False),
            log_model_topology=cfg.get("log_model_topology", False),
            log_system_metrics=cfg.get("log_system_metrics", True),
            offline_mode=cfg.get("offline_mode", False),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_delay=cfg.get("retry_delay", 1.0),
            save_code=cfg.get("save_code", True),
            save_config=cfg.get("save_config", True),
        )


class RewardConfig(eqx.Module):
    """Configuration for reward calculation.

    This config contains all settings related to reward computation,
    penalties, bonuses, and reward shaping with mode-aware enhancements.
    """

    # Basic reward settings
    reward_on_submit_only: bool = True
    step_penalty: float = -0.01
    success_bonus: float = 10.0
    similarity_weight: float = 1.0

    # Additional reward shaping
    progress_bonus: float = 0.0
    invalid_action_penalty: float = -0.1

    # Enhanced mode-specific reward settings
    control_operation_penalty: float = -0.01  # Penalty for control operations
    pair_switching_bonus: float = 0.0  # Bonus for beneficial pair switching

    # Mode-specific reward structures
    training_similarity_weight: float = 1.0  # Similarity weight in training mode
    evaluation_similarity_weight: float = (
        0.0  # Similarity weight in evaluation mode (masked)
    )

    # Pair-type specific bonuses
    demo_completion_bonus: float = 1.0  # Bonus for completing demonstration pairs
    test_completion_bonus: float = 5.0  # Bonus for completing test pairs

    # Advanced reward shaping
    consecutive_success_bonus: float = (
        0.0  # Bonus for solving multiple pairs in sequence
    )
    efficiency_bonus_threshold: int = 50  # Step threshold for efficiency bonus
    efficiency_bonus: float = 1.0  # Bonus for solving pairs efficiently

    def validate(self) -> tuple[str, ...]:
        """Validate reward configuration and return tuple of errors."""
        errors = []

        try:
            # Validate basic numeric fields with reasonable ranges
            validate_float_range(self.step_penalty, "step_penalty", -10.0, 1.0)
            validate_float_range(self.success_bonus, "success_bonus", -100.0, 1000.0)
            validate_float_range(self.similarity_weight, "similarity_weight", 0.0, 10.0)
            validate_float_range(self.progress_bonus, "progress_bonus", -10.0, 10.0)
            validate_float_range(
                self.invalid_action_penalty, "invalid_action_penalty", -10.0, 1.0
            )

            # Validate enhanced reward fields
            validate_float_range(
                self.control_operation_penalty, "control_operation_penalty", -10.0, 1.0
            )
            validate_float_range(
                self.pair_switching_bonus, "pair_switching_bonus", -10.0, 10.0
            )
            validate_float_range(
                self.training_similarity_weight, "training_similarity_weight", 0.0, 10.0
            )
            validate_float_range(
                self.evaluation_similarity_weight,
                "evaluation_similarity_weight",
                0.0,
                10.0,
            )
            validate_float_range(
                self.demo_completion_bonus, "demo_completion_bonus", -100.0, 100.0
            )
            validate_float_range(
                self.test_completion_bonus, "test_completion_bonus", -100.0, 100.0
            )
            validate_float_range(
                self.consecutive_success_bonus, "consecutive_success_bonus", -10.0, 10.0
            )
            validate_non_negative_int(
                self.efficiency_bonus_threshold, "efficiency_bonus_threshold"
            )
            validate_float_range(self.efficiency_bonus, "efficiency_bonus", -10.0, 10.0)

            # Issue warnings for potentially problematic configurations
            if self.reward_on_submit_only and self.progress_bonus != 0.0:
                logger.warning(
                    "progress_bonus is ignored when reward_on_submit_only=True"
                )

            if self.step_penalty > 0:
                logger.warning(
                    f"step_penalty should typically be negative or zero for proper learning, got {self.step_penalty}"
                )

            if self.success_bonus < 0:
                logger.warning(
                    f"success_bonus should typically be positive for proper learning, got {self.success_bonus}"
                )

            if self.invalid_action_penalty > 0:
                logger.warning(
                    f"invalid_action_penalty should typically be negative or zero, got {self.invalid_action_penalty}"
                )

            if self.control_operation_penalty > 0:
                logger.warning(
                    f"control_operation_penalty should typically be negative or zero, got {self.control_operation_penalty}"
                )

            # Warn about evaluation similarity weight
            if self.evaluation_similarity_weight > 0:
                logger.warning(
                    "evaluation_similarity_weight > 0 may lead to cheating in test mode due to target masking"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"RewardConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> RewardConfig:
        """Create reward config from Hydra DictConfig."""
        return cls(
            reward_on_submit_only=cfg.get("reward_on_submit_only", True),
            step_penalty=cfg.get("step_penalty", -0.01),
            success_bonus=cfg.get("success_bonus", 10.0),
            similarity_weight=cfg.get("similarity_weight", 1.0),
            progress_bonus=cfg.get("progress_bonus", 0.0),
            invalid_action_penalty=cfg.get("invalid_action_penalty", -0.1),
            control_operation_penalty=cfg.get("control_operation_penalty", -0.01),
            pair_switching_bonus=cfg.get("pair_switching_bonus", 0.0),
            training_similarity_weight=cfg.get("training_similarity_weight", 1.0),
            evaluation_similarity_weight=cfg.get("evaluation_similarity_weight", 0.0),
            demo_completion_bonus=cfg.get("demo_completion_bonus", 1.0),
            test_completion_bonus=cfg.get("test_completion_bonus", 5.0),
            consecutive_success_bonus=cfg.get("consecutive_success_bonus", 0.0),
            efficiency_bonus_threshold=cfg.get("efficiency_bonus_threshold", 50),
            efficiency_bonus=cfg.get("efficiency_bonus", 1.0),
        )


class GridInitializationConfig(eqx.Module):
    """Configuration for diverse grid initialization strategies.
    
    This config controls how working grids are initialized in the environment,
    supporting multiple modes including demo grids, permutations, empty grids,
    and random patterns for enhanced training diversity.
    """
    
    # Initialization mode selection
    mode: Literal["demo", "permutation", "empty", "random", "mixed"] = "demo"
    
    # Probability weights for mixed mode (must sum to 1.0)
    demo_weight: float = 0.25
    permutation_weight: float = 0.25
    empty_weight: float = 0.25
    random_weight: float = 0.25
    
    # Permutation configuration
    permutation_types: tuple[str, ...] = ("rotate", "reflect", "color_remap")
    
    # Random initialization configuration
    random_density: float = 0.3  # Density for random patterns (0.0 to 1.0)
    random_pattern_type: Literal["sparse", "dense", "structured", "noise"] = "sparse"
    
    # Fallback and error handling
    enable_fallback: bool = True  # Fallback to demo mode if other modes fail
    
    def __init__(self, **kwargs):
        """Initialize with automatic list-to-tuple conversion for permutation_types."""
        permutation_types = kwargs.get("permutation_types", ("rotate", "reflect", "color_remap"))
        if isinstance(permutation_types, str):
            permutation_types = (permutation_types,)
        elif hasattr(permutation_types, '__iter__') and not isinstance(permutation_types, (str, tuple)):
            # Handle ListConfig and other iterable types
            permutation_types = tuple(permutation_types) if permutation_types else ("rotate",)
        elif not isinstance(permutation_types, tuple):
            permutation_types = ("rotate", "reflect", "color_remap")
        
        # Set all fields
        self.mode = kwargs.get("mode", "demo")
        self.demo_weight = kwargs.get("demo_weight", 0.25)
        self.permutation_weight = kwargs.get("permutation_weight", 0.25)
        self.empty_weight = kwargs.get("empty_weight", 0.25)
        self.random_weight = kwargs.get("random_weight", 0.25)
        self.permutation_types = permutation_types
        self.random_density = kwargs.get("random_density", 0.3)
        self.random_pattern_type = kwargs.get("random_pattern_type", "sparse")
        self.enable_fallback = kwargs.get("enable_fallback", True)
    
    def validate(self) -> tuple[str, ...]:
        """Validate grid initialization configuration and return tuple of errors."""
        errors = []
        
        try:
            # Validate mode choices
            valid_modes = ("demo", "permutation", "empty", "random", "mixed")
            validate_string_choice(self.mode, "mode", valid_modes)
            
            # Validate pattern type choices
            valid_pattern_types = ("sparse", "dense", "structured", "noise")
            validate_string_choice(self.random_pattern_type, "random_pattern_type", valid_pattern_types)
            
            # Validate weight ranges
            validate_float_range(self.demo_weight, "demo_weight", 0.0, 1.0)
            validate_float_range(self.permutation_weight, "permutation_weight", 0.0, 1.0)
            validate_float_range(self.empty_weight, "empty_weight", 0.0, 1.0)
            validate_float_range(self.random_weight, "random_weight", 0.0, 1.0)
            validate_float_range(self.random_density, "random_density", 0.0, 1.0)
            
            # Validate weights sum to 1.0 (with small tolerance for floating point)
            total_weight = (
                self.demo_weight + 
                self.permutation_weight + 
                self.empty_weight + 
                self.random_weight
            )
            if abs(total_weight - 1.0) > 1e-6:
                errors.append(
                    f"Initialization weights must sum to 1.0, got {total_weight:.6f} "
                    f"(demo: {self.demo_weight}, permutation: {self.permutation_weight}, "
                    f"empty: {self.empty_weight}, random: {self.random_weight})"
                )
            
            # Validate permutation types
            valid_permutation_types = ("rotate", "reflect", "color_remap", "translate")
            if hasattr(self.permutation_types, "__iter__"):
                for ptype in self.permutation_types:
                    if ptype not in valid_permutation_types:
                        errors.append(
                            f"Invalid permutation type: {ptype}. "
                            f"Valid types: {valid_permutation_types}"
                        )
            
            # Cross-field validation warnings
            if self.mode != "mixed":
                # If not in mixed mode, weights are ignored
                if any(w != 0.25 for w in [self.demo_weight, self.permutation_weight, 
                                          self.empty_weight, self.random_weight]):
                    from loguru import logger
                    logger.warning(
                        f"Mode is '{self.mode}' but weights are specified. "
                        "Weights are only used in 'mixed' mode."
                    )
            
            if self.mode == "permutation" and not self.permutation_types:
                errors.append("permutation_types cannot be empty when mode is 'permutation'")
                
        except ConfigValidationError as e:
            errors.append(str(e))
        
        return tuple(errors)
    
    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"GridInitializationConfig must be hashable for JAX compatibility: {e}"
            )
    
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'GridInitializationConfig':
        """Create grid initialization config from Hydra DictConfig."""
        permutation_types = cfg.get("permutation_types", ["rotate", "reflect", "color_remap"])
        if isinstance(permutation_types, str):
            permutation_types = (permutation_types,)
        elif hasattr(permutation_types, '__iter__') and not isinstance(permutation_types, (str, tuple)):
            # Handle ListConfig and other iterable types
            permutation_types = tuple(permutation_types) if permutation_types else ("rotate",)
        elif not isinstance(permutation_types, tuple):
            permutation_types = ("rotate", "reflect", "color_remap")
        
        return cls(
            mode=cfg.get("mode", "demo"),
            demo_weight=cfg.get("demo_weight", 0.25),
            permutation_weight=cfg.get("permutation_weight", 0.25),
            empty_weight=cfg.get("empty_weight", 0.25),
            random_weight=cfg.get("random_weight", 0.25),
            permutation_types=permutation_types,
            random_density=cfg.get("random_density", 0.3),
            random_pattern_type=cfg.get("random_pattern_type", "sparse"),
            enable_fallback=cfg.get("enable_fallback", True),
        )


class ActionConfig(eqx.Module):
    """Configuration for action space and validation.

    This config contains all settings related to action handling,
    validation, and operation constraints, including dynamic action space control.
    """

    # Selection format
    selection_format: Literal["mask", "point", "bbox"] = "mask"

    # Selection parameters
    selection_threshold: float = 0.5
    allow_partial_selection: bool = True

    # Operation parameters
    max_operations: int = 42  # Updated to include enhanced control operations (0-41)
    allowed_operations: Optional[tuple[int, ...]] = (
        None  # Changed from List[Int] to tuple
    )

    # Validation settings
    validate_actions: bool = True
    allow_invalid_actions: bool = False  # Standardized naming: allow_* pattern

    # Dynamic action space control settings
    dynamic_action_filtering: bool = False  # Enable runtime operation filtering
    context_dependent_operations: bool = (
        False  # Allow context-based operation availability
    )
    invalid_operation_policy: Literal["clip", "reject", "passthrough", "penalize"] = (
        "clip"
    )

    def __init__(self, **kwargs):
        """Initialize with automatic list-to-tuple conversion."""
        allowed_operations = kwargs.get("allowed_operations", None)
        if allowed_operations is not None:
            if hasattr(allowed_operations, '__iter__') and not isinstance(allowed_operations, (str, tuple)):
                # Handle ListConfig and other iterable types
                allowed_operations = tuple(allowed_operations) if allowed_operations else None
            elif not isinstance(allowed_operations, tuple):
                allowed_operations = tuple(allowed_operations) if allowed_operations else None

        # Set all fields
        self.selection_format = kwargs.get("selection_format", "mask")
        self.selection_threshold = kwargs.get("selection_threshold", 0.5)
        self.allow_partial_selection = kwargs.get("allow_partial_selection", True)
        self.max_operations = kwargs.get("max_operations", 42)
        self.allowed_operations = allowed_operations
        self.validate_actions = kwargs.get("validate_actions", True)
        self.allow_invalid_actions = kwargs.get("allow_invalid_actions", False)
        self.dynamic_action_filtering = kwargs.get("dynamic_action_filtering", False)
        self.context_dependent_operations = kwargs.get(
            "context_dependent_operations", False
        )
        self.invalid_operation_policy = kwargs.get("invalid_operation_policy", "clip")

    def validate(self) -> tuple[str, ...]:
        """Validate action configuration and return tuple of errors."""
        errors = []

        try:
            # Validate selection format
            valid_formats = ("mask", "point", "bbox")
            validate_string_choice(
                self.selection_format, "selection_format", valid_formats
            )

            # Validate selection threshold
            validate_float_range(
                self.selection_threshold, "selection_threshold", 0.0, 1.0
            )

            # Validate operation parameters
            validate_positive_int(self.max_operations, "max_operations")
            if self.max_operations > 100:
                logger.warning(f"max_operations is very large: {self.max_operations}")

            # Validate allowed operations tuple if provided
            if self.allowed_operations is not None:
                if not isinstance(self.allowed_operations, tuple):
                    errors.append("allowed_operations must be a tuple or None")
                elif not self.allowed_operations:
                    errors.append("allowed_operations cannot be empty if specified")
                else:
                    for i, op in enumerate(self.allowed_operations):
                        if not isinstance(op, int):
                            errors.append(f"allowed_operations[{i}] must be an integer")
                        elif not 0 <= op < self.max_operations:
                            errors.append(
                                f"allowed_operations[{i}] must be in range [0, {self.max_operations})"
                            )

                    # Check for duplicates
                    if len(set(self.allowed_operations)) != len(
                        self.allowed_operations
                    ):
                        duplicates = [
                            op
                            for op in set(self.allowed_operations)
                            if self.allowed_operations.count(op) > 1
                        ]
                        errors.append(
                            f"allowed_operations contains duplicate operations: {duplicates}"
                        )

            # Validate dynamic action space control settings
            valid_policies = ("clip", "reject", "passthrough", "penalize")
            validate_string_choice(
                self.invalid_operation_policy,
                "invalid_operation_policy",
                valid_policies,
            )

            # Cross-field validation warnings
            if self.selection_format != "mask" and self.allow_partial_selection:
                logger.warning(
                    f"allow_partial_selection is ignored for selection_format='{self.selection_format}'"
                )

            if not self.validate_actions and not self.allow_invalid_actions:
                logger.warning(
                    "allow_invalid_actions has no effect when validate_actions=False"
                )

            if not self.dynamic_action_filtering and self.context_dependent_operations:
                logger.warning(
                    "context_dependent_operations has no effect when dynamic_action_filtering=False"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return errors

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"ActionConfig must be hashable for JAX compatibility: {e}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> ActionConfig:
        """Create action config from Hydra DictConfig."""
        allowed_ops = cfg.get("allowed_operations")
        if allowed_ops is not None:
            if hasattr(allowed_ops, '__iter__') and not isinstance(allowed_ops, (str, tuple)):
                # Handle ListConfig and other iterable types
                allowed_ops = tuple(allowed_ops) if allowed_ops else None
            elif not isinstance(allowed_ops, tuple):
                allowed_ops = tuple(allowed_ops) if allowed_ops else None

        return cls(
            allowed_operations=allowed_ops,  # Pass as keyword argument
            selection_format=cfg.get("selection_format", "mask"),
            selection_threshold=cfg.get("selection_threshold", 0.5),
            allow_partial_selection=cfg.get("allow_partial_selection", True),
            max_operations=cfg.get(
                "num_operations", 42
            ),  # Updated to include enhanced operations
            validate_actions=cfg.get("validate_actions", True),
            allow_invalid_actions=not cfg.get(
                "clip_invalid_actions", True
            ),  # Map from legacy name with inverted logic
            dynamic_action_filtering=cfg.get("dynamic_action_filtering", False),
            context_dependent_operations=cfg.get("context_dependent_operations", False),
            invalid_operation_policy=cfg.get("invalid_operation_policy", "clip"),
        )


class JaxArcConfig(eqx.Module):
    """Unified configuration for JaxARC using Equinox.

    This is the main configuration container that unifies all configuration aspects
    into a single, typed, JAX-compatible configuration object. It eliminates the
    dual configuration pattern and provides a single source of truth.

    All configuration parameters are organized into logical groups with clear
    separation of concerns. The configuration supports YAML serialization,
    Hydra integration, and comprehensive validation.
    """

    # Core configuration components
    environment: EnvironmentConfig
    dataset: DatasetConfig
    action: ActionConfig
    reward: RewardConfig
    grid_initialization: GridInitializationConfig
    visualization: VisualizationConfig
    storage: StorageConfig
    logging: LoggingConfig
    wandb: WandbConfig
    episode: ArcEpisodeConfig
    history: HistoryConfig

    def __init__(
        self,
        environment: Optional[EnvironmentConfig] = None,
        dataset: Optional[DatasetConfig] = None,
        action: Optional[ActionConfig] = None,
        reward: Optional[RewardConfig] = None,
        grid_initialization: Optional[GridInitializationConfig] = None,
        visualization: Optional[VisualizationConfig] = None,
        storage: Optional[StorageConfig] = None,
        logging: Optional[LoggingConfig] = None,
        wandb: Optional[WandbConfig] = None,
        episode: Optional[ArcEpisodeConfig] = None,
        history: Optional[HistoryConfig] = None,
    ):
        """Initialize unified configuration with optional component overrides."""
        self.environment = environment or EnvironmentConfig()
        self.dataset = dataset or DatasetConfig()
        self.action = action or ActionConfig()
        self.reward = reward or RewardConfig()
        self.grid_initialization = grid_initialization or GridInitializationConfig()
        self.visualization = visualization or VisualizationConfig.from_hydra(
            DictConfig({})
        )
        self.storage = storage or StorageConfig()
        self.logging = logging or LoggingConfig()
        self.wandb = wandb or WandbConfig.from_hydra(DictConfig({}))
        self.episode = episode or ArcEpisodeConfig()
        self.history = history or HistoryConfig()

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            raise ValueError(
                f"JaxArcConfig must be hashable for JAX compatibility: {e}"
            )

    def validate(self) -> tuple[str, ...]:
        """Comprehensive validation method that checks cross-config consistency.

        Returns:
            Tuple of validation error messages. Empty tuple means validation passed.
        """
        all_errors = []

        # Validate each configuration component
        all_errors.extend(self.environment.validate())
        all_errors.extend(self.dataset.validate())
        all_errors.extend(self.action.validate())
        all_errors.extend(self.reward.validate())
        all_errors.extend(self.grid_initialization.validate())
        all_errors.extend(self.visualization.validate())
        all_errors.extend(self.storage.validate())
        all_errors.extend(self.logging.validate())
        all_errors.extend(self.wandb.validate())
        all_errors.extend(self.episode.validate())

        # Validate history config (HistoryConfig uses @chex.dataclass validation in __post_init__)
        # The validation happens automatically during construction, so no explicit validate() call needed

        # Cross-configuration validation
        cross_validation_errors = self._validate_cross_config_consistency()
        all_errors.extend(cross_validation_errors)

        return tuple(all_errors)

    def _validate_cross_config_consistency(self) -> tuple[str, ...]:
        """Validate consistency between different configuration sections."""
        errors = []
        warnings = []

        try:
            # 1. Debug level consistency validation
            self._validate_debug_level_consistency(errors, warnings)

            # 2. Storage and output consistency validation
            self._validate_storage_consistency(errors, warnings)

            # 3. Performance and resource consistency validation
            self._validate_performance_consistency(errors, warnings)

            # 4. WandB integration consistency validation
            self._validate_wandb_consistency(errors, warnings)

            # 5. Action space and environment consistency validation
            self._validate_action_environment_consistency(errors, warnings)

            # 6. Reward and learning consistency validation
            self._validate_reward_consistency(errors, warnings)

            # 7. Dataset and environment consistency validation
            self._validate_dataset_consistency(errors, warnings)

            # 8. Logging and storage consistency validation
            self._validate_logging_consistency(errors, warnings)

            # Log all warnings
            for warning in warnings:
                logger.warning(warning)

        except Exception as e:
            errors.append(f"Cross-configuration validation error: {e}")

        return tuple(errors)

    def _validate_debug_level_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate debug level consistency across configurations."""
        debug_level = self.environment.debug_level

        # Debug level vs visualization consistency
        if debug_level == "off":
            if self.visualization.enabled and self.visualization.level != "off":
                warnings.append(
                    "Debug level is 'off' but visualization is enabled - consider disabling visualization for better performance"
                )
            if (
                self.logging.log_operations
                or self.logging.log_grid_changes
                or self.logging.log_rewards
            ):
                warnings.append(
                    "Debug level is 'off' but detailed logging is enabled - consider reducing log level"
                )

        # Debug level vs storage policy consistency
        expected_storage_policy = self.environment.computed_storage_policy
        if self.storage.policy != expected_storage_policy:
            warnings.append(
                f"Storage policy '{self.storage.policy}' doesn't match debug level '{debug_level}' (expected '{expected_storage_policy}')"
            )

        # Debug level vs visualization level consistency
        expected_viz_level = self.environment.computed_visualization_level
        if (
            self.visualization.enabled
            and self.visualization.level != expected_viz_level
        ):
            warnings.append(
                f"Visualization level '{self.visualization.level}' doesn't match debug level '{debug_level}' (expected '{expected_viz_level}')"
            )

    def _validate_storage_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate storage configuration consistency."""
        # Storage policy vs features consistency
        if self.storage.policy == "none":
            if self.visualization.save_intermediate_states:
                errors.append(
                    "Cannot save intermediate states when storage policy is 'none'"
                )
            if self.logging.structured_logging:
                warnings.append(
                    "Structured logging enabled but storage policy is 'none' - logs may not be persisted"
                )
            if self.storage.max_episodes_per_run > 0:
                warnings.append(
                    "Storage policy is 'none' but max_episodes_per_run > 0 - episodes won't be saved"
                )

        # Storage limits vs debug level consistency
        if self.environment.debug_level in ["verbose", "research"]:
            if self.storage.max_storage_gb < 5.0:
                warnings.append(
                    f"Debug level '{self.environment.debug_level}' may require more storage than {self.storage.max_storage_gb}GB"
                )

        # Directory configuration consistency
        if self.visualization.enabled and self.storage.policy != "none":
            if not self.storage.visualization_dir.strip():
                errors.append("Visualization enabled but visualization_dir is empty")

        if self.logging.structured_logging and self.storage.policy != "none":
            if not self.storage.logs_dir.strip():
                errors.append("Structured logging enabled but logs_dir is empty")

    def _validate_performance_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate performance-related configuration consistency."""
        # Memory limits vs visualization level
        if self.visualization.max_memory_mb < 100 and self.visualization.level in [
            "verbose",
            "full",
        ]:
            warnings.append(
                "Low memory limit with verbose visualization may cause performance issues"
            )

        # Episode count vs storage limits
        total_expected_storage = self.storage.max_episodes_per_run * (
            self.visualization.max_memory_mb / 1000
            if self.visualization.save_intermediate_states
            else 0.1
        )
        if total_expected_storage > self.storage.max_storage_gb:
            warnings.append(
                f"Expected storage usage ({total_expected_storage:.1f}GB) exceeds limit ({self.storage.max_storage_gb}GB)"
            )

        # Logging frequency vs performance
        if self.logging.log_frequency == 1 and self.environment.max_episode_steps > 100:
            warnings.append(
                "Logging every step with long episodes may impact performance"
            )



    def _validate_wandb_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate WandB integration consistency."""
        if self.wandb.enabled:
            # Required fields validation
            if not self.wandb.project_name.strip():
                errors.append("WandB enabled but project_name is empty")

            # Logging level consistency
            if self.logging.log_level == "ERROR":
                warnings.append(
                    "WandB enabled but log level is ERROR - may miss important metrics"
                )

            # Offline mode consistency
            if self.wandb.offline_mode and self.wandb.log_frequency < 10:
                warnings.append(
                    "WandB offline mode with frequent logging may create large local files"
                )

            # Image logging consistency
            if (
                self.wandb.image_format in ["png", "both"]
                and not self.visualization.enabled
            ):
                warnings.append(
                    "WandB image logging enabled but visualization is disabled"
                )

    def _validate_action_environment_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate action space and environment consistency."""
        # Action space vs episode length
        if self.action.max_operations > 50 and self.environment.max_episode_steps < 20:
            warnings.append(
                "Many operations available but few episode steps - may not explore action space effectively"
            )

        # Action validation vs environment strictness
        if not self.action.validate_actions and self.environment.strict_validation:
            warnings.append(
                "Environment strict validation enabled but action validation disabled - may cause inconsistencies"
            )

        # Invalid actions handling consistency
        if self.action.allow_invalid_actions and self.environment.strict_validation:
            warnings.append(
                "Allowing invalid actions conflicts with strict environment validation"
            )

        # Selection format vs episode length
        if (
            self.action.selection_format == "mask"
            and self.environment.max_episode_steps < 30
        ):
            warnings.append(
                "Mask selection with short episodes may not provide enough time for complex selections"
            )

    def _validate_reward_consistency(
        self, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate reward configuration consistency."""
        # Reward timing vs episode length
        if (
            self.reward.reward_on_submit_only
            and self.environment.max_episode_steps < 10
        ):
            warnings.append(
                "Submit-only rewards with very few steps may not provide enough exploration time"
            )

        # Step penalty vs episode length
        if abs(self.reward.step_penalty) * self.environment.max_episode_steps > abs(
            self.reward.success_bonus
        ):
            warnings.append(
                "Cumulative step penalties may exceed success bonus - consider adjusting reward balance"
            )

        # Progress bonus vs reward timing
        if self.reward.progress_bonus != 0.0 and self.reward.reward_on_submit_only:
            warnings.append("Progress bonus is ignored when reward_on_submit_only=True")

        # Invalid action penalty vs action validation
        if (
            self.reward.invalid_action_penalty != 0.0
            and not self.action.validate_actions
        ):
            warnings.append(
                "Invalid action penalty set but action validation is disabled"
            )

    def _validate_dataset_consistency(
        self, errors: List[str], warnings: List[str]
    ) -> None:
        """Validate dataset configuration consistency."""
        # Grid size vs episode length
        max_grid_area = self.dataset.max_grid_height * self.dataset.max_grid_width
        if max_grid_area > 400 and self.environment.max_episode_steps < 100:
            warnings.append(
                "Large grids with short episodes may not provide enough time for complex tasks"
            )

        # Color count vs operations
        if self.dataset.max_colors > 10 and self.action.allowed_operations:
            fill_ops = [op for op in self.action.allowed_operations if 0 <= op <= 9]
            if len(fill_ops) < self.dataset.max_colors:
                warnings.append(
                    f"Dataset allows {self.dataset.max_colors} colors but only {len(fill_ops)} fill operations available"
                )

        # Task pairs vs memory
        total_pairs = self.dataset.max_train_pairs + self.dataset.max_test_pairs
        if total_pairs > 20 and self.visualization.max_memory_mb < 500:
            warnings.append(
                "Many task pairs with low memory limit may cause visualization issues"
            )

    def _validate_logging_consistency(
        self, errors: List[str], warnings: List[str]
    ) -> None:
        """Validate logging configuration consistency."""
        # Structured logging vs format
        if self.logging.structured_logging and self.logging.log_format not in [
            "json",
            "structured",
        ]:
            warnings.append(
                f"Structured logging enabled but format is '{self.logging.log_format}' - consider using 'json' or 'structured'"
            )



        # Content-specific logging consistency
        detailed_logging = (
            self.logging.log_operations
            or self.logging.log_grid_changes
            or self.logging.log_rewards
        )
        if detailed_logging and self.logging.log_level in ["ERROR", "WARNING"]:
            warnings.append(
                "Detailed content logging enabled but log level may suppress the logs"
            )
        return errors

    def to_yaml(self) -> str:
        """Export configuration to YAML format.

        Returns:
            YAML string representation of the configuration.
        """
        try:
            # Convert Equinox modules to dictionaries
            config_dict = {
                "environment": self._config_to_dict(self.environment),
                "dataset": self._config_to_dict(self.dataset),
                "action": self._config_to_dict(self.action),
                "reward": self._config_to_dict(self.reward),
                "visualization": self._config_to_dict(self.visualization),
                "storage": self._config_to_dict(self.storage),
                "logging": self._config_to_dict(self.logging),
                "wandb": self._config_to_dict(self.wandb),
            }

            return yaml.dump(
                config_dict,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                encoding=None,
            )

        except Exception as e:
            raise ConfigValidationError(
                f"Failed to export configuration to YAML: {e}"
            ) from e

    def to_yaml_file(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration file.

        Raises:
            ConfigValidationError: If YAML saving fails.
        """
        try:
            yaml_path = Path(yaml_path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)

            yaml_content = self.to_yaml()

            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

        except Exception as e:
            raise ConfigValidationError(
                f"Failed to save configuration to YAML file: {e}"
            ) from e

    def _config_to_dict(self, config: eqx.Module) -> Dict[str, Any]:
        """Convert an Equinox module to a dictionary with proper serialization."""
        result = {}

        # Get all fields from the module
        for field_name in config.__annotations__.keys():
            if hasattr(config, field_name):
                value = getattr(config, field_name)

                # Handle special cases for YAML serialization
                if value is None:
                    result[field_name] = None
                elif isinstance(value, (list, tuple)):
                    # Convert to plain list, handling nested OmegaConf objects
                    result[field_name] = self._serialize_value(list(value))
                elif hasattr(value, "__dict__") and hasattr(value, "_content"):
                    # Handle OmegaConf objects
                    result[field_name] = self._serialize_value(value)
                else:
                    result[field_name] = self._serialize_value(value)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize values for YAML compatibility."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "_content"):
            # Handle OmegaConf objects by converting to container
            try:
                from omegaconf import OmegaConf

                return OmegaConf.to_container(value, resolve=True)
            except:
                return str(value)
        else:
            # For any other type, convert to string representation
            return str(value)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> JaxArcConfig:
        """Load configuration from YAML file with validation.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            JaxArcConfig instance loaded from YAML.

        Raises:
            ConfigValidationError: If YAML loading or validation fails.
        """
        try:
            yaml_path = Path(yaml_path)
            if not yaml_path.exists():
                raise ConfigValidationError(f"YAML file does not exist: {yaml_path}")

            with open(yaml_path, encoding="utf-8") as f:
                yaml_content = f.read()

            return cls.from_yaml_string(yaml_content)

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to load configuration from YAML file: {e}"
            ) from e

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> JaxArcConfig:
        """Load configuration from YAML string with validation.

        Args:
            yaml_string: YAML configuration as string.

        Returns:
            JaxArcConfig instance loaded from YAML string.

        Raises:
            ConfigValidationError: If YAML parsing or validation fails.
        """
        try:
            config_dict = yaml.safe_load(yaml_string)

            if not isinstance(config_dict, dict):
                raise ConfigValidationError(
                    "YAML string must contain a dictionary at root level"
                )

            # Create config from dictionary
            config = cls._from_dict(config_dict)

            # Validate the loaded configuration
            validation_errors = config.validate()
            if validation_errors:
                error_msg = f"Configuration validation failed with {len(validation_errors)} errors:\n"
                error_msg += "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigValidationError(error_msg)

            return config

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to load configuration from YAML string: {e}"
            ) from e

    @classmethod
    def from_hydra(cls, hydra_config: DictConfig) -> JaxArcConfig:
        """Create JaxArcConfig from Hydra configuration.

        This method eliminates the dual configuration pattern by converting
        Hydra DictConfig directly to typed JaxArcConfig.

        Args:
            hydra_config: Hydra DictConfig object.

        Returns:
            JaxArcConfig instance created from Hydra config.

        Raises:
            ConfigValidationError: If conversion or validation fails.
        """
        try:
            # Convert DictConfig to regular dict for processing
            config_dict = OmegaConf.to_container(hydra_config, resolve=True)

            if not isinstance(config_dict, dict):
                raise ConfigValidationError("Hydra config must be a dictionary")

            return cls._from_dict(config_dict)

        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(
                f"Failed to create configuration from Hydra: {e}"
            ) from e

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> JaxArcConfig:
        """Create JaxArcConfig from dictionary."""
        # Create individual config components from dictionary sections
        environment_cfg = config_dict.get("environment", {})
        dataset_cfg = config_dict.get("dataset", {})
        action_cfg = config_dict.get("action", {})
        reward_cfg = config_dict.get("reward", {})
        grid_initialization_cfg = config_dict.get("grid_initialization", {})
        visualization_cfg = config_dict.get("visualization", {})
        storage_cfg = config_dict.get("storage", {})
        logging_cfg = config_dict.get("logging", {})
        wandb_cfg = config_dict.get("wandb", {})
        episode_cfg = config_dict.get("episode", {})
        history_cfg = config_dict.get("history", {})

        # Handle legacy config structure - merge top-level keys into appropriate sections
        for key, value in config_dict.items():
            if key not in [
                "environment",
                "dataset",
                "action",
                "reward",
                "grid_initialization",
                "visualization",
                "storage",
                "logging",
                "wandb",
                "episode",
                "history",
            ]:
                # Try to map legacy keys to appropriate config sections
                if key in [
                    "max_episode_steps",
                    "auto_reset",
                    "strict_validation",
                    "allow_invalid_actions",
                    "debug_level",
                ]:
                    environment_cfg[key] = value
                elif key in [
                    "dataset_name",
                    "dataset_path",
                    "max_grid_height",
                    "max_grid_width",
                    "task_split",
                ]:
                    dataset_cfg[key] = value
                elif key in ["selection_format", "num_operations", "validate_actions"]:
                    action_cfg[key] = value
                elif key in ["reward_on_submit_only", "step_penalty", "success_bonus"]:
                    reward_cfg[key] = value
                elif key in [
                    "grid_init_mode", 
                    "demo_weight", 
                    "permutation_weight", 
                    "empty_weight", 
                    "random_weight",
                    "permutation_types",
                    "random_density",
                    "random_pattern_type",
                    "enable_fallback"
                ]:
                    grid_initialization_cfg[key] = value
                elif key in ["log_operations", "log_grid_changes", "log_rewards"]:
                    logging_cfg[key] = value
                elif key in [
                    "episode_mode",
                    "demo_selection_strategy",
                    "allow_demo_switching",
                    "test_selection_strategy",
                    "allow_test_switching",
                    "terminate_on_first_success",
                    "max_pairs_per_episode",
                    "success_threshold",
                    "training_reward_frequency",
                    "evaluation_reward_frequency",
                ]:
                    episode_cfg[key] = value
                elif key in [
                    "max_history_length",
                    "store_selection_data",
                    "store_intermediate_grids",
                    "compress_repeated_actions",
                ]:
                    history_cfg[key] = value

        # Handle Hydra debug configuration mapping
        # If we have a debug config but no explicit debug_level in environment, infer it
        if "environment" in config_dict and "debug_level" not in environment_cfg:
            # Try to infer debug level from other settings or use default
            if logging_cfg.get("log_operations") or logging_cfg.get("log_grid_changes"):
                environment_cfg["debug_level"] = "standard"
            elif visualization_cfg.get("level") == "full":
                environment_cfg["debug_level"] = "research"
            elif visualization_cfg.get("level") == "verbose":
                environment_cfg["debug_level"] = "verbose"
            elif visualization_cfg.get("enabled", True):
                environment_cfg["debug_level"] = "standard"
            else:
                environment_cfg["debug_level"] = "off"

        # Create config components using from_hydra methods
        environment = EnvironmentConfig.from_hydra(DictConfig(environment_cfg))
        dataset = DatasetConfig.from_hydra(DictConfig(dataset_cfg))
        action = ActionConfig.from_hydra(DictConfig(action_cfg))
        reward = RewardConfig.from_hydra(DictConfig(reward_cfg))
        grid_initialization = GridInitializationConfig.from_hydra(DictConfig(grid_initialization_cfg))
        visualization = VisualizationConfig.from_hydra(DictConfig(visualization_cfg))
        storage = StorageConfig.from_hydra(DictConfig(storage_cfg))
        logging = LoggingConfig.from_hydra(DictConfig(logging_cfg))
        wandb = WandbConfig.from_hydra(DictConfig(wandb_cfg))
        episode = ArcEpisodeConfig.from_hydra(episode_cfg)

        # Create history config (HistoryConfig uses @chex.dataclass, not from_hydra)
        history = HistoryConfig(
            enabled=history_cfg.get("enabled", True),
            max_history_length=history_cfg.get("max_history_length", 1000),
            store_selection_data=history_cfg.get("store_selection_data", True),
            store_intermediate_grids=history_cfg.get("store_intermediate_grids", False),
            compress_repeated_actions=history_cfg.get(
                "compress_repeated_actions", True
            ),
        )

        return cls(
            environment=environment,
            dataset=dataset,
            action=action,
            reward=reward,
            grid_initialization=grid_initialization,
            visualization=visualization,
            storage=storage,
            logging=logging,
            wandb=wandb,
            episode=episode,
            history=history,
        )

    # =========================================================================
    # Efficient Serialization Methods (Single Source of Truth)
    # =========================================================================

    def save(self, path: str) -> None:
        """Save configuration efficiently using Equinox serialization.
        
        This method saves the configuration to a file using Equinox's efficient
        serialization system. Since configurations are typically small and
        contain no large arrays, no special filtering is needed.
        
        Args:
            path: Path to save the serialized configuration
            
        Examples:
            ```python
            # Save configuration
            config.save("config.eqx")
            ```
        """
        import equinox as eqx
        
        eqx.tree_serialise_leaves(path, self)

    @classmethod
    def load(cls, path: str) -> 'JaxArcConfig':
        """Load configuration from a serialized file.
        
        This method loads a configuration that was saved with the save() method.
        
        Args:
            path: Path to the serialized configuration file
            
        Returns:
            JaxArcConfig with loaded configuration
            
        Examples:
            ```python
            # Load configuration
            config = JaxArcConfig.load("config.eqx")
            ```
        """
        import equinox as eqx
        
        # Create dummy config with correct structure for deserialization
        dummy_config = cls()
        
        # Load and return the configuration
        return eqx.tree_deserialise_leaves(path, dummy_config)

    def save_with_state(self, config_path: str, state_path: str, state) -> None:
        """Save both configuration and state efficiently in separate files.
        
        This method saves the configuration and state in separate files,
        allowing for efficient loading and better organization.
        
        Args:
            config_path: Path to save the configuration
            state_path: Path to save the state
            state: ArcEnvState to save
            
        Examples:
            ```python
            # Save config and state separately
            config.save_with_state("config.eqx", "state.eqx", state)
            ```
        """
        # Save configuration
        self.save(config_path)
        
        # Save state using its efficient save method
        state.save(state_path)

    @classmethod
    def load_with_state(cls, config_path: str, state_path: str, parser) -> tuple['JaxArcConfig', 'ArcEnvState']:
        """Load both configuration and state from separate files.
        
        This method loads both configuration and state that were saved with
        save_with_state(), reconstructing the state's task_data using the parser.
        
        Args:
            config_path: Path to the serialized configuration file
            state_path: Path to the serialized state file
            parser: ArcDataParserBase instance to reconstruct task_data
            
        Returns:
            Tuple of (JaxArcConfig, ArcEnvState) with loaded data
            
        Examples:
            ```python
            from jaxarc.parsers import ArcAgiParser
            from jaxarc.state import ArcEnvState
            
            # Load config and state
            parser = ArcAgiParser(dataset_config)
            config, state = JaxArcConfig.load_with_state("config.eqx", "state.eqx", parser)
            ```
        """
        from jaxarc.state import ArcEnvState
        
        # Load configuration
        config = cls.load(config_path)
        
        # Load state using its efficient load method
        state = ArcEnvState.load(state_path, parser)
        
        return config, state
