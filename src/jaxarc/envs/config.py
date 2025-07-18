"""
Configuration system for JaxARC environments with Hydra integration.

This module provides typed configuration dataclasses that integrate seamlessly
with Hydra while ensuring JAX compatibility and type safety.
"""

from __future__ import annotations

import os
import re
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chex
from loguru import logger
from omegaconf import DictConfig, OmegaConf


# Validation utilities for enhanced config validation
class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    pass


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise ConfigValidationError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ConfigValidationError(f"{field_name} must be positive, got {value}")


def validate_non_negative_int(value: int, field_name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int):
        raise ConfigValidationError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ConfigValidationError(f"{field_name} must be non-negative, got {value}")


def validate_float_range(value: float, field_name: str, min_val: float, max_val: float) -> None:
    """Validate that a float value is within a specified range."""
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{field_name} must be a number, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ConfigValidationError(f"{field_name} must be in range [{min_val}, {max_val}], got {value}")


def validate_string_choice(value: str, field_name: str, choices: List[str]) -> None:
    """Validate that a string value is one of the allowed choices."""
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    if value not in choices:
        raise ConfigValidationError(f"{field_name} must be one of {choices}, got '{value}'")


def validate_path_string(value: str, field_name: str, must_exist: bool = False) -> None:
    """Validate that a value is a valid path string."""
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    
    # Check for invalid path characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in value for char in invalid_chars):
        raise ConfigValidationError(f"{field_name} contains invalid path characters: {value}")
    
    if must_exist and value and not Path(value).exists():
        raise ConfigValidationError(f"{field_name} path does not exist: {value}")


def validate_operation_list(operations: Optional[List[int]], field_name: str, max_operations: int) -> None:
    """Validate a list of operation IDs."""
    if operations is None:
        return
    
    if not isinstance(operations, list):
        raise ConfigValidationError(f"{field_name} must be a list or None, got {type(operations).__name__}")
    
    if not operations:
        raise ConfigValidationError(f"{field_name} cannot be empty if specified")
    
    for i, op in enumerate(operations):
        if not isinstance(op, int):
            raise ConfigValidationError(f"{field_name}[{i}] must be an integer, got {type(op).__name__}")
        if not 0 <= op < max_operations:
            raise ConfigValidationError(f"{field_name}[{i}] must be in range [0, {max_operations}), got {op}")
    
    # Check for duplicates
    if len(set(operations)) != len(operations):
        duplicates = [op for op in set(operations) if operations.count(op) > 1]
        raise ConfigValidationError(f"{field_name} contains duplicate operations: {duplicates}")


def validate_dataset_name(name: str, field_name: str) -> None:
    """Validate dataset name format."""
    if not isinstance(name, str):
        raise ConfigValidationError(f"{field_name} must be a string, got {type(name).__name__}")
    
    if not name:
        raise ConfigValidationError(f"{field_name} cannot be empty")
    
    # Check for valid dataset name pattern (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ConfigValidationError(f"{field_name} must contain only alphanumeric characters, hyphens, and underscores, got '{name}'")


def validate_cross_field_consistency(config_obj: Any, validations: List[tuple]) -> None:
    """Validate cross-field consistency rules.
    
    Args:
        config_obj: Configuration object to validate
        validations: List of (condition_func, error_message) tuples
    """
    for condition_func, error_message in validations:
        try:
            if not condition_func(config_obj):
                raise ConfigValidationError(error_message)
        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            raise ConfigValidationError(f"Validation error: {error_message}") from e


@chex.dataclass(frozen=True)
class DebugConfig:
    """Debug configuration for development and analysis."""

    log_rl_steps: bool = False
    rl_steps_output_dir: str = "output/rl_steps"
    clear_output_dir: bool = True
    
    # Enhanced visualization settings
    enhanced_visualization_enabled: bool = False
    visualization_level: str = "standard"  # "off", "minimal", "standard", "verbose", "full"
    async_logging: bool = True
    wandb_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate debug configuration with comprehensive field validation."""
        try:
            # Validate log_rl_steps is boolean
            if not isinstance(self.log_rl_steps, bool):
                raise ConfigValidationError(f"log_rl_steps must be a boolean, got {type(self.log_rl_steps).__name__}")
            
            # Validate output directory path
            validate_path_string(self.rl_steps_output_dir, "rl_steps_output_dir")
            
            # Validate clear_output_dir is boolean
            if not isinstance(self.clear_output_dir, bool):
                raise ConfigValidationError(f"clear_output_dir must be a boolean, got {type(self.clear_output_dir).__name__}")
            
            # Validate enhanced visualization settings
            if not isinstance(self.enhanced_visualization_enabled, bool):
                raise ConfigValidationError(f"enhanced_visualization_enabled must be a boolean, got {type(self.enhanced_visualization_enabled).__name__}")
            
            valid_levels = ["off", "minimal", "standard", "verbose", "full"]
            validate_string_choice(self.visualization_level, "visualization_level", valid_levels)
            
            if not isinstance(self.async_logging, bool):
                raise ConfigValidationError(f"async_logging must be a boolean, got {type(self.async_logging).__name__}")
            
            if not isinstance(self.wandb_enabled, bool):
                raise ConfigValidationError(f"wandb_enabled must be a boolean, got {type(self.wandb_enabled).__name__}")
            
            # Cross-field validation: warn if logging is enabled but directory is empty
            if self.log_rl_steps and not self.rl_steps_output_dir.strip():
                logger.warning("log_rl_steps is enabled but rl_steps_output_dir is empty")
                
            # Warn about conflicting settings
            if self.enhanced_visualization_enabled and self.visualization_level == "off":
                logger.warning("enhanced_visualization_enabled=True but visualization_level='off' - no visualization will be generated")
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"DebugConfig validation failed: {e}") from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DebugConfig:
        """Create debug config from Hydra DictConfig with validation."""
        return cls(
            log_rl_steps=cfg.get("log_rl_steps", False),
            rl_steps_output_dir=cfg.get("rl_steps_output_dir", "output/rl_steps"),
            clear_output_dir=cfg.get("clear_output_dir", True),
            enhanced_visualization_enabled=cfg.get("enhanced_visualization_enabled", False),
            visualization_level=cfg.get("visualization_level", "standard"),
            async_logging=cfg.get("async_logging", True),
            wandb_enabled=cfg.get("wandb_enabled", False),
        )


@chex.dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward calculation."""

    # Basic reward settings
    reward_on_submit_only: bool = True
    step_penalty: float = -0.01
    success_bonus: float = 10.0
    similarity_weight: float = 1.0

    # Additional reward shaping
    progress_bonus: float = 0.0
    invalid_action_penalty: float = -0.1

    def __post_init__(self) -> None:
        """Validate reward configuration with comprehensive field validation."""
        try:
            # Validate boolean fields
            if not isinstance(self.reward_on_submit_only, bool):
                raise ConfigValidationError(f"reward_on_submit_only must be a boolean, got {type(self.reward_on_submit_only).__name__}")
            
            # Validate numeric fields with reasonable ranges
            if not isinstance(self.step_penalty, (int, float)):
                raise ConfigValidationError(f"step_penalty must be a number, got {type(self.step_penalty).__name__}")
            
            if not isinstance(self.success_bonus, (int, float)):
                raise ConfigValidationError(f"success_bonus must be a number, got {type(self.success_bonus).__name__}")
            
            if not isinstance(self.similarity_weight, (int, float)):
                raise ConfigValidationError(f"similarity_weight must be a number, got {type(self.similarity_weight).__name__}")
            
            if not isinstance(self.progress_bonus, (int, float)):
                raise ConfigValidationError(f"progress_bonus must be a number, got {type(self.progress_bonus).__name__}")
            
            if not isinstance(self.invalid_action_penalty, (int, float)):
                raise ConfigValidationError(f"invalid_action_penalty must be a number, got {type(self.invalid_action_penalty).__name__}")
            
            # Validate reasonable ranges
            validate_float_range(self.step_penalty, "step_penalty", -10.0, 1.0)
            validate_float_range(self.success_bonus, "success_bonus", -100.0, 1000.0)
            validate_float_range(self.similarity_weight, "similarity_weight", 0.0, 10.0)
            validate_float_range(self.progress_bonus, "progress_bonus", -10.0, 10.0)
            validate_float_range(self.invalid_action_penalty, "invalid_action_penalty", -10.0, 1.0)
            
            # Cross-field validation and warnings
            cross_validations = [
                (lambda cfg: not (cfg.reward_on_submit_only and cfg.progress_bonus != 0.0) or True,
                 "progress_bonus is ignored when reward_on_submit_only=True"),
                (lambda cfg: cfg.step_penalty <= 0 or True,
                 f"step_penalty should typically be negative or zero for proper learning, got {self.step_penalty}"),
                (lambda cfg: cfg.success_bonus >= 0 or True,
                 f"success_bonus should typically be positive for proper learning, got {self.success_bonus}"),
                (lambda cfg: cfg.invalid_action_penalty <= 0 or True,
                 f"invalid_action_penalty should typically be negative or zero, got {self.invalid_action_penalty}")
            ]
            
            # Issue warnings for potentially problematic configurations
            if self.reward_on_submit_only and self.progress_bonus != 0.0:
                logger.warning("progress_bonus is ignored when reward_on_submit_only=True")
            
            if self.step_penalty > 0:
                logger.warning(f"step_penalty should typically be negative or zero for proper learning, got {self.step_penalty}")
            
            if self.success_bonus < 0:
                logger.warning(f"success_bonus should typically be positive for proper learning, got {self.success_bonus}")
            
            if self.invalid_action_penalty > 0:
                logger.warning(f"invalid_action_penalty should typically be negative or zero, got {self.invalid_action_penalty}")
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"RewardConfig validation failed: {e}") from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> RewardConfig:
        """Create reward config from Hydra DictConfig with validation."""
        return cls(
            reward_on_submit_only=cfg.get("reward_on_submit_only", True),
            step_penalty=cfg.get("step_penalty", -0.01),
            success_bonus=cfg.get("success_bonus", 10.0),
            similarity_weight=cfg.get("similarity_weight", 1.0),
            progress_bonus=cfg.get("progress_bonus", 0.0),
            invalid_action_penalty=cfg.get("invalid_action_penalty", -0.1),
        )


@chex.dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset-specific settings."""

    # Dataset identification
    dataset_name: str = "arc-agi-1"  # "arc-agi-1", "concept-arc", "mini-arc", etc.
    dataset_path: str = ""

    # Dataset-specific grid constraints (override GridConfig if specified)
    dataset_max_grid_height: Optional[int] = None
    dataset_max_grid_width: Optional[int] = None
    dataset_min_grid_height: Optional[int] = None
    dataset_min_grid_width: Optional[int] = None
    dataset_max_colors: Optional[int] = None

    # Task sampling parameters
    task_split: str = "train"  # "train", "eval", "test"
    max_tasks: Optional[int] = None  # Limit number of tasks to load
    shuffle_tasks: bool = True

    def __post_init__(self) -> None:
        """Validate dataset configuration with comprehensive field validation."""
        try:
            # Validate dataset name
            validate_dataset_name(self.dataset_name, "dataset_name")
            
            # Validate dataset path
            validate_path_string(self.dataset_path, "dataset_path")
            
            # Validate optional grid constraints
            if self.dataset_max_grid_height is not None:
                validate_positive_int(self.dataset_max_grid_height, "dataset_max_grid_height")
                if self.dataset_max_grid_height > 100:  # Reasonable upper bound
                    logger.warning(f"dataset_max_grid_height is very large: {self.dataset_max_grid_height}")
            
            if self.dataset_max_grid_width is not None:
                validate_positive_int(self.dataset_max_grid_width, "dataset_max_grid_width")
                if self.dataset_max_grid_width > 100:  # Reasonable upper bound
                    logger.warning(f"dataset_max_grid_width is very large: {self.dataset_max_grid_width}")
            
            if self.dataset_min_grid_height is not None:
                validate_positive_int(self.dataset_min_grid_height, "dataset_min_grid_height")
                if self.dataset_min_grid_height < 1:
                    raise ConfigValidationError("dataset_min_grid_height must be at least 1")
            
            if self.dataset_min_grid_width is not None:
                validate_positive_int(self.dataset_min_grid_width, "dataset_min_grid_width")
                if self.dataset_min_grid_width < 1:
                    raise ConfigValidationError("dataset_min_grid_width must be at least 1")
            
            if self.dataset_max_colors is not None:
                validate_positive_int(self.dataset_max_colors, "dataset_max_colors")
                if self.dataset_max_colors < 2:
                    raise ConfigValidationError("dataset_max_colors must be at least 2")
                if self.dataset_max_colors > 20:  # Reasonable upper bound
                    logger.warning(f"dataset_max_colors is very large: {self.dataset_max_colors}")
            
            # Validate task sampling parameters
            if not isinstance(self.shuffle_tasks, bool):
                raise ConfigValidationError(f"shuffle_tasks must be a boolean, got {type(self.shuffle_tasks).__name__}")
            
            if self.max_tasks is not None:
                validate_positive_int(self.max_tasks, "max_tasks")
                if self.max_tasks > 10000:  # Reasonable upper bound
                    logger.warning(f"max_tasks is very large: {self.max_tasks}")
            
            # Validate task split based on dataset
            standard_splits = ["train", "eval", "test", "all"]
            dataset_specific_splits = {
                "ConceptARC": ["corpus"],
                "concept-arc": ["corpus"],
                "MiniARC": ["training", "evaluation"],
                "mini-arc": ["training", "evaluation"],
                "arc-agi-1": ["training", "evaluation"],
                "arc-agi-2": ["training", "evaluation"],
            }
            
            # Get valid splits for this dataset
            valid_splits = standard_splits.copy()
            if self.dataset_name in dataset_specific_splits:
                valid_splits.extend(dataset_specific_splits[self.dataset_name])
            
            validate_string_choice(self.task_split, "task_split", valid_splits)
            
            # Cross-field validation for grid constraints
            if (self.dataset_min_grid_height is not None and 
                self.dataset_max_grid_height is not None and
                self.dataset_min_grid_height > self.dataset_max_grid_height):
                raise ConfigValidationError(
                    f"dataset_min_grid_height ({self.dataset_min_grid_height}) > "
                    f"dataset_max_grid_height ({self.dataset_max_grid_height})"
                )
            
            if (self.dataset_min_grid_width is not None and 
                self.dataset_max_grid_width is not None and
                self.dataset_min_grid_width > self.dataset_max_grid_width):
                raise ConfigValidationError(
                    f"dataset_min_grid_width ({self.dataset_min_grid_width}) > "
                    f"dataset_max_grid_width ({self.dataset_max_grid_width})"
                )
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"DatasetConfig validation failed: {e}") from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DatasetConfig:
        """Create dataset config from Hydra DictConfig with validation."""
        return cls(
            dataset_name=cfg.get("dataset_name", "arc-agi-1"),
            dataset_path=cfg.get("dataset_path", ""),
            dataset_max_grid_height=cfg.get("dataset_max_grid_height"),
            dataset_max_grid_width=cfg.get("dataset_max_grid_width"),
            dataset_min_grid_height=cfg.get("dataset_min_grid_height"),
            dataset_min_grid_width=cfg.get("dataset_min_grid_width"),
            dataset_max_colors=cfg.get("dataset_max_colors"),
            task_split=cfg.get("task_split", "train"),
            max_tasks=cfg.get("max_tasks"),
            shuffle_tasks=cfg.get("shuffle_tasks", True),
        )

    def get_effective_grid_config(self, base_grid_config: GridConfig) -> GridConfig:
        """Get effective grid config with dataset overrides applied."""
        return GridConfig(
            max_grid_height=self.dataset_max_grid_height
            or base_grid_config.max_grid_height,
            max_grid_width=self.dataset_max_grid_width
            or base_grid_config.max_grid_width,
            min_grid_height=self.dataset_min_grid_height
            or base_grid_config.min_grid_height,
            min_grid_width=self.dataset_min_grid_width
            or base_grid_config.min_grid_width,
            max_colors=self.dataset_max_colors or base_grid_config.max_colors,
            background_color=base_grid_config.background_color,
        )


@chex.dataclass(frozen=True)
class GridConfig:
    """Configuration for grid dimensions and constraints."""

    # Grid size constraints
    max_grid_height: int = 30
    max_grid_width: int = 30
    min_grid_height: int = 3
    min_grid_width: int = 3

    # Color constraints
    max_colors: int = 10
    background_color: int = 0

    def __post_init__(self) -> None:
        """Validate grid configuration with comprehensive field validation."""
        try:
            # Validate all grid dimensions are positive integers
            validate_positive_int(self.max_grid_height, "max_grid_height")
            validate_positive_int(self.max_grid_width, "max_grid_width")
            validate_positive_int(self.min_grid_height, "min_grid_height")
            validate_positive_int(self.min_grid_width, "min_grid_width")
            
            # Validate reasonable bounds for grid dimensions
            if self.max_grid_height > 200:
                logger.warning(f"max_grid_height is very large: {self.max_grid_height}")
            if self.max_grid_width > 200:
                logger.warning(f"max_grid_width is very large: {self.max_grid_width}")
            
            # Validate color constraints
            validate_positive_int(self.max_colors, "max_colors")
            validate_non_negative_int(self.background_color, "background_color")
            
            if self.max_colors < 2:
                raise ConfigValidationError("max_colors must be at least 2")
            if self.max_colors > 50:  # Reasonable upper bound
                logger.warning(f"max_colors is very large: {self.max_colors}")
            
            # Cross-field validation
            if self.max_grid_height < self.min_grid_height:
                raise ConfigValidationError(
                    f"max_grid_height ({self.max_grid_height}) < min_grid_height ({self.min_grid_height})"
                )
            
            if self.max_grid_width < self.min_grid_width:
                raise ConfigValidationError(
                    f"max_grid_width ({self.max_grid_width}) < min_grid_width ({self.min_grid_width})"
                )
            
            if self.background_color >= self.max_colors:
                raise ConfigValidationError(
                    f"background_color ({self.background_color}) must be < max_colors ({self.max_colors})"
                )
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"GridConfig validation failed: {e}") from e

    @property
    def max_grid_size(self) -> tuple[int, int]:
        """Get maximum grid size as (height, width) tuple."""
        return (self.max_grid_height, self.max_grid_width)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> GridConfig:
        """Create grid config from Hydra DictConfig with validation."""
        return cls(
            max_grid_height=cfg.get("max_grid_height", 30),
            max_grid_width=cfg.get("max_grid_width", 30),
            min_grid_height=cfg.get("min_grid_height", 3),
            min_grid_width=cfg.get("min_grid_width", 3),
            max_colors=cfg.get("max_colors", 10),
            background_color=cfg.get("background_color", 0),
        )


@chex.dataclass(frozen=True)
class ActionConfig:
    """Configuration for action space and validation."""

    # Selection format
    selection_format: str = "mask"  # "mask", "point", "bbox"

    # Selection parameters
    selection_threshold: float = 0.5  # For converting continuous to discrete selection
    allow_partial_selection: bool = True

    # Operation parameters
    num_operations: int = 35  # Total number of operations (0-34)
    allowed_operations: Optional[list[int]] = (
        None  # Specific operations allowed (None = all)
    )

    # Validation settings
    validate_actions: bool = True
    clip_invalid_actions: bool = True

    def __post_init__(self) -> None:
        """Validate action configuration with comprehensive field validation."""
        try:
            # Validate selection format
            valid_formats = ["mask", "point", "bbox"]
            validate_string_choice(self.selection_format, "selection_format", valid_formats)
            
            # Validate selection threshold
            validate_float_range(self.selection_threshold, "selection_threshold", 0.0, 1.0)
            
            # Validate boolean fields
            if not isinstance(self.allow_partial_selection, bool):
                raise ConfigValidationError(f"allow_partial_selection must be a boolean, got {type(self.allow_partial_selection).__name__}")
            
            if not isinstance(self.validate_actions, bool):
                raise ConfigValidationError(f"validate_actions must be a boolean, got {type(self.validate_actions).__name__}")
            
            if not isinstance(self.clip_invalid_actions, bool):
                raise ConfigValidationError(f"clip_invalid_actions must be a boolean, got {type(self.clip_invalid_actions).__name__}")
            
            # Validate operation parameters
            validate_positive_int(self.num_operations, "num_operations")
            if self.num_operations > 100:  # Reasonable upper bound
                logger.warning(f"num_operations is very large: {self.num_operations}")
            
            # Validate allowed operations list
            validate_operation_list(self.allowed_operations, "allowed_operations", self.num_operations)
            
            # Cross-field validation and warnings
            if (self.selection_format != "mask" and self.allow_partial_selection):
                logger.warning(f"allow_partial_selection is ignored for selection_format='{self.selection_format}'")
            
            if not self.validate_actions and self.clip_invalid_actions:
                logger.warning("clip_invalid_actions has no effect when validate_actions=False")
            
            # Warn about potentially problematic configurations
            if self.selection_threshold == 0.5 and self.selection_format == "mask":
                logger.info("Using default selection_threshold=0.5 for mask format - consider tuning for your use case")
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"ActionConfig validation failed: {e}") from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> ActionConfig:
        """Create action config from Hydra DictConfig with validation."""
        allowed_ops = cfg.get("allowed_operations")
        if allowed_ops is not None and not isinstance(allowed_ops, list):
            allowed_ops = list(allowed_ops) if allowed_ops else None

        selection_format = cfg.get("selection_format", "mask")

        return cls(
            selection_format=selection_format,
            selection_threshold=cfg.get("selection_threshold", 0.5),
            allow_partial_selection=cfg.get("allow_partial_selection", True),
            num_operations=cfg.get("num_operations", 35),
            allowed_operations=allowed_ops,
            validate_actions=cfg.get("validate_actions", True),
            clip_invalid_actions=cfg.get("clip_invalid_actions", True),
        )


@chex.dataclass(frozen=True)
class ArcEnvConfig:
    """Complete configuration for ARC environment."""

    # Episode settings
    max_episode_steps: int = 100
    auto_reset: bool = True

    # Logging and debugging
    log_operations: bool = False
    log_grid_changes: bool = False
    log_rewards: bool = False

    # Environment behavior
    strict_validation: bool = True
    allow_invalid_actions: bool = False

    # Sub-configurations
    reward: RewardConfig = field(default_factory=RewardConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Task parser (can be set after creation)
    parser: Optional[Any] = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Validate environment configuration with comprehensive field validation."""
        try:
            # Validate episode settings
            validate_positive_int(self.max_episode_steps, "max_episode_steps")
            if self.max_episode_steps > 10000:  # Reasonable upper bound
                logger.warning(f"max_episode_steps is very large: {self.max_episode_steps}")
            
            # Validate boolean fields
            if not isinstance(self.auto_reset, bool):
                raise ConfigValidationError(f"auto_reset must be a boolean, got {type(self.auto_reset).__name__}")
            
            if not isinstance(self.log_operations, bool):
                raise ConfigValidationError(f"log_operations must be a boolean, got {type(self.log_operations).__name__}")
            
            if not isinstance(self.log_grid_changes, bool):
                raise ConfigValidationError(f"log_grid_changes must be a boolean, got {type(self.log_grid_changes).__name__}")
            
            if not isinstance(self.log_rewards, bool):
                raise ConfigValidationError(f"log_rewards must be a boolean, got {type(self.log_rewards).__name__}")
            
            if not isinstance(self.strict_validation, bool):
                raise ConfigValidationError(f"strict_validation must be a boolean, got {type(self.strict_validation).__name__}")
            
            if not isinstance(self.allow_invalid_actions, bool):
                raise ConfigValidationError(f"allow_invalid_actions must be a boolean, got {type(self.allow_invalid_actions).__name__}")
            
            # Validate sub-configs are properly typed
            if not isinstance(self.reward, RewardConfig):
                raise ConfigValidationError(f"reward must be RewardConfig, got {type(self.reward).__name__}")

            if not isinstance(self.grid, GridConfig):
                raise ConfigValidationError(f"grid must be GridConfig, got {type(self.grid).__name__}")

            if not isinstance(self.action, ActionConfig):
                raise ConfigValidationError(f"action must be ActionConfig, got {type(self.action).__name__}")

            if not isinstance(self.dataset, DatasetConfig):
                raise ConfigValidationError(f"dataset must be DatasetConfig, got {type(self.dataset).__name__}")
            
            if not isinstance(self.debug, DebugConfig):
                raise ConfigValidationError(f"debug must be DebugConfig, got {type(self.debug).__name__}")
            
            # Cross-field validation and warnings
            if self.strict_validation and self.allow_invalid_actions:
                logger.warning("allow_invalid_actions=True may conflict with strict_validation=True")
            
            if not self.strict_validation and not self.allow_invalid_actions:
                logger.warning("Both strict_validation and allow_invalid_actions are False - this may lead to unexpected behavior")
            
            # Validate cross-configuration consistency
            self._validate_cross_config_consistency()
                
        except ConfigValidationError as e:
            raise ConfigValidationError(f"ArcEnvConfig validation failed: {e}") from e
    
    def _validate_cross_config_consistency(self) -> None:
        """Validate consistency across different configuration sections."""
        # Check reward and action config consistency
        if (self.reward.reward_on_submit_only and 
            self.reward.progress_bonus != 0.0):
            logger.warning("progress_bonus is ignored when reward_on_submit_only=True")
        
        # Check grid and dataset config consistency
        if (self.dataset.dataset_max_grid_height is not None and
            self.dataset.dataset_max_grid_height != self.grid.max_grid_height):
            logger.info(f"Dataset overrides grid max_height: {self.dataset.dataset_max_grid_height} vs {self.grid.max_grid_height}")
        
        if (self.dataset.dataset_max_grid_width is not None and
            self.dataset.dataset_max_grid_width != self.grid.max_grid_width):
            logger.info(f"Dataset overrides grid max_width: {self.dataset.dataset_max_grid_width} vs {self.grid.max_grid_width}")
        
        # Check action and validation consistency
        if (self.action.selection_format != "mask" and 
            self.action.allow_partial_selection):
            logger.warning(f"allow_partial_selection is ignored for selection_format='{self.action.selection_format}'")
        
        # Check logging and debug consistency
        if (self.debug.log_rl_steps and 
            not any([self.log_operations, self.log_grid_changes, self.log_rewards])):
            logger.info("Debug RL step logging is enabled but no environment logging is enabled")
        
        # Performance warnings
        if (self.log_operations and self.log_grid_changes and self.log_rewards):
            logger.warning("All logging options are enabled - this may impact performance")
        
        # Episode length warnings based on action format
        if self.action.selection_format == "point" and self.max_episode_steps < 50:
            logger.warning("Point-based actions may need more steps - consider increasing max_episode_steps")
        elif self.action.selection_format == "mask" and self.max_episode_steps > 200:
            logger.warning("Mask-based actions typically need fewer steps - consider reducing max_episode_steps")

    @classmethod
    def from_hydra(cls, cfg: DictConfig, parser: Optional[Any] = None) -> ArcEnvConfig:
        """Create complete environment config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration
            parser: Optional pre-initialized parser (e.g., from Hydra instantiation)
        """
        # Handle nested configs
        reward_cfg = RewardConfig.from_hydra(cfg.get("reward", {}))
        grid_cfg = GridConfig.from_hydra(cfg.get("grid", {}))
        action_cfg = ActionConfig.from_hydra(cfg.get("action", {}))
        dataset_cfg = DatasetConfig.from_hydra(cfg.get("dataset", {}))
        debug_cfg = DebugConfig.from_hydra(cfg.get("debug", {}))

        # Apply dataset-specific overrides to grid config
        effective_grid_cfg = dataset_cfg.get_effective_grid_config(grid_cfg)

        return cls(
            max_episode_steps=cfg.get("max_episode_steps", 100),
            auto_reset=cfg.get("auto_reset", True),
            log_operations=cfg.get("log_operations", False),
            log_grid_changes=cfg.get("log_grid_changes", False),
            log_rewards=cfg.get("log_rewards", False),
            strict_validation=cfg.get("strict_validation", True),
            allow_invalid_actions=cfg.get("allow_invalid_actions", False),
            reward=reward_cfg,
            grid=effective_grid_cfg,
            action=action_cfg,
            dataset=dataset_cfg,
            debug=debug_cfg,
            parser=parser,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "max_episode_steps": self.max_episode_steps,
            "auto_reset": self.auto_reset,
            "log_operations": self.log_operations,
            "log_grid_changes": self.log_grid_changes,
            "log_rewards": self.log_rewards,
            "strict_validation": self.strict_validation,
            "allow_invalid_actions": self.allow_invalid_actions,
            "reward": {
                "reward_on_submit_only": self.reward.reward_on_submit_only,
                "step_penalty": self.reward.step_penalty,
                "success_bonus": self.reward.success_bonus,
                "similarity_weight": self.reward.similarity_weight,
                "progress_bonus": self.reward.progress_bonus,
                "invalid_action_penalty": self.reward.invalid_action_penalty,
            },
            "grid": {
                "max_grid_height": self.grid.max_grid_height,
                "max_grid_width": self.grid.max_grid_width,
                "min_grid_height": self.grid.min_grid_height,
                "min_grid_width": self.grid.min_grid_width,
                "max_colors": self.grid.max_colors,
                "background_color": self.grid.background_color,
            },
            "action": {
                "selection_format": self.action.selection_format,
                "selection_threshold": self.action.selection_threshold,
                "allow_partial_selection": self.action.allow_partial_selection,
                "num_operations": self.action.num_operations,
                "allowed_operations": self.action.allowed_operations,
                "validate_actions": self.action.validate_actions,
                "clip_invalid_actions": self.action.clip_invalid_actions,
            },
            "dataset": {
                "dataset_name": self.dataset.dataset_name,
                "dataset_path": self.dataset.dataset_path,
                "dataset_max_grid_height": self.dataset.dataset_max_grid_height,
                "dataset_max_grid_width": self.dataset.dataset_max_grid_width,
                "dataset_min_grid_height": self.dataset.dataset_min_grid_height,
                "dataset_min_grid_width": self.dataset.dataset_min_grid_width,
                "dataset_max_colors": self.dataset.dataset_max_colors,
                "task_split": self.dataset.task_split,
                "max_tasks": self.dataset.max_tasks,
                "shuffle_tasks": self.dataset.shuffle_tasks,
            },
        }


# Utility functions for config conversion and validation


def validate_config(config: ArcEnvConfig) -> None:
    """Validate configuration consistency with enhanced validation.
    
    This function provides additional validation beyond what's done in __post_init__.
    It's useful for runtime validation of configurations that may have been modified.
    """
    try:
        # The configuration should already be validated by __post_init__, but we can
        # perform additional runtime checks here if needed
        
        # Check reward config consistency
        if config.reward.reward_on_submit_only and config.reward.progress_bonus != 0.0:
            logger.warning("progress_bonus is ignored when reward_on_submit_only=True")

        # Check grid config consistency (this should already be caught in __post_init__)
        if config.grid.max_colors <= config.grid.background_color:
            raise ConfigValidationError(
                f"background_color ({config.grid.background_color}) must be < max_colors ({config.grid.max_colors})"
            )
        
        # Additional runtime validation for parser compatibility
        if config.parser is not None:
            # Check if parser is compatible with dataset configuration
            parser_class_name = config.parser.__class__.__name__
            expected_parsers = {
                "arc-agi-1": ["ArcAgiParser"],
                "arc-agi-2": ["ArcAgiParser"], 
                "concept-arc": ["ConceptArcParser"],
                "mini-arc": ["MiniArcParser"],
            }
            
            if config.dataset.dataset_name in expected_parsers:
                if parser_class_name not in expected_parsers[config.dataset.dataset_name]:
                    logger.warning(
                        f"Parser {parser_class_name} may not be optimal for dataset {config.dataset.dataset_name}. "
                        f"Expected: {expected_parsers[config.dataset.dataset_name]}"
                    )
        
        # Validate action config consistency with environment settings
        if (config.action.selection_format != "mask" and 
            config.action.allow_partial_selection):
            logger.warning(
                f"allow_partial_selection is ignored for selection_format='{config.action.selection_format}'"
            )
            
    except Exception as e:
        if isinstance(e, ConfigValidationError):
            raise
        raise ConfigValidationError(f"Configuration validation failed: {e}") from e

    # Check action config consistency
    if (
        config.action.selection_format != "mask"
        and config.action.allow_partial_selection
    ):
        logger.warning(
            f"allow_partial_selection is ignored for selection_format='{config.action.selection_format}'"
        )


def config_from_dict(config_dict: Dict[str, Any]) -> ArcEnvConfig:
    """Create config from dictionary."""
    return ArcEnvConfig(
        max_episode_steps=config_dict.get("max_episode_steps", 100),
        auto_reset=config_dict.get("auto_reset", True),
        log_operations=config_dict.get("log_operations", False),
        log_grid_changes=config_dict.get("log_grid_changes", False),
        log_rewards=config_dict.get("log_rewards", False),
        strict_validation=config_dict.get("strict_validation", True),
        allow_invalid_actions=config_dict.get("allow_invalid_actions", False),
        reward=RewardConfig(**config_dict.get("reward", {})),
        grid=GridConfig(**config_dict.get("grid", {})),
        action=ActionConfig(**config_dict.get("action", {})),
        dataset=DatasetConfig(**config_dict.get("dataset", {})),
    )


def merge_configs(base: ArcEnvConfig, override: DictConfig) -> ArcEnvConfig:
    """Merge base config with Hydra override config."""
    # Convert base config to dict
    base_dict = base.to_dict()

    # Merge with override using OmegaConf
    merged_dict = OmegaConf.merge(base_dict, override)

    # Convert back to typed config
    return config_from_dict(OmegaConf.to_container(merged_dict))



def get_config_summary(config: ArcEnvConfig) -> str:
    """Get human-readable config summary."""
    return f"""ARC Environment Configuration:
  Episode: max_steps={config.max_episode_steps}, auto_reset={config.auto_reset}
  Rewards: submit_only={config.reward.reward_on_submit_only}, success_bonus={config.reward.success_bonus}
  Grid: max_size=({config.grid.max_grid_height}, {config.grid.max_grid_width}), colors={config.grid.max_colors}
  Actions: format={config.action.selection_format}, operations={config.action.num_operations}, allowed={config.action.allowed_operations}
  Dataset: name={config.dataset.dataset_name}, split={config.dataset.task_split}
  Logging: operations={config.log_operations}, grid_changes={config.log_grid_changes}
"""
