"""
Configuration system for JaxARC environments with Hydra integration.

This module provides typed configuration dataclasses that integrate seamlessly
with Hydra while ensuring JAX compatibility and type safety.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, Optional

import chex
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@chex.dataclass(frozen=True)
class DebugConfig:
    """Debug configuration for development and analysis."""

    log_rl_steps: bool = False
    rl_steps_output_dir: str = "output/rl_steps"
    clear_output_dir: bool = True

    def __post_init__(self) -> None:
        """Validate debug configuration."""
        # Ensure output directory path is valid
        if not isinstance(self.rl_steps_output_dir, str):
            raise ValueError("rl_steps_output_dir must be a string")

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DebugConfig:
        """Create debug config from Hydra DictConfig."""
        return cls(
            log_rl_steps=cfg.get("log_rl_steps", False),
            rl_steps_output_dir=cfg.get("rl_steps_output_dir", "output/rl_steps"),
            clear_output_dir=cfg.get("clear_output_dir", True),
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
        """Validate reward configuration."""
        if self.step_penalty > 0:
            logger.warning(f"Step penalty should be negative, got {self.step_penalty}")

        if self.success_bonus < 0:
            logger.warning(
                f"Success bonus should be positive, got {self.success_bonus}"
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
        """Validate dataset configuration."""
        # Standard splits for most datasets
        standard_splits = ["train", "eval", "test", "all"]
        
        # Dataset-specific splits
        dataset_specific_splits = {
            "ConceptARC": ["corpus"],
            "MiniARC": ["training", "evaluation"],
            "arc-agi-1": ["training", "evaluation"],
            "arc-agi-2": ["training", "evaluation"],
        }
        
        # Get valid splits for this dataset
        valid_splits = standard_splits.copy()
        if self.dataset_name in dataset_specific_splits:
            valid_splits.extend(dataset_specific_splits[self.dataset_name])
        
        if self.task_split not in valid_splits:
            raise ValueError(
                f"task_split must be one of {valid_splits}, got {self.task_split}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> DatasetConfig:
        """Create dataset config from Hydra DictConfig."""
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
        """Validate grid configuration."""
        if self.max_grid_height < self.min_grid_height:
            raise ValueError(
                f"max_grid_height ({self.max_grid_height}) < min_grid_height ({self.min_grid_height})"
            )

        if self.max_grid_width < self.min_grid_width:
            raise ValueError(
                f"max_grid_width ({self.max_grid_width}) < min_grid_width ({self.min_grid_width})"
            )

        if self.max_colors < 2:
            raise ValueError(f"max_colors must be at least 2, got {self.max_colors}")

    @property
    def max_grid_size(self) -> tuple[int, int]:
        """Get maximum grid size as (height, width) tuple."""
        return (self.max_grid_height, self.max_grid_width)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> GridConfig:
        """Create grid config from Hydra DictConfig."""
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
        """Validate action configuration."""
        if self.selection_format not in ["mask", "point", "bbox"]:
            raise ValueError(f"Invalid selection_format: {self.selection_format}")

        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError(
                f"selection_threshold must be in [0, 1], got {self.selection_threshold}"
            )

        if self.num_operations < 1:
            raise ValueError(
                f"num_operations must be positive, got {self.num_operations}"
            )

        if self.allowed_operations is not None:
            if not all(0 <= op < self.num_operations for op in self.allowed_operations):
                raise ValueError(
                    f"allowed_operations must be in range [0, {self.num_operations})"
                )
            if len(set(self.allowed_operations)) != len(self.allowed_operations):
                raise ValueError("allowed_operations must not contain duplicates")

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> ActionConfig:
        """Create action config from Hydra DictConfig."""
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
        """Validate environment configuration."""
        if self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}"
            )

        # Validate sub-configs are properly typed (frozen dataclass can't mutate)
        if not isinstance(self.reward, RewardConfig):
            raise TypeError(f"reward must be RewardConfig, got {type(self.reward)}")

        if not isinstance(self.grid, GridConfig):
            raise TypeError(f"grid must be GridConfig, got {type(self.grid)}")

        if not isinstance(self.action, ActionConfig):
            raise TypeError(f"action must be ActionConfig, got {type(self.action)}")

        if not isinstance(self.dataset, DatasetConfig):
            raise TypeError(f"dataset must be DatasetConfig, got {type(self.dataset)}")

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
    """Validate configuration consistency."""
    # Check reward config consistency
    if config.reward.reward_on_submit_only and config.reward.progress_bonus != 0.0:
        logger.warning("progress_bonus is ignored when reward_on_submit_only=True")

    # Check grid config consistency
    if config.grid.max_colors <= config.grid.background_color:
        raise ValueError(
            f"background_color ({config.grid.background_color}) must be < max_colors ({config.grid.max_colors})"
        )

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
