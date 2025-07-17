"""
Factory functions for creating ARC environment configurations.

This module provides convenient factory functions for creating different types
of ARC environment configurations with sensible defaults and Hydra integration.

DEPRECATION NOTICE:
These factory functions are maintained for backward compatibility but are deprecated
in favor of Hydra's native composition system. For new code, prefer using Hydra
configuration files and the `create_complete_hydra_config()` function.

See the migration guide in docs/configuration.md for examples of how to replace
factory functions with Hydra configurations.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .config import ActionConfig, ArcEnvConfig, DatasetConfig, GridConfig, RewardConfig


def create_raw_config(
    max_episode_steps: int = 50,
    step_penalty: float = 0.0,
    success_bonus: float = 1.0,
    dataset_name: str = "arc-agi-1",
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create minimal configuration for basic ARC environment.

    DEPRECATED: Use Hydra configuration instead. See migration guide in docs/configuration.md
    
    Hydra equivalent:
        @hydra.main(config_path="conf", config_name="presets/raw")
        def main(cfg: DictConfig):
            config = create_complete_hydra_config(cfg)

    Suitable for simple experiments and debugging. Only allows fill colors (0-9),
    resize (33), and submit (34) operations.

    Args:
        max_episode_steps: Maximum steps per episode
        step_penalty: Penalty per step (should be <= 0)
        success_bonus: Bonus for solving the task
        dataset_name: Dataset name for configuration
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig with minimal settings
    """
    warnings.warn(
        "create_raw_config() is deprecated. Use Hydra configuration with "
        "create_complete_hydra_config() instead. See docs/configuration.md for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    reward_config = RewardConfig(
        reward_on_submit_only=False,
        step_penalty=step_penalty,
        success_bonus=success_bonus,
        similarity_weight=1.0,
        progress_bonus=0.0,
        invalid_action_penalty=0.0,
    )

    grid_config = GridConfig(
        max_grid_height=30,
        max_grid_width=30,
        min_grid_height=3,
        min_grid_width=3,
        max_colors=10,
        background_color=0,
    )

    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        num_operations=35,
        allowed_operations=[
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
            33,
            34,
        ],  # Fill colors + resize + submit
        validate_actions=False,
        clip_invalid_actions=True,
    )

    dataset_config = DatasetConfig(
        dataset_name=dataset_name,
        task_split="train",
    )

    config = ArcEnvConfig(
        max_episode_steps=max_episode_steps,
        auto_reset=True,
        log_operations=False,
        log_grid_changes=False,
        log_rewards=False,
        strict_validation=False,
        allow_invalid_actions=True,
        reward=reward_config,
        grid=grid_config,
        action=action_config,
        dataset=dataset_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_standard_config(
    max_episode_steps: int = 100,
    reward_on_submit_only: bool = True,
    step_penalty: float = -0.01,
    success_bonus: float = 10.0,
    log_operations: bool = False,
    dataset_name: str = "arc-agi-1",
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create standard configuration for ARC environment.

    DEPRECATED: Use Hydra configuration instead. See migration guide in docs/configuration.md
    
    Hydra equivalent:
        @hydra.main(config_path="conf", config_name="presets/standard")
        def main(cfg: DictConfig):
            config = create_complete_hydra_config(cfg)

    Balanced settings suitable for most training scenarios.

    Args:
        max_episode_steps: Maximum steps per episode
        reward_on_submit_only: Whether to give rewards only on submit
        step_penalty: Penalty per step (should be negative)
        success_bonus: Bonus for solving the task
        log_operations: Whether to log operations
        dataset_name: Dataset name for configuration
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig with standard settings
    """
    warnings.warn(
        "create_standard_config() is deprecated. Use Hydra configuration with "
        "create_complete_hydra_config() instead. See docs/configuration.md for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    reward_config = RewardConfig(
        reward_on_submit_only=reward_on_submit_only,
        step_penalty=step_penalty,
        success_bonus=success_bonus,
        similarity_weight=1.0,
        progress_bonus=0.1,
        invalid_action_penalty=-0.1,
    )

    grid_config = GridConfig(
        max_grid_height=30,
        max_grid_width=30,
        min_grid_height=3,
        min_grid_width=3,
        max_colors=10,
        background_color=0,
    )

    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        num_operations=35,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    dataset_config = DatasetConfig(
        dataset_name=dataset_name,
        task_split="train",
    )

    config = ArcEnvConfig(
        max_episode_steps=max_episode_steps,
        auto_reset=True,
        log_operations=log_operations,
        log_grid_changes=False,
        log_rewards=False,
        strict_validation=True,
        allow_invalid_actions=False,
        reward=reward_config,
        grid=grid_config,
        action=action_config,
        dataset=dataset_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_full_config(
    max_episode_steps: int = 200,
    dataset_name: str = "arc-agi-1",
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create full-featured configuration with all logging and validation enabled.

    Suitable for detailed analysis and debugging.

    Args:
        max_episode_steps: Maximum steps per episode
        dataset_name: Dataset name for configuration
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig with full features enabled
    """
    reward_config = RewardConfig(
        reward_on_submit_only=False,
        step_penalty=-0.01,
        success_bonus=10.0,
        similarity_weight=1.0,
        progress_bonus=0.2,
        invalid_action_penalty=-0.2,
    )

    grid_config = GridConfig(
        max_grid_height=30,
        max_grid_width=30,
        min_grid_height=3,
        min_grid_width=3,
        max_colors=10,
        background_color=0,
    )

    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        num_operations=35,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    dataset_config = DatasetConfig(
        dataset_name=dataset_name,
        task_split="train",
    )

    config = ArcEnvConfig(
        max_episode_steps=max_episode_steps,
        auto_reset=True,
        log_operations=True,
        log_grid_changes=True,
        log_rewards=True,
        strict_validation=True,
        allow_invalid_actions=False,
        reward=reward_config,
        grid=grid_config,
        action=action_config,
        dataset=dataset_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_point_config(
    max_episode_steps: int = 150,
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration for point-based actions.

    Actions specify single points rather than selection masks.

    Args:
        max_episode_steps: Maximum steps per episode
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig configured for point actions
    """
    base_config = create_standard_config(max_episode_steps=max_episode_steps)

    # Modify action config for point-based actions
    action_config = ActionConfig(
        selection_format="point",
        selection_threshold=0.5,
        allow_partial_selection=False,  # Not relevant for point actions
        num_operations=35,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    config = ArcEnvConfig(
        max_episode_steps=base_config.max_episode_steps,
        auto_reset=base_config.auto_reset,
        log_operations=base_config.log_operations,
        log_grid_changes=base_config.log_grid_changes,
        log_rewards=base_config.log_rewards,
        strict_validation=base_config.strict_validation,
        allow_invalid_actions=base_config.allow_invalid_actions,
        reward=base_config.reward,
        grid=base_config.grid,
        action=action_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_bbox_config(
    max_episode_steps: int = 120,
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration for bounding box actions.

    Actions specify rectangular regions rather than selection masks.

    Args:
        max_episode_steps: Maximum steps per episode
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig configured for bbox actions
    """
    base_config = create_standard_config(max_episode_steps=max_episode_steps)

    # Modify action config for bbox-based actions
    action_config = ActionConfig(
        selection_format="bbox",
        selection_threshold=0.5,
        allow_partial_selection=False,  # Not relevant for bbox actions
        num_operations=35,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    config = ArcEnvConfig(
        max_episode_steps=base_config.max_episode_steps,
        auto_reset=base_config.auto_reset,
        log_operations=base_config.log_operations,
        log_grid_changes=base_config.log_grid_changes,
        log_rewards=base_config.log_rewards,
        strict_validation=base_config.strict_validation,
        allow_invalid_actions=base_config.allow_invalid_actions,
        reward=base_config.reward,
        grid=base_config.grid,
        action=action_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_restricted_config(
    max_episode_steps: int = 80,
    allowed_operations: Optional[list[int]] = None,
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration with restricted action space.

    Suitable for curriculum learning or specialized training.

    Args:
        max_episode_steps: Maximum steps per episode
        allowed_operations: List of allowed operation IDs (None for all)
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig with restricted settings
    """
    # Start with standard config
    base_config = create_standard_config(max_episode_steps=max_episode_steps)

    # Restrict to basic operations if none specified
    if allowed_operations is None:
        allowed_operations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 34]  # Fill colors + submit

    # Modify action config
    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.7,  # Higher threshold for more decisive selections
        allow_partial_selection=False,
        num_operations=max(allowed_operations) + 1 if allowed_operations else 35,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    # More restrictive reward settings
    reward_config = RewardConfig(
        reward_on_submit_only=True,
        step_penalty=-0.02,  # Higher penalty to encourage efficiency
        success_bonus=15.0,  # Higher bonus for success
        similarity_weight=1.0,
        progress_bonus=0.0,  # No intermediate rewards
        invalid_action_penalty=-0.5,
    )

    config = ArcEnvConfig(
        max_episode_steps=base_config.max_episode_steps,
        auto_reset=base_config.auto_reset,
        log_operations=base_config.log_operations,
        log_grid_changes=base_config.log_grid_changes,
        log_rewards=base_config.log_rewards,
        strict_validation=True,
        allow_invalid_actions=False,
        reward=reward_config,
        grid=base_config.grid,
        action=action_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


def create_config_from_hydra(
    hydra_config: DictConfig,
    base_config: Optional[ArcEnvConfig] = None,
    parser: Optional[Any] = None,
) -> ArcEnvConfig:
    """
    Create configuration from Hydra config with optional base config and parser.

    Args:
        hydra_config: Hydra configuration
        base_config: Optional base configuration to merge with
        parser: Optional pre-initialized parser (e.g., from Hydra instantiation)

    Returns:
        ArcEnvConfig created from Hydra config
    """
    if base_config is None:
        return ArcEnvConfig.from_hydra(hydra_config, parser=parser)

    # Merge base config with Hydra config
    from .config import merge_configs

    merged_config = merge_configs(base_config, hydra_config)

    # Add parser if provided
    if parser is not None:
        merged_config = create_config_with_parser(merged_config, parser)

    return merged_config


def create_training_config(
    curriculum_level: str = "standard",
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration optimized for training at different curriculum levels.

    Args:
        curriculum_level: "basic", "standard", "advanced", or "expert"
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig optimized for training
    """
    if curriculum_level == "basic":
        # Extract max_episode_steps from kwargs to avoid conflict
        max_steps = kwargs.pop("max_episode_steps", 50)
        return create_restricted_config(
            max_episode_steps=max_steps,
            allowed_operations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 34],  # Fill + submit
            **kwargs,
        )
    if curriculum_level == "standard":
        # Extract conflicting parameters from kwargs
        max_steps = kwargs.pop("max_episode_steps", 100)
        return create_standard_config(
            max_episode_steps=max_steps,
            reward_on_submit_only=True,
            step_penalty=-0.01,
            success_bonus=10.0,
            **kwargs,
        )
    if curriculum_level == "advanced":
        # Extract max_episode_steps from kwargs to avoid conflict
        max_steps = kwargs.pop("max_episode_steps", 150)
        return create_full_config(max_episode_steps=max_steps, **kwargs)
    if curriculum_level == "expert":
        # Extract conflicting parameters from kwargs
        max_steps = kwargs.pop("max_episode_steps", 200)
        return create_standard_config(
            max_episode_steps=max_steps,
            reward_on_submit_only=False,
            step_penalty=-0.005,
            success_bonus=20.0,
            **kwargs,
        )
    raise ValueError(f"Unknown curriculum level: {curriculum_level}")


def create_evaluation_config(
    strict_mode: bool = True,
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration optimized for evaluation.

    Args:
        strict_mode: Whether to use strict validation
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig optimized for evaluation
    """
    reward_config = RewardConfig(
        reward_on_submit_only=True,
        step_penalty=0.0,  # No step penalty during evaluation
        success_bonus=1.0,  # Simple binary reward
        similarity_weight=1.0,
        progress_bonus=0.0,
        invalid_action_penalty=0.0,
    )

    grid_config = GridConfig(
        max_grid_height=30,
        max_grid_width=30,
        min_grid_height=3,
        min_grid_width=3,
        max_colors=10,
        background_color=0,
    )

    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        num_operations=35,
        validate_actions=strict_mode,
        clip_invalid_actions=True,
    )

    config = ArcEnvConfig(
        max_episode_steps=100,
        auto_reset=True,
        log_operations=False,
        log_grid_changes=False,
        log_rewards=False,
        strict_validation=strict_mode,
        allow_invalid_actions=not strict_mode,
        reward=reward_config,
        grid=grid_config,
        action=action_config,
    )

    # Apply any additional overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        from .config import config_from_dict

        config = config_from_dict(config_dict)

    return config


# Convenience dictionaries for easy access
CONFIG_PRESETS = {
    "raw": create_raw_config,
    "standard": create_standard_config,
    "full": create_full_config,
    "point": create_point_config,
    "bbox": create_bbox_config,
    "restricted": create_restricted_config,
    "evaluation": create_evaluation_config,
}

DATASET_PRESETS = {
    "arc-agi-1": lambda **kwargs: create_dataset_config("arc-agi-1", **kwargs),
    "concept-arc": lambda **kwargs: create_conceptarc_config(**kwargs),
    "mini-arc": lambda **kwargs: create_miniarc_config(**kwargs),
    "re-arc": lambda **kwargs: create_dataset_config("re-arc", **kwargs),
}

TRAINING_PRESETS = {
    "basic": lambda **kwargs: create_training_config("basic", **kwargs),
    "standard": lambda **kwargs: create_training_config("standard", **kwargs),
    "advanced": lambda **kwargs: create_training_config("advanced", **kwargs),
    "expert": lambda **kwargs: create_training_config("expert", **kwargs),
}


def create_dataset_config(
    dataset_name: str,
    task_split: str = "train",
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration optimized for specific datasets.

    Args:
        dataset_name: Dataset name ("arc-agi-1", "concept-arc", "mini-arc", etc.)
        task_split: Dataset split to use
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig optimized for the specified dataset
    """
    # Dataset-specific configurations
    dataset_configs = {
        "arc-agi-1": {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "max_colors": 10,
            "selection_format": "mask",
        },
        "concept-arc": {
            "max_grid_height": 15,
            "max_grid_width": 15,
            "max_colors": 10,
            "selection_format": "mask",
        },
        "mini-arc": {
            "max_grid_height": 5,
            "max_grid_width": 5,
            "max_colors": 10,
            "selection_format": "point",  # Point actions work well for small grids
        },
        "re-arc": {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "max_colors": 10,
            "selection_format": "mask",
        },
    }

    if dataset_name not in dataset_configs:
        logger.warning(
            f"Unknown dataset '{dataset_name}', using default ARC-AGI-1 settings"
        )
        dataset_settings = dataset_configs["arc-agi-1"]
    else:
        dataset_settings = dataset_configs[dataset_name]

    # Create base config with dataset-specific settings
    base_config = create_standard_config(dataset_name=dataset_name, **kwargs)

    # Override with dataset-specific settings
    dataset_config = DatasetConfig(
        dataset_name=dataset_name,
        task_split=task_split,
        dataset_max_grid_height=dataset_settings["max_grid_height"],
        dataset_max_grid_width=dataset_settings["max_grid_width"],
        dataset_max_colors=dataset_settings["max_colors"],
    )

    action_config = ActionConfig(
        selection_format=dataset_settings["selection_format"],
        selection_threshold=base_config.action.selection_threshold,
        allow_partial_selection=base_config.action.allow_partial_selection,
        num_operations=base_config.action.num_operations,
        allowed_operations=base_config.action.allowed_operations,
        validate_actions=base_config.action.validate_actions,
        clip_invalid_actions=base_config.action.clip_invalid_actions,
    )

    # Apply dataset overrides to grid config
    effective_grid_config = dataset_config.get_effective_grid_config(base_config.grid)

    return ArcEnvConfig(
        max_episode_steps=base_config.max_episode_steps,
        auto_reset=base_config.auto_reset,
        log_operations=base_config.log_operations,
        log_grid_changes=base_config.log_grid_changes,
        log_rewards=base_config.log_rewards,
        strict_validation=base_config.strict_validation,
        allow_invalid_actions=base_config.allow_invalid_actions,
        reward=base_config.reward,
        grid=effective_grid_config,
        action=action_config,
        dataset=dataset_config,
    )


def get_preset_config(preset_name: str, **kwargs: Any) -> ArcEnvConfig:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of the preset configuration
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig for the specified preset

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name in CONFIG_PRESETS:
        return CONFIG_PRESETS[preset_name](**kwargs)
    if preset_name in TRAINING_PRESETS:
        return TRAINING_PRESETS[preset_name](**kwargs)
    if preset_name in DATASET_PRESETS:
        return DATASET_PRESETS[preset_name](**kwargs)
    available = (
        list(CONFIG_PRESETS.keys())
        + list(TRAINING_PRESETS.keys())
        + list(DATASET_PRESETS.keys())
    )
    raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")


def create_config_with_parser(
    base_config: ArcEnvConfig,
    parser: Any,
) -> ArcEnvConfig:
    """
    Create config with a parser instance.

    Args:
        base_config: Base configuration
        parser: Parser instance (e.g., ArcAgiParser) that can sample tasks

    Returns:
        ArcEnvConfig with parser attached
    """
    # Create new config with parser (need to work around frozen dataclass)
    return ArcEnvConfig(
        max_episode_steps=base_config.max_episode_steps,
        auto_reset=base_config.auto_reset,
        log_operations=base_config.log_operations,
        log_grid_changes=base_config.log_grid_changes,
        log_rewards=base_config.log_rewards,
        strict_validation=base_config.strict_validation,
        allow_invalid_actions=base_config.allow_invalid_actions,
        reward=base_config.reward,
        grid=base_config.grid,
        action=base_config.action,
        dataset=base_config.dataset,
        parser=parser,
    )


def create_config_with_hydra_parser(
    hydra_config: DictConfig,
    dataset_config: Optional[DictConfig] = None,
) -> ArcEnvConfig:
    """
    Create config with parser instantiated from Hydra dataset configuration.

    Args:
        hydra_config: Complete Hydra configuration
        dataset_config: Optional dataset configuration override

    Returns:
        ArcEnvConfig with parser instantiated from Hydra config
    """
    from hydra.utils import instantiate

    # Use provided dataset config or extract from hydra_config
    dataset_cfg = dataset_config or hydra_config.get("dataset", {})

    # Instantiate parser from dataset config
    parser_config = dataset_cfg.get("parser", {})
    if parser_config and "_target_" in parser_config:
        # Create parser config for instantiation
        parser_cfg = OmegaConf.create(
            {
                **dataset_cfg,  # Include dataset settings
                **parser_config,  # Include parser-specific settings
            }
        )
        parser = instantiate(parser_cfg)
    else:
        logger.warning("No parser configuration found in dataset config")
        parser = None

    return create_config_from_hydra(hydra_config, parser=parser)


def create_complete_hydra_config(
    hydra_config: DictConfig,
) -> ArcEnvConfig:
    """
    Create a complete ARC environment configuration using existing Hydra infrastructure.

    This function leverages the existing dataset configurations and parser setup
    to create a fully functional environment configuration.

    Args:
        hydra_config: Complete Hydra configuration (typically from @hydra.main)

    Returns:
        ArcEnvConfig with parser instantiated from dataset configuration

    Example:
        ```python
        @hydra.main(config_path="../../conf", config_name="config")
        def main(cfg: DictConfig):
            env_config = create_complete_hydra_config(cfg)
            key = jax.random.PRNGKey(42)
            state, obs = arc_reset(key, env_config)
        ```
    """
    from hydra.utils import instantiate

    # Get dataset configuration
    dataset_cfg = hydra_config.get("dataset", {})

    # Instantiate parser from dataset configuration
    parser = None
    if "parser" in dataset_cfg and "_target_" in dataset_cfg.parser:
        try:
            # Create parser configuration with dataset settings
            parser_cfg = OmegaConf.create(
                {
                    **dataset_cfg,  # Include all dataset settings
                }
            )
            parser = instantiate(parser_cfg.parser, cfg=parser_cfg)
            logger.info(f"Instantiated parser: {parser.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to instantiate parser: {e}")
            parser = None

    # Get environment configuration
    env_cfg = hydra_config.get("environment", {})

    # Create base environment configuration
    base_config = ArcEnvConfig.from_hydra(env_cfg)

    # Apply dataset-specific grid config if present
    if dataset_cfg and "grid" in dataset_cfg:
        from .config import GridConfig

        # Extract grid settings from dataset config
        grid_cfg = dataset_cfg.grid
        updated_grid = GridConfig(
            max_grid_height=grid_cfg.get(
                "max_grid_height", base_config.grid.max_grid_height
            ),
            max_grid_width=grid_cfg.get(
                "max_grid_width", base_config.grid.max_grid_width
            ),
            min_grid_height=grid_cfg.get(
                "min_grid_height", base_config.grid.min_grid_height
            ),
            min_grid_width=grid_cfg.get(
                "min_grid_width", base_config.grid.min_grid_width
            ),
            max_colors=grid_cfg.get("max_colors", base_config.grid.max_colors),
            background_color=grid_cfg.get(
                "background_color", base_config.grid.background_color
            ),
        )

        base_config = ArcEnvConfig(
            max_episode_steps=base_config.max_episode_steps,
            auto_reset=base_config.auto_reset,
            log_operations=base_config.log_operations,
            log_grid_changes=base_config.log_grid_changes,
            log_rewards=base_config.log_rewards,
            strict_validation=base_config.strict_validation,
            allow_invalid_actions=base_config.allow_invalid_actions,
            reward=base_config.reward,
            grid=updated_grid,
            action=base_config.action,
            dataset=base_config.dataset,
            parser=base_config.parser,
        )

    # Attach parser to configuration
    if parser is not None:
        return create_config_with_parser(base_config, parser)
    logger.warning("No parser available - environment will use demo tasks")
    return base_config


def create_conceptarc_config(
    max_episode_steps: int = 120,
    task_split: str = "corpus",
    reward_on_submit_only: bool = True,
    step_penalty: float = -0.01,
    success_bonus: float = 10.0,
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration optimized for ConceptARC dataset.

    ConceptARC is organized around 16 concept groups with 10 tasks each,
    designed to systematically assess abstraction and generalization abilities.
    Uses standard ARC grid dimensions (up to 30x30) with concept-based organization.

    Args:
        max_episode_steps: Maximum steps per episode (default 120 for concept exploration)
        task_split: Dataset split to use (default "corpus" for ConceptARC)
        reward_on_submit_only: Whether to give rewards only on submit
        step_penalty: Penalty per step (should be negative)
        success_bonus: Bonus for solving the task
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig optimized for ConceptARC dataset

    Raises:
        ValueError: If configuration validation fails
    """
    try:
        # ConceptARC-specific reward configuration
        reward_config = RewardConfig(
            reward_on_submit_only=reward_on_submit_only,
            step_penalty=step_penalty,
            success_bonus=success_bonus,
            similarity_weight=1.0,
            progress_bonus=0.1,  # Small progress bonus for concept exploration
            invalid_action_penalty=-0.1,
        )

        # ConceptARC uses standard ARC grid dimensions
        grid_config = GridConfig(
            max_grid_height=30,  # Standard ARC dimensions
            max_grid_width=30,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
        )

        # ConceptARC works well with mask-based actions for concept reasoning
        action_config = ActionConfig(
            selection_format="mask",
            selection_threshold=0.5,
            allow_partial_selection=True,
            num_operations=35,
            validate_actions=True,
            clip_invalid_actions=True,
        )

        # ConceptARC-specific dataset configuration
        dataset_config = DatasetConfig(
            dataset_name="ConceptARC",
            task_split=task_split,
            dataset_max_grid_height=30,
            dataset_max_grid_width=30,
            dataset_max_colors=10,
            shuffle_tasks=True,
        )

        config = ArcEnvConfig(
            max_episode_steps=max_episode_steps,
            auto_reset=True,
            log_operations=False,
            log_grid_changes=False,
            log_rewards=False,
            strict_validation=True,
            allow_invalid_actions=False,
            reward=reward_config,
            grid=grid_config,
            action=action_config,
            dataset=dataset_config,
        )

        # Apply any additional overrides
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            from .config import config_from_dict

            config = config_from_dict(config_dict)

        # Validate the configuration
        from .config import validate_config

        validate_config(config)

        logger.info("Created ConceptARC configuration with concept-based organization")
        return config

    except Exception as e:
        logger.error(f"Failed to create ConceptARC configuration: {e}")
        raise ValueError(f"ConceptARC configuration error: {e}") from e


def create_miniarc_config(
    max_episode_steps: int = 80,
    task_split: str = "training",
    reward_on_submit_only: bool = True,
    step_penalty: float = -0.005,  # Lower penalty for faster iteration
    success_bonus: float = 5.0,  # Lower bonus for smaller tasks
    **kwargs: Any,
) -> ArcEnvConfig:
    """
    Create configuration optimized for MiniARC dataset.

    MiniARC is a 5x5 compact version of ARC with 400 training and 400 evaluation
    tasks, designed for faster experimentation and prototyping. Optimized for
    smaller grid dimensions and rapid iteration.

    Args:
        max_episode_steps: Maximum steps per episode (default 80 for faster iteration)
        task_split: Dataset split to use (default "training" for MiniARC)
        reward_on_submit_only: Whether to give rewards only on submit
        step_penalty: Penalty per step (lower for faster iteration)
        success_bonus: Bonus for solving the task (lower for smaller tasks)
        **kwargs: Additional config overrides

    Returns:
        ArcEnvConfig optimized for MiniARC dataset

    Raises:
        ValueError: If configuration validation fails
    """
    try:
        # MiniARC-specific reward configuration (optimized for smaller tasks)
        reward_config = RewardConfig(
            reward_on_submit_only=reward_on_submit_only,
            step_penalty=step_penalty,  # Lower penalty for rapid iteration
            success_bonus=success_bonus,  # Lower bonus for smaller tasks
            similarity_weight=1.0,
            progress_bonus=0.05,  # Small progress bonus for quick feedback
            invalid_action_penalty=-0.05,  # Lower penalty for experimentation
        )

        # MiniARC grid configuration optimized for 5x5 grids
        grid_config = GridConfig(
            max_grid_height=5,  # MiniARC 5x5 constraint
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
        )

        # MiniARC works well with point-based actions for small grids
        action_config = ActionConfig(
            selection_format="point",  # Point actions optimal for 5x5 grids
            selection_threshold=0.5,
            allow_partial_selection=False,  # Not relevant for point actions
            num_operations=35,
            validate_actions=True,
            clip_invalid_actions=True,
        )

        # MiniARC-specific dataset configuration
        dataset_config = DatasetConfig(
            dataset_name="MiniARC",
            task_split=task_split,
            dataset_max_grid_height=5,
            dataset_max_grid_width=5,
            dataset_max_colors=10,
            shuffle_tasks=True,
        )

        config = ArcEnvConfig(
            max_episode_steps=max_episode_steps,
            auto_reset=True,
            log_operations=False,
            log_grid_changes=False,
            log_rewards=False,
            strict_validation=True,
            allow_invalid_actions=False,
            reward=reward_config,
            grid=grid_config,
            action=action_config,
            dataset=dataset_config,
        )

        # Apply any additional overrides
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            from .config import config_from_dict

            config = config_from_dict(config_dict)

        # Validate the configuration
        from .config import validate_config

        validate_config(config)

        logger.info("Created MiniARC configuration optimized for 5x5 grids")
        return config

    except Exception as e:
        logger.error(f"Failed to create MiniARC configuration: {e}")
        raise ValueError(f"MiniARC configuration error: {e}") from e


def create_config_with_task_sampler(
    base_config: ArcEnvConfig,
    task_sampler: Callable,
) -> ArcEnvConfig:
    """
    Create config with a task sampler function.

    Args:
        base_config: Base configuration
        task_sampler: Function that takes (key, dataset_config) and returns JaxArcTask

    Returns:
        ArcEnvConfig with task sampler attached

    Deprecated: Use create_config_with_parser instead
    """
    warnings.warn(
        "create_config_with_task_sampler is deprecated. Use create_config_with_parser instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility, wrap the task_sampler in a simple parser-like object
    class TaskSamplerWrapper:
        def __init__(self, sampler):
            self.sampler = sampler

        def get_random_task(self, key):
            return self.sampler(key, base_config.dataset)

    wrapper = TaskSamplerWrapper(task_sampler)
    return create_config_with_parser(base_config, wrapper)
