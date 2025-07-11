"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
single-agent environments with grid operations.
"""

from __future__ import annotations

from .arc_base import ArcEnvironment, ArcEnvState
from .grid_operations import execute_grid_operation

# New config-based API
from .config import ArcEnvConfig, RewardConfig, GridConfig, ActionConfig, DatasetConfig
from .functional import arc_reset, arc_step, arc_reset_with_hydra, arc_step_with_hydra
from .factory import (
    create_raw_config,
    create_standard_config,
    create_full_config,
    create_point_config,
    create_bbox_config,
    create_restricted_config,
    create_config_from_hydra,
    create_training_config,
    create_evaluation_config,
    create_dataset_config,
    create_config_with_task_sampler,
    create_config_with_parser,
    create_config_with_hydra_parser,
    create_complete_hydra_config,
    get_preset_config,
    CONFIG_PRESETS,
    TRAINING_PRESETS,
    DATASET_PRESETS,
)


__all__ = [
    # Legacy ARC environments
    "ArcEnvironment",
    "ArcEnvState",
    "execute_grid_operation",
    # Config classes
    "ArcEnvConfig",
    "RewardConfig",
    "GridConfig",
    "ActionConfig",
    "DatasetConfig",
    # Functional API
    "arc_reset",
    "arc_step",
    "arc_reset_with_hydra",
    "arc_step_with_hydra",
    # Factory functions
    "create_raw_config",
    "create_standard_config",
    "create_full_config",
    "create_point_config",
    "create_bbox_config",
    "create_restricted_config",
    "create_config_from_hydra",
    "create_training_config",
    "create_evaluation_config",
    "create_dataset_config",
    "create_config_with_task_sampler",
    "create_config_with_parser",
    "create_config_with_hydra_parser",
    "create_complete_hydra_config",
    "get_preset_config",
    "CONFIG_PRESETS",
    "TRAINING_PRESETS",
    "DATASET_PRESETS",
]
