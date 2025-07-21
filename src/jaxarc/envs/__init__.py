"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
environments with both functional and class-based APIs.
"""

from __future__ import annotations

# State definition (centralized)
from ..state import ArcEnvState

# Action handlers
from .actions import bbox_handler, get_action_handler, mask_handler, point_handler

# Configuration system
from .config import (
    ActionConfig,
    ArcEnvConfig,
    DatasetConfig,
    GridConfig,
    RewardConfig,
    config_from_dict,
    get_config_summary,
    merge_configs,
    validate_config,
)

# Configuration factory system
from .config_factory import (
    ConfigFactory,
    ConfigPresets,
    create_development_config,
    create_production_config,
    create_research_config,
    from_hydra,
    from_preset,
    get_available_presets,
)

# Core environment classes
from .environment import ArcEnvironment
from .equinox_config import (
    ActionConfig as UnifiedActionConfig,
)

# Unified configuration system (Equinox-based)
from .equinox_config import (
    ConfigValidationError,
    EnvironmentConfig,
    JaxArcConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
    convert_arc_env_config_to_jax_arc_config,
)
from .equinox_config import (
    DatasetConfig as UnifiedDatasetConfig,
)
from .equinox_config import (
    RewardConfig as UnifiedRewardConfig,
)

# Factory functions for creating configurations
from .factory import (
    CONFIG_PRESETS,
    DATASET_PRESETS,
    TRAINING_PRESETS,
    create_bbox_config,
    create_complete_hydra_config,
    create_conceptarc_config,
    create_config_from_hydra,
    create_config_with_hydra_parser,
    create_config_with_parser,
    create_config_with_task_sampler,
    create_dataset_config,
    create_evaluation_config,
    create_full_config,
    create_miniarc_config,
    create_point_config,
    create_raw_config,
    create_restricted_config,
    create_standard_config,
    create_training_config,
    get_preset_config,
)

# Functional API
from .functional import (
    arc_reset,
    arc_reset_with_hydra,
    arc_step,
    arc_step_with_hydra,
)

# Grid operations
from .grid_operations import execute_grid_operation

# Operation definitions and utilities
from .operations import (
    OPERATION_NAMES,
    get_all_operation_ids,
    get_operation_category,
    get_operation_display_text,
    get_operation_name,
    get_operations_by_category,
    is_valid_operation_id,
)

# Action and observation spaces
from .spaces import MultiBinary, Space

__all__ = [
    # Core environment classes
    "ArcEnvironment",
    "ArcEnvState",
    # Configuration classes (legacy)
    "ArcEnvConfig",
    "RewardConfig",
    "GridConfig",
    "ActionConfig",
    "DatasetConfig",
    # Unified configuration classes
    "JaxArcConfig",
    "EnvironmentConfig",
    "UnifiedDatasetConfig",
    "UnifiedActionConfig",
    "UnifiedRewardConfig",
    "VisualizationConfig",
    "StorageConfig",
    "LoggingConfig",
    "WandbConfig",
    "ConfigValidationError",
    "convert_arc_env_config_to_jax_arc_config",
    # Configuration factory system
    "ConfigFactory",
    "ConfigPresets",
    "create_development_config",
    "create_research_config",
    "create_production_config",
    "from_hydra",
    "from_preset",
    "get_available_presets",
    # Config utilities
    "validate_config",
    "get_config_summary",
    "config_from_dict",
    "merge_configs",
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
    "create_conceptarc_config",
    "create_miniarc_config",
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
    # Action handlers
    "get_action_handler",
    "point_handler",
    "bbox_handler",
    "mask_handler",
    # Grid operations
    "execute_grid_operation",
    # Operation definitions
    "OPERATION_NAMES",
    "get_operation_name",
    "get_operation_display_text",
    "is_valid_operation_id",
    "get_all_operation_ids",
    "get_operations_by_category",
    "get_operation_category",
    # Action and observation spaces
    "Space",
    "MultiBinary",
]
