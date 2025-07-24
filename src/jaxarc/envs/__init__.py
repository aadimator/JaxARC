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

# Action space controller
from .action_space_controller import ActionSpaceController



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
from .config import (
    ActionConfig as UnifiedActionConfig,
)

# Unified configuration system (Equinox-based)
from .config import (
    ConfigValidationError,
    EnvironmentConfig,
    JaxArcConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)

# Episode management system
from .episode_manager import ArcEpisodeConfig, ArcEpisodeManager
from .config import (
    DatasetConfig as UnifiedDatasetConfig,
)
from .config import (
    RewardConfig as UnifiedRewardConfig,
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

# Observation system
from .observations import (
    ArcObservation,
    ObservationConfig,
    create_debug_observation,
    create_evaluation_observation,
    create_minimal_observation,
    create_observation,
    create_rich_observation,
    create_standard_observation,
    create_training_observation,
)

# Action and observation spaces
from .spaces import MultiBinary, Space

__all__ = [
    "OPERATION_NAMES",
    "ActionSpaceController",
    "ArcEnvState",
    "ArcEnvironment",
    "ArcEpisodeConfig",
    "ArcEpisodeManager",
    "ArcObservation",
    "ConfigFactory",
    "ConfigPresets",
    "ConfigValidationError",
    "EnvironmentConfig",
    "JaxArcConfig",
    "LoggingConfig",
    "MultiBinary",
    "ObservationConfig",
    "Space",
    "StorageConfig",
    "UnifiedActionConfig",
    "UnifiedDatasetConfig",
    "UnifiedRewardConfig",
    "VisualizationConfig",
    "WandbConfig",
    "arc_reset",
    "arc_reset_with_hydra",
    "arc_step",
    "arc_step_with_hydra",
    "bbox_handler",
    "create_debug_observation",
    "create_development_config",
    "create_evaluation_observation",
    "create_minimal_observation",
    "create_observation",
    "create_production_config",
    "create_research_config",
    "create_rich_observation",
    "create_standard_observation",
    "create_training_observation",
    "execute_grid_operation",
    "from_hydra",
    "from_preset",
    "get_action_handler",
    "get_all_operation_ids",
    "get_available_presets",
    "get_operation_category",
    "get_operation_display_text",
    "get_operation_name",
    "get_operations_by_category",
    "is_valid_operation_id",
    "mask_handler",
    "point_handler",
]
