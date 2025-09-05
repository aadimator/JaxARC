"""JaxARC utilities package.

This package contains essential utility functions and classes that support the core
functionality but are not part of the core environment or parsing logic.

This module has been simplified to only export utilities that are actually used
throughout the codebase, following the KISS principle.
"""

from __future__ import annotations

# Essential JAXTyping definitions (now in types.py)
from ..types import (
    EPISODE_MODE_TEST,
    EPISODE_MODE_TRAIN,
    MAX_GRID_SIZE,
    NUM_COLORS,
    # Constants
    NUM_OPERATIONS,
    ColorValue,
    # Core grid types (support both single and batched with *batch modifier)
    GridArray,
    GridHeight,
    GridWidth,
    MaskArray,
    ObservationArray,
    # Action types
    OperationId,
    OperationMask,
    PairIndex,
    # Utility types
    PRNGKey,
    RewardValue,
    SelectionArray,
    SimilarityScore,
    # Environment state types
    StepCount,
    TaskIndex,
    # Task data types
    TaskInputGrids,
    TaskInputMasks,
    TaskOutputGrids,
    TaskOutputMasks,
)

# Configuration utilities
from .core import (
    get_path,
    get_raw_path,
)

# Dataset utilities
from .dataset_manager import DatasetError, DatasetManager

# Task management utilities
from .task_manager import (
    TaskIDManager,
    TemporaryTaskManager,
    create_jax_task_index,
    extract_task_id_from_index,
    get_global_task_manager,
    get_jax_task_index,
    get_task_id_globally,
    get_task_index_globally,
    is_dummy_task_index,
    register_task_globally,
)

__all__ = [
    # Dataset utilities
    "DatasetManager",
    "DatasetError",
    # Configuration utilities
    "get_path",
    "get_raw_path",
    # Task management utilities
    "TaskIDManager",
    "get_global_task_manager",
    "register_task_globally",
    "get_task_index_globally",
    "get_task_id_globally",
    "get_jax_task_index",
    "create_jax_task_index",
    "extract_task_id_from_index",
    "is_dummy_task_index",
    "TemporaryTaskManager",
    # Essential JAXTyping exports
    "GridArray",
    "MaskArray",
    "SelectionArray",
    "TaskInputGrids",
    "TaskOutputGrids",
    "TaskInputMasks",
    "TaskOutputMasks",
    "OperationId",
    "OperationMask",
    "SimilarityScore",
    "RewardValue",
    "ObservationArray",
    "StepCount",
    "TaskIndex",
    "PairIndex",
    "ColorValue",
    "GridHeight",
    "GridWidth",
    "PRNGKey",
    "NUM_OPERATIONS",
    "NUM_COLORS",
    "MAX_GRID_SIZE",
    "EPISODE_MODE_TRAIN",
    "EPISODE_MODE_TEST",
]
