"""JaxARC utilities package.

This package contains utility functions and classes that support the core
functionality but are not part of the core environment or parsing logic.
"""

from __future__ import annotations

# Configuration utilities
from .config import (
    get_path,
    get_raw_path,
)

# Dataset utilities
from .dataset_downloader import DatasetDownloader, DatasetDownloadError
from .dataset_validation import get_dataset_recommendations, validate_dataset_config

# JAXTyping definitions for easy access
from .jax_types import (
    # Flexible types
    AnyGridArray,
    AnyMaskArray,
    AnySelectionArray,
    BboxActionData,
    BboxCoords,
    ColIndex,
    # Utility types
    ColorValue,
    ContinuousSelectionArray,
    EpisodeDone,
    EpisodeIndex,
    # Core grid types (support both single and batched with *batch modifier)
    GridArray,
    GridHeight,
    GridWidth,
    MaskActionData,
    MaskArray,
    OperationId,
    PointActionData,
    # Action types
    PointCoords,
    RewardValue,
    RowIndex,
    SelectionArray,
    # Scoring types (support both single and batched with *batch modifier)
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
    "DatasetDownloadError",
    "DatasetDownloader",
    "validate_dataset_config",
    "get_dataset_recommendations",
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
    # JAXTyping exports
    "GridArray",
    "MaskArray",
    "SelectionArray",
    "ContinuousSelectionArray",
    "TaskInputGrids",
    "TaskOutputGrids",
    "TaskInputMasks",
    "TaskOutputMasks",
    "PointCoords",
    "BboxCoords",
    "OperationId",
    "PointActionData",
    "BboxActionData",
    "MaskActionData",
    "SimilarityScore",
    "RewardValue",
    "StepCount",
    "EpisodeIndex",
    "TaskIndex",
    "EpisodeDone",
    "ColorValue",
    "GridHeight",
    "GridWidth",
    "RowIndex",
    "ColIndex",
    "AnyGridArray",
    "AnyMaskArray",
    "AnySelectionArray",
]
