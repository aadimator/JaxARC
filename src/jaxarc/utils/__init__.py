"""JaxARC utilities package.

This package contains utility functions and classes that support the core
functionality but are not part of the core environment or parsing logic.
"""

from __future__ import annotations

# Dataset utilities
from .dataset_downloader import DatasetDownloader, DatasetDownloadError
from .dataset_validation import validate_dataset_config, get_dataset_recommendations

# Configuration utilities  
from .config import (
    get_config,
    get_path,
    get_raw_path,
    get_processed_path,
    get_interim_path,
    get_external_path,
)

# Task management utilities
from .task_manager import (
    TaskIDManager,
    get_global_task_manager,
    register_task_globally,
    get_task_index_globally,
    get_task_id_globally,
    get_jax_task_index,
    create_jax_task_index,
    extract_task_id_from_index,
    is_dummy_task_index,
    TemporaryTaskManager,
)

# JAXTyping definitions for easy access
from .jax_types import (
    # Core grid types (support both single and batched with *batch modifier)
    GridArray,
    MaskArray,
    SelectionArray,
    ContinuousSelectionArray,
    
    # Task data types
    TaskInputGrids,
    TaskInputMasks,
    TaskOutputGrids,
    TaskOutputMasks,
    
    # Action types
    PointCoords,
    BboxCoords,
    OperationId,
    PointActionData,
    BboxActionData,
    MaskActionData,
    
    # Scoring types (support both single and batched with *batch modifier)
    SimilarityScore,
    RewardValue,
    
    # Environment state types
    StepCount,
    EpisodeIndex,
    TaskIndex,
    EpisodeDone,
    
    # Utility types
    ColorValue,
    GridHeight,
    GridWidth,
    RowIndex,
    ColIndex,
    
    # Flexible types
    AnyGridArray,
    AnyMaskArray,
    AnySelectionArray,
)

__all__ = [
    # Dataset utilities
    "DatasetDownloadError",
    "DatasetDownloader", 
    "validate_dataset_config",
    "get_dataset_recommendations",
    
    # Configuration utilities
    "get_config",
    "get_path",
    "get_raw_path", 
    "get_processed_path",
    "get_interim_path",
    "get_external_path",
    
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
