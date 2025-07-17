"""JaxARC utilities package."""

from __future__ import annotations

from .dataset_downloader import DatasetDownloader, DatasetDownloadError

# Import JAXTyping definitions for easy access
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
    "DatasetDownloadError",
    "DatasetDownloader",
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
