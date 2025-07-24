"""
Centralized JAXTyping definitions for JaxARC.

This module provides precise array type aliases using JAXTyping for better type safety,
documentation, and JAX compatibility. All array types include shape and dtype information
to catch errors early and improve code clarity.

Key Features:
- Precise shape annotations for all grid operations
- Unified types supporting both single and batched operations using *batch modifier
- Action-specific type aliases
- Task data structure types
- Runtime type validation support

JAXTyping Modifiers Used:
- *batch: Allows zero or more batch dimensions, enabling the same type to work for
  both single arrays (height, width) and batched arrays (batch1, batch2, ..., height, width)
- ...: Flexible variadic dimensions for utility types
- "": Scalar (zero-dimensional) arrays

Examples:
    # Single grid
    grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)  # shape: (2, 2)

    # Batched grids
    batch: GridArray = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32)  # shape: (2, 2, 2)

    # Both work with the same type annotation!
"""

from __future__ import annotations

from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int

# =============================================================================
# Core Grid Types (with optional batch dimensions using * modifier)
# =============================================================================

# Grid types that can handle both single grids and batched grids
GridArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing ARC grid(s) with color values 0-9.
Supports both single grids (height, width) and batched grids (*batch, height, width)."""

MaskArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing valid/invalid cells in grid(s).
Supports both single masks (height, width) and batched masks (*batch, height, width)."""

SelectionArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing selected cells for operations.
Supports both single selections (height, width) and batched selections (*batch, height, width)."""

ContinuousSelectionArray: TypeAlias = Float[Array, "*batch height width"]
"""Float array representing continuous selection weights in [0, 1].
Supports both single selections (height, width) and batched selections (*batch, height, width)."""

# =============================================================================
# Task Data Structure Types
# =============================================================================

# Task data arrays with maximum dimensions for padding
TaskInputGrids: TypeAlias = Int[Array, "max_pairs max_height max_width"]
"""Training/test input grids padded to maximum dimensions."""

TaskOutputGrids: TypeAlias = Int[Array, "max_pairs max_height max_width"]
"""Training/test output grids padded to maximum dimensions."""

TaskInputMasks: TypeAlias = Bool[Array, "max_pairs max_height max_width"]
"""Training/test input masks padded to maximum dimensions."""

TaskOutputMasks: TypeAlias = Bool[Array, "max_pairs max_height max_width"]
"""Training/test output masks padded to maximum dimensions."""

# =============================================================================
# Action Types
# =============================================================================

# Point-based action coordinates
PointCoords: TypeAlias = Int[Array, "2"]
"""Point coordinates as [row, col]."""

# Bounding box coordinates
BboxCoords: TypeAlias = Int[Array, "4"]
"""Bounding box coordinates as [r1, c1, r2, c2]."""

# Operation identifiers
OperationId: TypeAlias = Int[Array, ""]
"""Scalar integer representing an ARCLE operation (0-41 including enhanced control operations)."""

# Action data for different formats
PointActionData: TypeAlias = Int[Array, "2"]
"""Point action data: [row, col]."""

BboxActionData: TypeAlias = Int[Array, "4"]
"""Bounding box action data: [r1, c1, r2, c2]."""

MaskActionData: TypeAlias = Float[Array, "height_width"]
"""Flattened mask action data for reconstruction."""

# =============================================================================
# Similarity and Scoring Types (with optional batch dimensions)
# =============================================================================

SimilarityScore: TypeAlias = Float[Array, "*batch"]
"""Float array representing grid similarity score(s).
Supports both single scores () and batched scores (*batch)."""

# Reward and value types
RewardValue: TypeAlias = Float[Array, "*batch"]
"""Float array representing reward value(s).
Supports both single rewards () and batched rewards (*batch)."""

# Observation types
ObservationArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing observation(s) from the environment.
Supports both single observations (height, width) and batched observations (*batch, height, width)."""

# =============================================================================
# Environment State Types
# =============================================================================

# Step and episode counters
StepCount: TypeAlias = Int[Array, ""]
"""Scalar integer representing current step count."""

EpisodeIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing current episode/example index."""

TaskIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing task identifier."""

# Boolean flags
EpisodeDone: TypeAlias = Bool[Array, ""]
"""Scalar boolean indicating if episode is complete."""

# =============================================================================
# Utility Types
# =============================================================================

# Color values (0-9 for ARC grids)
ColorValue: TypeAlias = Int[Array, ""]
"""Scalar integer representing a color value (0-9)."""

# Padding value type
PaddingValue: TypeAlias = Int[Array, ""] | int
"""Padding value for grid operations (can be scalar or JAX array)."""

# PRNG key types
PRNGKey: TypeAlias = Int[Array, "2"]
"""JAX PRNG key array with shape (2,)."""

# Grid dimensions
GridHeight: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid height."""

GridWidth: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid width."""

# Coordinate types
RowIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing a row index."""

ColIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing a column index."""

# Bounding box type
BoundingBox: TypeAlias = tuple[RowIndex, RowIndex, ColIndex, ColIndex]
"""Bounding box coordinates as (min_row, max_row, min_col, max_col)."""

# =============================================================================
# Validation and Debug Types
# =============================================================================

# Types for validation and debugging
ValidationMask: TypeAlias = Bool[Array, "height width"]
"""Boolean mask for validation purposes."""

DebugInfo: TypeAlias = Float[Array, "..."]
"""Flexible array type for debug information."""

# =============================================================================
# Type Unions for Flexible APIs
# =============================================================================

# Union types for APIs that accept multiple formats
AnyGridArray: TypeAlias = Int[Array, "..."]
"""Flexible grid array type for APIs accepting various shapes."""

AnyMaskArray: TypeAlias = Bool[Array, "..."]
"""Flexible mask array type for APIs accepting various shapes."""

AnySelectionArray: TypeAlias = Bool[Array, "..."]
"""Flexible selection array type for APIs accepting various shapes."""

# =============================================================================
# Constants and Metadata
# =============================================================================

# Type annotations for common constants
MAX_GRID_SIZE = 30  # Maximum grid dimension in ARC
MAX_TRAIN_PAIRS = 10  # Maximum training pairs per task
MAX_TEST_PAIRS = 3  # Maximum test pairs per task
NUM_COLORS = 10  # Number of colors in ARC (0-9)
NUM_OPERATIONS = (
    42  # Number of ARCLE operations including enhanced control operations (0-41)
)

# Enhanced functionality constants
# Note: MAX_PAIRS removed - now dataset/config dependent
MAX_HISTORY_LENGTH = 1000  # Default maximum action history length
MAX_SELECTION_SIZE = MAX_GRID_SIZE * MAX_GRID_SIZE  # Maximum flattened selection size
MAX_GRID_SIZE_SQUARED = MAX_GRID_SIZE * MAX_GRID_SIZE  # For flattened grid operations
# Dynamic ACTION_RECORD_FIELDS calculation based on configuration
# This replaces the previous fixed constant approach


def get_selection_data_size(
    selection_format: str,
    max_grid_height: int = MAX_GRID_SIZE,
    max_grid_width: int = MAX_GRID_SIZE,
) -> int:
    """Calculate selection data size based on format and dataset configuration.

    This function determines the optimal size for storing selection data in action history
    based on the selection format and dataset constraints. This is much more efficient
    than using a fixed MAX_SELECTION_SIZE for all cases.

    Args:
        selection_format: Selection format ("point", "bbox", "mask")
        max_grid_height: Maximum grid height for the dataset
        max_grid_width: Maximum grid width for the dataset

    Returns:
        Number of elements needed to store selection data

    Examples:
        # MiniARC with point selection: only 2 elements needed
        size = get_selection_data_size("point", 5, 5)  # Returns 2

        # Full ARC with bbox selection: only 4 elements needed
        size = get_selection_data_size("bbox", 30, 30)  # Returns 4

        # MiniARC with mask selection: 25 elements (5x5)
        size = get_selection_data_size("mask", 5, 5)  # Returns 25

        # Full ARC with mask selection: 900 elements (30x30)
        size = get_selection_data_size("mask", 30, 30)  # Returns 900
    """
    if selection_format == "point":
        return 2  # [row, col]
    elif selection_format == "bbox":
        return 4  # [r1, c1, r2, c2]
    elif selection_format == "mask":
        return max_grid_height * max_grid_width  # flattened mask
    else:
        raise ValueError(f"Unknown selection format: {selection_format}")


def get_action_record_fields(
    selection_format: str,
    max_grid_height: int = MAX_GRID_SIZE,
    max_grid_width: int = MAX_GRID_SIZE,
) -> int:
    """Calculate total action record fields based on configuration.

    Args:
        selection_format: Selection format ("point", "bbox", "mask")
        max_grid_height: Maximum grid height for the dataset
        max_grid_width: Maximum grid width for the dataset

    Returns:
        Total number of fields in action record

    Examples:
        # MiniARC with point selection: 2 + 4 = 6 fields
        fields = get_action_record_fields("point", 5, 5)  # Returns 6

        # Full ARC with mask selection: 900 + 4 = 904 fields
        fields = get_action_record_fields("mask", 30, 30)  # Returns 904
    """
    selection_size = get_selection_data_size(
        selection_format, max_grid_height, max_grid_width
    )
    return selection_size + 4  # +4 for: operation_id, timestamp, pair_index, valid


# Backward compatibility: default ACTION_RECORD_FIELDS for mask format with max grid size
ACTION_RECORD_FIELDS = get_action_record_fields("mask", MAX_GRID_SIZE, MAX_GRID_SIZE)

# Default maximums for different datasets (can be overridden)
DEFAULT_MAX_TRAIN_PAIRS = 10  # Conservative default, can be increased for augmentation
DEFAULT_MAX_TEST_PAIRS = 4  # Reasonable default for most datasets

# Episode mode constants for JAX compatibility
EPISODE_MODE_TRAIN = 0  # Training mode
EPISODE_MODE_TEST = 1  # Test/evaluation mode

# Type aliases for these constants
MaxGridSize: TypeAlias = int
MaxTrainPairs: TypeAlias = int
MaxTestPairs: TypeAlias = int
NumColors: TypeAlias = int
NumOperations: TypeAlias = int
MaxTrainPairs: TypeAlias = int
MaxTestPairs: TypeAlias = int
MaxHistoryLength: TypeAlias = int
MaxSelectionSize: TypeAlias = int
ActionRecordFields: TypeAlias = int

# =============================================================================
# Enhanced ARC Step Logic Types
# =============================================================================

# Episode management types
EpisodeMode: TypeAlias = Int[Array, ""]
"""Scalar integer representing episode mode (0=train, 1=test) for JAX compatibility."""

# Flexible pair tracking types - sizes determined at runtime based on dataset config
AvailableTrainPairs: TypeAlias = Bool[Array, "max_train_pairs"]
"""Boolean mask indicating which training/demonstration pairs are available.
Size determined by dataset configuration and augmentation settings."""

AvailableTestPairs: TypeAlias = Bool[Array, "max_test_pairs"]
"""Boolean mask indicating which test pairs are available.
Size determined by dataset configuration (typically smaller than train pairs)."""

TrainCompletionStatus: TypeAlias = Bool[Array, "max_train_pairs"]
"""Boolean mask tracking completion status of training/demonstration pairs.
Size matches available train pairs for consistency."""

TestCompletionStatus: TypeAlias = Bool[Array, "max_test_pairs"]
"""Boolean mask tracking completion status of test pairs.
Size matches available test pairs for consistency."""

# Action history types
HistoryLength: TypeAlias = Int[Array, ""]
"""Scalar integer representing current length of action history."""

OperationMask: TypeAlias = Bool[Array, "num_operations"]
"""Boolean mask indicating which operations are currently allowed.
Uses num_operations dimension for dynamic action space control."""

# Action history storage types
SelectionData: TypeAlias = Float[Array, "max_selection_size"]
"""Flattened selection data for action history storage.
Accommodates point (2), bbox (4), or flattened mask data with padding."""

ActionSequence: TypeAlias = Int[Array, "sequence_length action_fields"]
"""Sequence of actions with fixed dimensions for JAX compatibility."""

# Action record structure for history tracking
# Note: ActionHistory will be defined as a structured array type
# Each record contains: selection_data, operation_id, timestamp, pair_index, valid flag
ActionHistory: TypeAlias = Float[Array, "max_history_length action_record_fields"]
"""Fixed-size action history storage with structured records.
Each record contains selection data, operation ID, timestamp, pair index, and validity flag.
Uses static shape with padding for JAX compatibility."""

# =============================================================================
# Enhanced Action Space Types
# =============================================================================

# Operation identifiers (updated range)
# Note: OperationId in the original section now covers the full range (0-41)

# Selection data variants for different action formats
PointSelectionData: TypeAlias = Int[Array, "2"]
"""Point selection data as [row, col] coordinates."""

BboxSelectionData: TypeAlias = Int[Array, "4"]
"""Bounding box selection data as [r1, c1, r2, c2] coordinates."""

MaskSelectionData: TypeAlias = Float[Array, "max_grid_size_squared"]
"""Flattened mask selection data with maximum grid size padding."""

# =============================================================================
# Episode Management Types
# =============================================================================

# Pair tracking and management
PairIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing index of current demonstration/test pair."""

PairCount: TypeAlias = Int[Array, ""]
"""Scalar integer representing total number of available pairs."""

# Episode termination and continuation
EpisodeContinuation: TypeAlias = Bool[Array, ""]
"""Scalar boolean indicating whether episode should continue."""

# Progress tracking
TrainProgressMetrics: TypeAlias = Float[Array, "max_train_pairs"]
"""Array of progress metrics for each training/demonstration pair."""

TestProgressMetrics: TypeAlias = Float[Array, "max_test_pairs"]
"""Array of progress metrics for each test pair."""

# =============================================================================
# Configuration and Validation Types
# =============================================================================

# Configuration validation
ConfigValidation: TypeAlias = Bool[Array, ""]
"""Scalar boolean indicating configuration validity."""

# Memory usage tracking
MemoryUsage: TypeAlias = Int[Array, ""]
"""Scalar integer representing memory usage in bytes."""

# Performance metrics
StepLatency: TypeAlias = Float[Array, ""]
"""Scalar float representing step execution latency in milliseconds."""

# =============================================================================
# Visualization and Color Types
# =============================================================================

# Visualization types
ColorHex: TypeAlias = str  # Hex color string like "#FF0000"
RGBColor: TypeAlias = tuple[int, int, int]  # RGB color tuple
