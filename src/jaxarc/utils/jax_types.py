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
"""Scalar integer representing an ARCLE operation (0-34)."""

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
NUM_OPERATIONS = 35  # Number of ARCLE operations (0-34)

# Type aliases for these constants
MaxGridSize: TypeAlias = int
MaxTrainPairs: TypeAlias = int
MaxTestPairs: TypeAlias = int
NumColors: TypeAlias = int
NumOperations: TypeAlias = int

# =============================================================================
# Visualization and Color Types
# =============================================================================

# Visualization types
ColorHex: TypeAlias = str  # Hex color string like "#FF0000"
RGBColor: TypeAlias = tuple[int, int, int]  # RGB color tuple
