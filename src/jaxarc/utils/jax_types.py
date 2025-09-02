"""
Essential JAX type definitions for JaxARC.

This module provides the core JAX array type aliases using JAXTyping for the
JaxARC environment. Only includes types that are actually used in the codebase.

Key Features:
- Core grid and mask array types with batch support
- Action space type definitions
- Task data structure types
- Environment state types
- Essential utility types

JAXTyping *batch modifier allows the same type to work for both single arrays
(height, width) and batched arrays (batch1, batch2, ..., height, width).
"""

from __future__ import annotations

from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int

# =============================================================================
# Core Grid Types (with optional batch dimensions)
# =============================================================================

GridArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing ARC grid(s) with color values 0-9."""

MaskArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing valid/invalid cells in grid(s)."""

SelectionArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing selected cells for operations."""

# =============================================================================
# Task Data Structure Types
# =============================================================================

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

OperationId: TypeAlias = Int[Array, ""]
"""Scalar integer representing an ARC operation (0-34)."""

OperationMask: TypeAlias = Bool[Array, "35"]
"""Boolean mask indicating which operations are currently allowed."""

# =============================================================================
# Environment State Types
# =============================================================================

StepCount: TypeAlias = Int[Array, ""]
"""Scalar integer representing current step count."""

TaskIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing task identifier."""

PairIndex: TypeAlias = Int[Array, ""]
"""Scalar integer representing current demonstration/test pair."""

SimilarityScore: TypeAlias = Float[Array, "*batch"]
"""Float array representing grid similarity score(s)."""

RewardValue: TypeAlias = Float[Array, "*batch"]
"""Float array representing reward value(s)."""

ObservationArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing observation(s) from the environment."""

# =============================================================================
# Utility Types
# =============================================================================

PRNGKey: TypeAlias = Int[Array, "2"]
"""JAX PRNG key array with shape (2,)."""

ColorValue: TypeAlias = Int[Array, ""]
"""Scalar integer representing a color value (0-9)."""

GridHeight: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid height."""

GridWidth: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid width."""

# =============================================================================
# Constants
# =============================================================================

# Core ARC constants
NUM_OPERATIONS = 35  # Number of ARC operations (0-34)
NUM_COLORS = 10  # Number of colors in ARC (0-9)
MAX_GRID_SIZE = 30  # Maximum grid dimension in ARC

# Episode mode constants for JAX compatibility
EPISODE_MODE_TRAIN = 0  # Training mode
EPISODE_MODE_TEST = 1  # Test/evaluation mode
