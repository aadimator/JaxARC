"""
Type definitions for the JaxARC project.

This module contains all the core data structures used throughout the project,
including grid representations, task data, agent states, and environment states.
All types are designed to be JAX-compatible with proper validation and JAXTyping annotations.

This module also provides the core JAX array type aliases using JAXTyping for the
JaxARC environment.

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

from typing import Any, TypeAlias

import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from stoa.env_types import StepType, TimeStep

# Import configuration types for EnvParams
from jaxarc.configs.action_config import ActionConfig
from jaxarc.configs.dataset_config import DatasetConfig
from jaxarc.configs.grid_initialization_config import GridInitializationConfig
from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.configs.reward_config import RewardConfig

StepType = StepType  # Re-export for convenience
TimeStep = TimeStep  # Re-export for convenience

# =============================================================================
# JAX Type Definitions (formerly in jax_types.py)
# =============================================================================

# Core Grid Types (with optional batch dimensions)
GridArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing ARC grid(s) with color values 0-9."""

MaskArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing valid/invalid cells in grid(s)."""

SelectionArray: TypeAlias = Bool[Array, "*batch height width"]
"""Boolean array representing selected cells for operations."""

# Task Data Structure Types
TaskInputGrids: TypeAlias = Int[Array, "max_pairs max_height max_width"]
"""Training/test input grids padded to maximum dimensions."""

TaskOutputGrids: TypeAlias = Int[Array, "max_pairs max_height max_width"]
"""Training/test output grids padded to maximum dimensions."""

TaskInputMasks: TypeAlias = Bool[Array, "max_pairs max_height max_width"]
"""Training/test input masks padded to maximum dimensions."""

TaskOutputMasks: TypeAlias = Bool[Array, "max_pairs max_height max_width"]
"""Training/test output masks padded to maximum dimensions."""

# Action Types
OperationId: TypeAlias = Int[Array, ""]
"""Scalar integer representing an ARC operation (0-34)."""

OperationMask: TypeAlias = Bool[Array, "35"]
"""Boolean mask indicating which operations are currently allowed."""

# Environment State Types
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

DiscountValue: TypeAlias = Float[Array, "*batch"]
"""Float array representing discount value(s)."""

ObservationArray: TypeAlias = Int[Array, "*batch height width"]
"""Integer array representing observation(s) from the environment."""

# Utility Types
PRNGKey: TypeAlias = Int[Array, "2"]
"""JAX PRNG key array with shape (2,)."""

ColorValue: TypeAlias = Int[Array, ""]
"""Scalar integer representing a color value (0-9)."""

GridHeight: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid height."""

GridWidth: TypeAlias = Int[Array, ""]
"""Scalar integer representing grid width."""

# =============================================================================
# Constants (formerly in jax_types.py)
# =============================================================================

# Core ARC constants
NUM_OPERATIONS = 35  # Number of ARC operations (0-34)
NUM_COLORS = 10  # Number of colors in ARC (0-9)
MAX_GRID_SIZE = 30  # Maximum grid dimension in ARC

# Episode mode constants for JAX compatibility
EPISODE_MODE_TRAIN = 0  # Training mode
EPISODE_MODE_TEST = 1  # Test/evaluation mode


class EnvParams(eqx.Module):
    """
    Clean environment parameters - only what's needed for environment logic.

    This is NOT a rename of JaxArcConfig. JaxArcConfig contains framework concerns
    (logging, visualization, storage) that don't belong in environment parameters.

    EnvParams now carries a JAX-native task buffer for JIT/vmap-compatible reset().
    The buffer is a stacked pytree of JAX arrays (batched JaxArcTask fields) and
    optional subset indices define a view into the buffer.
    """

    # Core configurations (references, not duplicated fields)
    dataset: DatasetConfig
    action: ActionConfig
    reward: RewardConfig
    grid_initialization: GridInitializationConfig

    # Episode-specific settings
    max_episode_steps: int

    # JAX-native task buffer (batched pytree of arrays) and optional indices view
    buffer: Any = None
    subset_indices: Any = None

    # Episode control
    episode_mode: int = 0  # 0=train, 1=test

    def __check_init__(self) -> None:
        # Basic validations
        assert isinstance(self.max_episode_steps, int)
        assert self.max_episode_steps > 0
        assert self.episode_mode in (0, 1)

        # Require a task buffer for JIT-compatible reset
        assert self.buffer is not None, (
            "EnvParams.buffer must be provided for JIT-compatible reset"
        )

    @classmethod
    def from_config(
        cls,
        config: JaxArcConfig,
        *,
        episode_mode: int = 0,
        buffer: Any = None,
        subset_indices: Any = None,
    ) -> EnvParams:
        """
        Extract environment parameters from the unified JaxArcConfig.

        Args:
            config: Full project configuration
            episode_mode: 0=train, 1=test
            buffer: Batched pytree of JAX arrays (stacked JaxArcTask fields)
            subset_indices: Optional indices defining a subview into the buffer
        """
        return cls(
            dataset=config.dataset,
            action=config.action,
            reward=config.reward,
            grid_initialization=config.grid_initialization,
            max_episode_steps=int(config.environment.max_episode_steps),
            buffer=buffer,
            subset_indices=subset_indices,
            episode_mode=int(episode_mode),
        )


class Grid(eqx.Module):
    """
    Represents a grid in the ARC challenge using Equinox Module.

    Equinox provides better JAX integration with automatic PyTree registration
    and improved compatibility with JAX transformations.

    Attributes:
        data: The grid data as a 2D integer array with JAXTyping shape annotation
        mask: Boolean mask indicating valid cells with JAXTyping shape annotation
    """

    data: GridArray  # JAXTyping: Int[Array, "height width"]
    mask: MaskArray  # JAXTyping: Bool[Array, "height width"]

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the valid region in the grid.

        Uses the mask to determine the actual meaningful grid dimensions,
        not the padded dimensions.

        Returns:
            Tuple of (height, width) representing the valid region dimensions
        """
        from .utils.grid_utils import get_actual_grid_shape_from_mask

        height, width = get_actual_grid_shape_from_mask(self.mask)
        return (int(height), int(width))

    def __check_init__(self) -> None:
        """Equinox validation method for grid structure."""
        if hasattr(self.data, "shape") and hasattr(self.mask, "shape"):
            # JAXTyping provides compile-time shape validation, but we keep runtime checks
            # for compatibility and additional safety during development
            chex.assert_rank(self.data, 2)
            chex.assert_rank(self.mask, 2)
            chex.assert_type(self.data, jnp.integer)
            chex.assert_type(self.mask, jnp.bool_)
            chex.assert_shape(self.mask, self.data.shape)

            # Additional JAXTyping-aware validation
            # Ensure grid values are in valid ARC color range (0-9)
            # Also, -1 for background masking as well
            if hasattr(self.data, "min") and hasattr(self.data, "max"):
                min_val = int(jnp.min(self.data))
                max_val = int(jnp.max(self.data))
                if not -1 <= min_val <= max_val <= 9:
                    msg = f"Grid color values must be in [-1, 9], got [{min_val}, {max_val}]"
                    raise ValueError(msg)


class TaskPair(eqx.Module):
    """
    Represents a single input-output pair in an ARC task using Equinox Module.

    Attributes:
        input_grid: Input grid for this pair
        output_grid: Expected output grid for this pair
    """

    input_grid: Grid
    output_grid: Grid


class JaxArcTask(eqx.Module):
    """
    JAX-compatible ARC task data with fixed-size arrays for efficient processing using Equinox Module.

    This structure contains all task data with fixed-size arrays padded to
    maximum dimensions for efficient batch processing and JAX transformations.
    All arrays now use JAXTyping annotations for better type safety and documentation.

    Attributes:
        # Training examples with JAXTyping annotations
        input_grids_examples: Training input grids with precise shape annotation
        input_masks_examples: Masks for training inputs with precise shape annotation
        output_grids_examples: Training output grids with precise shape annotation
        output_masks_examples: Masks for training outputs with precise shape annotation
        num_train_pairs: Number of valid training pairs

        # Test examples with JAXTyping annotations
        test_input_grids: Test input grids with precise shape annotation
        test_input_masks: Masks for test inputs with precise shape annotation
        true_test_output_grids: True test outputs with precise shape annotation
        true_test_output_masks: Masks for true test outputs with precise shape annotation
        num_test_pairs: Number of valid test pairs

        # Task metadata with JAXTyping annotation
        task_index: Integer index for task identification (JAX-compatible scalar)
    """

    # Training examples - JAXTyping: Int[Array, "max_pairs max_height max_width"]
    input_grids_examples: TaskInputGrids
    input_masks_examples: TaskInputMasks
    output_grids_examples: TaskOutputGrids
    output_masks_examples: TaskOutputMasks
    num_train_pairs: int

    # Test examples - JAXTyping: Int[Array, "max_pairs max_height max_width"]
    test_input_grids: TaskInputGrids
    test_input_masks: TaskInputMasks
    true_test_output_grids: TaskOutputGrids
    true_test_output_masks: TaskOutputMasks
    num_test_pairs: int

    # Task metadata - JAXTyping: Int[Array, ""]
    task_index: TaskIndex

    def __check_init__(self) -> None:
        """Equinox validation method for parsed task data structure."""
        # Skip validation during JAX transformations
        if not hasattr(self.input_grids_examples, "shape"):
            return

        try:
            # Validate training data shapes and types
            chex.assert_rank(self.input_grids_examples, 3)
            chex.assert_rank(self.input_masks_examples, 3)
            chex.assert_rank(self.output_grids_examples, 3)
            chex.assert_rank(self.output_masks_examples, 3)

            chex.assert_type(self.input_grids_examples, jnp.int32)
            chex.assert_type(self.input_masks_examples, jnp.bool_)
            chex.assert_type(self.output_grids_examples, jnp.int32)
            chex.assert_type(self.output_masks_examples, jnp.bool_)

            # Check consistent shapes across training examples
            train_shape = self.input_grids_examples.shape
            chex.assert_shape(self.input_masks_examples, train_shape)
            chex.assert_shape(self.output_grids_examples, train_shape)
            chex.assert_shape(self.output_masks_examples, train_shape)

            # Validate test data shapes and types
            chex.assert_rank(self.test_input_grids, 3)
            chex.assert_rank(self.test_input_masks, 3)
            chex.assert_rank(self.true_test_output_grids, 3)
            chex.assert_rank(self.true_test_output_masks, 3)

            chex.assert_type(self.test_input_grids, jnp.int32)
            chex.assert_type(self.test_input_masks, jnp.bool_)
            chex.assert_type(self.true_test_output_grids, jnp.int32)
            chex.assert_type(self.true_test_output_masks, jnp.bool_)

            # Check consistent shapes across test examples
            test_shape = self.test_input_grids.shape
            chex.assert_shape(self.test_input_masks, test_shape)
            chex.assert_shape(self.true_test_output_grids, test_shape)
            chex.assert_shape(self.true_test_output_masks, test_shape)

            # Validate that grid dimensions match between train and test
            if train_shape[1:] != test_shape[1:]:
                msg = f"Grid dimensions mismatch: train {train_shape[1:]} vs test {test_shape[1:]}"
                raise ValueError(msg)

            # Validate counts
            max_train_pairs = train_shape[0]
            max_test_pairs = test_shape[0]

            if not 0 <= self.num_train_pairs <= max_train_pairs:
                msg = f"Invalid num_train_pairs: {self.num_train_pairs} not in [0, {max_train_pairs}]"
                raise ValueError(msg)

            if not 0 <= self.num_test_pairs <= max_test_pairs:
                msg = f"Invalid num_test_pairs: {self.num_test_pairs} not in [0, {max_test_pairs}]"
                raise ValueError(msg)

            # Validate task_index
            chex.assert_type(self.task_index, jnp.int32)
            chex.assert_shape(self.task_index, ())

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass

    def get_train_input_grid(self, pair_idx: int) -> Grid:
        """Get training input grid at given index."""
        return Grid(
            data=self.input_grids_examples[pair_idx],
            mask=self.input_masks_examples[pair_idx],
        )

    def get_train_output_grid(self, pair_idx: int) -> Grid:
        """Get training output grid at given index."""
        return Grid(
            data=self.output_grids_examples[pair_idx],
            mask=self.output_masks_examples[pair_idx],
        )

    def get_test_input_grid(self, pair_idx: int) -> Grid:
        """Get test input grid at given index."""
        return Grid(
            data=self.test_input_grids[pair_idx], mask=self.test_input_masks[pair_idx]
        )

    def get_test_output_grid(self, pair_idx: int) -> Grid:
        """Get test output grid at given index."""
        return Grid(
            data=self.true_test_output_grids[pair_idx],
            mask=self.true_test_output_masks[pair_idx],
        )

    def get_train_pair(self, pair_idx: int) -> TaskPair:
        """Get training pair at given index."""
        return TaskPair(
            input_grid=self.get_train_input_grid(pair_idx),
            output_grid=self.get_train_output_grid(pair_idx),
        )

    def get_test_pair(self, pair_idx: int) -> TaskPair:
        """Get test pair at given index."""
        return TaskPair(
            input_grid=self.get_test_input_grid(pair_idx),
            output_grid=self.get_test_output_grid(pair_idx),
        )

    # =========================================================================
    # Enhanced Utility Methods for State Management
    # =========================================================================

    def get_available_demo_pairs(self) -> Bool[Array, ...]:
        """Get mask of available training pairs.

        Returns:
            JAX boolean array indicating which training pairs are available
            (based on num_train_pairs)
        """
        return jnp.arange(self.input_grids_examples.shape[0]) < self.num_train_pairs

    def get_available_test_pairs(self) -> Bool[Array, ...]:
        """Get mask of available test pairs.

        Returns:
            JAX boolean array indicating which test pairs are available
            (based on num_test_pairs)
        """
        return jnp.arange(self.test_input_grids.shape[0]) < self.num_test_pairs

    def get_demo_pair_data(
        self, pair_idx: int
    ) -> tuple[GridArray, GridArray, MaskArray, MaskArray]:
        """Get training pair data by index.

        Args:
            pair_idx: Index of the training pair to retrieve

        Returns:
            Tuple of (input_grid, output_grid, input_mask, output_mask)
        """
        return (
            self.input_grids_examples[pair_idx],
            self.output_grids_examples[pair_idx],
            self.input_masks_examples[pair_idx],
            self.output_masks_examples[pair_idx],
        )

    def get_test_pair_data(self, pair_idx: int) -> tuple[GridArray, MaskArray]:
        """Get test pair input data by index (no target during evaluation).

        Args:
            pair_idx: Index of the test pair to retrieve

        Returns:
            Tuple of (input_grid, input_mask)
        """
        return (self.test_input_grids[pair_idx], self.test_input_masks[pair_idx])

    def is_demo_pair_available(self, pair_idx: int) -> jnp.ndarray:
        """Check if a specific demonstration pair is available.

        Args:
            pair_idx: Index of the demonstration pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is available
        """
        return jnp.array((pair_idx >= 0) & (pair_idx < self.num_train_pairs))

    def is_test_pair_available(self, pair_idx: int) -> jnp.ndarray:
        """Check if a specific test pair is available.

        Args:
            pair_idx: Index of the test pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is available
        """
        return jnp.array((pair_idx >= 0) & (pair_idx < self.num_test_pairs))

    def get_max_train_pairs(self) -> int:
        """Get the maximum number of training pairs this task can hold.

        Returns:
            Maximum number of training pairs (array dimension)
        """
        return self.input_grids_examples.shape[0]

    def get_max_test_pairs(self) -> int:
        """Get the maximum number of test pairs this task can hold.

        Returns:
            Maximum number of test pairs (array dimension)
        """
        return self.test_input_grids.shape[0]

    def get_grid_shape(self) -> tuple[int, int]:
        """Get the grid dimensions for this task.

        Returns:
            Tuple of (height, width) for the grid dimensions
        """
        return (self.input_grids_examples.shape[1], self.input_grids_examples.shape[2])

    def get_task_summary(self) -> dict:
        """Get a summary of task information.

        Returns:
            Dictionary containing task metadata
        """
        return {
            "task_index": int(self.task_index),
            "num_train_pairs": self.num_train_pairs,
            "num_test_pairs": self.num_test_pairs,
            "max_train_pairs": self.get_max_train_pairs(),
            "max_test_pairs": self.get_max_test_pairs(),
            "grid_shape": self.get_grid_shape(),
        }

    def get_task_id(self) -> str | None:
        """Get the task ID for this task.

        This is a convenience method that looks up the task ID from the global
        task manager using the stored task_index.

        Note: This method is NOT JAX-compatible and should not be used
        within JAX transformations (jit, vmap, etc.). Use only for
        debugging, logging, visualization, or other non-JAX code.

        Returns:
            String task ID if found in the global task manager, None otherwise

        Example:
            ```python
            task = parser.get_task_by_id("some_task")
            task_id = task.get_task_id()  # Returns "some_task"
            ```
        """
        from jaxarc.utils import get_task_id_globally

        return get_task_id_globally(int(self.task_index))


# ARC-specific types
class ARCOperationType:
    """ARC operation types (grid + submit only).

    Pair control operations (35-41) have been removed to simplify the action space.
    Remaining valid operation IDs: 0-34.
    """

    # Fill operations (0-9)
    FILL_0 = 0
    FILL_1 = 1
    FILL_2 = 2
    FILL_3 = 3
    FILL_4 = 4
    FILL_5 = 5
    FILL_6 = 6
    FILL_7 = 7
    FILL_8 = 8
    FILL_9 = 9

    # Flood fill operations (10-19)
    FLOOD_FILL_0 = 10
    FLOOD_FILL_1 = 11
    FLOOD_FILL_2 = 12
    FLOOD_FILL_3 = 13
    FLOOD_FILL_4 = 14
    FLOOD_FILL_5 = 15
    FLOOD_FILL_6 = 16
    FLOOD_FILL_7 = 17
    FLOOD_FILL_8 = 18
    FLOOD_FILL_9 = 19

    # Move operations (20-23)
    MOVE_UP = 20
    MOVE_DOWN = 21
    MOVE_LEFT = 22
    MOVE_RIGHT = 23

    # Rotate operations (24-25)
    ROTATE_C = 24  # Clockwise
    ROTATE_CC = 25  # Counter-clockwise

    # Flip operations (26-27)
    FLIP_HORIZONTAL = 26
    FLIP_VERTICAL = 27

    # Clipboard operations (28-30)
    COPY = 28
    PASTE = 29
    CUT = 30

    # Grid operations (31-33)
    CLEAR = 31
    COPY_INPUT = 32
    RESIZE = 33

    # Submit operation (34)
    SUBMIT = 34
