"""
Type definitions for the JaxARC project.

This module contains all the core data structures used throughout the project,
including grid representations, task data, agent states, and environment states.
All types are designed to be JAX-compatible with proper validation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NewType

import chex
import jax.numpy as jnp


@chex.dataclass
class Grid:
    """
    Represents a grid in the ARC challenge.

    Attributes:
        data: The grid data as a 2D array
        mask: Boolean mask indicating valid cells
    """

    data: jnp.ndarray  # Shape: (height, width), dtype: int32
    mask: jnp.ndarray  # Shape: (height, width), dtype: bool

    def __post_init__(self) -> None:
        """Validate grid structure."""
        if hasattr(self.data, "shape") and hasattr(self.mask, "shape"):
            chex.assert_rank(self.data, 2)
            chex.assert_rank(self.mask, 2)
            chex.assert_type(self.data, jnp.integer)
            chex.assert_type(self.mask, jnp.bool_)
            chex.assert_shape(self.mask, self.data.shape)


@chex.dataclass
class TaskPair:
    """
    Represents a single input-output pair in an ARC task.

    Attributes:
        input_grid: Input grid for this pair
        output_grid: Expected output grid for this pair
    """

    input_grid: Grid
    output_grid: Grid


@chex.dataclass
class ArcTask:
    """
    Represents a complete ARC task with training and test examples.

    Attributes:
        training_pairs: List of training input-output pairs
        test_pairs: List of test input-output pairs (outputs may be unknown)
        task_id: Optional identifier for the task
    """

    training_pairs: list[TaskPair]
    test_pairs: list[TaskPair]
    task_id: str | None = None


@chex.dataclass
class ParsedTaskData:
    """
    Parsed and padded ARC task data optimized for JAX processing.

    This structure contains all task data with fixed-size arrays padded to
    maximum dimensions for efficient batch processing.

    Attributes:
        # Training examples
        input_grids_examples: Training input grids (num_train_pairs, max_h, max_w)
        input_masks_examples: Masks for training inputs
        output_grids_examples: Training output grids
        output_masks_examples: Masks for training outputs
        num_train_pairs: Number of valid training pairs

        # Test examples
        test_input_grids: Test input grids (num_test_pairs, max_h, max_w)
        test_input_masks: Masks for test inputs
        true_test_output_grids: True test outputs (for evaluation)
        true_test_output_masks: Masks for true test outputs
        num_test_pairs: Number of valid test pairs

        # Task metadata
        task_index: Integer index for task identification (JAX-compatible)
    """

    # Training examples - shape: (max_train_pairs, max_grid_h, max_grid_w)
    input_grids_examples: jnp.ndarray  # int32
    input_masks_examples: jnp.ndarray  # bool
    output_grids_examples: jnp.ndarray  # int32
    output_masks_examples: jnp.ndarray  # bool
    num_train_pairs: int

    # Test examples - shape: (max_test_pairs, max_grid_h, max_grid_w)
    test_input_grids: jnp.ndarray  # int32
    test_input_masks: jnp.ndarray  # bool
    true_test_output_grids: jnp.ndarray  # int32
    true_test_output_masks: jnp.ndarray  # bool
    num_test_pairs: int

    # Task metadata
    task_index: jnp.ndarray  # int32 scalar - integer ID for JAX compatibility

    def __post_init__(self) -> None:
        """Validate parsed task data structure."""
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

            if not (0 <= self.num_train_pairs <= max_train_pairs):
                msg = f"Invalid num_train_pairs: {self.num_train_pairs} not in [0, {max_train_pairs}]"
                raise ValueError(msg)

            if not (0 <= self.num_test_pairs <= max_test_pairs):
                msg = f"Invalid num_test_pairs: {self.num_test_pairs} not in [0, {max_test_pairs}]"
                raise ValueError(msg)

            # Validate task_index
            chex.assert_type(self.task_index, jnp.int32)
            chex.assert_shape(self.task_index, ())

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass


# Type Aliases for IDs
AgentID = NewType("AgentID", int)


# Enum types for actions
class PrimitiveType(IntEnum):
    """Enumeration of primitive operations available in the environment."""

    DRAW_PIXEL = 0
    DRAW_LINE = 1
    FLOOD_FILL = 2
    COPY_PASTE_RECT = 3


class ControlType(IntEnum):
    """Enumeration of control actions available in the environment."""

    SUBMIT = 0
    RESET = 1
    NO_OP = 2


class ActionCategory(IntEnum):
    """Enumeration of action categories."""

    PRIMITIVE = 0
    CONTROL = 1


@chex.dataclass
class AgentAction:
    """
    Represents an action taken by an agent in the MARL environment.

    Attributes:
        agent_id: ID of the agent taking the action
        action_type: Type/category of action (as integer)
        params: Parameters for the action (padded to max size)
        step_number: Step when action was taken
    """

    agent_id: AgentID
    action_type: jnp.ndarray  # int32 scalar
    params: jnp.ndarray  # Shape: (max_action_params,)
    step_number: jnp.ndarray  # int32 scalar

    def __post_init__(self) -> None:
        if not hasattr(self.action_type, "shape"):
            return
        try:
            chex.assert_type(self.action_type, jnp.int32)
            chex.assert_shape(self.action_type, ())
            chex.assert_type(self.step_number, jnp.int32)
            chex.assert_shape(self.step_number, ())
            chex.assert_rank(self.params, 1)  # Should be 1D array
        except (AttributeError, TypeError):
            pass


@chex.dataclass
class Hypothesis:
    """
    Represents an agent's hypothesis about a transformation.

    Attributes:
        description: Natural language description of the hypothesis
        confidence: Confidence score [0.0, 1.0]
        program: Sequence of actions implementing the hypothesis
        program_length: Number of valid actions in the program
        agent_id: ID of the agent proposing this hypothesis
        validation_score: Score from validation on training examples
        consensus_votes: Number of votes received from other agents
    """

    description: str
    confidence: jnp.ndarray  # float32 scalar
    program: jnp.ndarray  # Shape: (max_program_length, max_action_params), int32
    program_length: jnp.ndarray  # int32 scalar
    agent_id: int
    validation_score: jnp.ndarray  # float32 scalar
    consensus_votes: jnp.ndarray  # int32 scalar

    def __post_init__(self) -> None:
        """Validate hypothesis structure."""
        if not hasattr(self.confidence, "shape"):
            return
        try:
            chex.assert_type(self.confidence, jnp.float32)
            chex.assert_type(self.program, jnp.int32)
            chex.assert_type(self.program_length, jnp.int32)
            chex.assert_type(self.validation_score, jnp.float32)
            chex.assert_type(self.consensus_votes, jnp.int32)

            chex.assert_shape(self.confidence, ())
            chex.assert_rank(self.program, 2)
            chex.assert_shape(self.program_length, ())
            chex.assert_shape(self.validation_score, ())
            chex.assert_shape(self.consensus_votes, ())
        except (AttributeError, TypeError):
            pass


@chex.dataclass
class GridSelection:
    """
    Represents a selection of cells in a grid.

    Attributes:
        mask: Boolean mask indicating selected cells
        selection_type: Type of selection (rectangle, contour, etc.)
        params: Additional parameters for the selection
    """

    mask: jnp.ndarray  # Shape: (height, width), bool
    selection_type: jnp.ndarray  # int32 scalar
    params: jnp.ndarray  # Shape: (max_params,), int32

    def __post_init__(self) -> None:
        """Validate grid selection structure."""
        if not hasattr(self.mask, "shape"):
            return
        try:
            chex.assert_type(self.mask, jnp.bool_)
            chex.assert_type(self.selection_type, jnp.int32)
            chex.assert_type(self.params, jnp.int32)
            chex.assert_rank(self.mask, 2)
            chex.assert_shape(self.selection_type, ())
            chex.assert_rank(self.params, 1)
        except (AttributeError, TypeError):
            pass


@chex.dataclass
class Action:
    """
    Unified action representation for ARC environments.

    Attributes:
        category: Action category (primitive/control)
        type_id: Specific action type within category
        selection: Grid selection for the action
        params: Additional action parameters
        agent_id: ID of the agent taking this action
        timestamp: When the action was taken
    """

    category: jnp.ndarray  # int32 scalar
    type_id: jnp.ndarray  # int32 scalar
    selection: GridSelection
    params: jnp.ndarray  # Shape: (max_params,), int32
    agent_id: int
    timestamp: int

    def __post_init__(self) -> None:
        """Validate action structure."""
        if not hasattr(self.category, "shape"):
            return
        try:
            chex.assert_type(self.category, jnp.int32)
            chex.assert_type(self.type_id, jnp.int32)
            chex.assert_type(self.params, jnp.int32)
            chex.assert_shape(self.category, ())
            chex.assert_shape(self.type_id, ())
            chex.assert_rank(self.params, 1)
        except (AttributeError, TypeError):
            pass


@chex.dataclass
class AgentInternalState:
    """
    Internal state for a single agent.

    Attributes:
        hypotheses: Current hypotheses held by the agent
        confidence_scores: Confidence in each hypothesis
        active_hypothesis_idx: Index of currently active hypothesis
        memory: Agent's working memory
    """

    hypotheses: jnp.ndarray  # Shape: (max_hypotheses, hypothesis_dim), float32
    confidence_scores: jnp.ndarray  # Shape: (max_hypotheses,), float32
    active_hypothesis_idx: jnp.ndarray  # int32 scalar
    memory: jnp.ndarray  # Shape: (memory_dim,), float32

    def __post_init__(self) -> None:
        """Validate agent internal state."""
        if not hasattr(self.hypotheses, "shape"):
            return
        try:
            chex.assert_type(self.hypotheses, jnp.float32)
            chex.assert_type(self.confidence_scores, jnp.float32)
            chex.assert_type(self.active_hypothesis_idx, jnp.int32)
            chex.assert_type(self.memory, jnp.float32)

            chex.assert_rank(self.hypotheses, 2)
            chex.assert_rank(self.confidence_scores, 1)
            chex.assert_shape(self.active_hypothesis_idx, ())
            chex.assert_rank(self.memory, 1)

            # Check consistency
            max_hypotheses = self.hypotheses.shape[0]
            chex.assert_shape(self.confidence_scores, (max_hypotheses,))
        except (AttributeError, TypeError):
            pass


# ARCLE-specific types
class ARCLEOperationType:
    """ARCLE operation types."""

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
    ROTATE_90 = 24
    ROTATE_270 = 25

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


@chex.dataclass
class ARCLEAction:
    """
    ARCLE-specific action representation.

    Attributes:
        selection: Continuous selection mask for the grid
        operation: ARCLE operation ID (0-34)
        agent_id: ID of the agent taking this action
        timestamp: When the action was taken
    """

    selection: jnp.ndarray  # Shape: (height, width), float32
    operation: jnp.ndarray  # int32 scalar
    agent_id: int
    timestamp: int

    def __post_init__(self) -> None:
        """Validate ARCLE action structure."""
        if not hasattr(self.selection, "shape"):
            return
        try:
            chex.assert_type(self.selection, jnp.float32)
            chex.assert_type(self.operation, jnp.int32)
            chex.assert_rank(self.selection, 2)
            chex.assert_shape(self.operation, ())

            # Validate selection values are in [0, 1]
            if hasattr(self.selection, "min") and hasattr(self.selection, "max"):
                min_val = float(jnp.min(self.selection))
                max_val = float(jnp.max(self.selection))
                if not (0.0 <= min_val <= max_val <= 1.0):
                    msg = f"Selection values must be in [0, 1], got [{min_val}, {max_val}]"
                    raise ValueError(msg)

            # Validate operation ID
            if hasattr(self.operation, "item"):
                op_val = int(self.operation.item())
                if not (0 <= op_val <= 34):
                    msg = f"Operation ID must be in [0, 34], got {op_val}"
                    raise ValueError(msg)

        except (AttributeError, TypeError):
            pass


@chex.dataclass
class ARCLEState:
    """
    State representation for ARCLE environment.

    This includes all base environment state fields (equivalent to ArcEnvState)
    plus ARCLE-specific additions for grid manipulation, clipboard operations,
    and episode tracking.

    All arrays use maximum dimensions from config with actual dimensions
    tracked separately for dynamic grid size support.

    Attributes:
        # Base environment state fields (from ArcEnvState)
        done: Boolean indicating if environment is done
        step: Current step number in environment
        task_data: Parsed ARC task containing training/test examples
        active_train_pair_idx: Index of current training pair being worked on
        working_grid: Grid being modified by agents
        working_grid_mask: Mask indicating valid cells in working_grid
        program: Sequence of actions taken so far
        program_length: Current length of the program
        active_agents: Mask indicating which agents are active
        cumulative_rewards: Cumulative rewards for each agent

        # ARCLE-specific fields
        grid: Current working grid being modified by agent (same as working_grid)
        input_grid: Original input grid (immutable reference)
        target_grid: Target output grid for comparison
        selected: Current selection mask (for visualization)
        clipboard: Clipboard data for copy/paste operations
        grid_dim: Actual grid dimensions [height, width]
        target_dim: Target grid dimensions [height, width]
        max_grid_dim: Maximum grid dimensions from config [height, width]
        step_count: Number of steps taken in current episode
        terminated: Whether episode has ended (submit or max steps)
        similarity_score: Current similarity to target grid [0.0, 1.0]
    """

    # Base environment state fields (compatible with ArcEnvState)
    done: chex.Array  # bool scalar
    step: int
    task_data: ParsedTaskData
    active_train_pair_idx: jnp.ndarray  # int32 scalar
    working_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    working_grid_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w), bool
    program: jnp.ndarray  # Shape: (max_program_length, max_action_params)
    program_length: jnp.ndarray  # int32 scalar
    active_agents: jnp.ndarray  # Shape: (max_agents,), bool
    cumulative_rewards: jnp.ndarray  # Shape: (max_agents,), float32

    # ARCLE-specific state fields
    grid: jnp.ndarray  # (max_grid_h, max_grid_w) int32 - current working grid
    input_grid: jnp.ndarray  # (max_grid_h, max_grid_w) int32 - original input
    target_grid: jnp.ndarray  # (max_grid_h, max_grid_w) int32 - target output
    selected: jnp.ndarray  # (max_grid_h, max_grid_w) bool - selection mask
    clipboard: jnp.ndarray  # (max_grid_h, max_grid_w) int32 - clipboard data

    # Grid metadata
    grid_dim: jnp.ndarray  # (2,) int32 - actual [h, w]
    target_dim: jnp.ndarray  # (2,) int32 - target [h, w]
    max_grid_dim: jnp.ndarray  # (2,) int32 - max [h, w] from config

    # Episode state
    step_count: jnp.ndarray  # int32 scalar
    terminated: jnp.ndarray  # bool scalar
    similarity_score: jnp.ndarray  # float32 scalar

    def __post_init__(self) -> None:
        """Validate ARCLE state structure and consistency."""
        # Skip validation during JAX transformations
        if not (hasattr(self.done, "ndim") and hasattr(self.done, "shape")):
            return

        try:
            # Validate base fields
            chex.assert_type(self.done, jnp.bool_)
            chex.assert_shape(self.done, ())
            chex.assert_type(self.active_train_pair_idx, jnp.int32)
            chex.assert_shape(self.active_train_pair_idx, ())

            # Validate base grid state
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)

            # Validate program state
            chex.assert_rank(self.program, 2)
            chex.assert_type(self.program_length, jnp.int32)
            chex.assert_shape(self.program_length, ())

            # Validate agent arrays
            chex.assert_rank(self.active_agents, 1)
            chex.assert_rank(self.cumulative_rewards, 1)
            chex.assert_type(self.active_agents, jnp.bool_)
            chex.assert_type(self.cumulative_rewards, jnp.float32)
            num_agents = self.active_agents.shape[0]
            chex.assert_shape(self.cumulative_rewards, (num_agents,))

            # Validate ARCLE-specific grid arrays
            chex.assert_rank(self.grid, 2)
            chex.assert_rank(self.input_grid, 2)
            chex.assert_rank(self.target_grid, 2)
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)

            chex.assert_type(self.grid, jnp.integer)
            chex.assert_type(self.input_grid, jnp.integer)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_type(self.selected, jnp.bool_)
            chex.assert_type(self.clipboard, jnp.integer)

            # Validate grid arrays have consistent shapes
            grid_shape = self.grid.shape
            chex.assert_shape(self.input_grid, grid_shape)
            chex.assert_shape(self.target_grid, grid_shape)
            chex.assert_shape(self.selected, grid_shape)
            chex.assert_shape(self.clipboard, grid_shape)

            # Ensure grid and working_grid are consistent
            chex.assert_shape(self.working_grid, grid_shape)

            # Validate metadata arrays
            chex.assert_rank(self.grid_dim, 1)
            chex.assert_rank(self.target_dim, 1)
            chex.assert_rank(self.max_grid_dim, 1)
            chex.assert_shape(self.grid_dim, (2,))
            chex.assert_shape(self.target_dim, (2,))
            chex.assert_shape(self.max_grid_dim, (2,))

            chex.assert_type(self.grid_dim, jnp.int32)
            chex.assert_type(self.target_dim, jnp.int32)
            chex.assert_type(self.max_grid_dim, jnp.int32)

            # Validate episode state scalars
            chex.assert_type(self.step_count, jnp.int32)
            chex.assert_shape(self.step_count, ())
            chex.assert_type(self.terminated, jnp.bool_)
            chex.assert_shape(self.terminated, ())
            chex.assert_type(self.similarity_score, jnp.float32)
            chex.assert_shape(self.similarity_score, ())

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass
