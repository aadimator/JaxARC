from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import NewType

import chex
import jax.numpy as jnp


@chex.dataclass
class Grid:
    """Represents a 2D grid of colors.

    Attributes:
        array: A JAX ndarray representing the grid.
               Shape is (height, width). Values are integers (colors).
    """

    array: jnp.ndarray

    def __post_init__(self) -> None:
        chex.assert_rank(self.array, 2)
        # ARC colors are typically small integers (0-9).
        # We'll assert integer type, specific range checks can be done by the parser if needed.
        chex.assert_type(self.array, jnp.integer)


@chex.dataclass
class TaskPair:
    """Represents an input-output pair of grids for an ARC task.

    Attributes:
        input: The input grid.
        output: The output grid. Can be None for test inputs before solutions are known.
    """

    input: Grid
    output: Grid | None


@chex.dataclass
class ArcTask:
    """Represents a parsed ARC task.

    Attributes:
        task_id: An optional identifier for the task (e.g., filename).
        train_pairs: A sequence of training input-output grid pairs.
        test_pairs: A sequence of test input-output grid pairs.
                   For challenges, the 'output' in test_pairs is the target solution.
    """

    train_pairs: Sequence[TaskPair]
    test_pairs: Sequence[TaskPair]
    task_id: str | None = None


# --- Core Data Types for Agent Actions and Hypotheses ---

# Type Aliases for IDs
AgentID = NewType("AgentID", int)


@chex.dataclass
class ParsedTaskData:
    """
    JAX-compatible container for a preprocessed ARC task.

    This serves as the data contract between the ARC data parser and the
    environment's reset method. All arrays are pre-allocated and padded
    to maximum dimensions determined by the dataset configuration.

    Attributes:
        input_grids_examples: Padded input grids from training pairs
                            Shape: (max_train_pairs, max_grid_h, max_grid_w)
        input_masks_examples: Boolean masks for input_grids_examples
                            Shape: (max_train_pairs, max_grid_h, max_grid_w)
        output_grids_examples: Padded output grids from training pairs
                             Shape: (max_train_pairs, max_grid_h, max_grid_w)
        output_masks_examples: Boolean masks for output_grids_examples
                             Shape: (max_train_pairs, max_grid_h, max_grid_w)
        num_train_pairs: Actual number of training pairs (before padding)
        test_input_grids: Padded test input grids
                        Shape: (max_test_pairs, max_grid_h, max_grid_w)
        test_input_masks: Boolean masks for test_input_grids
                        Shape: (max_test_pairs, max_grid_h, max_grid_w)
        true_test_output_grids: Padded ground truth test output grids
                              Shape: (max_test_pairs, max_grid_h, max_grid_w)
        true_test_output_masks: Boolean masks for true_test_output_grids
                              Shape: (max_test_pairs, max_grid_h, max_grid_w)
        num_test_pairs: Actual number of test pairs (before padding)
        task_id: Optional identifier for the task
    """

    # Training data
    input_grids_examples: jnp.ndarray
    input_masks_examples: jnp.ndarray
    output_grids_examples: jnp.ndarray
    output_masks_examples: jnp.ndarray
    num_train_pairs: int

    # Test data
    test_input_grids: jnp.ndarray
    test_input_masks: jnp.ndarray
    true_test_output_grids: jnp.ndarray
    true_test_output_masks: jnp.ndarray
    num_test_pairs: int

    # Metadata
    task_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the structure and types of the ParsedTaskData."""
        # Check if we're in a JAX transformation context where array fields
        # have been transformed into non-array objects (e.g., during tree operations)
        if not (
            hasattr(self.input_grids_examples, "ndim")
            and hasattr(self.input_grids_examples, "shape")
        ):
            # Skip validation during JAX tree operations where fields are transformed
            return

        try:
            # Determine expected rank (3 for single data, 4 for batched data)
            expected_rank = self.input_grids_examples.ndim

            # Validate training data shapes and types (support both rank 3 and 4)
            chex.assert_rank(self.input_grids_examples, expected_rank)
            chex.assert_rank(self.input_masks_examples, expected_rank)
            chex.assert_rank(self.output_grids_examples, expected_rank)
            chex.assert_rank(self.output_masks_examples, expected_rank)

            # Validate test data shapes and types
            chex.assert_rank(self.test_input_grids, expected_rank)
            chex.assert_rank(self.test_input_masks, expected_rank)
            chex.assert_rank(self.true_test_output_grids, expected_rank)
            chex.assert_rank(self.true_test_output_masks, expected_rank)

            # Ensure training arrays have consistent shapes
            train_shape = self.input_grids_examples.shape
            chex.assert_shape(self.input_masks_examples, train_shape)
            chex.assert_shape(self.output_grids_examples, train_shape)
            chex.assert_shape(self.output_masks_examples, train_shape)

            # Ensure test arrays have consistent shapes
            test_shape = self.test_input_grids.shape
            chex.assert_shape(self.test_input_masks, test_shape)
            chex.assert_shape(self.true_test_output_grids, test_shape)
            chex.assert_shape(self.true_test_output_masks, test_shape)

            # Validate that grid dimensions are the same for train and test
            if train_shape[1:] != test_shape[1:]:
                error_msg = (
                    f"Training and test grid dimensions must match. "
                    f"Training: {train_shape[1:]}, Test: {test_shape[1:]}"
                )
                raise ValueError(error_msg)

            # Validate data types
            chex.assert_type(self.input_grids_examples, jnp.integer)
            chex.assert_type(self.output_grids_examples, jnp.integer)
            chex.assert_type(self.test_input_grids, jnp.integer)
            chex.assert_type(self.true_test_output_grids, jnp.integer)

            # Validate mask types (should be boolean)
            chex.assert_type(self.input_masks_examples, jnp.bool_)
            chex.assert_type(self.output_masks_examples, jnp.bool_)
            chex.assert_type(self.test_input_masks, jnp.bool_)
            chex.assert_type(self.true_test_output_masks, jnp.bool_)

            # Validate counts are non-negative and within bounds
            max_train_pairs = train_shape[0]
            max_test_pairs = test_shape[0]

            if not (0 <= self.num_train_pairs <= max_train_pairs):
                error_msg = (
                    f"num_train_pairs ({self.num_train_pairs}) must be between "
                    f"0 and {max_train_pairs}"
                )
                raise ValueError(error_msg)

            if not (0 <= self.num_test_pairs <= max_test_pairs):
                error_msg = (
                    f"num_test_pairs ({self.num_test_pairs}) must be between "
                    f"0 and {max_test_pairs}"
                )
                raise ValueError(error_msg)

        except (AttributeError, TypeError):
            # Skip validation only for specific JAX transformation errors
            # This preserves normal validation while allowing JAX operations
            pass

        # Note: Additional validation (shape compatibility, bounds checking) should be done
        # at data creation time outside of JAX-transformed functions, as Python control flow
        # with JAX tracers is not supported in JIT-compiled contexts.


# --- Additional Types for MARL Environment ---


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
        chex.assert_type(self.action_type, jnp.int32)
        chex.assert_shape(self.action_type, ())
        chex.assert_type(self.step_number, jnp.int32)
        chex.assert_shape(self.step_number, ())
        chex.assert_rank(self.params, 1)  # Should be 1D array


@chex.dataclass
class Hypothesis:
    """
    Represents a hypothesis generated by an agent in the MARL environment.

    Attributes:
        agent_id: ID of the agent generating the hypothesis
        hypothesis_id: Unique identifier for this hypothesis
        step_number: Step when hypothesis was created
        confidence: Confidence score (0.0 to 1.0)
        vote_count: Number of votes this hypothesis has received
        data: Generic data payload for hypothesis data (Shape: max_proposal_data_dim)
        description: Optional textual description of the hypothesis
        is_active: Whether this hypothesis is still valid/not superseded
    """

    agent_id: AgentID
    hypothesis_id: jnp.ndarray  # int32 scalar
    step_number: jnp.ndarray  # int32 scalar
    confidence: jnp.ndarray  # float32 scalar
    vote_count: jnp.ndarray  # int32 scalar
    data: jnp.ndarray | None = None  # Shape: (max_proposal_data_dim,)
    description: str | None = None
    is_active: jnp.ndarray | None = None  # bool scalar

    def __post_init__(self) -> None:
        chex.assert_type(self.hypothesis_id, jnp.int32)
        chex.assert_shape(self.hypothesis_id, ())
        chex.assert_type(self.step_number, jnp.int32)
        chex.assert_shape(self.step_number, ())
        chex.assert_type(self.confidence, jnp.float32)
        chex.assert_shape(self.confidence, ())
        chex.assert_type(self.vote_count, jnp.int32)
        chex.assert_shape(self.vote_count, ())

        if self.data is not None:
            chex.assert_rank(self.data, 1)  # Should be 1D array

        if self.is_active is not None:
            chex.assert_type(self.is_active, jnp.bool_)
            chex.assert_shape(self.is_active, ())


# --- Utility Types ---


@chex.dataclass
class GridSelection:
    """
    Represents a selection of pixels/regions in a grid.

    Attributes:
        mask: Boolean mask indicating selected pixels
        selection_type: Type of selection (e.g., by color, shape, etc.)
        metadata: Optional metadata about the selection
    """

    mask: jnp.ndarray  # Shape: (grid_h, grid_w), dtype: bool
    selection_type: jnp.ndarray  # int32 scalar
    metadata: jnp.ndarray | None = None  # Optional metadata array

    def __post_init__(self) -> None:
        chex.assert_rank(self.mask, 2)
        chex.assert_type(self.mask, jnp.bool_)
        chex.assert_type(self.selection_type, jnp.int32)
        chex.assert_shape(self.selection_type, ())

        if self.metadata is not None:
            chex.assert_rank(self.metadata, 1)  # Should be 1D array


# --- Primitive Environment Action System ---


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
class Action:
    """
    Represents an action in the primitive environment.

    Attributes:
        category: Category of action (PRIMITIVE or CONTROL)
        primitive_type: Type of primitive operation (if category is PRIMITIVE)
        control_type: Type of control action (if category is CONTROL)
        params: Parameters for the action (padded to max size)
        agent_id: ID of the agent taking the action
    """

    category: jnp.ndarray  # int32 scalar
    primitive_type: jnp.ndarray  # int32 scalar
    control_type: jnp.ndarray  # int32 scalar
    params: jnp.ndarray  # Shape: (max_action_params,)
    agent_id: AgentID

    def __post_init__(self) -> None:
        """Validate action structure and types."""
        chex.assert_type(self.category, jnp.int32)
        chex.assert_shape(self.category, ())
        chex.assert_type(self.primitive_type, jnp.int32)
        chex.assert_shape(self.primitive_type, ())
        chex.assert_type(self.control_type, jnp.int32)
        chex.assert_shape(self.control_type, ())
        chex.assert_rank(self.params, 1)  # Should be 1D array


@chex.dataclass
class AgentInternalState:
    """
    Internal state for a single agent in the primitive environment.

    Attributes:
        agent_id: ID of the agent
        active: Whether the agent is currently active
        program_buffer: Buffer storing the agent's current program
        last_action: Last action taken by the agent
    """

    agent_id: AgentID
    active: jnp.ndarray  # bool scalar
    program_buffer: jnp.ndarray  # Shape: (max_program_length, max_action_params)
    last_action: Action | None = None

    def __post_init__(self) -> None:
        """Validate agent internal state structure."""
        chex.assert_type(self.active, jnp.bool_)
        chex.assert_shape(self.active, ())
        chex.assert_rank(self.program_buffer, 2)  # Should be 2D array


# --- Configuration Management ---
# Configuration is handled via Hydra - see conf/environment/primitive_env.yaml
