from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import (
    AgentAction,
    AgentID,
    ArcTask,
    Grid,
    GridSelection,
    Hypothesis,
    ParsedTaskData,
    TaskPair,
)


# Test Grid class
def test_grid_creation():
    """Tests the creation of a Grid object with valid data."""
    array = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    grid = Grid(array=array)

    chex.assert_shape(grid.array, (2, 3))
    chex.assert_type(grid.array, jnp.integer)
    chex.assert_trees_all_equal(grid.array, array)


def test_grid_invalid_rank():
    """Tests that Grid raises assertion error for non-2D arrays."""
    with pytest.raises(AssertionError):
        Grid(array=jnp.array([1, 2, 3], dtype=jnp.int32))  # 1D array

    with pytest.raises(AssertionError):
        Grid(array=jnp.array([[[1]]], dtype=jnp.int32))  # 3D array


def test_grid_non_integer_type():
    """Tests that Grid raises assertion error for non-integer arrays."""
    with pytest.raises(AssertionError):
        Grid(array=jnp.array([[1.0, 2.0]], dtype=jnp.float32))


# Test TaskPair class
def test_task_pair_creation():
    """Tests the creation of a TaskPair object."""
    input_grid = Grid(array=jnp.array([[0, 1]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1, 0]], dtype=jnp.int32))

    task_pair = TaskPair(input=input_grid, output=output_grid)

    assert isinstance(task_pair.input, Grid)
    assert isinstance(task_pair.output, Grid)
    chex.assert_trees_all_equal(
        task_pair.input.array, jnp.array([[0, 1]], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        task_pair.output.array, jnp.array([[1, 0]], dtype=jnp.int32)
    )


# Test ArcTask class
def test_arc_task_creation():
    """Tests the creation of an ArcTask object."""
    input_grid = Grid(array=jnp.array([[0, 1]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1, 0]], dtype=jnp.int32))
    task_pair = TaskPair(input=input_grid, output=output_grid)

    arc_task = ArcTask(
        train_pairs=[task_pair], test_pairs=[task_pair], task_id="test_task_001"
    )

    assert len(arc_task.train_pairs) == 1
    assert len(arc_task.test_pairs) == 1
    assert arc_task.task_id == "test_task_001"
    assert isinstance(arc_task.train_pairs[0], TaskPair)
    assert isinstance(arc_task.test_pairs[0], TaskPair)


def test_arc_task_optional_id():
    """Tests ArcTask creation with optional task_id."""
    input_grid = Grid(array=jnp.array([[0]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1]], dtype=jnp.int32))
    task_pair = TaskPair(input=input_grid, output=output_grid)

    arc_task = ArcTask(train_pairs=[task_pair], test_pairs=[task_pair])

    assert arc_task.task_id is None


# Test Hypothesis class
def test_hypothesis_creation():
    """Tests the creation of a Hypothesis object with valid data."""
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        hypothesis_id=jnp.array(100, dtype=jnp.int32),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        vote_count=jnp.array(3, dtype=jnp.int32),
        data=jnp.array([1, 2, 3], dtype=jnp.int32),
        description="Test hypothesis",
        is_active=jnp.array(True, dtype=jnp.bool_),
    )

    assert hypothesis.agent_id == AgentID(1)
    chex.assert_shape(hypothesis.hypothesis_id, ())
    chex.assert_shape(hypothesis.step_number, ())
    chex.assert_shape(hypothesis.confidence, ())
    chex.assert_shape(hypothesis.vote_count, ())
    chex.assert_rank(hypothesis.data, 1)
    chex.assert_shape(hypothesis.is_active, ())
    assert hypothesis.description == "Test hypothesis"


def test_hypothesis_pytree_compatibility():
    """Tests if Hypothesis JAX arrays can be used in JAX transformations."""

    def process_array_fields(data: jax.Array) -> jax.Array:
        return data * 2

    # Test that the JAX arrays within Hypothesis can be processed
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        hypothesis_id=jnp.array(100, dtype=jnp.int32),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        vote_count=jnp.array(2, dtype=jnp.int32),
        data=jnp.array([1, 2, 3], dtype=jnp.int32),
        description="Test hypothesis",
    )

    # Compile the function that works on individual JAX arrays
    jitted_process = jax.jit(process_array_fields)
    processed_data = jitted_process(hypothesis.data)  # pylint: disable=not-callable
    chex.assert_trees_all_equal(processed_data, jnp.array([2, 4, 6], dtype=jnp.int32))

    # Verify that the Hypothesis fields are valid JAX arrays
    assert isinstance(hypothesis.data, jnp.ndarray)
    assert isinstance(hypothesis.step_number, jnp.ndarray)
    assert isinstance(hypothesis.confidence, jnp.ndarray)
    assert isinstance(hypothesis.vote_count, jnp.ndarray)


def test_hypothesis_optional_fields():
    """Tests Hypothesis creation with optional fields set to None."""
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        hypothesis_id=jnp.array(50, dtype=jnp.int32),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        vote_count=jnp.array(1, dtype=jnp.int32),
        # data, description, and is_active are None
    )

    assert hypothesis.data is None
    assert hypothesis.description is None
    assert hypothesis.is_active is None

    # Test with data provided but description as None
    hypothesis_with_data = Hypothesis(
        agent_id=AgentID(2),
        hypothesis_id=jnp.array(51, dtype=jnp.int32),
        step_number=jnp.array(1, dtype=jnp.int32),
        confidence=jnp.array(0.9, dtype=jnp.float32),
        vote_count=jnp.array(0, dtype=jnp.int32),
        data=jnp.array([4, 5], dtype=jnp.int32),
    )

    assert hypothesis_with_data.description is None
    chex.assert_trees_all_equal(
        hypothesis_with_data.data, jnp.array([4, 5], dtype=jnp.int32)
    )


def test_hypothesis_field_types():
    """Tests that Hypothesis fields enforce correct types where possible."""
    # Test that invalid JAX array types/shapes cause errors during __post_init__

    # Test with wrong dtype for step_number (should be int32)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(5, dtype=jnp.float32),  # Wrong dtype
            confidence=jnp.array(0.8, dtype=jnp.float32),
            vote_count=jnp.array(1, dtype=jnp.int32),
        )

    # Test with wrong dtype for confidence (should be float32)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(5, dtype=jnp.int32),
            confidence=jnp.array(0.8, dtype=jnp.int32),  # Wrong dtype
            vote_count=jnp.array(1, dtype=jnp.int32),
        )

    # Test with wrong shape for step_number (should be scalar)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array([5, 6], dtype=jnp.int32),  # Wrong shape
            confidence=jnp.array(0.8, dtype=jnp.float32),
            vote_count=jnp.array(1, dtype=jnp.int32),
        )

    # Test with wrong shape for confidence (should be scalar)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(5, dtype=jnp.int32),
            confidence=jnp.array([0.8, 0.9], dtype=jnp.float32),  # Wrong shape
            vote_count=jnp.array(1, dtype=jnp.int32),
        )


def test_hypothesis_confidence_bounds():
    """Tests that confidence values are properly validated."""
    # Valid confidence values should work
    for conf in [0.0, 0.5, 1.0]:
        hypothesis = Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(1, dtype=jnp.int32),
            confidence=jnp.array(conf, dtype=jnp.float32),
            vote_count=jnp.array(0, dtype=jnp.int32),
        )
        assert hypothesis.confidence == conf


# Additional Hypothesis Testing Scenarios
def test_hypothesis_voting_support():
    """Tests the voting and active state features of Hypothesis."""
    hypothesis_data = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)

    hypothesis = Hypothesis(
        agent_id=AgentID(2),
        hypothesis_id=jnp.array(100, dtype=jnp.int32),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.85, dtype=jnp.float32),
        vote_count=jnp.array(3, dtype=jnp.int32),
        data=hypothesis_data,
        description="Move blue objects down",
        is_active=jnp.array(True, dtype=jnp.bool_),
    )

    chex.assert_type(hypothesis.vote_count, jnp.int32)
    assert hypothesis.vote_count == 3
    chex.assert_type(hypothesis.is_active, jnp.bool_)
    assert hypothesis.is_active


def test_hypothesis_vote_count_validation():
    """Tests validation of vote_count field."""
    # Test with wrong dtype for vote_count
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(3, dtype=jnp.int32),
            confidence=jnp.array(0.7, dtype=jnp.float32),
            vote_count=jnp.array(1, dtype=jnp.float32),  # Wrong dtype
        )

    # Test with wrong shape for vote_count
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            hypothesis_id=jnp.array(50, dtype=jnp.int32),
            step_number=jnp.array(3, dtype=jnp.int32),
            confidence=jnp.array(0.7, dtype=jnp.float32),
            vote_count=jnp.array([1, 2], dtype=jnp.int32),  # Wrong shape
        )


# Test ParsedTaskData class
def test_parsed_task_data_creation():
    """Tests the creation of a ParsedTaskData object with valid data."""
    max_train_pairs, max_test_pairs = 3, 2
    max_grid_h, max_grid_w = 5, 5

    # Create sample data
    input_grids = jnp.zeros((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    input_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)
    output_grids = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    output_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)

    test_input_grids = jnp.zeros(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_input_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )
    test_output_grids = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_output_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )

    parsed_data = ParsedTaskData(
        input_grids_examples=input_grids,
        input_masks_examples=input_masks,
        output_grids_examples=output_grids,
        output_masks_examples=output_masks,
        num_train_pairs=2,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=test_output_grids,
        true_test_output_masks=test_output_masks,
        num_test_pairs=1,
        task_id="test_task_001",
    )

    # Verify shapes and types
    chex.assert_shape(
        parsed_data.input_grids_examples, (max_train_pairs, max_grid_h, max_grid_w)
    )
    chex.assert_shape(
        parsed_data.test_input_grids, (max_test_pairs, max_grid_h, max_grid_w)
    )
    chex.assert_type(parsed_data.input_grids_examples, jnp.integer)
    chex.assert_type(parsed_data.input_masks_examples, jnp.bool_)
    assert parsed_data.num_train_pairs == 2
    assert parsed_data.num_test_pairs == 1
    assert parsed_data.task_id == "test_task_001"


def test_parsed_task_data_shape_validation():
    """Tests that ParsedTaskData validates array shapes correctly."""
    max_train_pairs, max_test_pairs = 2, 1
    max_grid_h, max_grid_w = 3, 3

    # Create valid training data
    input_grids = jnp.zeros((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    input_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)
    output_grids = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    output_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)

    # Create test data with mismatched grid dimensions
    test_input_grids = jnp.zeros(
        (max_test_pairs, max_grid_h + 1, max_grid_w), dtype=jnp.int32
    )
    test_input_masks = jnp.ones(
        (max_test_pairs, max_grid_h + 1, max_grid_w), dtype=jnp.bool_
    )
    test_output_grids = jnp.ones(
        (max_test_pairs, max_grid_h + 1, max_grid_w), dtype=jnp.int32
    )
    test_output_masks = jnp.ones(
        (max_test_pairs, max_grid_h + 1, max_grid_w), dtype=jnp.bool_
    )

    # Should raise ValueError due to mismatched grid dimensions
    with pytest.raises(
        ValueError, match="Training and test grid dimensions must match"
    ):
        ParsedTaskData(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=2,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
        )


def test_parsed_task_data_count_validation():
    """Tests that ParsedTaskData validates pair counts correctly."""
    max_train_pairs, max_test_pairs = 2, 1
    max_grid_h, max_grid_w = 3, 3

    # Create valid data
    input_grids = jnp.zeros((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    input_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)
    output_grids = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    output_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)

    test_input_grids = jnp.zeros(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_input_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )
    test_output_grids = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_output_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )

    # Test with invalid num_train_pairs (too large)
    with pytest.raises(ValueError, match="num_train_pairs .* must be between"):
        ParsedTaskData(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=max_train_pairs + 1,  # Invalid
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=1,
        )

    # Test with invalid num_test_pairs (negative)
    with pytest.raises(ValueError, match="num_test_pairs .* must be between"):
        ParsedTaskData(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_output_grids,
            true_test_output_masks=test_output_masks,
            num_test_pairs=-1,  # Invalid
        )


def test_parsed_task_data_pytree_compatibility():
    """Tests that ParsedTaskData works with JAX transformations."""
    max_train_pairs, max_test_pairs = 2, 1
    max_grid_h, max_grid_w = 3, 3

    # Create sample data
    input_grids = jnp.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        dtype=jnp.int32,
    )
    input_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)
    output_grids = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.int32)
    output_masks = jnp.ones((max_train_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_)

    test_input_grids = jnp.zeros(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_input_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )
    test_output_grids = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.int32
    )
    test_output_masks = jnp.ones(
        (max_test_pairs, max_grid_h, max_grid_w), dtype=jnp.bool_
    )

    parsed_data = ParsedTaskData(
        input_grids_examples=input_grids,
        input_masks_examples=input_masks,
        output_grids_examples=output_grids,
        output_masks_examples=output_masks,
        num_train_pairs=2,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=test_output_grids,
        true_test_output_masks=test_output_masks,
        num_test_pairs=1,
    )

    # Test that we can use JAX transformations on the arrays
    @jax.jit
    def process_input_grids(grids):
        return jnp.sum(grids)

    result = process_input_grids(parsed_data.input_grids_examples)
    expected = jnp.sum(input_grids)
    chex.assert_trees_all_equal(result, expected)


# Test AgentAction class
def test_agent_action_creation():
    """Tests the creation of an AgentAction object."""
    action_params = jnp.array([1, 2, 3, 0, 0], dtype=jnp.int32)  # Padded to max size

    action = AgentAction(
        agent_id=AgentID(1),
        action_type=jnp.array(5, dtype=jnp.int32),
        params=action_params,
        step_number=jnp.array(10, dtype=jnp.int32),
    )

    assert action.agent_id == AgentID(1)
    chex.assert_shape(action.action_type, ())
    chex.assert_shape(action.step_number, ())
    chex.assert_rank(action.params, 1)
    chex.assert_trees_all_equal(action.params, action_params)


def test_agent_action_validation():
    """Tests AgentAction validation of field types and shapes."""
    action_params = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Test with wrong dtype for action_type
    with pytest.raises(AssertionError):
        AgentAction(
            agent_id=AgentID(1),
            action_type=jnp.array(5, dtype=jnp.float32),  # Wrong dtype
            params=action_params,
            step_number=jnp.array(10, dtype=jnp.int32),
        )

    # Test with wrong shape for step_number
    with pytest.raises(AssertionError):
        AgentAction(
            agent_id=AgentID(1),
            action_type=jnp.array(5, dtype=jnp.int32),
            params=action_params,
            step_number=jnp.array([10, 11], dtype=jnp.int32),  # Wrong shape
        )


# Test GridSelection class
def test_grid_selection_creation():
    """Tests the creation of a GridSelection object."""
    mask = jnp.array([[True, False, True], [False, True, False]], dtype=jnp.bool_)
    metadata = jnp.array([1, 5, 2], dtype=jnp.int32)  # e.g., [color, count, shape_type]

    selection = GridSelection(
        mask=mask,
        selection_type=jnp.array(1, dtype=jnp.int32),  # e.g., 1 = "by_color"
        metadata=metadata,
    )

    chex.assert_shape(selection.mask, (2, 3))
    chex.assert_type(selection.mask, jnp.bool_)
    chex.assert_shape(selection.selection_type, ())
    chex.assert_rank(selection.metadata, 1)
    chex.assert_trees_all_equal(selection.mask, mask)


def test_grid_selection_validation():
    """Tests GridSelection validation of field types and shapes."""
    mask = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)

    # Test with wrong dtype for mask
    with pytest.raises(AssertionError):
        GridSelection(
            mask=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),  # Wrong dtype
            selection_type=jnp.array(1, dtype=jnp.int32),
        )

    # Test with wrong shape for selection_type
    with pytest.raises(AssertionError):
        GridSelection(
            mask=mask,
            selection_type=jnp.array([1, 2], dtype=jnp.int32),  # Wrong shape
        )


def test_grid_selection_optional_metadata():
    """Tests GridSelection with optional metadata set to None."""
    mask = jnp.array([[True, False]], dtype=jnp.bool_)

    selection = GridSelection(
        mask=mask,
        selection_type=jnp.array(2, dtype=jnp.int32),
        # metadata is None
    )

    assert selection.metadata is None
    chex.assert_trees_all_equal(selection.mask, mask)
