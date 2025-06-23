from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import (
    AgentAction,
    ArcTask,
    Grid,
    GridSelection,
    ParsedTaskData,
    TaskPair,
)


# Test Grid class
def test_grid_creation():
    """Tests the creation of a Grid object with valid data."""
    data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    mask = jnp.ones((2, 3), dtype=jnp.bool_)
    grid = Grid(data=data, mask=mask)

    chex.assert_shape(grid.data, (2, 3))
    chex.assert_type(grid.data, jnp.integer)
    chex.assert_trees_all_equal(grid.data, data)
    chex.assert_trees_all_equal(grid.mask, mask)


def test_grid_invalid_rank():
    """Tests that Grid raises assertion error for non-2D arrays."""
    with pytest.raises(AssertionError):
        Grid(data=jnp.array([1, 2, 3], dtype=jnp.int32),
             mask=jnp.ones(3, dtype=jnp.bool_))  # 1D array

    with pytest.raises(AssertionError):
        Grid(data=jnp.array([[[1]]], dtype=jnp.int32),
             mask=jnp.ones((1, 1, 1), dtype=jnp.bool_))  # 3D array


def test_grid_non_integer_type():
    """Tests that Grid raises assertion error for non-integer arrays."""
    with pytest.raises(AssertionError):
        Grid(data=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
             mask=jnp.ones((1, 2), dtype=jnp.bool_))


# Test TaskPair class
def test_task_pair_creation():
    """Tests the creation of a TaskPair object."""
    input_grid = Grid(data=jnp.array([[0, 1]], dtype=jnp.int32),
                      mask=jnp.ones((1, 2), dtype=jnp.bool_))
    output_grid = Grid(data=jnp.array([[1, 0]], dtype=jnp.int32),
                       mask=jnp.ones((1, 2), dtype=jnp.bool_))

    task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

    assert isinstance(task_pair.input_grid, Grid)
    assert isinstance(task_pair.output_grid, Grid)
    chex.assert_trees_all_equal(task_pair.input_grid.data, input_grid.data)
    chex.assert_trees_all_equal(task_pair.output_grid.data, output_grid.data)

    # Test that they are separate objects
    assert task_pair.input_grid is not task_pair.output_grid


# Test ArcTask class
def test_arc_task_creation():
    """Tests the creation of an ArcTask object."""
    input_grid = Grid(data=jnp.array([[0, 1]], dtype=jnp.int32),
                      mask=jnp.ones((1, 2), dtype=jnp.bool_))
    output_grid = Grid(data=jnp.array([[1, 0]], dtype=jnp.int32),
                       mask=jnp.ones((1, 2), dtype=jnp.bool_))
    task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

    arc_task = ArcTask(
        training_pairs=[task_pair], test_pairs=[task_pair], task_id="test_task_001"
    )

    assert len(arc_task.training_pairs) == 1
    assert len(arc_task.test_pairs) == 1
    assert arc_task.task_id == "test_task_001"
    assert isinstance(arc_task.training_pairs[0], TaskPair)
    assert isinstance(arc_task.test_pairs[0], TaskPair)


def test_arc_task_optional_id():
    """Tests ArcTask creation with optional task_id."""
    input_grid = Grid(data=jnp.array([[0]], dtype=jnp.int32),
                      mask=jnp.ones((1, 1), dtype=jnp.bool_))
    output_grid = Grid(data=jnp.array([[1]], dtype=jnp.int32),
                       mask=jnp.ones((1, 1), dtype=jnp.bool_))
    task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

    arc_task = ArcTask(training_pairs=[task_pair], test_pairs=[task_pair])

    assert arc_task.task_id is None


# Test ParsedTaskData class
def test_parsed_task_data_creation():
    """Tests the creation of a ParsedTaskData object."""
    max_train_pairs, max_test_pairs = 3, 2
    grid_h, grid_w = 5, 5

    parsed_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((max_train_pairs, grid_h, grid_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((max_train_pairs, grid_h, grid_w), dtype=jnp.bool_),
        output_grids_examples=jnp.ones((max_train_pairs, grid_h, grid_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((max_train_pairs, grid_h, grid_w), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((max_test_pairs, grid_h, grid_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((max_test_pairs, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((max_test_pairs, grid_h, grid_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((max_test_pairs, grid_h, grid_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    assert parsed_data.num_train_pairs == 2
    assert parsed_data.num_test_pairs == 1
    chex.assert_shape(parsed_data.input_grids_examples, (max_train_pairs, grid_h, grid_w))
    chex.assert_shape(parsed_data.test_input_grids, (max_test_pairs, grid_h, grid_w))
    chex.assert_type(parsed_data.task_index, jnp.int32)


def test_parsed_task_data_shape_validation():
    """Tests ParsedTaskData shape validation during __post_init__."""
    grid_h, grid_w = 5, 5

    # This should work - matching shapes
    parsed_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        output_grids_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((1, grid_h, grid_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((1, grid_h, grid_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    assert parsed_data.num_train_pairs == 2


def test_parsed_task_data_count_validation():
    """Tests validation of pair counts in ParsedTaskData."""
    grid_h, grid_w = 3, 3

    parsed_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((3, grid_h, grid_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
        output_grids_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
        num_train_pairs=2,  # Should be <= 3
        test_input_grids=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        num_test_pairs=1,  # Should be <= 2
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    assert parsed_data.num_train_pairs == 2
    assert parsed_data.num_test_pairs == 1


def test_parsed_task_data_pytree_compatibility():
    """Tests that ParsedTaskData works with JAX transformations."""

    def transform_grids(parsed_data: ParsedTaskData) -> ParsedTaskData:
        return parsed_data.replace(
            input_grids_examples=parsed_data.input_grids_examples + 1
        )

    grid_h, grid_w = 3, 3
    parsed_data = ParsedTaskData(
        input_grids_examples=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
        input_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        output_grids_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
        output_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((1, grid_h, grid_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((1, grid_h, grid_w), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    # Test JAX transformation
    transformed = jax.jit(transform_grids)(parsed_data)
    assert jnp.all(transformed.input_grids_examples == 1)


# Test AgentAction class
def test_agent_action_creation():
    """Tests AgentAction creation and validation."""
    action = AgentAction(
        agent_id=1,
        action_type=jnp.array(2, dtype=jnp.int32),
        params=jnp.array([10, 20, 0, 0, 0], dtype=jnp.int32),
        step_number=jnp.array(0, dtype=jnp.int32),
    )

    chex.assert_type(action.action_type, jnp.int32)
    chex.assert_type(action.params, jnp.int32)
    chex.assert_type(action.step_number, jnp.int32)
    chex.assert_shape(action.action_type, ())
    chex.assert_shape(action.step_number, ())
    chex.assert_rank(action.params, 1)


def test_agent_action_validation():
    """Tests AgentAction validation logic."""
    # Valid action should pass validation
    action = AgentAction(
        agent_id=1,
        action_type=jnp.array(1, dtype=jnp.int32),
        params=jnp.array([5, 5, 1, 0, 0], dtype=jnp.int32),
        step_number=jnp.array(0, dtype=jnp.int32),
    )

    assert action.agent_id == 1
    assert action.action_type == 1


# Test GridSelection class
def test_grid_selection_creation():
    """Tests GridSelection creation and validation."""
    selection = GridSelection(
        mask=jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
        selection_type=jnp.array(1, dtype=jnp.int32),
        params=jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32),
    )

    chex.assert_shape(selection.mask, (2, 2))
    chex.assert_shape(selection.params, (5,))
    chex.assert_type(selection.mask, jnp.bool_)
    chex.assert_type(selection.params, jnp.int32)
    chex.assert_type(selection.selection_type, jnp.int32)


def test_grid_selection_validation():
    """Tests GridSelection validation logic."""
    selection = GridSelection(
        mask=jnp.array([[True, False, False], [False, True, False]], dtype=jnp.bool_),
        selection_type=jnp.array(0, dtype=jnp.int32),
        params=jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32),
    )

    assert selection.selection_type == 0


def test_grid_selection_optional_metadata():
    """Tests GridSelection with optional metadata fields."""
    selection = GridSelection(
        mask=jnp.array([[True]], dtype=jnp.bool_),
        selection_type=jnp.array(2, dtype=jnp.int32),
        params=jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32),
    )

    chex.assert_shape(selection.mask, (1, 1))
    assert selection.selection_type == 2
