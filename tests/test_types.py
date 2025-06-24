from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import (
    ARCLEAction,
    ArcTask,
    Grid,
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
        Grid(
            data=jnp.array([1, 2, 3], dtype=jnp.int32),
            mask=jnp.ones(3, dtype=jnp.bool_),
        )  # 1D array

    with pytest.raises(AssertionError):
        Grid(
            data=jnp.array([[[1]]], dtype=jnp.int32),
            mask=jnp.ones((1, 1, 1), dtype=jnp.bool_),
        )  # 3D array


def test_grid_non_integer_type():
    """Tests that Grid raises assertion error for non-integer arrays."""
    with pytest.raises(AssertionError):
        Grid(
            data=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )


# Test TaskPair class
def test_task_pair_creation():
    """Tests the creation of a TaskPair object."""
    input_grid = Grid(
        data=jnp.array([[0, 1]], dtype=jnp.int32),
        mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )
    output_grid = Grid(
        data=jnp.array([[1, 0]], dtype=jnp.int32),
        mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )

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
    input_grid = Grid(
        data=jnp.array([[0, 1]], dtype=jnp.int32),
        mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )
    output_grid = Grid(
        data=jnp.array([[1, 0]], dtype=jnp.int32),
        mask=jnp.ones((1, 2), dtype=jnp.bool_),
    )
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
    input_grid = Grid(
        data=jnp.array([[0]], dtype=jnp.int32), mask=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    output_grid = Grid(
        data=jnp.array([[1]], dtype=jnp.int32), mask=jnp.ones((1, 1), dtype=jnp.bool_)
    )
    task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

    arc_task = ArcTask(training_pairs=[task_pair], test_pairs=[task_pair])

    assert arc_task.task_id is None


# Test ParsedTaskData class
def test_parsed_task_data_creation():
    """Tests the creation of a ParsedTaskData object."""
    max_train_pairs, max_test_pairs = 3, 2
    grid_h, grid_w = 5, 5

    parsed_data = ParsedTaskData(
        input_grids_examples=jnp.zeros(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
        ),
        input_masks_examples=jnp.ones(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.bool_
        ),
        output_grids_examples=jnp.ones(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
        ),
        output_masks_examples=jnp.ones(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.bool_
        ),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((max_test_pairs, grid_h, grid_w), dtype=jnp.int32),
        test_input_masks=jnp.ones((max_test_pairs, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones(
            (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
        ),
        true_test_output_masks=jnp.ones(
            (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
        ),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    assert parsed_data.num_train_pairs == 2
    assert parsed_data.num_test_pairs == 1
    chex.assert_shape(
        parsed_data.input_grids_examples, (max_train_pairs, grid_h, grid_w)
    )
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


# Test ARCLEAction class
def test_arcle_action_creation():
    """Tests ARCLEAction creation and validation."""
    action = ARCLEAction(
        selection=jnp.array([[0.5, 1.0], [0.0, 0.8]], dtype=jnp.float32),
        operation=jnp.array(5, dtype=jnp.int32),
        agent_id=1,
        timestamp=100,
    )

    chex.assert_type(action.selection, jnp.float32)
    chex.assert_type(action.operation, jnp.int32)
    chex.assert_shape(action.selection, (2, 2))
    chex.assert_shape(action.operation, ())
    assert action.agent_id == 1
    assert action.timestamp == 100


def test_arcle_action_selection_bounds():
    """Tests ARCLEAction selection values are within [0, 1] bounds."""
    # Valid selection values
    action = ARCLEAction(
        selection=jnp.array([[0.0, 0.5], [1.0, 0.25]], dtype=jnp.float32),
        operation=jnp.array(10, dtype=jnp.int32),
        agent_id=2,
        timestamp=200,
    )

    assert jnp.min(action.selection) >= 0.0
    assert jnp.max(action.selection) <= 1.0


def test_arcle_action_operation_bounds():
    """Tests ARCLEAction operation ID validation."""
    # Valid operation ID (within 0-34 range)
    action = ARCLEAction(
        selection=jnp.array([[1.0]], dtype=jnp.float32),
        operation=jnp.array(34, dtype=jnp.int32),  # Maximum valid operation
        agent_id=0,
        timestamp=0,
    )

    assert 0 <= action.operation <= 34
