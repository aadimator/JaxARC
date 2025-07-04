from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import (
    ARCLEAction,
    Grid,
    JaxArcTask,
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


# Test JaxArcTask class
def test_jax_arc_task_creation():
    """Tests the creation of a JaxArcTask object."""
    max_train_pairs, max_test_pairs = 3, 2
    grid_h, grid_w = 5, 5

    parsed_data = JaxArcTask(
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


def test_jax_arc_task_shape_validation():
    """Tests JaxArcTask shape validation during __post_init__."""
    grid_h, grid_w = 5, 5

    # This should work - matching shapes
    parsed_data = JaxArcTask(
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


def test_jax_arc_task_count_validation():
    """Tests validation of pair counts in JaxArcTask."""
    grid_h, grid_w = 3, 3

    parsed_data = JaxArcTask(
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


def test_jax_arc_task_pytree_compatibility():
    """Tests that JaxArcTask works with JAX transformations."""

    def transform_grids(parsed_data: JaxArcTask) -> JaxArcTask:
        return parsed_data.replace(
            input_grids_examples=parsed_data.input_grids_examples + 1
        )

    grid_h, grid_w = 3, 3
    parsed_data = JaxArcTask(
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


def test_jax_arc_task_utility_methods():
    """Tests the utility methods of JaxArcTask."""
    grid_h, grid_w = 3, 3

    # Create test data
    train_input = jnp.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=jnp.int32)
    train_output = jnp.array([[[1, 0, 2], [4, 3, 5], [7, 6, 8]]], dtype=jnp.int32)
    test_input = jnp.array([[[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=jnp.int32)
    test_output = jnp.array([[[8, 9, 7], [5, 6, 4], [2, 3, 1]]], dtype=jnp.int32)

    jax_arc_task = JaxArcTask(
        input_grids_examples=train_input,
        input_masks_examples=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        output_grids_examples=train_output,
        output_masks_examples=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        num_train_pairs=1,
        test_input_grids=test_input,
        test_input_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        true_test_output_grids=test_output,
        true_test_output_masks=jnp.ones((1, grid_h, grid_w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    # Test get_train_input_grid
    train_input_grid = jax_arc_task.get_train_input_grid(0)
    chex.assert_trees_all_equal(train_input_grid.data, train_input[0])

    # Test get_train_output_grid
    train_output_grid = jax_arc_task.get_train_output_grid(0)
    chex.assert_trees_all_equal(train_output_grid.data, train_output[0])

    # Test get_test_input_grid
    test_input_grid = jax_arc_task.get_test_input_grid(0)
    chex.assert_trees_all_equal(test_input_grid.data, test_input[0])

    # Test get_test_output_grid
    test_output_grid = jax_arc_task.get_test_output_grid(0)
    chex.assert_trees_all_equal(test_output_grid.data, test_output[0])

    # Test get_train_pair
    train_pair = jax_arc_task.get_train_pair(0)
    chex.assert_trees_all_equal(train_pair.input_grid.data, train_input[0])
    chex.assert_trees_all_equal(train_pair.output_grid.data, train_output[0])

    # Test get_test_pair
    test_pair = jax_arc_task.get_test_pair(0)
    chex.assert_trees_all_equal(test_pair.input_grid.data, test_input[0])
    chex.assert_trees_all_equal(test_pair.output_grid.data, test_output[0])
