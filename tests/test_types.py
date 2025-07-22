"""
Comprehensive tests for core type system in JaxARC.

This module tests the core Equinox modules (Grid, JaxArcTask, ARCLEAction, TaskPair)
for proper initialization, validation, JAX compatibility, and JAXTyping compliance.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import (
    ARCLEAction,
    ARCLEOperationType,
    Grid,
    JaxArcTask,
    TaskPair,
)
from jaxarc.utils.jax_types import (
    ContinuousSelectionArray,
    GridArray,
    MaskArray,
    OperationId,
    TaskIndex,
    TaskInputGrids,
    TaskOutputGrids,
)
from tests.equinox_test_utils import (
    EquinoxMockFactory,
)


class TestGridModule:
    """Comprehensive tests for Grid Equinox module."""

    def test_grid_creation(self):
        """Tests the creation of a Grid object with valid data."""
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)

        chex.assert_shape(grid.data, (2, 3))
        chex.assert_type(grid.data, jnp.integer)
        chex.assert_trees_all_equal(grid.data, data)
        chex.assert_trees_all_equal(grid.mask, mask)

    def test_grid_invalid_rank(self):
        """Tests that Grid raises assertion error for non-2D arrays."""
        with pytest.raises(AssertionError):
            Grid(
                data=jnp.array([1, 2, 3], dtype=jnp.int32),
                mask=jnp.ones(3, dtype=jnp.bool_),
            ).__check_init__()  # 1D array

        with pytest.raises(AssertionError):
            Grid(
                data=jnp.array([[[1]]], dtype=jnp.int32),
                mask=jnp.ones((1, 1, 1), dtype=jnp.bool_),
            ).__check_init__()  # 3D array

    def test_grid_non_integer_type(self):
        """Tests that Grid raises assertion error for non-integer arrays."""
        with pytest.raises(AssertionError):
            Grid(
                data=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
                mask=jnp.ones((1, 2), dtype=jnp.bool_),
            ).__check_init__()

    def test_grid_color_validation(self):
        """Tests that Grid validates color values are in ARC range (0-9)."""
        # Valid color values (0-9)
        valid_data = jnp.array([[0, 5, 9], [1, 3, 7]], dtype=jnp.int32)
        valid_mask = jnp.ones((2, 3), dtype=jnp.bool_)

        grid = Grid(data=valid_data, mask=valid_mask)
        assert jnp.min(grid.data) >= 0
        assert jnp.max(grid.data) <= 9

        # Test with edge case values
        edge_data = jnp.array([[0, 9]], dtype=jnp.int32)
        edge_mask = jnp.ones((1, 2), dtype=jnp.bool_)

        edge_grid = Grid(data=edge_data, mask=edge_mask)
        assert jnp.min(edge_grid.data) == 0
        assert jnp.max(edge_grid.data) == 9

        # Test with -1 (background masking)
        bg_data = jnp.array([[-1, 0, 5]], dtype=jnp.int32)
        bg_mask = jnp.array([[False, True, True]], dtype=jnp.bool_)

        bg_grid = Grid(data=bg_data, mask=bg_mask)
        assert jnp.min(bg_grid.data) == -1
        assert jnp.max(bg_grid.data) <= 9

        # Test with invalid color values (> 9)
        with pytest.raises(ValueError):
            invalid_data = jnp.array([[10, 11]], dtype=jnp.int32)
            invalid_mask = jnp.ones((1, 2), dtype=jnp.bool_)
            invalid_grid = Grid(data=invalid_data, mask=invalid_mask)
            invalid_grid.__check_init__()

        # Test with invalid color values (< -1)
        with pytest.raises(ValueError):
            invalid_data = jnp.array([[-2, 0]], dtype=jnp.int32)
            invalid_mask = jnp.ones((1, 2), dtype=jnp.bool_)
            invalid_grid = Grid(data=invalid_data, mask=invalid_mask)
            invalid_grid.__check_init__()

    def test_grid_shape_property(self):
        """Tests the shape property of Grid."""
        # Full grid
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)
        assert grid.shape == (2, 3)

        # Partial grid with mask
        data = jnp.array([[0, 1, 2, 0], [3, 4, 5, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        mask = jnp.array(
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ],
            dtype=jnp.bool_,
        )
        grid = Grid(data=data, mask=mask)

        # The shape property should return the actual grid dimensions based on the mask
        # This requires mocking the grid_utils.get_actual_grid_shape_from_mask function
        # For now, we'll just check that it returns a tuple of two integers
        assert isinstance(grid.shape, tuple)
        assert len(grid.shape) == 2
        assert isinstance(grid.shape[0], int)
        assert isinstance(grid.shape[1], int)

    def test_grid_jax_compatibility(self):
        """Tests that Grid works with JAX transformations."""
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)

        # Test jit
        @jax.jit
        def grid_identity(g):
            return g

        jitted_grid = grid_identity(grid)
        # Check arrays are equal (avoiding weak type issues)
        assert jnp.array_equal(grid.data, jitted_grid.data)
        assert jnp.array_equal(grid.mask, jitted_grid.mask)

        # Test vmap with simpler approach - create individual grids and stack them
        grids = [
            Grid(data=data, mask=mask),
            Grid(data=data + 1, mask=mask),
            Grid(data=data + 2, mask=mask),
        ]

        @jax.vmap
        def extract_data(g):
            return g.data

        # Stack the grids into a pytree structure that vmap can handle
        batch_grid = jax.tree.map(lambda *args: jnp.stack(args), *grids)

        # Apply vmap
        result_data = extract_data(batch_grid)

        # Check result
        assert result_data.shape == (3, 2, 3)  # Batch, height, width

    def test_grid_equinox_module_properties(self):
        """Tests that Grid has all expected Equinox module properties."""
        # Create a grid instance
        data = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)

        # Test PyTree structure
        leaves, treedef = jax.tree.flatten(grid)
        reconstructed = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(grid.data, reconstructed.data)
        assert jnp.array_equal(grid.mask, reconstructed.mask)

        # Test immutability
        with pytest.raises(AttributeError):
            grid.data = jnp.zeros((2, 2), dtype=jnp.int32)

        # Test replace method
        new_data = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
        modified_grid = eqx.tree_at(lambda x: x.data, grid, new_data)
        assert jnp.array_equal(modified_grid.data, new_data)
        assert jnp.array_equal(modified_grid.mask, mask)
        assert jnp.array_equal(grid.data, data)  # Original unchanged

    def test_grid_jaxtyping_annotations(self):
        """Tests that Grid works with JAXTyping annotations."""
        # Create grid with explicit JAXTyping annotations
        data: GridArray = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask: MaskArray = jnp.ones((2, 3), dtype=jnp.bool_)
        grid = Grid(data=data, mask=mask)

        # Verify the types are preserved
        assert isinstance(grid.data, jnp.ndarray)
        assert isinstance(grid.mask, jnp.ndarray)
        chex.assert_type(grid.data, jnp.int32)
        chex.assert_type(grid.mask, jnp.bool_)
        chex.assert_shape(grid.data, (2, 3))
        chex.assert_shape(grid.mask, (2, 3))

        # Test with batch dimensions
        batch_data: GridArray = jnp.stack([data, data + 1])
        batch_mask: MaskArray = jnp.stack([mask, mask])

        assert batch_data.shape == (2, 2, 3)  # batch, height, width
        assert batch_mask.shape == (2, 2, 3)  # batch, height, width

    def test_grid_with_complex_mask(self):
        """Tests Grid with complex mask patterns."""
        # Create a grid with a complex mask pattern
        data = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]], dtype=jnp.int32)

        # L-shaped mask
        mask = jnp.array(
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, True, True, True],
            ],
            dtype=jnp.bool_,
        )

        grid = Grid(data=data, mask=mask)

        # Check that the grid was created correctly
        chex.assert_shape(grid.data, (3, 4))
        chex.assert_shape(grid.mask, (3, 4))

        # Test with JAX transformations
        jitted_grid = jax.jit(lambda g: g)(grid)
        assert eqx.tree_equal(grid, jitted_grid)


class TestTaskPairModule:
    """Comprehensive tests for TaskPair Equinox module."""

    def test_task_pair_creation(self):
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

    def test_task_pair_with_different_shapes(self):
        """Tests TaskPair with input and output grids of different shapes."""
        input_grid = Grid(
            data=jnp.array([[0, 1, 2]], dtype=jnp.int32),
            mask=jnp.ones((1, 3), dtype=jnp.bool_),
        )
        output_grid = Grid(
            data=jnp.array([[1, 0], [2, 3]], dtype=jnp.int32),
            mask=jnp.ones((2, 2), dtype=jnp.bool_),
        )

        task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        assert input_grid.data.shape != output_grid.data.shape
        assert task_pair.input_grid.data.shape == (1, 3)
        assert task_pair.output_grid.data.shape == (2, 2)

    def test_task_pair_jax_compatibility(self):
        """Tests that TaskPair works with JAX transformations."""
        input_grid = Grid(
            data=jnp.array([[0, 1]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
        output_grid = Grid(
            data=jnp.array([[1, 0]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
        task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Test jit
        @jax.jit
        def task_pair_identity(tp):
            return tp

        jitted_task_pair = task_pair_identity(task_pair)
        assert eqx.tree_equal(task_pair, jitted_task_pair)

        # Test that we can modify fields using eqx.tree_at
        new_input_grid = Grid(
            data=jnp.array([[2, 3]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
        modified_task_pair = eqx.tree_at(
            lambda tp: tp.input_grid, task_pair, new_input_grid
        )

        assert eqx.tree_equal(modified_task_pair.input_grid, new_input_grid)
        assert eqx.tree_equal(modified_task_pair.output_grid, task_pair.output_grid)

    def test_task_pair_equinox_module_properties(self):
        """Tests that TaskPair has all expected Equinox module properties."""
        input_grid = Grid(
            data=jnp.array([[0, 1]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
        output_grid = Grid(
            data=jnp.array([[1, 0]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )

        task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Test PyTree structure
        leaves, treedef = jax.tree.flatten(task_pair)
        reconstructed = jax.tree.unflatten(treedef, leaves)
        assert eqx.tree_equal(task_pair.input_grid, reconstructed.input_grid)
        assert eqx.tree_equal(task_pair.output_grid, reconstructed.output_grid)

        # Test immutability
        with pytest.raises(AttributeError):
            task_pair.input_grid = input_grid

        # Test replace method
        new_input_grid = Grid(
            data=jnp.array([[2, 3]], dtype=jnp.int32),
            mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
        modified_task_pair = eqx.tree_at(
            lambda tp: tp.input_grid, task_pair, new_input_grid
        )
        assert eqx.tree_equal(modified_task_pair.input_grid, new_input_grid)
        assert eqx.tree_equal(modified_task_pair.output_grid, task_pair.output_grid)

    def test_task_pair_with_complex_grids(self):
        """Tests TaskPair with complex grid structures."""
        # Create input grid with partial mask
        input_data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        input_mask = jnp.array(
            [[True, True, False], [True, True, False]], dtype=jnp.bool_
        )
        input_grid = Grid(data=input_data, mask=input_mask)

        # Create output grid with different mask
        output_data = jnp.array([[5, 4, 3], [2, 1, 0]], dtype=jnp.int32)
        output_mask = jnp.array(
            [[False, True, True], [False, True, True]], dtype=jnp.bool_
        )
        output_grid = Grid(data=output_data, mask=output_mask)

        # Create task pair
        task_pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Verify the grids were stored correctly
        assert eqx.tree_equal(task_pair.input_grid, input_grid)
        assert eqx.tree_equal(task_pair.output_grid, output_grid)

        # Test with JAX transformations
        jitted_task_pair = jax.jit(lambda tp: tp)(task_pair)
        assert eqx.tree_equal(task_pair, jitted_task_pair)


class TestJaxArcTaskModule:
    """Comprehensive tests for JaxArcTask Equinox module."""

    def test_jax_arc_task_creation(self):
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
            test_input_grids=jnp.zeros(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            test_input_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
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
        chex.assert_shape(
            parsed_data.test_input_grids, (max_test_pairs, grid_h, grid_w)
        )
        chex.assert_type(parsed_data.task_index, jnp.int32)

    def test_jax_arc_task_shape_validation(self):
        """Tests JaxArcTask shape validation during __check_init__."""
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

        # Test with mismatched shapes
        with pytest.raises(ValueError):
            JaxArcTask(
                input_grids_examples=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
                input_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                output_grids_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
                output_masks_examples=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                num_train_pairs=2,
                test_input_grids=jnp.zeros(
                    (1, grid_h + 1, grid_w), dtype=jnp.int32
                ),  # Different height
                test_input_masks=jnp.ones((1, grid_h + 1, grid_w), dtype=jnp.bool_),
                true_test_output_grids=jnp.ones(
                    (1, grid_h + 1, grid_w), dtype=jnp.int32
                ),
                true_test_output_masks=jnp.ones(
                    (1, grid_h + 1, grid_w), dtype=jnp.bool_
                ),
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            ).__check_init__()

    def test_jax_arc_task_count_validation(self):
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

        # Test with invalid counts
        with pytest.raises(ValueError):
            JaxArcTask(
                input_grids_examples=jnp.zeros((3, grid_h, grid_w), dtype=jnp.int32),
                input_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                output_grids_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.int32),
                output_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                num_train_pairs=4,  # Invalid: > max_train_pairs (3)
                test_input_grids=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
                test_input_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                true_test_output_grids=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
                true_test_output_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            ).__check_init__()

        with pytest.raises(ValueError):
            JaxArcTask(
                input_grids_examples=jnp.zeros((3, grid_h, grid_w), dtype=jnp.int32),
                input_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                output_grids_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.int32),
                output_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                num_train_pairs=2,
                test_input_grids=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
                test_input_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                true_test_output_grids=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
                true_test_output_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                num_test_pairs=3,  # Invalid: > max_test_pairs (2)
                task_index=jnp.array(0, dtype=jnp.int32),
            ).__check_init__()

        with pytest.raises(ValueError):
            JaxArcTask(
                input_grids_examples=jnp.zeros((3, grid_h, grid_w), dtype=jnp.int32),
                input_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                output_grids_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.int32),
                output_masks_examples=jnp.ones((3, grid_h, grid_w), dtype=jnp.bool_),
                num_train_pairs=-1,  # Invalid: < 0
                test_input_grids=jnp.zeros((2, grid_h, grid_w), dtype=jnp.int32),
                test_input_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                true_test_output_grids=jnp.ones((2, grid_h, grid_w), dtype=jnp.int32),
                true_test_output_masks=jnp.ones((2, grid_h, grid_w), dtype=jnp.bool_),
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            ).__check_init__()

    def test_jax_arc_task_pytree_compatibility(self):
        """Tests that JaxArcTask works with JAX transformations."""

        def transform_grids(parsed_data: JaxArcTask) -> JaxArcTask:
            return eqx.tree_at(
                lambda x: x.input_grids_examples,
                parsed_data,
                parsed_data.input_grids_examples + 1,
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

    def test_jax_arc_task_utility_methods(self):
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

    def test_jax_arc_task_equinox_module_properties(self):
        """Tests that JaxArcTask has all expected Equinox module properties."""
        # Use the EquinoxMockFactory to create a test JaxArcTask
        task = EquinoxMockFactory.create_mock_jax_arc_task()

        # Test PyTree structure
        leaves, treedef = jax.tree.flatten(task)
        reconstructed = jax.tree.unflatten(treedef, leaves)
        assert eqx.tree_equal(task, reconstructed)

        # Test immutability
        with pytest.raises(AttributeError):
            task.num_train_pairs = 5

        # Test replace method
        new_task_index = jnp.array(42, dtype=jnp.int32)
        modified_task = eqx.tree_at(lambda x: x.task_index, task, new_task_index)
        assert modified_task.task_index == 42
        assert task.task_index == 0  # Original unchanged

    def test_jax_arc_task_jaxtyping_annotations(self):
        """Tests that JaxArcTask works with JAXTyping annotations."""
        # Create task with explicit JAXTyping annotations
        grid_h, grid_w = 4, 4
        max_train_pairs, max_test_pairs = 2, 1

        input_grids: TaskInputGrids = jnp.zeros(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
        )
        input_masks = jnp.ones((max_train_pairs, grid_h, grid_w), dtype=jnp.bool_)
        output_grids: TaskOutputGrids = jnp.ones(
            (max_train_pairs, grid_h, grid_w), dtype=jnp.int32
        )
        output_masks = jnp.ones((max_train_pairs, grid_h, grid_w), dtype=jnp.bool_)
        task_idx: TaskIndex = jnp.array(42, dtype=jnp.int32)

        task = JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=max_train_pairs,
            test_input_grids=jnp.zeros(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            test_input_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            true_test_output_grids=jnp.zeros(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            true_test_output_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            num_test_pairs=max_test_pairs,
            task_index=task_idx,
        )

        # Verify the types are preserved
        chex.assert_type(task.input_grids_examples, jnp.int32)
        chex.assert_type(task.input_masks_examples, jnp.bool_)
        chex.assert_type(task.task_index, jnp.int32)
        chex.assert_shape(task.input_grids_examples, (max_train_pairs, grid_h, grid_w))
        chex.assert_shape(task.task_index, ())

    def test_jax_arc_task_with_zero_pairs(self):
        """Tests JaxArcTask with zero pairs (edge case)."""
        grid_h, grid_w = 3, 3
        max_train_pairs, max_test_pairs = 2, 1

        # Create task with zero train and test pairs
        task = JaxArcTask(
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
            num_train_pairs=0,  # Zero train pairs
            test_input_grids=jnp.zeros(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            test_input_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            true_test_output_grids=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.int32
            ),
            true_test_output_masks=jnp.ones(
                (max_test_pairs, grid_h, grid_w), dtype=jnp.bool_
            ),
            num_test_pairs=0,  # Zero test pairs
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Verify the counts
        assert task.num_train_pairs == 0
        assert task.num_test_pairs == 0

        # Test with JAX transformations
        jitted_task = jax.jit(lambda t: t)(task)
        assert jitted_task.num_train_pairs == 0
        assert jitted_task.num_test_pairs == 0


class TestARCLEActionModule:
    """Comprehensive tests for ARCLEAction Equinox module."""

    def test_arcle_action_creation(self):
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

    def test_arcle_action_selection_bounds(self):
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

        # Invalid selection values (< 0.0)
        with pytest.raises(ValueError):
            invalid_action = ARCLEAction(
                selection=jnp.array([[-0.1, 0.5], [1.0, 0.25]], dtype=jnp.float32),
                operation=jnp.array(10, dtype=jnp.int32),
                agent_id=2,
                timestamp=200,
            )
            invalid_action.__check_init__()

        # Invalid selection values (> 1.0)
        with pytest.raises(ValueError):
            invalid_action = ARCLEAction(
                selection=jnp.array([[0.0, 0.5], [1.1, 0.25]], dtype=jnp.float32),
                operation=jnp.array(10, dtype=jnp.int32),
                agent_id=2,
                timestamp=200,
            )
            invalid_action.__check_init__()

    def test_arcle_action_operation_bounds(self):
        """Tests ARCLEAction operation ID validation."""
        # Valid operation ID (within 0-34 range)
        action = ARCLEAction(
            selection=jnp.array([[1.0]], dtype=jnp.float32),
            operation=jnp.array(34, dtype=jnp.int32),  # Maximum valid operation
            agent_id=0,
            timestamp=0,
        )

        assert 0 <= action.operation <= 34

        # Invalid operation ID (> 34)
        with pytest.raises(ValueError):
            invalid_action = ARCLEAction(
                selection=jnp.array([[1.0]], dtype=jnp.float32),
                operation=jnp.array(35, dtype=jnp.int32),
                agent_id=0,
                timestamp=0,
            )
            invalid_action.__check_init__()

        # Invalid operation ID (< 0)
        with pytest.raises(ValueError):
            invalid_action = ARCLEAction(
                selection=jnp.array([[1.0]], dtype=jnp.float32),
                operation=jnp.array(-1, dtype=jnp.int32),
                agent_id=0,
                timestamp=0,
            )
            invalid_action.__check_init__()

    def test_arcle_action_jax_compatibility(self):
        """Tests that ARCLEAction works with JAX transformations."""
        action = ARCLEAction(
            selection=jnp.array([[0.5, 1.0], [0.0, 0.8]], dtype=jnp.float32),
            operation=jnp.array(5, dtype=jnp.int32),
            agent_id=1,
            timestamp=100,
        )

        # Test jit
        @jax.jit
        def action_identity(a):
            return a

        jitted_action = action_identity(action)
        # Check arrays are equal (avoiding weak type issues)
        assert jnp.array_equal(action.selection, jitted_action.selection)
        assert jnp.array_equal(action.operation, jitted_action.operation)
        # Note: agent_id and timestamp may become weak types after JIT

        # Test vmap with batch of actions
        batch_size = 3
        batch_actions = []
        for i in range(batch_size):
            # Create actions with different operation IDs
            batch_action = eqx.tree_at(
                lambda x: x.operation, action, jnp.array(i, dtype=jnp.int32)
            )
            batch_actions.append(batch_action)

        # Apply vmap to extract operation IDs
        @jax.vmap
        def extract_operation(a):
            return a.operation

        # Stack the actions into a pytree structure that vmap can handle
        batched_actions = jax.tree.map(lambda *args: jnp.stack(args), *batch_actions)

        # Apply vmap
        operations = extract_operation(batched_actions)
        assert operations.shape == (batch_size,)
        assert jnp.array_equal(operations, jnp.array([0, 1, 2], dtype=jnp.int32))

    def test_arcle_action_operation_types(self):
        """Tests ARCLEAction with different operation types."""
        # Test with fill operations (0-9)
        fill_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.FILL_5, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert fill_action.operation == ARCLEOperationType.FILL_5

        # Test with flood fill operations (10-19)
        flood_fill_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.FLOOD_FILL_3, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert flood_fill_action.operation == ARCLEOperationType.FLOOD_FILL_3

        # Test with move operations (20-23)
        move_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.MOVE_RIGHT, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert move_action.operation == ARCLEOperationType.MOVE_RIGHT

        # Test with rotate operations (24-25)
        rotate_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.ROTATE_C, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert rotate_action.operation == ARCLEOperationType.ROTATE_C

        # Test with flip operations (26-27)
        flip_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.FLIP_HORIZONTAL, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert flip_action.operation == ARCLEOperationType.FLIP_HORIZONTAL

        # Test with clipboard operations (28-30)
        clipboard_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.COPY, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert clipboard_action.operation == ARCLEOperationType.COPY

        # Test with grid operations (31-33)
        grid_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.CLEAR, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert grid_action.operation == ARCLEOperationType.CLEAR

        # Test with submit operation (34)
        submit_action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(ARCLEOperationType.SUBMIT, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )
        assert submit_action.operation == ARCLEOperationType.SUBMIT

    def test_arcle_action_equinox_module_properties(self):
        """Tests that ARCLEAction has all expected Equinox module properties."""
        action = ARCLEAction(
            selection=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            operation=jnp.array(5, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )

        # Test PyTree structure
        leaves, treedef = jax.tree.flatten(action)
        reconstructed = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(action.selection, reconstructed.selection)
        assert jnp.array_equal(action.operation, reconstructed.operation)

        # Test immutability
        with pytest.raises(AttributeError):
            action.selection = jnp.zeros((1, 2), dtype=jnp.float32)

        # Test replace method
        new_selection = jnp.array([[0.8, 0.2]], dtype=jnp.float32)
        modified_action = eqx.tree_at(lambda a: a.selection, action, new_selection)
        assert jnp.array_equal(modified_action.selection, new_selection)
        assert jnp.array_equal(modified_action.operation, action.operation)

    def test_arcle_action_jaxtyping_annotations(self):
        """Tests that ARCLEAction works with JAXTyping annotations."""
        # Create action with explicit JAXTyping annotations
        selection: ContinuousSelectionArray = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
        operation: OperationId = jnp.array(5, dtype=jnp.int32)

        action = ARCLEAction(
            selection=selection,
            operation=operation,
            agent_id=0,
            timestamp=0,
        )

        # Verify the types are preserved
        chex.assert_type(action.selection, jnp.float32)
        chex.assert_type(action.operation, jnp.int32)
        chex.assert_shape(action.selection, (1, 2))
        chex.assert_shape(action.operation, ())

    def test_arcle_action_with_complex_selection(self):
        """Tests ARCLEAction with complex selection patterns."""
        # Create a complex selection pattern (e.g., a circle)
        height, width = 5, 5
        selection = jnp.zeros((height, width), dtype=jnp.float32)

        # Create a circular selection pattern
        center_h, center_w = height // 2, width // 2
        radius = min(height, width) // 2 - 0.5

        for h in range(height):
            for w in range(width):
                dist = jnp.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
                # Smooth falloff from center
                selection = selection.at[h, w].set(
                    jnp.maximum(0.0, 1.0 - dist / radius)
                )

        # Create action with this selection
        action = ARCLEAction(
            selection=selection,
            operation=jnp.array(ARCLEOperationType.FILL_3, dtype=jnp.int32),
            agent_id=0,
            timestamp=0,
        )

        # Verify the selection was stored correctly
        chex.assert_shape(action.selection, (height, width))
        assert jnp.min(action.selection) >= 0.0
        assert jnp.max(action.selection) <= 1.0

        # Test with JAX transformations
        jitted_action = jax.jit(lambda a: a)(action)
        # Check arrays are equal (avoiding weak type issues)
        assert jnp.array_equal(action.selection, jitted_action.selection)
        assert jnp.array_equal(action.operation, jitted_action.operation)


def test_jaxtyping_runtime_validation():
    """Tests JAXTyping runtime validation for all core types."""
    # Test Grid with JAXTyping
    data: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    mask: MaskArray = jnp.ones((2, 2), dtype=jnp.bool_)
    grid = Grid(data=data, mask=mask)

    # Test TaskPair with JAXTyping
    task_pair = TaskPair(input_grid=grid, output_grid=grid)

    # Test JaxArcTask with JAXTyping
    input_grids: TaskInputGrids = jnp.zeros((2, 3, 3), dtype=jnp.int32)
    output_grids: TaskOutputGrids = jnp.ones((2, 3, 3), dtype=jnp.int32)
    task_idx: TaskIndex = jnp.array(0, dtype=jnp.int32)

    task = JaxArcTask(
        input_grids_examples=input_grids,
        input_masks_examples=jnp.ones((2, 3, 3), dtype=jnp.bool_),
        output_grids_examples=output_grids,
        output_masks_examples=jnp.ones((2, 3, 3), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=task_idx,
    )

    # Test ARCLEAction with JAXTyping
    selection: ContinuousSelectionArray = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
    operation: OperationId = jnp.array(5, dtype=jnp.int32)

    action = ARCLEAction(
        selection=selection,
        operation=operation,
        agent_id=0,
        timestamp=0,
    )

    # All objects should be created successfully with JAXTyping annotations
    assert isinstance(grid, Grid)
    assert isinstance(task_pair, TaskPair)
    assert isinstance(task, JaxArcTask)
    assert isinstance(action, ARCLEAction)


def test_comprehensive_jax_transformations():
    """Comprehensive test of JAX transformations for all core types."""
    # Create test instances
    grid = EquinoxMockFactory.create_mock_grid()
    task_pair = EquinoxMockFactory.create_mock_task_pair()
    task = EquinoxMockFactory.create_mock_jax_arc_task()
    action = EquinoxMockFactory.create_mock_arcle_action()

    # Test jit transformation for all types
    jitted_grid = jax.jit(lambda x: x)(grid)
    jitted_task_pair = jax.jit(lambda x: x)(task_pair)
    jitted_task = jax.jit(lambda x: x)(task)
    jitted_action = jax.jit(lambda x: x)(action)

    # Verify equality after jit - note that we need to check fields individually
    # because JIT can convert Python integers to weak JAX types
    assert eqx.tree_equal(grid, jitted_grid)
    assert eqx.tree_equal(task_pair, jitted_task_pair)

    # For task, check arrays directly
    assert jnp.array_equal(task.input_grids_examples, jitted_task.input_grids_examples)
    assert jnp.array_equal(task.input_masks_examples, jitted_task.input_masks_examples)
    assert jnp.array_equal(
        task.output_grids_examples, jitted_task.output_grids_examples
    )
    assert jnp.array_equal(
        task.output_masks_examples, jitted_task.output_masks_examples
    )
    assert jnp.array_equal(task.test_input_grids, jitted_task.test_input_grids)
    assert jnp.array_equal(task.test_input_masks, jitted_task.test_input_masks)
    assert jnp.array_equal(
        task.true_test_output_grids, jitted_task.true_test_output_grids
    )
    assert jnp.array_equal(
        task.true_test_output_masks, jitted_task.true_test_output_masks
    )
    assert task.num_train_pairs == jitted_task.num_train_pairs
    assert task.num_test_pairs == jitted_task.num_test_pairs
    assert jnp.array_equal(task.task_index, jitted_task.task_index)

    # For action, check fields directly
    assert jnp.array_equal(action.selection, jitted_action.selection)
    assert jnp.array_equal(action.operation, jitted_action.operation)
    assert action.agent_id == jitted_action.agent_id
    assert action.timestamp == jitted_action.timestamp


def test_jaxtyping_runtime_validation():
    """Tests JAXTyping runtime validation for all core types."""
    # Test Grid with JAXTyping
    data: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    mask: MaskArray = jnp.ones((2, 2), dtype=jnp.bool_)
    grid = Grid(data=data, mask=mask)

    # Test TaskPair with JAXTyping
    task_pair = TaskPair(input_grid=grid, output_grid=grid)

    # Test JaxArcTask with JAXTyping
    input_grids: TaskInputGrids = jnp.zeros((2, 3, 3), dtype=jnp.int32)
    output_grids: TaskOutputGrids = jnp.ones((2, 3, 3), dtype=jnp.int32)
    task_idx: TaskIndex = jnp.array(0, dtype=jnp.int32)

    task = JaxArcTask(
        input_grids_examples=input_grids,
        input_masks_examples=jnp.ones((2, 3, 3), dtype=jnp.bool_),
        output_grids_examples=output_grids,
        output_masks_examples=jnp.ones((2, 3, 3), dtype=jnp.bool_),
        num_train_pairs=2,
        test_input_grids=jnp.zeros((1, 3, 3), dtype=jnp.int32),
        test_input_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
        true_test_output_grids=jnp.ones((1, 3, 3), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((1, 3, 3), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=task_idx,
    )

    # Test ARCLEAction with JAXTyping
    selection: ContinuousSelectionArray = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
    operation: OperationId = jnp.array(5, dtype=jnp.int32)

    action = ARCLEAction(
        selection=selection,
        operation=operation,
        agent_id=0,
        timestamp=0,
    )

    # All objects should be created successfully with JAXTyping annotations
    assert isinstance(grid, Grid)
    assert isinstance(task_pair, TaskPair)
    assert isinstance(task, JaxArcTask)
    assert isinstance(action, ARCLEAction)
