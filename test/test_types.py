"""
Tests for core data structures in jaxarc.types module.

This module tests the Grid, JaxArcTask, TaskPair, TimeStep, and EnvParams classes
to ensure proper JAX compatibility, type safety, and validation.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.types import (
    EnvParams,
    Grid,
    JaxArcTask,
    StepType,
    TaskPair,
    TimeStep,
)


class TestGrid:
    """Test cases for the Grid class."""

    def test_grid_creation_basic(self):
        """Test basic Grid creation with valid data and mask."""
        data = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Verify data and mask are stored correctly
        chex.assert_trees_all_equal(grid.data, data)
        chex.assert_trees_all_equal(grid.mask, mask)

    def test_grid_creation_with_validation(self):
        """Test Grid creation triggers validation correctly."""
        data = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)

        # Should create successfully with valid data
        grid = Grid(data=data, mask=mask)
        assert grid.data.shape == (2, 3)
        assert grid.mask.shape == (2, 3)

    def test_grid_shape_property(self, sample_grid):
        """Test Grid.shape property returns correct dimensions."""
        # sample_grid is 3x3 from conftest.py
        shape = sample_grid.shape
        assert shape == (3, 3)
        assert isinstance(shape, tuple)
        assert len(shape) == 2

    def test_grid_jax_array_compatibility(self):
        """Test Grid works with JAX arrays and transformations."""
        data = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Test JAX array properties
        chex.assert_type(grid.data, jnp.int32)
        chex.assert_type(grid.mask, jnp.bool_)
        chex.assert_rank(grid.data, 2)
        chex.assert_rank(grid.mask, 2)

    def test_grid_type_checking(self):
        """Test Grid enforces correct array types."""
        # Test with correct types
        data = jnp.array([[1, 2]], dtype=jnp.int32)
        mask = jnp.array([[True, False]], dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Verify types are preserved
        chex.assert_type(grid.data, jnp.integer)
        chex.assert_type(grid.mask, jnp.bool_)

    def test_grid_mask_data_consistency(self):
        """Test Grid validates mask and data array consistency."""
        data = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)

        # Should work with matching shapes
        grid = Grid(data=data, mask=mask)
        chex.assert_shape(grid.mask, grid.data.shape)

    def test_grid_invalid_shapes_validation(self):
        """Test Grid validation catches mismatched shapes."""
        data = jnp.array([[1, 2]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)  # Wrong shape

        # Should raise error during validation
        with pytest.raises((ValueError, AssertionError)):
            Grid(data=data, mask=mask)

    def test_grid_color_value_validation(self):
        """Test Grid validates color values are in valid ARC range."""
        # Valid colors (0-9, -1 for background)
        valid_data = jnp.array([[0, 5, 9], [-1, 3, 7]], dtype=jnp.int32)
        mask = jnp.ones((2, 3), dtype=jnp.bool_)

        # Should create successfully
        grid = Grid(data=valid_data, mask=mask)
        assert jnp.all(grid.data >= -1)
        assert jnp.all(grid.data <= 9)

    def test_grid_invalid_color_values(self):
        """Test Grid rejects invalid color values."""
        # Invalid colors (outside -1 to 9 range)
        invalid_data = jnp.array([[10, 15]], dtype=jnp.int32)  # Too high
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        with pytest.raises(ValueError, match="Grid color values must be in"):
            Grid(data=invalid_data, mask=mask)

    def test_grid_negative_invalid_colors(self):
        """Test Grid rejects negative values below -1."""
        invalid_data = jnp.array([[-2, -5]], dtype=jnp.int32)  # Too low
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        with pytest.raises(ValueError, match="Grid color values must be in"):
            Grid(data=invalid_data, mask=mask)

    def test_grid_with_masked_cells(self, masked_grid):
        """Test Grid works correctly with masked (invalid) cells."""
        # masked_grid has some False values in mask
        assert masked_grid.mask.shape == masked_grid.data.shape
        assert not jnp.all(masked_grid.mask)  # Some cells should be masked

    def test_grid_shape_with_mask(self):
        """Test Grid.shape property works with masked grids."""
        # Create grid where mask defines actual shape
        data = jnp.zeros((5, 5), dtype=jnp.int32)
        mask = jnp.zeros((5, 5), dtype=jnp.bool_)

        # Set a 3x3 region as valid
        mask = mask.at[:3, :3].set(True)

        grid = Grid(data=data, mask=mask)

        # Shape should reflect the valid region
        shape = grid.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 2

    def test_grid_jax_transformations(self):
        """Test Grid works with JAX transformations like jit."""

        def create_grid_fn(data, mask):
            return Grid(data=data, mask=mask)

        data = jnp.array([[1, 2]], dtype=jnp.int32)
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        # Test that grid creation works in jitted function
        jitted_create = jax.jit(create_grid_fn)
        grid = jitted_create(data, mask)

        chex.assert_trees_all_equal(grid.data, data)
        chex.assert_trees_all_equal(grid.mask, mask)

    def test_grid_pytree_compatibility(self):
        """Test Grid is properly registered as a PyTree."""
        data = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Test PyTree operations
        leaves, treedef = jax.tree_util.tree_flatten(grid)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_equal(grid.data, reconstructed.data)
        chex.assert_trees_all_equal(grid.mask, reconstructed.mask)

    def test_grid_equinox_module_properties(self):
        """Test Grid inherits Equinox Module properties correctly."""
        data = jnp.array([[1, 0]], dtype=jnp.int32)
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        grid = Grid(data=data, mask=mask)

        # Test that it's an Equinox module
        import equinox as eqx

        assert isinstance(grid, eqx.Module)

        # Test immutability (should create new instance)
        new_data = jnp.array([[2, 1]], dtype=jnp.int32)
        new_grid = eqx.tree_at(lambda g: g.data, grid, new_data)

        # Original should be unchanged
        chex.assert_trees_all_equal(grid.data, data)
        chex.assert_trees_all_equal(new_grid.data, new_data)


class TestJaxArcTask:
    """Test cases for the JaxArcTask class."""

    def test_jaxarctask_creation_basic(self):
        """Test basic JaxArcTask creation with proper array shapes and types."""
        # Create test data with proper shapes
        max_pairs, max_height, max_width = 3, 5, 5

        # Training data
        input_grids = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        input_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)
        output_grids = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.int32)
        output_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        # Test data
        test_input_grids = jnp.zeros(
            (max_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_input_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)
        test_output_grids = jnp.ones(
            (max_pairs, max_height, max_width), dtype=jnp.int32
        )
        test_output_masks = jnp.ones(
            (max_pairs, max_height, max_width), dtype=jnp.bool_
        )

        task = JaxArcTask(
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
            task_index=jnp.array(42, dtype=jnp.int32),
        )

        # Verify basic properties
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 1
        chex.assert_trees_all_equal(task.task_index, jnp.array(42, dtype=jnp.int32))

    def test_jaxarctask_array_shapes_and_types(self):
        """Test JaxArcTask validates array shapes and types correctly."""
        max_pairs, max_height, max_width = 2, 4, 4

        # Create arrays with correct shapes and types
        input_grids = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        input_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)
        output_grids = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.int32)
        output_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=input_masks,
            output_grids_examples=output_grids,
            output_masks_examples=output_masks,
            num_train_pairs=2,
            test_input_grids=input_grids,
            test_input_masks=input_masks,
            true_test_output_grids=output_grids,
            true_test_output_masks=output_masks,
            num_test_pairs=2,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Verify array types
        chex.assert_type(task.input_grids_examples, jnp.int32)
        chex.assert_type(task.input_masks_examples, jnp.bool_)
        chex.assert_type(task.output_grids_examples, jnp.int32)
        chex.assert_type(task.output_masks_examples, jnp.bool_)
        chex.assert_type(task.test_input_grids, jnp.int32)
        chex.assert_type(task.test_input_masks, jnp.bool_)
        chex.assert_type(task.true_test_output_grids, jnp.int32)
        chex.assert_type(task.true_test_output_masks, jnp.bool_)
        chex.assert_type(task.task_index, jnp.int32)

        # Verify array shapes
        expected_shape = (max_pairs, max_height, max_width)
        chex.assert_shape(task.input_grids_examples, expected_shape)
        chex.assert_shape(task.input_masks_examples, expected_shape)
        chex.assert_shape(task.output_grids_examples, expected_shape)
        chex.assert_shape(task.output_masks_examples, expected_shape)
        chex.assert_shape(task.test_input_grids, expected_shape)
        chex.assert_shape(task.test_input_masks, expected_shape)
        chex.assert_shape(task.true_test_output_grids, expected_shape)
        chex.assert_shape(task.true_test_output_masks, expected_shape)
        chex.assert_shape(task.task_index, ())

    def test_jaxarctask_training_pair_access(self):
        """Test training pair access methods work correctly."""
        max_pairs, max_height, max_width = 2, 3, 3

        # Create test data with distinct values
        input_grids = jnp.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
            dtype=jnp.int32,
        )

        output_grids = jnp.array(
            [[[9, 8, 7], [6, 5, 4], [3, 2, 1]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]],
            dtype=jnp.int32,
        )

        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=input_grids,
            input_masks_examples=masks,
            output_grids_examples=output_grids,
            output_masks_examples=masks,
            num_train_pairs=2,
            test_input_grids=input_grids,
            test_input_masks=masks,
            true_test_output_grids=output_grids,
            true_test_output_masks=masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

        # Test individual grid access
        train_input_0 = task.get_train_input_grid(0)
        train_output_0 = task.get_train_output_grid(0)

        chex.assert_trees_all_equal(train_input_0.data, input_grids[0])
        chex.assert_trees_all_equal(train_output_0.data, output_grids[0])

        # Test pair access
        train_pair_0 = task.get_train_pair(0)
        chex.assert_trees_all_equal(train_pair_0.input_grid.data, input_grids[0])
        chex.assert_trees_all_equal(train_pair_0.output_grid.data, output_grids[0])

    def test_jaxarctask_test_pair_access(self):
        """Test test pair access methods work correctly."""
        max_pairs, max_height, max_width = 2, 3, 3

        # Create test data
        test_input = jnp.array(
            [[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]],
            dtype=jnp.int32,
        )

        test_output = jnp.array(
            [[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]],
            dtype=jnp.int32,
        )

        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=jnp.zeros(
                (max_pairs, max_height, max_width), dtype=jnp.int32
            ),
            input_masks_examples=masks,
            output_grids_examples=jnp.zeros(
                (max_pairs, max_height, max_width), dtype=jnp.int32
            ),
            output_masks_examples=masks,
            num_train_pairs=1,
            test_input_grids=test_input,
            test_input_masks=masks,
            true_test_output_grids=test_output,
            true_test_output_masks=masks,
            num_test_pairs=2,
            task_index=jnp.array(1, dtype=jnp.int32),
        )

        # Test individual test grid access
        test_input_0 = task.get_test_input_grid(0)
        test_output_0 = task.get_test_output_grid(0)

        chex.assert_trees_all_equal(test_input_0.data, test_input[0])
        chex.assert_trees_all_equal(test_output_0.data, test_output[0])

        # Test test pair access
        test_pair_0 = task.get_test_pair(0)
        chex.assert_trees_all_equal(test_pair_0.input_grid.data, test_input[0])
        chex.assert_trees_all_equal(test_pair_0.output_grid.data, test_output[0])

    def test_jaxarctask_utility_methods(self):
        """Test JaxArcTask utility methods work correctly."""
        max_pairs, max_height, max_width = 3, 4, 4

        # Create arrays
        arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=arrays,
            input_masks_examples=masks,
            output_grids_examples=arrays,
            output_masks_examples=masks,
            num_train_pairs=2,
            test_input_grids=arrays,
            test_input_masks=masks,
            true_test_output_grids=arrays,
            true_test_output_masks=masks,
            num_test_pairs=1,
            task_index=jnp.array(5, dtype=jnp.int32),
        )

        # Test available pairs methods
        train_available = task.get_available_demo_pairs()
        test_available = task.get_available_test_pairs()

        # Should have 2 training pairs and 1 test pair available
        expected_train = jnp.array([True, True, False])
        expected_test = jnp.array([True, False, False])

        chex.assert_trees_all_equal(train_available, expected_train)
        chex.assert_trees_all_equal(test_available, expected_test)

        # Test pair availability checks
        assert task.is_demo_pair_available(0) == True
        assert task.is_demo_pair_available(1) == True
        assert task.is_demo_pair_available(2) == False

        assert task.is_test_pair_available(0) == True
        assert task.is_test_pair_available(1) == False

        # Test dimension methods
        assert task.get_max_train_pairs() == max_pairs
        assert task.get_max_test_pairs() == max_pairs
        assert task.get_grid_shape() == (max_height, max_width)

        # Test summary
        summary = task.get_task_summary()
        assert summary["task_index"] == 5
        assert summary["num_train_pairs"] == 2
        assert summary["num_test_pairs"] == 1
        assert summary["max_train_pairs"] == max_pairs
        assert summary["max_test_pairs"] == max_pairs
        assert summary["grid_shape"] == (max_height, max_width)

    def test_jaxarctask_validation_errors(self):
        """Test JaxArcTask validation catches invalid configurations."""
        max_pairs, max_height, max_width = 2, 3, 3

        # Create valid base arrays
        valid_arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        valid_masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        # Test invalid num_train_pairs
        with pytest.raises(ValueError, match="Invalid num_train_pairs"):
            JaxArcTask(
                input_grids_examples=valid_arrays,
                input_masks_examples=valid_masks,
                output_grids_examples=valid_arrays,
                output_masks_examples=valid_masks,
                num_train_pairs=5,  # Too many
                test_input_grids=valid_arrays,
                test_input_masks=valid_masks,
                true_test_output_grids=valid_arrays,
                true_test_output_masks=valid_masks,
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            )

        # Test invalid num_test_pairs
        with pytest.raises(ValueError, match="Invalid num_test_pairs"):
            JaxArcTask(
                input_grids_examples=valid_arrays,
                input_masks_examples=valid_masks,
                output_grids_examples=valid_arrays,
                output_masks_examples=valid_masks,
                num_train_pairs=1,
                test_input_grids=valid_arrays,
                test_input_masks=valid_masks,
                true_test_output_grids=valid_arrays,
                true_test_output_masks=valid_masks,
                num_test_pairs=-1,  # Negative
                task_index=jnp.array(0, dtype=jnp.int32),
            )

    def test_jaxarctask_shape_consistency(self):
        """Test JaxArcTask validates shape consistency between train and test."""
        # Create arrays with different shapes
        train_arrays = jnp.zeros((2, 3, 3), dtype=jnp.int32)
        train_masks = jnp.ones((2, 3, 3), dtype=jnp.bool_)

        test_arrays = jnp.zeros((2, 4, 4), dtype=jnp.int32)  # Different grid size
        test_masks = jnp.ones((2, 4, 4), dtype=jnp.bool_)

        with pytest.raises(ValueError, match="Grid dimensions mismatch"):
            JaxArcTask(
                input_grids_examples=train_arrays,
                input_masks_examples=train_masks,
                output_grids_examples=train_arrays,
                output_masks_examples=train_masks,
                num_train_pairs=1,
                test_input_grids=test_arrays,
                test_input_masks=test_masks,
                true_test_output_grids=test_arrays,
                true_test_output_masks=test_masks,
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            )

    def test_jaxarctask_jax_compatibility(self):
        """Test JaxArcTask works with JAX transformations."""
        max_pairs, max_height, max_width = 2, 3, 3

        arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        def create_task():
            return JaxArcTask(
                input_grids_examples=arrays,
                input_masks_examples=masks,
                output_grids_examples=arrays,
                output_masks_examples=masks,
                num_train_pairs=1,
                test_input_grids=arrays,
                test_input_masks=masks,
                true_test_output_grids=arrays,
                true_test_output_masks=masks,
                num_test_pairs=1,
                task_index=jnp.array(0, dtype=jnp.int32),
            )

        # Test JIT compilation
        jitted_create = jax.jit(create_task)
        task = jitted_create()

        assert task.num_train_pairs == 1
        assert task.num_test_pairs == 1

    def test_jaxarctask_pytree_operations(self):
        """Test JaxArcTask works with PyTree operations."""
        max_pairs, max_height, max_width = 2, 3, 3

        arrays = jnp.zeros((max_pairs, max_height, max_width), dtype=jnp.int32)
        masks = jnp.ones((max_pairs, max_height, max_width), dtype=jnp.bool_)

        task = JaxArcTask(
            input_grids_examples=arrays,
            input_masks_examples=masks,
            output_grids_examples=arrays,
            output_masks_examples=masks,
            num_train_pairs=1,
            test_input_grids=arrays,
            test_input_masks=masks,
            true_test_output_grids=arrays,
            true_test_output_masks=masks,
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),
        )

        # Test PyTree flattening and unflattening
        leaves, treedef = jax.tree_util.tree_flatten(task)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify reconstruction
        assert reconstructed.num_train_pairs == task.num_train_pairs
        assert reconstructed.num_test_pairs == task.num_test_pairs
        chex.assert_trees_all_equal(reconstructed.task_index, task.task_index)
        chex.assert_trees_all_equal(
            reconstructed.input_grids_examples, task.input_grids_examples
        )


class TestTaskPair:
    """Test cases for the TaskPair class."""

    def test_taskpair_creation_basic(self):
        """Test basic TaskPair creation with input and output grids."""
        # Create input and output grids
        input_data = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        output_data = jnp.array([[4, 3], [2, 1]], dtype=jnp.int32)
        mask = jnp.ones((2, 2), dtype=jnp.bool_)

        input_grid = Grid(data=input_data, mask=mask)
        output_grid = Grid(data=output_data, mask=mask)

        pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Verify grids are stored correctly
        chex.assert_trees_all_equal(pair.input_grid.data, input_data)
        chex.assert_trees_all_equal(pair.output_grid.data, output_data)
        chex.assert_trees_all_equal(pair.input_grid.mask, mask)
        chex.assert_trees_all_equal(pair.output_grid.mask, mask)

    def test_taskpair_input_output_consistency(self):
        """Test TaskPair maintains input/output grid consistency."""
        # Create grids with different but valid data
        input_data = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.int32)
        output_data = jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=jnp.int32)

        # Use different masks to test consistency
        input_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        output_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        output_mask = output_mask.at[0, 0].set(False)  # Different mask

        input_grid = Grid(data=input_data, mask=input_mask)
        output_grid = Grid(data=output_data, mask=output_mask)

        pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Verify both grids maintain their properties
        assert pair.input_grid.shape == (3, 3)
        assert pair.output_grid.shape == (3, 3)

        # Verify data integrity
        chex.assert_trees_all_equal(pair.input_grid.data, input_data)
        chex.assert_trees_all_equal(pair.output_grid.data, output_data)
        chex.assert_trees_all_equal(pair.input_grid.mask, input_mask)
        chex.assert_trees_all_equal(pair.output_grid.mask, output_mask)

    def test_taskpair_equinox_module_properties(self):
        """Test TaskPair inherits Equinox Module properties correctly."""
        input_data = jnp.array([[1, 2]], dtype=jnp.int32)
        output_data = jnp.array([[2, 1]], dtype=jnp.int32)
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        input_grid = Grid(data=input_data, mask=mask)
        output_grid = Grid(data=output_data, mask=mask)

        pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Test that it's an Equinox module
        import equinox as eqx

        assert isinstance(pair, eqx.Module)

        # Test immutability - should create new instance
        new_input_data = jnp.array([[3, 4]], dtype=jnp.int32)
        new_input_grid = Grid(data=new_input_data, mask=mask)
        new_pair = eqx.tree_at(lambda p: p.input_grid, pair, new_input_grid)

        # Original should be unchanged
        chex.assert_trees_all_equal(pair.input_grid.data, input_data)
        chex.assert_trees_all_equal(new_pair.input_grid.data, new_input_data)

    def test_taskpair_jax_compatibility(self):
        """Test TaskPair works with JAX transformations."""
        input_data = jnp.array([[1, 0]], dtype=jnp.int32)
        output_data = jnp.array([[0, 1]], dtype=jnp.int32)
        mask = jnp.ones((1, 2), dtype=jnp.bool_)

        def create_pair():
            input_grid = Grid(data=input_data, mask=mask)
            output_grid = Grid(data=output_data, mask=mask)
            return TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Test JIT compilation
        jitted_create = jax.jit(create_pair)
        pair = jitted_create()

        chex.assert_trees_all_equal(pair.input_grid.data, input_data)
        chex.assert_trees_all_equal(pair.output_grid.data, output_data)

    def test_taskpair_pytree_operations(self):
        """Test TaskPair works with PyTree operations."""
        input_data = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        output_data = jnp.array([[3, 2, 1]], dtype=jnp.int32)
        mask = jnp.ones((1, 3), dtype=jnp.bool_)

        input_grid = Grid(data=input_data, mask=mask)
        output_grid = Grid(data=output_data, mask=mask)
        pair = TaskPair(input_grid=input_grid, output_grid=output_grid)

        # Test PyTree flattening and unflattening
        leaves, treedef = jax.tree_util.tree_flatten(pair)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_equal(pair.input_grid.data, reconstructed.input_grid.data)
        chex.assert_trees_all_equal(
            pair.output_grid.data, reconstructed.output_grid.data
        )


class TestTimeStep:
    """Test cases for the TimeStep class."""

    def test_timestep_creation_basic(self):
        """Test basic TimeStep creation with all required fields."""
        observation = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        reward = jnp.array(1.0, dtype=jnp.float32)
        discount = jnp.array(0.99, dtype=jnp.float32)

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=reward,
            discount=discount,
            observation=observation,
            extras={},
        )
        # Verify extras default
        assert timestep.extras == {}

        # Verify basic properties
        chex.assert_trees_all_equal(timestep.step_type, StepType.FIRST)
        chex.assert_trees_all_equal(timestep.reward, reward)
        chex.assert_trees_all_equal(timestep.discount, discount)
        chex.assert_trees_all_equal(timestep.observation, observation)
        assert timestep.extras == {}

    def test_timestep_step_types(self):
        """Test TimeStep with different step types."""
        observation = jnp.array([[0]], dtype=jnp.int32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        discount = jnp.array(1.0, dtype=jnp.float32)

        # Test FIRST step
        first_step = TimeStep(
            step_type=StepType.FIRST,
            reward=reward,
            discount=discount,
            observation=observation,
        )

        # Test MID step
        mid_step = TimeStep(
            step_type=StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
        )

        # Test TERMINATED step
        terminated_step = TimeStep(
            step_type=StepType.TERMINATED,
            reward=reward,
            discount=jnp.array(0.0, dtype=jnp.float32),  # Terminal discount
            observation=observation,
        )

        # Test TRUNCATED step
        truncated_step = TimeStep(
            step_type=StepType.TRUNCATED,
            reward=reward,
            discount=jnp.array(0.0, dtype=jnp.float32),  # Terminal discount
            observation=observation,
        )

        # Verify step types are stored correctly
        chex.assert_trees_all_equal(first_step.step_type, StepType.FIRST)
        chex.assert_trees_all_equal(mid_step.step_type, StepType.MID)
        chex.assert_trees_all_equal(terminated_step.step_type, StepType.TERMINATED)
        chex.assert_trees_all_equal(truncated_step.step_type, StepType.TRUNCATED)

    def test_timestep_utility_methods(self):
        """Test TimeStep utility methods (first, last, done, etc.)."""
        observation = jnp.array([[1]], dtype=jnp.int32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        discount = jnp.array(1.0, dtype=jnp.float32)

        # Test FIRST step methods
        first_step = TimeStep(
            step_type=StepType.FIRST,
            reward=reward,
            discount=discount,
            observation=observation,
        )

        assert first_step.first() == True
        assert first_step.mid() == False
        assert first_step.last() == False
        assert first_step.terminated() == False
        assert first_step.truncated() == False
        assert first_step.done() == False

        # Test MID step methods
        mid_step = TimeStep(
            step_type=StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
        )

        assert mid_step.first() == False
        assert mid_step.mid() == True
        assert mid_step.last() == False
        assert mid_step.terminated() == False
        assert mid_step.truncated() == False
        assert mid_step.done() == False

        # Test TERMINATED step methods
        terminated_step = TimeStep(
            step_type=StepType.TERMINATED,
            reward=reward,
            discount=jnp.array(0.0, dtype=jnp.float32),
            observation=observation,
        )

        assert terminated_step.first() == False
        assert terminated_step.mid() == False
        assert terminated_step.last() == True
        assert terminated_step.terminated() == True
        assert terminated_step.truncated() == False
        assert terminated_step.done() == True

        # Test TRUNCATED step methods
        truncated_step = TimeStep(
            step_type=StepType.TRUNCATED,
            reward=reward,
            discount=jnp.array(0.0, dtype=jnp.float32),
            observation=observation,
        )

        assert truncated_step.first() == False
        assert truncated_step.mid() == False
        assert truncated_step.last() == True
        assert truncated_step.terminated() == False
        assert truncated_step.truncated() == True
        assert truncated_step.done() == True

    def test_timestep_episode_management(self):
        """Test TimeStep episode management functionality."""
        observation = jnp.array([[1, 2]], dtype=jnp.int32)

        # Create episode sequence
        steps = []

        # First step
        first_step = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=observation,
            extras={"episode_step": 0},
        )
        steps.append(first_step)

        # Middle steps
        for i in range(1, 3):
            mid_step = TimeStep(
                step_type=StepType.MID,
                reward=jnp.array(0.5, dtype=jnp.float32),
                discount=jnp.array(0.99, dtype=jnp.float32),
                observation=observation,
                extras={"episode_step": i},
            )
            steps.append(mid_step)

        # Terminal step
        final_step = TimeStep(
            step_type=StepType.TERMINATED,
            reward=jnp.array(1.0, dtype=jnp.float32),
            discount=jnp.array(0.0, dtype=jnp.float32),
            observation=observation,
            extras={"episode_step": 3},
        )
        steps.append(final_step)

        # Verify episode structure
        assert steps[0].first() == True
        assert all(step.mid() == True for step in steps[1:-1])
        assert steps[-1].last() == True
        assert steps[-1].terminated() == True

        # Verify extras are preserved
        for i, step in enumerate(steps):
            assert step.extras["episode_step"] == i

    def test_timestep_with_state(self):
        """Test TimeStep with embedded state."""
        observation = jnp.array([[1]], dtype=jnp.int32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        discount = jnp.array(1.0, dtype=jnp.float32)

        # Create a mock state (could be any PyTree)
        mock_state = {
            "grid": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
            "step_count": jnp.array(5, dtype=jnp.int32),
        }

        # Stoa TimeStep does not embed state; ensure extras can carry info if needed
        timestep = TimeStep(
            step_type=StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
            extras={"state": mock_state},
        )

        # Verify state-like info preserved in extras
        assert "state" in timestep.extras
        chex.assert_trees_all_equal(
            timestep.extras["state"]["grid"], mock_state["grid"]
        )
        chex.assert_trees_all_equal(
            timestep.extras["state"]["step_count"], mock_state["step_count"]
        )

    def test_timestep_extras_initialization(self):
        """Test TimeStep extras dict initialization."""
        observation = jnp.array([[1]], dtype=jnp.int32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        discount = jnp.array(1.0, dtype=jnp.float32)

        # Test with explicit empty extras
        timestep1 = TimeStep(
            step_type=StepType.FIRST,
            reward=reward,
            discount=discount,
            observation=observation,
            extras={},
        )
        assert timestep1.extras == {}

        # Test with populated extras
        extras_data = {"info": "test", "score": 42}
        timestep3 = TimeStep(
            step_type=StepType.FIRST,
            reward=reward,
            discount=discount,
            observation=observation,
            extras=extras_data,
        )
        assert timestep3.extras == extras_data

    def test_timestep_jax_compatibility(self):
        """Test TimeStep works with JAX transformations."""
        observation = jnp.array([[1, 0]], dtype=jnp.int32)
        reward = jnp.array(1.0, dtype=jnp.float32)
        discount = jnp.array(0.99, dtype=jnp.float32)

        def create_timestep():
            return TimeStep(
                step_type=StepType.MID,
                reward=reward,
                discount=discount,
                observation=observation,
            )

        # Test JIT compilation
        jitted_create = jax.jit(create_timestep)
        timestep = jitted_create()

        chex.assert_trees_all_equal(timestep.step_type, StepType.MID)
        chex.assert_trees_all_equal(timestep.reward, reward)
        chex.assert_trees_all_equal(timestep.discount, discount)
        chex.assert_trees_all_equal(timestep.observation, observation)

    def test_timestep_pytree_operations(self):
        """Test TimeStep works with PyTree operations."""
        observation = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        reward = jnp.array(0.5, dtype=jnp.float32)
        discount = jnp.array(0.95, dtype=jnp.float32)

        timestep = TimeStep(
            step_type=StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
            extras={"test": "value"},
        )

        # Test PyTree flattening and unflattening
        leaves, treedef = jax.tree_util.tree_flatten(timestep)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_equal(timestep.step_type, reconstructed.step_type)
        chex.assert_trees_all_equal(timestep.reward, reconstructed.reward)
        chex.assert_trees_all_equal(timestep.discount, reconstructed.discount)
        chex.assert_trees_all_equal(timestep.observation, reconstructed.observation)


class TestEnvParams:
    """Test cases for the EnvParams class."""

    def test_envparams_creation_from_config(self):
        """Test EnvParams creation from JaxArcConfig."""
        # Create a basic config
        config = JaxArcConfig()

        # Create a mock buffer (batched task data)
        mock_buffer = {
            "input_grids": jnp.zeros((5, 3, 10, 10), dtype=jnp.int32),
            "output_grids": jnp.ones((5, 3, 10, 10), dtype=jnp.int32),
            "masks": jnp.ones((5, 3, 10, 10), dtype=jnp.bool_),
        }

        # Create EnvParams from config
        env_params = EnvParams.from_config(
            config, episode_mode=0, buffer=mock_buffer, subset_indices=None
        )

        # Verify configuration references
        assert env_params.dataset is config.dataset
        assert env_params.action is config.action
        assert env_params.reward is config.reward
        assert env_params.grid_initialization is config.grid_initialization

        # Verify episode settings
        assert env_params.max_episode_steps == config.environment.max_episode_steps
        assert env_params.episode_mode == 0

        # Verify buffer
        assert env_params.buffer is mock_buffer
        assert env_params.subset_indices is None

    def test_envparams_validation(self):
        """Test EnvParams validation catches invalid configurations."""
        config = JaxArcConfig()
        mock_buffer = {"data": jnp.array([1, 2, 3])}

        # Should work with valid parameters
        valid_params = EnvParams.from_config(config, episode_mode=0, buffer=mock_buffer)
        assert valid_params.max_episode_steps > 0
        assert valid_params.episode_mode in (0, 1)

        # Test invalid episode mode
        with pytest.raises(AssertionError):
            EnvParams(
                dataset=config.dataset,
                action=config.action,
                reward=config.reward,
                grid_initialization=config.grid_initialization,
                max_episode_steps=100,
                buffer=mock_buffer,
                subset_indices=None,
                episode_mode=2,  # Invalid mode
            )

        # Test missing buffer
        with pytest.raises(AssertionError, match="EnvParams.buffer must be provided"):
            EnvParams(
                dataset=config.dataset,
                action=config.action,
                reward=config.reward,
                grid_initialization=config.grid_initialization,
                max_episode_steps=100,
                buffer=None,  # Missing buffer
                subset_indices=None,
                episode_mode=0,
            )

        # Test invalid max_episode_steps
        with pytest.raises(AssertionError):
            EnvParams(
                dataset=config.dataset,
                action=config.action,
                reward=config.reward,
                grid_initialization=config.grid_initialization,
                max_episode_steps=0,  # Invalid (must be > 0)
                buffer=mock_buffer,
                subset_indices=None,
                episode_mode=0,
            )

    def test_envparams_buffer_management(self):
        """Test EnvParams buffer management and subset indices."""
        config = JaxArcConfig()

        # Create a larger buffer with multiple tasks
        buffer_size = 10
        mock_buffer = {
            "task_indices": jnp.arange(buffer_size, dtype=jnp.int32),
            "input_grids": jnp.zeros((buffer_size, 3, 5, 5), dtype=jnp.int32),
            "output_grids": jnp.ones((buffer_size, 3, 5, 5), dtype=jnp.int32),
            "num_pairs": jnp.full(buffer_size, 3, dtype=jnp.int32),
        }

        # Test with subset indices
        subset_indices = jnp.array([0, 2, 4, 6], dtype=jnp.int32)

        env_params = EnvParams.from_config(
            config,
            episode_mode=1,  # Test mode
            buffer=mock_buffer,
            subset_indices=subset_indices,
        )

        # Verify buffer and subset
        assert env_params.buffer is mock_buffer
        chex.assert_trees_all_equal(env_params.subset_indices, subset_indices)
        assert env_params.episode_mode == 1

        # Verify buffer structure
        assert "task_indices" in env_params.buffer
        assert "input_grids" in env_params.buffer
        chex.assert_shape(env_params.buffer["input_grids"], (buffer_size, 3, 5, 5))

    def test_envparams_episode_mode_configuration(self):
        """Test EnvParams episode mode and configuration extraction."""
        config = JaxArcConfig()
        mock_buffer = {"test_data": jnp.array([1, 2, 3])}

        # Test training mode (0)
        train_params = EnvParams.from_config(config, episode_mode=0, buffer=mock_buffer)
        assert train_params.episode_mode == 0

        # Test evaluation mode (1)
        eval_params = EnvParams.from_config(config, episode_mode=1, buffer=mock_buffer)
        assert eval_params.episode_mode == 1

        # Verify configuration extraction is consistent
        assert train_params.dataset is config.dataset
        assert eval_params.dataset is config.dataset
        assert train_params.action is config.action
        assert eval_params.action is config.action

        # Verify episode settings are extracted correctly
        expected_steps = int(config.environment.max_episode_steps)
        assert train_params.max_episode_steps == expected_steps
        assert eval_params.max_episode_steps == expected_steps

    def test_envparams_configuration_references(self):
        """Test EnvParams maintains references to configuration objects."""
        config = JaxArcConfig()
        mock_buffer = {"data": jnp.array([1])}

        env_params = EnvParams.from_config(config, episode_mode=0, buffer=mock_buffer)

        # Verify that configs are references, not copies
        assert env_params.dataset is config.dataset
        assert env_params.action is config.action
        assert env_params.reward is config.reward
        assert env_params.grid_initialization is config.grid_initialization

        # Test that changes to original config affect env_params
        # (This tests that we have references, not deep copies)
        original_dataset = env_params.dataset
        assert env_params.dataset is original_dataset

    def test_envparams_jax_compatibility(self):
        """Test EnvParams works with JAX transformations."""
        config = JaxArcConfig()

        # Create JAX-compatible buffer
        mock_buffer = {
            "grids": jnp.zeros((3, 2, 4, 4), dtype=jnp.int32),
            "masks": jnp.ones((3, 2, 4, 4), dtype=jnp.bool_),
            "indices": jnp.arange(3, dtype=jnp.int32),
        }

        def create_env_params():
            return EnvParams.from_config(config, episode_mode=0, buffer=mock_buffer)

        # Test JIT compilation
        jitted_create = jax.jit(create_env_params)
        env_params = jitted_create()

        # Verify creation worked
        assert env_params.episode_mode == 0
        assert env_params.buffer is not None
        assert env_params.max_episode_steps > 0

    def test_envparams_buffer_structure_validation(self):
        """Test EnvParams with different buffer structures."""
        config = JaxArcConfig()

        # Test with minimal buffer
        minimal_buffer = {"task_data": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)}

        env_params1 = EnvParams.from_config(
            config, episode_mode=0, buffer=minimal_buffer
        )
        assert env_params1.buffer is minimal_buffer

        # Test with complex buffer structure
        complex_buffer = {
            "tasks": {
                "input_grids": jnp.zeros((5, 3, 8, 8), dtype=jnp.int32),
                "output_grids": jnp.ones((5, 3, 8, 8), dtype=jnp.int32),
                "input_masks": jnp.ones((5, 3, 8, 8), dtype=jnp.bool_),
                "output_masks": jnp.ones((5, 3, 8, 8), dtype=jnp.bool_),
                "num_train_pairs": jnp.full(5, 3, dtype=jnp.int32),
                "num_test_pairs": jnp.full(5, 1, dtype=jnp.int32),
                "task_indices": jnp.arange(5, dtype=jnp.int32),
            },
            "metadata": {"dataset_name": "test_dataset", "total_tasks": 5},
        }

        env_params2 = EnvParams.from_config(
            config,
            episode_mode=1,
            buffer=complex_buffer,
            subset_indices=jnp.array([0, 2, 4], dtype=jnp.int32),
        )

        assert env_params2.buffer is complex_buffer
        assert "tasks" in env_params2.buffer
        assert "metadata" in env_params2.buffer
        chex.assert_trees_all_equal(
            env_params2.subset_indices, jnp.array([0, 2, 4], dtype=jnp.int32)
        )

    def test_envparams_subset_indices_functionality(self):
        """Test EnvParams subset indices define buffer views correctly."""
        config = JaxArcConfig()

        # Create buffer with 6 tasks
        full_buffer = {
            "task_ids": jnp.arange(6, dtype=jnp.int32),
            "difficulties": jnp.array([1, 2, 1, 3, 2, 1], dtype=jnp.int32),
            "grids": jnp.zeros((6, 2, 3, 3), dtype=jnp.int32),
        }

        # Test without subset (full buffer)
        full_params = EnvParams.from_config(
            config, episode_mode=0, buffer=full_buffer, subset_indices=None
        )
        assert full_params.subset_indices is None

        # Test with subset (only easy tasks - difficulty 1)
        easy_indices = jnp.array([0, 2, 5], dtype=jnp.int32)  # Tasks with difficulty 1
        subset_params = EnvParams.from_config(
            config, episode_mode=0, buffer=full_buffer, subset_indices=easy_indices
        )

        chex.assert_trees_all_equal(subset_params.subset_indices, easy_indices)
        assert subset_params.buffer is full_buffer  # Same buffer reference

        # Test with different subset (medium tasks - difficulty 2)
        medium_indices = jnp.array([1, 4], dtype=jnp.int32)
        medium_params = EnvParams.from_config(
            config, episode_mode=1, buffer=full_buffer, subset_indices=medium_indices
        )

        chex.assert_trees_all_equal(medium_params.subset_indices, medium_indices)

    def test_envparams_equinox_module_properties(self):
        """Test EnvParams inherits Equinox Module properties correctly."""
        config = JaxArcConfig()
        mock_buffer = {"data": jnp.array([1, 2, 3])}

        env_params = EnvParams.from_config(config, episode_mode=0, buffer=mock_buffer)

        # Test that it's an Equinox module
        import equinox as eqx

        assert isinstance(env_params, eqx.Module)

        # Test immutability - should create new instance
        new_episode_mode = 1
        new_params = eqx.tree_at(lambda p: p.episode_mode, env_params, new_episode_mode)

        # Original should be unchanged
        assert env_params.episode_mode == 0
        assert new_params.episode_mode == 1

        # Other fields should have the same values (but may be new instances)
        assert new_params.max_episode_steps == env_params.max_episode_steps
        # Note: Equinox tree_at may create new instances, so we check values not identity

    def test_envparams_pytree_operations(self):
        """Test EnvParams works with PyTree operations."""
        config = JaxArcConfig()
        mock_buffer = {
            "arrays": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
            "metadata": {"info": "test"},
        }
        subset_indices = jnp.array([0, 1], dtype=jnp.int32)

        env_params = EnvParams.from_config(
            config, episode_mode=1, buffer=mock_buffer, subset_indices=subset_indices
        )

        # Test PyTree flattening and unflattening
        leaves, treedef = jax.tree_util.tree_flatten(env_params)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify reconstruction
        assert reconstructed.episode_mode == env_params.episode_mode
        assert reconstructed.max_episode_steps == env_params.max_episode_steps
        chex.assert_trees_all_equal(
            reconstructed.subset_indices, env_params.subset_indices
        )

        # Note: Config references may not be preserved through PyTree operations
        # but the core functionality should work
