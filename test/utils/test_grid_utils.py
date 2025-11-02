"""Tests for grid utility functions."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.utils.grid_utils import (
    compute_grid_similarity,
    extract_bounding_box_region,
    extract_object_rectangle,
    get_actual_grid_shape_from_mask,
    get_selection_bounding_box,
    pad_array_sequence,
    pad_grid_to_size,
    validate_single_cell_selection,
)


class TestPaddingFunctions:
    """Test grid padding utilities."""

    def test_pad_grid_to_size_basic(self):
        """Test basic grid padding functionality."""
        grid = jnp.array([[1, 2], [3, 4]])
        padded_grid, mask = pad_grid_to_size(grid, 4, 3, fill_value=0)

        expected_grid = jnp.array([[1, 2, 0], [3, 4, 0], [0, 0, 0], [0, 0, 0]])
        expected_mask = jnp.array(
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
                [False, False, False],
            ]
        )

        chex.assert_trees_all_equal(padded_grid, expected_grid)
        chex.assert_trees_all_equal(mask, expected_mask)

    def test_pad_grid_to_size_no_padding_needed(self):
        """Test padding when grid already matches target size."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]])
        padded_grid, mask = pad_grid_to_size(grid, 2, 3, fill_value=0)

        chex.assert_trees_all_equal(padded_grid, grid)
        chex.assert_trees_all_equal(mask, jnp.ones_like(grid, dtype=jnp.bool_))

    def test_pad_grid_to_size_custom_fill_value(self):
        """Test padding with custom fill value."""
        grid = jnp.array([[1, 2]])
        padded_grid, mask = pad_grid_to_size(grid, 2, 3, fill_value=9)

        expected_grid = jnp.array([[1, 2, 9], [9, 9, 9]])
        expected_mask = jnp.array([[True, True, False], [False, False, False]])

        chex.assert_trees_all_equal(padded_grid, expected_grid)
        chex.assert_trees_all_equal(mask, expected_mask)

    def test_pad_grid_to_size_exceeds_target(self):
        """Test error when grid exceeds target dimensions."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(
            ValueError, match="Grid dimensions.*exceed target dimensions"
        ):
            pad_grid_to_size(grid, 1, 2)

    def test_pad_array_sequence_basic(self):
        """Test basic array sequence padding."""
        arrays = [jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6]])]

        padded_arrays, masks = pad_array_sequence(arrays, 3, 3, 3, fill_value=0)

        # Check shape
        chex.assert_shape(padded_arrays, (3, 3, 3))
        chex.assert_shape(masks, (3, 3, 3))

        # Check first array
        expected_first = jnp.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
        chex.assert_trees_all_equal(padded_arrays[0], expected_first)

        # Check second array
        expected_second = jnp.array([[5, 6, 0], [0, 0, 0], [0, 0, 0]])
        chex.assert_trees_all_equal(padded_arrays[1], expected_second)

        # Check third array (empty)
        expected_third = jnp.zeros((3, 3), dtype=jnp.int32)
        chex.assert_trees_all_equal(padded_arrays[2], expected_third)

    def test_pad_array_sequence_exceeds_length(self):
        """Test error when array count exceeds target length."""
        arrays = [jnp.array([[1]]), jnp.array([[2]]), jnp.array([[3]])]

        with pytest.raises(ValueError, match="Number of arrays.*exceeds target length"):
            pad_array_sequence(arrays, 2, 2, 2)

    def test_pad_array_sequence_empty_list(self):
        """Test padding empty array list."""
        arrays = []
        padded_arrays, masks = pad_array_sequence(arrays, 2, 2, 2, fill_value=7)

        expected_array = jnp.full((2, 2), 7, dtype=jnp.int32)
        expected_mask = jnp.zeros((2, 2), dtype=jnp.bool_)

        chex.assert_trees_all_equal(padded_arrays[0], expected_array)
        chex.assert_trees_all_equal(padded_arrays[1], expected_array)
        chex.assert_trees_all_equal(masks[0], expected_mask)
        chex.assert_trees_all_equal(masks[1], expected_mask)


class TestBoundingBoxUtilities:
    """Test bounding box calculation utilities."""

    def test_get_selection_bounding_box_basic(self):
        """Test basic bounding box calculation."""
        selection = jnp.array(
            [[False, True, False], [True, True, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 0
        assert max_row == 1
        assert min_col == 0
        assert max_col == 1

    def test_get_selection_bounding_box_single_cell(self):
        """Test bounding box for single selected cell."""
        selection = jnp.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 1
        assert max_row == 1
        assert min_col == 1
        assert max_col == 1

    def test_get_selection_bounding_box_empty(self):
        """Test bounding box for empty selection."""
        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == -1
        assert max_row == -1
        assert min_col == -1
        assert max_col == -1

    def test_get_selection_bounding_box_full_grid(self):
        """Test bounding box when entire grid is selected."""
        selection = jnp.ones((3, 4), dtype=jnp.bool_)

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 0
        assert max_row == 2
        assert min_col == 0
        assert max_col == 3

    def test_get_selection_bounding_box_jit_compatible(self):
        """Test that bounding box function is JIT compatible."""
        selection = jnp.array(
            [[False, True, False], [True, True, False], [False, False, False]]
        )

        jitted_fn = jax.jit(get_selection_bounding_box)
        result = jitted_fn(selection)
        expected = get_selection_bounding_box(selection)

        chex.assert_trees_all_equal(result, expected)


class TestRectangleExtraction:
    """Test rectangle extraction utilities."""

    def test_extract_object_rectangle_basic(self):
        """Test basic object rectangle extraction."""
        grid = jnp.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]])
        selection = jnp.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, False, False],
            ]
        )

        masked_grid, bbox = extract_object_rectangle(grid, selection)

        # Check bounding box coordinates
        min_row, max_row, min_col, max_col = bbox
        assert min_row == 0
        assert max_row == 1
        assert min_col == 0
        assert max_col == 2

        # Check that only bounding box region is preserved
        expected_masked = jnp.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]])
        # Only the bounding box region should be preserved
        expected_masked = jnp.where(
            (jnp.arange(3)[:, None] >= 0)
            & (jnp.arange(3)[:, None] <= 1)
            & (jnp.arange(4)[None, :] >= 0)
            & (jnp.arange(4)[None, :] <= 2),
            expected_masked,
            0,
        )

        chex.assert_trees_all_equal(masked_grid, expected_masked)

    def test_extract_object_rectangle_empty_selection(self):
        """Test extraction with empty selection."""
        grid = jnp.array([[1, 2], [3, 4]])
        selection = jnp.zeros_like(grid, dtype=jnp.bool_)

        masked_grid, bbox = extract_object_rectangle(grid, selection)

        # Should return all zeros and invalid bbox
        chex.assert_trees_all_equal(masked_grid, jnp.zeros_like(grid))
        assert bbox == (-1, -1, -1, -1)

    def test_extract_object_rectangle_jit_compatible(self):
        """Test that rectangle extraction is JIT compatible."""
        grid = jnp.array([[1, 2], [3, 4]])
        selection = jnp.array([[True, False], [False, True]])

        jitted_fn = jax.jit(extract_object_rectangle)
        result = jitted_fn(grid, selection)
        expected = extract_object_rectangle(grid, selection)

        chex.assert_trees_all_equal(result, expected)

    def test_extract_bounding_box_region_basic(self):
        """Test basic bounding box region extraction."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        extracted = extract_bounding_box_region(grid, 0, 1, 0, 1)

        expected = jnp.array([[1, 2, 0], [4, 5, 0], [0, 0, 0]])

        chex.assert_trees_all_equal(extracted, expected)

    def test_extract_bounding_box_region_invalid_coords(self):
        """Test extraction with invalid coordinates."""
        grid = jnp.array([[1, 2], [3, 4]])

        extracted = extract_bounding_box_region(grid, -1, -1, -1, -1)

        chex.assert_trees_all_equal(extracted, jnp.zeros_like(grid))

    def test_extract_bounding_box_region_jit_compatible(self):
        """Test that bounding box extraction is JIT compatible."""
        grid = jnp.array([[1, 2], [3, 4]])

        jitted_fn = jax.jit(extract_bounding_box_region)
        result = jitted_fn(grid, 0, 0, 0, 1)
        expected = extract_bounding_box_region(grid, 0, 0, 0, 1)

        chex.assert_trees_all_equal(result, expected)


class TestSelectionValidation:
    """Test selection validation utilities."""

    def test_validate_single_cell_selection_valid(self):
        """Test validation of valid single cell selection."""
        selection = jnp.array(
            [[False, True, False], [False, False, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == True

    def test_validate_single_cell_selection_multiple(self):
        """Test validation fails for multiple cell selection."""
        selection = jnp.array(
            [[True, True, False], [False, False, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == False

    def test_validate_single_cell_selection_empty(self):
        """Test validation fails for empty selection."""
        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == False

    def test_validate_single_cell_selection_jit_compatible(self):
        """Test that selection validation is JIT compatible."""
        selection = jnp.array([[False, True, False]])

        jitted_fn = jax.jit(validate_single_cell_selection)
        result = jitted_fn(selection)
        expected = validate_single_cell_selection(selection)

        chex.assert_trees_all_equal(result, expected)


class TestShapeAndMaskUtilities:
    """Test shape and mask utility functions."""

    def test_get_actual_grid_shape_from_mask_basic(self):
        """Test getting actual grid shape from mask."""
        mask = jnp.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
            ]
        )

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 2
        assert width == 2

    def test_get_actual_grid_shape_from_mask_full(self):
        """Test shape calculation for full mask."""
        mask = jnp.ones((3, 4), dtype=jnp.bool_)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 3
        assert width == 4

    def test_get_actual_grid_shape_from_mask_empty(self):
        """Test shape calculation for empty mask."""
        mask = jnp.zeros((3, 4), dtype=jnp.bool_)

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 0
        assert width == 0

    def test_get_actual_grid_shape_from_mask_single_cell(self):
        """Test shape calculation for single cell mask."""
        mask = jnp.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        height, width = get_actual_grid_shape_from_mask(mask)

        assert height == 2  # Row index 1 + 1
        assert width == 2  # Col index 1 + 1

    def test_compute_grid_similarity_perfect_match(self):
        """Test similarity computation for perfect match."""
        grid = jnp.array([[1, 2], [3, 4]])
        mask = jnp.ones_like(grid, dtype=jnp.bool_)

        similarity = compute_grid_similarity(grid, mask, grid, mask)

        chex.assert_trees_all_close(similarity, 1.0)

    def test_compute_grid_similarity_no_match(self):
        """Test similarity computation for no match."""
        working_grid = jnp.array([[1, 2], [3, 4]])
        target_grid = jnp.array([[5, 6], [7, 8]])
        mask = jnp.ones_like(working_grid, dtype=jnp.bool_)

        similarity = compute_grid_similarity(working_grid, mask, target_grid, mask)

        chex.assert_trees_all_close(similarity, 0.0)

    def test_compute_grid_similarity_partial_match(self):
        """Test similarity computation for partial match."""
        working_grid = jnp.array([[1, 2], [3, 4]])
        target_grid = jnp.array([[1, 6], [3, 8]])
        mask = jnp.ones_like(working_grid, dtype=jnp.bool_)

        similarity = compute_grid_similarity(working_grid, mask, target_grid, mask)

        # 2 out of 4 cells match
        chex.assert_trees_all_close(similarity, 0.5)

    def test_compute_grid_similarity_empty_target(self):
        """Test similarity with empty target."""
        working_grid = jnp.array([[1, 2], [3, 4]])
        working_mask = jnp.ones_like(working_grid, dtype=jnp.bool_)
        target_grid = jnp.array([[0, 0], [0, 0]])
        target_mask = jnp.zeros_like(target_grid, dtype=jnp.bool_)

        similarity = compute_grid_similarity(
            working_grid, working_mask, target_grid, target_mask
        )

        # Working has content but target is empty, so similarity should be 0
        chex.assert_trees_all_close(similarity, 0.0)

    def test_compute_grid_similarity_both_empty(self):
        """Test similarity when both grids are empty."""
        grid = jnp.array([[0, 0], [0, 0]])
        mask = jnp.zeros_like(grid, dtype=jnp.bool_)

        similarity = compute_grid_similarity(grid, mask, grid, mask)

        # Both empty, so perfect match
        chex.assert_trees_all_close(similarity, 1.0)

    def test_compute_grid_similarity_jit_compatible(self):
        """Test that similarity computation is JIT compatible."""
        grid = jnp.array([[1, 2], [3, 4]])
        mask = jnp.ones_like(grid, dtype=jnp.bool_)

        jitted_fn = jax.jit(compute_grid_similarity)
        result = jitted_fn(grid, mask, grid, mask)
        expected = compute_grid_similarity(grid, mask, grid, mask)

        chex.assert_trees_all_close(result, expected)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_grid(self):
        """Test utilities with single pixel grids."""
        grid = jnp.array([[5]])
        mask = jnp.array([[True]])

        # Test padding
        padded_grid, padded_mask = pad_grid_to_size(grid, 2, 2)
        expected_padded = jnp.array([[5, 0], [0, 0]])
        expected_mask = jnp.array([[True, False], [False, False]])
        chex.assert_trees_all_equal(padded_grid, expected_padded)
        chex.assert_trees_all_equal(padded_mask, expected_mask)

        # Test bounding box
        bbox = get_selection_bounding_box(mask)
        assert bbox == (0, 0, 0, 0)

        # Test shape calculation
        height, width = get_actual_grid_shape_from_mask(mask)
        assert height == 1
        assert width == 1

    def test_large_grid_operations(self):
        """Test utilities with larger grids."""
        # Create a 10x10 grid
        grid = jnp.arange(100).reshape(10, 10)
        mask = jnp.ones_like(grid, dtype=jnp.bool_)

        # Test that operations work with larger grids
        height, width = get_actual_grid_shape_from_mask(mask)
        assert height == 10
        assert width == 10

        # Test bounding box with full selection
        bbox = get_selection_bounding_box(mask)
        assert bbox == (0, 9, 0, 9)

    def test_irregular_selection_patterns(self):
        """Test with irregular selection patterns."""
        # L-shaped selection
        selection = jnp.array(
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        )

        bbox = get_selection_bounding_box(selection)
        assert bbox == (0, 2, 0, 2)

        # Diagonal selection
        selection = jnp.array(
            [[True, False, False], [False, True, False], [False, False, True]]
        )

        bbox = get_selection_bounding_box(selection)
        assert bbox == (0, 2, 0, 2)

    def test_dtype_preservation(self):
        """Test that data types are preserved correctly."""
        # Test with different dtypes
        grid_int32 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid_float32 = jnp.array([[1.5, 2.5], [3.5, 4.5]], dtype=jnp.float32)

        padded_int, _ = pad_grid_to_size(grid_int32, 3, 3)
        padded_float, _ = pad_grid_to_size(grid_float32, 3, 3)

        assert padded_int.dtype == jnp.int32
        assert padded_float.dtype == jnp.float32

    def test_memory_efficiency_patterns(self):
        """Test memory-efficient patterns with static shapes."""
        # Test that functions maintain static shapes for JIT efficiency
        grid = jnp.array([[1, 2, 3], [4, 5, 6]])
        selection = jnp.array([[True, False, True], [False, True, False]])

        # Extract rectangle should maintain original grid shape
        masked_grid, _ = extract_object_rectangle(grid, selection)
        chex.assert_shape(masked_grid, grid.shape)

        # Bounding box extraction should maintain shape
        extracted = extract_bounding_box_region(grid, 0, 1, 0, 2)
        chex.assert_shape(extracted, grid.shape)
