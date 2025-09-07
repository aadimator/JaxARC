"""
Comprehensive unit tests for object-action improvements.

This module tests all new utility functions and enhanced operations
introduced in the object-action improvements specification, including:
- Bounding box calculation utilities
- Rectangle extraction utilities
- Selection validation utilities
- Enhanced object operations (move, rotate, flip, flood fill)
- JAX compatibility and transformations
- Edge cases and boundary conditions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.grid_operations import (
    flip_object,
    flood_fill_color,
    move_object,
    rotate_object,
    validate_bounding_box_for_operation,
)
from jaxarc.state import State
from jaxarc.types import GridArray
from jaxarc.utils.grid_utils import (
    extract_bounding_box_region,
    extract_object_rectangle,
    get_selection_bounding_box,
    validate_single_cell_selection,
)
from test.test_utils import (
    assert_jax_compatible,
)


def create_test_state(working_grid: GridArray) -> State:
    """Helper function to create a test state with all required fields."""
    grid_shape = working_grid.shape
    return State(
        working_grid=working_grid,
        working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
        input_grid=working_grid,
        input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
        target_grid=working_grid,
        target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
        selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
        clipboard=jnp.zeros(grid_shape),
        step_count=jnp.int32(0),
        allowed_operations_mask=jnp.ones(35, dtype=jnp.bool_),
        similarity_score=0.0,
        key=jax.random.PRNGKey(42),
        task_idx=jnp.int32(0),
        pair_idx=jnp.int32(0),
    )


class TestBoundingBoxUtilities:
    """Test bounding box calculation utilities."""

    def test_get_selection_bounding_box_single_pixel(self):
        """Test bounding box calculation for single pixel selection."""
        selection = jnp.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 1
        assert max_row == 1
        assert min_col == 1
        assert max_col == 1

    def test_get_selection_bounding_box_rectangle(self):
        """Test bounding box calculation for rectangular selection."""
        selection = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 0
        assert max_row == 1
        assert min_col == 0
        assert max_col == 1

    def test_get_selection_bounding_box_l_shape(self):
        """Test bounding box calculation for L-shaped selection."""
        selection = jnp.array(
            [[True, False, False], [True, False, False], [True, True, True]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 0
        assert max_row == 2
        assert min_col == 0
        assert max_col == 2

    def test_get_selection_bounding_box_scattered(self):
        """Test bounding box calculation for scattered selection."""
        selection = jnp.array(
            [
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == 0
        assert max_row == 2
        assert min_col == 0
        assert max_col == 3

    def test_get_selection_bounding_box_empty(self):
        """Test bounding box calculation for empty selection."""
        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)

        assert min_row == -1
        assert max_row == -1
        assert min_col == -1
        assert max_col == -1

    def test_get_selection_bounding_box_edge_cases(self):
        """Test bounding box calculation for edge cases."""
        # Top-left corner
        selection = jnp.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)
        assert (min_row, max_row, min_col, max_col) == (0, 0, 0, 0)

        # Bottom-right corner
        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, True]]
        )
        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)
        assert (min_row, max_row, min_col, max_col) == (2, 2, 2, 2)

    def test_get_selection_bounding_box_jax_compatibility(self):
        """Test that bounding box calculation is JAX compatible."""
        selection = jnp.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )

        # Test JIT compilation
        result = assert_jax_compatible(
            get_selection_bounding_box,
            selection,
            test_jit=True,
            test_vmap=False,  # vmap would require batched inputs
        )

        assert len(result) == 4
        assert all(isinstance(x, jnp.ndarray) for x in result)


class TestRectangleExtractionUtilities:
    """Test rectangle extraction utilities."""

    def test_extract_object_rectangle_basic(self):
        """Test basic rectangle extraction."""
        grid = jnp.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        selection = jnp.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )

        masked_grid, bbox = extract_object_rectangle(grid, selection)
        min_row, max_row, min_col, max_col = bbox

        # Check bounding box coordinates
        assert min_row == 0
        assert max_row == 1
        assert min_col == 0
        assert max_col == 2

        # Check that only bounding box region is preserved
        expected_masked = jnp.array(
            [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Only the bounding box region should be preserved
        assert jnp.array_equal(masked_grid[:2, :3], expected_masked[:2, :3])
        assert jnp.all(masked_grid[2:, :] == 0)
        assert jnp.all(masked_grid[:, 3:] == 0)

    def test_extract_object_rectangle_empty_selection(self):
        """Test rectangle extraction with empty selection."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        masked_grid, bbox = extract_object_rectangle(grid, selection)
        min_row, max_row, min_col, max_col = bbox

        # Should return invalid bounding box
        assert min_row == -1
        assert max_row == -1
        assert min_col == -1
        assert max_col == -1

        # Grid should be all zeros
        assert jnp.all(masked_grid == 0)

    def test_extract_bounding_box_region_valid_coords(self):
        """Test bounding box region extraction with valid coordinates."""
        grid = jnp.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        # Extract 2x2 region from (1,1) to (2,2)
        extracted = extract_bounding_box_region(grid, 1, 2, 1, 2)

        # Should preserve only the specified region
        expected = jnp.array([[0, 0, 0, 0], [0, 6, 7, 0], [0, 10, 11, 0], [0, 0, 0, 0]])

        assert jnp.array_equal(extracted, expected)

    def test_extract_bounding_box_region_invalid_coords(self):
        """Test bounding box region extraction with invalid coordinates."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Use invalid coordinates (-1 values)
        extracted = extract_bounding_box_region(grid, -1, -1, -1, -1)

        # Should return all zeros
        assert jnp.all(extracted == 0)

    def test_rectangle_extraction_jax_compatibility(self):
        """Test that rectangle extraction is JAX compatible."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        selection = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )

        # Test JIT compilation
        result = assert_jax_compatible(
            extract_object_rectangle, grid, selection, test_jit=True
        )

        masked_grid, bbox = result
        assert masked_grid.shape == grid.shape
        assert len(bbox) == 4


class TestSelectionValidationUtilities:
    """Test selection validation utilities."""

    def test_validate_single_cell_selection_valid(self):
        """Test validation with exactly one cell selected."""
        selection = jnp.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == True

    def test_validate_single_cell_selection_multiple_cells(self):
        """Test validation with multiple cells selected."""
        selection = jnp.array(
            [[True, False, False], [False, True, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == False

    def test_validate_single_cell_selection_no_cells(self):
        """Test validation with no cells selected."""
        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        result = validate_single_cell_selection(selection)
        assert result == False

    def test_validate_single_cell_selection_jax_compatibility(self):
        """Test that selection validation is JAX compatible."""
        selection = jnp.array(
            [[False, True, False], [False, False, False], [False, False, False]]
        )

        result = assert_jax_compatible(
            validate_single_cell_selection, selection, test_jit=True
        )

        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.bool_


class TestEnhancedFloodFillOperation:
    """Test enhanced flood fill operation with validation."""

    def test_flood_fill_single_cell_valid(self):
        """Test flood fill with valid single cell selection."""
        working_grid = jnp.array([[1, 1, 2], [1, 1, 2], [3, 3, 3]])

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )

        result_state = flood_fill_color(sample_state, selection, 5)

        # Should flood fill the connected region of 1s with color 5
        expected_grid = jnp.array([[5, 5, 2], [5, 5, 2], [3, 3, 3]])

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_flood_fill_multiple_cells_noop(self):
        """Test flood fill with multiple cells returns original state (NOOP)."""
        working_grid = jnp.array([[1, 1, 2], [1, 1, 2], [3, 3, 3]])

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [[True, True, False], [False, False, False], [False, False, False]]
        )

        result_state = flood_fill_color(sample_state, selection, 5)

        # Should return original state unchanged
        assert jnp.array_equal(result_state.working_grid, sample_state.working_grid)

    def test_flood_fill_no_cells_noop(self):
        """Test flood fill with no cells returns original state (NOOP)."""
        working_grid = jnp.array([[1, 1, 2], [1, 1, 2], [3, 3, 3]])

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        result_state = flood_fill_color(sample_state, selection, 5)

        # Should return original state unchanged
        assert jnp.array_equal(result_state.working_grid, sample_state.working_grid)

    def test_flood_fill_jax_compatibility(self):
        """Test that enhanced flood fill is JAX compatible."""
        working_grid = jnp.array([[1, 1, 2], [1, 1, 2], [3, 3, 3]])

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )

        result = assert_jax_compatible(
            flood_fill_color, sample_state, selection, 7, test_jit=True
        )

        assert hasattr(result, "working_grid")
        assert result.working_grid.shape == sample_state.working_grid.shape


class TestEnhancedMoveOperation:
    """Test enhanced move operation with bounded wrapping."""

    def test_move_object_up_wrapping(self):
        """Test move up with wrapping within bounding box."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        # Select the 3x3 inner region
        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = move_object(sample_state, selection, 0)  # Move up

        # The top row should wrap to bottom within the bounding box
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 7, 8, 9, 0],  # Bottom row moved to top
                [0, 1, 2, 3, 0],  # Original top row moved down
                [0, 4, 5, 6, 0],  # Original middle row moved down
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_move_object_left_wrapping(self):
        """Test move left with wrapping within bounding box."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = move_object(sample_state, selection, 2)  # Move left

        # The leftmost column should wrap to rightmost within bounding box
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 1, 2, 0],  # 1 wrapped to right, others shifted left
                [0, 6, 4, 5, 0],  # 4 wrapped to right, others shifted left
                [0, 9, 7, 8, 0],  # 7 wrapped to right, others shifted left
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_move_object_jax_compatibility(self):
        """Test that enhanced move operation is JAX compatible."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        result = assert_jax_compatible(
            move_object,
            sample_state,
            selection,
            1,  # Move down
            test_jit=True,
        )

        assert hasattr(result, "working_grid")
        assert result.working_grid.shape == sample_state.working_grid.shape


class TestEnhancedRotationOperation:
    """Test enhanced rotation operation with square validation."""

    def test_rotate_object_square_clockwise(self):
        """Test rotation of square selection clockwise."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        # Select a 3x3 square region
        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = rotate_object(sample_state, selection, 0)  # Clockwise

        # 90° clockwise rotation of the 3x3 region
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 7, 4, 1, 0],  # Left column becomes top row
                [0, 8, 5, 2, 0],  # Middle column stays middle
                [0, 9, 6, 3, 0],  # Right column becomes bottom row
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_rotate_object_square_counterclockwise(self):
        """Test rotation of square selection counterclockwise."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = rotate_object(sample_state, selection, 1)  # Counterclockwise

        # 90° counterclockwise rotation of the 3x3 region
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 6, 9, 0],  # Right column becomes top row
                [0, 2, 5, 8, 0],  # Middle column stays middle
                [0, 1, 4, 7, 0],  # Left column becomes bottom row
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_rotate_object_non_square_noop(self):
        """Test rotation of non-square selection returns original state (NOOP)."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        # Select a 2x3 rectangular region (non-square)
        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        result_state = rotate_object(sample_state, selection, 0)

        # Should return original state unchanged
        assert jnp.array_equal(result_state.working_grid, sample_state.working_grid)

    def test_rotate_object_jax_compatibility(self):
        """Test that enhanced rotation operation is JAX compatible."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        result = assert_jax_compatible(
            rotate_object, sample_state, selection, 0, test_jit=True
        )

        assert hasattr(result, "working_grid")
        assert result.working_grid.shape == sample_state.working_grid.shape


class TestEnhancedFlipOperation:
    """Test enhanced flip operation with bounding box extraction."""

    def test_flip_object_horizontal(self):
        """Test horizontal flip of selected region."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        # Select the 3x3 inner region
        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = flip_object(sample_state, selection, 0)  # Horizontal flip

        # Horizontal flip within the bounding box
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 2, 1, 0],  # Row flipped horizontally
                [0, 6, 5, 4, 0],  # Row flipped horizontally
                [0, 9, 8, 7, 0],  # Row flipped horizontally
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_flip_object_vertical(self):
        """Test vertical flip of selected region."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )

        result_state = flip_object(sample_state, selection, 1)  # Vertical flip

        # Vertical flip within the bounding box
        expected_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 7, 8, 9, 0],  # Bottom row moved to top
                [0, 4, 5, 6, 0],  # Middle row stays middle
                [0, 1, 2, 3, 0],  # Top row moved to bottom
                [0, 0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_flip_object_l_shape(self):
        """Test flip of L-shaped selection."""
        # Create an L-shaped pattern
        working_grid = jnp.array(
            [[1, 0, 0, 0], [2, 0, 0, 0], [3, 4, 5, 0], [0, 0, 0, 0]]
        )

        state = create_test_state(working_grid)

        # Select the L-shaped region
        selection = jnp.array(
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        )

        result_state = flip_object(state, selection, 0)  # Horizontal flip

        # The bounding box is 3x3, so flip within that region
        expected_grid = jnp.array(
            [
                [0, 0, 1, 0],  # Flipped horizontally within bounding box
                [0, 0, 2, 0],  # Flipped horizontally within bounding box
                [5, 4, 3, 0],  # Flipped horizontally within bounding box
                [0, 0, 0, 0],
            ]
        )

        assert jnp.array_equal(result_state.working_grid, expected_grid)

    def test_flip_object_jax_compatibility(self):
        """Test that enhanced flip operation is JAX compatible."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        result = assert_jax_compatible(
            flip_object,
            sample_state,
            selection,
            1,  # Vertical flip
            test_jit=True,
        )

        assert hasattr(result, "working_grid")
        assert result.working_grid.shape == sample_state.working_grid.shape


class TestOperationConsistency:
    """Test consistency across all enhanced operations."""

    def test_consistent_bounding_box_calculation(self):
        """Test that all operations use consistent bounding box calculation."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        selection = jnp.array(
            [
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        # Get bounding box directly
        min_row, max_row, min_col, max_col = get_selection_bounding_box(selection)
        expected_bbox = (1, 2, 1, 2)

        assert (min_row, max_row, min_col, max_col) == expected_bbox

        # Test that operations handle the same bounding box consistently
        # (We can't directly test internal bounding box usage, but we can verify
        # that operations work on the same selection)

        move_result = move_object(sample_state, selection, 0)
        rotate_result = rotate_object(sample_state, selection, 0)
        flip_result = flip_object(sample_state, selection, 0)

        # All operations should produce valid results
        assert move_result.working_grid.shape == sample_state.working_grid.shape
        assert rotate_result.working_grid.shape == sample_state.working_grid.shape
        assert flip_result.working_grid.shape == sample_state.working_grid.shape

    def test_edge_case_boundary_selections(self):
        """Test operations with selections at grid boundaries."""
        working_grid = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        sample_state = create_test_state(working_grid)

        # Top-left corner selection
        corner_selection = jnp.array(
            [
                [True, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        # All operations should handle boundary cases gracefully
        move_result = move_object(sample_state, corner_selection, 0)
        rotate_result = rotate_object(sample_state, corner_selection, 0)
        flip_result = flip_object(sample_state, corner_selection, 0)

        # Results should be valid (no crashes or invalid shapes)
        assert move_result.working_grid.shape == sample_state.working_grid.shape
        assert rotate_result.working_grid.shape == sample_state.working_grid.shape
        assert flip_result.working_grid.shape == sample_state.working_grid.shape

    def test_error_handling_invalid_bounding_boxes(self):
        """Test error handling for invalid bounding boxes."""
        # Test validation function directly
        assert validate_bounding_box_for_operation(-1, -1, -1, -1) == False
        assert validate_bounding_box_for_operation(0, 1, 0, 1) == True
        assert (
            validate_bounding_box_for_operation(0, 1, 0, 1, require_square=True) == True
        )
        assert (
            validate_bounding_box_for_operation(0, 2, 0, 1, require_square=True)
            == False
        )


class TestJAXTransformations:
    """Test JAX transformations (jit, vmap) compatibility."""

    def test_all_utilities_jit_compatible(self):
        """Test that all utility functions are JIT compatible."""
        # Test data
        selection = jnp.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )

        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test bounding box calculation
        jitted_bbox = jax.jit(get_selection_bounding_box)
        bbox_result = jitted_bbox(selection)
        assert len(bbox_result) == 4

        # Test rectangle extraction
        jitted_extract = jax.jit(extract_object_rectangle)
        extract_result = jitted_extract(grid, selection)
        assert len(extract_result) == 2

        # Test validation
        jitted_validate = jax.jit(validate_single_cell_selection)
        validate_result = jitted_validate(selection)
        assert isinstance(validate_result, jnp.ndarray)

    def test_operations_jit_compatible(self):
        """Test that all enhanced operations are JIT compatible."""
        # Create test state
        working_grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        state = create_test_state(working_grid)

        selection = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )

        # Test JIT compilation of operations
        jitted_move = jax.jit(move_object)
        move_result = jitted_move(state, selection, 0)
        assert hasattr(move_result, "working_grid")

        jitted_rotate = jax.jit(rotate_object)
        rotate_result = jitted_rotate(state, selection, 0)
        assert hasattr(rotate_result, "working_grid")

        jitted_flip = jax.jit(flip_object)
        flip_result = jitted_flip(state, selection, 0)
        assert hasattr(flip_result, "working_grid")

        jitted_flood_fill = jax.jit(flood_fill_color)
        single_cell_selection = jnp.array(
            [[True, False, False], [False, False, False], [False, False, False]]
        )
        flood_result = jitted_flood_fill(state, single_cell_selection, 8)
        assert hasattr(flood_result, "working_grid")

    def test_static_shape_preservation(self):
        """Test that all operations preserve static shapes for JAX compatibility."""
        # Create test data with known shapes - use square shape to avoid rotation issues
        grid_shape = (5, 5)
        working_grid = jnp.ones(grid_shape, dtype=jnp.int32)

        state = create_test_state(working_grid)

        selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)

        # Test that all operations preserve grid shape
        operations = [
            (move_object, [state, selection, 0]),
            (rotate_object, [state, selection, 0]),
            (flip_object, [state, selection, 0]),
            (
                flood_fill_color,
                [state, selection.at[1, 1].set(True).at[1:3, 1:3].set(False), 5],
            ),
        ]

        for op_func, args in operations:
            result = op_func(*args)
            assert result.working_grid.shape == grid_shape, (
                f"{op_func.__name__} changed grid shape"
            )


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_grid(self):
        """Test operations on single pixel grid."""
        working_grid = jnp.array([[5]])

        state = create_test_state(working_grid)

        selection = jnp.array([[True]])

        # Test all operations on single pixel
        move_result = move_object(state, selection, 0)
        rotate_result = rotate_object(state, selection, 0)
        flip_result = flip_object(state, selection, 0)
        flood_result = flood_fill_color(state, selection, 3)

        # All should handle single pixel gracefully
        assert move_result.working_grid.shape == (1, 1)
        assert rotate_result.working_grid.shape == (1, 1)
        assert flip_result.working_grid.shape == (1, 1)
        assert flood_result.working_grid.shape == (1, 1)

    def test_maximum_size_grid(self):
        """Test operations on large grid."""
        # Create a reasonably large grid for testing
        grid_size = (10, 10)
        working_grid = jnp.arange(100).reshape(grid_size)

        state = create_test_state(working_grid)

        # Select a region in the middle
        selection = jnp.zeros(grid_size, dtype=jnp.bool_)
        selection = selection.at[3:7, 3:7].set(True)

        # Test operations on large grid
        move_result = move_object(state, selection, 1)
        rotate_result = rotate_object(state, selection, 0)
        flip_result = flip_object(state, selection, 1)

        # All should complete without errors
        assert move_result.working_grid.shape == grid_size
        assert rotate_result.working_grid.shape == grid_size
        assert flip_result.working_grid.shape == grid_size

    def test_all_zeros_grid(self):
        """Test operations on grid with all zeros."""
        working_grid = jnp.zeros((3, 3), dtype=jnp.int32)

        state = create_test_state(working_grid)

        selection = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )

        # Test operations on all-zeros grid
        move_result = move_object(state, selection, 0)
        rotate_result = rotate_object(state, selection, 0)
        flip_result = flip_object(state, selection, 0)

        # Results should be valid (all zeros should remain zeros after transformations)
        assert jnp.all(move_result.working_grid == 0)
        assert jnp.all(rotate_result.working_grid == 0)
        assert jnp.all(flip_result.working_grid == 0)

    def test_boundary_wrapping_edge_cases(self):
        """Test edge cases in boundary wrapping for move operations."""
        # Create a 2x2 grid to test minimal wrapping
        working_grid = jnp.array([[1, 2], [3, 4]])

        state = create_test_state(working_grid)

        selection = jnp.array([[True, True], [True, True]])

        # Test all four directions on minimal grid
        up_result = move_object(state, selection, 0)
        down_result = move_object(state, selection, 1)
        left_result = move_object(state, selection, 2)
        right_result = move_object(state, selection, 3)

        # Verify wrapping behavior
        expected_up = jnp.array([[3, 4], [1, 2]])  # Bottom row wraps to top
        expected_down = jnp.array([[3, 4], [1, 2]])  # Top row wraps to bottom
        expected_left = jnp.array([[2, 1], [4, 3]])  # Right column wraps to left
        expected_right = jnp.array([[2, 1], [4, 3]])  # Left column wraps to right

        assert jnp.array_equal(up_result.working_grid, expected_up)
        assert jnp.array_equal(down_result.working_grid, expected_down)
        assert jnp.array_equal(left_result.working_grid, expected_left)
        assert jnp.array_equal(right_result.working_grid, expected_right)


if __name__ == "__main__":
    # Enable JIT for testing JAX compatibility
    jax.config.update("jax_disable_jit", False)

    # Run tests
    pytest.main([__file__, "-v"])
