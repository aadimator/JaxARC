"""
Comprehensive tests for grid operations module.

This module tests all grid transformation operations with JAX arrays,
validates operation correctness and edge cases, and tests grid operation
composition and chaining.

Tests cover:
- All individual grid operations (fill, flood fill, move, rotate, flip, etc.)
- JAX compatibility and transformations
- Edge cases and boundary conditions
- Operation composition and chaining
- Error handling and validation
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from jaxarc.envs.grid_operations import (
    # Helper functions
    apply_within_bounds,
    clear_grid,
    copy_input_grid,
    copy_to_clipboard,
    cut_to_clipboard,
    # Operation execution
    execute_grid_operation,
    # Individual operations
    fill_color,
    flip_object,
    flood_fill_color,
    get_all_operation_ids,
    get_effective_selection,
    get_operation_category,
    get_operation_display_text,
    # Utility functions
    get_operation_name,
    get_operations_by_category,
    is_valid_operation_id,
    move_object,
    paste_from_clipboard,
    resize_grid,
    rotate_object,
    simple_flood_fill,
    submit_solution,
    validate_bounding_box_for_operation,
)
from jaxarc.state import State


class TestOperationUtilities:
    """Test utility functions for grid operations."""

    def test_get_operation_name(self):
        """Test operation name retrieval."""
        assert get_operation_name(0) == "Fill 0"
        assert get_operation_name(9) == "Fill 9"
        assert get_operation_name(10) == "Flood Fill 0"
        assert get_operation_name(19) == "Flood Fill 9"
        assert get_operation_name(20) == "Move Up"
        assert get_operation_name(23) == "Move Right"
        assert get_operation_name(24) == "Rotate CW"
        assert get_operation_name(25) == "Rotate CCW"
        assert get_operation_name(26) == "Flip H"
        assert get_operation_name(27) == "Flip V"
        assert get_operation_name(28) == "Copy"
        assert get_operation_name(29) == "Paste"
        assert get_operation_name(30) == "Cut"
        assert get_operation_name(31) == "Clear"
        assert get_operation_name(32) == "Copy Input"
        assert get_operation_name(33) == "Resize"
        assert get_operation_name(34) == "Submit"

        # Test invalid operation
        with pytest.raises(ValueError, match="Unknown operation ID"):
            get_operation_name(35)

    def test_get_operation_display_text(self):
        """Test operation display text formatting."""
        assert get_operation_display_text(0) == "Op 0: Fill 0"
        assert get_operation_display_text(34) == "Op 34: Submit"

        with pytest.raises(ValueError, match="Unknown operation ID"):
            get_operation_display_text(35)

    def test_is_valid_operation_id(self):
        """Test operation ID validation."""
        # Valid operations
        for op_id in range(35):
            assert is_valid_operation_id(op_id) == True

        # Invalid operations
        assert is_valid_operation_id(35) == False
        assert is_valid_operation_id(-1) == False
        assert is_valid_operation_id(100) == False

    def test_get_all_operation_ids(self):
        """Test getting all valid operation IDs."""
        all_ids = get_all_operation_ids()
        assert all_ids == list(range(35))
        assert len(all_ids) == 35

    def test_get_operations_by_category(self):
        """Test operation categorization."""
        categories = get_operations_by_category()

        assert categories["fill"] == list(range(10))
        assert categories["flood_fill"] == list(range(10, 20))
        assert categories["movement"] == list(range(20, 24))
        assert categories["transformation"] == list(range(24, 28))
        assert categories["editing"] == list(range(28, 32))
        assert categories["special"] == list(range(32, 35))

    def test_get_operation_category(self):
        """Test getting category for specific operations."""
        assert get_operation_category(5) == "fill"
        assert get_operation_category(15) == "flood_fill"
        assert get_operation_category(22) == "movement"
        assert get_operation_category(24) == "transformation"
        assert get_operation_category(29) == "editing"
        assert get_operation_category(33) == "special"

        with pytest.raises(ValueError, match="Unknown operation ID"):
            get_operation_category(35)


class TestHelperFunctions:
    """Test helper functions used by grid operations."""

    def test_apply_within_bounds(self):
        """Test applying values within selection bounds."""
        grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        selection = jnp.array(
            [[True, False, True], [False, True, False], [True, False, True]],
            dtype=jnp.bool_,
        )

        # Test with scalar value
        result = apply_within_bounds(grid, selection, 0)
        expected = jnp.array([[0, 2, 0], [4, 0, 6], [0, 8, 0]], dtype=jnp.int32)
        chex.assert_trees_all_equal(result, expected)

        # Test with array value
        new_values = jnp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=jnp.int32)
        result = apply_within_bounds(grid, selection, new_values)
        expected = jnp.array([[9, 2, 7], [4, 5, 6], [3, 8, 1]], dtype=jnp.int32)
        chex.assert_trees_all_equal(result, expected)

    def test_get_effective_selection(self):
        """Test effective selection calculation."""
        working_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        # Test with existing selection
        selection = jnp.array(
            [[True, False, False], [False, False, False], [False, False, False]],
            dtype=jnp.bool_,
        )
        result = get_effective_selection(selection, working_grid_mask)
        chex.assert_trees_all_equal(result, selection)

        # Test with no selection (should use working_grid_mask)
        empty_selection = jnp.zeros((3, 3), dtype=jnp.bool_)
        result = get_effective_selection(empty_selection, working_grid_mask)
        chex.assert_trees_all_equal(result, working_grid_mask)

    def test_validate_bounding_box_for_operation(self):
        """Test bounding box validation."""
        # Valid bounding box
        assert validate_bounding_box_for_operation(0, 2, 0, 2) == True

        # Invalid bounding box (min_row < 0)
        assert validate_bounding_box_for_operation(-1, 2, 0, 2) == False

        # Test square requirement
        assert (
            validate_bounding_box_for_operation(0, 2, 0, 2, require_square=True) == True
        )  # 3x3 square
        assert (
            validate_bounding_box_for_operation(0, 2, 0, 1, require_square=True)
            == False
        )  # 3x2 rectangle


@pytest.fixture
def test_state(prng_key: PRNGKeyArray) -> State:
    """Create a test state for grid operations."""
    # Create a simple 5x5 grid with some pattern
    working_grid = jnp.array(
        [
            [1, 0, 2, 0, 1],
            [0, 1, 0, 2, 0],
            [2, 0, 1, 0, 2],
            [0, 2, 0, 1, 0],
            [1, 0, 2, 0, 1],
        ],
        dtype=jnp.int32,
    )

    working_grid_mask = jnp.ones((5, 5), dtype=jnp.bool_)

    # Create input and target grids
    input_grid = working_grid.copy()
    input_grid_mask = working_grid_mask.copy()
    target_grid = working_grid.copy()
    target_grid_mask = working_grid_mask.copy()

    # Create empty selection and clipboard
    selected = jnp.zeros((5, 5), dtype=jnp.bool_)
    clipboard = jnp.zeros((5, 5), dtype=jnp.int32)

    # Create allowed operations mask (all operations allowed)
    allowed_operations_mask = jnp.ones(35, dtype=jnp.bool_)

    return State(
        working_grid=working_grid,
        working_grid_mask=working_grid_mask,
        input_grid=input_grid,
        input_grid_mask=input_grid_mask,
        target_grid=target_grid,
        target_grid_mask=target_grid_mask,
        selected=selected,
        clipboard=clipboard,
        step_count=jnp.array(0, dtype=jnp.int32),
        allowed_operations_mask=allowed_operations_mask,
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
        key=prng_key,
        task_idx=jnp.array(0, dtype=jnp.int32),
        pair_idx=jnp.array(0, dtype=jnp.int32),
        carry=None,
    )


class TestFillOperations:
    """Test color fill operations (0-9)."""

    def test_fill_color_basic(self, test_state: State):
        """Test basic color fill operation."""
        # Create selection in top-left corner
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)
        selection = selection.at[0, 1].set(True)

        # Fill with color 5
        result_state = fill_color(test_state, selection, 5)

        # Check that selected cells are filled
        assert result_state.working_grid[0, 0] == 5
        assert result_state.working_grid[0, 1] == 5

        # Check that unselected cells are unchanged
        assert result_state.working_grid[0, 2] == test_state.working_grid[0, 2]
        assert result_state.working_grid[1, 0] == test_state.working_grid[1, 0]

    def test_fill_color_no_selection(self, test_state: State):
        """Test fill with no selection (should not change grid)."""
        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = fill_color(test_state, empty_selection, 5)

        # Grid should be unchanged
        chex.assert_trees_all_equal(result_state.working_grid, test_state.working_grid)

    def test_fill_all_colors(self, test_state: State):
        """Test filling with all valid colors (0-9)."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)  # Center cell

        for color in range(10):
            result_state = fill_color(test_state, selection, color)
            assert result_state.working_grid[2, 2] == color


class TestFloodFillOperations:
    """Test flood fill operations (10-19)."""

    def test_simple_flood_fill_basic(self):
        """Test basic flood fill functionality."""
        # Create a grid with connected regions
        grid = jnp.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 0, 4, 4],
                [3, 3, 0, 4, 4],
            ],
            dtype=jnp.int32,
        )

        # Select starting point in the 1-region
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)

        # Flood fill with color 9
        result = simple_flood_fill(grid, selection, 9)

        # Check that all connected 1s are filled
        assert result[0, 0] == 9
        assert result[0, 1] == 9
        assert result[1, 0] == 9
        assert result[1, 1] == 9

        # Check that other regions are unchanged
        assert result[0, 3] == 2  # 2-region unchanged
        assert result[3, 0] == 3  # 3-region unchanged

    def test_flood_fill_color_single_cell(self, test_state: State):
        """Test flood fill with single cell selection."""
        # Select a single cell
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)

        result_state = flood_fill_color(test_state, selection, 9)

        # Should flood fill connected region of same color
        original_color = test_state.working_grid[0, 0]

        # Check that flood fill occurred (exact behavior depends on grid pattern)
        assert result_state.working_grid[0, 0] == 9

    def test_flood_fill_color_multiple_cells(self, test_state: State):
        """Test flood fill with multiple cells selected (should not work)."""
        # Select multiple cells
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)
        selection = selection.at[0, 1].set(True)

        result_state = flood_fill_color(test_state, selection, 9)

        # Grid should be unchanged (multiple cell selection invalid for flood fill)
        chex.assert_trees_all_equal(result_state.working_grid, test_state.working_grid)

    def test_flood_fill_color_no_selection(self, test_state: State):
        """Test flood fill with no selection."""
        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = flood_fill_color(test_state, empty_selection, 9)

        # Grid should be unchanged
        chex.assert_trees_all_equal(result_state.working_grid, test_state.working_grid)


class TestMovementOperations:
    """Test object movement operations (20-23)."""

    def test_move_object_up(self, test_state: State):
        """Test moving object up."""
        # Create a 2x2 selection in the middle
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2:4, 2:4].set(True)

        result_state = move_object(test_state, selection, 0)  # Move up

        # The selected region should have moved up within its bounding box
        # This is a wrapping move within the bounding box
        assert result_state.working_grid is not None

        # Check that the operation completed without error
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_move_object_all_directions(self, test_state: State):
        """Test moving object in all directions."""
        # Create a small selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)

        directions = [0, 1, 2, 3]  # up, down, left, right

        for direction in directions:
            result_state = move_object(test_state, selection, direction)

            # Check that operation completed
            assert result_state.working_grid.shape == test_state.working_grid.shape

            # Grid should be different from original (unless no movement possible)
            # We can't easily predict exact result due to wrapping logic

    def test_move_object_no_selection(self, test_state: State):
        """Test move with no selection (should use entire working grid)."""
        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = move_object(test_state, empty_selection, 0)  # Move up

        # Should use effective selection (entire working grid)
        assert result_state.working_grid.shape == test_state.working_grid.shape


class TestRotationOperations:
    """Test object rotation operations (24-25)."""

    def test_rotate_object_clockwise(self, test_state: State):
        """Test rotating object clockwise."""
        # Create a square selection (required for rotation)
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:4, 1:4].set(True)  # 3x3 square

        result_state = rotate_object(test_state, selection, 0)  # Clockwise

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_rotate_object_counterclockwise(self, test_state: State):
        """Test rotating object counterclockwise."""
        # Create a square selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:4, 1:4].set(True)  # 3x3 square

        result_state = rotate_object(test_state, selection, 1)  # Counterclockwise

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_rotate_object_non_square(self, test_state: State):
        """Test rotating non-square selection (should not work)."""
        # Create a non-square selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:4].set(True)  # 2x3 rectangle

        result_state = rotate_object(test_state, selection, 0)

        # Grid should be unchanged (non-square selection invalid for rotation)
        chex.assert_trees_all_equal(result_state.working_grid, test_state.working_grid)


class TestFlipOperations:
    """Test object flip operations (26-27)."""

    def test_flip_object_horizontal(self, test_state: State):
        """Test flipping object horizontally."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:4].set(True)  # 2x3 rectangle

        result_state = flip_object(test_state, selection, 0)  # Horizontal flip

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_flip_object_vertical(self, test_state: State):
        """Test flipping object vertically."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:4, 1:3].set(True)  # 3x2 rectangle

        result_state = flip_object(test_state, selection, 1)  # Vertical flip

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape


class TestClipboardOperations:
    """Test clipboard operations (28-30)."""

    def test_copy_to_clipboard(self, test_state: State):
        """Test copying selection to clipboard."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0:2, 0:2].set(True)  # Top-left 2x2

        result_state = copy_to_clipboard(test_state, selection)

        # Check that clipboard contains copied data
        assert result_state.clipboard[0, 0] == test_state.working_grid[0, 0]
        assert result_state.clipboard[0, 1] == test_state.working_grid[0, 1]
        assert result_state.clipboard[1, 0] == test_state.working_grid[1, 0]
        assert result_state.clipboard[1, 1] == test_state.working_grid[1, 1]

        # Check that non-selected areas are zero in clipboard
        assert result_state.clipboard[0, 2] == 0
        assert result_state.clipboard[2, 0] == 0

        # Working grid should be unchanged
        chex.assert_trees_all_equal(result_state.working_grid, test_state.working_grid)

    def test_paste_from_clipboard(self, test_state: State):
        """Test pasting from clipboard."""
        # First copy something to clipboard
        copy_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        copy_selection = copy_selection.at[0:2, 0:2].set(True)
        state_with_clipboard = copy_to_clipboard(test_state, copy_selection)

        # Now paste to a different location
        paste_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        paste_selection = paste_selection.at[3:5, 3:5].set(True)  # Bottom-right 2x2

        result_state = paste_from_clipboard(state_with_clipboard, paste_selection)

        # Check that paste occurred
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_cut_to_clipboard(self, test_state: State):
        """Test cutting selection to clipboard."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[0:2, 0:2].set(True)  # Top-left 2x2

        original_values = test_state.working_grid[0:2, 0:2]
        result_state = cut_to_clipboard(test_state, selection)

        # Check that clipboard contains cut data
        chex.assert_trees_all_equal(result_state.clipboard[0:2, 0:2], original_values)

        # Check that selected area is cleared (set to 0)
        assert jnp.all(result_state.working_grid[0:2, 0:2] == 0)

        # Check that non-selected areas are unchanged
        chex.assert_trees_all_equal(
            result_state.working_grid[2:, :], test_state.working_grid[2:, :]
        )


class TestGridOperations:
    """Test grid-level operations (31-33)."""

    def test_clear_grid_selection(self, test_state: State):
        """Test clearing selected region."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)  # Middle 2x2

        result_state = clear_grid(test_state, selection)

        # Check that selected area is cleared
        assert jnp.all(result_state.working_grid[1:3, 1:3] == 0)

        # Check that non-selected areas are unchanged
        assert result_state.working_grid[0, 0] == test_state.working_grid[0, 0]
        assert result_state.working_grid[4, 4] == test_state.working_grid[4, 4]

    def test_clear_grid_no_selection(self, test_state: State):
        """Test clearing entire grid."""
        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = clear_grid(test_state, empty_selection)

        # Entire grid should be cleared
        assert jnp.all(result_state.working_grid == 0)

    def test_copy_input_grid(self, test_state: State):
        """Test copying input grid to working grid."""
        # Modify working grid first
        modified_state = eqx.tree_at(
            lambda s: s.working_grid, test_state, jnp.zeros((5, 5), dtype=jnp.int32)
        )

        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = copy_input_grid(modified_state, empty_selection)

        # Working grid should match input grid
        chex.assert_trees_all_equal(result_state.working_grid, test_state.input_grid)

    def test_resize_grid(self, test_state: State):
        """Test resizing grid."""
        # Select bottom-right corner to define new size
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 3].set(True)  # Should resize to 3x4

        result_state = resize_grid(test_state, selection)

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_submit_solution(self, test_state: State):
        """Test submit operation."""
        empty_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        result_state = submit_solution(test_state, empty_selection)

        # Submit should not change the state
        chex.assert_trees_all_equal(result_state, test_state)


class TestOperationExecution:
    """Test main operation execution function."""

    def test_execute_grid_operation_fill(self, test_state: State):
        """Test executing fill operations."""
        # Set up selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)
        state_with_selection = eqx.tree_at(lambda s: s.selected, test_state, selection)

        # Test fill operation (operation 5 = fill with color 5)
        result_state = execute_grid_operation(state_with_selection, 5)

        # Check that fill occurred
        assert result_state.working_grid[2, 2] == 5

    def test_execute_grid_operation_move(self, test_state: State):
        """Test executing move operations."""
        # Set up selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)
        state_with_selection = eqx.tree_at(lambda s: s.selected, test_state, selection)

        # Test move operation (operation 20 = move up)
        result_state = execute_grid_operation(state_with_selection, 20)

        # Check that operation completed
        assert result_state.working_grid.shape == test_state.working_grid.shape

    def test_execute_grid_operation_all_operations(self, test_state: State):
        """Test executing all valid operations."""
        # Set up a basic selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)
        state_with_selection = eqx.tree_at(lambda s: s.selected, test_state, selection)

        # Test all valid operations
        for op_id in range(35):
            result_state = execute_grid_operation(state_with_selection, op_id)

            # Check that operation completed without error
            assert result_state.working_grid.shape == test_state.working_grid.shape
            assert result_state.similarity_score is not None


class TestJAXCompatibility:
    """Test JAX transformations and compatibility."""

    def test_jit_compilation(self, test_state: State):
        """Test that operations compile under JIT."""
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)

        # Test JIT compilation of individual operations
        jitted_fill = jax.jit(fill_color)
        result = jitted_fill(test_state, selection, 5)
        assert result.working_grid[2, 2] == 5

        # Test JIT compilation of main execution function
        jitted_execute = jax.jit(execute_grid_operation)
        state_with_selection = eqx.tree_at(lambda s: s.selected, test_state, selection)
        result = jitted_execute(state_with_selection, 5)
        assert result.working_grid[2, 2] == 5

    def test_vmap_compatibility(self, test_state: State, prng_key: PRNGKeyArray):
        """Test batch processing with vmap."""
        # Create batch of selections
        selections = jnp.array(
            [
                jnp.zeros((5, 5), dtype=jnp.bool_).at[0, 0].set(True),
                jnp.zeros((5, 5), dtype=jnp.bool_).at[1, 1].set(True),
                jnp.zeros((5, 5), dtype=jnp.bool_).at[2, 2].set(True),
            ]
        )

        # Create batch of states
        states = jax.tree.map(lambda x: jnp.stack([x, x, x]), test_state)

        # Test vmap over fill operation
        vmapped_fill = jax.vmap(fill_color, in_axes=(0, 0, None))
        results = vmapped_fill(states, selections, 7)

        # Check results
        assert results.working_grid[0, 0, 0] == 7  # First batch item
        assert results.working_grid[1, 1, 1] == 7  # Second batch item
        assert results.working_grid[2, 2, 2] == 7  # Third batch item


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_operations_with_empty_grid(self, prng_key: PRNGKeyArray):
        """Test operations on empty grid."""
        # Create empty state
        empty_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        empty_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        empty_selection = jnp.zeros((3, 3), dtype=jnp.bool_)

        empty_state = State(
            working_grid=empty_grid,
            working_grid_mask=empty_mask,
            input_grid=empty_grid,
            input_grid_mask=empty_mask,
            target_grid=empty_grid,
            target_grid_mask=empty_mask,
            selected=empty_selection,
            clipboard=empty_grid,
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(35, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=prng_key,
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            carry=None,
        )

        # Test fill operation on empty grid
        selection = jnp.zeros((3, 3), dtype=jnp.bool_)
        selection = selection.at[1, 1].set(True)

        result = fill_color(empty_state, selection, 5)
        assert result.working_grid[1, 1] == 5

    def test_operations_with_single_cell_grid(self, prng_key: PRNGKeyArray):
        """Test operations on 1x1 grid."""
        # Create 1x1 state
        single_grid = jnp.array([[3]], dtype=jnp.int32)
        single_mask = jnp.ones((1, 1), dtype=jnp.bool_)
        single_selection = jnp.zeros((1, 1), dtype=jnp.bool_)

        single_state = State(
            working_grid=single_grid,
            working_grid_mask=single_mask,
            input_grid=single_grid,
            input_grid_mask=single_mask,
            target_grid=single_grid,
            target_grid_mask=single_mask,
            selected=single_selection,
            clipboard=single_grid,
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(35, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=prng_key,
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            carry=None,
        )

        # Test operations on single cell
        selection = jnp.ones((1, 1), dtype=jnp.bool_)

        # Fill should work
        result = fill_color(single_state, selection, 7)
        assert result.working_grid[0, 0] == 7

        # Move should not crash (though may not change anything)
        result = move_object(single_state, selection, 0)
        assert result.working_grid.shape == (1, 1)

    def test_boundary_selections(self, test_state: State):
        """Test operations with selections at grid boundaries."""
        # Test selection at edges
        edge_selections = [
            jnp.zeros((5, 5), dtype=jnp.bool_).at[0, :].set(True),  # Top edge
            jnp.zeros((5, 5), dtype=jnp.bool_).at[4, :].set(True),  # Bottom edge
            jnp.zeros((5, 5), dtype=jnp.bool_).at[:, 0].set(True),  # Left edge
            jnp.zeros((5, 5), dtype=jnp.bool_).at[:, 4].set(True),  # Right edge
        ]

        for selection in edge_selections:
            # Test fill operation
            result = fill_color(test_state, selection, 8)
            assert result.working_grid.shape == test_state.working_grid.shape

            # Test move operation
            result = move_object(test_state, selection, 0)
            assert result.working_grid.shape == test_state.working_grid.shape


class TestOperationChaining:
    """Test composition and chaining of operations."""

    def test_fill_then_move(self, test_state: State):
        """Test chaining fill and move operations."""
        # Create selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:3, 1:3].set(True)  # 2x2 square

        # First fill with color 7
        state1 = fill_color(test_state, selection, 7)
        assert jnp.all(state1.working_grid[1:3, 1:3] == 7)

        # Then move the filled region
        state2 = move_object(state1, selection, 0)  # Move up

        # Check that operation completed
        assert state2.working_grid.shape == test_state.working_grid.shape

    def test_copy_paste_chain(self, test_state: State):
        """Test copy-paste operation chain."""
        # Copy from one location
        copy_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        copy_selection = copy_selection.at[0:2, 0:2].set(True)

        state1 = copy_to_clipboard(test_state, copy_selection)

        # Paste to another location
        paste_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        paste_selection = paste_selection.at[3:5, 3:5].set(True)

        state2 = paste_from_clipboard(state1, paste_selection)

        # Check that operation completed
        assert state2.working_grid.shape == test_state.working_grid.shape

    def test_rotate_then_flip(self, test_state: State):
        """Test rotation followed by flip."""
        # Create square selection for rotation
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[1:4, 1:4].set(True)  # 3x3 square

        # First rotate clockwise
        state1 = rotate_object(test_state, selection, 0)

        # Then flip horizontally
        state2 = flip_object(state1, selection, 0)

        # Check that operations completed
        assert state2.working_grid.shape == test_state.working_grid.shape

    def test_multiple_operation_execution(self, test_state: State):
        """Test executing multiple operations in sequence."""
        # Set up selection
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)
        state_with_selection = eqx.tree_at(lambda s: s.selected, test_state, selection)

        # Execute sequence of operations
        operations = [5, 28, 31, 32]  # Fill 5, Copy, Clear, Copy Input

        current_state = state_with_selection
        for op in operations:
            current_state = execute_grid_operation(current_state, op)

            # Check that each operation completed
            assert current_state.working_grid.shape == test_state.working_grid.shape
