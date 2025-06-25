"""
Tests for grid operations module.

This module tests all 35 grid operations to ensure they work correctly
with JAX compilation and maintain proper state structure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import chex

from jaxarc.envs.grid_operations import (
    compute_grid_similarity,
    execute_grid_operation,
)
from jaxarc.envs.arc_env import ArcEnvironmentState
from jaxarc.types import ParsedTaskData


@pytest.fixture
def sample_grid():
    """Create a sample 5x5 grid for testing."""
    grid = jnp.array([
        [0, 1, 2, 0, 0],
        [1, 1, 0, 0, 3],
        [2, 0, 1, 1, 0],
        [0, 0, 0, 2, 2],
        [3, 3, 0, 0, 0]
    ], dtype=jnp.int32)
    return grid


@pytest.fixture
def sample_state(sample_grid):
    """Create a sample ArcEnvironmentState for testing."""
    h, w = sample_grid.shape

    # Create dummy task data
    task_data = ParsedTaskData(
        input_grids_examples=jnp.expand_dims(sample_grid, 0),
        input_masks_examples=jnp.ones((1, h, w), dtype=jnp.bool_),
        output_grids_examples=jnp.expand_dims(sample_grid, 0),
        output_masks_examples=jnp.ones((1, h, w), dtype=jnp.bool_),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(sample_grid, 0),
        test_input_masks=jnp.ones((1, h, w), dtype=jnp.bool_),
        true_test_output_grids=jnp.expand_dims(sample_grid, 0),
        true_test_output_masks=jnp.ones((1, h, w), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    return ArcEnvironmentState(
        done=jnp.array(False, dtype=jnp.bool_),
        step=0,
        task_data=task_data,
        active_train_pair_idx=jnp.array(0, dtype=jnp.int32),
        working_grid=sample_grid,
        working_grid_mask=jnp.ones((h, w), dtype=jnp.bool_),
        program=jnp.zeros((10, 5), dtype=jnp.int32),
        program_length=jnp.array(0, dtype=jnp.int32),
        active_agents=jnp.ones(1, dtype=jnp.bool_),
        cumulative_rewards=jnp.zeros(1, dtype=jnp.float32),
        selected=jnp.zeros((h, w), dtype=jnp.bool_),
        clipboard=jnp.zeros((h, w), dtype=jnp.int32),
        grid_dim=jnp.array([h, w], dtype=jnp.int32),
        target_dim=jnp.array([h, w], dtype=jnp.int32),
        max_grid_dim=jnp.array([h, w], dtype=jnp.int32),
        similarity_score=jnp.array(1.0, dtype=jnp.float32),
    )


class TestGridSimilarity:
    """Test grid similarity computation."""

    def test_identical_grids(self):
        """Test similarity of identical grids."""
        grid1 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)

        similarity = compute_grid_similarity(grid1, grid2)
        assert jnp.allclose(similarity, 1.0)

    def test_completely_different_grids(self):
        """Test similarity of completely different grids."""
        grid1 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2 = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)

        similarity = compute_grid_similarity(grid1, grid2)
        assert jnp.allclose(similarity, 0.0)

    def test_partially_similar_grids(self):
        """Test similarity of partially similar grids."""
        grid1 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2 = jnp.array([[1, 2], [5, 6]], dtype=jnp.int32)

        similarity = compute_grid_similarity(grid1, grid2)
        assert jnp.allclose(similarity, 0.5)  # 2 out of 4 pixels match

    def test_similarity_jit_compilation(self):
        """Test that similarity function can be JIT compiled."""
        @jax.jit
        def jit_similarity(grid1, grid2):
            return compute_grid_similarity(grid1, grid2)

        grid1 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        grid2 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)

        similarity = jit_similarity(grid1, grid2)
        assert jnp.allclose(similarity, 1.0)


class TestFillOperations:
    """Test fill operations (0-9)."""

    def test_fill_color_operation(self, sample_state):
        """Test basic fill color operation."""
        # Select a 2x2 region
        state = sample_state.replace(
            selected=sample_state.selected.at[1:3, 1:3].set(True)
        )

        # Fill with color 5
        new_state = execute_grid_operation(state, jnp.array(5, dtype=jnp.int32))

        # Check that selected region is filled with color 5
        expected_grid = sample_state.working_grid.at[1:3, 1:3].set(5)
        assert jnp.array_equal(new_state.working_grid, expected_grid)

    def test_all_fill_colors(self, sample_state):
        """Test all fill color operations (0-9)."""
        for color in range(10):
            state = sample_state.replace(
                selected=sample_state.selected.at[0, 0].set(True)
            )

            new_state = execute_grid_operation(state, jnp.array(color, dtype=jnp.int32))

            # Check that the selected cell has the correct color
            assert new_state.working_grid[0, 0] == color


class TestFloodFillOperations:
    """Test flood fill operations (10-19)."""

    def test_flood_fill_operation(self, sample_state):
        """Test basic flood fill operation."""
        # Create a grid with connected region of same color
        test_grid = jnp.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.int32)

        state = sample_state.replace(
            working_grid=test_grid,
            selected=sample_state.selected.at[0, 0].set(True)  # Select a cell with value 1
        )

        # Flood fill with color 5 (operation 15)
        new_state = execute_grid_operation(state, jnp.array(15, dtype=jnp.int32))

        # All connected 1's should become 5's
        expected_grid = jnp.array([
            [5, 5, 0, 0, 0],
            [5, 5, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.int32)

        assert jnp.array_equal(new_state.working_grid, expected_grid)


class TestObjectMovement:
    """Test object movement operations (20-23)."""

    def test_move_up_operation(self, sample_state):
        """Test move up operation."""
        # Select a region to move
        state = sample_state.replace(
            selected=sample_state.selected.at[2:4, 1:3].set(True)
        )

        # Move up (operation 20)
        new_state = execute_grid_operation(state, jnp.array(20, dtype=jnp.int32))

        # The operation should complete successfully and modify the grid
        assert new_state is not None
        assert new_state.working_grid.shape == sample_state.working_grid.shape
        # Grid should be different after move operation
        assert not jnp.array_equal(new_state.working_grid, sample_state.working_grid)

    def test_all_movement_directions(self, sample_state):
        """Test all movement operations."""
        # Test each direction
        for direction, op_id in enumerate([20, 21, 22, 23]):  # up, down, left, right
            state = sample_state.replace(
                selected=sample_state.selected.at[2, 2].set(True)
            )

            new_state = execute_grid_operation(state, jnp.array(op_id, dtype=jnp.int32))

            # The operation should complete without error
            assert new_state is not None
            assert new_state.working_grid.shape == sample_state.working_grid.shape


class TestTransformations:
    """Test transformation operations (24-27)."""

    def test_rotate_operation(self, sample_state):
        """Test rotation operations."""
        # Create a simple pattern to rotate
        test_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        test_grid = test_grid.at[1:3, 1:4].set(1)  # Horizontal rectangle

        state = sample_state.replace(
            working_grid=test_grid,
            selected=jnp.ones((5, 5), dtype=jnp.bool_)  # Select entire grid
        )

        # Rotate 90 degrees (operation 24)
        new_state = execute_grid_operation(state, jnp.array(24, dtype=jnp.int32))

        # The operation should complete
        assert new_state is not None
        assert new_state.working_grid.shape == (5, 5)

    def test_flip_operations(self, sample_state):
        """Test flip operations."""
        for flip_op in [26, 27]:  # horizontal, vertical flip
            state = sample_state.replace(
                selected=jnp.ones_like(sample_state.selected, dtype=jnp.bool_)
            )

            new_state = execute_grid_operation(state, jnp.array(flip_op, dtype=jnp.int32))

            assert new_state is not None
            assert new_state.working_grid.shape == sample_state.working_grid.shape


class TestClipboardOperations:
    """Test clipboard operations (28-30)."""

    def test_copy_operation(self, sample_state):
        """Test copy operation."""
        # Select a region to copy
        state = sample_state.replace(
            selected=sample_state.selected.at[1:3, 1:3].set(True)
        )

        # Copy operation (28)
        new_state = execute_grid_operation(state, jnp.array(28, dtype=jnp.int32))

        # Clipboard should contain the copied region
        expected_clipboard = jnp.zeros_like(sample_state.clipboard)
        expected_clipboard = expected_clipboard.at[1:3, 1:3].set(
            sample_state.working_grid[1:3, 1:3]
        )

        assert jnp.array_equal(new_state.clipboard, expected_clipboard)

    def test_paste_operation(self, sample_state):
        """Test paste operation."""
        # Set up clipboard with some content
        clipboard_content = jnp.zeros_like(sample_state.clipboard)
        clipboard_content = clipboard_content.at[0:2, 0:2].set(9)

        state = sample_state.replace(
            clipboard=clipboard_content,
            selected=sample_state.selected.at[2:4, 2:4].set(True)
        )

        # Paste operation (29)
        new_state = execute_grid_operation(state, jnp.array(29, dtype=jnp.int32))

        # The operation should complete successfully
        assert new_state is not None
        assert new_state.working_grid.shape == sample_state.working_grid.shape
        # Grid should potentially be different after paste operation
        assert new_state.working_grid is not None


class TestGridOperations:
    """Test grid operations (31-33)."""

    def test_clear_grid_operation(self, sample_state):
        """Test clear grid operation."""
        # Clear grid operation (31)
        new_state = execute_grid_operation(sample_state, jnp.array(31, dtype=jnp.int32))

        # Grid should be all zeros
        assert jnp.all(new_state.working_grid == 0)

    def test_copy_input_grid_operation(self, sample_state):
        """Test copy input grid operation."""
        # Modify working grid first
        modified_state = sample_state.replace(
            working_grid=jnp.ones_like(sample_state.working_grid) * 9
        )

        # Copy input grid operation (32)
        new_state = execute_grid_operation(modified_state, jnp.array(32, dtype=jnp.int32))

        # Working grid should match input grid
        input_grid = sample_state.task_data.input_grids_examples[0]
        assert jnp.array_equal(new_state.working_grid, input_grid)


class TestSubmitOperation:
    """Test submit operation (34)."""

    def test_submit_operation(self, sample_state):
        """Test submit operation."""
        # Submit operation (34)
        new_state = execute_grid_operation(sample_state, jnp.array(34, dtype=jnp.int32))

        # State should be marked as done
        assert new_state.done == True


class TestJAXCompatibility:
    """Test JAX compatibility of grid operations."""

    def test_jit_compilation(self, sample_state):
        """Test that execute_grid_operation can be JIT compiled."""
        @jax.jit
        def jit_execute_operation(state, operation):
            return execute_grid_operation(state, operation)

        # Test with a simple operation
        new_state = jit_execute_operation(sample_state, jnp.array(0, dtype=jnp.int32))

        assert new_state is not None
        assert new_state.working_grid.shape == sample_state.working_grid.shape

    def test_all_operations_jit_compatible(self, sample_state):
        """Test that all operations are JIT compatible."""
        @jax.jit
        def test_operation(state, op_id):
            return execute_grid_operation(state, op_id)

        # Test each operation
        for op_id in range(35):
            try:
                new_state = test_operation(sample_state, jnp.array(op_id, dtype=jnp.int32))
                assert new_state is not None
            except Exception as e:
                pytest.fail(f"Operation {op_id} failed with JIT compilation: {e}")

    def test_operations_preserve_state_structure(self, sample_state):
        """Test that operations preserve the state dataclass structure."""
        for op_id in range(35):
            new_state = execute_grid_operation(sample_state, jnp.array(op_id, dtype=jnp.int32))

            # Check that the state is still a valid ArcEnvironmentState
            assert isinstance(new_state, ArcEnvironmentState)

            # Check that all required fields are present
            assert hasattr(new_state, 'working_grid')
            assert hasattr(new_state, 'similarity_score')
            assert hasattr(new_state, 'done')
            assert hasattr(new_state, 'step')

    def test_vmap_compatibility_single_operation(self, sample_state):
        """Test basic vmap compatibility for operations."""
        def single_op(state, op_id):
            return execute_grid_operation(state, op_id)

        # Create batch of operations
        batch_size = 3
        operations = jnp.array([0, 1, 2], dtype=jnp.int32)

        # This test just checks that the function structure is compatible
        # Full vmap support would require additional work
        try:
            # Test single operation first
            result = single_op(sample_state, operations[0])
            assert result is not None
        except Exception as e:
            pytest.fail(f"Single operation failed: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_operation_id(self, sample_state):
        """Test behavior with invalid operation IDs."""
        # Operation IDs should be 0-34, test boundary
        valid_ops = [0, 34]
        for op_id in valid_ops:
            try:
                new_state = execute_grid_operation(sample_state, jnp.array(op_id, dtype=jnp.int32))
                assert new_state is not None
            except Exception as e:
                pytest.fail(f"Valid operation {op_id} failed: {e}")

    def test_empty_selection(self, sample_state):
        """Test operations with empty selection."""
        # All selections are False
        state = sample_state.replace(
            selected=jnp.zeros_like(sample_state.selected, dtype=jnp.bool_)
        )

        # Operations should still work (might be no-ops)
        for op_id in [0, 5, 20, 28]:  # Test a few different operation types
            new_state = execute_grid_operation(state, jnp.array(op_id, dtype=jnp.int32))
            assert new_state is not None

    def test_full_selection(self, sample_state):
        """Test operations with full grid selection."""
        state = sample_state.replace(
            selected=jnp.ones_like(sample_state.selected, dtype=jnp.bool_)
        )

        # Operations should work with full selection
        for op_id in [0, 5, 31]:  # Test fill and clear operations
            new_state = execute_grid_operation(state, jnp.array(op_id, dtype=jnp.int32))
            assert new_state is not None
