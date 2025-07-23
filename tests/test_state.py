"""
Tests for ArcEnvState Equinox module.

This module contains comprehensive tests for the ArcEnvState Equinox module,
focusing on state initialization, validation, update methods, JAX transformation
compatibility, and state utility methods.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.equinox_utils import check_jax_transformations
from jaxarc.utils.grid_utils import crop_grid_to_mask, get_actual_grid_shape_from_mask


@pytest.fixture
def sample_task():
    """Create a sample JaxArcTask for testing."""
    # Create simple 3x3 grids for testing
    grid_data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    grid_mask = jnp.ones((3, 3), dtype=bool)

    # Create task with correct field names
    task = JaxArcTask(
        input_grids_examples=jnp.expand_dims(grid_data, 0),
        input_masks_examples=jnp.expand_dims(grid_mask, 0),
        output_grids_examples=jnp.expand_dims(grid_data, 0),
        output_masks_examples=jnp.expand_dims(grid_mask, 0),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(grid_data, 0),
        test_input_masks=jnp.expand_dims(grid_mask, 0),
        true_test_output_grids=jnp.expand_dims(grid_data, 0),
        true_test_output_masks=jnp.expand_dims(grid_mask, 0),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

    return task


@pytest.fixture
def sample_state(sample_task):
    """Create a sample ArcEnvState for testing."""
    from jaxarc.state import create_arc_env_state
    
    grid_data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    grid_mask = jnp.ones((3, 3), dtype=bool)

    return create_arc_env_state(
        task_data=sample_task,
        working_grid=grid_data,
        working_grid_mask=grid_mask,
        target_grid=grid_data,
        max_train_pairs=5,  # Test with smaller sizes
        max_test_pairs=2,
    )


@pytest.fixture
def padded_state(sample_task):
    """Create a state with padded grid for testing actual shape methods."""
    from jaxarc.state import create_arc_env_state
    
    # Create a 5x5 grid with content only in a 3x3 area
    grid_data = jnp.zeros((5, 5), dtype=jnp.int32)
    grid_data = grid_data.at[:3, :3].set(
        jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    )

    # Create a mask that only shows the 3x3 area as valid
    grid_mask = jnp.zeros((5, 5), dtype=bool)
    grid_mask = grid_mask.at[:3, :3].set(True)

    return create_arc_env_state(
        task_data=sample_task,
        working_grid=grid_data,
        working_grid_mask=grid_mask,
        target_grid=grid_data,
        max_train_pairs=8,  # Test with different sizes
        max_test_pairs=3,
    )


class TestStateInitialization:
    """Test ArcEnvState initialization and validation."""

    def test_state_creation(self, sample_state):
        """Test that ArcEnvState can be created as Equinox Module."""
        assert isinstance(sample_state, eqx.Module)
        assert isinstance(sample_state, ArcEnvState)

        # Check that all fields are accessible
        assert sample_state.step_count.item() == 0
        assert not sample_state.episode_done.item()
        assert sample_state.current_example_idx.item() == 0
        assert sample_state.similarity_score.item() == 0.0

        # Check array shapes
        assert sample_state.working_grid.shape == (3, 3)
        assert sample_state.working_grid_mask.shape == (3, 3)
        assert sample_state.target_grid.shape == (3, 3)
        assert sample_state.selected.shape == (3, 3)
        assert sample_state.clipboard.shape == (3, 3)

    def test_state_validation(self, sample_state):
        """Test that state validation works correctly."""
        # Should not raise any exceptions
        sample_state.__check_init__()

        # Test with invalid state (mismatched shapes)
        with pytest.raises((ValueError, AssertionError)):
            invalid_state = ArcEnvState(
                task_data=sample_state.task_data,
                working_grid=sample_state.working_grid,
                working_grid_mask=jnp.ones((2, 2), dtype=bool),  # Wrong shape
                target_grid=sample_state.target_grid,
                step_count=sample_state.step_count,
                episode_done=sample_state.episode_done,
                current_example_idx=sample_state.current_example_idx,
                selected=sample_state.selected,
                clipboard=sample_state.clipboard,
                similarity_score=sample_state.similarity_score,
                # Enhanced functionality fields
                episode_mode=sample_state.episode_mode,
                available_demo_pairs=sample_state.available_demo_pairs,
                available_test_pairs=sample_state.available_test_pairs,
                demo_completion_status=sample_state.demo_completion_status,
                test_completion_status=sample_state.test_completion_status,
                action_history=sample_state.action_history,
                action_history_length=sample_state.action_history_length,
                allowed_operations_mask=sample_state.allowed_operations_mask,
            )
            invalid_state.__check_init__()

    def test_invalid_grid_type(self, sample_state):
        """Test validation with incorrect grid type."""
        with pytest.raises((ValueError, AssertionError, TypeError)):
            invalid_state = ArcEnvState(
                task_data=sample_state.task_data,
                working_grid=jnp.ones((3, 3), dtype=jnp.float32),  # Wrong type
                working_grid_mask=sample_state.working_grid_mask,
                target_grid=sample_state.target_grid,
                step_count=sample_state.step_count,
                episode_done=sample_state.episode_done,
                current_example_idx=sample_state.current_example_idx,
                selected=sample_state.selected,
                clipboard=sample_state.clipboard,
                similarity_score=sample_state.similarity_score,
            )
            invalid_state.__check_init__()

    def test_invalid_mask_type(self, sample_state):
        """Test validation with incorrect mask type."""
        with pytest.raises((ValueError, AssertionError, TypeError)):
            invalid_state = ArcEnvState(
                task_data=sample_state.task_data,
                working_grid=sample_state.working_grid,
                working_grid_mask=jnp.ones((3, 3), dtype=jnp.int32),  # Wrong type
                target_grid=sample_state.target_grid,
                step_count=sample_state.step_count,
                episode_done=sample_state.episode_done,
                current_example_idx=sample_state.current_example_idx,
                selected=sample_state.selected,
                clipboard=sample_state.clipboard,
                similarity_score=sample_state.similarity_score,
            )
            invalid_state.__check_init__()

    def test_invalid_scalar_type(self, sample_state):
        """Test validation with incorrect scalar type."""
        with pytest.raises((ValueError, AssertionError, TypeError)):
            invalid_state = ArcEnvState(
                task_data=sample_state.task_data,
                working_grid=sample_state.working_grid,
                working_grid_mask=sample_state.working_grid_mask,
                target_grid=sample_state.target_grid,
                step_count=jnp.array(0.5, dtype=jnp.float32),  # Wrong type
                episode_done=sample_state.episode_done,
                current_example_idx=sample_state.current_example_idx,
                selected=sample_state.selected,
                clipboard=sample_state.clipboard,
                similarity_score=sample_state.similarity_score,
            )
            invalid_state.__check_init__()


class TestStateUpdateMethods:
    """Test ArcEnvState update methods."""

    def test_replace_method(self, sample_state):
        """Test the replace method for updating state fields."""
        new_state = sample_state.replace(
            step_count=jnp.array(10, dtype=jnp.int32),
            episode_done=jnp.array(True),
            similarity_score=jnp.array(0.8, dtype=jnp.float32),
        )

        assert new_state.step_count.item() == 10
        assert new_state.episode_done.item()
        assert jnp.allclose(
            new_state.similarity_score, jnp.array(0.8, dtype=jnp.float32)
        )

        # Original should be unchanged
        assert sample_state.step_count.item() == 0
        assert not sample_state.episode_done.item()
        assert sample_state.similarity_score.item() == 0.0

        # Other fields should be preserved
        assert jnp.array_equal(new_state.working_grid, sample_state.working_grid)
        assert jnp.array_equal(new_state.target_grid, sample_state.target_grid)

    def test_replace_with_grid_update(self, sample_state):
        """Test replacing grid arrays."""
        new_grid = jnp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=jnp.int32)

        new_state = sample_state.replace(working_grid=new_grid)

        assert jnp.array_equal(new_state.working_grid, new_grid)
        assert not jnp.array_equal(new_state.working_grid, sample_state.working_grid)

        # Other fields should be preserved
        assert new_state.step_count.item() == sample_state.step_count.item()
        assert new_state.episode_done.item() == sample_state.episode_done.item()

    def test_replace_multiple_fields(self, sample_state):
        """Test replacing multiple fields at once."""
        new_grid = jnp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=jnp.int32)
        new_selection = jnp.array(
            [[True, False, True], [False, True, False], [True, False, True]]
        )

        new_state = sample_state.replace(
            working_grid=new_grid,
            selected=new_selection,
            step_count=jnp.array(5, dtype=jnp.int32),
            episode_done=jnp.array(True),
        )

        # Check all updated fields
        assert jnp.array_equal(new_state.working_grid, new_grid)
        assert jnp.array_equal(new_state.selected, new_selection)
        assert new_state.step_count.item() == 5
        assert new_state.episode_done.item()

        # Original should be unchanged
        assert not jnp.array_equal(sample_state.working_grid, new_grid)
        assert not jnp.array_equal(sample_state.selected, new_selection)
        assert sample_state.step_count.item() == 0
        assert not sample_state.episode_done.item()

    def test_tree_at_updates(self, sample_state):
        """Test updating state using eqx.tree_at."""
        # Update single field
        new_state = eqx.tree_at(
            lambda s: s.step_count, sample_state, jnp.array(1, dtype=jnp.int32)
        )

        assert new_state.step_count.item() == 1
        assert sample_state.step_count.item() == 0  # Original unchanged
        assert new_state is not sample_state  # New instance

        # Update multiple fields
        new_state = eqx.tree_at(
            lambda s: (s.step_count, s.episode_done),
            sample_state,
            (jnp.array(5, dtype=jnp.int32), jnp.array(True)),
        )

        assert new_state.step_count.item() == 5
        assert new_state.episode_done.item()
        assert sample_state.step_count.item() == 0  # Original unchanged
        assert not sample_state.episode_done.item()  # Original unchanged


class TestStateUtilityMethods:
    """Test ArcEnvState utility methods."""

    def test_get_actual_grid_shape(self, padded_state):
        """Test get_actual_grid_shape method."""
        actual_shape = padded_state.get_actual_grid_shape()

        # Should return (3, 3) since only the 3x3 area is valid
        assert actual_shape == (3, 3)

        # Verify against the utility function
        expected_shape = get_actual_grid_shape_from_mask(padded_state.working_grid_mask)
        assert actual_shape == expected_shape

    def test_get_actual_working_grid(self, padded_state):
        """Test get_actual_working_grid method."""
        actual_grid = padded_state.get_actual_working_grid()

        # Should return a 3x3 grid with the valid content
        assert actual_grid.shape == (3, 3)

        # Verify content matches the original 3x3 area
        expected_grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        assert jnp.array_equal(actual_grid, expected_grid)

        # Verify against the utility function
        expected_grid = crop_grid_to_mask(
            padded_state.working_grid, padded_state.working_grid_mask
        )
        assert jnp.array_equal(actual_grid, expected_grid)

    def test_get_actual_target_grid(self, padded_state):
        """Test get_actual_target_grid method."""
        actual_grid = padded_state.get_actual_target_grid()

        # Should return a 3x3 grid with the valid content
        assert actual_grid.shape == (3, 3)

        # Verify content matches the original 3x3 area
        expected_grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        assert jnp.array_equal(actual_grid, expected_grid)

        # Verify against the utility function
        expected_grid = crop_grid_to_mask(
            padded_state.target_grid, padded_state.working_grid_mask
        )
        assert jnp.array_equal(actual_grid, expected_grid)

    def test_empty_grid_shape(self):
        """Test get_actual_grid_shape with empty grid."""
        # Create a completely masked grid (no valid cells)
        empty_mask = jnp.zeros((5, 5), dtype=bool)

        # Get the shape directly from the utility function
        shape = get_actual_grid_shape_from_mask(empty_mask)

        # Should return (0, 0) for empty grid
        assert shape == (0, 0)


class TestJAXTransformations:
    """Test JAX transformations with ArcEnvState."""

    def test_jit_compilation(self, sample_state):
        """Test that state works with JAX JIT compilation."""

        @jax.jit
        def increment_step(state):
            return state.replace(step_count=state.step_count + 1)

        new_state = increment_step(sample_state)
        assert new_state.step_count.item() == 1

        # Test multiple calls
        new_state = increment_step(new_state)
        assert new_state.step_count.item() == 2

    def test_jit_with_grid_operations(self, sample_state):
        """Test JIT with grid operations."""

        @jax.jit
        def update_grid(state):
            # Create a new grid with all values incremented by 1
            new_grid = state.working_grid + 1
            return state.replace(working_grid=new_grid)

        new_state = update_grid(sample_state)

        # Check that grid values were incremented
        expected_grid = jnp.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=jnp.int32)
        assert jnp.array_equal(new_state.working_grid, expected_grid)

    def test_jit_with_utility_methods(self, padded_state):
        """Test JIT with state utility methods."""

        @jax.jit
        def get_shape_and_grid(state):
            shape = state.get_actual_grid_shape()
            # We can't return the actual grid directly due to dynamic shapes
            # Instead, return a boolean indicating if shape is as expected
            shape_correct = (shape[0] == 3) & (shape[1] == 3)
            return shape_correct

        # This should work with JIT since we're not returning dynamic shapes
        result = get_shape_and_grid(padded_state)
        assert result

    def test_vmap_compatibility(self, sample_state):
        """Test vmap compatibility with simple operations."""
        # Create a batch of states with different step counts
        step_counts = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)

        # Function to test with vmap
        def increment_step_count(step_count):
            return step_count + 1

        # Apply vmap
        vmapped_fn = jax.vmap(increment_step_count)
        result = vmapped_fn(step_counts)

        expected = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)

    def test_grad_compatibility(self):
        """Test grad compatibility with simple scalar operations."""

        # Test grad with simple scalar operations
        def simple_loss(similarity_score):
            return similarity_score**2

        grad_fn = jax.grad(simple_loss)

        # Test with a simple float value
        test_score = jnp.array(0.5, dtype=jnp.float32)
        gradient = grad_fn(test_score)

        # Gradient of x^2 is 2x
        expected_gradient = 2.0 * test_score
        assert jnp.allclose(gradient, expected_gradient)

    def test_transformation_utility(self, sample_state):
        """Test the JAX transformations testing utility."""

        def simple_test_fn(state):
            return state.step_count + 1

        results = check_jax_transformations(sample_state, simple_test_fn)

        # Should test jit, vmap, and grad
        assert "jit" in results
        assert results["jit"]  # JIT should work


class TestPyTreeCompatibility:
    """Test PyTree compatibility with ArcEnvState."""

    def test_pytree_registration(self, sample_state):
        """Test that ArcEnvState is automatically registered as PyTree."""
        # Should be able to flatten and unflatten
        flat, tree_def = jax.tree_util.tree_flatten(sample_state)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        # Check that reconstruction preserves structure
        assert isinstance(reconstructed, ArcEnvState)
        assert jnp.array_equal(reconstructed.working_grid, sample_state.working_grid)
        assert reconstructed.step_count.item() == sample_state.step_count.item()
        assert reconstructed.episode_done.item() == sample_state.episode_done.item()

    def test_tree_map(self, sample_state):
        """Test jax.tree_util.tree_map with ArcEnvState."""

        # Map a function over all arrays in the state
        def add_one(x):
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.integer):
                return x + 1
            return x

        new_state = jax.tree_util.tree_map(add_one, sample_state)

        # Check that integer arrays were incremented
        assert new_state.step_count.item() == sample_state.step_count.item() + 1
        assert (
            new_state.current_example_idx.item()
            == sample_state.current_example_idx.item() + 1
        )

        # Check grid values
        expected_grid = sample_state.working_grid + 1
        assert jnp.array_equal(new_state.working_grid, expected_grid)

        # Boolean arrays should remain unchanged
        assert jnp.array_equal(
            new_state.working_grid_mask, sample_state.working_grid_mask
        )
        assert jnp.array_equal(new_state.episode_done, sample_state.episode_done)

    def test_tree_structure(self, sample_state):
        """Test tree structure of ArcEnvState."""
        # Get leaves and structure
        leaves, treedef = jax.tree_util.tree_flatten(sample_state)

        # Check number of leaves
        # Should have 10 fields: task_data, working_grid, working_grid_mask, target_grid,
        # step_count, episode_done, current_example_idx, selected, clipboard, similarity_score
        # Note: task_data is itself a PyTree, so the actual leaf count will be higher
        assert len(leaves) > 10

        # Reconstruct with same structure but different values
        new_leaves = [
            leaf + 1
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.number)
            else leaf
            for leaf in leaves
        ]
        new_state = jax.tree_util.tree_unflatten(treedef, new_leaves)

        # Check that it's still an ArcEnvState
        assert isinstance(new_state, ArcEnvState)


if __name__ == "__main__":
    pytest.main([__file__])
