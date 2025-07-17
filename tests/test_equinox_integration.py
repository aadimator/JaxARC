"""
Integration tests for Equinox-based state management.

This module tests the key functionality of the Equinox-based ArcEnvState
to ensure it meets the requirements for task 9.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask, Grid


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
    grid_data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    grid_mask = jnp.ones((3, 3), dtype=bool)
    
    return ArcEnvState(
        task_data=sample_task,
        working_grid=grid_data,
        working_grid_mask=grid_mask,
        target_grid=grid_data,
        step_count=jnp.array(0, dtype=jnp.int32),
        episode_done=jnp.array(False),
        current_example_idx=jnp.array(0, dtype=jnp.int32),
        selected=jnp.zeros((3, 3), dtype=bool),
        clipboard=jnp.zeros((3, 3), dtype=jnp.int32),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
    )


def test_equinox_module_creation(sample_state):
    """Test that ArcEnvState is properly created as an Equinox Module."""
    # Requirement 5.1: JAX modules SHALL use Equinox Module classes where appropriate
    assert isinstance(sample_state, eqx.Module)
    assert isinstance(sample_state, ArcEnvState)


def test_automatic_pytree_registration(sample_state):
    """Test that Equinox provides automatic PyTree registration."""
    # Requirement 5.2: Equinox PyTree structures SHALL be preferred over chex dataclasses where beneficial
    
    # Should be able to flatten and unflatten without manual registration
    flat, tree_def = jax.tree_util.tree_flatten(sample_state)
    reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)
    
    # Check that reconstruction preserves structure and values
    assert isinstance(reconstructed, ArcEnvState)
    assert jnp.array_equal(reconstructed.working_grid, sample_state.working_grid)
    assert reconstructed.step_count == sample_state.step_count
    assert reconstructed.episode_done == sample_state.episode_done


def test_equinox_validation(sample_state):
    """Test that Equinox validation works correctly."""
    # Requirement 5.6: Equinox and JAXTyping integration SHALL improve code clarity, type safety, and JAX performance
    
    # Should not raise any exceptions for valid state
    sample_state.__check_init__()
    
    # Test with invalid state (mismatched shapes) - should raise exception
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
        )
        invalid_state.__check_init__()


def test_equinox_state_updates(sample_state):
    """Test that Equinox state updates work correctly."""
    # Test using eqx.tree_at for updates
    new_state = eqx.tree_at(
        lambda s: s.step_count,
        sample_state,
        sample_state.step_count + 1
    )
    
    assert new_state.step_count == 1
    assert sample_state.step_count == 0  # Original unchanged
    assert new_state is not sample_state  # New instance
    
    # Test multiple field updates
    new_state = eqx.tree_at(
        lambda s: (s.step_count, s.episode_done),
        sample_state,
        (jnp.array(5), jnp.array(True))
    )
    
    assert new_state.step_count == 5
    assert new_state.episode_done == True
    assert sample_state.step_count == 0  # Original unchanged
    assert sample_state.episode_done == False  # Original unchanged


def test_jax_transformations_compatibility(sample_state):
    """Test that Equinox state works with JAX transformations."""
    # Requirement 5.7: Migration SHALL maintain backward compatibility
    
    @jax.jit
    def increment_step(state):
        return eqx.tree_at(
            lambda s: s.step_count,
            state,
            state.step_count + 1
        )
    
    # Should work with JIT compilation
    new_state = increment_step(sample_state)
    assert new_state.step_count == 1
    
    # Test multiple calls
    new_state = increment_step(new_state)
    assert new_state.step_count == 2


def test_backward_compatibility(sample_state):
    """Test that the Equinox implementation maintains backward compatibility."""
    # Requirement 5.7: Migration SHALL maintain backward compatibility
    
    # Field access should work the same as before
    assert sample_state.step_count == 0
    assert sample_state.episode_done == False
    assert sample_state.current_example_idx == 0
    assert sample_state.similarity_score == 0.0
    
    # Array fields should be accessible
    assert sample_state.working_grid.shape == (3, 3)
    assert sample_state.working_grid_mask.shape == (3, 3)
    assert sample_state.target_grid.shape == (3, 3)
    assert sample_state.selected.shape == (3, 3)
    assert sample_state.clipboard.shape == (3, 3)
    
    # Should remain immutable
    with pytest.raises(AttributeError):
        sample_state.step_count = 5


def test_replace_method_compatibility(sample_state):
    """Test that the replace method works for backward compatibility."""
    new_state = sample_state.replace(
        step_count=jnp.array(10),
        episode_done=jnp.array(True),
        similarity_score=jnp.array(0.8)
    )
    
    assert new_state.step_count == 10
    assert new_state.episode_done == True
    assert new_state.similarity_score == 0.8
    
    # Original should be unchanged
    assert sample_state.step_count == 0
    assert sample_state.episode_done == False
    assert sample_state.similarity_score == 0.0
    
    # Other fields should be preserved
    assert jnp.array_equal(new_state.working_grid, sample_state.working_grid)
    assert jnp.array_equal(new_state.target_grid, sample_state.target_grid)


if __name__ == "__main__":
    pytest.main([__file__])