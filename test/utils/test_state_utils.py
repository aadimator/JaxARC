"""Tests for state utility functions."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.state import State
from jaxarc.utils.state_utils import (
    increment_step_count,
    update_grid_and_similarity,
    update_multiple_fields,
    update_selection,
    update_similarity_score,
    update_working_grid,
    validate_state_consistency,
)


@pytest.fixture
def sample_state():
    """Create a sample state for testing."""
    return State(
        working_grid=jnp.array([[1, 2], [3, 4]]),
        working_grid_mask=jnp.array([[True, True], [True, True]]),
        input_grid=jnp.array([[0, 1], [2, 3]]),
        input_grid_mask=jnp.array([[True, True], [True, True]]),
        target_grid=jnp.array([[5, 6], [7, 8]]),
        target_grid_mask=jnp.array([[True, True], [True, True]]),
        selected=jnp.array([[False, True], [False, False]]),
        clipboard=jnp.array([[0, 0], [0, 0]]),
        step_count=jnp.int32(5),
        allowed_operations_mask=jnp.ones(35, dtype=bool),
        similarity_score=jnp.float32(0.75),
        key=jax.random.PRNGKey(42),
        task_idx=jnp.int32(0),
        pair_idx=jnp.int32(0),
        carry={},
    )


class TestUpdateMultipleFields:
    """Test the update_multiple_fields utility function."""

    def test_update_single_field(self, sample_state):
        """Test updating a single field."""
        new_step_count = jnp.int32(10)
        updated_state = update_multiple_fields(sample_state, step_count=new_step_count)

        assert updated_state.step_count == new_step_count
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(
            updated_state.working_grid, sample_state.working_grid
        )
        chex.assert_trees_all_equal(
            updated_state.similarity_score, sample_state.similarity_score
        )

    def test_update_multiple_fields_basic(self, sample_state):
        """Test updating multiple fields simultaneously."""
        new_grid = jnp.array([[9, 8], [7, 6]])
        new_score = jnp.float32(0.9)
        new_step = jnp.int32(15)

        updated_state = update_multiple_fields(
            sample_state,
            working_grid=new_grid,
            similarity_score=new_score,
            step_count=new_step,
        )

        chex.assert_trees_all_equal(updated_state.working_grid, new_grid)
        assert updated_state.similarity_score == new_score
        assert updated_state.step_count == new_step
        # Unchanged fields should remain the same
        chex.assert_trees_all_equal(updated_state.target_grid, sample_state.target_grid)

    def test_update_no_fields(self, sample_state):
        """Test that updating with no fields returns the same state."""
        updated_state = update_multiple_fields(sample_state)

        # Should return the exact same state
        chex.assert_trees_all_equal(updated_state, sample_state)

    def test_update_nonexistent_field(self, sample_state):
        """Test error when trying to update nonexistent field."""
        with pytest.raises(AttributeError, match="has no field 'nonexistent_field'"):
            update_multiple_fields(sample_state, nonexistent_field=42)

    def test_update_all_fields(self, sample_state):
        """Test updating all fields at once."""
        new_working_grid = jnp.array([[10, 11], [12, 13]])
        new_working_mask = jnp.array([[True, False], [False, True]])
        new_target_grid = jnp.array([[14, 15], [16, 17]])
        new_target_mask = jnp.array([[False, True], [True, False]])
        new_selected = jnp.array([[True, False], [True, False]])
        new_step_count = jnp.int32(20)
        new_similarity = jnp.float32(0.95)
        new_carry = {"test": "value"}

        updated_state = update_multiple_fields(
            sample_state,
            working_grid=new_working_grid,
            working_grid_mask=new_working_mask,
            target_grid=new_target_grid,
            target_grid_mask=new_target_mask,
            selected=new_selected,
            step_count=new_step_count,
            similarity_score=new_similarity,
            carry=new_carry,
        )

        chex.assert_trees_all_equal(updated_state.working_grid, new_working_grid)
        chex.assert_trees_all_equal(updated_state.working_grid_mask, new_working_mask)
        chex.assert_trees_all_equal(updated_state.target_grid, new_target_grid)
        chex.assert_trees_all_equal(updated_state.target_grid_mask, new_target_mask)
        chex.assert_trees_all_equal(updated_state.selected, new_selected)
        assert updated_state.step_count == new_step_count
        assert updated_state.similarity_score == new_similarity
        assert updated_state.carry == new_carry


class TestValidateStateConsistency:
    """Test state validation functionality."""

    def test_validate_consistent_state(self, sample_state):
        """Test validation of a consistent state."""
        # Should not raise any errors
        validated_state = validate_state_consistency(sample_state)
        chex.assert_trees_all_equal(validated_state, sample_state)

    def test_validate_mismatched_grid_shapes(self, sample_state):
        """Test validation fails for mismatched grid shapes."""
        # Create state with mismatched grid shapes
        invalid_state = update_multiple_fields(
            sample_state,
            target_grid=jnp.array([[1, 2, 3], [4, 5, 6]]),  # Different shape
        )

        # Should raise error during validation
        with pytest.raises(Exception):  # equinox.error_if raises generic Exception
            validate_state_consistency(invalid_state)

    def test_validate_mismatched_mask_shape(self, sample_state):
        """Test validation fails for mismatched mask shape."""
        # Create state with mismatched mask shape
        invalid_state = update_multiple_fields(
            sample_state,
            working_grid_mask=jnp.array([[True, True, True]]),  # Different shape
        )

        with pytest.raises(Exception):
            validate_state_consistency(invalid_state)

    def test_validate_negative_step_count(self, sample_state):
        """Test validation fails for negative step count."""
        invalid_state = update_multiple_fields(sample_state, step_count=jnp.int32(-1))

        with pytest.raises(Exception):
            validate_state_consistency(invalid_state)

    def test_validate_invalid_similarity_score_low(self, sample_state):
        """Test validation fails for similarity score below 0."""
        invalid_state = update_multiple_fields(
            sample_state, similarity_score=jnp.float32(-0.1)
        )

        with pytest.raises(Exception):
            validate_state_consistency(invalid_state)

    def test_validate_invalid_similarity_score_high(self, sample_state):
        """Test validation fails for similarity score above 1."""
        invalid_state = update_multiple_fields(
            sample_state, similarity_score=jnp.float32(1.1)
        )

        with pytest.raises(Exception):
            validate_state_consistency(invalid_state)

    def test_validate_boundary_similarity_scores(self, sample_state):
        """Test validation passes for boundary similarity scores."""
        # Test score of 0.0
        state_zero = update_multiple_fields(
            sample_state, similarity_score=jnp.float32(0.0)
        )
        validated_zero = validate_state_consistency(state_zero)
        assert validated_zero.similarity_score == 0.0

        # Test score of 1.0
        state_one = update_multiple_fields(
            sample_state, similarity_score=jnp.float32(1.0)
        )
        validated_one = validate_state_consistency(state_one)
        assert validated_one.similarity_score == 1.0

    def test_validate_state_jit_compatible(self, sample_state):
        """Test that state validation is JIT compatible."""
        jitted_validate = jax.jit(validate_state_consistency)

        validated_state = jitted_validate(sample_state)
        expected_state = validate_state_consistency(sample_state)

        chex.assert_trees_all_equal(validated_state, expected_state)


class TestSpecificUpdateFunctions:
    """Test specific state update utility functions."""

    def test_update_working_grid(self, sample_state):
        """Test updating working grid specifically."""
        new_grid = jnp.array([[9, 8], [7, 6]])
        updated_state = update_working_grid(sample_state, new_grid)

        chex.assert_trees_all_equal(updated_state.working_grid, new_grid)
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(updated_state.target_grid, sample_state.target_grid)
        assert updated_state.step_count == sample_state.step_count

    def test_update_selection(self, sample_state):
        """Test updating selection mask specifically."""
        new_selection = jnp.array([[True, False], [True, True]])
        updated_state = update_selection(sample_state, new_selection)

        chex.assert_trees_all_equal(updated_state.selected, new_selection)
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(
            updated_state.working_grid, sample_state.working_grid
        )
        assert updated_state.step_count == sample_state.step_count

    def test_increment_step_count(self, sample_state):
        """Test incrementing step count specifically."""
        updated_state = increment_step_count(sample_state)

        assert updated_state.step_count == sample_state.step_count + 1
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(
            updated_state.working_grid, sample_state.working_grid
        )
        chex.assert_trees_all_equal(
            updated_state.similarity_score, sample_state.similarity_score
        )

    def test_update_similarity_score(self, sample_state):
        """Test updating similarity score specifically."""
        new_score = jnp.float32(0.95)
        updated_state = update_similarity_score(sample_state, new_score)

        assert updated_state.similarity_score == new_score
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(
            updated_state.working_grid, sample_state.working_grid
        )
        assert updated_state.step_count == sample_state.step_count

    def test_update_grid_and_similarity(self, sample_state):
        """Test updating both grid and similarity score together."""
        new_grid = jnp.array([[10, 11], [12, 13]])
        new_score = jnp.float32(0.85)

        updated_state = update_grid_and_similarity(sample_state, new_grid, new_score)

        chex.assert_trees_all_equal(updated_state.working_grid, new_grid)
        assert updated_state.similarity_score == new_score
        # Other fields should remain unchanged
        chex.assert_trees_all_equal(updated_state.target_grid, sample_state.target_grid)
        assert updated_state.step_count == sample_state.step_count


class TestJAXCompatibility:
    """Test JAX compatibility of state utilities."""

    def test_update_functions_jit_compatible(self, sample_state):
        """Test that update functions work with JIT compilation."""
        new_grid = jnp.array([[9, 8], [7, 6]])
        new_selection = jnp.array([[True, False], [False, True]])
        new_score = jnp.float32(0.9)

        # Test individual functions
        jitted_update_grid = jax.jit(update_working_grid)
        jitted_update_selection = jax.jit(update_selection)
        jitted_increment_step = jax.jit(increment_step_count)
        jitted_update_score = jax.jit(update_similarity_score)

        # Test that JIT versions produce same results
        grid_result = jitted_update_grid(sample_state, new_grid)
        expected_grid = update_working_grid(sample_state, new_grid)
        chex.assert_trees_all_equal(grid_result, expected_grid)

        selection_result = jitted_update_selection(sample_state, new_selection)
        expected_selection = update_selection(sample_state, new_selection)
        chex.assert_trees_all_equal(selection_result, expected_selection)

        step_result = jitted_increment_step(sample_state)
        expected_step = increment_step_count(sample_state)
        chex.assert_trees_all_equal(step_result, expected_step)

        score_result = jitted_update_score(sample_state, new_score)
        expected_score = update_similarity_score(sample_state, new_score)
        chex.assert_trees_all_equal(score_result, expected_score)

    def test_update_multiple_fields_jit_compatible(self, sample_state):
        """Test that update_multiple_fields works with JIT."""
        new_grid = jnp.array([[9, 8], [7, 6]])
        new_score = jnp.float32(0.9)
        new_step = jnp.int32(10)

        # Create a JIT-compatible wrapper
        @jax.jit
        def jitted_update(state, grid, score, step):
            return update_multiple_fields(
                state, working_grid=grid, similarity_score=score, step_count=step
            )

        jit_result = jitted_update(sample_state, new_grid, new_score, new_step)
        expected_result = update_multiple_fields(
            sample_state,
            working_grid=new_grid,
            similarity_score=new_score,
            step_count=new_step,
        )

        chex.assert_trees_all_equal(jit_result, expected_result)

    def test_state_immutability(self, sample_state):
        """Test that state updates preserve immutability."""
        original_grid = sample_state.working_grid.copy()
        original_step = sample_state.step_count

        # Update the state
        new_grid = jnp.array([[9, 8], [7, 6]])
        updated_state = update_working_grid(sample_state, new_grid)

        # Original state should be unchanged
        chex.assert_trees_all_equal(sample_state.working_grid, original_grid)
        assert sample_state.step_count == original_step

        # Updated state should have new values
        chex.assert_trees_all_equal(updated_state.working_grid, new_grid)
        assert updated_state.step_count == original_step  # Should be unchanged


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_update_with_different_dtypes(self, sample_state):
        """Test updating with different data types."""
        # Test with different integer types
        new_step_int8 = jnp.int8(10)
        new_step_int64 = jnp.int64(20)

        updated_int8 = update_multiple_fields(sample_state, step_count=new_step_int8)
        updated_int64 = update_multiple_fields(sample_state, step_count=new_step_int64)

        assert updated_int8.step_count == 10
        assert updated_int64.step_count == 20

        # Test with different float types
        new_score_float64 = jnp.float64(0.95)
        updated_float64 = update_multiple_fields(
            sample_state, similarity_score=new_score_float64
        )
        chex.assert_trees_all_close(updated_float64.similarity_score, 0.95)

    def test_update_with_empty_carry(self, sample_state):
        """Test updating carry field with different values."""
        # Test with empty dict
        updated_empty = update_multiple_fields(sample_state, carry={})
        assert updated_empty.carry == {}

        # Test with populated dict
        new_carry = {"key1": "value1", "key2": 42}
        updated_carry = update_multiple_fields(sample_state, carry=new_carry)
        assert updated_carry.carry == new_carry

    def test_update_with_large_grids(self):
        """Test updating with larger grid sizes."""
        # Create state with larger grids
        large_grid = jnp.ones((10, 10), dtype=jnp.int32)
        large_mask = jnp.ones((10, 10), dtype=jnp.bool_)
        large_selection = jnp.zeros((10, 10), dtype=jnp.bool_)

        large_state = State(
            working_grid=large_grid,
            working_grid_mask=large_mask,
            input_grid=large_grid,
            input_grid_mask=large_mask,
            target_grid=large_grid * 2,
            target_grid_mask=large_mask,
            selected=large_selection,
            clipboard=jnp.zeros((10, 10), dtype=jnp.int32),
            step_count=jnp.int32(0),
            allowed_operations_mask=jnp.ones(35, dtype=bool),
            similarity_score=jnp.float32(0.0),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.int32(0),
            pair_idx=jnp.int32(0),
            carry={},
        )

        # Test updates work with large grids
        new_large_grid = jnp.full((10, 10), 5, dtype=jnp.int32)
        updated_large = update_working_grid(large_state, new_large_grid)

        chex.assert_trees_all_equal(updated_large.working_grid, new_large_grid)
        chex.assert_shape(updated_large.working_grid, (10, 10))

    def test_sequential_updates(self, sample_state):
        """Test multiple sequential updates."""
        # Chain multiple updates
        state1 = increment_step_count(sample_state)
        state2 = update_similarity_score(state1, jnp.float32(0.9))
        state3 = update_working_grid(state2, jnp.array([[9, 8], [7, 6]]))

        # Verify final state has all updates
        assert state3.step_count == sample_state.step_count + 1
        assert state3.similarity_score == 0.9
        chex.assert_trees_all_equal(state3.working_grid, jnp.array([[9, 8], [7, 6]]))

        # Verify original state is unchanged
        assert sample_state.step_count == 5
        assert sample_state.similarity_score == 0.75
        chex.assert_trees_all_equal(
            sample_state.working_grid, jnp.array([[1, 2], [3, 4]])
        )

    def test_update_with_zero_step_count(self, sample_state):
        """Test updating with zero step count."""
        zero_step_state = update_multiple_fields(sample_state, step_count=jnp.int32(0))

        # Should be valid
        validated_state = validate_state_consistency(zero_step_state)
        assert validated_state.step_count == 0
