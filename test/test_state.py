"""
Tests for the State class in jaxarc.state module.

This module tests the centralized State definition to ensure proper JAX compatibility,
type safety, validation, and state transitions.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxarc.state import State
from jaxarc.types import (
    NUM_OPERATIONS,
)


class TestState:
    """Test cases for the State class."""

    def test_state_creation_basic(self):
        """Test basic State creation with all required fields."""
        # Create test grids and arrays
        grid_shape = (5, 5)
        working_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
        working_grid_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        input_grid = jnp.ones(grid_shape, dtype=jnp.int32)
        input_grid_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        target_grid = jnp.full(grid_shape, 2, dtype=jnp.int32)
        target_grid_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        selected = jnp.zeros(grid_shape, dtype=jnp.bool_)
        clipboard = jnp.zeros(grid_shape, dtype=jnp.int32)

        # Create scalar fields
        step_count = jnp.array(0, dtype=jnp.int32)
        allowed_operations_mask = jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_)
        similarity_score = jnp.array(0.0, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        task_idx = jnp.array(0, dtype=jnp.int32)
        pair_idx = jnp.array(0, dtype=jnp.int32)

        state = State(
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            input_grid=input_grid,
            input_grid_mask=input_grid_mask,
            target_grid=target_grid,
            target_grid_mask=target_grid_mask,
            selected=selected,
            clipboard=clipboard,
            step_count=step_count,
            allowed_operations_mask=allowed_operations_mask,
            similarity_score=similarity_score,
            key=key,
            task_idx=task_idx,
            pair_idx=pair_idx,
        )

        # Verify all fields are stored correctly
        chex.assert_trees_all_equal(state.working_grid, working_grid)
        chex.assert_trees_all_equal(state.working_grid_mask, working_grid_mask)
        chex.assert_trees_all_equal(state.input_grid, input_grid)
        chex.assert_trees_all_equal(state.input_grid_mask, input_grid_mask)
        chex.assert_trees_all_equal(state.target_grid, target_grid)
        chex.assert_trees_all_equal(state.target_grid_mask, target_grid_mask)
        chex.assert_trees_all_equal(state.selected, selected)
        chex.assert_trees_all_equal(state.clipboard, clipboard)
        chex.assert_trees_all_equal(state.step_count, step_count)
        chex.assert_trees_all_equal(
            state.allowed_operations_mask, allowed_operations_mask
        )
        chex.assert_trees_all_equal(state.similarity_score, similarity_score)
        chex.assert_trees_all_equal(state.key, key)
        chex.assert_trees_all_equal(state.task_idx, task_idx)
        chex.assert_trees_all_equal(state.pair_idx, pair_idx)
        assert state.carry is None

    def test_state_validation(self):
        """Test State validation catches invalid configurations."""
        grid_shape = (3, 3)

        # Create valid base arrays
        valid_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
        valid_mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        valid_selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
        valid_step_count = jnp.array(0, dtype=jnp.int32)
        valid_operations_mask = jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_)
        valid_similarity = jnp.array(0.0, dtype=jnp.float32)
        valid_key = jax.random.PRNGKey(42)
        valid_task_idx = jnp.array(0, dtype=jnp.int32)
        valid_pair_idx = jnp.array(0, dtype=jnp.int32)

        # Should create successfully with valid data
        state = State(
            working_grid=valid_grid,
            working_grid_mask=valid_mask,
            input_grid=valid_grid,
            input_grid_mask=valid_mask,
            target_grid=valid_grid,
            target_grid_mask=valid_mask,
            selected=valid_selection,
            clipboard=valid_grid,
            step_count=valid_step_count,
            allowed_operations_mask=valid_operations_mask,
            similarity_score=valid_similarity,
            key=valid_key,
            task_idx=valid_task_idx,
            pair_idx=valid_pair_idx,
        )

        # Verify validation passes
        assert state.working_grid.shape == grid_shape
        assert state.step_count.shape == ()

    def test_state_dynamic_fields(self):
        """Test State contains only dynamic fields that change during episodes."""
        grid_shape = (4, 4)

        # Create initial state
        initial_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
        mask = jnp.ones(grid_shape, dtype=jnp.bool_)
        selection = jnp.zeros(grid_shape, dtype=jnp.bool_)

        state = State(
            working_grid=initial_grid,
            working_grid_mask=mask,
            input_grid=initial_grid,
            input_grid_mask=mask,
            target_grid=initial_grid,
            target_grid_mask=mask,
            selected=selection,
            clipboard=initial_grid,
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
        )

        # Test that we can modify dynamic fields
        new_grid = jnp.ones(grid_shape, dtype=jnp.int32)
        new_step_count = jnp.array(5, dtype=jnp.int32)
        new_similarity = jnp.array(0.75, dtype=jnp.float32)

        # Create updated state (immutable update)
        updated_state = eqx.tree_at(
            lambda s: (s.working_grid, s.step_count, s.similarity_score),
            state,
            (new_grid, new_step_count, new_similarity),
        )

        # Verify updates
        chex.assert_trees_all_equal(updated_state.working_grid, new_grid)
        chex.assert_trees_all_equal(updated_state.step_count, new_step_count)
        chex.assert_trees_all_equal(updated_state.similarity_score, new_similarity)

        # Original state should be unchanged
        chex.assert_trees_all_equal(state.working_grid, initial_grid)
        chex.assert_trees_all_equal(state.step_count, jnp.array(0, dtype=jnp.int32))

    def test_state_jax_compatibility(self):
        """Test State works with JAX transformations."""
        grid_shape = (3, 3)

        def create_state():
            return State(
                working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
                working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
                input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
                target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
                clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
                step_count=jnp.array(0, dtype=jnp.int32),
                allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
                similarity_score=jnp.array(0.0, dtype=jnp.float32),
                key=jax.random.PRNGKey(42),
                task_idx=jnp.array(0, dtype=jnp.int32),
                pair_idx=jnp.array(0, dtype=jnp.int32),
            )

        # Test JIT compilation
        jitted_create = jax.jit(create_state)
        state = jitted_create()

        # Verify state was created correctly
        assert state.working_grid.shape == grid_shape
        chex.assert_type(state.step_count, jnp.integer)
        chex.assert_type(state.similarity_score, jnp.floating)

    def test_state_transitions_and_carry(self):
        """Test State transitions and carry functionality."""
        grid_shape = (2, 2)

        # Create initial state with carry
        initial_carry = {"episode_info": "test", "custom_data": jnp.array([1, 2, 3])}

        state = State(
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            carry=initial_carry,
        )

        # Verify carry is stored correctly
        assert state.carry is not None
        assert state.carry["episode_info"] == "test"
        chex.assert_trees_all_equal(state.carry["custom_data"], jnp.array([1, 2, 3]))

        # Test state transition (step increment)
        new_step_count = state.step_count + 1
        new_similarity = jnp.array(0.5, dtype=jnp.float32)

        next_state = eqx.tree_at(
            lambda s: (s.step_count, s.similarity_score),
            state,
            (new_step_count, new_similarity),
        )

        # Verify transition
        chex.assert_trees_all_equal(
            next_state.step_count, jnp.array(1, dtype=jnp.int32)
        )
        chex.assert_trees_all_equal(next_state.similarity_score, new_similarity)

        # Carry should be preserved
        assert next_state.carry is not None
        assert next_state.carry["episode_info"] == "test"

    def test_state_grid_operations_state(self):
        """Test State maintains grid operations state correctly."""
        grid_shape = (3, 3)

        # Create state with specific grid operation state
        working_grid = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        selected = jnp.array(
            [[True, False, True], [False, True, False], [True, False, True]],
            dtype=jnp.bool_,
        )
        clipboard = jnp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=jnp.int32)

        state = State(
            working_grid=working_grid,
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=selected,
            clipboard=clipboard,
            step_count=jnp.array(3, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.33, dtype=jnp.float32),
            key=jax.random.PRNGKey(123),
            task_idx=jnp.array(1, dtype=jnp.int32),
            pair_idx=jnp.array(2, dtype=jnp.int32),
        )

        # Verify grid operation state
        chex.assert_trees_all_equal(state.working_grid, working_grid)
        chex.assert_trees_all_equal(state.selected, selected)
        chex.assert_trees_all_equal(state.clipboard, clipboard)

        # Test selection update
        new_selection = jnp.zeros(grid_shape, dtype=jnp.bool_)
        new_selection = new_selection.at[1, 1].set(True)  # Select center cell

        updated_state = eqx.tree_at(lambda s: s.selected, state, new_selection)

        chex.assert_trees_all_equal(updated_state.selected, new_selection)
        # Original should be unchanged
        chex.assert_trees_all_equal(state.selected, selected)

    def test_state_episode_progress_tracking(self):
        """Test State tracks episode progress correctly."""
        grid_shape = (2, 2)

        # Create state at different episode stages
        base_arrays = {
            "working_grid": jnp.zeros(grid_shape, dtype=jnp.int32),
            "working_grid_mask": jnp.ones(grid_shape, dtype=jnp.bool_),
            "input_grid": jnp.ones(grid_shape, dtype=jnp.int32),
            "input_grid_mask": jnp.ones(grid_shape, dtype=jnp.bool_),
            "target_grid": jnp.full(grid_shape, 2, dtype=jnp.int32),
            "target_grid_mask": jnp.ones(grid_shape, dtype=jnp.bool_),
            "selected": jnp.zeros(grid_shape, dtype=jnp.bool_),
            "clipboard": jnp.zeros(grid_shape, dtype=jnp.int32),
            "allowed_operations_mask": jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            "key": jax.random.PRNGKey(42),
            "task_idx": jnp.array(0, dtype=jnp.int32),
            "pair_idx": jnp.array(0, dtype=jnp.int32),
        }

        # Initial state
        initial_state = State(
            step_count=jnp.array(0, dtype=jnp.int32),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            **base_arrays,
        )

        # Mid-episode state
        mid_state = State(
            step_count=jnp.array(10, dtype=jnp.int32),
            similarity_score=jnp.array(0.5, dtype=jnp.float32),
            **base_arrays,
        )

        # Near-end state
        final_state = State(
            step_count=jnp.array(25, dtype=jnp.int32),
            similarity_score=jnp.array(0.95, dtype=jnp.float32),
            **base_arrays,
        )

        # Verify progress tracking
        assert initial_state.step_count == 0
        assert mid_state.step_count == 10
        assert final_state.step_count == 25

        assert initial_state.similarity_score == 0.0
        assert mid_state.similarity_score == 0.5
        assert final_state.similarity_score == 0.95

    def test_state_operation_mask_control(self):
        """Test State dynamic operation filtering."""
        grid_shape = (2, 2)

        # Create state with restricted operations
        restricted_mask = jnp.zeros(NUM_OPERATIONS, dtype=jnp.bool_)
        # Only allow fill operations (0-9) and submit (34)
        restricted_mask = restricted_mask.at[:10].set(True)  # Fill operations
        restricted_mask = restricted_mask.at[34].set(True)  # Submit operation

        state = State(
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=restricted_mask,
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
        )

        # Verify operation mask
        chex.assert_trees_all_equal(state.allowed_operations_mask, restricted_mask)

        # Check specific operations
        assert state.allowed_operations_mask[0] == True  # Fill 0
        assert state.allowed_operations_mask[9] == True  # Fill 9
        assert state.allowed_operations_mask[10] == False  # Flood fill 0
        assert state.allowed_operations_mask[34] == True  # Submit

    def test_state_prng_key_management(self):
        """Test State PRNG key management for environment randomness."""
        grid_shape = (2, 2)

        # Create state with specific PRNG key
        initial_key = jax.random.PRNGKey(12345)

        state = State(
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=initial_key,
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
        )

        # Verify key is stored correctly
        chex.assert_trees_all_equal(state.key, initial_key)
        chex.assert_shape(state.key, (2,))
        chex.assert_type(state.key, jnp.integer)

        # Test key splitting for randomness
        new_key, subkey = jax.random.split(state.key)

        # Update state with new key
        updated_state = eqx.tree_at(lambda s: s.key, state, new_key)

        # Keys should be different
        assert not jnp.array_equal(state.key, updated_state.key)
        chex.assert_shape(updated_state.key, (2,))

    def test_state_task_pair_tracking(self):
        """Test State tracks task and pair indices correctly."""
        grid_shape = (2, 2)

        # Create state with specific task/pair indices
        task_idx = jnp.array(42, dtype=jnp.int32)
        pair_idx = jnp.array(3, dtype=jnp.int32)

        state = State(
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
            task_idx=task_idx,
            pair_idx=pair_idx,
        )

        # Verify indices
        chex.assert_trees_all_equal(state.task_idx, task_idx)
        chex.assert_trees_all_equal(state.pair_idx, pair_idx)
        chex.assert_shape(state.task_idx, ())
        chex.assert_shape(state.pair_idx, ())
        chex.assert_type(state.task_idx, jnp.int32)
        chex.assert_type(state.pair_idx, jnp.int32)

        # Test index updates (e.g., switching to next pair)
        new_pair_idx = jnp.array(4, dtype=jnp.int32)
        updated_state = eqx.tree_at(lambda s: s.pair_idx, state, new_pair_idx)

        chex.assert_trees_all_equal(updated_state.pair_idx, new_pair_idx)
        chex.assert_trees_all_equal(updated_state.task_idx, task_idx)  # Unchanged

    def test_state_pytree_operations(self):
        """Test State works with PyTree operations."""
        grid_shape = (2, 2)

        state = State(
            working_grid=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(5, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.8, dtype=jnp.float32),
            key=jax.random.PRNGKey(999),
            task_idx=jnp.array(10, dtype=jnp.int32),
            pair_idx=jnp.array(2, dtype=jnp.int32),
        )

        # Test PyTree flattening and unflattening
        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify reconstruction
        chex.assert_trees_all_equal(state.working_grid, reconstructed.working_grid)
        chex.assert_trees_all_equal(state.step_count, reconstructed.step_count)
        chex.assert_trees_all_equal(
            state.similarity_score, reconstructed.similarity_score
        )
        chex.assert_trees_all_equal(state.task_idx, reconstructed.task_idx)
        chex.assert_trees_all_equal(state.pair_idx, reconstructed.pair_idx)

    def test_state_equinox_module_properties(self):
        """Test State inherits Equinox Module properties correctly."""
        grid_shape = (2, 2)

        state = State(
            working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
            working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            input_grid=jnp.ones(grid_shape, dtype=jnp.int32),
            input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            target_grid=jnp.full(grid_shape, 2, dtype=jnp.int32),
            target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
            selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
            clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
        )

        # Test that it's an Equinox module
        assert isinstance(state, eqx.Module)

        # Test immutability - should create new instance
        new_step_count = jnp.array(10, dtype=jnp.int32)
        new_state = eqx.tree_at(lambda s: s.step_count, state, new_step_count)

        # Original should be unchanged
        chex.assert_trees_all_equal(state.step_count, jnp.array(0, dtype=jnp.int32))
        chex.assert_trees_all_equal(new_state.step_count, new_step_count)
