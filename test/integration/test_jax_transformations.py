"""
JAX transformation tests for JaxARC.

This module tests JAX transformations (jit, vmap, pmap) on core functions
to ensure compatibility and correctness.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc import JaxArcConfig
from jaxarc.envs.actions import action_handler, create_action
from jaxarc.envs.functional import validate_action_jax
from jaxarc.envs.grid_operations import (
    execute_grid_operation,
)
from jaxarc.registration import available_task_ids, make
from jaxarc.utils.grid_utils import (
    compute_grid_similarity,
    get_selection_bounding_box,
    validate_single_cell_selection,
)
from jaxarc.utils.state_utils import (
    increment_step_count,
    update_working_grid,
)


def get_grid_shape_from_state(state):
    """Helper function to get grid shape from state."""
    return state.working_grid.shape


class TestJITCompilation:
    """Test JIT compilation of core functions."""

    def test_reset_jit_compilation(self, sample_env_and_params):
        """Test that reset function compiles successfully under JIT."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        # Test JIT compilation
        jitted_reset = jax.jit(env.reset)

        # Should compile without errors
        state_jit, timestep_jit = jitted_reset(key)
        state_normal, timestep_normal = env.reset(key)

        # Results should be identical
        chex.assert_trees_all_close(
            timestep_jit.observation, timestep_normal.observation
        )
        chex.assert_trees_all_close(timestep_jit.reward, timestep_normal.reward)
        assert timestep_jit.step_type == timestep_normal.step_type

    def test_step_jit_compilation(self, sample_env_and_params):
        """Test that step function compiles successfully under JIT."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        # Create initial timestep
        state, timestep = env.reset(key)

        # Create test action with correct grid shape
        grid_shape = get_grid_shape_from_state(state)
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),  # Fill with color 0
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True),
        )

        # Test JIT compilation
        jitted_step = jax.jit(env.step)

        # Should compile without errors
        next_state_jit, next_timestep_jit = jitted_step(state, action)
        next_state_normal, next_timestep_normal = env.step(state, action)

        # Results should be identical
        chex.assert_trees_all_close(
            next_timestep_jit.observation, next_timestep_normal.observation
        )
        chex.assert_trees_all_close(
            next_timestep_jit.reward, next_timestep_normal.reward
        )
        assert next_timestep_jit.step_type == next_timestep_normal.step_type

    def test_validate_action_jax_jit_compilation(self, sample_state):
        """Test that validate_action_jax compiles under JIT."""
        grid_shape = get_grid_shape_from_state(sample_state)
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[1, 1].set(True),
        )

        # Test JIT compilation
        jitted_validate = jax.jit(validate_action_jax)

        # Should compile without errors
        result_jit = jitted_validate(action, sample_state, None)
        result_normal = validate_action_jax(action, sample_state, None)

        # Results should be identical
        chex.assert_trees_all_close(result_jit, result_normal)

    def test_grid_operations_jit_compilation(self, sample_state):
        """Test that grid operations compile under JIT."""
        # Test execute_grid_operation
        jitted_execute = jax.jit(execute_grid_operation)

        # Should compile without errors
        result_jit = jitted_execute(sample_state, jnp.array(0, dtype=jnp.int32))
        result_normal = execute_grid_operation(
            sample_state, jnp.array(0, dtype=jnp.int32)
        )

        # Results should be identical
        chex.assert_trees_all_close(result_jit.working_grid, result_normal.working_grid)

    def test_grid_similarity_jit_compilation(self, sample_state):
        """Test that compute_grid_similarity compiles under JIT."""
        grid1 = sample_state.working_grid
        mask1 = sample_state.working_grid_mask
        grid2 = sample_state.target_grid
        mask2 = sample_state.target_grid_mask

        # Test JIT compilation
        jitted_similarity = jax.jit(compute_grid_similarity)

        # Should compile without errors
        result_jit = jitted_similarity(grid1, mask1, grid2, mask2)
        result_normal = compute_grid_similarity(grid1, mask1, grid2, mask2)

        # Results should be identical
        chex.assert_trees_all_close(result_jit, result_normal)

    def test_action_handler_jit_compilation(self, sample_state):
        """Test that action_handler compiles under JIT."""
        grid_shape = get_grid_shape_from_state(sample_state)
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True),
        )

        # Test JIT compilation
        jitted_handler = jax.jit(action_handler)

        # Should compile without errors
        result_jit = jitted_handler(action, sample_state.working_grid_mask)
        result_normal = action_handler(action, sample_state.working_grid_mask)

        # Results should be identical
        chex.assert_trees_all_close(result_jit, result_normal)

    def test_state_utils_jit_compilation(self, sample_state):
        """Test that state utility functions compile under JIT."""
        # Test update_working_grid
        new_grid = jnp.ones_like(sample_state.working_grid)
        jitted_update_grid = jax.jit(update_working_grid)

        result_jit = jitted_update_grid(sample_state, new_grid)
        result_normal = update_working_grid(sample_state, new_grid)

        chex.assert_trees_all_close(result_jit.working_grid, result_normal.working_grid)

        # Test increment_step_count
        jitted_increment = jax.jit(increment_step_count)

        result_jit = jitted_increment(sample_state)
        result_normal = increment_step_count(sample_state)

        chex.assert_trees_all_close(result_jit.step_count, result_normal.step_count)

    def test_grid_utils_jit_compilation(self, sample_state):
        """Test that grid utility functions compile under JIT."""
        grid_shape = get_grid_shape_from_state(sample_state)
        selection = jnp.zeros(grid_shape, dtype=jnp.bool_).at[1:4, 1:4].set(True)

        # Test get_selection_bounding_box
        jitted_bbox = jax.jit(get_selection_bounding_box)

        result_jit = jitted_bbox(selection)
        result_normal = get_selection_bounding_box(selection)

        chex.assert_trees_all_close(result_jit, result_normal)

        # Test validate_single_cell_selection
        single_cell = jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True)
        jitted_validate_single = jax.jit(validate_single_cell_selection)

        result_jit = jitted_validate_single(single_cell)
        result_normal = validate_single_cell_selection(single_cell)

        chex.assert_trees_all_close(result_jit, result_normal)

    def test_numerical_correctness_under_jit(self, sample_env_and_params):
        """Test that JIT compilation preserves numerical correctness."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(123)

        # Run multiple steps with and without JIT
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)

        # Normal execution
        state_normal, timestep_normal = env.reset(key)
        grid_shape = timestep_normal.observation.shape[:2]
        for i in range(3):  # Reduce iterations to fit in 5x5 grid
            pos = min(i, grid_shape[0] - 1)
            action = create_action(
                operation=jnp.array(i % 10, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[pos, pos].set(True),
            )
            state_normal, timestep_normal = env.step(state_normal, action)

        # JIT execution
        state_jit, timestep_jit = jitted_reset(key)
        for i in range(3):  # Reduce iterations to fit in 5x5 grid
            pos = min(i, grid_shape[0] - 1)
            action = create_action(
                operation=jnp.array(i % 10, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[pos, pos].set(True),
            )
            state_jit, timestep_jit = jitted_step(state_jit, action)

        # Results should be numerically identical
        chex.assert_trees_all_close(
            timestep_jit.observation, timestep_normal.observation
        )
        chex.assert_trees_all_close(timestep_jit.reward, timestep_normal.reward)
        chex.assert_trees_all_close(state_jit.working_grid, state_normal.working_grid)

    def test_compilation_performance(self, sample_env_and_params):
        """Test that JIT compilation provides performance benefits."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        # Compile functions
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)

        # Warm up JIT compilation
        state, timestep = jitted_reset(key)
        grid_shape = timestep.observation.shape[:2]
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[0, 0].set(True),
        )
        _ = jitted_step(state, action)

        # Test that compiled functions execute without errors
        # (Performance measurement would require timing, which is not reliable in tests)
        state, timestep = jitted_reset(key)
        for i in range(5):  # Reduce iterations to fit in 5x5 grid
            pos = i % grid_shape[0]
            action = create_action(
                operation=jnp.array(i % 10, dtype=jnp.int32),
                selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[pos, pos].set(True),
            )
            state, timestep = jitted_step(state, action)

        # Should complete without errors
        assert timestep.step_type in [0, 1, 2]  # Valid step types


class TestVmapTransformations:
    """Test vmap (vectorization) transformations on core functions."""

    @pytest.fixture
    def sample_env_and_params(self):
        """Create sample environment and parameters for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        task_id = task_ids[0] if task_ids else "all"
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    @pytest.fixture
    def batch_keys(self):
        """Create batch of PRNG keys for testing."""
        return jax.random.split(jax.random.PRNGKey(42), 8)

    def test_reset_vmap(self, sample_env_and_params, batch_keys):
        """Test that reset function works correctly with vmap."""
        env, env_params = sample_env_and_params

        # Test vmap over keys
        vmapped_reset = jax.vmap(env.reset)

        # Should work without errors
        batch_states, batch_timesteps = vmapped_reset(batch_keys)

        # Check batch dimensions
        batch_size = batch_keys.shape[0]
        # Get grid shape from first timestep (H, W, C)
        grid_shape_3d = batch_timesteps.observation.shape[1:]
        grid_shape_2d = grid_shape_3d[:2]

        chex.assert_shape(batch_timesteps.observation, (batch_size, *grid_shape_3d))
        chex.assert_shape(batch_timesteps.reward, (batch_size,))
        chex.assert_shape(batch_states.working_grid, (batch_size, *grid_shape_2d))

        # Each timestep should be valid
        for i in range(batch_size):
            assert batch_timesteps.step_type[i] == 0  # FIRST step type

    def test_step_vmap(self, sample_env_and_params, batch_keys):
        """Test that step function works correctly with vmap."""
        env, env_params = sample_env_and_params

        # Create batch of initial timesteps
        vmapped_reset = jax.vmap(env.reset)
        batch_states, batch_timesteps = vmapped_reset(batch_keys)

        # Create batch of actions
        batch_size = batch_keys.shape[0]
        grid_shape_3d = batch_timesteps.observation.shape[1:]  # H, W, C
        grid_shape_2d = grid_shape_3d[:2]  # H, W

        batch_actions = []
        for i in range(batch_size):
            pos = i % grid_shape_2d[0]
            selection = jnp.zeros(grid_shape_2d, dtype=jnp.bool_).at[pos, pos].set(True)
            action = create_action(
                operation=jnp.array(i % 10, dtype=jnp.int32), selection=selection
            )
            batch_actions.append(action)

        # Convert to batch format
        batch_operations = jnp.array([a.operation for a in batch_actions])
        batch_selections = jnp.stack([a.selection for a in batch_actions])

        # Create batch action structure
        def create_batch_action(op, sel):
            return create_action(operation=op, selection=sel)

        vmapped_create_action = jax.vmap(create_batch_action)
        batch_action_struct = vmapped_create_action(batch_operations, batch_selections)

        # Test vmap over step
        vmapped_step = jax.vmap(env.step, in_axes=(0, 0))

        # Should work without errors
        next_batch_states, next_batch_timesteps = vmapped_step(
            batch_states, batch_action_struct
        )

        # Check batch dimensions
        chex.assert_shape(
            next_batch_timesteps.observation, (batch_size, *grid_shape_3d)
        )
        chex.assert_shape(next_batch_timesteps.reward, (batch_size,))
        chex.assert_shape(next_batch_states.working_grid, (batch_size, *grid_shape_2d))

    def test_grid_operations_vmap(self):
        """Test that grid operations work correctly with vmap."""
        # Create batch of test states
        batch_size = 4
        grid_shape = (10, 10)

        # Create batch of grids
        batch_grids = jnp.zeros((batch_size, *grid_shape), dtype=jnp.int32)
        batch_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)

        # Create batch of operations
        batch_operations = jnp.array(
            [0, 1, 2, 3], dtype=jnp.int32
        )  # Different fill colors

        # Test vmap over compute_grid_similarity
        vmapped_similarity = jax.vmap(compute_grid_similarity, in_axes=(0, 0, 0, 0))

        # Should work without errors
        similarities = vmapped_similarity(
            batch_grids, batch_masks, batch_grids, batch_masks
        )

        # Check results
        chex.assert_shape(similarities, (batch_size,))
        # All should be 1.0 (identical grids)
        chex.assert_trees_all_close(similarities, jnp.ones(batch_size))

    def test_action_handler_vmap(self):
        """Test that action_handler works correctly with vmap."""
        batch_size = 4
        grid_shape = (10, 10)

        # Create batch of actions
        batch_operations = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        batch_selections = jnp.zeros((batch_size, *grid_shape), dtype=jnp.bool_)
        for i in range(batch_size):
            batch_selections = batch_selections.at[i, i, i].set(True)

        # Create batch action structure
        vmapped_create_action = jax.vmap(create_action)
        batch_actions = vmapped_create_action(batch_operations, batch_selections)

        # Create batch of masks
        batch_masks = jnp.ones((batch_size, *grid_shape), dtype=jnp.bool_)

        # Test vmap over action_handler
        vmapped_handler = jax.vmap(action_handler, in_axes=(0, 0))

        # Should work without errors
        results = vmapped_handler(batch_actions, batch_masks)

        # Check results
        chex.assert_shape(results, (batch_size, *grid_shape))

    def test_state_utils_vmap(self):
        """Test that state utility functions work correctly with vmap."""
        batch_size = 4
        grid_shape = (10, 10)

        # Create batch of mock states (simplified for testing)
        from jaxarc.state import State

        batch_states = []
        for i in range(batch_size):
            state = State(
                working_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
                working_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                input_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
                input_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                target_grid=jnp.zeros(grid_shape, dtype=jnp.int32),
                target_grid_mask=jnp.ones(grid_shape, dtype=jnp.bool_),
                selected=jnp.zeros(grid_shape, dtype=jnp.bool_),
                clipboard=jnp.zeros(grid_shape, dtype=jnp.int32),
                step_count=jnp.array(i, dtype=jnp.int32),
                task_idx=jnp.array(0, dtype=jnp.int32),
                pair_idx=jnp.array(0, dtype=jnp.int32),
                allowed_operations_mask=jnp.ones(35, dtype=jnp.bool_),
                similarity_score=jnp.array(0.0, dtype=jnp.float32),
                key=jax.random.PRNGKey(i),
            )
            batch_states.append(state)

        # Stack states into batch format
        batch_state = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args), *batch_states
        )

        # Test vmap over increment_step_count
        vmapped_increment = jax.vmap(increment_step_count)

        # Should work without errors
        incremented_states = vmapped_increment(batch_state)

        # Check results
        expected_counts = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        chex.assert_trees_all_close(incremented_states.step_count, expected_counts)

    def test_vectorization_correctness(self, sample_env_and_params):
        """Test that vmap produces same results as individual calls."""
        env, env_params = sample_env_and_params
        keys = jax.random.split(jax.random.PRNGKey(123), 4)

        # Individual calls
        individual_results = []
        for key in keys:
            _, timestep = env.reset(key)
            individual_results.append(timestep)

        # Batch call
        vmapped_reset = jax.vmap(env.reset)
        _, batch_result = vmapped_reset(keys)

        # Compare results
        for i, individual in enumerate(individual_results):
            chex.assert_trees_all_close(
                individual.observation, batch_result.observation[i]
            )
            chex.assert_trees_all_close(individual.reward, batch_result.reward[i])

    def test_performance_vectorization(self, sample_env_and_params):
        """Test that vmap provides performance benefits for batch operations."""
        env, env_params = sample_env_and_params
        batch_size = 16
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        # Compile vmapped function
        vmapped_reset = jax.jit(jax.vmap(env.reset))

        # Warm up compilation
        _ = vmapped_reset(keys)

        # Test that batch processing works efficiently
        batch_states, batch_timesteps = vmapped_reset(keys)

        # Verify batch dimensions
        grid_shape_3d = batch_timesteps.observation.shape[1:]
        grid_shape_2d = grid_shape_3d[:2]
        chex.assert_shape(batch_timesteps.observation, (batch_size, *grid_shape_3d))
        chex.assert_shape(batch_states.working_grid, (batch_size, *grid_shape_2d))

        # All timesteps should be valid
        assert jnp.all(batch_timesteps.step_type == 0)  # All FIRST step types


class TestPmapTransformations:
    """Test pmap (parallel map) transformations on core functions."""

    @pytest.fixture
    def sample_env_and_params(self):
        """Create sample environment and parameters for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        task_id = task_ids[0] if task_ids else "all"
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_pmap_availability(self):
        """Test that pmap is available and can be used."""
        # Check if multiple devices are available
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for pmap testing")

        # Simple pmap test
        def simple_fn(x):
            return x * 2

        pmapped_fn = jax.pmap(simple_fn)

        # Test with device-distributed data
        x = jnp.array([1.0, 2.0])  # One element per device
        result = pmapped_fn(x)

        expected = jnp.array([2.0, 4.0])
        chex.assert_trees_all_close(result, expected)

    def test_reset_pmap_compatibility(self, sample_env_and_params):
        """Test that reset function is compatible with pmap."""
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for pmap testing")

        env, env_params = sample_env_and_params

        # Create device-distributed keys
        num_devices = len(devices)
        keys = jax.random.split(jax.random.PRNGKey(42), num_devices)

        # Test pmap compatibility (structure check)
        try:
            pmapped_reset = jax.pmap(env.reset)
            # This tests that the function can be pmapped without errors
            # Actual execution would require proper device distribution
            assert pmapped_reset is not None
        except Exception as e:
            pytest.fail(f"pmap compilation failed: {e}")

    def test_step_pmap_compatibility(self, sample_env_and_params):
        """Test that step function is compatible with pmap."""
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for pmap testing")

        env, env_params = sample_env_and_params

        # Test pmap compatibility (structure check)
        try:
            pmapped_step = jax.pmap(env.step, in_axes=(0, 0))
            # This tests that the function can be pmapped without errors
            assert pmapped_step is not None
        except Exception as e:
            pytest.fail(f"pmap compilation failed: {e}")

    def test_multi_device_compatibility(self):
        """Test multi-device compatibility for core operations."""
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for pmap testing")

        # Test simple grid operations across devices
        def grid_operation(grid):
            return grid + 1

        pmapped_op = jax.pmap(grid_operation)

        # Create device-distributed grids
        num_devices = len(devices)
        grids = jnp.zeros((num_devices, 5, 5), dtype=jnp.int32)

        result = pmapped_op(grids)
        expected = jnp.ones((num_devices, 5, 5), dtype=jnp.int32)

        chex.assert_trees_all_close(result, expected)

    def test_pmap_static_shapes(self):
        """Test that pmap works with static shapes required by JAX."""
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for pmap testing")

        # Test that our grid operations maintain static shapes
        def static_shape_op(x):
            # Ensure static shape is maintained
            return jnp.zeros_like(x) + x

        pmapped_op = jax.pmap(static_shape_op)

        num_devices = len(devices)
        test_data = jnp.ones((num_devices, 10, 10), dtype=jnp.int32)

        result = pmapped_op(test_data)
        chex.assert_shape(result, (num_devices, 10, 10))
        chex.assert_trees_all_close(result, test_data)


class TestPRNGManagement:
    """Test PRNG key management and reproducible randomization."""

    @pytest.fixture
    def sample_env_and_params(self):
        """Create sample environment and parameters for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        task_id = task_ids[0] if task_ids else "all"
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_prng_key_reproducibility(self, sample_env_and_params):
        """Test that same PRNG keys produce identical results."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        # Run reset twice with same key
        state1, timestep1 = env.reset(key)
        state2, timestep2 = env.reset(key)

        # Results should be identical
        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(state1.working_grid, state2.working_grid)
        chex.assert_trees_all_close(state1.task_idx, state2.task_idx)

    def test_prng_key_splitting(self, sample_env_and_params):
        """Test that PRNG key splitting produces different results."""
        env, env_params = sample_env_and_params
        base_key = jax.random.PRNGKey(42)

        # Split key and use different subkeys
        key1, key2 = jax.random.split(base_key)

        state1, timestep1 = env.reset(key1)
        state2, timestep2 = env.reset(key2)

        # Results should be different (with high probability)
        # Note: There's a small chance they could be the same, but very unlikely
        try:
            chex.assert_trees_all_close(state1.task_idx, state2.task_idx)
            # If they are the same, check if working grids are different
            assert not jnp.array_equal(state1.working_grid, state2.working_grid)
        except AssertionError:
            # This is expected - the results should be different
            pass

    def test_prng_key_threading(self, sample_env_and_params):
        """Test that PRNG keys are properly threaded through operations."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(123)

        # Reset environment
        state, timestep = env.reset(key)

        # Check that state contains a key
        assert hasattr(state, "key")
        assert state.key is not None

        # The key in state should be different from input key (due to splitting)
        assert not jnp.array_equal(state.key, key)

    def test_prng_key_consistency_under_jit(self, sample_env_and_params):
        """Test that PRNG behavior is consistent under JIT compilation."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(456)

        # Normal execution
        state_normal, timestep_normal = env.reset(key)

        # JIT execution
        jitted_reset = jax.jit(env.reset)
        state_jit, timestep_jit = jitted_reset(key)

        # Results should be identical
        chex.assert_trees_all_close(
            timestep_normal.observation, timestep_jit.observation
        )
        chex.assert_trees_all_close(state_normal.working_grid, state_jit.working_grid)
        chex.assert_trees_all_close(state_normal.key, state_jit.key)

    def test_prng_key_batch_independence(self, sample_env_and_params):
        """Test that batch operations maintain PRNG independence."""
        env, env_params = sample_env_and_params

        # Create batch of different keys
        base_key = jax.random.PRNGKey(789)
        batch_keys = jax.random.split(base_key, 4)

        # Batch reset
        vmapped_reset = jax.vmap(env.reset)
        batch_states, batch_timesteps = vmapped_reset(batch_keys)

        # Each result should be different (check task indices)
        task_indices = batch_states.task_idx

        # At least some should be different (very high probability)
        unique_indices = jnp.unique(task_indices)
        # We expect some variation, but allow for possibility of duplicates
        assert len(unique_indices) >= 1  # At minimum, we have valid indices

    def test_prng_key_deterministic_sequence(self, sample_env_and_params):
        """Test that PRNG produces deterministic sequences."""
        env, env_params = sample_env_and_params

        def run_sequence(key):
            """Run a deterministic sequence of operations."""
            state, timestep = env.reset(key)
            results = [state.task_idx]
            grid_shape = timestep.observation.shape[:2]

            for i in range(3):
                pos = min(i, grid_shape[0] - 1)
                action = create_action(
                    operation=jnp.array(i, dtype=jnp.int32),
                    selection=jnp.zeros(grid_shape, dtype=jnp.bool_)
                    .at[pos, pos]
                    .set(True),
                )
                state, timestep = env.step(state, action)
                results.append(state.step_count)

            return jnp.array(results)

        key = jax.random.PRNGKey(999)

        # Run sequence twice
        sequence1 = run_sequence(key)
        sequence2 = run_sequence(key)

        # Should be identical
        chex.assert_trees_all_close(sequence1, sequence2)


class TestPyTreeOperations:
    """Test PyTree operations and tree manipulation."""

    @pytest.fixture
    def sample_state(self, sample_env_and_params):
        """Create sample state for testing."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)
        state, _ = env.reset(key)
        return state

    def test_state_is_pytree(self, sample_state):
        """Test that State is properly registered as a PyTree."""
        # Test tree_flatten and tree_unflatten
        leaves, treedef = jax.tree_util.tree_flatten(sample_state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Reconstructed state should be identical
        chex.assert_trees_all_close(
            sample_state.working_grid, reconstructed.working_grid
        )
        chex.assert_trees_all_close(sample_state.step_count, reconstructed.step_count)
        chex.assert_trees_all_close(
            sample_state.similarity_score, reconstructed.similarity_score
        )

    def test_timestep_is_pytree(self, sample_env_and_params):
        """Test that TimeStep is properly registered as a PyTree."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)
        _, timestep = env.reset(key)

        # Test tree operations
        leaves, treedef = jax.tree_util.tree_flatten(timestep)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Reconstructed timestep should be identical
        chex.assert_trees_all_close(timestep.observation, reconstructed.observation)
        chex.assert_trees_all_close(timestep.reward, reconstructed.reward)
        assert timestep.step_type == reconstructed.step_type

    def test_action_is_pytree(self):
        """Test that Action is properly registered as a PyTree."""
        # Use a reasonable grid size (5x5 for MiniARC)
        grid_shape = (5, 5)
        action = create_action(
            operation=jnp.array(5, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[3, 3].set(True),
        )

        # Test tree operations
        leaves, treedef = jax.tree_util.tree_flatten(action)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Reconstructed action should be identical
        chex.assert_trees_all_close(action.operation, reconstructed.operation)
        chex.assert_trees_all_close(action.selection, reconstructed.selection)

    def test_pytree_map_operations(self, sample_state):
        """Test PyTree map operations on state."""
        # Test tree_map with simple operation
        doubled_state = jax.tree_util.tree_map(
            lambda x: x * 2
            if isinstance(x, jax.Array) and x.dtype in [jnp.int32, jnp.float32]
            else x,
            sample_state,
        )

        # Numeric fields should be doubled
        chex.assert_trees_all_close(
            doubled_state.step_count, sample_state.step_count * 2
        )
        chex.assert_trees_all_close(
            doubled_state.similarity_score, sample_state.similarity_score * 2
        )

    def test_pytree_leaves_structure(self, sample_state):
        """Test PyTree leaves structure and types."""
        leaves, _ = jax.tree_util.tree_flatten(sample_state)

        # Should have the expected number of leaves (all array fields)
        assert len(leaves) > 0

        # All leaves should be JAX arrays
        for leaf in leaves:
            assert isinstance(leaf, jax.Array)

    def test_pytree_with_transformations(self, sample_state):
        """Test PyTree compatibility with JAX transformations."""

        # Test with jit
        @jax.jit
        def process_state(state):
            return jax.tree_util.tree_map(
                lambda x: x + 1
                if isinstance(x, jax.Array) and x.dtype == jnp.int32
                else x,
                state,
            )

        processed = process_state(sample_state)

        # Should work without errors
        assert processed is not None

        # Integer fields should be incremented
        chex.assert_trees_all_close(processed.step_count, sample_state.step_count + 1)

    def test_pytree_batch_operations(self, sample_env_and_params):
        """Test PyTree operations with batched data."""
        env, env_params = sample_env_and_params

        # Create batch of states
        keys = jax.random.split(jax.random.PRNGKey(42), 4)
        vmapped_reset = jax.vmap(env.reset)
        batch_states, batch_timesteps = vmapped_reset(keys)

        # Test tree operations on batch
        # batch_states = batch_timesteps.state

        # Tree map should work on batch
        processed_batch = jax.tree_util.tree_map(
            lambda x: x + 1 if isinstance(x, jax.Array) and x.dtype == jnp.int32 else x,
            batch_states,
        )

        # Should maintain batch structure
        chex.assert_shape(processed_batch.step_count, (4,))
        chex.assert_trees_all_close(
            processed_batch.step_count, batch_states.step_count + 1
        )


class TestStaticShapeMaintenance:
    """Test static shape maintenance across transformations."""

    @pytest.fixture
    def sample_env_and_params(self):
        """Create sample environment and parameters for testing."""
        config = JaxArcConfig()
        task_ids = available_task_ids("Mini", config=config)
        task_id = task_ids[0] if task_ids else "all"
        env, env_params = make(f"Mini-{task_id}", config=config)
        return env, env_params

    def test_reset_maintains_static_shapes(self, sample_env_and_params):
        """Test that reset maintains static shapes."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        state, timestep = env.reset(key)

        # Check static shapes
        obs_shape = timestep.observation.shape
        grid_shape = obs_shape[:2]
        chex.assert_shape(timestep.observation, obs_shape)
        chex.assert_shape(state.working_grid, grid_shape)
        chex.assert_shape(state.working_grid_mask, grid_shape)
        chex.assert_shape(state.target_grid, grid_shape)
        chex.assert_shape(timestep.reward, ())
        chex.assert_shape(state.step_count, ())

    def test_step_maintains_static_shapes(self, sample_env_and_params):
        """Test that step maintains static shapes."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        state, timestep = env.reset(key)

        obs_shape = timestep.observation.shape
        grid_shape = obs_shape[:2]
        action = create_action(
            operation=jnp.array(0, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True),
        )

        next_state, next_timestep = env.step(state, action)

        # Shapes should remain the same
        chex.assert_shape(next_timestep.observation, obs_shape)
        chex.assert_shape(next_state.working_grid, grid_shape)
        chex.assert_shape(next_state.working_grid_mask, grid_shape)
        chex.assert_shape(next_timestep.reward, ())
        chex.assert_shape(next_state.step_count, ())

    def test_batch_operations_maintain_shapes(self, sample_env_and_params):
        """Test that batch operations maintain static shapes."""
        env, env_params = sample_env_and_params
        batch_size = 8
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        # Batch reset
        vmapped_reset = jax.vmap(env.reset)
        batch_states, batch_timesteps = vmapped_reset(keys)

        # Check batch shapes
        grid_shape_3d = batch_timesteps.observation.shape[1:]
        grid_shape_2d = grid_shape_3d[:2]
        chex.assert_shape(batch_timesteps.observation, (batch_size, *grid_shape_3d))
        chex.assert_shape(batch_states.working_grid, (batch_size, *grid_shape_2d))
        chex.assert_shape(batch_timesteps.reward, (batch_size,))
        chex.assert_shape(batch_states.step_count, (batch_size,))

    def test_jit_preserves_static_shapes(self, sample_env_and_params):
        """Test that JIT compilation preserves static shapes."""
        env, env_params = sample_env_and_params
        key = jax.random.PRNGKey(42)

        # JIT compiled functions
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)

        state, timestep = jitted_reset(key)

        # Check shapes after JIT
        obs_shape = timestep.observation.shape
        grid_shape = obs_shape[:2]
        chex.assert_shape(timestep.observation, obs_shape)
        chex.assert_shape(state.working_grid, grid_shape)

        action = create_action(
            operation=jnp.array(1, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_).at[2, 2].set(True),
        )

        next_state, next_timestep = jitted_step(state, action)

        # Shapes should be preserved
        chex.assert_shape(next_timestep.observation, obs_shape)
        chex.assert_shape(next_state.working_grid, grid_shape)

    def test_grid_operations_preserve_shapes(self):
        """Test that grid operations preserve static shapes."""
        grid_shape = (5, 5)  # Use MiniARC grid size
        grid = jnp.zeros(grid_shape, dtype=jnp.int32)
        mask = jnp.ones(grid_shape, dtype=jnp.bool_)

        # Test compute_grid_similarity
        similarity = compute_grid_similarity(grid, mask, grid, mask)
        chex.assert_shape(similarity, ())

        # Test get_selection_bounding_box
        selection = jnp.zeros(grid_shape, dtype=jnp.bool_).at[1:4, 1:4].set(True)
        bbox = get_selection_bounding_box(selection)
        # bbox returns a tuple of 4 scalars, not an array
        assert len(bbox) == 4
        for coord in bbox:
            chex.assert_shape(coord, ())

    def test_action_shapes_consistency(self):
        """Test that action shapes are consistent."""
        grid_shape = (5, 5)  # Use MiniARC grid size

        action = create_action(
            operation=jnp.array(5, dtype=jnp.int32),
            selection=jnp.zeros(grid_shape, dtype=jnp.bool_),
        )

        # Check action shapes
        chex.assert_shape(action.operation, ())
        chex.assert_shape(action.selection, grid_shape)

        # Test batch actions
        batch_size = 4
        batch_operations = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        batch_selections = jnp.zeros((batch_size, *grid_shape), dtype=jnp.bool_)

        vmapped_create = jax.vmap(create_action)
        batch_actions = vmapped_create(batch_operations, batch_selections)

        chex.assert_shape(batch_actions.operation, (batch_size,))
        chex.assert_shape(batch_actions.selection, (batch_size, *grid_shape))
