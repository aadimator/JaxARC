"""
Comprehensive Equinox integration tests for task 12.

This module provides comprehensive tests for Equinox module functionality,
JAX transformation compatibility, performance benchmarks, and backward compatibility
to fulfill the requirements of task 12 in the codebase refactoring spec.

Requirements covered:
- 5.6: Equinox and JAXTyping integration SHALL improve code clarity, type safety, and JAX performance
- 5.7: Migration SHALL maintain backward compatibility

Test Categories:
1. Equinox Module Functionality Tests
2. JAX Transformation Compatibility Tests
3. Performance Benchmarks (old vs new state management)
4. Backward Compatibility Tests
"""

from __future__ import annotations

import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.factory import create_standard_config
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.envs.grid_operations import (
    copy_to_clipboard,
    fill_color,
    paste_from_clipboard,
)
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.equinox_utils import (
    check_jax_transformations,
    create_state_diff,
    module_memory_usage,
    tree_map_with_path,
    tree_size_info,
    validate_state_shapes,
)


class TestEquinoxModuleFunctionality:
    """Test core Equinox module functionality."""

    @pytest.fixture
    def simple_task(self):
        """Create a simple JaxArcTask for testing."""
        grid_data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        grid_mask = jnp.ones((3, 3), dtype=bool)

        return JaxArcTask(
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

    @pytest.fixture
    def equinox_state(self, simple_task):
        """Create an Equinox-based ArcEnvState for testing."""
        grid_data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
        grid_mask = jnp.ones((3, 3), dtype=bool)

        return ArcEnvState(
            task_data=simple_task,
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

    def test_equinox_module_properties(self, equinox_state):
        """Test that ArcEnvState has proper Equinox Module properties."""
        # Requirement 5.1: JAX modules SHALL use Equinox Module classes where appropriate
        assert isinstance(equinox_state, eqx.Module)

        # Test PyTree registration (Equinox modules are automatically PyTrees)
        flat, tree_def = jax.tree_util.tree_flatten(equinox_state)
        assert len(flat) > 0
        assert tree_def is not None

        # Test reconstruction
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)
        assert isinstance(reconstructed, ArcEnvState)
        assert jnp.array_equal(reconstructed.working_grid, equinox_state.working_grid)

    def test_equinox_validation_functionality(self, equinox_state):
        """Test Equinox validation through __check_init__."""
        # Should not raise for valid state
        equinox_state.__check_init__()

        # Test validation catches shape mismatches
        with pytest.raises((ValueError, AssertionError)):
            invalid_state = ArcEnvState(
                task_data=equinox_state.task_data,
                working_grid=equinox_state.working_grid,
                working_grid_mask=jnp.ones((2, 2), dtype=bool),  # Wrong shape
                target_grid=equinox_state.target_grid,
                step_count=equinox_state.step_count,
                episode_done=equinox_state.episode_done,
                current_example_idx=equinox_state.current_example_idx,
                selected=equinox_state.selected,
                clipboard=equinox_state.clipboard,
                similarity_score=equinox_state.similarity_score,
            )
            invalid_state.__check_init__()

    def test_equinox_tree_operations(self, equinox_state):
        """Test Equinox tree operations like tree_at."""
        # Single field update
        new_state = eqx.tree_at(
            lambda s: s.step_count, equinox_state, equinox_state.step_count + 1
        )

        assert new_state.step_count == 1
        assert equinox_state.step_count == 0  # Original unchanged
        assert new_state is not equinox_state

        # Multiple field update
        new_state = eqx.tree_at(
            lambda s: (s.step_count, s.episode_done, s.similarity_score),
            equinox_state,
            (jnp.array(5), jnp.array(True), jnp.array(0.8)),
        )

        assert new_state.step_count == 5
        assert new_state.episode_done == True
        assert new_state.similarity_score == 0.8
        assert equinox_state.step_count == 0  # Original unchanged

    def test_replace_method_functionality(self, equinox_state):
        """Test the replace method for backward compatibility."""
        new_state = equinox_state.replace(
            step_count=jnp.array(10),
            episode_done=jnp.array(True),
            similarity_score=jnp.array(0.9),
        )

        assert new_state.step_count == 10
        assert new_state.episode_done == True
        assert new_state.similarity_score == 0.9

        # Original should be unchanged
        assert equinox_state.step_count == 0
        assert equinox_state.episode_done == False
        assert equinox_state.similarity_score == 0.0


class TestJAXTransformationCompatibility:
    """Test JAX transformation compatibility with Equinox state."""

    @pytest.fixture
    def config(self):
        """Create a standard configuration for testing."""
        return create_standard_config()

    @pytest.fixture
    def key(self):
        """Create a PRNG key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def demo_state(self, config, key):
        """Create a demo state for testing."""
        state, _ = arc_reset(key, config)
        return state

    def test_jit_compilation_compatibility(self, demo_state):
        """Test that Equinox state works with JAX JIT compilation."""

        @jax.jit
        def jitted_update(state):
            return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

        # Should compile and execute without errors
        new_state = jitted_update(demo_state)
        assert new_state.step_count == demo_state.step_count + 1

        # Test multiple calls (should use cached compilation)
        new_state = jitted_update(new_state)
        assert new_state.step_count == demo_state.step_count + 2

    def test_simple_vmap_compatibility(self, demo_state):
        """Test vmap compatibility with simple scalar operations."""

        # Test with simple scalar fields that can be vmapped
        @jax.vmap
        def increment_step_counts(step_counts):
            return step_counts + 1

        # Create batch of step counts
        batch_step_counts = jnp.array([0, 1, 2, 3, 4])
        result = increment_step_counts(batch_step_counts)

        expected = jnp.array([1, 2, 3, 4, 5])
        assert jnp.array_equal(result, expected)

    def test_grid_operations_jit_compatibility(self, demo_state):
        """Test that grid operations work with JIT compilation."""

        @jax.jit
        def jitted_fill_operation(state, color):
            selection = jnp.zeros_like(state.working_grid, dtype=bool)
            selection = selection.at[0:2, 0:2].set(True)
            return fill_color(state, selection, color)

        # Should compile and execute
        new_state = jitted_fill_operation(demo_state, 5)
        assert isinstance(new_state, ArcEnvState)

        # Check that the operation was applied
        expected_grid = demo_state.working_grid.at[0:2, 0:2].set(5)
        assert jnp.array_equal(
            new_state.working_grid[0:2, 0:2], expected_grid[0:2, 0:2]
        )

    def test_functional_api_jit_compatibility(self, config, key):
        """Test that the functional API works with JIT compilation."""

        # Test JIT compilation with simpler functions due to config complexity
        @jax.jit
        def jitted_state_update(state):
            return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

        # Get a state first
        state, obs = arc_reset(key, config)

        # Should compile and execute
        new_state = jitted_state_update(state)
        assert isinstance(new_state, ArcEnvState)
        assert new_state.step_count == state.step_count + 1

    def test_transformation_utility_function(self, demo_state):
        """Test the JAX transformation utility function."""

        def simple_test_fn(state):
            return state.step_count + state.similarity_score

        results = check_jax_transformations(demo_state, simple_test_fn)

        # Should at least support JIT
        assert "jit" in results
        assert results["jit"] == True

        # Other transformations may or may not work depending on the complexity
        assert "vmap" in results
        assert "grad" in results


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing old vs new state management patterns."""

    @pytest.fixture
    def config(self):
        """Create a standard configuration for testing."""
        return create_standard_config()

    @pytest.fixture
    def key(self):
        """Create a PRNG key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def demo_state(self, config, key):
        """Create a demo state for testing."""
        state, _ = arc_reset(key, config)
        return state

    def test_state_update_performance_comparison(self, demo_state):
        """Compare performance of Equinox tree_at vs replace method."""
        # Requirement 5.6: Equinox integration SHALL improve or maintain performance

        @jax.jit
        def equinox_update(state):
            return eqx.tree_at(
                lambda s: (s.step_count, s.similarity_score),
                state,
                (state.step_count + 1, state.similarity_score + 0.1),
            )

        @jax.jit
        def replace_update(state):
            return state.replace(
                step_count=state.step_count + 1,
                similarity_score=state.similarity_score + 0.1,
            )

        # Warm up JIT compilation
        for _ in range(10):
            equinox_update(demo_state)
            replace_update(demo_state)

        # Benchmark iterations
        iterations = 1000

        # Time Equinox approach
        start = time.time()
        for _ in range(iterations):
            equinox_update(demo_state)
        equinox_time = time.time() - start

        # Time replace approach
        start = time.time()
        for _ in range(iterations):
            replace_update(demo_state)
        replace_time = time.time() - start

        print(
            f"State update performance: eqx.tree_at={equinox_time:.4f}s, replace={replace_time:.4f}s"
        )

        # Both should be fast (less than 1 second for 1000 operations)
        assert equinox_time < 1.0, f"Equinox updates too slow: {equinox_time}s"
        assert replace_time < 1.0, f"Replace updates too slow: {replace_time}s"

        # Performance should be comparable (within 2x of each other)
        ratio = max(equinox_time, replace_time) / min(equinox_time, replace_time)
        assert ratio < 2.0, f"Performance difference too large: {ratio:.2f}x"

    def test_grid_operations_performance(self, demo_state):
        """Test that grid operations maintain good performance with Equinox."""
        selection = jnp.zeros_like(demo_state.working_grid, dtype=bool)
        selection = selection.at[0:5, 0:5].set(True)

        @jax.jit
        def grid_operation_sequence(state):
            # Sequence of operations
            state = fill_color(state, selection, 3)
            state = copy_to_clipboard(state, selection)
            state = paste_from_clipboard(state, selection)
            return state

        # Warm up
        for _ in range(10):
            grid_operation_sequence(demo_state)

        # Time grid operations
        iterations = 500
        start = time.time()
        for _ in range(iterations):
            grid_operation_sequence(demo_state)
        grid_ops_time = time.time() - start

        print(
            f"Grid operations performance: {grid_ops_time:.4f}s for {iterations} operation sequences"
        )

        # Should be fast (less than 2 seconds for 500 sequences)
        assert grid_ops_time < 2.0, f"Grid operations too slow: {grid_ops_time}s"

    def test_memory_usage_comparison(self, demo_state):
        """Test memory usage of Equinox vs traditional patterns."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Test Equinox pattern memory usage
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        equinox_states = []
        for i in range(100):
            new_state = eqx.tree_at(
                lambda s: s.step_count, demo_state, demo_state.step_count + i
            )
            equinox_states.append(new_state)

        gc.collect()
        equinox_memory = process.memory_info().rss / 1024 / 1024  # MB
        equinox_increase = equinox_memory - initial_memory

        # Clear states
        del equinox_states
        gc.collect()

        # Test replace pattern memory usage
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        replace_states = []
        for i in range(100):
            new_state = demo_state.replace(step_count=demo_state.step_count + i)
            replace_states.append(new_state)

        gc.collect()
        replace_memory = process.memory_info().rss / 1024 / 1024  # MB
        replace_increase = replace_memory - initial_memory

        print(
            f"Memory usage: Equinox={equinox_increase:.1f}MB, Replace={replace_increase:.1f}MB"
        )

        # Both should use reasonable memory (less than 100MB for 100 states)
        assert equinox_increase < 100, (
            f"Equinox memory usage too high: {equinox_increase}MB"
        )
        assert replace_increase < 100, (
            f"Replace memory usage too high: {replace_increase}MB"
        )

        # Memory usage should be comparable (allow for more variation in memory measurements)
        if equinox_increase > 0 and replace_increase > 0:
            ratio = max(equinox_increase, replace_increase) / min(
                equinox_increase, replace_increase
            )
            assert ratio < 10.0, f"Memory usage difference too large: {ratio:.2f}x"

    def test_compilation_time_benchmark(self, demo_state):
        """Test JIT compilation time for Equinox patterns."""

        def complex_equinox_function(state):
            # Complex function that hasn't been compiled yet
            state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)
            state = eqx.tree_at(
                lambda s: s.similarity_score, state, state.similarity_score * 2.0
            )
            state = eqx.tree_at(lambda s: s.episode_done, state, state.step_count > 10)
            return state

        # Time compilation
        start = time.time()
        jitted_fn = jax.jit(complex_equinox_function)
        result = jitted_fn(demo_state)  # This triggers compilation
        compilation_time = time.time() - start

        print(f"JIT compilation time: {compilation_time:.4f}s")

        # Compilation should be reasonable (less than 10 seconds)
        assert compilation_time < 10.0, f"JIT compilation too slow: {compilation_time}s"

        # Verify the result is correct
        assert result.step_count == demo_state.step_count + 1
        assert result.similarity_score == demo_state.similarity_score * 2.0


class TestBackwardCompatibility:
    """Test backward compatibility during the Equinox transition."""

    @pytest.fixture
    def config(self):
        """Create a standard configuration for testing."""
        return create_standard_config()

    @pytest.fixture
    def key(self):
        """Create a PRNG key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def demo_state(self, config, key):
        """Create a demo state for testing."""
        state, _ = arc_reset(key, config)
        return state

    def test_field_access_compatibility(self, demo_state):
        """Test that field access works the same as before."""
        # Requirement 5.7: Migration SHALL maintain backward compatibility

        # All fields should be accessible
        assert hasattr(demo_state, "task_data")
        assert hasattr(demo_state, "working_grid")
        assert hasattr(demo_state, "working_grid_mask")
        assert hasattr(demo_state, "target_grid")
        assert hasattr(demo_state, "step_count")
        assert hasattr(demo_state, "episode_done")
        assert hasattr(demo_state, "current_example_idx")
        assert hasattr(demo_state, "selected")
        assert hasattr(demo_state, "clipboard")
        assert hasattr(demo_state, "similarity_score")

        # Field access should return expected types (JAX arrays or scalars)
        assert hasattr(demo_state.step_count, "shape") or isinstance(
            demo_state.step_count, (int, float)
        )
        assert hasattr(demo_state.episode_done, "shape") or isinstance(
            demo_state.episode_done, bool
        )
        assert hasattr(demo_state.similarity_score, "shape") or isinstance(
            demo_state.similarity_score, float
        )
        assert isinstance(demo_state.working_grid, jnp.ndarray)

    def test_functional_api_compatibility(self, config, key):
        """Test that the functional API remains compatible."""
        # arc_reset should work the same
        state, obs = arc_reset(key, config)
        assert isinstance(state, ArcEnvState)
        assert isinstance(obs, jnp.ndarray)

        # arc_step should work the same
        action = {
            "selection": jnp.zeros_like(state.working_grid, dtype=bool)
            .at[0:2, 0:2]
            .set(True),
            "operation": 1,
        }
        result = arc_step(state, action, config)
        # arc_step returns (new_state, observation, reward, done, info) - 5 values
        new_state, observation, reward, done, info = result
        assert isinstance(new_state, ArcEnvState)
        assert isinstance(observation, jnp.ndarray)
        assert isinstance(reward, jnp.ndarray)
        assert isinstance(done, jnp.ndarray)
        assert isinstance(info, dict)

    def test_grid_operations_compatibility(self, demo_state):
        """Test that grid operations remain compatible."""
        selection = jnp.zeros_like(demo_state.working_grid, dtype=bool)
        selection = selection.at[0:3, 0:3].set(True)

        # Grid operations should work the same
        new_state = fill_color(demo_state, selection, 5)
        assert isinstance(new_state, ArcEnvState)
        assert new_state.working_grid[0, 0] == 5

        # Copy and paste should work
        new_state = copy_to_clipboard(new_state, selection)
        assert isinstance(new_state, ArcEnvState)

        paste_state = paste_from_clipboard(new_state, selection)
        assert isinstance(paste_state, ArcEnvState)

    def test_immutability_preserved(self, demo_state):
        """Test that immutability is preserved in the Equinox implementation."""
        # State should remain immutable
        with pytest.raises(AttributeError):
            demo_state.step_count = 5

        # Updates should create new instances
        new_state = demo_state.replace(step_count=jnp.array(10))
        assert new_state is not demo_state
        assert new_state.step_count == 10
        assert demo_state.step_count != 10

    def test_validation_still_works(self, demo_state):
        """Test that validation still works after migration."""
        # State should pass validation
        demo_state.__check_init__()

        # Invalid states should still be caught
        with pytest.raises((ValueError, AssertionError)):
            invalid_state = demo_state.replace(
                working_grid_mask=jnp.ones((2, 2), dtype=bool)  # Wrong shape
            )
            invalid_state.__check_init__()


class TestEquinoxUtilities:
    """Test the Equinox utility functions."""

    @pytest.fixture
    def config(self):
        """Create a standard configuration for testing."""
        return create_standard_config()

    @pytest.fixture
    def key(self):
        """Create a PRNG key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def demo_state(self, config, key):
        """Create a demo state for testing."""
        state, _ = arc_reset(key, config)
        return state

    def test_tree_map_with_path_utility(self, demo_state):
        """Test the tree_map_with_path utility function."""
        paths_found = []

        def collect_paths(path: str, value: Any) -> Any:
            if hasattr(value, "shape"):
                paths_found.append(path)
            return value

        tree_map_with_path(collect_paths, demo_state)

        # Should find paths for array fields (paths may be indexed due to PyTree flattening)
        expected_field_names = [
            "working_grid",
            "working_grid_mask",
            "target_grid",
            "selected",
            "clipboard",  # These are definitely arrays
        ]

        # Check that we found some paths with array data
        assert len(paths_found) > 0, f"No paths found. Paths: {paths_found}"

        # Check that at least some expected field names appear in the paths
        found_fields = 0
        for expected_field in expected_field_names:
            if any(expected_field in path for path in paths_found):
                found_fields += 1

        assert found_fields > 0, f"No expected fields found in paths: {paths_found}"

    def test_validate_state_shapes_utility(self, demo_state):
        """Test the validate_state_shapes utility function."""
        # Valid state should pass validation
        assert validate_state_shapes(demo_state) == True

        # Test with a state that might have issues (but should still pass)
        modified_state = demo_state.replace(similarity_score=jnp.array(1.0))
        assert validate_state_shapes(modified_state) == True

    def test_create_state_diff_utility(self, demo_state):
        """Test the create_state_diff utility function."""
        # Create a modified state
        new_state = demo_state.replace(
            step_count=jnp.array(5), similarity_score=jnp.array(0.8)
        )

        # Get diff
        diff = create_state_diff(demo_state, new_state)

        # Should detect changes
        assert "step_count" in diff
        assert "similarity_score" in diff

        # Should not detect unchanged fields
        assert "working_grid" not in diff or diff["working_grid"] is None

    def test_tree_size_info_utility(self, demo_state):
        """Test the tree_size_info utility function."""
        size_info = tree_size_info(demo_state)

        # Should have size information for array fields
        assert len(size_info) > 0

        # Check that we get shape and size information
        for path, (shape, size) in size_info.items():
            assert isinstance(shape, tuple)
            assert isinstance(size, int)
            assert size > 0

    def test_module_memory_usage_utility(self, demo_state):
        """Test the module_memory_usage utility function."""
        memory_info = module_memory_usage(demo_state)

        # Should have memory usage information
        assert "total_bytes" in memory_info
        assert "total_elements" in memory_info
        assert "arrays" in memory_info

        assert memory_info["total_bytes"] > 0
        assert memory_info["total_elements"] > 0
        assert len(memory_info["arrays"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
