"""
Performance benchmarks for Equinox state management migration.

This module benchmarks the performance of Equinox patterns vs traditional patterns
to ensure the migration maintains or improves performance.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.factory import create_standard_config
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.envs.grid_operations import fill_color


class TestEquinoxPerformance:
    """Performance benchmarks for Equinox migration."""

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

    def test_jit_compilation_performance(self, demo_state):
        """Test that JIT compilation works efficiently with Equinox patterns."""

        @jax.jit
        def equinox_update(state):
            return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

        @jax.jit
        def replace_update(state):
            return state.replace(step_count=state.step_count + 1)

        # Warm up JIT compilation
        for _ in range(10):
            equinox_update(demo_state)
            replace_update(demo_state)

        # Time JIT-compiled execution
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
            f"JIT Performance: eqx.tree_at={equinox_time:.4f}s, replace={replace_time:.4f}s"
        )

        # Both should be very fast after JIT compilation
        assert equinox_time < 1.0, f"Equinox JIT too slow: {equinox_time}s"
        assert replace_time < 1.0, f"Replace JIT too slow: {replace_time}s"

    def test_grid_operations_performance(self, demo_state):
        """Test that grid operations maintain good performance with Equinox."""

        selection = jnp.zeros_like(demo_state.working_grid, dtype=jnp.bool_)
        selection = selection.at[0:5, 0:5].set(True)

        # Warm up JIT
        for _ in range(10):
            fill_color(demo_state, selection, 3)

        # Time grid operations
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            fill_color(demo_state, selection, 3)
        grid_ops_time = time.time() - start

        print(
            f"Grid operations performance: {grid_ops_time:.4f}s for {iterations} operations"
        )

        # Should be fast (less than 1 second for 1000 operations)
        assert grid_ops_time < 1.0, f"Grid operations too slow: {grid_ops_time}s"

    def test_functional_api_performance(self, config, key):
        """Test that the functional API maintains good performance."""

        # Warm up
        for _ in range(5):
            state, _ = arc_reset(key, config)
            action = {
                "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                .at[0:2, 0:2]
                .set(True),
                "operation": 1,
            }
            arc_step(state, action, config)

        # Time full reset-step cycle
        iterations = 100  # Fewer iterations since this is more expensive
        start = time.time()
        for i in range(iterations):
            state, _ = arc_reset(jax.random.PRNGKey(i), config)
            action = {
                "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                .at[0:2, 0:2]
                .set(True),
                "operation": 1,
            }
            arc_step(state, action, config)
        api_time = time.time() - start

        print(
            f"Functional API performance: {api_time:.4f}s for {iterations} reset-step cycles"
        )

        # Should complete reasonably quickly (less than 10 seconds for 100 cycles)
        assert api_time < 10.0, f"Functional API too slow: {api_time}s"

    def test_memory_efficiency(self, demo_state):
        """Test that Equinox patterns don't create excessive memory overhead."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create many state updates
        states = []
        for i in range(100):
            new_state = eqx.tree_at(
                lambda s: s.step_count, demo_state, demo_state.step_count + i
            )
            states.append(new_state)

        # Force garbage collection
        gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage: initial={initial_memory:.1f}MB, final={final_memory:.1f}MB, increase={memory_increase:.1f}MB"
        )

        # Memory increase should be reasonable (less than 100MB for 100 states)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase}MB"

    def test_batch_operations_performance(self, demo_state):
        """Test performance of batch operations with Equinox patterns."""

        # Create batch of step counts
        batch_step_counts = jnp.arange(100)

        @jax.jit
        @jax.vmap
        def batch_increment(step_count):
            return step_count + 1

        # Warm up
        for _ in range(10):
            batch_increment(batch_step_counts)

        # Time batch operations
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            batch_increment(batch_step_counts)
        batch_time = time.time() - start

        print(
            f"Batch operations performance: {batch_time:.4f}s for {iterations} batch operations"
        )

        # Should be very fast due to vectorization
        assert batch_time < 1.0, f"Batch operations too slow: {batch_time}s"

    def test_complex_state_updates_performance(self, demo_state):
        """Test performance of complex multi-field state updates."""

        @jax.jit
        def complex_equinox_update(state):
            # Update multiple fields using Equinox
            return eqx.tree_at(
                lambda s: (s.step_count, s.episode_done, s.similarity_score),
                state,
                (state.step_count + 1, True, state.similarity_score + 0.1),
            )

        @jax.jit
        def complex_replace_update(state):
            # Update multiple fields using replace
            return state.replace(
                step_count=state.step_count + 1,
                episode_done=True,
                similarity_score=state.similarity_score + 0.1,
            )

        # Warm up
        for _ in range(10):
            complex_equinox_update(demo_state)
            complex_replace_update(demo_state)

        # Time complex updates
        iterations = 1000

        # Time Equinox approach
        start = time.time()
        for _ in range(iterations):
            complex_equinox_update(demo_state)
        equinox_time = time.time() - start

        # Time replace approach
        start = time.time()
        for _ in range(iterations):
            complex_replace_update(demo_state)
        replace_time = time.time() - start

        print(
            f"Complex updates: eqx.tree_at={equinox_time:.4f}s, replace={replace_time:.4f}s"
        )

        # Both should be reasonably fast
        assert equinox_time < 2.0, f"Complex Equinox updates too slow: {equinox_time}s"
        assert replace_time < 2.0, f"Complex replace updates too slow: {replace_time}s"

    def test_compilation_time(self, demo_state):
        """Test that JIT compilation time is reasonable for Equinox patterns."""

        def new_equinox_function(state):
            # A function that hasn't been compiled yet
            return eqx.tree_at(
                lambda s: (s.step_count, s.similarity_score),
                state,
                (state.step_count * 2, state.similarity_score * 1.5),
            )

        # Time first compilation
        start = time.time()
        jitted_fn = jax.jit(new_equinox_function)
        result = jitted_fn(demo_state)  # This triggers compilation
        compilation_time = time.time() - start

        print(f"JIT compilation time: {compilation_time:.4f}s")

        # Compilation should be reasonable (less than 10 seconds)
        assert compilation_time < 10.0, f"JIT compilation too slow: {compilation_time}s"

        # Verify the result is correct
        assert result.step_count == demo_state.step_count * 2
        assert abs(result.similarity_score - demo_state.similarity_score * 1.5) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
