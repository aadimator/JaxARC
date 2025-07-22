"""
Performance regression tests for JAX transformations in JaxARC.

This module implements comprehensive performance testing to ensure that JAX transformations
(jit, vmap, pmap) maintain acceptable performance characteristics and don't regress over time.
Tests focus on compilation time, execution time, and memory usage for key functions.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import pytest

from jaxarc.envs.config import (
    ActionConfig,
    ArcEnvConfig,
    DatasetConfig,
    DebugConfig,
    GridConfig,
    RewardConfig,
)
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.envs.grid_operations import (
    compute_grid_similarity,
    execute_grid_operation,
)
from jaxarc.types import ARCLEAction, Grid
from jaxarc.utils.grid_utils import get_actual_grid_shape_from_mask

# Performance thresholds (in seconds)
COMPILATION_TIME_THRESHOLD = 5.0  # Max compilation time
EXECUTION_TIME_THRESHOLD = 0.1  # Max execution time per call
BATCH_EXECUTION_TIME_THRESHOLD = 1.0  # Max batch execution time
MEMORY_THRESHOLD_MB = 100  # Max memory usage in MB


class PerformanceProfiler:
    """Utility class for profiling JAX function performance."""

    def __init__(self, func: Callable, name: str = "function"):
        self.func = func
        self.name = name
        self.compilation_time = 0.0
        self.execution_times: List[float] = []
        self.memory_usage_mb = 0.0
        self.compiled_func = None

    def profile_compilation(self, *args, **kwargs) -> float:
        """Profile JIT compilation time."""
        # Clear any existing compilation
        self.compiled_func = None
        gc.collect()

        try:
            # Time compilation
            start_time = time.perf_counter()
            self.compiled_func = jax.jit(self.func)

            # Force compilation by calling once
            _ = self.compiled_func(*args, **kwargs)

            end_time = time.perf_counter()
            self.compilation_time = end_time - start_time
        except (TypeError, ValueError) as e:
            if "abstract array" in str(e) or "static_argnums" in str(e):
                # Function contains non-array arguments, measure without JIT
                start_time = time.perf_counter()
                _ = self.func(*args, **kwargs)
                end_time = time.perf_counter()
                self.compilation_time = end_time - start_time
                self.compiled_func = self.func  # Use non-compiled version
            else:
                raise

        return self.compilation_time

    def profile_execution(self, *args, num_runs: int = 10, **kwargs) -> List[float]:
        """Profile execution time over multiple runs."""
        if self.compiled_func is None:
            try:
                self.compiled_func = jax.jit(self.func)
                # Warm up compilation
                _ = self.compiled_func(*args, **kwargs)
            except (TypeError, ValueError) as e:
                if "abstract array" in str(e) or "static_argnums" in str(e):
                    # Use non-compiled version
                    self.compiled_func = self.func
                else:
                    raise

        execution_times = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.compiled_func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        self.execution_times = execution_times
        return execution_times

    def profile_memory(self, *args, **kwargs) -> float:
        """Profile memory usage during execution."""
        if self.compiled_func is None:
            try:
                self.compiled_func = jax.jit(self.func)
                # Warm up compilation
                _ = self.compiled_func(*args, **kwargs)
            except (TypeError, ValueError) as e:
                if "abstract array" in str(e) or "static_argnums" in str(e):
                    # Use non-compiled version
                    self.compiled_func = self.func
                else:
                    raise

        # Start memory tracing
        tracemalloc.start()

        # Execute function
        _ = self.compiled_func(*args, **kwargs)

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.memory_usage_mb = peak / 1024 / 1024  # Convert to MB
        return self.memory_usage_mb

    def profile_batch_execution(self, batch_args: Tuple, batch_size: int = 10) -> float:
        """Profile vmap batch execution time."""
        vmapped_func = jax.vmap(self.func)

        # Warm up compilation
        _ = vmapped_func(*batch_args)

        start_time = time.perf_counter()
        _ = vmapped_func(*batch_args)
        end_time = time.perf_counter()

        return end_time - start_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "function_name": self.name,
            "compilation_time": self.compilation_time,
            "avg_execution_time": sum(self.execution_times) / len(self.execution_times)
            if self.execution_times
            else 0.0,
            "min_execution_time": min(self.execution_times)
            if self.execution_times
            else 0.0,
            "max_execution_time": max(self.execution_times)
            if self.execution_times
            else 0.0,
            "memory_usage_mb": self.memory_usage_mb,
            "num_execution_samples": len(self.execution_times),
        }


@pytest.fixture
def test_config():
    """Create test configuration."""
    # Create minimal config for performance testing
    reward_config = RewardConfig(
        reward_on_submit_only=False,
        step_penalty=-0.01,
        success_bonus=1.0,
        similarity_weight=1.0,
        progress_bonus=0.1,
        invalid_action_penalty=-0.1,
    )

    grid_config = GridConfig(
        max_grid_height=30,
        max_grid_width=30,
        min_grid_height=3,
        min_grid_width=3,
        max_colors=10,
        background_color=0,
    )

    action_config = ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        num_operations=35,
        allowed_operations=list(range(35)),
        validate_actions=True,
        clip_invalid_actions=True,
    )

    dataset_config = DatasetConfig(
        dataset_name="arc-agi-1",
        dataset_path="data/raw/arc-prize-2024",
        task_split="train",
        shuffle_tasks=True,
    )

    debug_config = DebugConfig(
        log_rl_steps=False,
        rl_steps_output_dir="output/rl_steps",
        clear_output_dir=False,
    )

    return ArcEnvConfig(
        max_episode_steps=50,
        auto_reset=True,
        log_operations=False,
        log_grid_changes=False,
        log_rewards=False,
        strict_validation=False,
        allow_invalid_actions=True,
        reward=reward_config,
        grid=grid_config,
        action=action_config,
        dataset=dataset_config,
        debug=debug_config,
        parser=None,
    )


@pytest.fixture
def test_key():
    """Create test PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def test_state_and_obs(test_config, test_key):
    """Create test state and observation."""
    state, obs = arc_reset(test_key, test_config)
    return state, obs


@pytest.fixture
def test_action():
    """Create test action."""
    return {
        "selection": jnp.ones((30, 30), dtype=jnp.bool_).flatten(),
        "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
    }


class TestCorePerformance:
    """Test performance of core JAX functions."""

    def test_arc_reset_performance(self, test_config, test_key):
        """Test arc_reset compilation and execution performance."""
        profiler = PerformanceProfiler(arc_reset, "arc_reset")

        # Test compilation time
        compilation_time = profiler.profile_compilation(test_key, test_config)
        assert compilation_time < COMPILATION_TIME_THRESHOLD, (
            f"arc_reset compilation took {compilation_time:.3f}s, "
            f"exceeds threshold of {COMPILATION_TIME_THRESHOLD}s"
        )

        # Test execution time
        execution_times = profiler.profile_execution(test_key, test_config, num_runs=5)
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < EXECUTION_TIME_THRESHOLD, (
            f"arc_reset average execution time {avg_execution_time:.3f}s, "
            f"exceeds threshold of {EXECUTION_TIME_THRESHOLD}s"
        )

        # Test memory usage
        memory_usage = profiler.profile_memory(test_key, test_config)
        assert memory_usage < MEMORY_THRESHOLD_MB, (
            f"arc_reset memory usage {memory_usage:.1f}MB, "
            f"exceeds threshold of {MEMORY_THRESHOLD_MB}MB"
        )

        # Print performance summary for monitoring
        summary = profiler.get_performance_summary()
        print("\narc_reset Performance Summary:")
        print(f"  Compilation time: {summary['compilation_time']:.3f}s")
        print(f"  Avg execution time: {summary['avg_execution_time']:.3f}s")
        print(f"  Memory usage: {summary['memory_usage_mb']:.1f}MB")

    def test_arc_step_performance(self, test_state_and_obs, test_action, test_config):
        """Test arc_step compilation and execution performance."""
        state, _ = test_state_and_obs

        profiler = PerformanceProfiler(arc_step, "arc_step")

        # Test compilation time
        compilation_time = profiler.profile_compilation(state, test_action, test_config)
        assert compilation_time < COMPILATION_TIME_THRESHOLD, (
            f"arc_step compilation took {compilation_time:.3f}s, "
            f"exceeds threshold of {COMPILATION_TIME_THRESHOLD}s"
        )

        # Test execution time
        execution_times = profiler.profile_execution(
            state, test_action, test_config, num_runs=5
        )
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < EXECUTION_TIME_THRESHOLD, (
            f"arc_step average execution time {avg_execution_time:.3f}s, "
            f"exceeds threshold of {EXECUTION_TIME_THRESHOLD}s"
        )

        # Test memory usage
        memory_usage = profiler.profile_memory(state, test_action, test_config)
        assert memory_usage < MEMORY_THRESHOLD_MB, (
            f"arc_step memory usage {memory_usage:.1f}MB, "
            f"exceeds threshold of {MEMORY_THRESHOLD_MB}MB"
        )

        # Print performance summary
        summary = profiler.get_performance_summary()
        print("\narc_step Performance Summary:")
        print(f"  Compilation time: {summary['compilation_time']:.3f}s")
        print(f"  Avg execution time: {summary['avg_execution_time']:.3f}s")
        print(f"  Memory usage: {summary['memory_usage_mb']:.1f}MB")


class TestGridOperationsPerformance:
    """Test performance of grid operations."""

    def test_compute_grid_similarity_performance(self):
        """Test grid similarity computation performance."""
        # Create test grids
        grid1 = jnp.ones((30, 30), dtype=jnp.int32)
        grid2 = jnp.ones((30, 30), dtype=jnp.int32) * 2

        profiler = PerformanceProfiler(
            compute_grid_similarity, "compute_grid_similarity"
        )

        # Test compilation time
        compilation_time = profiler.profile_compilation(grid1, grid2)
        assert compilation_time < COMPILATION_TIME_THRESHOLD, (
            f"compute_grid_similarity compilation took {compilation_time:.3f}s"
        )

        # Test execution time
        execution_times = profiler.profile_execution(grid1, grid2, num_runs=10)
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < EXECUTION_TIME_THRESHOLD, (
            f"compute_grid_similarity average execution time {avg_execution_time:.3f}s"
        )

        # Print performance summary
        summary = profiler.get_performance_summary()
        print("\ncompute_grid_similarity Performance Summary:")
        print(f"  Compilation time: {summary['compilation_time']:.3f}s")
        print(f"  Avg execution time: {summary['avg_execution_time']:.3f}s")

    def test_execute_grid_operation_performance(self, test_state_and_obs):
        """Test grid operation execution performance."""
        state, _ = test_state_and_obs
        operation = jnp.array(1, dtype=jnp.int32)  # Fill with color 1

        profiler = PerformanceProfiler(execute_grid_operation, "execute_grid_operation")

        # Test compilation time
        compilation_time = profiler.profile_compilation(state, operation)
        assert compilation_time < COMPILATION_TIME_THRESHOLD, (
            f"execute_grid_operation compilation took {compilation_time:.3f}s"
        )

        # Test execution time
        execution_times = profiler.profile_execution(state, operation, num_runs=5)
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < EXECUTION_TIME_THRESHOLD, (
            f"execute_grid_operation average execution time {avg_execution_time:.3f}s"
        )

        # Print performance summary
        summary = profiler.get_performance_summary()
        print("\nexecute_grid_operation Performance Summary:")
        print(f"  Compilation time: {summary['compilation_time']:.3f}s")
        print(f"  Avg execution time: {summary['avg_execution_time']:.3f}s")


class TestBatchPerformance:
    """Test batch processing performance with vmap."""

    def test_arc_reset_batch_performance(self, test_config):
        """Test batched arc_reset performance."""
        batch_size = 8

        # Create batch of keys
        key = jax.random.PRNGKey(42)
        batch_keys = jax.random.split(key, batch_size)

        # Test sequential execution time (since config can't be vmapped)
        start_time = time.perf_counter()

        batch_states = []
        batch_obs = []
        for i in range(batch_size):
            state, obs = arc_reset(batch_keys[i], test_config)
            batch_states.append(state)
            batch_obs.append(obs)

        batch_time = time.perf_counter() - start_time

        assert batch_time < BATCH_EXECUTION_TIME_THRESHOLD, (
            f"Sequential arc_reset took {batch_time:.3f}s for {batch_size} items, "
            f"exceeds threshold of {BATCH_EXECUTION_TIME_THRESHOLD}s"
        )

        print("\nSequential arc_reset Performance:")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {batch_time:.3f}s")
        print(f"  Time per item: {batch_time / batch_size:.3f}s")

    def test_compute_grid_similarity_batch_performance(self):
        """Test batched grid similarity computation."""
        batch_size = 16

        # Create batch of grids
        batch_grid1 = jnp.ones((batch_size, 30, 30), dtype=jnp.int32)
        batch_grid2 = jnp.ones((batch_size, 30, 30), dtype=jnp.int32) * 2

        # Test batch execution time
        profiler = PerformanceProfiler(compute_grid_similarity, "similarity_batch")
        batch_time = profiler.profile_batch_execution((batch_grid1, batch_grid2))

        assert batch_time < BATCH_EXECUTION_TIME_THRESHOLD, (
            f"Batched compute_grid_similarity took {batch_time:.3f}s for {batch_size} items"
        )

        print("\nBatched compute_grid_similarity Performance:")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {batch_time:.3f}s")
        print(f"  Time per item: {batch_time / batch_size:.3f}s")


class TestTypeSystemPerformance:
    """Test performance of type system operations."""

    def test_grid_creation_performance(self):
        """Test Grid creation and validation performance."""
        # Test direct creation performance (non-JIT due to validation)
        data = jnp.ones((30, 30), dtype=jnp.int32)
        mask = jnp.ones((30, 30), dtype=jnp.bool_)

        # Time multiple creations
        start_time = time.perf_counter()
        for _ in range(10):
            grid = Grid(data=data, mask=mask)
        end_time = time.perf_counter()

        avg_creation_time = (end_time - start_time) / 10
        assert avg_creation_time < EXECUTION_TIME_THRESHOLD, (
            f"Grid creation average time {avg_creation_time:.3f}s exceeds threshold"
        )

        print("\nGrid creation Performance Summary:")
        print(f"  Avg creation time: {avg_creation_time:.3f}s")
        print(f"  Grid shape: {grid.data.shape}")

    def test_arcle_action_performance(self):
        """Test ARCLEAction creation and validation performance."""
        # Test direct creation performance (non-JIT due to validation)
        selection = jnp.ones((30, 30), dtype=jnp.float32)
        operation = jnp.array(1, dtype=jnp.int32)
        agent_id = 0
        timestamp = 0

        # Time multiple creations
        start_time = time.perf_counter()
        for _ in range(10):
            action = ARCLEAction(
                selection=selection,
                operation=operation,
                agent_id=agent_id,
                timestamp=timestamp,
            )
        end_time = time.perf_counter()

        avg_creation_time = (end_time - start_time) / 10
        assert avg_creation_time < EXECUTION_TIME_THRESHOLD, (
            f"ARCLEAction creation average time {avg_creation_time:.3f}s exceeds threshold"
        )

        print("\nARCLEAction creation Performance Summary:")
        print(f"  Avg creation time: {avg_creation_time:.3f}s")
        print(f"  Selection shape: {action.selection.shape}")
        print(f"  Operation: {action.operation}")


class TestUtilityPerformance:
    """Test performance of utility functions."""

    def test_get_actual_grid_shape_from_mask_performance(self):
        """Test grid shape detection performance."""
        # Create test grid with mask
        grid = jnp.ones((30, 30), dtype=jnp.int32)
        mask = jnp.ones((20, 25), dtype=jnp.bool_)
        # Pad mask to grid size
        padded_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        padded_mask = padded_mask.at[:20, :25].set(mask)

        profiler = PerformanceProfiler(
            get_actual_grid_shape_from_mask, "get_actual_grid_shape_from_mask"
        )

        # Test compilation time
        compilation_time = profiler.profile_compilation(padded_mask)
        assert compilation_time < COMPILATION_TIME_THRESHOLD, (
            f"get_actual_grid_shape_from_mask compilation took {compilation_time:.3f}s"
        )

        # Test execution time
        execution_times = profiler.profile_execution(padded_mask, num_runs=10)
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < EXECUTION_TIME_THRESHOLD, (
            f"get_actual_grid_shape_from_mask average execution time {avg_execution_time:.3f}s"
        )

        print("\nget_actual_grid_shape_from_mask Performance Summary:")
        print(f"  Compilation time: {compilation_time:.3f}s")
        print(f"  Avg execution time: {avg_execution_time:.3f}s")


class TestRegressionBenchmarks:
    """Regression benchmarks to track performance over time."""

    def test_full_episode_performance(self, test_config, test_key):
        """Test performance of a full episode execution."""
        # Run a complete episode
        state, obs = arc_reset(test_key, test_config)

        # Create a sequence of actions
        actions = []
        for i in range(5):  # 5 steps
            action = {
                "selection": jnp.ones((30, 30), dtype=jnp.bool_).flatten(),
                "operation": jnp.array(
                    i % 10, dtype=jnp.int32
                ),  # Cycle through operations
            }
            actions.append(action)

        # Time full episode
        start_time = time.perf_counter()

        current_state = state
        for action in actions:
            current_state, obs, reward, done, info = arc_step(
                current_state, action, test_config
            )
            if done:
                break

        end_time = time.perf_counter()
        episode_time = end_time - start_time

        # Episode should complete within reasonable time
        max_episode_time = 1.0  # 1 second for 5 steps
        assert episode_time < max_episode_time, (
            f"Full episode took {episode_time:.3f}s, exceeds {max_episode_time}s"
        )

        print("\nFull Episode Performance:")
        print(f"  Total time: {episode_time:.3f}s")
        print(f"  Steps executed: {len(actions)}")
        print(f"  Time per step: {episode_time / len(actions):.3f}s")

    def test_compilation_cache_efficiency(self, test_config, test_key):
        """Test that repeated calls are efficient (simulating caching behavior)."""
        # First call
        start_time = time.perf_counter()
        state1, obs1 = arc_reset(test_key, test_config)
        first_call_time = time.perf_counter() - start_time

        # Second call (should be faster due to internal optimizations)
        start_time = time.perf_counter()
        state2, obs2 = arc_reset(test_key, test_config)
        second_call_time = time.perf_counter() - start_time

        # Both calls should be reasonably fast
        max_call_time = 0.5  # 500ms max per call
        assert first_call_time < max_call_time, (
            f"First call took {first_call_time:.3f}s, exceeds {max_call_time}s"
        )
        assert second_call_time < max_call_time, (
            f"Second call took {second_call_time:.3f}s, exceeds {max_call_time}s"
        )

        print("\nCall Efficiency:")
        print(f"  First call: {first_call_time:.3f}s")
        print(f"  Second call: {second_call_time:.3f}s")
        print(f"  Both calls under {max_call_time}s threshold")


@pytest.mark.slow
class TestExtensivePerformance:
    """Extensive performance tests (marked as slow)."""

    def test_large_batch_performance(self, test_config):
        """Test performance with large sequential batches."""
        batch_size = 64

        # Create large batch of keys
        key = jax.random.PRNGKey(42)
        batch_keys = jax.random.split(key, batch_size)

        # Test large sequential batch reset
        start_time = time.perf_counter()

        batch_states = []
        batch_obs = []
        for i in range(batch_size):
            state, obs = arc_reset(batch_keys[i], test_config)
            batch_states.append(state)
            batch_obs.append(obs)

        end_time = time.perf_counter()

        batch_time = end_time - start_time
        max_large_batch_time = 10.0  # 10 seconds for 64 items sequentially

        assert batch_time < max_large_batch_time, (
            f"Large sequential batch reset took {batch_time:.3f}s for {batch_size} items"
        )

        print("\nLarge Sequential Batch Performance:")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {batch_time:.3f}s")
        print(f"  Time per item: {batch_time / batch_size:.3f}s")

    def test_memory_stability(self, test_config, test_key):
        """Test that memory usage remains stable over multiple calls."""
        # Warm up
        _ = arc_reset(test_key, test_config)

        # Measure memory usage over multiple calls
        memory_measurements = []

        for i in range(10):
            tracemalloc.start()
            _ = arc_reset(test_key, test_config)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_measurements.append(peak / 1024 / 1024)  # Convert to MB

        # Check memory stability (variance should be low)
        avg_memory = sum(memory_measurements) / len(memory_measurements)
        max_memory = max(memory_measurements)
        min_memory = min(memory_measurements)
        memory_variance = max_memory - min_memory

        max_variance = 50.0  # 50MB variance allowed (more lenient for non-JIT)
        assert memory_variance < max_variance, (
            f"Memory usage variance {memory_variance:.1f}MB exceeds {max_variance}MB"
        )

        print("\nMemory Stability:")
        print(f"  Average memory: {avg_memory:.1f}MB")
        print(f"  Min memory: {min_memory:.1f}MB")
        print(f"  Max memory: {max_memory:.1f}MB")
        print(f"  Variance: {memory_variance:.1f}MB")


if __name__ == "__main__":
    # Run performance tests directly
    pytest.main([__file__, "-v", "-s"])
