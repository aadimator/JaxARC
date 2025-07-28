#!/usr/bin/env python3
"""
Batch Processing Performance Demo for JaxARC

This example demonstrates the batch processing capabilities of JaxARC using jax.vmap
for parallel environment execution. It shows how to efficiently process multiple
environments simultaneously and compares performance across different batch sizes.

Key Features Demonstrated:
- Batch environment reset and stepping
- PRNG key management for deterministic batch processing
- Performance scaling analysis
- Memory usage optimization
- Practical batch processing patterns

Usage:
    pixi run python examples/advanced/batch_processing_demo.py
"""

import time
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs import JaxArcConfig, EnvironmentConfig, UnifiedDatasetConfig, UnifiedActionConfig
from jaxarc.envs.functional import arc_reset, arc_step, batch_reset, batch_step
from jaxarc.envs.structured_actions import PointAction, BboxAction
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.jax_types import PRNGKey


def create_test_config() -> JaxArcConfig:
    """Create a test configuration."""
    return JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=100),
        dataset=UnifiedDatasetConfig(max_grid_height=30, max_grid_width=30),
        action=UnifiedActionConfig(selection_format="point")
    )


def demonstrate_basic_batch_processing():
    """Demonstrate basic batch processing operations."""
    logger.info("🔄 Basic Batch Processing Demo")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    batch_size = 8
    
    logger.info(f"  Creating batch of {batch_size} environments...")
    
    # Create batch of PRNG keys
    master_key = jax.random.PRNGKey(42)
    batch_keys = jax.random.split(master_key, batch_size)
    
    logger.info(f"  PRNG keys shape: {batch_keys.shape}")
    
    # Batch reset
    logger.info("  Performing batch reset...")
    start_time = time.perf_counter()
    batch_states, batch_obs = batch_reset(batch_keys, config, task)
    reset_time = time.perf_counter() - start_time
    
    logger.info(f"  Reset completed in {reset_time*1000:.2f}ms")
    logger.info(f"  Batch states shape: {batch_states.working_grid.shape}")
    logger.info(f"  Batch observations shape: {batch_obs.shape}")
    
    # Create batch actions
    logger.info("  Creating batch actions...")
    batch_actions = PointAction(
        operation=jnp.zeros(batch_size, dtype=jnp.int32),  # Fill operation
        row=jnp.array([5, 6, 7, 8, 9, 10, 11, 12]),  # Different rows
        col=jnp.array([5, 5, 5, 5, 5, 5, 5, 5])      # Same column
    )
    
    logger.info(f"  Action operations: {batch_actions.operation}")
    logger.info(f"  Action rows: {batch_actions.row}")
    logger.info(f"  Action cols: {batch_actions.col}")
    
    # Batch step
    logger.info("  Performing batch step...")
    start_time = time.perf_counter()
    new_states, new_obs, rewards, dones, infos = batch_step(
        batch_states, batch_actions, config
    )
    step_time = time.perf_counter() - start_time
    
    logger.info(f"  Step completed in {step_time*1000:.2f}ms")
    logger.info(f"  Rewards: {rewards}")
    logger.info(f"  Dones: {dones}")
    
    # Verify deterministic behavior
    logger.info("  Verifying deterministic behavior...")
    
    # Reset with same keys should produce identical results
    batch_states_2, batch_obs_2 = batch_reset(batch_keys, config, task)
    
    states_identical = jnp.allclose(
        batch_states.working_grid, 
        batch_states_2.working_grid
    )
    obs_identical = jnp.allclose(batch_obs, batch_obs_2)
    
    logger.info(f"  States identical: {states_identical}")
    logger.info(f"  Observations identical: {obs_identical}")
    
    return {
        'batch_size': batch_size,
        'reset_time': reset_time,
        'step_time': step_time,
        'deterministic': states_identical and obs_identical
    }


def analyze_batch_scaling():
    """Analyze how performance scales with batch size."""
    logger.info("📊 Batch Scaling Analysis")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = {}
    
    logger.info(f"  Testing batch sizes: {batch_sizes}")
    
    for batch_size in batch_sizes:
        logger.info(f"  Testing batch size {batch_size}...")
        
        # Create keys
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        
        # Warm up (important for JIT compilation)
        if batch_size == batch_sizes[0]:
            logger.info("    Warming up JIT compilation...")
            batch_reset(keys, config, task)
        
        # Benchmark reset
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple runs for accuracy
            batch_states, batch_obs = batch_reset(keys, config, task)
        reset_time = (time.perf_counter() - start_time) / 10
        
        # Benchmark step
        batch_actions = PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.full(batch_size, 5, dtype=jnp.int32),
            col=jnp.full(batch_size, 5, dtype=jnp.int32)
        )
        
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple runs for accuracy
            new_states, new_obs, rewards, dones, infos = batch_step(
                batch_states, batch_actions, config
            )
        step_time = (time.perf_counter() - start_time) / 10
        
        # Calculate metrics
        per_env_reset_time = reset_time / batch_size
        per_env_step_time = step_time / batch_size
        throughput = batch_size / step_time
        
        results[batch_size] = {
            'total_reset_time': reset_time,
            'total_step_time': step_time,
            'per_env_reset_time': per_env_reset_time,
            'per_env_step_time': per_env_step_time,
            'throughput': throughput
        }
        
        logger.info(f"    Reset: {reset_time*1000:.2f}ms total, "
                   f"{per_env_reset_time*1000:.3f}ms per env")
        logger.info(f"    Step:  {step_time*1000:.2f}ms total, "
                   f"{per_env_step_time*1000:.3f}ms per env")
        logger.info(f"    Throughput: {throughput:.0f} envs/sec")
    
    # Analyze scaling efficiency
    logger.info("  📈 Scaling Analysis:")
    baseline_per_env_time = results[1]['per_env_step_time']
    
    for batch_size in [8, 32, 128, 512]:
        if batch_size in results:
            current_per_env_time = results[batch_size]['per_env_step_time']
            efficiency = baseline_per_env_time / current_per_env_time
            logger.info(f"    Batch {batch_size}: {efficiency:.1f}x efficiency vs single env")
    
    # Find optimal batch size
    max_throughput = max(results[bs]['throughput'] for bs in results)
    optimal_batch_size = max(results.keys(), key=lambda bs: results[bs]['throughput'])
    
    logger.info(f"  🎯 Optimal batch size: {optimal_batch_size} "
               f"({max_throughput:.0f} envs/sec)")
    
    return results


def demonstrate_memory_scaling():
    """Demonstrate how memory usage scales with batch size."""
    logger.info("💾 Memory Scaling Analysis")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    
    batch_sizes = [1, 8, 32, 128]
    memory_results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"  Testing memory usage for batch size {batch_size}...")
        
        # Create batch
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        batch_states, batch_obs = batch_reset(keys, config, task)
        
        # Calculate memory usage
        state_memory = batch_states.working_grid.nbytes
        obs_memory = batch_obs.nbytes
        total_memory = state_memory + obs_memory
        
        per_env_memory = total_memory / batch_size
        
        memory_results[batch_size] = {
            'total_memory_mb': total_memory / (1024 * 1024),
            'per_env_memory_mb': per_env_memory / (1024 * 1024),
            'state_memory_mb': state_memory / (1024 * 1024),
            'obs_memory_mb': obs_memory / (1024 * 1024)
        }
        
        logger.info(f"    Total memory: {memory_results[batch_size]['total_memory_mb']:.2f} MB")
        logger.info(f"    Per env: {memory_results[batch_size]['per_env_memory_mb']:.3f} MB")
    
    # Analyze memory scaling
    logger.info("  📊 Memory Scaling Analysis:")
    single_env_memory = memory_results[1]['per_env_memory_mb']
    
    for batch_size in batch_sizes[1:]:
        current_per_env = memory_results[batch_size]['per_env_memory_mb']
        overhead = (current_per_env / single_env_memory - 1) * 100
        logger.info(f"    Batch {batch_size}: {overhead:+.1f}% memory overhead per env")
    
    return memory_results


def demonstrate_advanced_patterns():
    """Demonstrate advanced batch processing patterns."""
    logger.info("🚀 Advanced Batch Processing Patterns")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    batch_size = 16
    
    # Pattern 1: Different actions per environment
    logger.info("  Pattern 1: Different actions per environment")
    
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    batch_states, batch_obs = batch_reset(keys, config, task)
    
    # Create diverse actions
    mixed_actions = PointAction(
        operation=jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),  # Alternating ops
        row=jnp.arange(batch_size) % 10 + 5,  # Different rows
        col=jnp.arange(batch_size) % 10 + 5   # Different cols
    )
    
    new_states, new_obs, rewards, dones, infos = batch_step(
        batch_states, mixed_actions, config
    )
    
    logger.info(f"    Unique operations used: {jnp.unique(mixed_actions.operation)}")
    logger.info(f"    Reward range: {jnp.min(rewards):.3f} to {jnp.max(rewards):.3f}")
    
    # Pattern 2: Conditional processing based on state
    logger.info("  Pattern 2: Conditional processing")
    
    # Create actions based on current state
    step_counts = batch_states.step_count
    conditional_actions = PointAction(
        operation=jnp.where(step_counts < 5, 0, 1),  # Different ops based on step count
        row=jnp.where(step_counts < 5, 5, 10),       # Different positions
        col=jnp.where(step_counts < 5, 5, 10)
    )
    
    logger.info(f"    Step counts: {step_counts}")
    logger.info(f"    Conditional operations: {conditional_actions.operation}")
    
    # Pattern 3: Episode rollouts with early termination
    logger.info("  Pattern 3: Episode rollouts with early termination")
    
    # Reset for clean rollout
    batch_states, batch_obs = batch_reset(keys, config, task)
    
    max_steps = 20
    episode_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
    active_mask = jnp.ones(batch_size, dtype=jnp.bool_)
    
    for step in range(max_steps):
        # Only process active environments
        if not jnp.any(active_mask):
            break
            
        # Create actions for active environments
        actions = PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.full(batch_size, step % 10 + 5, dtype=jnp.int32),
            col=jnp.full(batch_size, 5, dtype=jnp.int32)
        )
        
        # Step all environments (inactive ones will be ignored in analysis)
        batch_states, batch_obs, rewards, dones, infos = batch_step(
            batch_states, actions, config
        )
        
        # Update active mask and episode lengths
        active_mask = active_mask & ~dones
        episode_lengths = jnp.where(active_mask, episode_lengths + 1, episode_lengths)
    
    logger.info(f"    Episode lengths: {episode_lengths}")
    logger.info(f"    Completed episodes: {jnp.sum(~active_mask)}")
    
    return {
        'mixed_actions_rewards': rewards,
        'episode_lengths': episode_lengths,
        'completion_rate': jnp.mean(~active_mask)
    }


def run_performance_comparison():
    """Compare batch processing vs sequential processing."""
    logger.info("⚡ Performance Comparison: Batch vs Sequential")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    
    num_envs = 64
    num_steps = 50
    
    # Sequential processing
    logger.info(f"  Sequential processing: {num_envs} envs × {num_steps} steps...")
    
    sequential_start = time.perf_counter()
    
    for env_idx in range(num_envs):
        key = jax.random.PRNGKey(42 + env_idx)
        state, obs = arc_reset(key, config, task)
        
        for step in range(num_steps):
            action = PointAction(
                operation=jnp.array(0),
                row=jnp.array(step % 10 + 5),
                col=jnp.array(5)
            )
            state, obs, reward, done, info = arc_step(state, action, config)
    
    sequential_time = time.perf_counter() - sequential_start
    
    # Batch processing
    logger.info(f"  Batch processing: {num_envs} envs × {num_steps} steps...")
    
    batch_start = time.perf_counter()
    
    keys = jax.random.split(jax.random.PRNGKey(42), num_envs)
    batch_states, batch_obs = batch_reset(keys, config, task)
    
    for step in range(num_steps):
        batch_actions = PointAction(
            operation=jnp.zeros(num_envs, dtype=jnp.int32),
            row=jnp.full(num_envs, step % 10 + 5, dtype=jnp.int32),
            col=jnp.full(num_envs, 5, dtype=jnp.int32)
        )
        batch_states, batch_obs, rewards, dones, infos = batch_step(
            batch_states, batch_actions, config
        )
    
    batch_time = time.perf_counter() - batch_start
    
    # Calculate metrics
    total_steps = num_envs * num_steps
    sequential_steps_per_sec = total_steps / sequential_time
    batch_steps_per_sec = total_steps / batch_time
    speedup = sequential_time / batch_time
    
    logger.info("  📊 Performance Comparison Results:")
    logger.info(f"    Sequential: {sequential_time:.2f}s ({sequential_steps_per_sec:.0f} steps/sec)")
    logger.info(f"    Batch:      {batch_time:.2f}s ({batch_steps_per_sec:.0f} steps/sec)")
    logger.info(f"    Speedup:    {speedup:.1f}x")
    logger.info(f"    Efficiency: {speedup/num_envs*100:.1f}% of theoretical maximum")
    
    return {
        'sequential_time': sequential_time,
        'batch_time': batch_time,
        'speedup': speedup,
        'batch_steps_per_sec': batch_steps_per_sec
    }


def main():
    """Run all batch processing demonstrations."""
    logger.info("🔄 JaxARC Batch Processing Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run demonstrations
        basic_results = demonstrate_basic_batch_processing()
        scaling_results = analyze_batch_scaling()
        memory_results = demonstrate_memory_scaling()
        pattern_results = demonstrate_advanced_patterns()
        comparison_results = run_performance_comparison()
        
        # Summary
        logger.info("\n🎉 Batch Processing Summary")
        logger.info("=" * 60)
        
        max_throughput = max(scaling_results[bs]['throughput'] for bs in scaling_results)
        optimal_batch_size = max(scaling_results.keys(), 
                               key=lambda bs: scaling_results[bs]['throughput'])
        
        logger.info(f"✅ Basic batch processing: {basic_results['deterministic']} deterministic")
        logger.info(f"✅ Optimal batch size: {optimal_batch_size} ({max_throughput:.0f} envs/sec)")
        logger.info(f"✅ Memory efficiency: Linear scaling with minimal overhead")
        logger.info(f"✅ Advanced patterns: Conditional processing and early termination")
        logger.info(f"✅ Performance gain: {comparison_results['speedup']:.1f}x vs sequential")
        
        logger.info(f"\n🚀 Batch processing achieves {comparison_results['batch_steps_per_sec']:.0f} steps/sec!")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()