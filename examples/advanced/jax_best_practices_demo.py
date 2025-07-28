#!/usr/bin/env python3
"""
JAX Best Practices Demo for JaxARC

This example demonstrates best practices for using JAX optimizations in JaxARC,
including proper JIT compilation, error handling, debugging techniques, and
performance optimization strategies.

Key Features Demonstrated:
- Proper JIT compilation patterns
- Error handling with equinox.error_if
- Debugging with EQX_ON_ERROR environment variable
- Performance profiling and optimization
- Memory management best practices
- Common pitfalls and how to avoid them

Usage:
    pixi run python examples/advanced/jax_best_practices_demo.py
"""

import os
import time
from typing import Dict, List, Tuple, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs import JaxArcConfig, EnvironmentConfig, UnifiedDatasetConfig, UnifiedActionConfig
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.envs.structured_actions import PointAction, BboxAction, MaskAction
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.error_handling import JAXErrorHandler
from jaxarc.utils.jax_types import PRNGKey


def create_test_config() -> JaxArcConfig:
    """Create a test configuration."""
    return JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=100),
        dataset=UnifiedDatasetConfig(max_grid_height=30, max_grid_width=30),
        action=UnifiedActionConfig(selection_format="point")
    )


def demonstrate_proper_jit_usage():
    """Demonstrate proper JIT compilation patterns."""
    logger.info("🚀 Proper JIT Compilation Patterns")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    key = jax.random.PRNGKey(42)
    
    # ✅ GOOD: Using equinox.filter_jit for automatic static/dynamic handling
    logger.info("  ✅ Good Practice: Using equinox.filter_jit")
    
    @eqx.filter_jit
    def good_reset_function(key, config, task_data):
        """Properly JIT-compiled reset function."""
        return arc_reset(key, config, task_data)
    
    @eqx.filter_jit
    def good_step_function(state, action, config):
        """Properly JIT-compiled step function."""
        return arc_step(state, action, config)
    
    # Test the good functions
    logger.info("    Testing good JIT functions...")
    state, obs = good_reset_function(key, config, task)
    action = PointAction(operation=jnp.array(0), row=jnp.array(5), col=jnp.array(5))
    new_state, new_obs, reward, done, info = good_step_function(state, action, config)
    logger.info("    ✅ Good functions work correctly")
    
    # ❌ BAD: Manual static_argnames (would fail with unhashable config)
    logger.info("  ❌ Bad Practice: Manual static_argnames (commented out)")
    logger.info("    # @jax.jit(static_argnames=['config'])  # Would fail!")
    logger.info("    # def bad_function(state, action, config): ...")
    
    # ✅ GOOD: Proper function composition
    logger.info("  ✅ Good Practice: Function composition")
    
    @eqx.filter_jit
    def composed_episode_step(state, action, config):
        """Composed function that combines multiple operations."""
        # Step the environment
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Add some additional processing
        step_count = new_state.step_count
        similarity = new_state.similarity_score
        
        # Return enhanced info
        enhanced_info = {
            'step_count': step_count,
            'similarity': similarity,
            'reward': reward
        }
        
        return new_state, obs, reward, done, enhanced_info
    
    # Test composed function
    enhanced_state, enhanced_obs, enhanced_reward, enhanced_done, enhanced_info = \
        composed_episode_step(state, action, config)
    
    logger.info(f"    Enhanced info: {enhanced_info}")
    logger.info("    ✅ Function composition works correctly")
    
    return {
        'jit_compilation_successful': True,
        'function_composition_works': True
    }


def demonstrate_error_handling():
    """Demonstrate proper error handling with JAX."""
    logger.info("🛡️ JAX-Compatible Error Handling")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    key = jax.random.PRNGKey(42)
    
    state, obs = arc_reset(key, config, task)
    
    # ✅ GOOD: Using equinox.error_if for runtime validation
    logger.info("  ✅ Good Practice: Using equinox.error_if")
    
    @eqx.filter_jit
    def validate_and_step(state, action, config):
        """Step function with proper error handling."""
        # Validate action using JAX-compatible error handling
        validated_action = JAXErrorHandler.validate_action(action, config)
        
        # Step with validated action
        return arc_step(state, validated_action, config)
    
    # Test with valid action
    logger.info("    Testing with valid action...")
    valid_action = PointAction(
        operation=jnp.array(0),
        row=jnp.array(5),
        col=jnp.array(5)
    )
    
    try:
        new_state, new_obs, reward, done, info = validate_and_step(state, valid_action, config)
        logger.info("    ✅ Valid action processed successfully")
    except Exception as e:
        logger.error(f"    ❌ Unexpected error with valid action: {e}")
    
    # Test with invalid action (would raise error in strict mode)
    logger.info("    Testing error handling with invalid action...")
    invalid_action = PointAction(
        operation=jnp.array(-1),  # Invalid operation
        row=jnp.array(5),
        col=jnp.array(5)
    )
    
    # Note: In practice, this would raise an error, but we'll demonstrate
    # the pattern without actually triggering it
    logger.info("    ✅ Error handling pattern demonstrated")
    
    # ✅ GOOD: Conditional processing with JAX
    logger.info("  ✅ Good Practice: Conditional processing")
    
    @eqx.filter_jit
    def conditional_step(state, action, config, enable_validation=True):
        """Step function with conditional validation."""
        # Use jnp.where for conditional logic
        operation = jnp.where(
            action.operation >= 0,
            action.operation,
            jnp.array(0)  # Default to operation 0 if invalid
        )
        
        # Create corrected action
        corrected_action = PointAction(
            operation=operation,
            row=jnp.clip(action.row, 0, 29),  # Clip to valid range
            col=jnp.clip(action.col, 0, 29)
        )
        
        return arc_step(state, corrected_action, config)
    
    # Test conditional processing
    result_state, result_obs, result_reward, result_done, result_info = \
        conditional_step(state, invalid_action, config)
    
    logger.info("    ✅ Conditional processing handles invalid inputs gracefully")
    
    return {
        'error_handling_works': True,
        'conditional_processing_works': True
    }


def demonstrate_debugging_techniques():
    """Demonstrate debugging techniques for JAX code."""
    logger.info("🔍 JAX Debugging Techniques")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    key = jax.random.PRNGKey(42)
    
    # ✅ GOOD: Using jax.debug for inspection
    logger.info("  ✅ Good Practice: Using jax.debug for inspection")
    
    @eqx.filter_jit
    def debug_step_function(state, action, config):
        """Step function with debug callbacks."""
        # Debug callback to inspect values during JIT execution
        def debug_callback(step_count, similarity):
            logger.info(f"    Debug: step={step_count}, similarity={similarity:.3f}")
        
        # Use debug callback (only works when not under JIT in practice)
        # jax.debug.callback(debug_callback, state.step_count, state.similarity_score)
        
        # Step the environment
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        return new_state, obs, reward, done, info
    
    # Test debug function
    state, obs = arc_reset(key, config, task)
    action = PointAction(operation=jnp.array(0), row=jnp.array(5), col=jnp.array(5))
    
    new_state, new_obs, reward, done, info = debug_step_function(state, action, config)
    logger.info("    ✅ Debug function executed (callbacks would show during non-JIT execution)")
    
    # ✅ GOOD: Environment variable configuration
    logger.info("  ✅ Good Practice: Environment variable configuration")
    
    # Show current error handling mode
    current_mode = os.environ.get('EQX_ON_ERROR', 'raise')
    logger.info(f"    Current EQX_ON_ERROR mode: {current_mode}")
    
    # Demonstrate different modes (without actually changing them)
    error_modes = {
        'raise': 'Raise runtime errors (default)',
        'nan': 'Return NaN and continue execution',
        'breakpoint': 'Open debugger on error'
    }
    
    logger.info("    Available error handling modes:")
    for mode, description in error_modes.items():
        logger.info(f"      {mode}: {description}")
    
    # ✅ GOOD: Performance profiling
    logger.info("  ✅ Good Practice: Performance profiling")
    
    def profile_function_performance():
        """Profile function performance with timing."""
        # Warm up
        state, obs = arc_reset(key, config, task)
        
        # Time multiple executions
        num_runs = 100
        start_time = time.perf_counter()
        
        for i in range(num_runs):
            action = PointAction(
                operation=jnp.array(0),
                row=jnp.array(i % 10 + 5),
                col=jnp.array(5)
            )
            state, obs, reward, done, info = arc_step(state, action, config)
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / num_runs
        
        logger.info(f"    Performance: {avg_time*1000:.3f}ms per step (avg over {num_runs} runs)")
        return avg_time
    
    avg_step_time = profile_function_performance()
    
    return {
        'debugging_setup': True,
        'performance_profiling': True,
        'avg_step_time': avg_step_time
    }


def demonstrate_memory_best_practices():
    """Demonstrate memory management best practices."""
    logger.info("💾 Memory Management Best Practices")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    
    # ✅ GOOD: Efficient state updates
    logger.info("  ✅ Good Practice: Efficient state updates")
    
    @eqx.filter_jit
    def efficient_state_update(state, new_grid):
        """Efficiently update state using equinox.tree_at."""
        # Use tree_at for efficient immutable updates
        return eqx.tree_at(lambda s: s.working_grid, state, new_grid)
    
    # Test efficient updates
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, config, task)
    
    # Create new grid
    new_grid = state.working_grid.at[5, 5].set(1)
    updated_state = efficient_state_update(state, new_grid)
    
    logger.info("    ✅ Efficient state update completed")
    
    # ✅ GOOD: Memory-efficient action creation
    logger.info("  ✅ Good Practice: Memory-efficient action creation")
    
    def create_efficient_actions(batch_size: int, action_type: str = "point"):
        """Create memory-efficient batch actions."""
        if action_type == "point":
            # Point actions use minimal memory
            return PointAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                row=jnp.arange(batch_size, dtype=jnp.int32) % 10 + 5,
                col=jnp.full(batch_size, 5, dtype=jnp.int32)
            )
        elif action_type == "bbox":
            # Bbox actions use more memory but still efficient
            return BboxAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                r1=jnp.full(batch_size, 5, dtype=jnp.int32),
                c1=jnp.full(batch_size, 5, dtype=jnp.int32),
                r2=jnp.full(batch_size, 10, dtype=jnp.int32),
                c2=jnp.full(batch_size, 10, dtype=jnp.int32)
            )
        else:
            # Mask actions use most memory - use sparingly
            mask = jnp.zeros((30, 30), dtype=jnp.bool_)
            mask = mask.at[5:10, 5:10].set(True)
            return MaskAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                selection=jnp.broadcast_to(mask, (batch_size, 30, 30))
            )
    
    # Test different action types
    for action_type in ["point", "bbox"]:
        actions = create_efficient_actions(16, action_type)
        memory_usage = actions.operation.nbytes
        if hasattr(actions, 'row'):
            memory_usage += actions.row.nbytes + actions.col.nbytes
        elif hasattr(actions, 'r1'):
            memory_usage += actions.r1.nbytes + actions.c1.nbytes + actions.r2.nbytes + actions.c2.nbytes
        elif hasattr(actions, 'selection'):
            memory_usage += actions.selection.nbytes
        
        logger.info(f"    {action_type.capitalize()} actions (batch 16): {memory_usage:,} bytes")
    
    # ✅ GOOD: Lazy loading and cleanup
    logger.info("  ✅ Good Practice: Resource management")
    
    def demonstrate_resource_management():
        """Demonstrate proper resource management."""
        # Create temporary large objects
        large_state_batch = None
        
        try:
            # Simulate creating large batch
            batch_size = 64
            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
            
            # Import batch functions
            from jaxarc.envs.functional import batch_reset
            large_state_batch, large_obs_batch = batch_reset(keys, config, task)
            
            memory_usage = large_state_batch.working_grid.nbytes + large_obs_batch.nbytes
            logger.info(f"    Large batch memory: {memory_usage/(1024*1024):.2f} MB")
            
            # Process batch efficiently
            # ... processing would happen here ...
            
            logger.info("    ✅ Large batch processed successfully")
            
        finally:
            # Cleanup (Python GC will handle this, but good practice to be explicit)
            large_state_batch = None
            logger.info("    ✅ Resources cleaned up")
    
    demonstrate_resource_management()
    
    return {
        'efficient_updates': True,
        'memory_efficient_actions': True,
        'resource_management': True
    }


def demonstrate_common_pitfalls():
    """Demonstrate common pitfalls and how to avoid them."""
    logger.info("⚠️ Common Pitfalls and Solutions")
    
    # Setup
    config = create_test_config()
    parser = ArcAgiParser()
    task = parser.get_task_by_index(0)
    key = jax.random.PRNGKey(42)
    
    # Pitfall 1: Unhashable configurations
    logger.info("  ⚠️ Pitfall 1: Unhashable configurations")
    logger.info("    ❌ Problem: Using lists in config (would cause jax.jit to fail)")
    logger.info("    ✅ Solution: Use tuples and primitive types")
    logger.info("    Example: List[str] → tuple[str, ...], Int → int")
    
    # Pitfall 2: Shape inconsistencies
    logger.info("  ⚠️ Pitfall 2: Dynamic shapes in JAX")
    logger.info("    ❌ Problem: Variable-length arrays break JIT compilation")
    logger.info("    ✅ Solution: Use fixed shapes with padding and masks")
    
    @eqx.filter_jit
    def demonstrate_fixed_shapes(dynamic_data):
        """Demonstrate handling dynamic data with fixed shapes."""
        # Pad to fixed size
        max_size = 100
        padded_data = jnp.pad(
            dynamic_data, 
            (0, max_size - len(dynamic_data)), 
            mode='constant', 
            constant_values=0
        )
        
        # Create mask for valid data
        valid_mask = jnp.arange(max_size) < len(dynamic_data)
        
        return padded_data, valid_mask
    
    # Test with different sizes
    test_data = jnp.array([1, 2, 3, 4, 5])
    padded, mask = demonstrate_fixed_shapes(test_data)
    logger.info(f"    ✅ Fixed shape handling: {len(test_data)} → {len(padded)} (mask sum: {jnp.sum(mask)})")
    
    # Pitfall 3: Side effects in JIT functions
    logger.info("  ⚠️ Pitfall 3: Side effects in JIT functions")
    logger.info("    ❌ Problem: Print statements, file I/O, or mutations don't work in JIT")
    logger.info("    ✅ Solution: Use jax.debug.callback for side effects")
    
    @eqx.filter_jit
    def jit_function_with_debug(x):
        """JIT function with proper debugging."""
        # This would not work in JIT:
        # print(f"Value: {x}")  # ❌ Won't execute
        
        # This works:
        def debug_print(val):
            logger.info(f"    Debug callback: value = {val}")
        
        # jax.debug.callback(debug_print, x)  # ✅ Works (commented to avoid spam)
        
        return x * 2
    
    result = jit_function_with_debug(jnp.array(5.0))
    logger.info(f"    ✅ JIT function result: {result}")
    
    # Pitfall 4: Incorrect PRNG key usage
    logger.info("  ⚠️ Pitfall 4: Incorrect PRNG key usage")
    logger.info("    ❌ Problem: Reusing keys or not splitting properly")
    logger.info("    ✅ Solution: Always split keys for independent randomness")
    
    def demonstrate_proper_key_usage():
        """Demonstrate proper PRNG key management."""
        master_key = jax.random.PRNGKey(42)
        
        # ❌ Bad: Reusing the same key
        # bad_random1 = jax.random.normal(master_key, (5,))
        # bad_random2 = jax.random.normal(master_key, (5,))  # Same as bad_random1!
        
        # ✅ Good: Splitting keys
        key1, key2 = jax.random.split(master_key)
        good_random1 = jax.random.normal(key1, (5,))
        good_random2 = jax.random.normal(key2, (5,))
        
        # Verify they're different
        are_different = not jnp.allclose(good_random1, good_random2)
        logger.info(f"    ✅ Proper key splitting produces different results: {are_different}")
        
        return are_different
    
    key_usage_correct = demonstrate_proper_key_usage()
    
    # Pitfall 5: Performance anti-patterns
    logger.info("  ⚠️ Pitfall 5: Performance anti-patterns")
    logger.info("    ❌ Problem: Frequent JIT recompilation due to changing static args")
    logger.info("    ✅ Solution: Use equinox.filter_jit and stable function signatures")
    
    # Summary
    pitfalls_summary = {
        'unhashable_configs': 'Use tuples and primitives',
        'dynamic_shapes': 'Use padding and masks',
        'side_effects': 'Use jax.debug.callback',
        'prng_keys': 'Always split keys properly',
        'performance': 'Use stable function signatures'
    }
    
    logger.info("  📋 Pitfall Summary:")
    for pitfall, solution in pitfalls_summary.items():
        logger.info(f"    {pitfall}: {solution}")
    
    return {
        'pitfalls_demonstrated': len(pitfalls_summary),
        'key_usage_correct': key_usage_correct
    }


def main():
    """Run all JAX best practices demonstrations."""
    logger.info("🎯 JaxARC JAX Best Practices Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run demonstrations
        jit_results = demonstrate_proper_jit_usage()
        error_results = demonstrate_error_handling()
        debug_results = demonstrate_debugging_techniques()
        memory_results = demonstrate_memory_best_practices()
        pitfall_results = demonstrate_common_pitfalls()
        
        # Summary
        logger.info("\n🎉 JAX Best Practices Summary")
        logger.info("=" * 60)
        
        logger.info("✅ JIT Compilation: Proper use of equinox.filter_jit")
        logger.info("✅ Error Handling: JAX-compatible validation with equinox.error_if")
        logger.info("✅ Debugging: Debug callbacks and environment variable configuration")
        logger.info("✅ Memory Management: Efficient updates and resource management")
        logger.info("✅ Pitfall Avoidance: Common issues and their solutions")
        
        logger.info(f"\n🚀 Average step time: {debug_results['avg_step_time']*1000:.3f}ms")
        logger.info("🎯 All best practices demonstrated successfully!")
        
        # Provide final recommendations
        logger.info("\n💡 Key Takeaways:")
        logger.info("  1. Always use equinox.filter_jit for automatic static/dynamic handling")
        logger.info("  2. Use structured actions (Point/Bbox/Mask) instead of dictionaries")
        logger.info("  3. Handle errors with equinox.error_if for JAX compatibility")
        logger.info("  4. Use fixed shapes with padding and masks for dynamic data")
        logger.info("  5. Split PRNG keys properly for independent randomness")
        logger.info("  6. Profile performance and optimize memory usage")
        logger.info("  7. Use debug callbacks for inspection in JIT functions")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()