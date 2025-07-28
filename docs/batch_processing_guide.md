# Batch Processing Guide

This guide covers JaxARC's batch processing capabilities, which enable parallel execution of multiple environments using JAX's vectorization features. Batch processing provides significant performance improvements and is essential for efficient RL training.

## Overview

JaxARC's batch processing system provides:

- **Vectorized Operations**: Use `jax.vmap` for parallel environment execution
- **Efficient Memory Usage**: Optimized memory layout for batch operations
- **Deterministic Behavior**: Proper PRNG key management for reproducible results
- **Scalable Performance**: Linear scaling with batch size up to hardware limits
- **JIT Compatibility**: Full JIT compilation support for maximum performance

## Quick Start

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import JaxArcConfig, EnvironmentConfig, UnifiedDatasetConfig, UnifiedActionConfig
from jaxarc.envs.functional import batch_reset, batch_step
from jaxarc.envs.structured_actions import PointAction

# Configuration
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    dataset=UnifiedDatasetConfig(max_grid_height=30, max_grid_width=30),
    action=UnifiedActionConfig(selection_format="point")
)

# Create mock task
task = create_mock_task()

# Batch processing
batch_size = 16
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

# Batch reset
batch_states, batch_obs = batch_reset(keys, config, task)

# Create batch actions
batch_actions = PointAction(
    operation=jnp.zeros(batch_size, dtype=jnp.int32),
    row=jnp.full(batch_size, 5, dtype=jnp.int32),
    col=jnp.full(batch_size, 5, dtype=jnp.int32)
)

# Batch step
batch_states, batch_obs, rewards, dones, infos = batch_step(
    batch_states, batch_actions, config
)

print(f"Batch shape: {batch_states.working_grid.shape}")  # (16, 30, 30)
print(f"Rewards: {rewards}")  # (16,)
```

## Core Functions

### batch_reset

Reset multiple environments in parallel.

```python
def batch_reset(
    keys: jnp.ndarray,           # Shape: (batch_size, 2)
    config: JaxArcConfig,        # Environment configuration
    task_data: JaxArcTask        # Task data (shared across batch)
) -> tuple[ArcEnvState, jnp.ndarray]:
    """Reset multiple environments in parallel.
    
    Args:
        keys: PRNG keys for each environment
        config: Environment configuration (shared)
        task_data: Task data (shared across all environments)
        
    Returns:
        batch_states: Batched environment states
        batch_obs: Batched observations
    """
```

#### Usage Example

```python
# Create batch of PRNG keys
master_key = jax.random.PRNGKey(42)
batch_size = 32
keys = jax.random.split(master_key, batch_size)

# Reset batch of environments
batch_states, batch_obs = batch_reset(keys, config, task)

# Verify batch dimensions
assert batch_states.working_grid.shape == (batch_size, 30, 30)
assert batch_obs.shape == (batch_size, 30, 30)
```

### batch_step

Step multiple environments in parallel.

```python
def batch_step(
    states: ArcEnvState,         # Batched states
    actions: StructuredAction,   # Batched actions
    config: JaxArcConfig         # Environment configuration
) -> tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Step multiple environments in parallel.
    
    Args:
        states: Batched environment states
        actions: Batched structured actions
        config: Environment configuration (shared)
        
    Returns:
        new_states: Updated batched states
        new_obs: Updated batched observations
        rewards: Batch of rewards
        dones: Batch of done flags
        infos: Batch of info dictionaries
    """
```

#### Usage Example

```python
# Create batch actions
batch_actions = PointAction(
    operation=jnp.array([0, 1, 0, 1] * (batch_size // 4)),  # Mixed operations
    row=jnp.arange(batch_size) % 10 + 5,                    # Different rows
    col=jnp.full(batch_size, 7, dtype=jnp.int32)           # Same column
)

# Step batch of environments
new_states, new_obs, rewards, dones, infos = batch_step(
    batch_states, batch_actions, config
)

# Process results
print(f"Reward range: {jnp.min(rewards):.3f} to {jnp.max(rewards):.3f}")
print(f"Completed episodes: {jnp.sum(dones)}")
```

## PRNG Key Management

Proper PRNG key management is crucial for deterministic batch processing.

### Key Splitting

Always split keys properly for independent randomness:

```python
# ✅ Good: Proper key splitting
master_key = jax.random.PRNGKey(42)
batch_keys = jax.random.split(master_key, batch_size)

# Each environment gets a unique key
for i, key in enumerate(batch_keys):
    print(f"Env {i}: {key}")

# ❌ Bad: Reusing the same key
# batch_keys = jnp.array([master_key] * batch_size)  # All identical!
```

### Deterministic Behavior

Verify deterministic behavior across runs:

```python
def test_deterministic_batch():
    """Test that batch processing is deterministic."""
    key = jax.random.PRNGKey(42)
    
    # First run
    keys1 = jax.random.split(key, batch_size)
    states1, obs1 = batch_reset(keys1, config, task)
    
    # Second run with same key
    keys2 = jax.random.split(key, batch_size)
    states2, obs2 = batch_reset(keys2, config, task)
    
    # Should be identical
    assert jnp.allclose(states1.working_grid, states2.working_grid)
    assert jnp.allclose(obs1, obs2)
    
    print("✅ Batch processing is deterministic")

test_deterministic_batch()
```

### Key Management in Episodes

Manage keys throughout episode execution:

```python
def run_batch_episode(initial_keys, config, task, max_steps=50):
    """Run complete batch episode with proper key management."""
    # Reset environments
    states, obs = batch_reset(initial_keys, config, task)
    
    # Track episode data
    episode_rewards = jnp.zeros(batch_size)
    episode_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
    active_mask = jnp.ones(batch_size, dtype=jnp.bool_)
    
    for step in range(max_steps):
        # Create actions (could be from policy)
        actions = PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.full(batch_size, step % 10 + 5, dtype=jnp.int32),
            col=jnp.full(batch_size, 5, dtype=jnp.int32)
        )
        
        # Step environments
        states, obs, rewards, dones, infos = batch_step(states, actions, config)
        
        # Update tracking
        episode_rewards += rewards * active_mask
        episode_lengths = jnp.where(active_mask, episode_lengths + 1, episode_lengths)
        active_mask = active_mask & ~dones
        
        # Early termination if all done
        if not jnp.any(active_mask):
            break
    
    return {
        'final_states': states,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'completion_rate': jnp.mean(~active_mask)
    }

# Run batch episode
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
results = run_batch_episode(keys, config, task)

print(f"Average reward: {jnp.mean(results['episode_rewards']):.3f}")
print(f"Average length: {jnp.mean(results['episode_lengths']):.1f}")
print(f"Completion rate: {results['completion_rate']:.1%}")
```

## Performance Optimization

### Batch Size Selection

Choose optimal batch size based on hardware and memory constraints:

```python
def find_optimal_batch_size(config, task, max_batch_size=1024):
    """Find optimal batch size for current hardware."""
    import time
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Test batch processing
            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
            
            # Warm up
            batch_states, batch_obs = batch_reset(keys, config, task)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(10):  # Multiple runs for accuracy
                batch_states, batch_obs = batch_reset(keys, config, task)
            elapsed = (time.perf_counter() - start_time) / 10
            
            # Calculate metrics
            per_env_time = elapsed / batch_size
            throughput = batch_size / elapsed
            
            results[batch_size] = {
                'per_env_time': per_env_time,
                'throughput': throughput,
                'total_time': elapsed
            }
            
            print(f"Batch {batch_size:3d}: {per_env_time*1000:.2f}ms/env, {throughput:.0f} envs/sec")
            
        except Exception as e:
            print(f"Batch {batch_size:3d}: Failed ({e})")
            break
    
    # Find optimal batch size
    optimal_batch_size = max(results.keys(), key=lambda bs: results[bs]['throughput'])
    print(f"\nOptimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size, results

# Find optimal batch size
optimal_bs, perf_results = find_optimal_batch_size(config, task)
```

### Memory Usage Analysis

Monitor memory usage across batch sizes:

```python
def analyze_memory_usage(config, task, batch_sizes=[1, 16, 64, 256]):
    """Analyze memory usage for different batch sizes."""
    memory_results = {}
    
    for batch_size in batch_sizes:
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        batch_states, batch_obs = batch_reset(keys, config, task)
        
        # Calculate memory usage
        state_memory = batch_states.working_grid.nbytes
        obs_memory = batch_obs.nbytes
        total_memory = state_memory + obs_memory
        
        per_env_memory = total_memory / batch_size
        
        memory_results[batch_size] = {
            'total_mb': total_memory / (1024 * 1024),
            'per_env_mb': per_env_memory / (1024 * 1024),
            'state_mb': state_memory / (1024 * 1024),
            'obs_mb': obs_memory / (1024 * 1024)
        }
        
        print(f"Batch {batch_size:3d}: {memory_results[batch_size]['total_mb']:.1f}MB total, "
              f"{memory_results[batch_size]['per_env_mb']:.3f}MB per env")
    
    # Analyze scaling efficiency
    baseline_per_env = memory_results[1]['per_env_mb']
    
    print("\nMemory scaling efficiency:")
    for batch_size in batch_sizes[1:]:
        current_per_env = memory_results[batch_size]['per_env_mb']
        overhead = (current_per_env / baseline_per_env - 1) * 100
        print(f"  Batch {batch_size:3d}: {overhead:+.1f}% overhead per env")
    
    return memory_results

# Analyze memory usage
memory_analysis = analyze_memory_usage(config, task)
```

### JIT Compilation Optimization

Optimize JIT compilation for batch processing:

```python
# Pre-compile batch functions with specific shapes
@eqx.filter_jit
def compiled_batch_reset(keys, config, task_data):
    """Pre-compiled batch reset function."""
    return batch_reset(keys, config, task_data)

@eqx.filter_jit
def compiled_batch_step(states, actions, config):
    """Pre-compiled batch step function."""
    return batch_step(states, actions, config)

def warm_up_batch_functions(batch_size, config, task):
    """Warm up batch functions to avoid compilation overhead."""
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    
    # Warm up reset
    batch_states, batch_obs = compiled_batch_reset(keys, config, task)
    
    # Warm up step
    actions = PointAction(
        operation=jnp.zeros(batch_size, dtype=jnp.int32),
        row=jnp.zeros(batch_size, dtype=jnp.int32),
        col=jnp.zeros(batch_size, dtype=jnp.int32)
    )
    
    compiled_batch_step(batch_states, actions, config)
    print(f"✅ Warmed up batch functions for batch size {batch_size}")

# Warm up for your target batch size
warm_up_batch_functions(64, config, task)
```

## Advanced Patterns

### Conditional Batch Processing

Process environments conditionally based on state:

```python
def conditional_batch_processing(batch_states, config):
    """Process environments conditionally based on state."""
    # Get step counts
    step_counts = batch_states.step_count
    
    # Create different actions based on step count
    early_actions = PointAction(
        operation=jnp.zeros(batch_size, dtype=jnp.int32),  # Fill
        row=jnp.full(batch_size, 5, dtype=jnp.int32),
        col=jnp.full(batch_size, 5, dtype=jnp.int32)
    )
    
    late_actions = PointAction(
        operation=jnp.ones(batch_size, dtype=jnp.int32),   # Different operation
        row=jnp.full(batch_size, 10, dtype=jnp.int32),
        col=jnp.full(batch_size, 10, dtype=jnp.int32)
    )
    
    # Select actions based on step count
    use_early = step_counts < 25
    
    conditional_actions = PointAction(
        operation=jnp.where(use_early, early_actions.operation, late_actions.operation),
        row=jnp.where(use_early, early_actions.row, late_actions.row),
        col=jnp.where(use_early, early_actions.col, late_actions.col)
    )
    
    return conditional_actions

# Usage
conditional_actions = conditional_batch_processing(batch_states, config)
new_states, new_obs, rewards, dones, infos = batch_step(
    batch_states, conditional_actions, config
)
```

### Dynamic Batch Sizes

Handle dynamic batch sizes with padding:

```python
def process_dynamic_batch(states_list, actions_list, config, max_batch_size=64):
    """Process variable-length batch with padding."""
    actual_batch_size = len(states_list)
    
    if actual_batch_size > max_batch_size:
        # Process in chunks
        results = []
        for i in range(0, actual_batch_size, max_batch_size):
            chunk_states = states_list[i:i+max_batch_size]
            chunk_actions = actions_list[i:i+max_batch_size]
            
            # Convert to batch format
            batch_states = stack_states(chunk_states)
            batch_actions = stack_actions(chunk_actions)
            
            # Process chunk
            chunk_results = batch_step(batch_states, batch_actions, config)
            results.append(chunk_results)
        
        # Combine results
        return combine_batch_results(results)
    
    else:
        # Pad to consistent size for JIT efficiency
        padded_states = pad_states(states_list, max_batch_size)
        padded_actions = pad_actions(actions_list, max_batch_size)
        
        # Process with padding
        batch_results = batch_step(padded_states, padded_actions, config)
        
        # Remove padding from results
        return unpad_results(batch_results, actual_batch_size)

def stack_states(states_list):
    """Stack list of states into batch format."""
    # Implementation depends on state structure
    pass

def stack_actions(actions_list):
    """Stack list of actions into batch format."""
    # Implementation depends on action type
    pass
```

### Hierarchical Batch Processing

Process batches at multiple levels:

```python
def hierarchical_batch_processing(
    num_episodes=100,
    envs_per_episode=16,
    steps_per_episode=50,
    config=None,
    task=None
):
    """Process multiple episodes, each with multiple environments."""
    
    episode_results = []
    
    for episode in range(num_episodes):
        # Create episode-specific keys
        episode_key = jax.random.PRNGKey(episode)
        env_keys = jax.random.split(episode_key, envs_per_episode)
        
        # Reset environments for this episode
        batch_states, batch_obs = batch_reset(env_keys, config, task)
        
        # Run episode
        episode_rewards = jnp.zeros(envs_per_episode)
        
        for step in range(steps_per_episode):
            # Create step-specific actions
            actions = PointAction(
                operation=jnp.zeros(envs_per_episode, dtype=jnp.int32),
                row=jnp.full(envs_per_episode, step % 10 + 5, dtype=jnp.int32),
                col=jnp.full(envs_per_episode, 5, dtype=jnp.int32)
            )
            
            # Step all environments
            batch_states, batch_obs, rewards, dones, infos = batch_step(
                batch_states, actions, config
            )
            
            episode_rewards += rewards
            
            # Handle done environments (could reset or mask)
            # For simplicity, continue with all environments
        
        episode_results.append({
            'episode': episode,
            'rewards': episode_rewards,
            'mean_reward': jnp.mean(episode_rewards),
            'std_reward': jnp.std(episode_rewards)
        })
        
        if episode % 10 == 0:
            print(f"Episode {episode}: mean reward = {episode_results[-1]['mean_reward']:.3f}")
    
    return episode_results

# Run hierarchical processing
results = hierarchical_batch_processing(
    num_episodes=50,
    envs_per_episode=32,
    steps_per_episode=100,
    config=config,
    task=task
)

# Analyze results
all_rewards = jnp.concatenate([r['rewards'] for r in results])
print(f"Overall mean reward: {jnp.mean(all_rewards):.3f}")
print(f"Overall std reward: {jnp.std(all_rewards):.3f}")
```

## Integration with RL Training

### Policy Integration

Integrate batch processing with policy networks:

```python
def batch_policy_rollout(policy_fn, batch_states, config, num_steps=10):
    """Perform policy rollout on batch of environments."""
    
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'log_probs': []
    }
    
    current_states = batch_states
    
    for step in range(num_steps):
        # Get policy actions
        policy_output = policy_fn(current_states)
        actions = policy_output['actions']
        log_probs = policy_output['log_probs']
        
        # Step environments
        new_states, obs, rewards, dones, infos = batch_step(
            current_states, actions, config
        )
        
        # Store trajectory
        trajectory['states'].append(current_states)
        trajectory['actions'].append(actions)
        trajectory['rewards'].append(rewards)
        trajectory['dones'].append(dones)
        trajectory['log_probs'].append(log_probs)
        
        current_states = new_states
    
    # Stack trajectory data
    for key in trajectory:
        trajectory[key] = jnp.stack(trajectory[key], axis=0)  # (steps, batch, ...)
    
    return trajectory

# Example policy function
def example_policy(states):
    """Example random policy."""
    batch_size = states.working_grid.shape[0]
    
    # Random actions
    actions = PointAction(
        operation=jax.random.randint(
            jax.random.PRNGKey(0), (batch_size,), 0, 10
        ),
        row=jax.random.randint(
            jax.random.PRNGKey(1), (batch_size,), 0, 30
        ),
        col=jax.random.randint(
            jax.random.PRNGKey(2), (batch_size,), 0, 30
        )
    )
    
    # Dummy log probabilities
    log_probs = jnp.zeros(batch_size)
    
    return {'actions': actions, 'log_probs': log_probs}

# Run policy rollout
trajectory = batch_policy_rollout(example_policy, batch_states, config)
print(f"Trajectory shape: {trajectory['rewards'].shape}")  # (steps, batch)
```

### Value Function Training

Use batch processing for value function training:

```python
def compute_batch_values(value_fn, batch_states):
    """Compute value function on batch of states."""
    # Extract features from states
    features = extract_state_features(batch_states)
    
    # Compute values
    values = value_fn(features)
    
    return values

def extract_state_features(batch_states):
    """Extract features from batch of states."""
    # Simple feature extraction
    features = {
        'working_grid': batch_states.working_grid,
        'target_grid': batch_states.target_grid,
        'similarity': batch_states.similarity_score,
        'step_count': batch_states.step_count
    }
    
    return features

# Example value function
def example_value_fn(features):
    """Example value function."""
    # Simple heuristic based on similarity
    return features['similarity'] * 10.0

# Compute batch values
batch_values = compute_batch_values(example_value_fn, batch_states)
print(f"Batch values: {batch_values}")
```

## Debugging and Monitoring

### Batch State Inspection

Inspect batch states for debugging:

```python
def inspect_batch_states(batch_states, indices=[0, 1, 2]):
    """Inspect specific environments in batch."""
    batch_size = batch_states.working_grid.shape[0]
    
    print(f"Batch size: {batch_size}")
    print(f"Grid shape: {batch_states.working_grid.shape[1:]}")
    
    for i in indices:
        if i < batch_size:
            print(f"\nEnvironment {i}:")
            print(f"  Step count: {batch_states.step_count[i]}")
            print(f"  Similarity: {batch_states.similarity_score[i]:.3f}")
            print(f"  Episode done: {batch_states.episode_done[i]}")
            
            # Show grid summary
            grid = batch_states.working_grid[i]
            unique_colors = jnp.unique(grid)
            print(f"  Unique colors: {unique_colors}")

# Inspect batch
inspect_batch_states(batch_states)
```

### Performance Monitoring

Monitor batch processing performance:

```python
class BatchPerformanceMonitor:
    """Monitor batch processing performance."""
    
    def __init__(self):
        self.reset_times = []
        self.step_times = []
        self.batch_sizes = []
    
    def time_batch_reset(self, keys, config, task):
        """Time batch reset operation."""
        import time
        
        start_time = time.perf_counter()
        result = batch_reset(keys, config, task)
        elapsed = time.perf_counter() - start_time
        
        self.reset_times.append(elapsed)
        self.batch_sizes.append(len(keys))
        
        return result
    
    def time_batch_step(self, states, actions, config):
        """Time batch step operation."""
        import time
        
        start_time = time.perf_counter()
        result = batch_step(states, actions, config)
        elapsed = time.perf_counter() - start_time
        
        self.step_times.append(elapsed)
        
        return result
    
    def get_stats(self):
        """Get performance statistics."""
        if not self.reset_times:
            return "No data collected"
        
        reset_times = jnp.array(self.reset_times)
        step_times = jnp.array(self.step_times)
        batch_sizes = jnp.array(self.batch_sizes)
        
        return {
            'reset_time_mean': jnp.mean(reset_times),
            'reset_time_std': jnp.std(reset_times),
            'step_time_mean': jnp.mean(step_times),
            'step_time_std': jnp.std(step_times),
            'avg_batch_size': jnp.mean(batch_sizes),
            'total_operations': len(self.reset_times) + len(self.step_times)
        }

# Use performance monitor
monitor = BatchPerformanceMonitor()

# Monitored operations
keys = jax.random.split(jax.random.PRNGKey(42), 32)
batch_states, batch_obs = monitor.time_batch_reset(keys, config, task)

actions = PointAction(
    operation=jnp.zeros(32, dtype=jnp.int32),
    row=jnp.full(32, 5, dtype=jnp.int32),
    col=jnp.full(32, 5, dtype=jnp.int32)
)

new_states, new_obs, rewards, dones, infos = monitor.time_batch_step(
    batch_states, actions, config
)

# Get statistics
stats = monitor.get_stats()
print(f"Performance stats: {stats}")
```

## Best Practices

### Memory Management

1. **Choose appropriate batch sizes**: Balance throughput and memory usage
2. **Use memory-efficient action formats**: Prefer point/bbox over mask actions
3. **Monitor memory usage**: Track memory consumption across batch sizes
4. **Clean up large batches**: Explicitly delete large batch objects when done

### Performance Optimization

1. **Warm up JIT functions**: Always warm up before benchmarking
2. **Use consistent batch sizes**: Avoid frequent recompilation
3. **Profile your code**: Identify bottlenecks in batch processing
4. **Optimize PRNG usage**: Minimize key splitting overhead

### Debugging

1. **Start small**: Debug with small batch sizes first
2. **Verify determinism**: Ensure reproducible results
3. **Monitor individual environments**: Inspect specific environments in batch
4. **Use performance monitoring**: Track timing and memory usage

### Integration

1. **Design for batches**: Structure your RL code around batch processing
2. **Handle variable batch sizes**: Support dynamic batch sizes when needed
3. **Separate concerns**: Keep batch processing logic separate from policy logic
4. **Test thoroughly**: Verify batch processing with your specific use case

This comprehensive guide covers all aspects of batch processing in JaxARC. For practical examples and advanced patterns, see the example scripts in `examples/advanced/batch_processing_demo.py`.