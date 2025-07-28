# Performance Optimization Guide

This guide provides comprehensive strategies for optimizing JaxARC performance, covering JIT compilation, memory management, batch processing, and debugging techniques. Following these guidelines can achieve 100x+ performance improvements and 90%+ memory reduction.

## Overview

JaxARC performance optimization focuses on four key areas:

1. **JIT Compilation**: Leveraging JAX's Just-In-Time compilation for speed
2. **Memory Management**: Optimizing memory usage through efficient data structures
3. **Batch Processing**: Parallelizing operations across multiple environments
4. **Profiling & Debugging**: Identifying and resolving performance bottlenecks

## Performance Targets

### Expected Improvements

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| Step Time | 50-100ms | 0.1-1ms | 50-1000x |
| Memory per Environment | 3.5MB | 0.05-3.5MB | 1-70x |
| Batch Throughput | Not possible | 10,000+ steps/sec | ∞ |
| JIT Compilation | 0/6 functions | 6/6 functions | ∞ |

### Performance Benchmarks

```python
# Target performance metrics
TARGET_STEP_TIME = 1.0  # ms per step
TARGET_THROUGHPUT = 10000  # steps per second
TARGET_MEMORY_REDUCTION = 0.9  # 90% reduction
TARGET_BATCH_EFFICIENCY = 0.8  # 80% of theoretical maximum
```

## JIT Compilation Optimization

### Using equinox.filter_jit

Always use `equinox.filter_jit` for automatic static/dynamic argument handling:

```python
import equinox as eqx
from jaxarc.envs.functional import arc_reset, arc_step

# ✅ Good: Automatic JIT compilation
@eqx.filter_jit
def optimized_reset(key, config, task_data):
    """JIT-compiled reset function."""
    return arc_reset(key, config, task_data)

@eqx.filter_jit
def optimized_step(state, action, config):
    """JIT-compiled step function."""
    return arc_step(state, action, config)

# ❌ Bad: Manual static_argnames (would fail with unhashable config)
# @jax.jit(static_argnames=['config'])  # Would fail!
# def bad_step(state, action, config): ...
```

### JIT Compilation Best Practices

#### 1. Warm Up Functions

Always warm up JIT functions before benchmarking:

```python
def warm_up_functions(config, task):
    """Warm up JIT-compiled functions."""
    key = jax.random.PRNGKey(42)
    
    # Warm up reset
    state, obs = arc_reset(key, config, task)
    
    # Warm up step
    action = PointAction(
        operation=jnp.array(0),
        row=jnp.array(5),
        col=jnp.array(5)
    )
    arc_step(state, action, config)
    
    print("✅ Functions warmed up")

# Always warm up before benchmarking
warm_up_functions(config, task)
```

#### 2. Consistent Function Signatures

Keep function signatures consistent to avoid recompilation:

```python
# ✅ Good: Consistent signatures
@eqx.filter_jit
def process_action(state, action, config):
    """Process any structured action type."""
    return arc_step(state, action, config)

# Use with different action types (same signature)
point_result = process_action(state, point_action, config)
bbox_result = process_action(state, bbox_action, config)
mask_result = process_action(state, mask_action, config)

# ❌ Bad: Different signatures cause recompilation
# def process_point_action(state, point_action, config): ...
# def process_bbox_action(state, bbox_action, config): ...
```

#### 3. Hashable Configurations

Ensure all configuration objects are hashable:

```python
def verify_config_hashability(config):
    """Verify that configuration is hashable."""
    try:
        hash(config)
        print("✅ Configuration is hashable")
        return True
    except TypeError as e:
        print(f"❌ Configuration is not hashable: {e}")
        return False

# Test your configuration
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    action=UnifiedActionConfig(selection_format="point")
)

verify_config_hashability(config)
```

### JIT Performance Profiling

Profile JIT compilation overhead:

```python
import time

def profile_jit_compilation(config, task, num_runs=100):
    """Profile JIT compilation performance."""
    key = jax.random.PRNGKey(42)
    
    # Create JIT functions
    @eqx.filter_jit
    def jit_reset(key, config, task_data):
        return arc_reset(key, config, task_data)
    
    @eqx.filter_jit
    def jit_step(state, action, config):
        return arc_step(state, action, config)
    
    # Measure compilation time
    print("Measuring JIT compilation time...")
    
    # First call includes compilation
    start_time = time.perf_counter()
    state, obs = jit_reset(key, config, task)
    compilation_time = time.perf_counter() - start_time
    
    # Subsequent calls are fast
    start_time = time.perf_counter()
    for _ in range(num_runs):
        state, obs = jit_reset(key, config, task)
    execution_time = (time.perf_counter() - start_time) / num_runs
    
    print(f"Compilation time: {compilation_time*1000:.1f}ms")
    print(f"Execution time: {execution_time*1000:.3f}ms")
    print(f"Speedup after compilation: {compilation_time/execution_time:.0f}x")
    
    return {
        'compilation_time': compilation_time,
        'execution_time': execution_time,
        'speedup': compilation_time / execution_time
    }

# Profile JIT performance
jit_profile = profile_jit_compilation(config, task)
```

## Memory Management Optimization

### Action Format Selection

Choose the most memory-efficient action format for your use case:

```python
def analyze_memory_usage_by_format():
    """Analyze memory usage for different action formats."""
    formats = ['point', 'bbox', 'mask']
    memory_usage = {}
    
    for format_name in formats:
        config = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=1000),
            action=UnifiedActionConfig(selection_format=format_name)
        )
        
        # Create state
        key = jax.random.PRNGKey(42)
        task = create_mock_task()
        state, obs = arc_reset(key, config, task)
        
        # Calculate memory usage
        action_history_mb = state.action_history.nbytes / (1024 * 1024)
        
        memory_usage[format_name] = {
            'action_history_mb': action_history_mb,
            'fields_per_action': get_fields_per_action(format_name)
        }
        
        print(f"{format_name.capitalize()} format: {action_history_mb:.3f} MB")
    
    # Calculate savings
    mask_memory = memory_usage['mask']['action_history_mb']
    point_savings = (1 - memory_usage['point']['action_history_mb'] / mask_memory) * 100
    bbox_savings = (1 - memory_usage['bbox']['action_history_mb'] / mask_memory) * 100
    
    print(f"\nMemory savings:")
    print(f"Point vs Mask: {point_savings:.1f}% reduction")
    print(f"Bbox vs Mask: {bbox_savings:.1f}% reduction")
    
    return memory_usage

def get_fields_per_action(format_name):
    """Get number of fields per action for each format."""
    return {
        'point': 6,   # operation, row, col, timestamp, pair_index, valid
        'bbox': 8,    # operation, r1, c1, r2, c2, timestamp, pair_index, valid
        'mask': 904   # operation + 30*30 mask + timestamp, pair_index, valid
    }[format_name]

# Analyze memory usage
memory_analysis = analyze_memory_usage_by_format()
```

### Memory-Efficient Configuration

Optimize configuration for memory usage:

```python
def create_memory_optimized_config(
    max_episode_steps=50,      # Reduce episode length
    grid_size=15,              # Smaller grids when possible
    action_format="point"      # Most memory-efficient format
):
    """Create memory-optimized configuration."""
    return JaxArcConfig(
        environment=EnvironmentConfig(
            max_episode_steps=max_episode_steps,
            debug_level=0  # Disable debug features
        ),
        dataset=UnifiedDatasetConfig(
            max_grid_height=grid_size,
            max_grid_width=grid_size
        ),
        action=UnifiedActionConfig(
            selection_format=action_format
        ),
        visualization=VisualizationConfig(
            enabled=False  # Disable visualization for performance
        )
    )

# Create optimized config
optimized_config = create_memory_optimized_config()
```

### Memory Profiling

Profile memory usage across different scenarios:

```python
def profile_memory_usage(config, task, batch_sizes=[1, 16, 64, 256]):
    """Profile memory usage for different batch sizes."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_profile = {}
    
    for batch_size in batch_sizes:
        # Measure memory before
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create batch
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        batch_states, batch_obs = batch_reset(keys, config, task)
        
        # Measure memory after
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = memory_after - memory_before
        
        memory_profile[batch_size] = {
            'total_memory_mb': memory_used,
            'per_env_memory_mb': memory_used / batch_size,
            'memory_before': memory_before,
            'memory_after': memory_after
        }
        
        print(f"Batch {batch_size:3d}: {memory_used:.1f}MB total, "
              f"{memory_used/batch_size:.3f}MB per env")
        
        # Clean up
        del batch_states, batch_obs
    
    return memory_profile

# Profile memory usage
memory_profile = profile_memory_usage(optimized_config, task)
```

### Memory Optimization Strategies

#### 1. Lazy Loading

Implement lazy loading for large data structures:

```python
class LazyTaskLoader:
    """Lazy loader for task data."""
    
    def __init__(self, task_index, parser):
        self.task_index = task_index
        self.parser = parser
        self._task_data = None
    
    @property
    def task_data(self):
        """Load task data on first access."""
        if self._task_data is None:
            self._task_data = self.parser.get_task_by_index(self.task_index)
        return self._task_data
    
    def clear_cache(self):
        """Clear cached task data to free memory."""
        self._task_data = None

# Usage
lazy_loader = LazyTaskLoader(0, parser)
task_data = lazy_loader.task_data  # Loads on first access
lazy_loader.clear_cache()  # Free memory when done
```

#### 2. Memory Pooling

Implement memory pooling for frequent allocations:

```python
class StateMemoryPool:
    """Memory pool for environment states."""
    
    def __init__(self, pool_size=100, config=None):
        self.pool_size = pool_size
        self.config = config
        self.available_states = []
        self.in_use_states = set()
    
    def get_state(self):
        """Get a state from the pool."""
        if self.available_states:
            state = self.available_states.pop()
            self.in_use_states.add(id(state))
            return state
        else:
            # Create new state if pool is empty
            return self._create_new_state()
    
    def return_state(self, state):
        """Return a state to the pool."""
        state_id = id(state)
        if state_id in self.in_use_states:
            self.in_use_states.remove(state_id)
            if len(self.available_states) < self.pool_size:
                self.available_states.append(state)
    
    def _create_new_state(self):
        """Create a new state."""
        key = jax.random.PRNGKey(42)
        task = create_mock_task()
        state, _ = arc_reset(key, self.config, task)
        return state

# Usage
pool = StateMemoryPool(pool_size=50, config=optimized_config)
state = pool.get_state()
# ... use state ...
pool.return_state(state)
```

## Batch Processing Optimization

### Optimal Batch Size Selection

Find the optimal batch size for your hardware:

```python
def find_optimal_batch_size(config, task, max_batch_size=1024, target_memory_gb=8):
    """Find optimal batch size based on performance and memory constraints."""
    import time
    import psutil
    
    batch_sizes = [2**i for i in range(0, 11) if 2**i <= max_batch_size]  # 1, 2, 4, 8, ..., 1024
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Memory check
            keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
            
            # Estimate memory usage
            memory_before = psutil.Process().memory_info().rss / (1024**3)  # GB
            batch_states, batch_obs = batch_reset(keys, config, task)
            memory_after = psutil.Process().memory_info().rss / (1024**3)  # GB
            memory_used = memory_after - memory_before
            
            if memory_used > target_memory_gb:
                print(f"Batch {batch_size}: Exceeds memory limit ({memory_used:.1f}GB > {target_memory_gb}GB)")
                break
            
            # Performance benchmark
            actions = PointAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                row=jnp.full(batch_size, 5, dtype=jnp.int32),
                col=jnp.full(batch_size, 5, dtype=jnp.int32)
            )
            
            # Warm up
            batch_step(batch_states, actions, config)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(10):
                batch_states, batch_obs, rewards, dones, infos = batch_step(
                    batch_states, actions, config
                )
            elapsed = (time.perf_counter() - start_time) / 10
            
            throughput = batch_size / elapsed
            per_env_time = elapsed / batch_size
            
            results[batch_size] = {
                'throughput': throughput,
                'per_env_time': per_env_time,
                'memory_used_gb': memory_used,
                'efficiency': throughput / batch_size  # Higher is better
            }
            
            print(f"Batch {batch_size:4d}: {throughput:6.0f} envs/sec, "
                  f"{per_env_time*1000:5.2f}ms/env, {memory_used:.2f}GB")
            
            # Clean up
            del batch_states, batch_obs
            
        except Exception as e:
            print(f"Batch {batch_size}: Failed ({e})")
            break
    
    # Find optimal batch size
    if results:
        optimal_batch_size = max(results.keys(), key=lambda bs: results[bs]['throughput'])
        optimal_throughput = results[optimal_batch_size]['throughput']
        
        print(f"\n🎯 Optimal batch size: {optimal_batch_size}")
        print(f"   Throughput: {optimal_throughput:.0f} envs/sec")
        print(f"   Memory usage: {results[optimal_batch_size]['memory_used_gb']:.2f}GB")
        
        return optimal_batch_size, results
    else:
        return None, {}

# Find optimal batch size
optimal_batch_size, batch_results = find_optimal_batch_size(optimized_config, task)
```

### Batch Processing Patterns

#### 1. Hierarchical Batching

Process multiple levels of batching for maximum throughput:

```python
def hierarchical_batch_processing(
    num_workers=4,
    envs_per_worker=64,
    episodes_per_worker=10,
    config=None,
    task=None
):
    """Hierarchical batch processing across workers and environments."""
    
    total_envs = num_workers * envs_per_worker
    total_episodes = num_workers * episodes_per_worker
    
    print(f"Processing {total_episodes} episodes across {total_envs} environments")
    
    # Create worker-specific keys
    worker_keys = jax.random.split(jax.random.PRNGKey(42), num_workers)
    
    all_results = []
    
    for worker_id in range(num_workers):
        print(f"Worker {worker_id + 1}/{num_workers}")
        
        # Create environment keys for this worker
        env_keys = jax.random.split(worker_keys[worker_id], envs_per_worker)
        
        # Process episodes for this worker
        worker_results = []
        
        for episode in range(episodes_per_worker):
            # Reset environments
            batch_states, batch_obs = batch_reset(env_keys, config, task)
            
            # Run episode
            episode_rewards = jnp.zeros(envs_per_worker)
            
            for step in range(50):  # 50 steps per episode
                # Create actions
                actions = PointAction(
                    operation=jnp.zeros(envs_per_worker, dtype=jnp.int32),
                    row=jnp.full(envs_per_worker, step % 10 + 5, dtype=jnp.int32),
                    col=jnp.full(envs_per_worker, 5, dtype=jnp.int32)
                )
                
                # Step environments
                batch_states, batch_obs, rewards, dones, infos = batch_step(
                    batch_states, actions, config
                )
                
                episode_rewards += rewards
            
            worker_results.append(episode_rewards)
        
        all_results.extend(worker_results)
    
    # Analyze results
    all_rewards = jnp.concatenate(all_results)
    
    print(f"\n📊 Results:")
    print(f"   Total episodes: {len(all_results)}")
    print(f"   Total environment steps: {len(all_results) * envs_per_worker}")
    print(f"   Mean reward: {jnp.mean(all_rewards):.3f}")
    print(f"   Std reward: {jnp.std(all_rewards):.3f}")
    
    return all_results

# Run hierarchical processing
hierarchical_results = hierarchical_batch_processing(
    num_workers=2,
    envs_per_worker=32,
    episodes_per_worker=5,
    config=optimized_config,
    task=task
)
```

#### 2. Dynamic Batch Sizing

Adapt batch size based on available memory:

```python
class DynamicBatchProcessor:
    """Dynamically adjust batch size based on memory usage."""
    
    def __init__(self, config, task, initial_batch_size=64, memory_limit_gb=4):
        self.config = config
        self.task = task
        self.current_batch_size = initial_batch_size
        self.memory_limit_gb = memory_limit_gb
        self.performance_history = []
    
    def process_batch(self, num_episodes=10):
        """Process batch with dynamic sizing."""
        import psutil
        import time
        
        results = []
        
        for episode in range(num_episodes):
            # Check memory usage
            memory_gb = psutil.Process().memory_info().rss / (1024**3)
            
            if memory_gb > self.memory_limit_gb * 0.8:  # 80% threshold
                self._reduce_batch_size()
            elif memory_gb < self.memory_limit_gb * 0.4:  # 40% threshold
                self._increase_batch_size()
            
            # Process episode
            keys = jax.random.split(jax.random.PRNGKey(episode), self.current_batch_size)
            
            start_time = time.perf_counter()
            batch_states, batch_obs = batch_reset(keys, self.config, self.task)
            
            # Run episode steps
            episode_rewards = jnp.zeros(self.current_batch_size)
            
            for step in range(20):
                actions = PointAction(
                    operation=jnp.zeros(self.current_batch_size, dtype=jnp.int32),
                    row=jnp.full(self.current_batch_size, 5, dtype=jnp.int32),
                    col=jnp.full(self.current_batch_size, 5, dtype=jnp.int32)
                )
                
                batch_states, batch_obs, rewards, dones, infos = batch_step(
                    batch_states, actions, self.config
                )
                
                episode_rewards += rewards
            
            elapsed = time.perf_counter() - start_time
            throughput = self.current_batch_size / elapsed
            
            self.performance_history.append({
                'episode': episode,
                'batch_size': self.current_batch_size,
                'throughput': throughput,
                'memory_gb': memory_gb
            })
            
            results.append(episode_rewards)
            
            if episode % 5 == 0:
                print(f"Episode {episode}: batch_size={self.current_batch_size}, "
                      f"throughput={throughput:.0f} envs/sec, memory={memory_gb:.1f}GB")
        
        return results
    
    def _reduce_batch_size(self):
        """Reduce batch size to save memory."""
        new_size = max(1, self.current_batch_size // 2)
        if new_size != self.current_batch_size:
            print(f"Reducing batch size: {self.current_batch_size} → {new_size}")
            self.current_batch_size = new_size
    
    def _increase_batch_size(self):
        """Increase batch size for better throughput."""
        new_size = min(512, self.current_batch_size * 2)
        if new_size != self.current_batch_size:
            print(f"Increasing batch size: {self.current_batch_size} → {new_size}")
            self.current_batch_size = new_size

# Use dynamic batch processor
processor = DynamicBatchProcessor(optimized_config, task)
dynamic_results = processor.process_batch(num_episodes=20)
```

## Profiling and Debugging

### Performance Profiling Tools

#### 1. JAX Profiler Integration

Use JAX's built-in profiling tools:

```python
def profile_with_jax_profiler(config, task, output_dir="./jax_profile"):
    """Profile JaxARC using JAX profiler."""
    import jax.profiler
    
    # Start profiling
    jax.profiler.start_trace(output_dir)
    
    try:
        # Run workload
        key = jax.random.PRNGKey(42)
        state, obs = arc_reset(key, config, task)
        
        for i in range(100):
            action = PointAction(
                operation=jnp.array(0),
                row=jnp.array(i % 10 + 5),
                col=jnp.array(5)
            )
            state, obs, reward, done, info = arc_step(state, action, config)
    
    finally:
        # Stop profiling
        jax.profiler.stop_trace()
    
    print(f"Profile saved to {output_dir}")
    print("View with: tensorboard --logdir={output_dir}")

# Profile with JAX profiler
# profile_with_jax_profiler(optimized_config, task)
```

#### 2. Custom Performance Monitor

Create a custom performance monitoring system:

```python
class PerformanceMonitor:
    """Monitor JaxARC performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'step_times': [],
            'reset_times': [],
            'memory_usage': [],
            'throughput': [],
            'jit_compilation_times': {}
        }
    
    def time_function(self, func, *args, **kwargs):
        """Time a function call."""
        import time
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        return result, elapsed
    
    def monitor_step(self, state, action, config):
        """Monitor step performance."""
        result, elapsed = self.time_function(arc_step, state, action, config)
        self.metrics['step_times'].append(elapsed)
        return result
    
    def monitor_reset(self, key, config, task):
        """Monitor reset performance."""
        result, elapsed = self.time_function(arc_reset, key, config, task)
        self.metrics['reset_times'].append(elapsed)
        return result
    
    def monitor_batch_step(self, states, actions, config):
        """Monitor batch step performance."""
        batch_size = states.working_grid.shape[0]
        result, elapsed = self.time_function(batch_step, states, actions, config)
        
        throughput = batch_size / elapsed
        self.metrics['throughput'].append(throughput)
        
        return result
    
    def get_statistics(self):
        """Get performance statistics."""
        stats = {}
        
        if self.metrics['step_times']:
            step_times = jnp.array(self.metrics['step_times'])
            stats['step_time_mean'] = float(jnp.mean(step_times))
            stats['step_time_std'] = float(jnp.std(step_times))
            stats['step_time_min'] = float(jnp.min(step_times))
            stats['step_time_max'] = float(jnp.max(step_times))
        
        if self.metrics['reset_times']:
            reset_times = jnp.array(self.metrics['reset_times'])
            stats['reset_time_mean'] = float(jnp.mean(reset_times))
            stats['reset_time_std'] = float(jnp.std(reset_times))
        
        if self.metrics['throughput']:
            throughput = jnp.array(self.metrics['throughput'])
            stats['throughput_mean'] = float(jnp.mean(throughput))
            stats['throughput_max'] = float(jnp.max(throughput))
        
        return stats
    
    def print_report(self):
        """Print performance report."""
        stats = self.get_statistics()
        
        print("📊 Performance Report")
        print("=" * 50)
        
        if 'step_time_mean' in stats:
            print(f"Step Time:")
            print(f"  Mean: {stats['step_time_mean']*1000:.2f}ms")
            print(f"  Std:  {stats['step_time_std']*1000:.2f}ms")
            print(f"  Min:  {stats['step_time_min']*1000:.2f}ms")
            print(f"  Max:  {stats['step_time_max']*1000:.2f}ms")
        
        if 'reset_time_mean' in stats:
            print(f"Reset Time:")
            print(f"  Mean: {stats['reset_time_mean']*1000:.2f}ms")
            print(f"  Std:  {stats['reset_time_std']*1000:.2f}ms")
        
        if 'throughput_mean' in stats:
            print(f"Throughput:")
            print(f"  Mean: {stats['throughput_mean']:.0f} envs/sec")
            print(f"  Max:  {stats['throughput_max']:.0f} envs/sec")

# Use performance monitor
monitor = PerformanceMonitor()

# Monitor operations
key = jax.random.PRNGKey(42)
state, obs = monitor.monitor_reset(key, optimized_config, task)

for i in range(50):
    action = PointAction(
        operation=jnp.array(0),
        row=jnp.array(i % 10 + 5),
        col=jnp.array(5)
    )
    state, obs, reward, done, info = monitor.monitor_step(state, action, optimized_config)

# Print report
monitor.print_report()
```

### Debugging Performance Issues

#### 1. JIT Compilation Issues

Debug JIT compilation problems:

```python
def debug_jit_compilation(config, task):
    """Debug JIT compilation issues."""
    
    # Test configuration hashability
    try:
        hash(config)
        print("✅ Configuration is hashable")
    except TypeError as e:
        print(f"❌ Configuration is not hashable: {e}")
        return
    
    # Test JIT compilation
    try:
        @eqx.filter_jit
        def test_reset(key, config, task_data):
            return arc_reset(key, config, task_data)
        
        @eqx.filter_jit
        def test_step(state, action, config):
            return arc_step(state, action, config)
        
        # Test compilation
        key = jax.random.PRNGKey(42)
        state, obs = test_reset(key, config, task)
        
        action = PointAction(
            operation=jnp.array(0),
            row=jnp.array(5),
            col=jnp.array(5)
        )
        
        state, obs, reward, done, info = test_step(state, action, config)
        
        print("✅ JIT compilation successful")
        
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
        import traceback
        traceback.print_exc()

# Debug JIT compilation
debug_jit_compilation(optimized_config, task)
```

#### 2. Memory Leak Detection

Detect memory leaks in long-running processes:

```python
def detect_memory_leaks(config, task, num_iterations=1000):
    """Detect memory leaks in JaxARC operations."""
    import psutil
    import gc
    
    process = psutil.Process()
    memory_samples = []
    
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, config, task)
    
    for i in range(num_iterations):
        # Sample memory usage
        if i % 100 == 0:
            gc.collect()  # Force garbage collection
            memory_mb = process.memory_info().rss / (1024 * 1024)
            memory_samples.append((i, memory_mb))
            print(f"Iteration {i}: {memory_mb:.1f}MB")
        
        # Perform operations
        action = PointAction(
            operation=jnp.array(0),
            row=jnp.array(i % 10 + 5),
            col=jnp.array(5)
        )
        
        state, obs, reward, done, info = arc_step(state, action, config)
        
        # Reset occasionally
        if i % 100 == 0:
            state, obs = arc_reset(key, config, task)
    
    # Analyze memory trend
    if len(memory_samples) > 2:
        initial_memory = memory_samples[0][1]
        final_memory = memory_samples[-1][1]
        memory_growth = final_memory - initial_memory
        
        print(f"\n📊 Memory Analysis:")
        print(f"   Initial memory: {initial_memory:.1f}MB")
        print(f"   Final memory: {final_memory:.1f}MB")
        print(f"   Memory growth: {memory_growth:.1f}MB")
        
        if memory_growth > 100:  # More than 100MB growth
            print("⚠️  Potential memory leak detected")
        else:
            print("✅ No significant memory leak detected")
    
    return memory_samples

# Detect memory leaks
memory_samples = detect_memory_leaks(optimized_config, task, num_iterations=500)
```

## Advanced Optimization Techniques

### 1. Custom JAX Transformations

Create custom JAX transformations for specific use cases:

```python
def create_optimized_episode_runner(config, task, max_steps=50):
    """Create optimized episode runner with custom transformations."""
    
    @eqx.filter_jit
    def run_episode(key, policy_params):
        """Run complete episode with JIT compilation."""
        # Reset environment
        state, obs = arc_reset(key, config, task)
        
        # Initialize episode data
        episode_rewards = jnp.array(0.0)
        episode_length = jnp.array(0)
        
        # Episode loop (unrolled for JIT efficiency)
        def episode_step(carry, step_idx):
            state, episode_rewards, episode_length = carry
            
            # Generate action (simplified policy)
            action = PointAction(
                operation=jnp.array(0),
                row=jnp.array(step_idx % 10 + 5),
                col=jnp.array(5)
            )
            
            # Step environment
            new_state, obs, reward, done, info = arc_step(state, action, config)
            
            # Update episode data
            new_episode_rewards = episode_rewards + reward
            new_episode_length = episode_length + 1
            
            # Early termination condition
            should_continue = ~done & (episode_length < max_steps)
            
            return (
                jnp.where(should_continue, new_state, state),
                jnp.where(should_continue, new_episode_rewards, episode_rewards),
                jnp.where(should_continue, new_episode_length, episode_length)
            ), None
        
        # Run episode steps
        (final_state, final_rewards, final_length), _ = jax.lax.scan(
            episode_step,
            (state, episode_rewards, episode_length),
            jnp.arange(max_steps)
        )
        
        return {
            'final_state': final_state,
            'episode_reward': final_rewards,
            'episode_length': final_length
        }
    
    return run_episode

# Create and use optimized episode runner
episode_runner = create_optimized_episode_runner(optimized_config, task)

# Run episodes
keys = jax.random.split(jax.random.PRNGKey(42), 10)
policy_params = None  # Placeholder

results = []
for key in keys:
    result = episode_runner(key, policy_params)
    results.append(result)

print(f"Average episode reward: {jnp.mean([r['episode_reward'] for r in results]):.3f}")
```

### 2. Memory-Mapped Data Loading

Use memory-mapped files for large datasets:

```python
class MemoryMappedTaskLoader:
    """Memory-mapped task loader for large datasets."""
    
    def __init__(self, data_path, max_tasks=1000):
        self.data_path = data_path
        self.max_tasks = max_tasks
        self._mmap_data = None
        self._task_index = None
    
    def _initialize_mmap(self):
        """Initialize memory-mapped data."""
        if self._mmap_data is None:
            # Create memory-mapped array for task data
            # This is a simplified example - actual implementation would
            # depend on your data format
            import numpy as np
            
            # Create dummy memory-mapped array
            self._mmap_data = np.memmap(
                'task_data.dat',
                dtype=np.float32,
                mode='w+',
                shape=(self.max_tasks, 30, 30, 4)  # tasks, height, width, channels
            )
            
            # Initialize with dummy data
            self._mmap_data[:] = np.random.rand(self.max_tasks, 30, 30, 4)
            
            self._task_index = np.arange(self.max_tasks)
    
    def get_task(self, task_id):
        """Get task data without loading entire dataset into memory."""
        self._initialize_mmap()
        
        if task_id >= self.max_tasks:
            raise ValueError(f"Task ID {task_id} exceeds maximum {self.max_tasks}")
        
        # Return task data as JAX array
        task_data = jnp.array(self._mmap_data[task_id])
        
        return create_jax_task_from_data(task_data, task_id)
    
    def get_batch_tasks(self, task_ids):
        """Get batch of tasks efficiently."""
        self._initialize_mmap()
        
        # Load batch of tasks
        batch_data = jnp.array(self._mmap_data[task_ids])
        
        return [
            create_jax_task_from_data(batch_data[i], task_ids[i])
            for i in range(len(task_ids))
        ]

def create_jax_task_from_data(task_data, task_id):
    """Create JaxArcTask from raw data."""
    # Simplified task creation
    return JaxArcTask(
        input_grids_examples=task_data[:3],
        input_masks_examples=jnp.ones((3, 30, 30), dtype=jnp.bool_),
        output_grids_examples=task_data[:3],
        output_masks_examples=jnp.ones((3, 30, 30), dtype=jnp.bool_),
        num_train_pairs=3,
        test_input_grids=task_data[3:4],
        test_input_masks=jnp.ones((1, 30, 30), dtype=jnp.bool_),
        true_test_output_grids=task_data[3:4],
        true_test_output_masks=jnp.ones((1, 30, 30), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=jnp.array(task_id, dtype=jnp.int32)
    )

# Use memory-mapped loader
# mmap_loader = MemoryMappedTaskLoader('large_dataset.dat')
# task = mmap_loader.get_task(0)
```

## Performance Optimization Checklist

Use this checklist to ensure optimal performance:

### JIT Compilation
- [ ] All core functions use `equinox.filter_jit`
- [ ] Configuration objects are hashable
- [ ] Functions are warmed up before benchmarking
- [ ] Function signatures are consistent
- [ ] No abstract array interpretation errors

### Memory Management
- [ ] Appropriate action format selected (point/bbox vs mask)
- [ ] Episode length optimized for use case
- [ ] Memory usage profiled and optimized
- [ ] Large objects cleaned up when not needed
- [ ] Memory leaks detected and fixed

### Batch Processing
- [ ] Optimal batch size determined
- [ ] PRNG keys properly split
- [ ] Batch operations use consistent shapes
- [ ] Memory constraints respected
- [ ] Throughput measured and optimized

### Profiling & Debugging
- [ ] Performance monitoring implemented
- [ ] Bottlenecks identified and addressed
- [ ] Memory usage tracked
- [ ] JIT compilation issues resolved
- [ ] Error handling optimized for performance

### Advanced Optimizations
- [ ] Custom JAX transformations where beneficial
- [ ] Memory-mapped data loading for large datasets
- [ ] Hierarchical batch processing implemented
- [ ] Dynamic batch sizing considered
- [ ] Performance regression tests in place

## Troubleshooting Common Issues

### Issue 1: Slow JIT Compilation

**Symptoms**: Long delays on first function call
**Solutions**:
- Warm up functions during initialization
- Use consistent function signatures
- Avoid complex control flow in JIT functions

### Issue 2: High Memory Usage

**Symptoms**: Out of memory errors, slow performance
**Solutions**:
- Use point/bbox actions instead of mask actions
- Reduce batch size or episode length
- Implement memory pooling
- Profile memory usage patterns

### Issue 3: Poor Batch Performance

**Symptoms**: Batch processing slower than expected
**Solutions**:
- Find optimal batch size for your hardware
- Ensure consistent batch shapes
- Use proper PRNG key management
- Profile batch scaling efficiency

### Issue 4: Inconsistent Performance

**Symptoms**: Variable execution times, unexpected slowdowns
**Solutions**:
- Check for JIT recompilation
- Monitor memory usage patterns
- Implement performance monitoring
- Use deterministic operations

## Summary

Performance optimization in JaxARC involves:

1. **JIT Compilation**: Use `equinox.filter_jit` with hashable configurations
2. **Memory Management**: Choose appropriate action formats and optimize memory usage
3. **Batch Processing**: Find optimal batch sizes and use efficient batching patterns
4. **Profiling**: Monitor performance and identify bottlenecks
5. **Advanced Techniques**: Implement custom optimizations for specific use cases

Following these guidelines can achieve:
- **100x+ speedup** through JIT compilation
- **90%+ memory reduction** with efficient action formats
- **10,000+ steps/sec** throughput with batch processing
- **Consistent performance** through proper profiling and optimization

For practical examples of these optimization techniques, see the example scripts in `examples/advanced/`.