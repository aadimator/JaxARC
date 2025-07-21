# JAX Integration and Performance Optimization Guide

This guide covers the JAX-compatible callback system and memory management
features implemented for the JaxARC visualization system.

## Overview

The enhanced visualization system provides JAX-compatible callbacks and memory
optimization features that maintain performance while enabling rich
visualization capabilities during training and debugging.

## Key Features

### 1. JAX-Compatible Callback System

The callback system ensures visualization functions work correctly with JAX
transformations:

- **Safe Error Handling**: Callback errors don't break JAX execution
- **Performance Monitoring**: Track callback performance impact
- **Array Serialization**: Proper handling of JAX arrays in callbacks
- **Adaptive Logging**: Reduce callback frequency based on performance impact

### 2. Memory Management and Optimization

Comprehensive memory management for visualization workloads:

- **Lazy Loading**: Load large datasets only when needed
- **Compressed Storage**: Efficient storage with automatic compression
- **Visualization Cache**: Memory-efficient caching with LRU eviction
- **Garbage Collection Optimization**: Tuned GC settings for visualization
- **Array Optimization**: Automatic dtype optimization to reduce memory usage

## Usage Examples

### Basic JAX Callback Usage

```python
import jax
import jax.numpy as jnp
from jaxarc.utils.visualization import jax_log_grid, jax_log_episode_summary


@jax.jit
def process_grid(grid_data, mask_data):
    """Process grid with JAX transformations and logging."""
    # Log the input grid
    jax_log_grid(grid_data, mask_data, "Input Grid")

    # Perform computation
    processed = jnp.where(mask_data, grid_data + 1, 0)

    # Log the processed grid
    jax_log_grid(processed, mask_data, "Processed Grid")

    return processed


# Use with JAX arrays
grid = jnp.array([[1, 2, 0], [3, 4, 1], [0, 2, 3]])
mask = jnp.array([[True, True, False], [True, True, True], [False, True, True]])
result = process_grid(grid, mask)
```

### Episode Summary Logging

```python
@jax.jit
def simulate_episode(episode_num, steps, base_reward):
    """Simulate an episode with logging."""
    total_reward = base_reward * steps
    final_similarity = jnp.tanh(total_reward / 10.0)
    success = final_similarity > 0.8

    # Log episode summary
    jax_log_episode_summary(episode_num, steps, total_reward, final_similarity, success)

    return total_reward, final_similarity, success
```

### Performance Monitoring

```python
from jaxarc.utils.visualization import (
    get_callback_performance_stats,
    print_callback_performance_report,
    reset_callback_performance_stats,
)

# Reset stats for clean measurement
reset_callback_performance_stats()

# Run your JAX functions with callbacks
# ...

# Get performance report
print_callback_performance_report()

# Get specific stats
stats = get_callback_performance_stats()
for callback_name, callback_stats in stats.items():
    print(f"{callback_name}: {callback_stats['avg_time_ms']:.2f}ms average")
```

### Memory Management

```python
from jaxarc.utils.visualization import (
    MemoryManager,
    LazyLoader,
    CompressedStorage,
    optimize_array_memory,
)

# Use memory manager context
with MemoryManager(max_total_memory_mb=100.0) as manager:
    # Create lazy loader for expensive data
    def load_large_dataset():
        return {"data": jnp.random.rand(1000, 1000)}

    lazy_loader = LazyLoader(load_large_dataset)
    manager.register_lazy_loader(lazy_loader)

    # Load data only when needed
    data = lazy_loader.get()

    # Optimize arrays for memory efficiency
    optimized_data = optimize_array_memory(np.asarray(data["data"]))

    # Get memory report
    report = manager.get_memory_report()
    print(f"Memory usage: {report['monitor_stats']['current_mb']:.2f} MB")

# Automatic cleanup on context exit
```

### Compressed Storage

```python
from pathlib import Path
from jaxarc.utils.visualization import CompressedStorage

# Create compressed storage
storage = CompressedStorage(Path("visualization_cache"))

# Save large visualization data
viz_data = {
    "episode_grids": [jnp.random.randint(0, 10, (30, 30)) for _ in range(100)],
    "metadata": {"episode_id": "test_001", "total_steps": 100},
}

# Convert JAX arrays to numpy for storage
numpy_data = {
    "episode_grids": [np.asarray(grid) for grid in viz_data["episode_grids"]],
    "metadata": viz_data["metadata"],
}

# Save with compression
storage.save(numpy_data, "episode_001")

# Load when needed
loaded_data = storage.load("episode_001")

# Convert back to JAX arrays if needed
jax_data = {
    "episode_grids": [jnp.array(grid) for grid in loaded_data["episode_grids"]],
    "metadata": loaded_data["metadata"],
}
```

## Advanced Features

### Adaptive Visualization

The system automatically reduces callback frequency when performance impact is
high:

```python
from jaxarc.utils.visualization import adaptive_visualization_callback


def expensive_visualization(data):
    # Expensive visualization operation
    pass


@jax.jit
def training_step(state):
    # This will automatically reduce frequency if too slow
    adaptive_visualization_callback(
        expensive_visualization,
        state,
        callback_name="expensive_viz",
        max_avg_time_ms=5.0,  # Reduce frequency if average time > 5ms
    )
    return state + 1
```

### Custom Lazy Loaders

```python
from jaxarc.utils.visualization import create_lazy_visualization_loader


def custom_loader(path):
    """Custom loader for specific data format."""
    # Load and process data
    return processed_data


# Create lazy loader with automatic cleanup
lazy_loader = create_lazy_visualization_loader(
    Path("data/large_dataset.pkl"), custom_loader, max_idle_time=300.0  # 5 minutes
)

# Data is loaded only when accessed
data = lazy_loader.get()
```

### Array Memory Optimization

```python
from jaxarc.utils.visualization import optimize_array_memory

# Optimize different array types
arrays = [
    np.array([1, 2, 3, 4, 5], dtype=np.int64),  # -> int8
    np.array([1000, 2000, 3000], dtype=np.int64),  # -> int16
    np.array([1.1, 2.2, 3.3], dtype=np.float64),  # -> float32
]

total_savings = 0
for i, arr in enumerate(arrays):
    original_size = arr.nbytes
    optimized = optimize_array_memory(arr)
    savings = original_size - optimized.nbytes
    total_savings += savings

    print(f"Array {i}: {arr.dtype} -> {optimized.dtype}, saved {savings} bytes")

print(f"Total memory savings: {total_savings} bytes")
```

## Performance Considerations

### JAX Compatibility

- All callbacks use `jax.debug.callback` for proper JAX integration
- Array serialization handles JAX arrays safely
- Error handling prevents callback failures from breaking JAX execution
- Performance monitoring tracks callback overhead

### Memory Efficiency

- Lazy loading prevents unnecessary memory usage
- Compressed storage reduces disk space requirements
- LRU cache eviction manages memory limits
- Garbage collection optimization reduces pause times
- Array dtype optimization minimizes memory footprint

### Best Practices

1. **Use Performance Monitoring**: Always monitor callback performance in
   training loops
2. **Implement Lazy Loading**: For large datasets that aren't always needed
3. **Optimize Arrays**: Use `optimize_array_memory` for long-term storage
4. **Manage Memory**: Use `MemoryManager` context for automatic cleanup
5. **Handle Errors Gracefully**: Callback errors shouldn't break training

## Configuration

The system integrates with Hydra configuration:

```yaml
# conf/visualization/debug_standard.yaml
debug:
  level: "standard"
  async_logging: true
  performance_monitoring: true
  memory_management:
    max_cache_mb: 100.0
    max_total_mb: 500.0
    lazy_loading: true
    compression: true
```

## Troubleshooting

### Common Issues

1. **High Callback Overhead**: Use adaptive callbacks or reduce logging
   frequency
2. **Memory Usage**: Enable memory management and use lazy loading
3. **JAX Compilation Errors**: Ensure callbacks don't use traced values
   inappropriately
4. **Storage Issues**: Use compressed storage for large datasets

### Debugging Tools

```python
# Check callback performance
from jaxarc.utils.visualization import print_callback_performance_report

print_callback_performance_report()

# Monitor memory usage
from jaxarc.utils.visualization import get_memory_manager

manager = get_memory_manager()
report = manager.get_memory_report()
print(f"Memory report: {report}")

# Force cleanup if needed
cleanup_stats = manager.cleanup_memory()
print(f"Cleanup freed: {cleanup_stats}")
```

## Integration with Existing Code

The JAX callback system integrates seamlessly with existing visualization
functions:

```python
# Existing function
from jaxarc.utils.visualization import log_grid_to_console

# JAX-compatible version
from jaxarc.utils.visualization import jax_log_grid


@jax.jit
def my_function(grid_data):
    # Use JAX-compatible version in JIT-compiled functions
    jax_log_grid(grid_data, title="Debug Grid")
    return grid_data * 2


# Outside JAX context, either works
log_grid_to_console(grid_data, title="Debug Grid")
```

This system provides a comprehensive solution for visualization in JAX-based
training while maintaining performance and memory efficiency.
