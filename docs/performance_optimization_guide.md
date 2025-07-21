# Performance Optimization Guide

This guide covers optimizing the performance of JaxARC's enhanced visualization
and logging system to minimize impact on training while maximizing debugging
capabilities.

## Performance Overview

The enhanced visualization system is designed to have minimal impact on JAX
performance through:

- **Asynchronous processing**: Visualization operations run in background
  threads
- **JAX debug callbacks**: Integration that doesn't break JIT compilation
- **Memory management**: Efficient memory usage and cleanup
- **Adaptive logging**: Performance-based adjustment of logging frequency
- **Batch operations**: Grouping I/O operations for efficiency

## Quick Performance Setup

### Minimal Impact Configuration

```python
from jaxarc.utils.visualization import (
    VisualizationConfig,
    AsyncLoggerConfig,
    EpisodeConfig,
    EnhancedVisualizer,
)

# Performance-optimized configuration
vis_config = VisualizationConfig(
    debug_level="minimal",  # Minimal visualization
    output_formats=["svg"],  # Single format
    image_quality="medium",  # Reduced quality
    lazy_loading=True,  # Enable lazy loading
    memory_limit_mb=200,  # Limit memory usage
)

# Async logger for background processing
async_config = AsyncLoggerConfig(
    queue_size=500,  # Smaller queue
    worker_threads=1,  # Single worker thread
    batch_size=20,  # Larger batches
    flush_interval=10.0,  # Less frequent flushing
)

# Episode management with cleanup
episode_config = EpisodeConfig(
    cleanup_policy="size_based",
    max_storage_gb=2.0,  # Limit storage
    max_episodes_per_run=100,
)

# Create optimized visualizer
visualizer = EnhancedVisualizer(
    vis_config=vis_config,
    async_logger_config=async_config,
    episode_config=episode_config,
)
```

### JAX-Optimized Training Loop

```python
import jax
import jax.numpy as jnp
from jaxarc.utils.visualization import jax_debug_callback


# JIT-compiled step function with visualization
@jax.jit
def optimized_training_step(state, action, config, visualizer_callback):
    """JIT-compiled training step with minimal visualization overhead."""

    # Environment step
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Visualization callback (runs outside JIT)
    jax.debug.callback(visualizer_callback, state, action, new_state, reward, info)

    return new_state, obs, reward, done, info


# Lightweight visualization callback
def lightweight_viz_callback(state, action, new_state, reward, info):
    """Lightweight visualization callback for JIT compatibility."""

    # Only visualize every N steps to reduce overhead
    step_num = info.get("step_num", 0)
    if step_num % 10 == 0:  # Every 10th step
        visualizer.visualize_step_async(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step_num,
        )


# Training loop
key = jax.random.PRNGKey(42)
state, obs = arc_reset(key, config)

for step in range(1000):
    action = agent.select_action(obs)

    # Use JIT-compiled step with visualization
    state, obs, reward, done, info = optimized_training_step(
        state, action, config, lightweight_viz_callback
    )

    if done:
        break
```

## Performance Monitoring

### Built-in Performance Monitor

```python
from jaxarc.utils.visualization import PerformanceMonitor

# Initialize performance monitor
perf_monitor = PerformanceMonitor(
    target_overhead=0.05,  # 5% maximum overhead
    measurement_window=100,  # Measure over 100 steps
    auto_adjust=True,  # Automatically adjust settings
)


# Measure step performance
@perf_monitor.measure_step_impact
def step_with_monitoring(state, action, config):
    """Step function with performance monitoring."""

    # Environment step
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Visualization (monitored)
    visualizer.visualize_step(state, action, new_state, reward, info, step_num)

    return new_state, obs, reward, done, info


# Check performance regularly
if step % 100 == 0:
    report = perf_monitor.get_performance_report()

    print(f"Average step time: {report['avg_step_time']:.3f}s")
    print(f"Visualization overhead: {report['visualization_overhead']:.1f}%")

    # Auto-adjust if overhead too high
    if report["visualization_overhead"] > 0.1:  # 10%
        print("⚠️  High visualization overhead detected")
        perf_monitor.suggest_optimizations()
```

### Custom Performance Measurement

```python
import time
import jax


def measure_performance(func, *args, warmup_runs=5, measurement_runs=10):
    """Measure function performance with JAX compilation."""

    # Warmup runs for JIT compilation
    for _ in range(warmup_runs):
        _ = func(*args)

    # Ensure all computations are complete
    jax.block_until_ready(func(*args))

    # Measurement runs
    start_time = time.perf_counter()
    for _ in range(measurement_runs):
        result = func(*args)
        jax.block_until_ready(result)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / measurement_runs
    return avg_time, result


# Measure baseline performance
baseline_time, _ = measure_performance(lambda: arc_step(state, action, config))

# Measure with visualization
viz_time, _ = measure_performance(
    lambda: step_with_visualization(state, action, config)
)

overhead_percent = ((viz_time - baseline_time) / baseline_time) * 100
print(f"Baseline: {baseline_time:.3f}s")
print(f"With visualization: {viz_time:.3f}s")
print(f"Overhead: {overhead_percent:.1f}%")
```

## Memory Optimization

### Memory Management Configuration

```python
from jaxarc.utils.visualization import MemoryManager

# Configure memory management
memory_manager = MemoryManager(
    max_memory_mb=1000,  # 1GB limit
    cleanup_threshold=0.8,  # Cleanup at 80% usage
    enable_lazy_loading=True,  # Load data on demand
    compression_level=6,  # Moderate compression
    gc_frequency=100,  # Garbage collect every 100 steps
)


# Monitor memory usage
def check_memory_usage():
    """Check and report memory usage."""

    stats = memory_manager.get_memory_stats()

    print(f"Current usage: {stats['current_mb']:.1f} MB")
    print(f"Peak usage: {stats['peak_mb']:.1f} MB")
    print(f"Cache hits: {stats['cache_hit_rate']:.1f}%")

    # Trigger cleanup if needed
    if stats["usage_ratio"] > 0.9:
        print("🧹 Triggering memory cleanup...")
        memory_manager.cleanup_old_data()


# Use in training loop
if step % 50 == 0:  # Check every 50 steps
    check_memory_usage()
```

### Efficient Image Storage

```python
def optimize_image_storage(visualizer):
    """Configure efficient image storage."""

    # Use compressed formats
    visualizer.set_output_formats(["svg"])  # SVG is more compact than PNG

    # Reduce image quality for storage
    visualizer.set_image_quality("medium")

    # Enable compression
    visualizer.enable_compression(level=6)

    # Limit image dimensions
    visualizer.set_max_image_size((600, 450))

    # Use lazy loading
    visualizer.enable_lazy_loading()


# Apply optimizations
optimize_image_storage(visualizer)
```

### Memory-Efficient Data Structures

```python
import jax.numpy as jnp
from jaxarc.utils.visualization import CompactStateRepresentation


def create_compact_state(state):
    """Create memory-efficient state representation."""

    # Use compact data types
    compact_state = CompactStateRepresentation(
        working_grid=state.working_grid.astype(jnp.uint8),  # Use uint8 instead of int32
        target_grid=state.target_grid.astype(jnp.uint8),
        step_count=jnp.uint16(state.step_count),  # Use uint16 for step count
        # Only store essential information
        essential_info_only=True,
    )

    return compact_state


# Use in visualization
compact_state = create_compact_state(state)
visualizer.visualize_step_compact(compact_state, action, reward, step_num)
```

## Asynchronous Processing Optimization

### Optimal Async Configuration

```python
import multiprocessing

# Determine optimal thread count
cpu_count = multiprocessing.cpu_count()
optimal_threads = max(1, cpu_count // 4)  # Use 1/4 of available CPUs

async_config = AsyncLoggerConfig(
    queue_size=1000,
    worker_threads=optimal_threads,
    batch_size=50,  # Larger batches for efficiency
    flush_interval=5.0,  # Balance between latency and efficiency
    enable_compression=True,
    priority_levels=3,  # Use priority queues
)

async_logger = AsyncLogger(async_config)
```

### Priority-Based Logging

```python
# Define logging priorities
PRIORITY_HIGH = 0  # Important events (episode end, errors)
PRIORITY_MEDIUM = 1  # Regular steps
PRIORITY_LOW = 2  # Debug information

# Use priorities in logging
async_logger.log_step_visualization(step_data, priority=PRIORITY_MEDIUM)

async_logger.log_episode_summary(
    episode_data, priority=PRIORITY_HIGH  # High priority for episode summaries
)

async_logger.log_debug_info(
    debug_data, priority=PRIORITY_LOW  # Low priority for debug info
)
```

### Batch Processing Optimization

```python
class BatchedVisualizer:
    """Batched visualizer for improved performance."""

    def __init__(self, base_visualizer, batch_size=20):
        self.base_visualizer = base_visualizer
        self.batch_size = batch_size
        self.step_batch = []
        self.summary_batch = []

    def add_step(self, step_data):
        """Add step to batch."""
        self.step_batch.append(step_data)

        if len(self.step_batch) >= self.batch_size:
            self.flush_steps()

    def flush_steps(self):
        """Process batched steps."""
        if not self.step_batch:
            return

        # Process all steps in batch
        self.base_visualizer.visualize_steps_batch(self.step_batch)
        self.step_batch.clear()

    def flush_all(self):
        """Flush all batched data."""
        self.flush_steps()
        if self.summary_batch:
            self.base_visualizer.visualize_summaries_batch(self.summary_batch)
            self.summary_batch.clear()


# Usage
batched_viz = BatchedVisualizer(visualizer, batch_size=30)

# In training loop
batched_viz.add_step(
    {"state": state, "action": action, "reward": reward, "step_num": step}
)

# Flush at episode end
batched_viz.flush_all()
```

## JAX-Specific Optimizations

### Efficient JAX Debug Callbacks

```python
import jax


def create_efficient_callback(visualizer, log_frequency=10):
    """Create efficient JAX debug callback."""

    def callback_fn(state, action, reward, step_num):
        """Efficient callback function."""

        # Only process every N steps
        if step_num % log_frequency != 0:
            return

        # Minimal data extraction
        essential_data = {
            "working_grid": state.working_grid,
            "action_op": action["operation"],
            "reward": reward,
            "step_num": step_num,
        }

        # Queue for async processing
        visualizer.queue_visualization(essential_data)

    return callback_fn


# Use in JIT-compiled functions
efficient_callback = create_efficient_callback(visualizer, log_frequency=20)


@jax.jit
def jit_step_with_viz(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Efficient callback
    jax.debug.callback(efficient_callback, new_state, action, reward, info["step_num"])

    return new_state, obs, reward, done, info
```

### Static Argument Optimization

```python
# Mark visualization config as static for JIT
@jax.jit
def step_with_static_config(state, action, env_config, viz_config):
    """Step function with static visualization config."""

    new_state, obs, reward, done, info = arc_step(state, action, env_config)

    # Visualization with static config
    if viz_config.enabled:
        jax.debug.callback(viz_config.callback_fn, state, action, new_state, reward)

    return new_state, obs, reward, done, info


# Use static_argnums for config
jit_step = jax.jit(step_with_static_config, static_argnums=(2, 3))
```

### Memory-Efficient Array Handling

```python
def efficient_array_serialization(array):
    """Efficiently serialize JAX arrays for visualization."""

    # Convert to numpy only when necessary
    if isinstance(array, jnp.ndarray):
        # Use minimal precision for visualization
        if array.dtype == jnp.float32:
            array = array.astype(jnp.float16)  # Reduce precision
        elif array.dtype == jnp.int32:
            array = array.astype(jnp.int8)  # Use smaller int type

        # Convert to numpy for serialization
        array = np.array(array)

    return array


# Use in visualization pipeline
def visualize_with_efficient_arrays(state, action, reward):
    """Visualize with memory-efficient array handling."""

    # Efficiently serialize arrays
    working_grid = efficient_array_serialization(state.working_grid)
    target_grid = efficient_array_serialization(state.target_grid)

    # Create visualization with reduced memory footprint
    visualizer.create_step_visualization(
        working_grid=working_grid,
        target_grid=target_grid,
        action=action,
        reward=float(reward),  # Convert scalar to Python float
    )
```

## Storage and I/O Optimization

### Efficient File I/O

```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor


class AsyncFileWriter:
    """Asynchronous file writer for better I/O performance."""

    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.write_queue = asyncio.Queue()

    async def write_file_async(self, filepath, content):
        """Write file asynchronously."""

        def write_sync():
            with open(filepath, "w") as f:
                f.write(content)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, write_sync)

    async def batch_write_files(self, file_data_pairs):
        """Write multiple files in batch."""

        tasks = [
            self.write_file_async(filepath, content)
            for filepath, content in file_data_pairs
        ]

        await asyncio.gather(*tasks)


# Usage
async_writer = AsyncFileWriter()

# Batch write visualizations
file_pairs = [
    ("step_001.svg", step_1_svg),
    ("step_002.svg", step_2_svg),
    ("step_003.svg", step_3_svg),
]

asyncio.run(async_writer.batch_write_files(file_pairs))
```

### Compression Optimization

```python
import gzip
import pickle
import lz4  # Fast compression library


def compress_visualization_data(data, method="lz4"):
    """Compress visualization data efficiently."""

    # Serialize data
    serialized = pickle.dumps(data)

    if method == "lz4":
        # Fast compression with good ratio
        compressed = lz4.frame.compress(serialized)
    elif method == "gzip":
        # Better compression ratio, slower
        compressed = gzip.compress(serialized)
    else:
        compressed = serialized

    compression_ratio = len(serialized) / len(compressed)
    return compressed, compression_ratio


def decompress_visualization_data(compressed_data, method="lz4"):
    """Decompress visualization data."""

    if method == "lz4":
        serialized = lz4.frame.decompress(compressed_data)
    elif method == "gzip":
        serialized = gzip.decompress(compressed_data)
    else:
        serialized = compressed_data

    return pickle.loads(serialized)


# Use in storage
episode_data = {"steps": steps, "summary": summary}
compressed, ratio = compress_visualization_data(episode_data, method="lz4")
print(f"Compression ratio: {ratio:.1f}x")
```

## Adaptive Performance Tuning

### Auto-Tuning System

```python
class AdaptivePerformanceTuner:
    """Automatically tune visualization performance."""

    def __init__(self, visualizer, target_overhead=0.05):
        self.visualizer = visualizer
        self.target_overhead = target_overhead
        self.performance_history = []
        self.adjustment_history = []

    def measure_and_adjust(self, step_function, *args):
        """Measure performance and adjust settings."""

        # Measure current performance
        start_time = time.perf_counter()
        result = step_function(*args)
        end_time = time.perf_counter()

        step_time = end_time - start_time
        self.performance_history.append(step_time)

        # Calculate recent average
        if len(self.performance_history) > 10:
            recent_avg = np.mean(self.performance_history[-10:])
            baseline_avg = np.mean(self.performance_history[:5])  # First 5 measurements

            current_overhead = (recent_avg - baseline_avg) / baseline_avg

            # Adjust if overhead too high
            if current_overhead > self.target_overhead:
                self.reduce_visualization_load()
            elif current_overhead < self.target_overhead * 0.5:
                self.increase_visualization_detail()

        return result

    def reduce_visualization_load(self):
        """Reduce visualization load to improve performance."""

        current_level = self.visualizer.get_debug_level()

        if current_level == "full":
            self.visualizer.set_debug_level("verbose")
        elif current_level == "verbose":
            self.visualizer.set_debug_level("standard")
        elif current_level == "standard":
            self.visualizer.set_debug_level("minimal")

        # Reduce logging frequency
        current_freq = self.visualizer.get_log_frequency()
        self.visualizer.set_log_frequency(current_freq * 2)

        print(
            f"🔧 Reduced visualization load: {current_level} -> {self.visualizer.get_debug_level()}"
        )

    def increase_visualization_detail(self):
        """Increase visualization detail when performance allows."""

        current_level = self.visualizer.get_debug_level()

        if current_level == "minimal":
            self.visualizer.set_debug_level("standard")
        elif current_level == "standard":
            self.visualizer.set_debug_level("verbose")
        elif current_level == "verbose":
            self.visualizer.set_debug_level("full")

        print(
            f"📈 Increased visualization detail: {current_level} -> {self.visualizer.get_debug_level()}"
        )


# Usage
tuner = AdaptivePerformanceTuner(visualizer, target_overhead=0.03)

# In training loop
result = tuner.measure_and_adjust(step_with_visualization, state, action, config)
```

### Performance Profiles

```python
# Define performance profiles for different scenarios
PERFORMANCE_PROFILES = {
    "development": {
        "debug_level": "verbose",
        "log_frequency": 1,
        "async_workers": 1,
        "memory_limit_mb": 500,
        "image_quality": "high",
    },
    "training": {
        "debug_level": "standard",
        "log_frequency": 10,
        "async_workers": 2,
        "memory_limit_mb": 200,
        "image_quality": "medium",
    },
    "production": {
        "debug_level": "minimal",
        "log_frequency": 100,
        "async_workers": 1,
        "memory_limit_mb": 100,
        "image_quality": "low",
    },
    "benchmark": {
        "debug_level": "off",
        "log_frequency": 0,
        "async_workers": 0,
        "memory_limit_mb": 50,
        "image_quality": "low",
    },
}


def apply_performance_profile(visualizer, profile_name):
    """Apply a performance profile to the visualizer."""

    if profile_name not in PERFORMANCE_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")

    profile = PERFORMANCE_PROFILES[profile_name]

    visualizer.set_debug_level(profile["debug_level"])
    visualizer.set_log_frequency(profile["log_frequency"])
    visualizer.set_async_workers(profile["async_workers"])
    visualizer.set_memory_limit(profile["memory_limit_mb"])
    visualizer.set_image_quality(profile["image_quality"])

    print(f"✅ Applied performance profile: {profile_name}")


# Usage
apply_performance_profile(visualizer, "training")
```

## Benchmarking and Testing

### Performance Benchmarks

```python
def run_performance_benchmark(visualizer, num_steps=1000):
    """Run comprehensive performance benchmark."""

    print("🚀 Running performance benchmark...")

    # Setup
    key = jax.random.PRNGKey(42)
    config = create_standard_config()
    state, obs = arc_reset(key, config)

    # Benchmark without visualization
    print("📊 Baseline performance (no visualization)...")
    baseline_times = []

    for i in range(num_steps):
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(1),
        }

        start = time.perf_counter()
        new_state, obs, reward, done, info = arc_step(state, action, config)
        jax.block_until_ready(new_state)
        end = time.perf_counter()

        baseline_times.append(end - start)
        state = new_state

        if done:
            state, obs = arc_reset(key, config)

    baseline_avg = np.mean(baseline_times)

    # Benchmark with visualization
    print("📊 Performance with visualization...")
    viz_times = []
    state, obs = arc_reset(key, config)

    for i in range(num_steps):
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(1),
        }

        start = time.perf_counter()
        new_state, obs, reward, done, info = arc_step(state, action, config)
        visualizer.visualize_step(state, action, new_state, reward, info, i)
        jax.block_until_ready(new_state)
        end = time.perf_counter()

        viz_times.append(end - start)
        state = new_state

        if done:
            state, obs = arc_reset(key, config)

    viz_avg = np.mean(viz_times)
    overhead = ((viz_avg - baseline_avg) / baseline_avg) * 100

    # Results
    print(f"\n📈 Benchmark Results:")
    print(f"Baseline average: {baseline_avg:.4f}s")
    print(f"With visualization: {viz_avg:.4f}s")
    print(f"Overhead: {overhead:.1f}%")
    print(f"Steps per second (baseline): {1/baseline_avg:.1f}")
    print(f"Steps per second (with viz): {1/viz_avg:.1f}")

    return {
        "baseline_avg": baseline_avg,
        "visualization_avg": viz_avg,
        "overhead_percent": overhead,
        "baseline_sps": 1 / baseline_avg,
        "visualization_sps": 1 / viz_avg,
    }


# Run benchmark
results = run_performance_benchmark(visualizer, num_steps=500)
```

### Memory Profiling

```python
import psutil
import tracemalloc


def profile_memory_usage(visualizer, num_episodes=10):
    """Profile memory usage during visualization."""

    # Start memory tracing
    tracemalloc.start()
    process = psutil.Process()

    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_snapshots = [initial_memory]

    print(f"🧠 Initial memory usage: {initial_memory:.1f} MB")

    # Run episodes with memory tracking
    for episode in range(num_episodes):
        key = jax.random.PRNGKey(episode)
        config = create_standard_config()
        state, obs = arc_reset(key, config)

        visualizer.start_episode(episode)

        for step in range(50):  # 50 steps per episode
            action = {
                "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
                "operation": jnp.array(step % 10),
            }

            new_state, obs, reward, done, info = arc_step(state, action, config)
            visualizer.visualize_step(state, action, new_state, reward, info, step)

            state = new_state
            if done:
                break

        visualizer.visualize_episode_summary(episode)

        # Record memory usage
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_snapshots.append(current_memory)

        if episode % 2 == 0:
            print(f"Episode {episode}: {current_memory:.1f} MB")

    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_growth = final_memory - initial_memory

    print(f"\n🧠 Memory Profile Results:")
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Memory growth: {memory_growth:.1f} MB")
    print(f"Peak traced memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Average per episode: {memory_growth / num_episodes:.1f} MB")

    return {
        "initial_mb": initial_memory,
        "final_mb": final_memory,
        "growth_mb": memory_growth,
        "peak_traced_mb": peak / 1024 / 1024,
        "avg_per_episode_mb": memory_growth / num_episodes,
        "snapshots": memory_snapshots,
    }


# Run memory profiling
memory_results = profile_memory_usage(visualizer, num_episodes=5)
```

## Best Practices Summary

### Performance Best Practices

1. **Use appropriate debug levels** for your use case
2. **Enable asynchronous processing** for I/O operations
3. **Monitor performance regularly** and adjust settings
4. **Use JAX debug callbacks** for JIT compatibility
5. **Batch operations** when possible
6. **Limit memory usage** with cleanup policies
7. **Profile before optimizing** to identify bottlenecks

### Configuration Best Practices

```python
# Production configuration
production_config = VisualizationConfig(
    debug_level="minimal",  # Minimal overhead
    output_formats=["svg"],  # Single efficient format
    image_quality="medium",  # Balance quality/size
    lazy_loading=True,  # Load on demand
    memory_limit_mb=200,  # Reasonable limit
    async_processing=True,  # Background processing
    compression_enabled=True,  # Compress storage
    log_frequency=50,  # Infrequent logging
)

# Development configuration
development_config = VisualizationConfig(
    debug_level="verbose",  # Rich debugging info
    output_formats=["svg", "png"],  # Multiple formats
    image_quality="high",  # High quality for analysis
    lazy_loading=False,  # Immediate loading
    memory_limit_mb=1000,  # Higher limit for dev
    async_processing=True,  # Still use async
    compression_enabled=False,  # Faster access
    log_frequency=1,  # Log every step
)
```

### Monitoring Best Practices

```python
# Regular performance monitoring
def monitor_performance(visualizer, step_num):
    """Monitor performance and adjust if needed."""

    if step_num % 100 == 0:  # Check every 100 steps
        # Get performance metrics
        perf_stats = visualizer.get_performance_stats()

        # Check overhead
        if perf_stats["overhead_percent"] > 10:
            print(f"⚠️  High overhead: {perf_stats['overhead_percent']:.1f}%")
            visualizer.reduce_logging_frequency()

        # Check memory usage
        memory_stats = visualizer.get_memory_stats()
        if memory_stats["usage_ratio"] > 0.8:
            print(f"🧠 High memory usage: {memory_stats['current_mb']:.1f} MB")
            visualizer.cleanup_old_data()

        # Check queue size
        queue_stats = visualizer.get_queue_stats()
        if queue_stats["queue_size"] > queue_stats["max_size"] * 0.8:
            print(f"📦 Queue getting full: {queue_stats['queue_size']}")
            visualizer.increase_worker_threads()


# Use in training loop
monitor_performance(visualizer, step_num)
```

This comprehensive performance optimization guide ensures that JaxARC's enhanced
visualization system provides maximum debugging value with minimal impact on
training performance.
