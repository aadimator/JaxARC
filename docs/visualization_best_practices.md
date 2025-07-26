# Visualization Best Practices Guide

This guide provides best practices, recommendations, and common patterns for
using JaxARC's enhanced visualization and logging system effectively.

## Configuration Best Practices

### 1. Choose the Right Debug Level

Select debug levels based on your use case:

```python
# Development and debugging
vis_config = VisualizationConfig(debug_level="verbose")

# Training and experimentation
vis_config = VisualizationConfig(debug_level="standard")

# Production deployment
vis_config = VisualizationConfig(debug_level="minimal")

# Performance benchmarking
vis_config = VisualizationConfig(debug_level="off")
```

### 2. Optimize Logging Frequency

Balance information needs with performance:

```python
# Development: Frequent logging for detailed analysis
vis_config = VisualizationConfig(log_frequency=1)  # Every step

# Training: Moderate logging for progress tracking
vis_config = VisualizationConfig(log_frequency=10)  # Every 10 steps

# Production: Infrequent logging for monitoring
vis_config = VisualizationConfig(log_frequency=100)  # Every 100 steps

# Adaptive logging based on performance
if performance_monitor.get_overhead() > 0.1:
    visualizer.set_log_frequency(visualizer.get_log_frequency() * 2)
```

### 3. Memory Management

Implement proper memory management strategies:

```python
# Set appropriate memory limits
vis_config = VisualizationConfig(
    memory_limit_mb=500,  # Adjust based on available RAM
    lazy_loading=True,  # Load data on demand
    enable_compression=True,  # Compress stored data
)

# Enable automatic cleanup
memory_manager = MemoryManager(
    max_memory_mb=1000,
    cleanup_threshold=0.8,  # Cleanup at 80% usage
    gc_frequency=100,  # Garbage collect every 100 steps
)

# Monitor memory usage
if step % 50 == 0:
    memory_stats = memory_manager.get_memory_stats()
    if memory_stats["usage_ratio"] > 0.9:
        memory_manager.cleanup_old_data()
```

### 4. Storage Optimization

Manage storage efficiently:

```python
# Configure storage limits
episode_manager = EpisodeManager(
    max_storage_gb=5.0,
    cleanup_policy="size_based",  # or "oldest_first"
    max_episodes_per_run=100,
)

# Use compression for long-term storage
vis_config = VisualizationConfig(
    output_formats=["svg"],  # SVG is more compact than PNG
    image_quality="medium",  # Balance quality and size
    enable_compression=True,
)

# Regular cleanup
if episode % 100 == 0:
    episode_manager.cleanup_old_episodes(keep_recent=50)
```

## Performance Best Practices

### 1. JAX Integration

Ensure proper JAX compatibility:

```python
# Use JAX debug callbacks for JIT compatibility
@jax.jit
def training_step(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Proper debug callback usage
    jax.debug.callback(
        visualizer.jax_callback_function, state, action, new_state, reward, info
    )

    return new_state, obs, reward, done, info


# Mark visualization config as static
@jax.jit
def step_with_static_viz_config(state, action, env_config, viz_config):
    # Implementation here
    pass


jit_step = jax.jit(step_with_static_viz_config, static_argnums=(3,))
```

### 2. Asynchronous Processing

Leverage async processing for better performance:

```python
# Configure async processing
async_logger = AsyncLogger(
    queue_size=1000,
    worker_threads=2,  # Adjust based on CPU cores
    batch_size=20,  # Larger batches for efficiency
    flush_interval=5.0,  # Balance latency and efficiency
)

# Use priority queues for important events
async_logger.log_step_visualization(step_data, priority=1)  # Normal priority
async_logger.log_episode_summary(episode_data, priority=0)  # High priority
async_logger.log_debug_info(debug_data, priority=2)  # Low priority
```

### 3. Performance Monitoring

Continuously monitor performance impact:

```python
# Set up performance monitoring
perf_monitor = PerformanceMonitor(
    target_overhead=0.05,  # 5% maximum overhead
    measurement_window=100,
    auto_adjust=True,
)


# Regular performance checks
@perf_monitor.measure_step_impact
def step_with_monitoring(state, action, config):
    # Your step implementation
    pass


# Adaptive performance tuning
if perf_monitor.get_avg_overhead() > 0.1:
    visualizer.reduce_logging_frequency()
    visualizer.set_debug_level("minimal")
```

## Development Workflow Best Practices

### 1. Development Phase

```python
# Development configuration
dev_config = VisualizationConfig(
    debug_level="verbose",
    output_formats=["svg", "png"],
    show_operation_names=True,
    highlight_changes=True,
    log_frequency=5,
    save_intermediate_states=True,
)

# Enable all debugging features
visualizer = Visualizer(
    vis_config=dev_config,
    enable_performance_monitoring=True,
    enable_memory_monitoring=True,
)

# Test specific scenarios
test_scenarios = [
    {"name": "Basic Operations", "operations": [1, 2, 3, 4, 5]},
    {"name": "Complex Operations", "operations": [10, 15, 20, 25, 30]},
]

for scenario in test_scenarios:
    # Test each scenario with detailed logging
    test_scenario(scenario, visualizer)
```

### 2. Training Phase

```python
# Training configuration
training_config = VisualizationConfig(
    debug_level="standard",
    output_formats=["svg"],
    log_frequency=10,
    memory_limit_mb=500,
    enable_comparisons=True,
)

# Set up experiment tracking
wandb_integration = WandbIntegration(
    {
        "project_name": "my-arc-experiments",
        "tags": ["training", "baseline"],
        "log_frequency": 20,
    }
)

# Training loop with visualization
for episode in range(num_episodes):
    # Your training logic
    if episode % 10 == 0:  # Periodic detailed visualization
        visualizer.set_debug_level("verbose")
    else:
        visualizer.set_debug_level("standard")
```

### 3. Production Phase

```python
# Production configuration
prod_config = VisualizationConfig(
    debug_level="minimal",
    output_formats=["svg"],
    log_frequency=100,
    memory_limit_mb=200,
    enable_comparisons=False,
)

# Minimal monitoring
visualizer = Visualizer(
    vis_config=prod_config, enable_auto_cleanup=True, enable_performance_monitoring=True
)

# Production monitoring
if step % 1000 == 0:  # Periodic health checks
    health_check(visualizer)
```

## Experiment Tracking Best Practices

### 1. Structured Experiment Organization

```python
# Organize experiments hierarchically
experiment_config = {
    "experiment_group": "baseline_studies",
    "experiment_name": "dqn_baseline_v1",
    "algorithm": "DQN",
    "dataset": "arc-agi-1",
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epsilon_decay": 0.995,
    },
}

# Use consistent naming conventions
run_name = f"{experiment_config['algorithm']}_{experiment_config['dataset']}_v1"
```

### 2. Comprehensive Metric Logging

```python
# Organize metrics into logical groups
def log_training_metrics(step, metrics):
    # Agent metrics
    wandb_integration.log_step(
        step,
        {
            "agent/epsilon": metrics["epsilon"],
            "agent/learning_rate": metrics["lr"],
            "agent/loss": metrics["loss"],
        },
    )

    # Environment metrics
    wandb_integration.log_step(
        step,
        {
            "env/reward": metrics["reward"],
            "env/similarity": metrics["similarity"],
            "env/success_rate": metrics["success_rate"],
        },
    )

    # Performance metrics
    wandb_integration.log_step(
        step,
        {
            "perf/steps_per_second": metrics["sps"],
            "perf/memory_usage": metrics["memory_mb"],
        },
    )
```

### 3. Artifact Management

```python
# Save important artifacts
def save_experiment_artifacts(episode_num):
    # Model checkpoints
    if episode_num % 100 == 0:
        model_path = f"checkpoints/model_ep_{episode_num}.pkl"
        save_model(model, model_path)
        wandb_integration.log_artifact(model_path, "model")

    # Visualization summaries
    if episode_num % 50 == 0:
        summary_path = visualizer.create_episode_summary(episode_num)
        wandb_integration.log_artifact(summary_path, "visualization")

    # Configuration snapshots
    config_path = f"configs/config_ep_{episode_num}.yaml"
    save_config(current_config, config_path)
    wandb_integration.log_artifact(config_path, "config")
```

## Error Handling Best Practices

### 1. Graceful Degradation

```python
def robust_visualization_step(state, action, new_state, reward, info, step_num):
    """Robust visualization with graceful error handling."""

    try:
        # Attempt full visualization
        visualizer.visualize_step(state, action, new_state, reward, info, step_num)

    except MemoryError:
        # Handle memory issues
        print("⚠️  Memory issue, reducing visualization quality")
        visualizer.set_image_quality("low")
        visualizer.cleanup_old_data()

        # Retry with reduced quality
        try:
            visualizer.visualize_step(state, action, new_state, reward, info, step_num)
        except:
            print("❌ Visualization failed, continuing without visualization")

    except Exception as e:
        # Log error but continue training
        print(f"⚠️  Visualization error: {e}")

        # Optionally disable visualization temporarily
        if isinstance(e, (IOError, OSError)):
            visualizer.disable_temporarily(duration=100)  # Disable for 100 steps
```

### 2. System Recovery

```python
def setup_automatic_recovery():
    """Set up automatic recovery mechanisms."""

    # Auto-restart failed workers
    visualizer.enable_worker_auto_restart(failure_threshold=3, restart_delay=5.0)

    # Auto-cleanup on memory pressure
    visualizer.enable_auto_cleanup(memory_threshold=0.8, cleanup_frequency=100)

    # Fallback configurations
    fallback_configs = [
        {"debug_level": "standard"},
        {"debug_level": "minimal"},
        {"debug_level": "off"},
    ]

    visualizer.set_fallback_configs(fallback_configs)
```

### 3. Monitoring and Alerting

```python
def setup_monitoring_alerts():
    """Set up monitoring and alerting."""

    # Performance alerts
    def performance_alert(overhead_percent):
        if overhead_percent > 15:
            print(f"🚨 High visualization overhead: {overhead_percent:.1f}%")
            visualizer.reduce_logging_frequency()

    # Memory alerts
    def memory_alert(usage_mb, limit_mb):
        if usage_mb > limit_mb * 0.9:
            print(f"🚨 High memory usage: {usage_mb:.1f}/{limit_mb} MB")
            visualizer.cleanup_old_data()

    # Storage alerts
    def storage_alert(used_gb, limit_gb):
        if used_gb > limit_gb * 0.9:
            print(f"🚨 High storage usage: {used_gb:.1f}/{limit_gb} GB")
            visualizer.cleanup_old_episodes()

    # Register alerts
    visualizer.register_alert("performance", performance_alert)
    visualizer.register_alert("memory", memory_alert)
    visualizer.register_alert("storage", storage_alert)
```

## Testing and Validation Best Practices

### 1. Unit Testing Visualization Components

```python
import pytest


def test_visualizer_basic_functionality():
    """Test basic visualizer functionality."""

    vis_config = VisualizationConfig(debug_level="minimal")
    visualizer = Visualizer(vis_config)

    # Test initialization
    assert visualizer.is_initialized()

    # Test configuration
    assert visualizer.get_debug_level() == "minimal"

    # Test basic operations
    visualizer.start_episode(0)
    assert visualizer.get_current_episode() == 0


def test_performance_impact():
    """Test that visualization doesn't significantly impact performance."""

    # Measure baseline performance
    baseline_time = measure_baseline_performance()

    # Measure with visualization
    vis_time = measure_with_visualization()

    # Assert overhead is acceptable
    overhead = (vis_time - baseline_time) / baseline_time
    assert overhead < 0.1, f"Visualization overhead too high: {overhead:.1f}"


def test_memory_management():
    """Test memory management functionality."""

    memory_manager = MemoryManager(max_memory_mb=100)

    # Simulate memory usage
    for i in range(50):
        memory_manager.allocate_visualization_data(size_mb=5)

    # Check that cleanup occurs
    assert memory_manager.get_memory_usage() < 100
```

### 2. Integration Testing

```python
def test_full_workflow_integration():
    """Test complete workflow integration."""

    # Setup complete system
    visualizer = create_complete_visualizer()

    # Run mini training loop
    for episode in range(5):
        visualizer.start_episode(episode)

        for step in range(10):
            # Simulate step
            step_data = create_test_step_data(step)
            visualizer.visualize_step(**step_data)

        visualizer.visualize_episode_summary(episode)

    # Verify outputs
    assert visualizer.get_episodes_count() == 5
    assert all(visualizer.episode_exists(i) for i in range(5))


def test_error_recovery():
    """Test error recovery mechanisms."""

    visualizer = Visualizer()

    # Simulate various error conditions
    with pytest.raises(MemoryError):
        # Force memory error
        visualizer.force_memory_error()

    # Verify recovery
    assert visualizer.is_operational()

    # Test graceful degradation
    visualizer.simulate_disk_full()
    assert visualizer.get_debug_level() == "minimal"  # Should auto-reduce
```

### 3. Performance Benchmarking

```python
def benchmark_visualization_performance():
    """Comprehensive performance benchmark."""

    configs = [
        {"name": "Minimal", "debug_level": "minimal"},
        {"name": "Standard", "debug_level": "standard"},
        {"name": "Verbose", "debug_level": "verbose"},
        {"name": "Full", "debug_level": "full"},
    ]

    results = {}

    for config in configs:
        visualizer = Visualizer(VisualizationConfig(**config))

        # Benchmark
        start_time = time.perf_counter()
        run_benchmark_scenario(visualizer)
        end_time = time.perf_counter()

        results[config["name"]] = {
            "time": end_time - start_time,
            "memory": visualizer.get_peak_memory_usage(),
        }

    # Generate performance report
    generate_performance_report(results)
```

## Migration Best Practices

### 1. Gradual Migration

```python
# Phase 1: Add basic visualization alongside existing code
def phase_1_migration():
    # Keep existing visualization
    old_visualizer.log_step(step_data)

    # Add new visualization
    try:
        new_visualizer.visualize_step(**step_data)
    except Exception as e:
        print(f"New visualization failed: {e}")


# Phase 2: Switch to new system with fallback
def phase_2_migration():
    try:
        new_visualizer.visualize_step(**step_data)
    except Exception:
        # Fallback to old system
        old_visualizer.log_step(step_data)


# Phase 3: Full migration
def phase_3_migration():
    new_visualizer.visualize_step(**step_data)
```

### 2. Configuration Migration

```python
def migrate_old_config_to_new(old_config):
    """Migrate old configuration format to new format."""

    # Map old settings to new settings
    debug_level_mapping = {
        "none": "off",
        "basic": "minimal",
        "detailed": "standard",
        "full": "verbose",
    }

    new_config = VisualizationConfig(
        debug_level=debug_level_mapping.get(old_config.get("level"), "standard"),
        output_formats=old_config.get("formats", ["svg"]),
        log_frequency=old_config.get("frequency", 10),
        memory_limit_mb=old_config.get("memory_limit", 500),
    )

    return new_config
```

### 3. Data Migration

```python
def migrate_visualization_data():
    """Migrate existing visualization data to new format."""

    old_data_dir = "outputs/old_format"
    new_data_dir = "outputs/new_format"

    for episode_dir in os.listdir(old_data_dir):
        old_episode_path = os.path.join(old_data_dir, episode_dir)

        # Convert old format to new format
        episode_data = load_old_format(old_episode_path)
        converted_data = convert_to_new_format(episode_data)

        # Save in new format
        new_episode_path = os.path.join(new_data_dir, episode_dir)
        save_new_format(converted_data, new_episode_path)
```

## Summary

### Key Takeaways

1. **Choose appropriate configurations** for your use case (development,
   training, production)
2. **Monitor performance impact** and adjust settings accordingly
3. **Implement proper error handling** and recovery mechanisms
4. **Use structured experiment tracking** for reproducible research
5. **Test thoroughly** before deploying to production
6. **Migrate gradually** when upgrading from older systems

### Configuration Quick Reference

```python
# Development
VisualizationConfig(debug_level="verbose", log_frequency=5, memory_limit_mb=1000)

# Training
VisualizationConfig(debug_level="standard", log_frequency=10, memory_limit_mb=500)

# Production
VisualizationConfig(debug_level="minimal", log_frequency=100, memory_limit_mb=200)

# Debugging
VisualizationConfig(debug_level="full", log_frequency=1, memory_limit_mb=2000)

# Benchmarking
VisualizationConfig(debug_level="off", log_frequency=0, memory_limit_mb=50)
```

### Performance Targets

- **Development**: < 20% overhead acceptable
- **Training**: < 10% overhead target
- **Production**: < 5% overhead required
- **Benchmarking**: < 1% overhead essential

Following these best practices will help you get the most value from JaxARC's
enhanced visualization system while maintaining optimal performance for your
specific use case.
