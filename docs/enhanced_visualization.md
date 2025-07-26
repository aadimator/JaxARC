# Enhanced Visualization and Logging System

JaxARC's enhanced visualization and logging system provides comprehensive
episode management, performance optimization, and integration with experiment
tracking tools like Weights & Biases (wandb). This system maintains JAX
performance while offering rich visualization capabilities for research and
debugging.

## Quick Start

```python
import jax
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.utils.visualization import Visualizer, VisualizationConfig

# Create configuration with enhanced visualization
config = create_standard_config()
vis_config = VisualizationConfig(
    debug_level="standard",
    output_formats=["svg"],
    show_operation_names=True,
    highlight_changes=True,
)

# Initialize enhanced visualizer
visualizer = Visualizer(vis_config)

# Use in training loop
key = jax.random.PRNGKey(42)
state, obs = arc_reset(key, config)

for step in range(100):
    # Your action selection logic here
    action = {"selection": selection_mask, "operation": operation_id}

    # Step environment
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Visualize step (async, minimal performance impact)
    visualizer.visualize_step(
        before_state=state,
        action=action,
        after_state=new_state,
        reward=reward,
        info=info,
        step_num=step,
    )

    state = new_state
    if done:
        # Generate episode summary
        visualizer.visualize_episode_summary(episode_num=0)
        break
```

## Core Components

### 1. Enhanced Visualizer

The `Visualizer` is the main interface that integrates all visualization
components:

```python
from jaxarc.utils.visualization import (
    Visualizer,
    VisualizationConfig,
    EpisodeManager,
    AsyncLogger,
    WandbIntegration,
)

# Configure visualization
vis_config = VisualizationConfig(
    debug_level="verbose",  # "off", "minimal", "standard", "verbose", "full"
    output_formats=["svg", "png"],
    image_quality="high",
    show_coordinates=True,
    show_operation_names=True,
    highlight_changes=True,
    include_metrics=True,
    color_scheme="default",  # "default", "colorblind", "high_contrast"
)

# Initialize components
episode_manager = EpisodeManager(base_output_dir="outputs/episodes")
async_logger = AsyncLogger(queue_size=1000, worker_threads=2)
wandb_integration = WandbIntegration(project_name="my-arc-experiments")

# Create enhanced visualizer
visualizer = Visualizer(
    vis_config=vis_config,
    episode_manager=episode_manager,
    async_logger=async_logger,
    wandb_integration=wandb_integration,
)
```

### 2. Episode Management

The episode management system organizes visualizations by training runs and
episodes:

```python
from jaxarc.utils.visualization import EpisodeManager, EpisodeConfig

# Configure episode management
episode_config = EpisodeConfig(
    base_output_dir="outputs/episodes",
    run_name="experiment_001",  # Auto-generated if None
    episode_dir_format="episode_{episode:04d}",
    step_file_format="step_{step:03d}",
    max_episodes_per_run=1000,
    cleanup_policy="size_based",  # "oldest_first", "size_based", "manual"
    max_storage_gb=10.0,
)

episode_manager = EpisodeManager(episode_config)

# Start new training run
run_dir = episode_manager.start_new_run("my_experiment")
print(f"Training run directory: {run_dir}")

# Start new episode
episode_dir = episode_manager.start_new_episode(episode_num=0)
print(f"Episode directory: {episode_dir}")

# Get file paths for step visualizations
step_path = episode_manager.get_step_path(step_num=5, file_type="svg")
print(f"Step 5 visualization: {step_path}")
```

### 3. Asynchronous Logging

The async logging system minimizes performance impact on JAX computations:

```python
from jaxarc.utils.visualization import AsyncLogger, AsyncLoggerConfig

# Configure async logging
logger_config = AsyncLoggerConfig(
    queue_size=1000,
    worker_threads=2,
    batch_size=10,
    flush_interval=5.0,  # seconds
    enable_compression=True,
)

async_logger = AsyncLogger(logger_config)

# Log step visualization (non-blocking)
step_data = {
    "step_num": 5,
    "before_grid": before_grid,
    "after_grid": after_grid,
    "action": action,
    "reward": reward,
    "info": info,
}
async_logger.log_step_visualization(step_data, priority=0)

# Force flush all pending logs
async_logger.flush()

# Proper cleanup
async_logger.shutdown()
```

### 4. Weights & Biases Integration

Seamless integration with wandb for experiment tracking:

```python
from jaxarc.utils.visualization import WandbIntegration, WandbConfig

# Configure wandb integration
wandb_config = WandbConfig(
    enabled=True,
    project_name="jaxarc-experiments",
    entity="my-team",  # Optional
    tags=["baseline", "experiment-1"],
    log_frequency=10,  # Log every N steps
    image_format="png",  # "png", "svg", "both"
    max_image_size=(800, 600),
    log_gradients=False,
    log_model_topology=False,
)

wandb_integration = WandbIntegration(wandb_config)

# Initialize wandb run
experiment_config = {"learning_rate": 0.001, "batch_size": 32, "max_episodes": 1000}
wandb_integration.initialize_run(experiment_config, run_name="baseline_run_1")

# Log step metrics and visualizations
wandb_integration.log_step(
    step_num=10,
    metrics={"reward": 0.5, "similarity": 0.8},
    images={"step_visualization": step_image},
)

# Log episode summary
wandb_integration.log_episode_summary(
    episode_num=0,
    summary_data={"total_reward": 10.0, "steps": 50, "success": True},
    summary_image=episode_summary_image,
)

# Finish run
wandb_integration.finish_run()
```

## Configuration Options

### Debug Levels

The system supports multiple debug levels with different visualization
granularity:

#### Off (`debug_level: "off"`)

- No visualizations generated
- Minimal performance impact
- Use for production training

#### Minimal (`debug_level: "minimal"`)

- Episode summaries only
- Final state visualizations
- Low storage requirements

#### Standard (`debug_level: "standard"`)

- Key steps and state changes
- Action effect highlighting
- Balanced information/performance

#### Verbose (`debug_level: "verbose"`)

- All steps and actions logged
- Detailed intermediate states
- Rich debugging information

#### Full (`debug_level: "full"`)

- Complete state dumps
- Timing information
- Maximum debugging detail

### Hydra Configuration

The visualization system integrates with Hydra for configuration management:

```yaml
# conf/visualization/debug_standard.yaml
debug_level: "standard"
enabled: true

output_formats: ["svg"]
image_quality: "high"
output_dir: "outputs/debug"

show_coordinates: false
show_operation_names: true
highlight_changes: true
include_metrics: true
color_scheme: "default"

visualize_episodes: true
episode_summaries: true
step_visualizations: true

log_frequency: 10
log_episode_end: true
log_episode_start: true
log_key_moments: true

enable_comparisons: true
save_intermediate_states: false

lazy_loading: true
memory_limit_mb: 500
```

Use with Hydra:

```python
import hydra
from omegaconf import DictConfig
from jaxarc.utils.visualization import create_visualizer_from_config


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Create visualizer from Hydra config
    visualizer = create_visualizer_from_config(cfg.visualization)

    # Your training code here
    # ...


if __name__ == "__main__":
    main()
```

Override settings from command line:

```bash
# Change debug level
python train.py visualization.debug_level=verbose

# Enable wandb
python train.py visualization.wandb.enabled=true visualization.wandb.project_name=my-project

# Change output directory
python train.py visualization.output_dir=custom/output/path
```

## Performance Optimization

### JAX Compatibility

The visualization system is designed to work seamlessly with JAX
transformations:

```python
import jax
from jaxarc.utils.visualization import jax_debug_callback_visualizer


# Create JAX-compatible visualization callback
@jax_debug_callback_visualizer
def visualize_callback(state, action, reward, step_num):
    # This runs outside JAX transformations
    visualizer.visualize_step(state, action, reward, step_num)


# Use in JIT-compiled functions
@jax.jit
def training_step(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)

    # Visualization callback (doesn't break JIT)
    jax.debug.callback(visualize_callback, new_state, action, reward, step_num)

    return new_state, reward, done


# Batch processing with vmap
batch_step = jax.vmap(training_step, in_axes=(0, 0, None))
```

### Memory Management

The system includes comprehensive memory management:

```python
from jaxarc.utils.visualization import MemoryManager

# Configure memory management
memory_manager = MemoryManager(
    max_memory_mb=1000,
    cleanup_threshold=0.8,
    enable_lazy_loading=True,
    compression_level=6,
)

# Monitor memory usage
memory_stats = memory_manager.get_memory_stats()
print(f"Current usage: {memory_stats['current_mb']:.1f} MB")
print(f"Peak usage: {memory_stats['peak_mb']:.1f} MB")

# Automatic cleanup when threshold reached
if memory_manager.should_cleanup():
    memory_manager.cleanup_old_data()
```

### Performance Monitoring

Monitor visualization performance impact:

```python
from jaxarc.utils.visualization import PerformanceMonitor

performance_monitor = PerformanceMonitor()


# Measure step performance with visualization
@performance_monitor.measure_step_impact
def step_with_visualization(state, action, config):
    new_state, obs, reward, done, info = arc_step(state, action, config)
    visualizer.visualize_step(state, action, new_state, reward, info, step_num)
    return new_state, reward, done


# Get performance report
report = performance_monitor.get_performance_report()
print(f"Average step time: {report['avg_step_time']:.3f}s")
print(f"Visualization overhead: {report['visualization_overhead']:.1f}%")

# Adaptive logging based on performance
if performance_monitor.should_reduce_logging():
    visualizer.set_debug_level("minimal")
```

## Structured Logging and Replay

### Episode Logging

The system provides structured logging for episode replay and analysis:

```python
from jaxarc.utils.logging import StructuredLogger

# Initialize structured logger
logger = StructuredLogger(output_dir="outputs/logs")

# Start episode logging
logger.start_episode(episode_num=0, task_id="task_001", config_hash="abc123")

# Log each step
logger.log_step(
    step_num=5,
    before_state=before_state,
    action=action,
    after_state=after_state,
    reward=reward,
    info=info,
    visualization_path="outputs/episodes/episode_0000/step_005.svg",
)

# End episode
logger.end_episode(
    summary_visualization_path="outputs/episodes/episode_0000/summary.svg"
)
```

### Episode Replay

Replay episodes for analysis and debugging:

```python
from jaxarc.utils.visualization import ReplaySystem

replay_system = ReplaySystem(log_dir="outputs/logs")

# Load episode data
episode_data = replay_system.load_episode(episode_num=0)
print(f"Episode {episode_data.episode_num}: {episode_data.total_steps} steps")
print(f"Total reward: {episode_data.total_reward}")
print(f"Success: {episode_data.final_similarity > 0.9}")

# Replay episode with visualization
replay_system.replay_episode(
    episode_num=0, output_dir="outputs/replay", regenerate_visualizations=True
)

# Filter episodes by criteria
successful_episodes = replay_system.filter_episodes(
    min_reward=5.0, min_similarity=0.8, max_steps=100
)

# Analyze failure modes
failure_analysis = replay_system.analyze_failures(
    failed_episodes=failed_episodes, analysis_type="step_by_step"
)
```

## Advanced Features

### Comparison Visualizations

Create comparison visualizations across multiple episodes:

```python
# Compare reward progression across episodes
comparison_viz = visualizer.create_comparison_visualization(
    episodes=[episode_1, episode_2, episode_3], comparison_type="reward_progression"
)

# Compare action patterns
action_comparison = visualizer.create_comparison_visualization(
    episodes=episodes, comparison_type="action_patterns"
)

# Compare final states
state_comparison = visualizer.create_comparison_visualization(
    episodes=episodes, comparison_type="final_states"
)
```

### Custom Visualization Functions

Extend the system with custom visualization functions:

```python
from jaxarc.utils.visualization import register_custom_visualizer


@register_custom_visualizer("heatmap")
def create_action_heatmap(episode_data, **kwargs):
    """Create heatmap of action frequency across grid positions."""
    # Your custom visualization logic
    return heatmap_svg


# Use custom visualizer
visualizer.add_custom_visualization("heatmap", episode_data)
```

### Integration with External Tools

#### TensorBoard Integration

```python
from jaxarc.utils.visualization import TensorBoardIntegration

tb_integration = TensorBoardIntegration(log_dir="outputs/tensorboard")

# Log scalars
tb_integration.log_scalar("reward", reward, step_num)
tb_integration.log_scalar("similarity", similarity, step_num)

# Log images
tb_integration.log_image("step_visualization", step_image, step_num)

# Log histograms
tb_integration.log_histogram("action_distribution", action_probs, step_num)
```

#### MLflow Integration

```python
from jaxarc.utils.visualization import MLflowIntegration

mlflow_integration = MLflowIntegration(experiment_name="arc-experiments")

# Log parameters
mlflow_integration.log_params(config_dict)

# Log metrics
mlflow_integration.log_metric("episode_reward", total_reward, step=episode_num)

# Log artifacts
mlflow_integration.log_artifact("episode_summary.svg")
```

## Error Handling and Troubleshooting

### Common Issues

#### Storage Errors

```python
from jaxarc.utils.visualization import StorageError, handle_storage_error

try:
    episode_manager.start_new_episode(episode_num=0)
except StorageError as e:
    # Handle storage issues gracefully
    fallback_dir = handle_storage_error(e, fallback_dir="tmp/episodes")
    episode_manager = EpisodeManager(base_output_dir=fallback_dir)
```

#### Performance Issues

```python
# Monitor and adjust performance
if performance_monitor.get_avg_overhead() > 0.05:  # 5% overhead
    # Reduce visualization frequency
    visualizer.set_log_frequency(50)  # Log every 50 steps instead of 10

    # Switch to minimal debug level
    visualizer.set_debug_level("minimal")

    # Disable expensive features
    visualizer.disable_comparisons()
```

#### Memory Issues

```python
# Handle memory pressure
if memory_manager.get_memory_usage() > 0.9:  # 90% of limit
    # Force cleanup
    memory_manager.cleanup_old_data()

    # Enable more aggressive compression
    memory_manager.set_compression_level(9)

    # Reduce image quality
    visualizer.set_image_quality("medium")
```

### Debugging Tips

1. **Enable verbose logging** during development:

   ```python
   visualizer.set_debug_level("verbose")
   ```

2. **Check performance impact** regularly:

   ```python
   report = performance_monitor.get_performance_report()
   if report["visualization_overhead"] > 0.1:  # 10%
       print("Warning: High visualization overhead")
   ```

3. **Monitor storage usage**:

   ```python
   storage_stats = episode_manager.get_storage_stats()
   print(f"Storage used: {storage_stats['used_gb']:.1f} GB")
   ```

4. **Validate configurations**:

   ```python
   from jaxarc.utils.visualization import validate_visualization_config

   try:
       validate_visualization_config(vis_config)
   except ConfigValidationError as e:
       print(f"Configuration error: {e}")
   ```

## Best Practices

### Performance

1. **Use appropriate debug levels** for your use case
2. **Enable async logging** for better performance
3. **Monitor memory usage** and enable cleanup policies
4. **Use JAX debug callbacks** for JIT compatibility
5. **Batch visualization operations** when possible

### Storage

1. **Set reasonable storage limits** to prevent disk space issues
2. **Use compression** for long-term storage
3. **Implement cleanup policies** for automatic maintenance
4. **Organize by runs and episodes** for easy navigation

### Experiment Tracking

1. **Use consistent naming conventions** for runs and experiments
2. **Tag experiments** appropriately for easy filtering
3. **Log hyperparameters** along with visualizations
4. **Use offline mode** when network is unreliable

### Development

1. **Start with minimal visualization** and increase as needed
2. **Test performance impact** before production use
3. **Use structured logging** for reproducible analysis
4. **Validate configurations** before long training runs

## Migration from Basic Visualization

### Step-by-Step Migration

1. **Replace basic visualization calls**:

   ```python
   # Old way
   from jaxarc.utils.visualization import log_grid_to_console

   log_grid_to_console(grid, "Step 5")

   # New way
   visualizer.visualize_step(before_state, action, after_state, reward, info, 5)
   ```

2. **Update configuration**:

   ```python
   # Old way
   debug_config = {"enabled": True, "log_steps": True}

   # New way
   vis_config = VisualizationConfig(debug_level="standard", step_visualizations=True)
   ```

3. **Add episode management**:

   ```python
   # Initialize episode management
   episode_manager = EpisodeManager()
   episode_manager.start_new_run("migration_test")
   ```

4. **Enable async logging**:
   ```python
   # Add async logging for better performance
   async_logger = AsyncLogger()
   visualizer = Visualizer(vis_config, async_logger=async_logger)
   ```

### Backward Compatibility

The enhanced system maintains backward compatibility with existing visualization
functions:

```python
# These still work
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg

log_grid_to_console(grid, "Debug output")
svg_content = draw_grid_svg(grid, title="Step visualization")
```

## Examples

Complete examples are available in the `examples/` directory:

- `enhanced_visualization_demo.py` - Comprehensive visualization examples
- `wandb_integration_demo.py` - Weights & Biases integration
- `performance_optimization_demo.py` - Performance monitoring and optimization
- `replay_analysis_demo.py` - Episode replay and analysis

Run examples with:

```bash
pixi run python examples/enhanced_visualization_demo.py
pixi run python examples/wandb_integration_demo.py
```

## API Reference

For detailed API documentation, see the
[API Reference](api_reference.md#visualization-system).
