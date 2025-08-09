# JaxARC Logging System

JaxARC provides a comprehensive logging system designed for both single-environment and batched training scenarios. The system is built around the `ExperimentLogger` class, which coordinates multiple specialized handlers for different logging concerns.

## Overview

The logging system features:

- **Handler-based architecture** with single responsibility principle
- **Batched logging support** for high-performance training
- **Error isolation** between handlers
- **JAX compatibility** through existing callback mechanisms
- **Configuration-driven** handler initialization
- **Multiple output formats** (JSON, console, SVG, Weights & Biases)

## Core Components

### ExperimentLogger

The `ExperimentLogger` serves as the central coordinator for all logging operations:

```python
from jaxarc.envs import JaxArcConfig, LoggingConfig
from jaxarc.utils.logging import ExperimentLogger

# Create configuration
logging_config = LoggingConfig(
    structured_logging=True,
    log_format="json",
    log_frequency=10,
)
config = JaxArcConfig(logging=logging_config)

# Initialize logger
logger = ExperimentLogger(config)
```

### Logging Handlers

The system includes several specialized handlers:

- **FileHandler**: Structured JSON and pickle file output
- **SVGHandler**: Visual grid representations and episode summaries
- **RichHandler**: Rich console output with formatted tables
- **WandbHandler**: Weights & Biases integration (when enabled)

## Single-Environment Logging

### Basic Usage

For traditional single-environment training:

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step, JaxArcConfig
from jaxarc.utils.logging import ExperimentLogger

# Initialize environment and logger
config = JaxArcConfig()
logger = ExperimentLogger(config)

# Training loop
for episode in range(100):
    state, observation = arc_reset(key, config, task_data=task)
    
    # Log task start
    task_data = {
        "task_id": task.task_id,
        "task_object": task,
        "episode_num": episode,
        "num_train_pairs": task.num_train_pairs,
        "num_test_pairs": task.num_test_pairs,
    }
    logger.log_task_start(task_data)
    
    total_reward = 0.0
    for step in range(config.environment.max_episode_steps):
        # Take action
        action = select_action(observation)
        state, observation, reward, done, info = arc_step(state, action, config)
        total_reward += reward
        
        # Log step data
        step_data = {
            "step_num": step,
            "before_state": state,  # Optional: can be expensive
            "after_state": state,   # Optional: can be expensive
            "action": action,
            "reward": reward,
            "info": info,
        }
        logger.log_step(step_data)
        
        if done:
            break
    
    # Log episode summary
    summary_data = {
        "episode_num": episode,
        "total_steps": step + 1,
        "total_reward": total_reward,
        "final_similarity": info.get("similarity", 0.0),
        "success": info.get("success", False),
        "task_id": task.task_id,
    }
    logger.log_episode_summary(summary_data)

# Clean shutdown
logger.close()
```

## Batched Logging

### Overview

Batched logging is designed for high-performance training scenarios where multiple environments run in parallel. It provides:

- **Aggregated metrics** computed efficiently using JAX operations
- **Representative sampling** for detailed episode logging
- **Frequency-based control** to minimize performance impact
- **Extensible metric handling** for downstream algorithms

### Configuration

Enable batched logging in your configuration:

```python
from jaxarc.envs import LoggingConfig, JaxArcConfig

# Basic batched logging configuration
logging_config = LoggingConfig(
    # Enable batched logging
    batched_logging_enabled=True,
    
    # Aggregated metrics frequency
    log_frequency=10,  # Log aggregated metrics every 10 updates
    
    # Sampling configuration
    sampling_enabled=True,
    num_samples=3,  # Sample 3 environments for detailed logging
    sample_frequency=50,  # Sample every 50 updates
    
    # Metric selection
    log_aggregated_rewards=True,
    log_aggregated_similarity=True,
    log_loss_metrics=True,
    log_gradient_norms=True,
    log_episode_lengths=True,
    log_success_rates=True,
)

config = JaxArcConfig(logging=logging_config)
```

### Usage in Training Loops

#### Basic Batched Training

```python
import jax
import jax.numpy as jnp
from jaxarc.utils.logging import ExperimentLogger

# Initialize logger with batched configuration
logger = ExperimentLogger(config)

# Training loop
batch_size = 64
for update_step in range(1000):
    # Simulate batch training step
    # In practice, this would come from your training algorithm
    
    batch_data = {
        "update_step": update_step,
        
        # Episode-level metrics (arrays of length batch_size)
        "episode_returns": episode_returns,  # [batch_size]
        "episode_lengths": episode_lengths,  # [batch_size]
        "similarity_scores": similarity_scores,  # [batch_size]
        "success_mask": success_mask,  # [batch_size] boolean array
        
        # Training metrics (scalars)
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "gradient_norm": gradient_norm,
        
        # Optional: Additional metrics for extensibility
        "entropy": entropy,
        "learning_rate": learning_rate,
        
        # Optional: For detailed sampling (expensive)
        "task_ids": task_ids,  # [batch_size] list of task identifiers
        "initial_states": initial_states,  # [batch_size] initial environment states
        "final_states": final_states,  # [batch_size] final environment states
    }
    
    # Log batch data (handles aggregation and sampling automatically)
    logger.log_batch_step(batch_data)

logger.close()
```

#### PureJaxRL Integration

For integration with PureJaxRL-style training loops:

```python
def train_step(carry, unused):
    """PureJaxRL-style training step with batched logging."""
    rng, update_step = carry
    
    # Your training logic here
    # ...
    
    # Prepare logging data
    log_data = {
        "update_step": update_step,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "similarity_scores": similarity_scores,
        "success_mask": success_mask,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "gradient_norm": gradient_norm,
    }
    
    # Log using JAX debug callback
    jax.debug.callback(logger.log_batch_step, log_data)
    
    return (rng, update_step + 1), metrics

# Run training loop
initial_carry = (jax.random.PRNGKey(0), 0)
final_carry, metrics_history = jax.lax.scan(
    train_step, initial_carry, None, length=num_updates
)
```

### Aggregated Metrics

The system automatically computes statistical aggregations for episode-level metrics:

- **Mean, Standard Deviation, Min, Max** for continuous metrics
- **Success Rate** for boolean success indicators
- **Scalar Training Metrics** passed through directly

Example aggregated output:
```
Batch Metrics - Step 100
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                ┃    Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Rewards               │          │
│   Reward Mean         │    5.174 │
│   Reward Std          │    1.108 │
│   Reward Max          │    7.180 │
│   Reward Min          │    3.045 │
│ Training              │          │
│   Policy Loss         │    1.000 │
│   Value Loss          │   0.5000 │
│   Gradient Norm       │    2.000 │
│ Other                 │          │
│   Success Rate        │   0.6562 │
└───────────────────────┴──────────┘
```

### Representative Sampling

When sampling is enabled, the system selects a subset of environments for detailed episode logging:

- **Deterministic sampling** based on update step for reproducibility
- **Episode summary format** compatible with existing visualization
- **Configurable sample size** and frequency
- **Optional state inclusion** for SVG visualization

## Configuration Options

### Logging Frequency Control

```python
# Performance-optimized configuration
performance_config = LoggingConfig(
    batched_logging_enabled=True,
    log_frequency=100,  # Less frequent for better performance
    sampling_enabled=True,
    num_samples=2,  # Fewer samples
    sample_frequency=500,  # Much less frequent sampling
)

# Debug-optimized configuration
debug_config = LoggingConfig(
    batched_logging_enabled=True,
    log_frequency=1,  # Log every update
    sampling_enabled=True,
    num_samples=8,  # Many samples for debugging
    sample_frequency=5,  # Frequent sampling
)
```

### Metric Selection

Control which metrics are computed and logged:

```python
logging_config = LoggingConfig(
    batched_logging_enabled=True,
    
    # Episode-level aggregations
    log_aggregated_rewards=True,
    log_aggregated_similarity=True,
    log_episode_lengths=True,
    log_success_rates=True,
    
    # Training metrics
    log_loss_metrics=True,
    log_gradient_norms=True,
    
    # Expensive features (disable for performance)
    log_operations=False,
    log_grid_changes=False,
    include_full_states=False,
)
```

### Output Control

```python
logging_config = LoggingConfig(
    # Output format
    structured_logging=True,
    log_format="json",  # "json", "text", "structured"
    compression=True,
    
    # Content control
    include_full_states=False,  # Expensive: full environment states
    log_operations=False,       # Expensive: detailed operation logs
    log_grid_changes=False,     # Expensive: grid change tracking
)
```

## Integration with Weights & Biases

Batched logging integrates seamlessly with Weights & Biases:

```python
from jaxarc.envs import WandbConfig

# Configure wandb with batched logging
wandb_config = WandbConfig(
    enabled=True,
    project_name="jaxarc-batched-training",
    tags=("batched", "research"),
    log_frequency=10,  # Align with batched logging frequency
)

config = JaxArcConfig(
    logging=logging_config,
    wandb=wandb_config
)

logger = ExperimentLogger(config)
# Aggregated metrics and sampled episodes automatically logged to wandb
```

## Performance Considerations

### Frequency Tuning

- **High-frequency logging** (every 1-10 updates): Good for debugging, expensive for training
- **Medium-frequency logging** (every 10-50 updates): Balanced for research
- **Low-frequency logging** (every 100+ updates): Optimized for production training

### Memory Management

- **Disable expensive features** in production:
  - `include_full_states=False`
  - `log_operations=False`
  - `log_grid_changes=False`
- **Reduce sampling** for large batch sizes
- **Use compression** for file outputs

### JAX Compatibility

The logging system is designed to work efficiently with JAX:

- **JAX debug callbacks** for logging within JIT-compiled functions
- **JAX array operations** for efficient metric aggregation
- **Static shapes** maintained throughout logging pipeline

## Troubleshooting

### Common Issues

#### Batched Logging Not Working

```python
# Check configuration
assert config.logging.batched_logging_enabled == True
assert config.logging.log_frequency > 0

# Verify batch data format
batch_data = {
    "update_step": 0,  # Required
    "episode_returns": jnp.array([1.0, 2.0, 3.0]),  # Required
    # ... other required fields
}
```

#### Performance Issues

```python
# Reduce logging frequency
config.logging.log_frequency = 100  # Instead of 10

# Disable expensive features
config.logging.include_full_states = False
config.logging.log_operations = False
config.logging.sampling_enabled = False  # If not needed
```

#### Memory Issues

```python
# Enable compression
config.logging.compression = True

# Reduce sample size
config.logging.num_samples = 1  # Instead of 5

# Disable full state logging
config.logging.include_full_states = False
```

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.getLogger("jaxarc.utils.logging").setLevel(logging.DEBUG)

# Or set debug level in environment config
config.environment.debug_level = "verbose"
```

## Backward Compatibility

The logging system maintains full backward compatibility:

- **Existing single-environment code** works unchanged
- **Old configuration files** load without batched logging settings
- **No performance regression** in single-environment scenarios
- **Graceful degradation** when handlers fail

```python
# Old-style configuration still works
old_config = JaxArcConfig(
    logging=LoggingConfig(
        batched_logging_enabled=False,  # Default
        structured_logging=True,
        log_format="json",
    )
)

logger = ExperimentLogger(old_config)
# All existing logging methods work as before
logger.log_step(step_data)
logger.log_episode_summary(summary_data)
```

## Best Practices

### Configuration

1. **Start with presets** and customize as needed
2. **Align frequencies** between batched logging and external tools (wandb)
3. **Test performance impact** before production deployment
4. **Use appropriate debug levels** for your use case

### Usage

1. **Batch size considerations**: Larger batches benefit more from aggregation
2. **Metric selection**: Enable only necessary metrics for efficiency
3. **Sampling strategy**: Use sampling for detailed analysis, not all episodes
4. **Error handling**: The system handles handler failures gracefully

### Performance

1. **Profile your training loop** to measure logging overhead
2. **Adjust frequencies** based on training speed requirements
3. **Monitor memory usage** especially with large batch sizes
4. **Use compression** for long training runs

## Migration Guide

### From Single to Batched Logging

1. **Update configuration**:
   ```python
   # Add batched logging settings
   config.logging.batched_logging_enabled = True
   config.logging.log_frequency = 50
   ```

2. **Modify training loop**:
   ```python
   # Replace individual episode logging
   # logger.log_episode_summary(episode_data)
   
   # With batch logging
   logger.log_batch_step(batch_data)
   ```

3. **Test and tune** frequencies for your use case

### From Old Visualization System

The old visualization system has been replaced by the `ExperimentLogger`:

```python
# Old (deprecated)
# from jaxarc.utils.visualization import Visualizer
# visualizer = Visualizer(config)

# New
from jaxarc.utils.logging import ExperimentLogger
logger = ExperimentLogger(config)
```

## Examples

See the following example files for complete usage patterns:

- `examples/basic/batched_logging_basic.py` - Simple batched logging setup
- `examples/advanced/batched_logging_demo.py` - Comprehensive usage patterns
- `examples/integration/batched_logging_configs.py` - Configuration examples

## API Reference

For detailed API documentation, see:

- `ExperimentLogger` class documentation
- `LoggingConfig` configuration options
- Individual handler documentation (FileHandler, SVGHandler, etc.)

## Evaluation Summary Logging

### Overview

An additional terminal logging stage captures final evaluation outcomes across test grids/tasks. Call `ExperimentLogger.log_evaluation_summary(eval_data)` once evaluation is complete. Each enabled handler that implements `log_evaluation_summary` will persist or display results in its modality.

### Data Schema

Minimal expected keys inside `eval_data`:

```python
eval_data = {
    "task_id": str,                       # Optional identifier
    "success_rate": float,                # Fraction of successful evaluation episodes
    "average_episode_length": float,      # Mean steps per evaluation episode
    "num_timeouts": int,                  # Count of timed-out episodes
    "test_results": [                     # Optional detailed per-test entries
        {
            "success": bool,
            "episode_length": int,
            "trajectory": [               # Optional: first trajectory used for SVGs
                (before_state, action, after_state, info_dict),
                ...
            ]
        },
        # ... more results
    ],
}
```

All keys are optional; handlers gracefully ignore missing fields.

### Handler Behaviors

- FileHandler: Writes `evaluation_summary.json` (adds timestamp & config hash) to the run directory.
- RichHandler: Renders a concise table (Evaluation Summary) with core metrics plus count of test results.
- WandbHandler: Logs metrics under an `eval/` namespace and mirrors them into the run summary for dashboard visibility.
- SVGHandler: (Gated) Renders at most one representative trajectory (first entry containing a `trajectory` list) as a synthetic episode (episode 0) using existing step SVG generation. Limited to first 100 steps to guard size.

### Gating & Performance

SVG evaluation rendering reuses episode summary gating (`_should_generate_episode_summary`). To disable heavy artifact generation during evaluation, set visualization level to `off` or disable episode summaries in config.

### Usage Example

```python
eval_data = compute_eval_metrics(agent, eval_envs)
logger.log_evaluation_summary(eval_data)
```

### Failure Isolation

Like other logging operations, each handler invocation is wrapped in an error boundary— a failure in one handler (e.g., wandb network issue) will not prevent others (e.g., file or console) from recording the evaluation results.

### Extensibility

You can attach additional scalar fields (e.g., `median_reward`, `std_similarity`) or nested dictionaries; unsupported / non-scalar entries are skipped by strict handlers (Wandb/Rich) but still serialized in full by FileHandler.
