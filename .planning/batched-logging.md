## Batched Logging Implementation Plan

Here is a detailed plan to implement a comprehensive, configurable logging solution for batched training within the `JaxARC` repository.

### 1. Philosophy and Goals

- **Location:** The core logic will be added to the `JaxARC` repository, specifically within `jaxarc/utils/logging`, to make it a foundational feature. The PPO training loop in `jaxarc-benchmarks` will then utilize this core functionality.
- **Simplicity:** We will extend the existing `ExperimentLogger` rather than creating a new system. This maintains a single, consistent logging interface.
- **Configurability:** All new features will be controllable via a new Hydra configuration file, `conf/logging/batched.yaml`, allowing users to enable, disable, and tune batched logging features easily.
- **Performance:** The design will prioritize minimizing performance impact on training. This will be achieved through two main strategies:
    - **Aggregation:** Logging mean, standard deviation, min, and max for metrics across the entire batch.
    - **Sampling:** Logging detailed, per-environment information (like SVG visualizations) for only a small, random subset of environments in the batch.

### 2. Implementation Steps

#### Step 2.1: Create Batched Logging Configuration (`JaxARC/conf/logging/batched.yaml`)

First, we'll create a new configuration file to control batched logging. This provides a centralized place for all related settings.

```yaml
# @package logging
# Batched logging configuration for high-performance training

# Core logging settings
structured_logging: true
log_format: "json"
log_level: "INFO"
compression: true
include_full_states: false # Keep this false for performance

# Batched Training Specifics
batched_logging_enabled: true
log_frequency: 10 # Log aggregated metrics every 10 updates

# Representative Sampling (for detailed logs like SVGs)
sampling_enabled: true
num_samples: 3 # Number of environments to sample from the batch for detailed logs
sample_frequency: 50 # Log samples every 50 updates

# What to log (for aggregated batch metrics)
log_aggregated_rewards: true
log_aggregated_similarity: true
log_loss_metrics: true
log_gradient_norms: true

# Async logging settings (tuned for batched workloads)
queue_size: 5000
worker_threads: 4
batch_size: 100
flush_interval: 10.0
enable_compression: true
```

We also need to update `JaxARC/envs/config.py` to include these new fields in the `LoggingConfig` class.

#### Step 2.2: Enhance `ExperimentLogger` for Batched Data (`JaxARC/utils/logging/experiment_logger.py`)

Next, we'll modify the `ExperimentLogger` to understand and process batched data. This involves adding a new method specifically for batched steps.

```python
class ExperimentLogger:
    # … (existing __init__ and other methods) …

    def log_batch_step(self, batch_data: Dict[str, Any]) -> None:
        """
        Log data from a batched training step.

        This method orchestrates logging for a full batch of environments by
        calculating aggregate metrics and logging representative samples.

        Args:
            batch_data: Dictionary containing batched training metrics,
                        including episode_returns, episode_lengths, policy_loss, etc.
                        It should also contain the current training step/update number.
        """
        # Extract the current update number
        update_step = batch_data.get("update_step", 0)

        # Log aggregated metrics at the specified frequency
        if self.config.logging.batched_logging_enabled and update_step % self.config.logging.log_frequency == 0:
            aggregated_metrics = self._aggregate_batch_metrics(batch_data)
            for handler in self.handlers.values():
                if hasattr(handler, 'log_aggregated_metrics'):
                    handler.log_aggregated_metrics(aggregated_metrics, update_step)

        # Log sampled, detailed steps at its specified frequency
        if self.config.logging.sampling_enabled and update_step % self.config.logging.sample_frequency == 0:
            sampled_steps = self._sample_steps_from_batch(batch_data)
            for step_data in sampled_steps:
                self.log_step(step_data) # Reuse the existing log_step for individual samples

    def _aggregate_batch_metrics(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate metrics from a batch of training data."""
        metrics = {}
        # Example for rewards
        if 'episode_returns' in batch_data:
            returns = batch_data['episode_returns']
            metrics['reward_mean'] = jnp.mean(returns)
            metrics['reward_std'] = jnp.std(returns)
            metrics['reward_max'] = jnp.max(returns)
            metrics['reward_min'] = jnp.min(returns)
        # … similar aggregation for other metrics like similarity, loss, etc.
        return metrics

    def _sample_steps_from_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract a sample of individual step data from the batch for detailed logging."""
        num_samples = self.config.logging.num_samples
        num_envs = batch_data['episode_returns'].shape[0]
        
        if num_samples >= num_envs:
            sample_indices = jnp.arange(num_envs)
        else:
            # Use a deterministic key for sampling to ensure consistency
            key = jax.random.PRNGKey(batch_data.get("update_step", 0))
            sample_indices = jax.random.choice(key, num_envs, shape=(num_samples,), replace=False)

        sampled_data = []
        for i in sample_indices:
            # Reconstruct a single `step_data` dictionary for one environment
            step_info = {
                "step_num": batch_data.get("update_step", 0),
                "reward": batch_data['episode_returns'][i],
                # … reconstruct other necessary fields for log_step
            }
            sampled_data.append(step_info)
            
        return sampled_data

```

#### Step 2.3: Update Handlers to Process Batched Metrics

Each handler needs a new method, `log_aggregated_metrics`, to process the summarized batch data. The existing `log_step` will automatically handle the sampled data.

- **`FileHandler` & `RichHandler`:** These can simply format and print/save the aggregated dictionary.
    
- **`WandbHandler`:** This handler will log the aggregated metrics to Weights & Biases, which is ideal for creating smooth training curves.
    
- **`SVGHandler`:** This handler will _not_ implement `log_aggregated_metrics`. Its strength is in detailed, step-by-step visualization, which is perfectly served by the sampling mechanism feeding into the existing `log_step` method.
    

Example for `WandbHandler` (`JaxARC/utils/logging/wandb_handler.py`):

```python
class WandbHandler:
    # … (existing methods) …

    def log_aggregated_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log aggregated batch metrics to wandb."""
        if self.run is None:
            return
        
        try:
            # Wandb can log a dictionary of metrics directly
            self.run.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Wandb aggregated logging failed: {e}")
```

### 3. Conclusion

This plan provides a clear path to implementing a robust, configurable, and performant logging system for batched training. By extending the existing `ExperimentLogger` and adding a dedicated configuration, we maintain a clean and consistent API while introducing powerful new capabilities. The use of aggregation and sampling ensures that we can gather rich insights from large-scale experiments without incurring significant performance overhead.