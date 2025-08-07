#!/usr/bin/env python3
"""
Basic Batched Logging Example

This example shows the simplest way to use batched logging in JaxARC.
Perfect for getting started with batched training scenarios.

Requirements: 1.1, 1.2, 7.3, 7.4
"""

import jax
import jax.numpy as jnp

from jaxarc.envs import JaxArcConfig, LoggingConfig
from jaxarc.utils.logging import ExperimentLogger


def create_simple_batch_data(batch_size: int, update_step: int):
    """Create simple batch training data for demonstration."""
    key = jax.random.PRNGKey(update_step)
    
    # Simple batch data with required fields
    return {
        "update_step": update_step,
        "episode_returns": jax.random.normal(key, (batch_size,)) + 5.0,
        "episode_lengths": jnp.full(batch_size, 50),  # Fixed length for simplicity
        "similarity_scores": jax.random.uniform(key, (batch_size,)),
        "success_mask": jax.random.uniform(key, (batch_size,)) > 0.5,
        "policy_loss": 1.0,  # Simple scalar loss
        "value_loss": 0.5,
        "gradient_norm": 2.0,
    }


def main():
    """Basic batched logging example."""
    print("Basic Batched Logging Example")
    print("=" * 40)
    
    # 1. Create configuration with batched logging enabled
    logging_config = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=5,  # Log every 5 updates
        sampling_enabled=True,
        num_samples=2,  # Sample 2 environments for detailed logging
        sample_frequency=10,  # Sample every 10 updates
    )
    config = JaxArcConfig(logging=logging_config)
    
    print("Created configuration with batched logging enabled")
    
    # 2. Initialize logger
    logger = ExperimentLogger(config)
    print(f"Initialized logger with {len(logger.handlers)} handlers")
    
    # 3. Simulate training loop
    batch_size = 32
    num_updates = 25
    
    print(f"\nRunning {num_updates} training updates with batch size {batch_size}")
    print("Aggregated metrics will be logged every 5 updates")
    print("Detailed samples will be logged every 10 updates")
    print()
    
    for update_step in range(num_updates):
        # Create batch training data
        batch_data = create_simple_batch_data(batch_size, update_step)
        
        # Log batch data - this handles both aggregation and sampling automatically
        logger.log_batch_step(batch_data)
        
        # Show what's happening
        avg_reward = float(jnp.mean(batch_data["episode_returns"]))
        success_rate = float(jnp.mean(batch_data["success_mask"]))
        
        # Indicate when logging occurs
        will_log_metrics = (update_step % config.logging.log_frequency == 0)
        will_log_samples = (update_step % config.logging.sample_frequency == 0)
        
        status = []
        if will_log_metrics:
            status.append("METRICS")
        if will_log_samples:
            status.append("SAMPLES")
        
        status_str = f" [{', '.join(status)}]" if status else ""
        
        print(f"Update {update_step:2d}: reward={avg_reward:.2f}, "
              f"success={success_rate:.2f}{status_str}")
    
    # 4. Clean shutdown
    logger.close()
    
    print(f"\nBasic example completed!")
    print("Check the output files in your configured logging directory.")


if __name__ == "__main__":
    main()