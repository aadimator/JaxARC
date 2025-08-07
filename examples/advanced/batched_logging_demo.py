#!/usr/bin/env python3
"""
Batched Logging Demo

This example demonstrates how to use the batched logging system for high-performance
training scenarios. It shows:
- Setting up batched logging configuration
- Integrating with training loops
- Using aggregated metrics and sampling
- Different configuration patterns for various use cases

Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 7.3, 7.4
"""

import jax
import jax.numpy as jnp
import time
from typing import Dict, Any

from jaxarc.envs import JaxArcConfig, LoggingConfig, EnvironmentConfig
from jaxarc.utils.logging import ExperimentLogger


def create_batched_logging_config():
    """Create configuration optimized for batched training with logging."""
    logging_config = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=10,
        sampling_enabled=True,
        num_samples=3,
        sample_frequency=50,
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    
    environment_config = EnvironmentConfig(
        max_episode_steps=50,
        debug_level="standard"
    )
    
    return JaxArcConfig(
        logging=logging_config,
        environment=environment_config
    )


def create_performance_logging_config():
    """Create configuration optimized for high-performance training."""
    logging_config = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=100,  # Log less frequently for performance
        sampling_enabled=True,
        num_samples=2,  # Fewer samples for performance
        sample_frequency=500,  # Sample less frequently
        # Disable expensive logging features
        log_operations=False,
        log_grid_changes=False,
        include_full_states=False,
        # Essential metrics only
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    
    return JaxArcConfig(logging=logging_config)


def create_debug_logging_config():
    """Create configuration optimized for debugging with detailed logging."""
    logging_config = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=10,  # Log frequently for debugging
        sampling_enabled=True,
        num_samples=5,  # More samples for debugging
        sample_frequency=20,  # Sample frequently
        # Enable detailed logging
        log_operations=True,
        log_grid_changes=True,
        include_full_states=True,
        # All metrics enabled
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    
    return JaxArcConfig(logging=logging_config)


def simulate_batch_training_data(batch_size: int, update_step: int) -> Dict[str, Any]:
    """Simulate realistic batch training data for demonstration.
    
    Args:
        batch_size: Number of environments in the batch
        update_step: Current training update number
        
    Returns:
        Dictionary containing simulated batch training data
    """
    key = jax.random.PRNGKey(update_step)
    keys = jax.random.split(key, 6)
    
    # Simulate episode returns with some variation
    base_reward = 5.0 + 2.0 * jnp.sin(update_step * 0.01)  # Trending reward
    episode_returns = base_reward + jax.random.normal(keys[0], (batch_size,)) * 2.0
    
    # Simulate episode lengths (typically 20-80 steps)
    episode_lengths = 50 + jax.random.randint(keys[1], (batch_size,), -30, 30)
    episode_lengths = jnp.clip(episode_lengths, 1, 100)
    
    # Simulate similarity scores (0.0 to 1.0)
    similarity_scores = jax.random.uniform(keys[2], (batch_size,))
    
    # Simulate success mask (boolean array)
    success_probability = 0.3 + 0.4 * jnp.tanh(update_step * 0.001)  # Improving over time
    success_mask = jax.random.uniform(keys[3], (batch_size,)) < success_probability
    
    # Simulate training metrics (scalars)
    policy_loss = 2.0 * jnp.exp(-update_step * 0.0001) + 0.1  # Decreasing loss
    value_loss = 1.5 * jnp.exp(-update_step * 0.0001) + 0.05
    gradient_norm = 1.0 + 0.5 * jax.random.normal(keys[4])
    
    # Optional: Simulate additional metrics for extensibility demo
    entropy = 2.0 * jnp.exp(-update_step * 0.0002) + 0.5  # Decreasing entropy
    learning_rate = 0.001 * jnp.exp(-update_step * 0.00005)  # Decaying LR
    
    return {
        "update_step": update_step,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "similarity_scores": similarity_scores,
        "success_mask": success_mask,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "gradient_norm": gradient_norm,
        "entropy": entropy,
        "learning_rate": learning_rate,
        # Optional: Task IDs for detailed logging
        "task_ids": [f"task_{update_step}_{i}" for i in range(batch_size)],
    }


def demonstrate_basic_batched_logging():
    """Demonstrate basic batched logging functionality."""
    print("\n=== Basic Batched Logging Demo ===")
    
    # Create configuration with batched logging
    config = create_batched_logging_config()
    
    # Initialize logger
    logger = ExperimentLogger(config)
    print(f"Initialized logger with {len(logger.handlers)} handlers")
    
    # Simulate a short training run
    batch_size = 64
    num_updates = 50
    
    print(f"Simulating {num_updates} training updates with batch size {batch_size}")
    
    for update_step in range(num_updates):
        # Generate batch training data
        batch_data = simulate_batch_training_data(batch_size, update_step)
        
        # Log batch data (respects frequency settings)
        logger.log_batch_step(batch_data)
        
        # Show progress
        if update_step % 10 == 0:
            avg_reward = float(jnp.mean(batch_data["episode_returns"]))
            success_rate = float(jnp.mean(batch_data["success_mask"]))
            print(f"  Update {update_step}: avg_reward={avg_reward:.2f}, "
                  f"success_rate={success_rate:.2f}")
    
    # Clean shutdown
    logger.close()
    print("Basic batched logging demo completed")


def demonstrate_performance_optimized_logging():
    """Demonstrate performance-optimized batched logging."""
    print("\n=== Performance-Optimized Logging Demo ===")
    
    # Create performance-optimized configuration
    config = create_performance_logging_config()
    logger = ExperimentLogger(config)
    
    # Simulate high-performance training
    batch_size = 1000  # Large batch size
    num_updates = 1000
    
    print(f"Simulating high-performance training: {num_updates} updates, "
          f"batch size {batch_size}")
    print(f"Log frequency: {config.logging.log_frequency}, "
          f"Sample frequency: {config.logging.sample_frequency}")
    
    start_time = time.time()
    
    for update_step in range(num_updates):
        # Generate large batch data
        batch_data = simulate_batch_training_data(batch_size, update_step)
        
        # Log batch data (minimal overhead due to frequency settings)
        logger.log_batch_step(batch_data)
        
        # Show progress less frequently
        if update_step % 200 == 0:
            elapsed = time.time() - start_time
            avg_reward = float(jnp.mean(batch_data["episode_returns"]))
            print(f"  Update {update_step}: avg_reward={avg_reward:.2f}, "
                  f"elapsed={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    updates_per_second = num_updates / total_time
    
    print(f"Performance results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Updates per second: {updates_per_second:.1f}")
    print(f"  Environments per second: {updates_per_second * batch_size:.0f}")
    
    logger.close()
    print("Performance-optimized logging demo completed")


def demonstrate_debug_logging():
    """Demonstrate debug-focused batched logging with detailed sampling."""
    print("\n=== Debug-Focused Logging Demo ===")
    
    # Create debug-optimized configuration
    config = create_debug_logging_config()
    logger = ExperimentLogger(config)
    
    # Simulate debugging scenario with smaller batch
    batch_size = 16  # Smaller batch for debugging
    num_updates = 30
    
    print(f"Simulating debug training: {num_updates} updates, batch size {batch_size}")
    print(f"Log frequency: {config.logging.log_frequency}, "
          f"Sample frequency: {config.logging.sample_frequency}")
    
    for update_step in range(num_updates):
        # Generate batch data
        batch_data = simulate_batch_training_data(batch_size, update_step)
        
        # Add some debugging information
        batch_data["debug_info"] = f"Debug update {update_step}"
        
        # Log batch data (frequent logging for debugging)
        logger.log_batch_step(batch_data)
        
        # Show detailed progress
        if update_step % 5 == 0:
            avg_reward = float(jnp.mean(batch_data["episode_returns"]))
            success_rate = float(jnp.mean(batch_data["success_mask"]))
            policy_loss = float(batch_data["policy_loss"])
            print(f"  Update {update_step}: avg_reward={avg_reward:.2f}, "
                  f"success_rate={success_rate:.2f}, policy_loss={policy_loss:.3f}")
    
    logger.close()
    print("Debug-focused logging demo completed")


def demonstrate_integration_with_purejaxrl_pattern():
    """Demonstrate integration with PureJaxRL-style training patterns."""
    print("\n=== PureJaxRL Integration Pattern Demo ===")
    
    # This demonstrates how batched logging would integrate with a typical
    # PureJaxRL training loop structure
    
    config = create_batched_logging_config()
    logger = ExperimentLogger(config)
    
    # Simulate PureJaxRL-style training function
    def train_step(carry, unused):
        """Simulated PureJaxRL training step function."""
        rng, update_step = carry
        rng, *step_rngs = jax.random.split(rng, 5)
        
        # Simulate batch environment step
        batch_size = 128
        batch_data = simulate_batch_training_data(batch_size, update_step)
        
        # In real PureJaxRL, this would be actual training metrics
        # Here we simulate the structure
        train_metrics = {
            "policy_loss": batch_data["policy_loss"],
            "value_loss": batch_data["value_loss"],
            "gradient_norm": batch_data["gradient_norm"],
        }
        
        # Combine environment and training metrics for logging
        log_data = {**batch_data, **train_metrics}
        
        # Log using JAX debug callback (PureJaxRL pattern)
        jax.debug.callback(logger.log_batch_step, log_data)
        
        return (rng, update_step + 1), train_metrics
    
    # Simulate training loop
    print("Running PureJaxRL-style training loop...")
    num_updates = 100
    initial_carry = (jax.random.PRNGKey(0), 0)
    
    # Use jax.lax.scan for efficient training loop (PureJaxRL pattern)
    final_carry, metrics_history = jax.lax.scan(
        train_step, initial_carry, None, length=num_updates
    )
    
    # Analyze results
    final_rng, final_update = final_carry
    avg_policy_loss = float(jnp.mean(metrics_history["policy_loss"]))
    avg_value_loss = float(jnp.mean(metrics_history["value_loss"]))
    
    print(f"Training completed:")
    print(f"  Final update: {final_update}")
    print(f"  Average policy loss: {avg_policy_loss:.3f}")
    print(f"  Average value loss: {avg_value_loss:.3f}")
    
    logger.close()
    print("PureJaxRL integration demo completed")


def demonstrate_configuration_patterns():
    """Demonstrate different configuration patterns for various use cases."""
    print("\n=== Configuration Patterns Demo ===")
    
    # Pattern 1: Research configuration with detailed logging
    print("1. Research Configuration:")
    research_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=10,
        sampling_enabled=True,
        num_samples=5,
        sample_frequency=25,
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=True,
        log_success_rates=True,
    )
    research_config = JaxArcConfig(logging=research_logging)
    print(f"   Batched logging: {research_config.logging.batched_logging_enabled}")
    print(f"   Log frequency: {research_config.logging.log_frequency}")
    print(f"   Sampling: {research_config.logging.sampling_enabled}")
    print(f"   Num samples: {research_config.logging.num_samples}")
    
    # Pattern 2: Training configuration with performance focus
    print("\n2. Training Configuration:")
    training_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=50,
        sampling_enabled=True,
        num_samples=2,
        sample_frequency=200,
        log_aggregated_rewards=True,
        log_aggregated_similarity=False,  # Disable for performance
        log_loss_metrics=True,
        log_gradient_norms=True,
        log_episode_lengths=False,  # Disable for performance
        log_success_rates=True,
    )
    training_config = JaxArcConfig(logging=training_logging)
    print(f"   Batched logging: {training_config.logging.batched_logging_enabled}")
    print(f"   Log frequency: {training_config.logging.log_frequency}")
    print(f"   Num samples: {training_config.logging.num_samples}")
    
    # Pattern 3: Custom configuration for specific needs
    print("\n3. Custom Configuration:")
    custom_logging = LoggingConfig(
        batched_logging_enabled=True,
        log_frequency=25,
        sampling_enabled=True,
        num_samples=4,
        sample_frequency=100,
        log_aggregated_rewards=True,
        log_aggregated_similarity=True,
        log_loss_metrics=True,
        log_gradient_norms=False,  # Disable for this experiment
        log_episode_lengths=True,
        log_success_rates=True,
    )
    custom_config = JaxArcConfig(logging=custom_logging)
    print(f"   Log frequency: {custom_config.logging.log_frequency}")
    print(f"   Sample frequency: {custom_config.logging.sample_frequency}")
    print(f"   Gradient norms: {custom_config.logging.log_gradient_norms}")
    
    print("Configuration patterns demo completed")


def main():
    """Main demonstration of batched logging usage patterns."""
    print("Batched Logging Usage Patterns Demo")
    print("=" * 50)
    
    try:
        # Demonstrate basic functionality
        demonstrate_basic_batched_logging()
        
        # Demonstrate performance optimization
        demonstrate_performance_optimized_logging()
        
        # Demonstrate debug-focused logging
        demonstrate_debug_logging()
        
        # Demonstrate PureJaxRL integration
        demonstrate_integration_with_purejaxrl_pattern()
        
        # Demonstrate configuration patterns
        demonstrate_configuration_patterns()
        
        print("\n" + "=" * 50)
        print("All batched logging demos completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()