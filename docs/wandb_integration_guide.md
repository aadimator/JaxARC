# Weights & Biases Integration Guide

This guide covers setting up and using Weights & Biases (wandb) with JaxARC's enhanced visualization system for experiment tracking and collaboration.

## Installation and Setup

### 1. Install Weights & Biases

```bash
# Install wandb (already included in JaxARC dependencies)
pixi add wandb

# Or if using pip
pip install wandb
```

### 2. Login to Weights & Biases

```bash
# Login with your wandb account
wandb login

# Or set API key as environment variable
export WANDB_API_KEY=your_api_key_here
```

### 3. Verify Installation

```python
import wandb
print(f"wandb version: {wandb.__version__}")

# Test connection
wandb.login()
```

## Basic Configuration

### Configuration File Setup

Create a wandb configuration file:

```yaml
# conf/logging/wandb_basic.yaml
wandb:
  enabled: true
  project_name: "jaxarc-experiments"
  entity: null  # Use default entity
  tags: []
  log_frequency: 10
  image_format: "png"
  max_image_size: [800, 600]
  log_gradients: false
  log_model_topology: false
  offline_mode: false
  sync_tensorboard: false
```

### Programmatic Configuration

```python
from jaxarc.utils.visualization import WandbConfig, WandbIntegration

# Basic configuration
wandb_config = WandbConfig(
    enabled=True,
    project_name="my-arc-project",
    entity="my-team",  # Optional: your team/organization
    tags=["baseline", "experiment-1"],
    log_frequency=10,  # Log every 10 steps
    image_format="png",
    max_image_size=(800, 600)
)

# Initialize integration
wandb_integration = WandbIntegration(wandb_config)
```

## Integration with Enhanced Visualizer

### Complete Setup

```python
import jax
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.utils.visualization import (
    EnhancedVisualizer,
    VisualizationConfig,
    EpisodeManager,
    AsyncLogger,
    WandbIntegration,
    WandbConfig
)

# Configure wandb
wandb_config = WandbConfig(
    enabled=True,
    project_name="jaxarc-arc-solving",
    entity="research-team",
    tags=["dqn", "baseline", "arc-agi-1"],
    log_frequency=5,
    image_format="png",
    max_image_size=(1024, 768),
    log_gradients=True,
    log_model_topology=True
)

# Initialize components
episode_manager = EpisodeManager(base_output_dir="outputs/episodes")
async_logger = AsyncLogger(queue_size=1000)
wandb_integration = WandbIntegration(wandb_config)

# Create enhanced visualizer
vis_config = VisualizationConfig(debug_level="standard")
visualizer = EnhancedVisualizer(
    vis_config=vis_config,
    episode_manager=episode_manager,
    async_logger=async_logger,
    wandb_integration=wandb_integration
)

# Initialize wandb run
experiment_config = {
    "algorithm": "DQN",
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_episodes": 1000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "target_update_freq": 100,
    "replay_buffer_size": 10000
}

wandb_integration.initialize_run(
    experiment_config=experiment_config,
    run_name="dqn_baseline_v1"
)
```

### Training Loop Integration

```python
# Training loop with wandb logging
key = jax.random.PRNGKey(42)
config = create_standard_config()

for episode in range(1000):
    state, obs = arc_reset(key, config)
    episode_reward = 0.0
    episode_steps = 0
    
    # Start episode in visualizer
    visualizer.start_episode(episode)
    
    for step in range(100):
        # Your agent's action selection
        action = agent.select_action(obs)
        
        # Environment step
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Update metrics
        episode_reward += reward
        episode_steps += 1
        
        # Visualize step (includes wandb logging)
        visualizer.visualize_step(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step
        )
        
        # Log additional metrics to wandb
        wandb_integration.log_step(
            step_num=episode * 100 + step,
            metrics={
                "step_reward": reward,
                "cumulative_reward": episode_reward,
                "epsilon": agent.epsilon,
                "loss": agent.last_loss if hasattr(agent, 'last_loss') else 0.0
            }
        )
        
        state = new_state
        if done:
            break
    
    # Log episode summary
    episode_summary = {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "success": info.get("success", False),
        "final_similarity": info.get("similarity", 0.0)
    }
    
    wandb_integration.log_episode_summary(
        episode_num=episode,
        summary_data=episode_summary
    )
    
    # Generate episode summary visualization
    visualizer.visualize_episode_summary(episode_num=episode)
    
    print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}")

# Finish wandb run
wandb_integration.finish_run()
```

## Advanced Configuration

### Full Configuration Options

```python
wandb_config = WandbConfig(
    # Basic settings
    enabled=True,
    project_name="jaxarc-advanced-experiments",
    entity="research-lab",
    
    # Run configuration
    run_name=None,  # Auto-generated if None
    run_id=None,    # Resume existing run if provided
    resume="allow", # "allow", "must", "never"
    
    # Logging settings
    tags=["ppo", "curriculum", "multi-task"],
    notes="Curriculum learning experiment with PPO",
    log_frequency=5,
    save_frequency=100,  # Save model every N episodes
    
    # Image settings
    image_format="both",  # "png", "svg", "both"
    max_image_size=(1200, 900),
    image_quality=0.9,  # JPEG quality for PNG compression
    
    # Advanced logging
    log_gradients=True,
    log_model_topology=True,
    log_system_metrics=True,
    log_code=True,  # Log source code
    
    # Performance settings
    offline_mode=False,
    sync_tensorboard=True,
    upload_frequency=10,  # Upload every N steps
    
    # Storage settings
    save_artifacts=True,
    artifact_types=["model", "dataset", "visualization"],
    
    # Integration settings
    slack_webhook=None,  # Slack notifications
    email_notifications=False
)
```

### Environment-Specific Configuration

```yaml
# conf/logging/wandb_development.yaml
wandb:
  enabled: true
  project_name: "jaxarc-dev"
  tags: ["development", "debug"]
  log_frequency: 1  # Log every step for debugging
  offline_mode: true  # Work offline during development
  save_artifacts: false  # Don't save artifacts in dev

# conf/logging/wandb_production.yaml
wandb:
  enabled: true
  project_name: "jaxarc-production"
  tags: ["production", "final"]
  log_frequency: 50  # Less frequent logging
  offline_mode: false
  save_artifacts: true
  log_system_metrics: true
  
# conf/logging/wandb_research.yaml
wandb:
  enabled: true
  project_name: "jaxarc-research"
  entity: "research-team"
  tags: ["research", "paper"]
  log_frequency: 10
  log_gradients: true
  log_model_topology: true
  save_artifacts: true
  artifact_types: ["model", "visualization", "analysis"]
```

## Logging Best Practices

### Structured Metrics Logging

```python
# Organize metrics into logical groups
def log_training_metrics(wandb_integration, step, metrics):
    """Log training metrics with proper grouping."""
    
    # Agent metrics
    wandb_integration.log_step(step, {
        "agent/epsilon": metrics["epsilon"],
        "agent/learning_rate": metrics["lr"],
        "agent/loss": metrics["loss"],
        "agent/q_value_mean": metrics["q_mean"],
        "agent/q_value_std": metrics["q_std"]
    })
    
    # Environment metrics
    wandb_integration.log_step(step, {
        "env/reward": metrics["reward"],
        "env/similarity": metrics["similarity"],
        "env/steps_taken": metrics["steps"],
        "env/success_rate": metrics["success_rate"]
    })
    
    # Performance metrics
    wandb_integration.log_step(step, {
        "perf/fps": metrics["fps"],
        "perf/memory_usage": metrics["memory_mb"],
        "perf/gpu_utilization": metrics["gpu_util"]
    })

# Usage in training loop
training_metrics = {
    "epsilon": agent.epsilon,
    "lr": optimizer.learning_rate,
    "loss": current_loss,
    "q_mean": q_values.mean(),
    "q_std": q_values.std(),
    "reward": episode_reward,
    "similarity": final_similarity,
    "steps": episode_steps,
    "success_rate": success_rate,
    "fps": steps_per_second,
    "memory_mb": memory_usage_mb,
    "gpu_util": gpu_utilization
}

log_training_metrics(wandb_integration, global_step, training_metrics)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_custom_plots(episode_data, wandb_integration, episode_num):
    """Create custom plots for wandb logging."""
    
    # Reward progression plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_data["reward_progression"])
    plt.title("Reward Progression")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    
    # Log to wandb
    wandb_integration.log_step(episode_num, {
        "custom/reward_progression": wandb.Image(plt)
    })
    plt.close()
    
    # Action distribution heatmap
    action_counts = episode_data["action_counts"]
    plt.figure(figsize=(12, 8))
    sns.heatmap(action_counts, annot=True, fmt='d', cmap='Blues')
    plt.title("Action Distribution Heatmap")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    
    wandb_integration.log_step(episode_num, {
        "custom/action_heatmap": wandb.Image(plt)
    })
    plt.close()
    
    # Similarity over time
    similarities = episode_data["similarity_progression"]
    plt.figure(figsize=(10, 6))
    plt.plot(similarities, label="Similarity")
    plt.axhline(y=0.9, color='r', linestyle='--', label="Success Threshold")
    plt.title("Similarity Progression")
    plt.xlabel("Step")
    plt.ylabel("Similarity Score")
    plt.legend()
    plt.grid(True)
    
    wandb_integration.log_step(episode_num, {
        "custom/similarity_progression": wandb.Image(plt)
    })
    plt.close()
```

### Model and Artifact Logging

```python
def log_model_artifacts(wandb_integration, model, episode_num):
    """Log model checkpoints and artifacts."""
    
    # Save model checkpoint
    model_path = f"checkpoints/model_episode_{episode_num}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Log as wandb artifact
    artifact = wandb.Artifact(
        name=f"model_checkpoint_ep_{episode_num}",
        type="model",
        description=f"Model checkpoint at episode {episode_num}"
    )
    artifact.add_file(model_path)
    wandb_integration.run.log_artifact(artifact)
    
    # Log model summary
    if hasattr(model, 'summary'):
        wandb_integration.log_step(episode_num, {
            "model/parameters": model.count_params(),
            "model/layers": len(model.layers),
            "model/memory_mb": model.get_memory_usage()
        })

def log_dataset_artifacts(wandb_integration, dataset_info):
    """Log dataset information as artifacts."""
    
    # Create dataset artifact
    dataset_artifact = wandb.Artifact(
        name="arc_dataset",
        type="dataset",
        description="ARC dataset used for training"
    )
    
    # Add dataset files
    dataset_artifact.add_dir("data/raw/ARC-AGI-1")
    
    # Add metadata
    dataset_artifact.metadata = {
        "num_training_tasks": dataset_info["num_training"],
        "num_evaluation_tasks": dataset_info["num_evaluation"],
        "dataset_version": dataset_info["version"],
        "preprocessing": dataset_info["preprocessing"]
    }
    
    wandb_integration.run.log_artifact(dataset_artifact)
```

## Offline Mode and Syncing

### Working Offline

```python
# Configure for offline mode
wandb_config = WandbConfig(
    enabled=True,
    project_name="jaxarc-offline",
    offline_mode=True,  # Work offline
    sync_on_finish=True  # Sync when run finishes
)

# Or set environment variable
import os
os.environ["WANDB_MODE"] = "offline"

# Initialize as usual
wandb_integration = WandbIntegration(wandb_config)
```

### Syncing Offline Runs

```bash
# Sync all offline runs
wandb sync outputs/wandb/offline-*

# Sync specific run
wandb sync outputs/wandb/offline-run-20240101_120000-abc123

# Sync with specific project
wandb sync --project jaxarc-experiments outputs/wandb/offline-*
```

### Programmatic Syncing

```python
from jaxarc.utils.visualization import WandbSyncManager

# Initialize sync manager
sync_manager = WandbSyncManager(
    offline_dir="outputs/wandb",
    project_name="jaxarc-experiments",
    max_retries=3,
    retry_delay=60  # seconds
)

# Sync all offline runs
sync_results = sync_manager.sync_all_runs()
for result in sync_results:
    if result["success"]:
        print(f"✅ Synced run: {result['run_id']}")
    else:
        print(f"❌ Failed to sync run: {result['run_id']} - {result['error']}")

# Sync specific run
sync_manager.sync_run("offline-run-20240101_120000-abc123")
```

## Error Handling and Troubleshooting

### Common Issues and Solutions

#### Authentication Issues

```python
import wandb

try:
    wandb.login()
except wandb.errors.AuthenticationError:
    print("❌ Authentication failed. Please check your API key.")
    print("Run 'wandb login' or set WANDB_API_KEY environment variable.")
```

#### Network Issues

```python
from jaxarc.utils.visualization import WandbIntegration, WandbConfig

# Configure with retry and fallback
wandb_config = WandbConfig(
    enabled=True,
    project_name="jaxarc-robust",
    offline_mode=False,
    max_retries=5,
    retry_delay=30,
    fallback_to_offline=True  # Fallback to offline if network fails
)

wandb_integration = WandbIntegration(wandb_config)

# The integration will automatically handle network issues
try:
    wandb_integration.log_step(step_num, metrics)
except Exception as e:
    print(f"Logging failed, but continuing: {e}")
```

#### Storage Issues

```python
# Monitor wandb storage usage
def check_wandb_storage(wandb_integration):
    """Check wandb storage usage and clean up if needed."""
    
    try:
        # Get storage info
        storage_info = wandb_integration.get_storage_info()
        
        if storage_info["usage_gb"] > storage_info["limit_gb"] * 0.9:
            print("⚠️  Approaching wandb storage limit")
            
            # Clean up old runs
            wandb_integration.cleanup_old_runs(keep_recent=10)
            
            # Reduce image quality
            wandb_integration.config.image_quality = 0.7
            wandb_integration.config.max_image_size = (600, 450)
            
    except Exception as e:
        print(f"Storage check failed: {e}")

# Use in training loop
if episode % 100 == 0:  # Check every 100 episodes
    check_wandb_storage(wandb_integration)
```

### Debugging Wandb Integration

```python
import logging

# Enable wandb debug logging
logging.getLogger("wandb").setLevel(logging.DEBUG)

# Enable JaxARC visualization debug logging
logging.getLogger("jaxarc.utils.visualization").setLevel(logging.DEBUG)

# Test wandb connection
def test_wandb_connection():
    """Test wandb connection and configuration."""
    
    try:
        # Test login
        wandb.login()
        print("✅ wandb login successful")
        
        # Test project access
        api = wandb.Api()
        projects = api.projects(entity="your-entity")
        print(f"✅ Found {len(projects)} projects")
        
        # Test run creation
        test_run = wandb.init(
            project="test-project",
            name="connection-test",
            mode="disabled"  # Don't actually log
        )
        test_run.finish()
        print("✅ Run creation successful")
        
    except Exception as e:
        print(f"❌ wandb connection test failed: {e}")

# Run connection test
test_wandb_connection()
```

## Performance Optimization

### Efficient Logging

```python
# Batch logging for better performance
class BatchedWandbLogger:
    def __init__(self, wandb_integration, batch_size=10):
        self.wandb_integration = wandb_integration
        self.batch_size = batch_size
        self.batch_metrics = []
        self.batch_images = []
    
    def add_metrics(self, step_num, metrics):
        """Add metrics to batch."""
        self.batch_metrics.append((step_num, metrics))
        
        if len(self.batch_metrics) >= self.batch_size:
            self.flush_metrics()
    
    def add_image(self, step_num, name, image):
        """Add image to batch."""
        self.batch_images.append((step_num, name, image))
        
        if len(self.batch_images) >= self.batch_size:
            self.flush_images()
    
    def flush_metrics(self):
        """Flush batched metrics."""
        for step_num, metrics in self.batch_metrics:
            self.wandb_integration.log_step(step_num, metrics)
        self.batch_metrics.clear()
    
    def flush_images(self):
        """Flush batched images."""
        for step_num, name, image in self.batch_images:
            self.wandb_integration.log_step(step_num, {name: image})
        self.batch_images.clear()
    
    def flush_all(self):
        """Flush all batched data."""
        self.flush_metrics()
        self.flush_images()

# Usage
batched_logger = BatchedWandbLogger(wandb_integration, batch_size=20)

# In training loop
batched_logger.add_metrics(step_num, {"reward": reward, "loss": loss})
batched_logger.add_image(step_num, "visualization", step_image)

# Flush at episode end
batched_logger.flush_all()
```

### Memory-Efficient Image Logging

```python
def optimize_image_for_wandb(image_array, max_size=(800, 600), quality=0.8):
    """Optimize image for wandb upload."""
    
    from PIL import Image
    import io
    
    # Convert to PIL Image
    if isinstance(image_array, jnp.ndarray):
        image_array = np.array(image_array)
    
    img = Image.fromarray(image_array)
    
    # Resize if too large
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Compress
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True, quality=int(quality * 100))
    buffer.seek(0)
    
    return wandb.Image(buffer)

# Use in visualization
optimized_image = optimize_image_for_wandb(step_visualization)
wandb_integration.log_step(step_num, {"step_viz": optimized_image})
```

## Integration Examples

### Complete Training Script

```python
#!/usr/bin/env python3
"""Complete training script with wandb integration."""

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig

from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.utils.visualization import (
    EnhancedVisualizer,
    create_visualizer_from_config
)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function with wandb integration."""
    
    # Create visualizer from config (includes wandb)
    visualizer = create_visualizer_from_config(cfg.visualization)
    
    # Initialize wandb run
    wandb_config = {
        "algorithm": cfg.algorithm.name,
        "learning_rate": cfg.algorithm.lr,
        "batch_size": cfg.algorithm.batch_size,
        "max_episodes": cfg.training.max_episodes,
        "dataset": cfg.dataset.name,
        "environment": cfg.environment.name
    }
    
    visualizer.wandb_integration.initialize_run(
        experiment_config=wandb_config,
        run_name=f"{cfg.algorithm.name}_{cfg.dataset.name}_run"
    )
    
    # Training setup
    key = jax.random.PRNGKey(cfg.seed)
    env_config = create_standard_config()
    
    # Training loop
    for episode in range(cfg.training.max_episodes):
        state, obs = arc_reset(key, env_config)
        episode_reward = 0.0
        
        visualizer.start_episode(episode)
        
        for step in range(cfg.training.max_steps_per_episode):
            # Agent action (placeholder)
            action = select_random_action(state, key)
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, env_config)
            episode_reward += reward
            
            # Visualize and log
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            state = new_state
            if done:
                break
        
        # Episode summary
        visualizer.visualize_episode_summary(episode_num=episode)
        
        # Log episode metrics
        visualizer.wandb_integration.log_episode_summary(
            episode_num=episode,
            summary_data={
                "episode_reward": episode_reward,
                "episode_steps": step + 1,
                "success": info.get("success", False)
            }
        )
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}")
    
    # Finish wandb run
    visualizer.wandb_integration.finish_run()
    print("Training completed!")

def select_random_action(state, key):
    """Placeholder random action selection."""
    # This would be replaced with your actual agent
    selection = jnp.ones_like(state.working_grid, dtype=jnp.bool_)
    operation = jax.random.randint(key, (), 0, 10)
    return {"selection": selection, "operation": operation}

if __name__ == "__main__":
    main()
```

### Configuration File

```yaml
# conf/config.yaml
defaults:
  - algorithm: dqn
  - dataset: arc_agi_1
  - environment: standard
  - visualization: debug_standard
  - logging: wandb_research

seed: 42

training:
  max_episodes: 1000
  max_steps_per_episode: 100

visualization:
  wandb:
    enabled: true
    project_name: "jaxarc-research-v2"
    entity: "research-team"
    tags: ["dqn", "arc-agi-1", "baseline"]
    log_frequency: 5
    image_format: "png"
    max_image_size: [1024, 768]
```

Run the training script:

```bash
# Basic run
pixi run python train.py

# Override wandb settings
pixi run python train.py visualization.wandb.project_name=my-custom-project

# Disable wandb
pixi run python train.py visualization.wandb.enabled=false

# Change debug level
pixi run python train.py visualization.debug_level=verbose
```

This comprehensive guide covers all aspects of integrating Weights & Biases with JaxARC's enhanced visualization system. The integration provides powerful experiment tracking capabilities while maintaining the performance benefits of JAX.