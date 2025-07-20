#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # JaxARC Comprehensive Demo: MiniArc + Enhanced Visualization + Wandb
# 
# This notebook demonstrates the complete JaxARC ecosystem with:
# - **MiniArc Dataset**: Compact 5x5 grid tasks for rapid experimentation
# - **Enhanced Visualization**: Rich SVG/PNG rendering with debug capabilities
# - **Wandb Integration**: Experiment tracking and logging
# - **Bbox Actions**: Bounding box-based action format
# - **Raw Environment**: Minimal operations for focused learning
# - **JAX-Compliant Random Agent**: Fully JIT-compiled agent implementation
# 
# ## Key Features Demonstrated
# 1. Dataset loading and task sampling
# 2. Environment configuration and setup
# 3. Enhanced visualization system
# 4. Wandb experiment tracking
# 5. JAX-optimized agent implementation
# 6. Episode management and storage
# 7. Performance monitoring and optimization

# %% [markdown]
# ## Setup and Imports

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Tuple, Optional
import time
from pathlib import Path
import numpy as np

# JaxARC core imports
from jaxarc.envs import (
    arc_reset, arc_step, create_bbox_config, 
    create_raw_config, ArcEnvConfig
)
from jaxarc.parsers import MiniArcParser
from jaxarc.types import JaxArcTask, ARCLEAction
from jaxarc.utils.config import get_config

# Enhanced visualization and logging
from jaxarc.utils.visualization import (
    EnhancedVisualizer, VisualizationConfig,
    EpisodeManager, WandbIntegration, WandbConfig,
    AsyncLogger, MemoryManager, PerformanceMonitor,
    create_research_wandb_config
)

# Set up JAX for optimal performance
jax.config.update("jax_enable_x64", False)  # Use float32 for speed
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration Setup
# 
# We'll create a comprehensive configuration that combines:
# - MiniArc dataset for fast iteration
# - Bbox action format for intuitive control
# - Raw environment with minimal operations
# - Enhanced visualization for rich debugging
# - Wandb integration for experiment tracking

# %%
def create_comprehensive_config() -> Dict[str, Any]:
    """Create a comprehensive configuration for our demo."""
    
    # Load base MiniArc configuration
    base_config = get_config(overrides=[
        "dataset=mini_arc",
        "action=bbox", 
        "environment=raw",
        "debug=on"
    ])
    
    # Enhanced visualization configuration
    vis_config = VisualizationConfig(
        debug_level="standard",
        output_formats=["svg", "png"],
        image_quality="high",
        show_coordinates=True,
        show_operation_names=True,
        highlight_changes=True,
        include_metrics=True,
        color_scheme="default",
        log_frequency=1,  # Log every step
        enable_comparisons=True,
        save_intermediate_states=True,
        lazy_loading=False,
        memory_limit_mb=500
    )
    
    # Wandb configuration for experiment tracking
    wandb_config = create_research_wandb_config(
        project_name="jaxarc-comprehensive-demo",
        entity=None  # Use default
    )
    wandb_config.enabled = True
    wandb_config.offline_mode = True  # Use offline mode for demo
    wandb_config.tags = ["demo", "miniarc", "bbox", "raw", "comprehensive"]
    
    # Episode management
    episode_config = {
        "base_output_dir": "outputs/comprehensive_demo",
        "run_name": f"demo_run_{int(time.time())}",
        "max_episodes_per_run": 10,
        "cleanup_policy": "size_based",
        "max_storage_gb": 2.0
    }
    
    # Performance monitoring
    perf_config = {
        "target_overhead": 0.1,  # 10% max overhead
        "measurement_window": 10,
        "auto_adjust": True
    }
    
    return {
        "base_config": base_config,
        "visualization": vis_config,
        "wandb": wandb_config,
        "episode_management": episode_config,
        "performance": perf_config
    }

# Create configuration
config_dict = create_comprehensive_config()
base_config = config_dict["base_config"]

print("✅ Configuration created successfully!")
print(f"Dataset: {base_config.dataset.dataset_name}")
print(f"Action format: {base_config.action.selection_format}")
print(f"Environment: Raw (minimal operations)")
print(f"Max grid size: {base_config.grid.max_grid_height}x{base_config.grid.max_grid_width}")

# %% [markdown]
# ## Dataset Loading and Task Sampling
# 
# Let's load the MiniArc dataset and examine some sample tasks.

# %%
def setup_dataset_and_parser():
    """Set up the MiniArc dataset and parser."""
    
    # Initialize MiniArc parser
    data_path = Path(base_config.dataset.data_root)
    
    if not data_path.exists():
        print(f"⚠️  Dataset not found at {data_path}")
        print("Please download MiniArc dataset or use demo tasks")
        return None, []
    
    try:
        parser = MiniArcParser(data_dir=str(data_path))
        
        # Load a sample of training tasks
        print("Loading MiniArc training tasks...")
        tasks = parser.load_tasks(split="training", max_tasks=5)
        
        print(f"✅ Loaded {len(tasks)} training tasks")
        
        # Display task information
        for i, task in enumerate(tasks[:3]):
            print(f"\nTask {i}:")
            print(f"  Training pairs: {task.num_train_pairs}")
            print(f"  Test pairs: {task.num_test_pairs}")
            print(f"  Grid shape: {task.input_grids_examples.shape}")
        
        return parser, tasks
    
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Using demo task instead...")
        return None, []

# Load dataset
parser, sample_tasks = setup_dataset_and_parser()

# %% [markdown]
# ## Enhanced Visualization System Setup
# 
# Set up the comprehensive visualization system with all components.

# %%
def setup_visualization_system(config_dict: Dict[str, Any]) -> EnhancedVisualizer:
    """Set up the complete visualization system."""
    
    # Create episode manager
    episode_manager = EpisodeManager(
        base_output_dir=config_dict["episode_management"]["base_output_dir"],
        run_name=config_dict["episode_management"]["run_name"],
        max_episodes_per_run=config_dict["episode_management"]["max_episodes_per_run"],
        cleanup_policy=config_dict["episode_management"]["cleanup_policy"],
        max_storage_gb=config_dict["episode_management"]["max_storage_gb"]
    )
    
    # Create async logger for performance
    async_logger = AsyncLogger(
        queue_size=500,
        worker_threads=1,
        batch_size=10,
        flush_interval=5.0,
        enable_compression=True
    )
    
    # Create memory manager
    memory_manager = MemoryManager(
        max_memory_mb=config_dict["visualization"].memory_limit_mb,
        cleanup_threshold=0.8,
        enable_lazy_loading=config_dict["visualization"].lazy_loading
    )
    
    # Create performance monitor
    perf_monitor = PerformanceMonitor(
        target_overhead=config_dict["performance"]["target_overhead"],
        measurement_window=config_dict["performance"]["measurement_window"],
        auto_adjust=config_dict["performance"]["auto_adjust"]
    )
    
    # Create Wandb integration
    wandb_integration = WandbIntegration(config_dict["wandb"])
    
    # Create enhanced visualizer
    visualizer = EnhancedVisualizer(
        vis_config=config_dict["visualization"],
        episode_manager=episode_manager,
        async_logger=async_logger,
        memory_manager=memory_manager,
        performance_monitor=perf_monitor,
        wandb_integration=wandb_integration
    )
    
    print("✅ Enhanced visualization system initialized")
    print(f"Output directory: {episode_manager.get_current_run_dir()}")
    
    return visualizer

# Setup visualization system
visualizer = setup_visualization_system(config_dict)

# %% [markdown]
# ## JAX-Compliant Random Agent Implementation
# 
# Create a fully JAX-compatible random agent that can be JIT-compiled for optimal performance.

# %%
@jax.jit
def create_random_bbox_action(key: jax.Array, grid_shape: Tuple[int, int]) -> Dict[str, jax.Array]:
    """Create a random bounding box action (JAX-compiled)."""
    
    height, width = grid_shape
    
    # Generate random bounding box coordinates
    key1, key2, key3, key4, key5 = jr.split(key, 5)
    
    # Random top-left corner
    y1 = jr.randint(key1, (), 0, height)
    x1 = jr.randint(key2, (), 0, width)
    
    # Random bottom-right corner (ensuring it's larger than top-left)
    y2 = jr.randint(key3, (), y1, height)
    x2 = jr.randint(key4, (), x1, width)
    
    # Random operation (for raw environment, use limited operations)
    operation = jr.randint(key5, (), 0, 5)  # Operations 0-4 for raw env
    
    return {
        "bbox": jnp.array([y1, x1, y2, x2], dtype=jnp.int32),
        "operation": operation
    }

@jax.jit 
def random_agent_step(key: jax.Array, state, config) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single step of the random agent (JAX-compiled)."""
    
    # Create random action
    action = create_random_bbox_action(key, state.working_grid.shape)
    
    # Take environment step
    new_state, obs, reward, done, info = arc_step(state, action, config)
    
    return new_state, obs, reward, done, info, action

@jax.jit
def random_agent_episode(key: jax.Array, config, max_steps: int = 20):
    """Run a complete episode with the random agent (JAX-compiled)."""
    
    # Reset environment
    reset_key, episode_key = jr.split(key)
    state, obs = arc_reset(reset_key, config)
    
    # Episode loop
    total_reward = 0.0
    step_count = 0
    
    def episode_step(carry, step_key):
        state, total_reward, step_count, done = carry
        
        # Only take action if not done
        new_state, obs, reward, new_done, info, action = jax.lax.cond(
            done,
            lambda: (state, obs, 0.0, done, {}, {"bbox": jnp.zeros(4, dtype=jnp.int32), "operation": jnp.array(0)}),
            lambda: random_agent_step(step_key, state, config)
        )
        
        new_total_reward = total_reward + reward
        new_step_count = step_count + 1
        
        return (new_state, new_total_reward, new_step_count, new_done), {
            "state": new_state,
            "reward": reward,
            "done": new_done,
            "action": action,
            "step": step_count
        }
    
    # Run episode steps
    step_keys = jr.split(episode_key, max_steps)
    final_carry, episode_data = jax.lax.scan(
        episode_step,
        (state, total_reward, step_count, False),
        step_keys
    )
    
    final_state, final_reward, final_steps, final_done = final_carry
    
    return {
        "final_state": final_state,
        "total_reward": final_reward,
        "episode_length": final_steps,
        "episode_data": episode_data,
        "success": final_done
    }

print("✅ JAX-compliant random agent implemented")
print("All functions are JIT-compiled for optimal performance")

# %% [markdown]
# ## Environment Configuration and Testing
# 
# Set up the environment with our specific configuration and test basic functionality.

# %%
def setup_environment():
    """Set up and test the environment configuration."""
    
    # Create environment configuration
    env_config = create_bbox_config(
        max_episode_steps=25,
        reward_on_submit_only=False,
        success_bonus=10.0,
        step_penalty=-0.01,
        log_operations=True
    )
    
    # Override with raw environment settings
    env_config = env_config.replace(
        action=env_config.action.replace(num_operations=5),  # Limit to 5 operations for raw
        strict_validation=False,  # More lenient for experimentation
        allow_invalid_actions=True
    )
    
    print("Environment Configuration:")
    print(f"  Max episode steps: {env_config.max_episode_steps}")
    print(f"  Action format: {env_config.action.selection_format}")
    print(f"  Number of operations: {env_config.action.num_operations}")
    print(f"  Grid size: {env_config.grid.max_grid_height}x{env_config.grid.max_grid_width}")
    
    # Test basic environment functionality
    print("\nTesting environment...")
    key = jr.PRNGKey(42)
    
    try:
        state, obs = arc_reset(key, env_config)
        print(f"✅ Environment reset successful")
        print(f"  Initial grid shape: {obs.shape}")
        print(f"  Initial similarity: {state.similarity_score:.3f}")
        
        # Test a simple action
        test_action = {
            "bbox": jnp.array([0, 0, 2, 2], dtype=jnp.int32),
            "operation": jnp.array(1, dtype=jnp.int32)
        }
        
        new_state, new_obs, reward, done, info = arc_step(state, test_action, env_config)
        print(f"✅ Environment step successful")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        
        return env_config
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        raise

# Setup environment
env_config = setup_environment()

# %% [markdown]
# ## Wandb Experiment Initialization
# 
# Initialize the Wandb experiment for tracking our runs.

# %%
def initialize_wandb_experiment():
    """Initialize Wandb experiment tracking."""
    
    experiment_config = {
        "agent_type": "RandomAgent",
        "environment": "JaxARC",
        "dataset": "MiniARC",
        "action_format": "bbox",
        "environment_type": "raw",
        "max_episodes": 5,
        "max_steps_per_episode": 25,
        "grid_size": "5x5",
        "num_operations": 5,
        "jax_backend": jax.default_backend(),
        "jax_devices": len(jax.devices())
    }
    
    # Initialize Wandb run
    success = visualizer.wandb_integration.initialize_run(
        experiment_config=experiment_config,
        run_name="comprehensive_demo_run"
    )
    
    if success:
        print("✅ Wandb experiment initialized")
        print(f"Run ID: {visualizer.wandb_integration.run_id}")
        print(f"Project: {visualizer.wandb_integration.config.project_name}")
    else:
        print("⚠️  Wandb initialization failed (continuing without Wandb)")
    
    return success

# Initialize Wandb
wandb_success = initialize_wandb_experiment()

# %% [markdown]
# ## Main Training Loop with Comprehensive Logging
# 
# Run multiple episodes with the random agent while demonstrating all features.

# %%
def run_comprehensive_demo(num_episodes: int = 5):
    """Run the comprehensive demo with all features."""
    
    print(f"🚀 Starting comprehensive demo with {num_episodes} episodes")
    print("=" * 60)
    
    # Initialize metrics tracking
    episode_metrics = []
    total_start_time = time.perf_counter()
    
    # Main episode loop
    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Start episode tracking
        visualizer.start_episode(episode)
        episode_start_time = time.perf_counter()
        
        # Generate episode key
        episode_key = jr.PRNGKey(42 + episode)
        
        # Run JAX-compiled episode
        print("Running JAX-compiled episode...")
        episode_result = random_agent_episode(episode_key, env_config, max_steps=25)
        
        episode_end_time = time.perf_counter()
        episode_duration = episode_end_time - episode_start_time
        
        # Extract episode data
        final_state = episode_result["final_state"]
        total_reward = float(episode_result["total_reward"])
        episode_length = int(episode_result["episode_length"])
        episode_data = episode_result["episode_data"]
        success = bool(episode_result["success"])
        
        print(f"Episode completed in {episode_duration:.3f}s")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Episode length: {episode_length}")
        print(f"Success: {success}")
        
        # Detailed step-by-step visualization (for first few steps)
        print("Generating step visualizations...")
        
        # Reset environment to replay episode with visualization
        viz_key = jr.PRNGKey(42 + episode)
        state, obs = arc_reset(viz_key, env_config)
        
        # Visualize first 10 steps in detail
        max_viz_steps = min(10, episode_length)
        step_keys = jr.split(viz_key, max_viz_steps)
        
        for step in range(max_viz_steps):
            # Get action from episode data
            action = episode_data["action"][step]
            
            # Take step
            new_state, obs, reward, done, info = arc_step(state, action, env_config)
            
            # Visualize step
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=float(reward),
                info={k: float(v) if isinstance(v, jnp.ndarray) else v for k, v in info.items()},
                step_num=step
            )
            
            # Log to Wandb
            if wandb_success:
                step_metrics = {
                    "episode": episode,
                    "step": step,
                    "step_reward": float(reward),
                    "cumulative_reward": float(jnp.sum(episode_data["reward"][:step+1])),
                    "similarity": float(info.get("similarity", 0.0)),
                    "grid_changes": int(jnp.sum(jnp.abs(new_state.working_grid - state.working_grid)))
                }
                
                visualizer.wandb_integration.log_step(
                    step_num=episode * 25 + step,
                    metrics=step_metrics
                )
            
            state = new_state
            
            if done:
                break
        
        # Generate episode summary
        episode_summary = {
            "episode_reward": total_reward,
            "episode_steps": episode_length,
            "episode_duration": episode_duration,
            "final_similarity": float(info.get("similarity", 0.0)),
            "success": success,
            "steps_per_second": episode_length / episode_duration if episode_duration > 0 else 0
        }
        
        # Visualize episode summary
        visualizer.visualize_episode_summary(episode)
        
        # Log episode summary to Wandb
        if wandb_success:
            visualizer.wandb_integration.log_episode_summary(episode, episode_summary)
        
        # Store metrics
        episode_metrics.append(episode_summary)
        
        print(f"✅ Episode {episode + 1} completed and logged")
    
    # Final summary
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive Demo Completed!")
    print("=" * 60)
    
    # Calculate summary statistics
    total_rewards = [ep["episode_reward"] for ep in episode_metrics]
    episode_lengths = [ep["episode_steps"] for ep in episode_metrics]
    success_rate = sum(ep["success"] for ep in episode_metrics) / len(episode_metrics)
    
    print(f"\n📈 Summary Statistics:")
    print(f"Total episodes: {num_episodes}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Episodes per second: {num_episodes / total_duration:.2f}")
    
    # Performance monitoring report
    if hasattr(visualizer, 'performance_monitor'):
        perf_report = visualizer.performance_monitor.get_performance_report()
        print(f"\n⚡ Performance Report:")
        print(f"Visualization overhead: {perf_report.get('visualization_overhead', 0):.1f}%")
        print(f"Average step time: {perf_report.get('avg_step_time', 0):.4f}s")
    
    # Memory usage report
    if hasattr(visualizer, 'memory_manager'):
        memory_stats = visualizer.memory_manager.get_memory_stats()
        print(f"Memory usage: {memory_stats.get('current_mb', 0):.1f} MB")
    
    # Output directory information
    output_dir = visualizer.episode_manager.get_current_run_dir()
    print(f"\n📁 Output Directory: {output_dir}")
    print("Generated files:")
    if output_dir.exists():
        svg_files = list(output_dir.glob("**/*.svg"))
        png_files = list(output_dir.glob("**/*.png"))
        print(f"  SVG files: {len(svg_files)}")
        print(f"  PNG files: {len(png_files)}")
    
    return episode_metrics

# Run the comprehensive demo
episode_metrics = run_comprehensive_demo(num_episodes=5)

# %% [markdown]
# ## Performance Analysis and Optimization
# 
# Analyze the performance characteristics of our JAX-compiled agent.

# %%
def analyze_performance():
    """Analyze and demonstrate performance characteristics."""
    
    print("🔬 Performance Analysis")
    print("=" * 40)
    
    # Test JIT compilation performance
    print("\n1. JIT Compilation Performance:")
    
    key = jr.PRNGKey(123)
    
    # Time first call (includes compilation)
    start_time = time.perf_counter()
    result1 = random_agent_episode(key, env_config, max_steps=10)
    first_call_time = time.perf_counter() - start_time
    
    # Time subsequent calls (compiled)
    times = []
    for i in range(10):
        key = jr.PRNGKey(123 + i)
        start_time = time.perf_counter()
        result = random_agent_episode(key, env_config, max_steps=10)
        times.append(time.perf_counter() - start_time)
    
    avg_compiled_time = np.mean(times)
    speedup = first_call_time / avg_compiled_time
    
    print(f"  First call (with compilation): {first_call_time:.4f}s")
    print(f"  Average compiled call: {avg_compiled_time:.4f}s")
    print(f"  Speedup after compilation: {speedup:.1f}x")
    
    # Test batch processing
    print("\n2. Batch Processing Performance:")
    
    batch_sizes = [1, 5, 10, 20]
    batch_times = []
    
    for batch_size in batch_sizes:
        keys = jr.split(jr.PRNGKey(456), batch_size)
        
        # Vectorized batch processing
        batch_fn = jax.vmap(lambda k: random_agent_episode(k, env_config, max_steps=10))
        
        start_time = time.perf_counter()
        batch_results = batch_fn(keys)
        batch_time = time.perf_counter() - start_time
        
        batch_times.append(batch_time)
        episodes_per_second = batch_size / batch_time
        
        print(f"  Batch size {batch_size:2d}: {batch_time:.4f}s ({episodes_per_second:.1f} eps/s)")
    
    # Memory efficiency analysis
    print("\n3. Memory Efficiency:")
    
    # Check memory usage of different components
    if hasattr(visualizer, 'memory_manager'):
        memory_stats = visualizer.memory_manager.get_memory_stats()
        print(f"  Current memory usage: {memory_stats.get('current_mb', 0):.1f} MB")
        print(f"  Peak memory usage: {memory_stats.get('peak_mb', 0):.1f} MB")
        print(f"  Memory efficiency: {memory_stats.get('efficiency', 0):.1%}")
    
    # JAX device memory
    try:
        for device in jax.devices():
            memory_info = device.memory_stats()
            print(f"  {device}: {memory_info.get('bytes_in_use', 0) / 1e6:.1f} MB used")
    except:
        print("  Device memory info not available")
    
    print("\n✅ Performance analysis completed")

# Run performance analysis
analyze_performance()

# %% [markdown]
# ## Visualization Gallery
# 
# Display some of the generated visualizations and demonstrate the rich output.

# %%
def display_visualization_gallery():
    """Display a gallery of generated visualizations."""
    
    print("🎨 Visualization Gallery")
    print("=" * 30)
    
    output_dir = visualizer.episode_manager.get_current_run_dir()
    
    if not output_dir.exists():
        print("No visualizations found. Run the demo first.")
        return
    
    # Find generated files
    svg_files = sorted(list(output_dir.glob("**/*.svg")))
    png_files = sorted(list(output_dir.glob("**/*.png")))
    
    print(f"Generated {len(svg_files)} SVG files and {len(png_files)} PNG files")
    
    # Display file structure
    print(f"\n📁 Output Structure:")
    for episode_dir in sorted(output_dir.glob("episode_*")):
        if episode_dir.is_dir():
            episode_files = list(episode_dir.glob("*"))
            print(f"  {episode_dir.name}/")
            for file in sorted(episode_files)[:5]:  # Show first 5 files
                print(f"    {file.name}")
            if len(episode_files) > 5:
                print(f"    ... and {len(episode_files) - 5} more files")
    
    # Show sample file paths for manual inspection
    print(f"\n🔍 Sample Files for Manual Inspection:")
    if svg_files:
        print(f"  First SVG: {svg_files[0]}")
        print(f"  Last SVG: {svg_files[-1]}")
    
    if png_files:
        print(f"  First PNG: {png_files[0]}")
        print(f"  Last PNG: {png_files[-1]}")
    
    # File size analysis
    total_size = sum(f.stat().st_size for f in svg_files + png_files)
    print(f"\n📊 Storage Statistics:")
    print(f"  Total files: {len(svg_files) + len(png_files)}")
    print(f"  Total size: {total_size / 1e6:.2f} MB")
    print(f"  Average file size: {total_size / len(svg_files + png_files) / 1e3:.1f} KB")
    
    print("\n💡 To view visualizations:")
    print(f"  1. Open any .svg file in a web browser")
    print(f"  2. View .png files in any image viewer")
    print(f"  3. Check the episode summaries for overview")

# Display visualization gallery
display_visualization_gallery()

# %% [markdown]
# ## Cleanup and Finalization
# 
# Clean up resources and finalize the experiment.

# %%
def cleanup_and_finalize():
    """Clean up resources and finalize the experiment."""
    
    print("🧹 Cleanup and Finalization")
    print("=" * 35)
    
    # Finalize Wandb run
    if wandb_success:
        print("Finalizing Wandb run...")
        visualizer.wandb_integration.finish_run()
        print("✅ Wandb run finalized")
    
    # Flush async logger
    if hasattr(visualizer, 'async_logger'):
        print("Flushing async logger...")
        # Force flush any remaining logs
        visualizer.async_logger.flush()
        print("✅ Async logger flushed")
    
    # Memory cleanup
    if hasattr(visualizer, 'memory_manager'):
        print("Performing memory cleanup...")
        cleanup_stats = visualizer.memory_manager.cleanup_memory()
        print(f"✅ Memory cleanup completed: {cleanup_stats}")
    
    # Final performance report
    if hasattr(visualizer, 'performance_monitor'):
        print("\n📊 Final Performance Report:")
        final_report = visualizer.performance_monitor.get_performance_report()
        for key, value in final_report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Storage summary
    output_dir = visualizer.episode_manager.get_current_run_dir()
    if output_dir.exists():
        total_files = len(list(output_dir.glob("**/*")))
        total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
        print(f"\n💾 Final Storage Summary:")
        print(f"  Output directory: {output_dir}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size / 1e6:.2f} MB")
    
    print("\n🎉 Comprehensive demo completed successfully!")
    print("\nKey achievements:")
    print("✅ MiniArc dataset integration")
    print("✅ Enhanced visualization system")
    print("✅ Wandb experiment tracking")
    print("✅ JAX-compiled random agent")
    print("✅ Bbox action format")
    print("✅ Raw environment configuration")
    print("✅ Performance optimization")
    print("✅ Comprehensive logging")

# Cleanup and finalize
cleanup_and_finalize()

# %% [markdown]
# ## Summary and Next Steps
# 
# This comprehensive demo has showcased the complete JaxARC ecosystem:
# 
# ### What We Demonstrated
# 1. **Dataset Integration**: Loaded and processed MiniArc tasks with 5x5 grids
# 2. **Environment Configuration**: Set up bbox actions with raw environment
# 3. **Enhanced Visualization**: Generated rich SVG/PNG visualizations with step-by-step analysis
# 4. **Wandb Integration**: Tracked experiments with comprehensive metrics
# 5. **JAX Optimization**: Implemented fully JIT-compiled agent for maximum performance
# 6. **Episode Management**: Organized outputs with automatic cleanup and storage management
# 7. **Performance Monitoring**: Analyzed computational efficiency and memory usage
# 
# ### Key Performance Results
# - **JAX Compilation**: Achieved significant speedup after JIT compilation
# - **Batch Processing**: Demonstrated scalable batch episode processing
# - **Memory Efficiency**: Maintained controlled memory usage with cleanup
# - **Visualization Overhead**: Kept visualization impact minimal through optimization
# 
# ### Generated Outputs
# - **Visualizations**: Step-by-step SVG and PNG files for each episode
# - **Episode Summaries**: Comprehensive analysis of each episode
# - **Performance Metrics**: Detailed timing and efficiency measurements
# - **Wandb Logs**: Experiment tracking data (offline mode)
# 
# ### Next Steps for Development
# 1. **Agent Development**: Replace random agent with learning algorithms
# 2. **Curriculum Learning**: Implement progressive difficulty scaling
# 3. **Multi-Task Learning**: Extend to handle multiple ARC task types
# 4. **Hyperparameter Optimization**: Use Wandb sweeps for systematic tuning
# 5. **Distributed Training**: Scale to multiple devices/machines
# 6. **Advanced Visualization**: Add interactive analysis tools
# 
# ### Usage Instructions
# ```python
# # To run this notebook:
# # 1. Ensure MiniArc dataset is downloaded
# # 2. Install wandb: pip install wandb
# # 3. Run: pixi run jupyter notebook jaxarc_comprehensive_demo.py
# # 4. Execute cells sequentially
# # 5. Check outputs/ directory for visualizations
# ```
# 
# This demo provides a complete foundation for ARC research and development with JaxARC!

# %%