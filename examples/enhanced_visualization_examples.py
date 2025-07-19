#!/usr/bin/env python3
"""
Enhanced Visualization System Examples

This script demonstrates various configurations and workflows for the enhanced
visualization and logging system in JaxARC.
"""

import jax
import jax.numpy as jnp
import time
import os
from pathlib import Path

from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.utils.visualization import (
    EnhancedVisualizer,
    VisualizationConfig,
    EpisodeManager,
    AsyncLogger,
    WandbIntegration,
    MemoryManager,
    PerformanceMonitor
)


def example_1_basic_setup():
    """Example 1: Basic enhanced visualization setup."""
    
    print("=" * 60)
    print("Example 1: Basic Enhanced Visualization Setup")
    print("=" * 60)
    
    # Basic configuration
    vis_config = VisualizationConfig(
        debug_level="standard",
        output_formats=["svg"],
        show_operation_names=True,
        highlight_changes=True,
        include_metrics=True
    )
    
    # Create enhanced visualizer
    visualizer = EnhancedVisualizer(vis_config)
    
    # Setup environment
    key = jax.random.PRNGKey(42)
    config = create_standard_config()
    state, obs = arc_reset(key, config)
    
    print(f"✅ Visualizer initialized")
    print(f"Output directory: {visualizer.get_output_directory()}")
    
    # Run a few steps with visualization
    for step in range(5):
        # Simple action: fill with color
        selection = jnp.ones_like(state.working_grid, dtype=jnp.bool_)
        action = {"selection": selection, "operation": jnp.array(step % 5 + 1)}
        
        # Environment step
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        # Visualize step
        visualizer.visualize_step(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step
        )
        
        print(f"Step {step}: Reward={reward:.3f}")
        state = new_state
        
        if done:
            break
    
    # Generate episode summary
    visualizer.visualize_episode_summary(episode_num=0)
    print("✅ Basic visualization example completed")


def example_2_performance_optimized():
    """Example 2: Performance-optimized configuration."""
    
    print("\n" + "=" * 60)
    print("Example 2: Performance-Optimized Configuration")
    print("=" * 60)
    
    # Performance-optimized configuration
    vis_config = VisualizationConfig(
        debug_level="minimal",
        output_formats=["svg"],
        image_quality="medium",
        lazy_loading=True,
        memory_limit_mb=200,
        log_frequency=10  # Log every 10th step
    )
    
    # Async logger for background processing
    async_logger = AsyncLogger(
        queue_size=500,
        worker_threads=1,
        batch_size=20,
        flush_interval=5.0
    )
    
    # Memory manager
    memory_manager = MemoryManager(
        max_memory_mb=300,
        cleanup_threshold=0.8,
        enable_lazy_loading=True
    )
    
    # Performance monitor
    perf_monitor = PerformanceMonitor(
        target_overhead=0.05,  # 5% max overhead
        auto_adjust=True
    )
    
    # Create optimized visualizer
    visualizer = EnhancedVisualizer(
        vis_config=vis_config,
        async_logger=async_logger,
        memory_manager=memory_manager,
        performance_monitor=perf_monitor
    )
    
    # Setup environment
    key = jax.random.PRNGKey(123)
    config = create_standard_config()
    
    print("🚀 Running performance test...")
    
    # Measure performance
    start_time = time.perf_counter()
    
    for episode in range(3):
        state, obs = arc_reset(key, config)
        visualizer.start_episode(episode)
        
        for step in range(20):
            # Random action
            selection = jax.random.bernoulli(key, 0.3, state.working_grid.shape)
            operation = jax.random.randint(key, (), 0, 10)
            action = {"selection": selection, "operation": operation}
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, config)
            
            # Visualize (only every 10th step due to log_frequency)
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        visualizer.visualize_episode_summary(episode)
    
    end_time = time.perf_counter()
    
    # Performance report
    perf_report = perf_monitor.get_performance_report()
    memory_stats = memory_manager.get_memory_stats()
    
    print(f"⏱️  Total time: {end_time - start_time:.2f}s")
    print(f"📊 Visualization overhead: {perf_report.get('visualization_overhead', 0):.1f}%")
    print(f"🧠 Memory usage: {memory_stats['current_mb']:.1f} MB")
    print("✅ Performance-optimized example completed")


def example_3_research_workflow():
    """Example 3: Research workflow with comprehensive logging."""
    
    print("\n" + "=" * 60)
    print("Example 3: Research Workflow Configuration")
    print("=" * 60)
    
    # Research configuration with detailed logging
    vis_config = VisualizationConfig(
        debug_level="verbose",
        output_formats=["svg", "png"],
        image_quality="high",
        show_coordinates=True,
        show_operation_names=True,
        highlight_changes=True,
        include_metrics=True,
        enable_comparisons=True,
        save_intermediate_states=True
    )
    
    # Episode management for organized storage
    episode_manager = EpisodeManager(
        base_output_dir="outputs/research_experiment",
        run_name="baseline_study_v1",
        max_episodes_per_run=100,
        cleanup_policy="size_based",
        max_storage_gb=5.0
    )
    
    # Wandb integration for experiment tracking
    wandb_config = {
        "enabled": True,
        "project_name": "jaxarc-research-demo",
        "tags": ["demo", "research", "baseline"],
        "log_frequency": 5,
        "image_format": "png",
        "offline_mode": True  # Use offline mode for demo
    }
    
    wandb_integration = WandbIntegration(wandb_config)
    
    # Create research visualizer
    visualizer = EnhancedVisualizer(
        vis_config=vis_config,
        episode_manager=episode_manager,
        wandb_integration=wandb_integration
    )
    
    # Initialize wandb run
    experiment_config = {
        "algorithm": "Random",
        "max_episodes": 2,
        "max_steps": 15,
        "visualization_level": "verbose"
    }
    
    wandb_integration.initialize_run(
        experiment_config=experiment_config,
        run_name="research_demo_run"
    )
    
    print(f"📁 Research output directory: {episode_manager.get_current_run_dir()}")
    
    # Run research experiment
    key = jax.random.PRNGKey(456)
    config = create_standard_config()
    
    for episode in range(2):
        print(f"\n🔬 Running research episode {episode}")
        
        state, obs = arc_reset(key, config)
        visualizer.start_episode(episode)
        
        episode_reward = 0.0
        
        for step in range(15):
            # More sophisticated action selection for research
            if step < 5:
                # Exploration phase
                selection = jax.random.bernoulli(key, 0.2, state.working_grid.shape)
                operation = jax.random.randint(key, (), 0, 5)
            else:
                # Focused phase
                selection = jax.random.bernoulli(key, 0.1, state.working_grid.shape)
                operation = jax.random.randint(key, (), 1, 3)
            
            action = {"selection": selection, "operation": operation}
            
            # Environment step
            new_state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            
            # Comprehensive visualization
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            # Log to wandb
            wandb_integration.log_step(
                step_num=episode * 15 + step,
                metrics={
                    "step_reward": reward,
                    "cumulative_reward": episode_reward,
                    "exploration_phase": step < 5
                }
            )
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        # Episode summary with research metrics
        episode_summary = {
            "episode_reward": episode_reward,
            "episode_steps": step + 1,
            "final_similarity": info.get("similarity", 0.0),
            "success": info.get("success", False)
        }
        
        visualizer.visualize_episode_summary(episode)
        wandb_integration.log_episode_summary(episode, episode_summary)
        
        print(f"📊 Episode {episode} summary: Reward={episode_reward:.3f}, Steps={step + 1}")
    
    # Finish wandb run
    wandb_integration.finish_run()
    print("✅ Research workflow example completed")


def example_4_production_deployment():
    """Example 4: Production deployment configuration."""
    
    print("\n" + "=" * 60)
    print("Example 4: Production Deployment Configuration")
    print("=" * 60)
    
    # Production configuration - minimal overhead
    vis_config = VisualizationConfig(
        debug_level="minimal",
        output_formats=["svg"],
        image_quality="low",
        lazy_loading=True,
        memory_limit_mb=100,
        log_frequency=100,  # Very infrequent logging
        enable_comparisons=False,
        save_intermediate_states=False
    )
    
    # Minimal async processing
    async_logger = AsyncLogger(
        queue_size=200,
        worker_threads=1,
        batch_size=50,
        flush_interval=30.0  # Less frequent flushing
    )
    
    # Aggressive memory management
    memory_manager = MemoryManager(
        max_memory_mb=150,
        cleanup_threshold=0.7,
        enable_lazy_loading=True,
        compression_level=9  # Maximum compression
    )
    
    # Episode management with strict limits
    episode_manager = EpisodeManager(
        base_output_dir="outputs/production",
        cleanup_policy="oldest_first",
        max_episodes_per_run=50,
        max_storage_gb=1.0  # Strict storage limit
    )
    
    # Create production visualizer
    visualizer = EnhancedVisualizer(
        vis_config=vis_config,
        async_logger=async_logger,
        memory_manager=memory_manager,
        episode_manager=episode_manager
    )
    
    print("🏭 Production configuration active")
    print(f"Memory limit: {memory_manager.max_memory_mb} MB")
    print(f"Storage limit: {episode_manager.max_storage_gb} GB")
    print(f"Log frequency: Every {vis_config.log_frequency} steps")
    
    # Simulate production training
    key = jax.random.PRNGKey(789)
    config = create_standard_config()
    
    # Run multiple episodes quickly
    for episode in range(5):
        state, obs = arc_reset(key, config)
        
        if episode % vis_config.log_frequency == 0:  # Only visualize some episodes
            visualizer.start_episode(episode)
        
        for step in range(10):  # Short episodes for demo
            selection = jax.random.bernoulli(key, 0.1, state.working_grid.shape)
            operation = jax.random.randint(key, (), 0, 5)
            action = {"selection": selection, "operation": operation}
            
            new_state, obs, reward, done, info = arc_step(state, action, config)
            
            # Minimal visualization
            if episode % vis_config.log_frequency == 0 and step % vis_config.log_frequency == 0:
                visualizer.visualize_step(
                    before_state=state,
                    action=action,
                    after_state=new_state,
                    reward=reward,
                    info=info,
                    step_num=step
                )
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        if episode % 2 == 0:
            print(f"Episode {episode} completed")
    
    # Check resource usage
    memory_stats = memory_manager.get_memory_stats()
    storage_stats = episode_manager.get_storage_stats()
    
    print(f"🧠 Final memory usage: {memory_stats['current_mb']:.1f} MB")
    print(f"💾 Storage used: {storage_stats.get('used_gb', 0):.2f} GB")
    print("✅ Production deployment example completed")


def example_5_debugging_workflow():
    """Example 5: Debugging workflow with maximum detail."""
    
    print("\n" + "=" * 60)
    print("Example 5: Debugging Workflow Configuration")
    print("=" * 60)
    
    # Maximum debugging configuration
    vis_config = VisualizationConfig(
        debug_level="full",
        output_formats=["svg", "png"],
        image_quality="high",
        show_coordinates=True,
        show_operation_names=True,
        highlight_changes=True,
        include_metrics=True,
        enable_comparisons=True,
        save_intermediate_states=True,
        log_frequency=1  # Log every step
    )
    
    # Performance monitoring for debugging
    perf_monitor = PerformanceMonitor(
        target_overhead=0.5,  # Allow high overhead for debugging
        measurement_window=10,
        auto_adjust=False  # Don't auto-adjust during debugging
    )
    
    # Create debugging visualizer
    visualizer = EnhancedVisualizer(
        vis_config=vis_config,
        performance_monitor=perf_monitor
    )
    
    print("🐛 Debug mode active - maximum detail logging")
    
    # Setup environment
    key = jax.random.PRNGKey(999)
    config = create_standard_config()
    state, obs = arc_reset(key, config)
    
    # Debug a specific scenario
    print("🔍 Debugging specific action sequence...")
    
    debug_actions = [
        {"selection": jnp.array([[True, False], [False, True]]), "operation": jnp.array(1)},
        {"selection": jnp.array([[False, True], [True, False]]), "operation": jnp.array(2)},
        {"selection": jnp.array([[True, True], [False, False]]), "operation": jnp.array(3)},
    ]
    
    visualizer.start_episode(0)
    
    for step, action in enumerate(debug_actions):
        print(f"\n🔍 Debug step {step}:")
        print(f"  Action: {action}")
        
        # Environment step
        new_state, obs, reward, done, info = arc_step(state, action, config)
        
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        # Detailed visualization
        visualizer.visualize_step(
            before_state=state,
            action=action,
            after_state=new_state,
            reward=reward,
            info=info,
            step_num=step
        )
        
        # Additional debug information
        if hasattr(visualizer, 'get_step_debug_info'):
            debug_info = visualizer.get_step_debug_info(step)
            print(f"  Debug info: {debug_info}")
        
        state = new_state
        
        if done:
            print("  Episode terminated")
            break
    
    # Generate comprehensive episode summary
    visualizer.visualize_episode_summary(episode_num=0)
    
    # Performance report
    perf_report = perf_monitor.get_performance_report()
    print(f"\n📊 Debug session performance:")
    print(f"  Average step time: {perf_report.get('avg_step_time', 0):.3f}s")
    print(f"  Visualization overhead: {perf_report.get('visualization_overhead', 0):.1f}%")
    
    print("✅ Debugging workflow example completed")


def example_6_custom_configuration():
    """Example 6: Custom configuration for specific needs."""
    
    print("\n" + "=" * 60)
    print("Example 6: Custom Configuration Example")
    print("=" * 60)
    
    # Custom configuration combining different aspects
    vis_config = VisualizationConfig(
        debug_level="standard",
        output_formats=["svg"],
        image_quality="high",
        show_operation_names=True,
        highlight_changes=True,
        include_metrics=True,
        color_scheme="high_contrast",  # Custom color scheme
        log_frequency=5,
        enable_comparisons=True
    )
    
    # Custom episode management
    episode_manager = EpisodeManager(
        base_output_dir="outputs/custom_experiment",
        run_name="custom_config_demo",
        episode_dir_format="ep_{episode:03d}",
        step_file_format="step_{step:02d}",
        max_episodes_per_run=20
    )
    
    # Custom async configuration
    async_logger = AsyncLogger(
        queue_size=800,
        worker_threads=2,
        batch_size=15,
        flush_interval=8.0,
        enable_compression=True
    )
    
    # Create custom visualizer
    visualizer = EnhancedVisualizer(
        vis_config=vis_config,
        episode_manager=episode_manager,
        async_logger=async_logger
    )
    
    print("🎨 Custom configuration loaded")
    print(f"Color scheme: {vis_config.color_scheme}")
    print(f"Episode format: {episode_manager.episode_dir_format}")
    print(f"Async workers: {async_logger.worker_threads}")
    
    # Demonstrate custom workflow
    key = jax.random.PRNGKey(111)
    config = create_standard_config()
    
    for episode in range(2):
        state, obs = arc_reset(key, config)
        visualizer.start_episode(episode)
        
        print(f"\n🎨 Custom episode {episode}")
        
        for step in range(8):
            # Custom action pattern
            if step % 2 == 0:
                # Even steps: small selections
                selection = jax.random.bernoulli(key, 0.1, state.working_grid.shape)
            else:
                # Odd steps: larger selections
                selection = jax.random.bernoulli(key, 0.3, state.working_grid.shape)
            
            operation = jax.random.randint(key, (), 1, 6)
            action = {"selection": selection, "operation": operation}
            
            new_state, obs, reward, done, info = arc_step(state, action, config)
            
            visualizer.visualize_step(
                before_state=state,
                action=action,
                after_state=new_state,
                reward=reward,
                info=info,
                step_num=step
            )
            
            state = new_state
            key = jax.random.split(key)[0]
            
            if done:
                break
        
        visualizer.visualize_episode_summary(episode)
        print(f"Episode {episode} completed with custom visualization")
    
    print("✅ Custom configuration example completed")


def main():
    """Run all enhanced visualization examples."""
    
    print("🚀 JaxARC Enhanced Visualization System Examples")
    print("This script demonstrates various configurations and workflows.")
    print("Each example shows different use cases and optimization strategies.")
    
    try:
        # Run all examples
        example_1_basic_setup()
        example_2_performance_optimized()
        example_3_research_workflow()
        example_4_production_deployment()
        example_5_debugging_workflow()
        example_6_custom_configuration()
        
        print("\n" + "=" * 60)
        print("🎉 All Enhanced Visualization Examples Completed!")
        print("=" * 60)
        print("\nCheck the 'outputs/' directory for generated visualizations.")
        print("Each example created different types of output demonstrating")
        print("various configuration options and use cases.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()