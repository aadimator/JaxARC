"""
JaxARC Enhanced Visualization Notebook Demo.

This example demonstrates the complete enhanced visualization and logging system including:
- Enhanced visualization with different debug levels
- Weights & Biases integration
- Episode management and replay
- Asynchronous logging
- Performance monitoring
- Configuration composition
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.utils.config import get_config
from jaxarc.utils.visualization import (
    EnhancedVisualizer,
    VisualizationConfig,
    EpisodeManager,
    EpisodeConfig,
    AsyncLogger,
    AsyncLoggerConfig,
    WandbIntegration,
    create_development_wandb_config,
    EpisodeReplaySystem,
    ReplayConfig,
    EpisodeAnalysisTools,
    AnalysisConfig,
)


def demo_enhanced_visualization_levels():
    """Demonstrate different visualization debug levels."""
    console = Console()
    console.print("\n[bold blue]Demo 1: Enhanced Visualization Debug Levels[/bold blue]")
    
    debug_levels = ["off", "minimal", "standard", "verbose", "full"]
    
    for level in debug_levels:
        console.print(f"\n[green]Testing debug level: {level}[/green]")
        
        # Create configuration with specific debug level
        config_overrides = [
            f"debug/={level}",
            "action.selection_format=point",
            "max_episode_steps=5",
        ]
        
        hydra_config = get_config(overrides=config_overrides)
        typed_config = ArcEnvConfig.from_hydra(hydra_config)
        
        # Create environment with enhanced visualization
        env = ArcEnvironment(typed_config, hydra_config)
        
        # Run a short episode
        key = jr.PRNGKey(42 + hash(level) % 1000)
        state, obs = env.reset(key)
        
        # Take a few actions
        for i in range(3):
            action_key, key = jr.split(key)
            
            # Create a simple point action
            row, col = jr.randint(action_key, shape=(2,), minval=0, maxval=5)
            action = {
                "point": jnp.array([row, col]),
                "operation": jr.randint(action_key, shape=(), minval=0, maxval=10),
            }
            
            state, obs, reward, info = env.step(action)
            
            if env.is_done:
                break
        
        # Clean up
        env.close()
        
        console.print(f"  Completed episode with {state.step_count} steps")
        if hasattr(env, '_enhanced_visualizer') and env._enhanced_visualizer:
            console.print(f"  Enhanced visualization: Enabled")
        else:
            console.print(f"  Enhanced visualization: Disabled")


def demo_wandb_integration():
    """Demonstrate Weights & Biases integration."""
    console = Console()
    console.print("\n[bold blue]Demo 2: Weights & Biases Integration[/bold blue]")
    
    # Create configuration with wandb enabled
    config_overrides = [
        "debug/=on",
        "action.selection_format=bbox",
        "max_episode_steps=10",
    ]
    
    # Create wandb configuration
    wandb_config = {
        "wandb": {
            "enabled": True,
            "project_name": "jaxarc-demo",
            "tags": ["demo", "enhanced-visualization"],
            "log_frequency": 2,
        }
    }
    
    hydra_config = get_config(overrides=config_overrides)
    # Merge wandb config
    hydra_config = OmegaConf.merge(hydra_config, OmegaConf.create(wandb_config))
    
    typed_config = ArcEnvConfig.from_hydra(hydra_config)
    
    console.print("[green]Creating environment with wandb integration...[/green]")
    
    try:
        # Create environment with wandb integration
        env = ArcEnvironment(typed_config, hydra_config)
        
        # Run episode
        key = jr.PRNGKey(123)
        state, obs = env.reset(key)
        
        total_reward = 0.0
        for i in range(5):
            action_key, key = jr.split(key)
            
            # Create a bbox action
            coords = jr.randint(action_key, shape=(4,), minval=0, maxval=5)
            action = {
                "bbox": coords,
                "operation": jr.randint(action_key, shape=(), minval=0, maxval=15),
            }
            
            state, obs, reward, info = env.step(action)
            total_reward += float(reward)
            
            console.print(f"  Step {i+1}: reward={float(reward):.3f}, similarity={float(info['similarity']):.3f}")
            
            if env.is_done:
                break
        
        console.print(f"[green]Episode completed! Total reward: {total_reward:.3f}[/green]")
        
        # Clean up
        env.close()
        
    except ImportError:
        console.print("[yellow]wandb not available - skipping wandb integration demo[/yellow]")
    except Exception as e:
        console.print(f"[yellow]wandb integration failed: {e}[/yellow]")


def demo_episode_management():
    """Demonstrate episode management and storage."""
    console = Console()
    console.print("\n[bold blue]Demo 3: Episode Management and Storage[/bold blue]")
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        console.print(f"[green]Using temporary directory: {temp_dir}[/green]")
        
        # Create episode manager
        episode_config = EpisodeConfig(
            base_output_dir=temp_dir,
            cleanup_policy="size_based",
            max_storage_gb=0.1,  # Small limit for demo
        )
        
        episode_manager = EpisodeManager(episode_config)
        
        # Start a new run
        run_dir = episode_manager.start_new_run("demo_run")
        console.print(f"Started new run: {run_dir}")
        
        # Create multiple episodes
        for episode_num in range(3):
            episode_dir = episode_manager.start_new_episode(episode_num)
            console.print(f"  Episode {episode_num}: {episode_dir}")
            
            # Simulate saving some step visualizations
            for step in range(5):
                step_path = episode_manager.get_step_path(step, "svg")
                step_path.parent.mkdir(parents=True, exist_ok=True)
                step_path.write_text(f"<svg>Step {step} visualization</svg>")
        
        # Check storage usage
        console.print(f"[green]Created {len(list(Path(temp_dir).rglob('*.svg')))} visualization files[/green]")
        
        # Demonstrate cleanup
        episode_manager.cleanup_old_data()
        console.print("[green]Cleanup completed[/green]")


def demo_async_logging():
    """Demonstrate asynchronous logging system."""
    console = Console()
    console.print("\n[bold blue]Demo 4: Asynchronous Logging System[/bold blue]")
    
    # Create async logger
    async_config = AsyncLoggerConfig(
        queue_size=100,
        worker_threads=2,
        batch_size=5,
        flush_interval=1.0,
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        async_logger = AsyncLogger(async_config)
        
        console.print("[green]Created async logger with 2 worker threads[/green]")
        
        # Simulate logging many entries
        console.print("[green]Logging 20 entries asynchronously...[/green]")
        
        for i in range(20):
            log_entry = {
                "step": i,
                "reward": float(jr.normal(jr.PRNGKey(i))),
                "similarity": float(jr.uniform(jr.PRNGKey(i + 100))),
                "timestamp": i * 0.1,
            }
            
            async_logger.log_step_data(log_entry)
        
        # Force flush
        async_logger.flush()
        console.print("[green]All entries logged and flushed[/green]")
        
        # Clean up
        async_logger.close()
        console.print("[green]Async logger closed[/green]")


def demo_episode_replay():
    """Demonstrate episode replay and analysis."""
    console = Console()
    console.print("\n[bold blue]Demo 5: Episode Replay and Analysis[/bold blue]")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create replay system
        replay_config = ReplayConfig(
            replay_data_dir=temp_dir,
            validation_enabled=True,
            compression_enabled=True,
        )
        
        replay_system = EpisodeReplaySystem(replay_config)
        
        console.print("[green]Created episode replay system[/green]")
        
        # Create analysis tools
        analysis_config = AnalysisConfig(
            enable_failure_analysis=True,
            enable_performance_metrics=True,
            similarity_threshold=0.8,
        )
        
        analysis_tools = EpisodeAnalysisTools(analysis_config)
        
        console.print("[green]Created analysis tools[/green]")
        
        # Simulate some episode data for analysis
        episode_data = {
            "episode_num": 1,
            "total_steps": 10,
            "total_reward": 5.5,
            "final_similarity": 0.75,
            "success": False,
            "steps": [
                {"step": i, "reward": 0.5 + i * 0.1, "similarity": 0.1 + i * 0.07}
                for i in range(10)
            ]
        }
        
        # Analyze episode
        metrics = analysis_tools.analyze_episode(episode_data)
        console.print(f"[green]Episode analysis completed[/green]")
        console.print(f"  Average reward per step: {metrics.get('avg_reward_per_step', 0):.3f}")
        console.print(f"  Similarity improvement: {metrics.get('similarity_improvement', 0):.3f}")
        console.print(f"  Success: {metrics.get('success', False)}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    console = Console()
    console.print("\n[bold blue]Demo 6: Performance Monitoring[/bold blue]")
    
    # Create configuration with performance monitoring
    config_overrides = [
        "debug/=verbose",
        "action.selection_format=mask",
        "max_episode_steps=15",
    ]
    
    hydra_config = get_config(overrides=config_overrides)
    typed_config = ArcEnvConfig.from_hydra(hydra_config)
    
    # Add performance monitoring config
    perf_config = {
        "performance_monitoring": {
            "enabled": True,
            "memory_tracking": True,
            "timing_enabled": True,
        }
    }
    hydra_config = OmegaConf.merge(hydra_config, OmegaConf.create(perf_config))
    
    console.print("[green]Running environment with performance monitoring...[/green]")
    
    # Create environment
    env = ArcEnvironment(typed_config, hydra_config)
    
    # Run episode with timing
    import time
    start_time = time.time()
    
    key = jr.PRNGKey(456)
    state, obs = env.reset(key)
    
    for i in range(10):
        action_key, key = jr.split(key)
        
        # Create a mask action
        mask = jr.bernoulli(action_key, 0.3, shape=state.working_grid.shape)
        action = {
            "mask": mask,
            "operation": jr.randint(action_key, shape=(), minval=0, maxval=20),
        }
        
        state, obs, reward, info = env.step(action)
        
        if env.is_done:
            break
    
    end_time = time.time()
    
    console.print(f"[green]Episode completed in {end_time - start_time:.3f} seconds[/green]")
    console.print(f"  Steps: {state.step_count}")
    console.print(f"  Final similarity: {float(state.similarity_score):.3f}")
    
    # Clean up
    env.close()


def demo_configuration_composition():
    """Demonstrate configuration composition and validation."""
    console = Console()
    console.print("\n[bold blue]Demo 7: Configuration Composition[/bold blue]")
    
    # Demonstrate different configuration compositions
    compositions = [
        {
            "name": "Research Configuration",
            "overrides": [
                "debug/=full",
                "visualization/=debug_full",
                "storage/=research",
                "logging/=wandb_full",
            ]
        },
        {
            "name": "Development Configuration", 
            "overrides": [
                "debug/=on",
                "visualization/=debug_standard",
                "storage/=development",
                "logging/=local_only",
            ]
        },
        {
            "name": "Performance Configuration",
            "overrides": [
                "debug/=off",
                "visualization/=debug_off",
                "storage/=development",
                "logging/=local_only",
            ]
        }
    ]
    
    for comp in compositions:
        console.print(f"\n[green]{comp['name']}:[/green]")
        
        try:
            hydra_config = get_config(overrides=comp['overrides'])
            typed_config = ArcEnvConfig.from_hydra(hydra_config)
            
            console.print(f"  Debug level: {hydra_config.get('visualization', {}).get('debug_level', 'N/A')}")
            console.print(f"  Enhanced visualization: {hydra_config.get('enhanced_visualization', {}).get('enabled', False)}")
            console.print(f"  Async logging: {typed_config.debug.async_logging}")
            console.print(f"  Wandb enabled: {typed_config.debug.wandb_enabled}")
            console.print("  ✓ Configuration valid")
            
        except Exception as e:
            console.print(f"  ✗ Configuration error: {e}")


def main():
    """Main demo function."""
    logger.info("Starting JaxARC Enhanced Visualization Notebook Demo")
    
    console = Console()
    console.print("[bold yellow]JaxARC Enhanced Visualization Notebook Demo[/bold yellow]")
    console.print("This comprehensive demo showcases:")
    console.print("1. Enhanced visualization debug levels")
    console.print("2. Weights & Biases integration")
    console.print("3. Episode management and storage")
    console.print("4. Asynchronous logging system")
    console.print("5. Episode replay and analysis")
    console.print("6. Performance monitoring")
    console.print("7. Configuration composition")
    
    try:
        # Run all demos
        demo_enhanced_visualization_levels()
        demo_wandb_integration()
        demo_episode_management()
        demo_async_logging()
        demo_episode_replay()
        demo_performance_monitoring()
        demo_configuration_composition()
        
        console.print("\n[bold green]All demos completed successfully![/bold green]")
        console.print("\nThe enhanced visualization system provides:")
        console.print("• Multiple debug levels for different use cases")
        console.print("• Seamless wandb integration for experiment tracking")
        console.print("• Efficient episode storage and management")
        console.print("• Asynchronous logging for minimal performance impact")
        console.print("• Comprehensive replay and analysis capabilities")
        console.print("• Built-in performance monitoring")
        console.print("• Flexible configuration composition")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        console.print(f"[bold red]Demo failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()