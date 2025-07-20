"""
Debug Levels Demo for JaxARC Enhanced Visualization.

This example demonstrates the different debug levels available in the enhanced
visualization system and how they affect performance and output.

Usage:
    pixi run python examples/debug_levels_demo.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.utils.config import get_config


def demo_debug_level(level: str, console: Console) -> dict:
    """Demonstrate a specific debug level and return performance metrics."""
    console.print(f"\n[bold blue]Testing Debug Level: {level}[/bold blue]")
    
    # Create configuration with specific debug level
    config_overrides = [
        f"debug/={level}",
        "action.selection_format=point",
        "max_episode_steps=10",
    ]
    
    try:
        hydra_config = get_config(overrides=config_overrides)
        typed_config = ArcEnvConfig.from_hydra(hydra_config)
        
        # Create temporary directory for this debug level
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory to use temp dir
            if hasattr(hydra_config, 'visualization') and hydra_config.visualization:
                vis_config = OmegaConf.create({
                    "visualization": {
                        **hydra_config.visualization,
                        "output_dir": temp_dir
                    }
                })
                hydra_config = OmegaConf.merge(hydra_config, vis_config)
            
            # Convert to unified config and create environment
            from jaxarc.envs.equinox_config import JaxArcConfig
            unified_config = JaxArcConfig.from_hydra(hydra_config)
            env = ArcEnvironment(unified_config)
            
            # Measure performance
            start_time = time.time()
            
            # Run episode
            key = jr.PRNGKey(42)
            state, obs = env.reset(key)
            
            step_times = []
            for i in range(8):
                step_start = time.time()
                
                action_key, key = jr.split(key)
                
                # Create a simple point action
                row, col = jr.randint(action_key, shape=(2,), minval=0, maxval=5)
                action = {
                    "point": jnp.array([row, col]),
                    "operation": jr.randint(action_key, shape=(), minval=0, maxval=10),
                }
                
                state, obs, reward, info = env.step(action)
                
                step_end = time.time()
                step_times.append(step_end - step_start)
                
                if env.is_done:
                    break
            
            end_time = time.time()
            
            # Count generated files
            output_files = list(Path(temp_dir).rglob("*"))
            svg_files = list(Path(temp_dir).rglob("*.svg"))
            png_files = list(Path(temp_dir).rglob("*.png"))
            json_files = list(Path(temp_dir).rglob("*.json"))
            
            # Clean up
            env.close()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_step_time = sum(step_times) / len(step_times) if step_times else 0
            
            metrics = {
                "level": level,
                "total_time": total_time,
                "avg_step_time": avg_step_time,
                "steps_completed": len(step_times),
                "total_files": len(output_files),
                "svg_files": len(svg_files),
                "png_files": len(png_files),
                "json_files": len(json_files),
                "final_similarity": float(state.similarity_score),
                "enhanced_viz_enabled": hasattr(env, '_enhanced_visualizer') and env._enhanced_visualizer is not None,
            }
            
            # Display results
            console.print(f"  Total time: {total_time:.3f}s")
            console.print(f"  Avg step time: {avg_step_time:.3f}s")
            console.print(f"  Steps completed: {len(step_times)}")
            console.print(f"  Files generated: {len(output_files)} total")
            console.print(f"    - SVG files: {len(svg_files)}")
            console.print(f"    - PNG files: {len(png_files)}")
            console.print(f"    - JSON files: {len(json_files)}")
            console.print(f"  Enhanced visualization: {'Enabled' if metrics['enhanced_viz_enabled'] else 'Disabled'}")
            console.print(f"  Final similarity: {float(state.similarity_score):.3f}")
            
            return metrics
            
    except Exception as e:
        console.print(f"  [red]Error testing {level}: {e}[/red]")
        return {
            "level": level,
            "error": str(e),
            "total_time": 0,
            "avg_step_time": 0,
            "steps_completed": 0,
            "total_files": 0,
            "svg_files": 0,
            "png_files": 0,
            "json_files": 0,
            "final_similarity": 0,
            "enhanced_viz_enabled": False,
        }


def create_comparison_table(results: list[dict]) -> Table:
    """Create a comparison table of debug level results."""
    table = Table(title="Debug Level Performance Comparison")
    
    table.add_column("Debug Level", style="cyan", no_wrap=True)
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Avg Step Time (s)", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Files Generated", justify="right")
    table.add_column("Enhanced Viz", justify="center")
    table.add_column("Final Similarity", justify="right")
    
    for result in results:
        if "error" in result:
            table.add_row(
                result["level"],
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
                "[red]ERROR[/red]",
            )
        else:
            table.add_row(
                result["level"],
                f"{result['total_time']:.3f}",
                f"{result['avg_step_time']:.3f}",
                str(result["steps_completed"]),
                str(result["total_files"]),
                "✓" if result["enhanced_viz_enabled"] else "✗",
                f"{result['final_similarity']:.3f}",
            )
    
    return table


def demo_configuration_impact():
    """Demonstrate how different configurations impact visualization."""
    console = Console()
    console.print("\n[bold blue]Configuration Impact Demo[/bold blue]")
    
    configurations = [
        {
            "name": "Minimal Config",
            "overrides": [
                "debug/=minimal",
                "visualization/=debug_minimal",
                "action.selection_format=point",
            ]
        },
        {
            "name": "Standard Config",
            "overrides": [
                "debug/=on",
                "visualization/=debug_standard",
                "action.selection_format=bbox",
            ]
        },
        {
            "name": "Verbose Config",
            "overrides": [
                "debug/=verbose",
                "visualization/=debug_verbose",
                "action.selection_format=mask",
            ]
        }
    ]
    
    for config_info in configurations:
        console.print(f"\n[green]{config_info['name']}:[/green]")
        
        try:
            hydra_config = get_config(overrides=config_info['overrides'])
            
            # Display key configuration values
            if hasattr(hydra_config, 'visualization'):
                vis_cfg = hydra_config.visualization
                console.print(f"  Debug level: {vis_cfg.get('debug_level', 'N/A')}")
                console.print(f"  Output formats: {vis_cfg.get('output_formats', [])}")
                console.print(f"  Show operation names: {vis_cfg.get('show_operation_names', False)}")
                console.print(f"  Highlight changes: {vis_cfg.get('highlight_changes', False)}")
                console.print(f"  Include metrics: {vis_cfg.get('include_metrics', False)}")
            
            if hasattr(hydra_config, 'enhanced_visualization'):
                enhanced_cfg = hydra_config.enhanced_visualization
                console.print(f"  Enhanced visualization: {enhanced_cfg.get('enabled', False)}")
                console.print(f"  Enhanced level: {enhanced_cfg.get('level', 'N/A')}")
            
            console.print("  ✓ Configuration valid")
            
        except Exception as e:
            console.print(f"  ✗ Configuration error: {e}")


def demo_performance_impact():
    """Demonstrate performance impact of different visualization levels."""
    console = Console()
    console.print("\n[bold blue]Performance Impact Analysis[/bold blue]")
    
    # Test with and without visualization
    test_configs = [
        ("No Visualization", ["debug/=off"]),
        ("Minimal Visualization", ["debug/=minimal"]),
        ("Standard Visualization", ["debug/=on"]),
        ("Full Visualization", ["debug/=full"]),
    ]
    
    performance_results = []
    
    for config_name, overrides in test_configs:
        console.print(f"\n[green]Testing {config_name}...[/green]")
        
        try:
            # Run multiple episodes to get average performance
            times = []
            
            for run in range(3):  # 3 runs for averaging
                config_overrides = overrides + [
                    "action.selection_format=point",
                    "max_episode_steps=5",  # Shorter episodes for faster testing
                ]
                
                hydra_config = get_config(overrides=config_overrides)
                typed_config = ArcEnvConfig.from_hydra(hydra_config)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Update output directory
                    if hasattr(hydra_config, 'visualization') and hydra_config.visualization:
                        vis_config = OmegaConf.create({
                            "visualization": {
                                **hydra_config.visualization,
                                "output_dir": temp_dir
                            }
                        })
                        hydra_config = OmegaConf.merge(hydra_config, vis_config)
                    
                    unified_config = JaxArcConfig.from_hydra(hydra_config)
                    env = ArcEnvironment(unified_config)
                    
                    start_time = time.time()
                    
                    key = jr.PRNGKey(42 + run)
                    state, obs = env.reset(key)
                    
                    for i in range(5):
                        action_key, key = jr.split(key)
                        row, col = jr.randint(action_key, shape=(2,), minval=0, maxval=3)
                        action = {
                            "point": jnp.array([row, col]),
                            "operation": jr.randint(action_key, shape=(), minval=0, maxval=5),
                        }
                        state, obs, reward, info = env.step(action)
                        
                        if env.is_done:
                            break
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    env.close()
            
            avg_time = sum(times) / len(times)
            performance_results.append((config_name, avg_time, times))
            
            console.print(f"  Average time: {avg_time:.3f}s")
            console.print(f"  Individual runs: {[f'{t:.3f}s' for t in times]}")
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            performance_results.append((config_name, float('inf'), []))
    
    # Create performance comparison
    if performance_results:
        console.print("\n[bold yellow]Performance Summary:[/bold yellow]")
        baseline_time = performance_results[0][1] if performance_results else 1.0
        
        for config_name, avg_time, _ in performance_results:
            if avg_time != float('inf'):
                overhead = ((avg_time - baseline_time) / baseline_time) * 100
                console.print(f"  {config_name}: {avg_time:.3f}s ({overhead:+.1f}% vs baseline)")
            else:
                console.print(f"  {config_name}: ERROR")


def main():
    """Main demo function."""
    logger.info("Starting Debug Levels Demo")
    
    console = Console()
    console.print("[bold yellow]JaxARC Debug Levels Demo[/bold yellow]")
    console.print("This demo compares different debug levels and their impact on:")
    console.print("• Performance (execution time)")
    console.print("• File generation (visualizations, logs)")
    console.print("• Memory usage")
    console.print("• Feature availability")
    
    # Test all debug levels
    debug_levels = ["off", "minimal", "on", "verbose", "full"]
    results = []
    
    console.print("\n[bold cyan]Testing Debug Levels...[/bold cyan]")
    
    for level in debug_levels:
        result = demo_debug_level(level, console)
        results.append(result)
    
    # Display comparison table
    console.print("\n")
    comparison_table = create_comparison_table(results)
    console.print(comparison_table)
    
    # Additional demos
    demo_configuration_impact()
    demo_performance_impact()
    
    console.print("\n[bold green]Debug Levels Demo Completed![/bold green]")
    console.print("\nKey Takeaways:")
    console.print("• 'off' level provides maximum performance with no visualization")
    console.print("• 'minimal' level provides episode summaries with minimal overhead")
    console.print("• 'on'/'standard' level balances information and performance")
    console.print("• 'verbose' level provides detailed step-by-step information")
    console.print("• 'full' level provides complete debugging information")
    console.print("\nChoose the appropriate level based on your use case:")
    console.print("• Training: 'off' or 'minimal'")
    console.print("• Development: 'on' or 'verbose'")
    console.print("• Debugging: 'full'")


if __name__ == "__main__":
    main()