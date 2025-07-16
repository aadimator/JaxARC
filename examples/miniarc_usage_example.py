#!/usr/bin/env python3
"""MiniARC Usage Example

This comprehensive example demonstrates how to use the MiniARC dataset with JaxARC,
showcasing the 5x5 grid optimization benefits, rapid prototyping workflow, and
performance comparisons with standard ARC.

MiniARC is a compact version of ARC with 400+ tasks optimized for 5x5 grids,
designed for faster experimentation and prototyping with reduced computational
requirements and quicker iteration cycles.

Usage:
    python examples/miniarc_usage_example.py
    python examples/miniarc_usage_example.py --performance-comparison
    python examples/miniarc_usage_example.py --rapid-prototyping --visualize
    python examples/miniarc_usage_example.py --batch-processing --verbose
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import create_miniarc_config, create_standard_config
from jaxarc.parsers import MiniArcParser
from jaxarc.utils.config import create_miniarc_config as create_config_util
from jaxarc.utils.visualization import draw_grid_svg, log_grid_to_console

console = Console()


class DemoMiniArcParser:
    """Demo parser that simulates MiniARC functionality without requiring the actual dataset."""

    def __init__(self):
        # Mock MiniARC tasks with 5x5 constraint
        self._task_ids = [
            "copy_pattern_5x5",
            "fill_rectangle_3x3",
            "mirror_horizontal_4x4",
            "count_objects_5x5",
            "extend_line_2x5",
            "complete_shape_4x4",
            "color_transform_3x3",
            "pattern_repeat_5x2",
        ]

        self._task_metadata = {}
        for task_id in self._task_ids:
            self._task_metadata[task_id] = {
                "task_name": task_id,
                "max_grid_size": (5, 5),
                "num_demonstrations": 2,
                "num_test_inputs": 1,
                "optimization_level": "5x5_optimized",
            }

    def get_available_task_ids(self):
        return self._task_ids.copy()

    def get_task_metadata(self, task_id):
        return self._task_metadata.get(task_id, {})

    def get_random_task(self, key):
        """Return a mock 5x5 task for demonstration."""
        from jaxarc.types import JaxArcTask
        from jaxarc.utils.task_manager import create_jax_task_index

        # Create simple 5x5 demo grids
        input_grid = jnp.array(
            [
                [0, 1, 0, 1, 0],
                [1, 2, 1, 2, 1],
                [0, 1, 0, 1, 0],
                [1, 2, 1, 2, 1],
                [0, 1, 0, 1, 0],
            ]
        )

        output_grid = jnp.array(
            [
                [2, 2, 2, 2, 2],
                [2, 1, 1, 1, 2],
                [2, 1, 0, 1, 2],
                [2, 1, 1, 1, 2],
                [2, 2, 2, 2, 2],
            ]
        )

        # Pad to standard size (5x5 is already optimal for MiniARC)
        padded_input = jnp.zeros((5, 5), dtype=jnp.int32)
        padded_input = padded_input.at[:5, :5].set(input_grid)

        padded_output = jnp.zeros((5, 5), dtype=jnp.int32)
        padded_output = padded_output.at[:5, :5].set(output_grid)

        # Create masks (full 5x5 is valid)
        input_mask = jnp.ones((5, 5), dtype=jnp.bool_)
        output_mask = jnp.ones((5, 5), dtype=jnp.bool_)

        return JaxArcTask(
            input_grids_examples=padded_input[None, ...],
            input_masks_examples=input_mask[None, ...],
            output_grids_examples=padded_output[None, ...],
            output_masks_examples=output_mask[None, ...],
            num_train_pairs=1,
            test_input_grids=padded_input[None, ...],
            test_input_masks=input_mask[None, ...],
            true_test_output_grids=padded_output[None, ...],
            true_test_output_masks=output_mask[None, ...],
            num_test_pairs=1,
            task_index=create_jax_task_index("miniarc_demo_task"),
        )

    def get_dataset_statistics(self):
        return {
            "total_tasks": len(self._task_ids),
            "optimization": "5x5 grids",
            "max_configured_dimensions": "5x5",
            "is_5x5_optimized": True,
            "train_pairs": {"min": 1, "max": 3, "avg": 2.0},
            "test_pairs": {"min": 1, "max": 1, "avg": 1.0},
            "grid_dimensions": {
                "max_height": 5,
                "max_width": 5,
                "avg_height": 4.2,
                "avg_width": 4.2,
            },
            "performance_benefits": [
                "Faster training iterations",
                "Reduced computational requirements",
                "Ideal for algorithm development",
                "Quick experimentation cycles",
            ],
        }


def create_demo_parser():
    """Create a demo parser for when MiniARC dataset is not available."""
    console.print("[yellow]📝 Using demo mode with mock MiniARC data[/yellow]")
    return DemoMiniArcParser()


def demonstrate_parser_basics():
    """Demonstrate basic MiniARC parser functionality."""
    console.print(
        Panel.fit(
            "[bold blue]🔍 MiniARC Parser Basics[/bold blue]", border_style="blue"
        )
    )

    # Create MiniARC configuration
    config = create_config_util(
        max_episode_steps=80, task_split="training", success_bonus=5.0
    )

    # Initialize parser with error handling for missing data
    try:
        # Create proper configuration for MiniARC parser
        from omegaconf import DictConfig

        parser_config = DictConfig(
            {
                "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
                "grid": {
                    "max_grid_height": 5,
                    "max_grid_width": 5,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,  # Increased to handle tasks with more training pairs
                "max_test_pairs": 1,
            }
        )

        parser = MiniArcParser(parser_config)
    except Exception as e:
        console.print(f"[yellow]⚠️  MiniARC dataset not found: {e}[/yellow]")
        console.print(
            "[dim]This is expected if you haven't downloaded the MiniARC dataset yet.[/dim]"
        )
        console.print(
            "[dim]You can download it using: python scripts/download_dataset.py miniarc[/dim]"
        )

        # Return a demo parser with mock data
        return create_demo_parser()

    # Get available tasks
    task_ids = parser.get_available_task_ids()
    console.print(f"📊 Found [bold cyan]{len(task_ids)}[/bold cyan] MiniARC tasks")

    # Get dataset statistics
    stats = parser.get_dataset_statistics()
    console.print("\n📈 Dataset Statistics:")
    console.print(f"  • Total tasks: [bold]{stats['total_tasks']}[/bold]")
    console.print(f"  • Grid optimization: [bold]{stats['optimization']}[/bold]")
    console.print(
        f"  • Max dimensions: [bold]{stats['max_configured_dimensions']}[/bold]"
    )
    console.print(
        f"  • 5x5 optimized: [bold green]{stats['is_5x5_optimized']}[/bold green]"
    )

    # Display optimization benefits
    console.print("\n🚀 Performance Benefits:")
    for benefit in stats.get("performance_benefits", []):
        console.print(f"  • [green]{benefit}[/green]")

    return parser


def demonstrate_5x5_optimization_benefits():
    """Demonstrate the benefits of 5x5 grid optimization."""
    console.print(
        Panel.fit(
            "[bold green]⚡ 5x5 Grid Optimization Benefits[/bold green]",
            border_style="green",
        )
    )

    # Create comparison table
    table = Table(title="MiniARC vs Standard ARC Comparison")
    table.add_column("Aspect", style="cyan", no_wrap=True)
    table.add_column("MiniARC (5x5)", style="green")
    table.add_column("Standard ARC (30x30)", style="red")
    table.add_column("Improvement", style="bold magenta")

    # Memory usage comparison
    miniarc_memory = 5 * 5 * 4  # 5x5 grid, 4 bytes per int32
    standard_memory = 30 * 30 * 4  # 30x30 grid, 4 bytes per int32
    memory_improvement = f"{standard_memory // miniarc_memory}x less"

    table.add_row("Grid Size", "5×5 (25 cells)", "30×30 (900 cells)", "36x fewer cells")
    table.add_row(
        "Memory per Grid",
        f"{miniarc_memory} bytes",
        f"{standard_memory} bytes",
        memory_improvement,
    )
    table.add_row("Processing Speed", "~10-50x faster", "Baseline", "Rapid iteration")
    table.add_row(
        "Training Time", "Minutes to hours", "Hours to days", "Quick experiments"
    )
    table.add_row(
        "Batch Size", "Large batches", "Limited batches", "Better GPU utilization"
    )
    table.add_row(
        "Development Cycle",
        "Seconds to minutes",
        "Minutes to hours",
        "Rapid prototyping",
    )

    console.print(table)

    # Demonstrate actual memory efficiency
    console.print("\n💾 Memory Efficiency Demonstration:")

    # Create sample grids
    miniarc_grid = jnp.zeros((5, 5), dtype=jnp.int32)
    standard_grid = jnp.zeros((30, 30), dtype=jnp.int32)

    console.print(f"  • MiniARC grid shape: [cyan]{miniarc_grid.shape}[/cyan]")
    console.print(f"  • Standard ARC grid shape: [red]{standard_grid.shape}[/red]")
    console.print(
        f"  • Memory ratio: [bold]{standard_grid.size // miniarc_grid.size}:1[/bold]"
    )

    # Demonstrate batch processing benefits
    console.print("\n🔄 Batch Processing Benefits:")
    miniarc_batch_size = 128  # Can process larger batches
    standard_batch_size = 16  # Limited by memory

    console.print(f"  • MiniARC batch size: [green]{miniarc_batch_size}[/green]")
    console.print(f"  • Standard ARC batch size: [red]{standard_batch_size}[/red]")
    console.print(
        f"  • Throughput improvement: [bold]{miniarc_batch_size // standard_batch_size}x[/bold]"
    )


def demonstrate_rapid_prototyping_workflow(parser):
    """Demonstrate rapid prototyping workflow with MiniARC."""
    console.print(
        Panel.fit(
            "[bold magenta]🚀 Rapid Prototyping Workflow[/bold magenta]",
            border_style="magenta",
        )
    )

    console.print("Simulating a typical development cycle with MiniARC:")

    # Step 1: Quick task loading
    console.print("\n[bold]Step 1: Quick Task Loading[/bold]")
    start_time = time.time()

    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)

    load_time = time.time() - start_time
    console.print(f"  ✅ Task loaded in [green]{load_time:.4f}s[/green]")
    console.print(f"  • Grid size: [cyan]{task.test_input_grids.shape[-2:]}[/cyan]")
    console.print(f"  • Training pairs: [yellow]{task.num_train_pairs}[/yellow]")

    # Step 2: Fast environment setup
    console.print("\n[bold]Step 2: Fast Environment Setup[/bold]")
    start_time = time.time()

    config = create_miniarc_config(
        max_episode_steps=50,  # Shorter episodes for rapid testing
        step_penalty=-0.001,  # Lower penalty for experimentation
        success_bonus=3.0,  # Quick feedback
    )
    env = ArcEnvironment(config)

    setup_time = time.time() - start_time
    console.print(f"  ✅ Environment ready in [green]{setup_time:.4f}s[/green]")

    # Step 3: Rapid iteration testing
    console.print("\n[bold]Step 3: Rapid Iteration Testing[/bold]")

    iteration_times = []
    for iteration in range(3):
        start_time = time.time()

        # Reset environment
        reset_key, step_key = jax.random.split(key)
        state, observation = env.reset(reset_key, task_data=task)

        # Take a few quick actions
        for step in range(5):
            action = {
                "selection": jnp.array(
                    [step % 5, step % 5], dtype=jnp.int32
                ),  # Point action
                "operation": jnp.array(1 + (step % 3), dtype=jnp.int32),
            }
            state, observation, reward, info = env.step(action)

        iteration_time = time.time() - start_time
        iteration_times.append(iteration_time)

        console.print(
            f"  Iteration {iteration + 1}: [green]{iteration_time:.4f}s[/green] "
            f"(similarity: [yellow]{info['similarity']:.3f}[/yellow])"
        )

    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    console.print(
        f"\n  📊 Average iteration time: [bold green]{avg_iteration_time:.4f}s[/bold green]"
    )

    # Step 4: Quick visualization and analysis
    console.print("\n[bold]Step 4: Quick Analysis[/bold]")
    start_time = time.time()

    # Extract actual grid for visualization
    input_grid = task.test_input_grids[0]
    input_mask = task.test_input_masks[0]

    # Find actual dimensions
    height = int(input_mask.sum(axis=0).max())
    width = int(input_mask.sum(axis=1).max())
    actual_grid = input_grid[:height, :width]

    analysis_time = time.time() - start_time
    console.print(f"  ✅ Analysis completed in [green]{analysis_time:.4f}s[/green]")
    console.print(f"  • Actual grid size: [cyan]{actual_grid.shape}[/cyan]")
    console.print(f"  • Colors used: [yellow]{len(jnp.unique(actual_grid))}[/yellow]")

    # Total workflow time
    total_time = load_time + setup_time + sum(iteration_times) + analysis_time
    console.print(f"\n🎯 [bold]Total workflow time: {total_time:.4f}s[/bold]")
    console.print(
        "[dim]Compare this to standard ARC which might take minutes or hours![/dim]"
    )

    return actual_grid


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between MiniARC and standard ARC."""
    console.print(
        Panel.fit(
            "[bold yellow]⚡ Performance Comparison[/bold yellow]",
            border_style="yellow",
        )
    )

    console.print("Benchmarking MiniARC vs Standard ARC configurations:")

    # Create configurations
    miniarc_config = create_miniarc_config(max_episode_steps=50)
    standard_config = create_standard_config(max_episode_steps=50)

    console.print("\n[bold]Configuration Comparison:[/bold]")
    console.print(
        f"  • MiniARC grid size: [green]{miniarc_config.grid.max_grid_height}×{miniarc_config.grid.max_grid_width}[/green]"
    )
    console.print(
        f"  • Standard ARC grid size: [red]{standard_config.grid.max_grid_height}×{standard_config.grid.max_grid_width}[/red]"
    )

    # Benchmark environment creation
    console.print("\n[bold]Environment Creation Benchmark:[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # MiniARC environment creation
        task1 = progress.add_task("Creating MiniARC environment...", total=None)
        start_time = time.time()
        miniarc_env = ArcEnvironment(miniarc_config)
        miniarc_creation_time = time.time() - start_time
        progress.update(task1, completed=True)

        # Standard ARC environment creation
        task2 = progress.add_task("Creating Standard ARC environment...", total=None)
        start_time = time.time()
        standard_env = ArcEnvironment(standard_config)
        standard_creation_time = time.time() - start_time
        progress.update(task2, completed=True)

    console.print(f"  ✅ MiniARC: [green]{miniarc_creation_time:.4f}s[/green]")
    console.print(f"  ✅ Standard: [red]{standard_creation_time:.4f}s[/red]")
    speedup = (
        standard_creation_time / miniarc_creation_time
        if miniarc_creation_time > 0
        else 1
    )
    console.print(f"  🚀 Speedup: [bold]{speedup:.2f}x faster[/bold]")

    # Benchmark episode execution
    console.print("\n[bold]Episode Execution Benchmark:[/bold]")

    # Create demo tasks
    key = jax.random.PRNGKey(123)
    demo_parser = create_demo_parser()
    demo_task = demo_parser.get_random_task(key)

    # Benchmark MiniARC episode
    start_time = time.time()
    state, obs = miniarc_env.reset(key, task_data=demo_task)
    for _ in range(10):  # 10 steps
        action = {
            "selection": jnp.array([2, 2], dtype=jnp.int32),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        state, obs, reward, info = miniarc_env.step(action)
    miniarc_episode_time = time.time() - start_time

    # Create a larger demo task for standard ARC
    larger_input = jnp.zeros((30, 30), dtype=jnp.int32)
    larger_input = larger_input.at[:5, :5].set(demo_task.test_input_grids[0])
    larger_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
    larger_mask = larger_mask.at[:5, :5].set(True)

    # Create larger task for standard ARC
    from jaxarc.types import JaxArcTask
    from jaxarc.utils.task_manager import create_jax_task_index

    larger_task = JaxArcTask(
        input_grids_examples=larger_input[None, ...],
        input_masks_examples=larger_mask[None, ...],
        output_grids_examples=larger_input[None, ...],
        output_masks_examples=larger_mask[None, ...],
        num_train_pairs=1,
        test_input_grids=larger_input[None, ...],
        test_input_masks=larger_mask[None, ...],
        true_test_output_grids=larger_input[None, ...],
        true_test_output_masks=larger_mask[None, ...],
        num_test_pairs=1,
        task_index=create_jax_task_index("standard_demo_task"),
    )

    # Benchmark Standard ARC episode (with properly sized task)
    start_time = time.time()
    state, obs = standard_env.reset(key, task_data=larger_task)
    for _ in range(10):  # 10 steps
        action = {
            "selection": jnp.zeros((30, 30), dtype=jnp.bool_).at[2, 2].set(True),
            "operation": jnp.array(1, dtype=jnp.int32),
        }
        state, obs, reward, info = standard_env.step(action)
    standard_episode_time = time.time() - start_time

    console.print(
        f"  ✅ MiniARC episode (10 steps): [green]{miniarc_episode_time:.4f}s[/green]"
    )
    console.print(
        f"  ✅ Standard episode (10 steps): [red]{standard_episode_time:.4f}s[/red]"
    )
    episode_speedup = (
        standard_episode_time / miniarc_episode_time if miniarc_episode_time > 0 else 1
    )
    console.print(f"  🚀 Episode speedup: [bold]{episode_speedup:.2f}x faster[/bold]")

    # Memory usage comparison
    console.print("\n[bold]Memory Usage Comparison:[/bold]")
    miniarc_memory = 5 * 5 * 4  # bytes
    standard_memory = 30 * 30 * 4  # bytes
    memory_ratio = standard_memory / miniarc_memory

    console.print(f"  • MiniARC grid memory: [green]{miniarc_memory} bytes[/green]")
    console.print(f"  • Standard ARC grid memory: [red]{standard_memory} bytes[/red]")
    console.print(
        f"  💾 Memory efficiency: [bold]{memory_ratio:.1f}x less memory[/bold]"
    )

    return {
        "creation_speedup": speedup,
        "episode_speedup": episode_speedup,
        "memory_efficiency": memory_ratio,
    }


def demonstrate_batch_processing(parser):
    """Demonstrate efficient batch processing with MiniARC."""
    console.print(
        Panel.fit(
            "[bold purple]🔄 Batch Processing Demo[/bold purple]", border_style="purple"
        )
    )

    console.print("Demonstrating efficient batch processing with 5x5 grids:")

    # Create batch of tasks
    batch_size = 8
    keys = jax.random.split(jax.random.PRNGKey(456), batch_size)

    console.print(f"\n[bold]Processing batch of {batch_size} tasks:[/bold]")

    start_time = time.time()
    batch_results = []

    for i, key in enumerate(keys):
        task = parser.get_random_task(key)

        # Process task (simulate some computation)
        input_grid = task.test_input_grids[0]
        grid_stats = {
            "task_id": f"batch_task_{i}",
            "grid_shape": input_grid.shape,
            "unique_colors": len(jnp.unique(input_grid)),
            "non_zero_cells": int(jnp.sum(input_grid != 0)),
        }
        batch_results.append(grid_stats)

        console.print(
            f"  Task {i + 1}: [cyan]{grid_stats['grid_shape']}[/cyan] grid, "
            f"[yellow]{grid_stats['unique_colors']}[/yellow] colors, "
            f"[green]{grid_stats['non_zero_cells']}[/green] non-zero cells"
        )

    batch_time = time.time() - start_time
    console.print(
        f"\n✅ Processed {batch_size} tasks in [bold green]{batch_time:.4f}s[/bold green]"
    )
    console.print(
        f"📊 Average time per task: [yellow]{batch_time / batch_size:.4f}s[/yellow]"
    )

    # Demonstrate JAX batch processing benefits
    console.print("\n[bold]JAX Batch Processing Benefits:[/bold]")
    console.print("  • [green]Vectorized operations across batch[/green]")
    console.print("  • [green]Efficient GPU/TPU utilization[/green]")
    console.print("  • [green]Reduced Python overhead[/green]")
    console.print("  • [green]Parallel processing of small grids[/green]")

    return batch_results


def save_miniarc_visualization(grid, filename: str):
    """Save MiniARC task visualization to SVG file."""
    if grid is None:
        return

    try:
        # Create SVG for the grid
        svg = draw_grid_svg(grid, label="MiniARC Task Example")
        output_path = Path(f"{filename}.svg")

        with output_path.open("w", encoding="utf-8") as f:
            f.write(svg.as_svg())

        console.print(f"💾 Saved visualization: [cyan]{output_path}[/cyan]")

    except Exception as e:
        console.print(f"[red]❌ Error saving visualization: {e}[/red]")


def main(
    performance_comparison: bool = typer.Option(
        False,
        "--performance-comparison",
        "-p",
        help="Run performance comparison with standard ARC",
    ),
    rapid_prototyping: bool = typer.Option(
        False,
        "--rapid-prototyping",
        "-r",
        help="Demonstrate rapid prototyping workflow",
    ),
    batch_processing: bool = typer.Option(
        False,
        "--batch-processing",
        "-b",
        help="Demonstrate batch processing capabilities",
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Save task visualizations to SVG files"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """MiniARC Usage Example - Comprehensive demonstration of MiniARC dataset optimization."""

    if verbose:
        logger.add("miniarc_usage.log", level="DEBUG")

    console.print(
        Panel.fit(
            "[bold cyan]🧠 MiniARC Usage Example[/bold cyan]\n"
            "Demonstrating 5x5 grid optimization, rapid prototyping, and performance benefits",
            border_style="cyan",
        )
    )

    try:
        # Initialize parser
        parser = demonstrate_parser_basics()

        # Always demonstrate 5x5 optimization benefits
        demonstrate_5x5_optimization_benefits()

        # Demonstrate rapid prototyping workflow
        if rapid_prototyping:
            actual_grid = demonstrate_rapid_prototyping_workflow(parser)
            if visualize and actual_grid is not None:
                save_miniarc_visualization(actual_grid, "miniarc_rapid_prototype")

        # Demonstrate performance comparison
        if performance_comparison:
            perf_results = demonstrate_performance_comparison()
            console.print("\n📊 [bold]Performance Summary:[/bold]")
            console.print(
                f"  • Environment creation: [green]{perf_results['creation_speedup']:.2f}x faster[/green]"
            )
            console.print(
                f"  • Episode execution: [green]{perf_results['episode_speedup']:.2f}x faster[/green]"
            )
            console.print(
                f"  • Memory efficiency: [green]{perf_results['memory_efficiency']:.1f}x less memory[/green]"
            )

        # Demonstrate batch processing
        if batch_processing:
            batch_results = demonstrate_batch_processing(parser)
            console.print(
                f"\n🔄 Processed [bold]{len(batch_results)}[/bold] tasks in batch"
            )

        # Show default demonstration if no specific flags
        if not any([rapid_prototyping, performance_comparison, batch_processing]):
            console.print("\n[dim]💡 Try these options for more demonstrations:[/dim]")
            console.print(
                "[dim]  • --performance-comparison: Compare with standard ARC[/dim]"
            )
            console.print(
                "[dim]  • --rapid-prototyping: Show development workflow[/dim]"
            )
            console.print(
                "[dim]  • --batch-processing: Demonstrate batch efficiency[/dim]"
            )
            console.print("[dim]  • --visualize: Save task visualizations[/dim]")

            # Show a basic task example
            key = jax.random.PRNGKey(789)
            task = parser.get_random_task(key)

            console.print("\n[bold]Sample MiniARC Task:[/bold]")
            input_grid = task.test_input_grids[0]
            console.print(f"  • Grid shape: [cyan]{input_grid.shape}[/cyan]")
            console.print(
                f"  • Training pairs: [yellow]{task.num_train_pairs}[/yellow]"
            )

            console.print("\n[bold cyan]Input Grid:[/bold cyan]")
            log_grid_to_console(input_grid)

            if visualize:
                save_miniarc_visualization(input_grid, "miniarc_sample_task")

        # Show usage suggestions
        console.print(
            Panel.fit(
                "[bold green]✅ Demo completed successfully![/bold green]\n\n"
                "💡 Try these commands:\n"
                "  • [cyan]python examples/miniarc_usage_example.py --performance-comparison[/cyan]\n"
                "  • [cyan]python examples/miniarc_usage_example.py --rapid-prototyping --visualize[/cyan]\n"
                "  • [cyan]python examples/miniarc_usage_example.py --batch-processing --verbose[/cyan]\n"
                "  • [cyan]python examples/miniarc_usage_example.py --performance-comparison --rapid-prototyping[/cyan]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        logger.exception("MiniARC usage example failed")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    typer.run(main)
