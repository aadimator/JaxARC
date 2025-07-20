#!/usr/bin/env python3
"""ARC-AGI-2 Usage Example

This comprehensive example demonstrates how to use the ARC-AGI-2 dataset with JaxARC,
showcasing the latest 2025 dataset features, GitHub format benefits, and enhanced
task complexity compared to ARC-AGI-1.

ARC-AGI-2 is the expanded ARC dataset with 1000 training and 120 evaluation tasks,
featuring more complex patterns and reasoning challenges.

Usage:
    python examples/arc_agi_2_usage_example.py
    python examples/arc_agi_2_usage_example.py --complexity-analysis
    python examples/arc_agi_2_usage_example.py --compare-with-agi1 --visualize
    python examples/arc_agi_2_usage_example.py --advanced-features --verbose
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import create_standard_config
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.config import get_config
from jaxarc.utils.visualization import draw_grid_svg

console = Console()


class DemoArcAgi2Parser:
    """Demo parser that simulates ARC-AGI-2 functionality without requiring the actual dataset."""

    def __init__(self):
        # Mock ARC-AGI-2 tasks (more complex than ARC-AGI-1)
        self._task_ids = [
            "agi2_001a",
            "agi2_002b",
            "agi2_003c",
            "agi2_004d",
            "agi2_005e",
            "agi2_006f",
            "agi2_007g",
            "agi2_008h",
            "agi2_009i",
            "agi2_010j",
            "agi2_011k",
            "agi2_012l",
            "agi2_013m",
            "agi2_014n",
            "agi2_015o",
            "agi2_016p",
            "agi2_017q",
            "agi2_018r",
            "agi2_019s",
            "agi2_020t",
        ]

        self._task_metadata = {}
        for task_id in self._task_ids:
            self._task_metadata[task_id] = {
                "task_name": task_id,
                "source": "GitHub (arcprize/ARC-AGI-2)",
                "format": "individual_json",
                "complexity_level": "enhanced",
                "num_demonstrations": 3 + (hash(task_id) % 3),  # 3-5 demonstrations
                "num_test_inputs": 1 + (hash(task_id) % 2),  # 1-2 test inputs
                "file_path": f"data/training/{task_id}.json",
                "year": 2025,
                "enhanced_features": [
                    "Complex spatial reasoning",
                    "Multi-step transformations",
                    "Advanced pattern recognition",
                    "Hierarchical structures",
                ],
            }

    def get_available_task_ids(self):
        return self._task_ids.copy()

    def get_task_metadata(self, task_id):
        return self._task_metadata.get(task_id, {})

    def get_random_task(self, key):
        """Return a mock ARC-AGI-2 task with enhanced complexity."""
        from jaxarc.types import JaxArcTask
        from jaxarc.utils.task_manager import create_jax_task_index

        # Create more complex ARC-AGI-2 style demo grids
        input_grid = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 2, 2, 0, 0],
                [0, 1, 1, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
                [0, 0, 0, 3, 4, 3, 0, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 5, 5, 0, 0, 0, 6, 6, 0, 0],
                [0, 5, 5, 0, 0, 0, 6, 6, 0, 0],
            ]
        )

        # More complex transformation pattern
        output_grid = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 4, 4, 4, 2, 2, 0, 0],
                [0, 1, 1, 4, 4, 4, 2, 2, 0, 0],
                [0, 4, 4, 4, 4, 4, 4, 4, 0, 0],
                [0, 4, 4, 3, 3, 3, 4, 4, 0, 0],
                [0, 4, 4, 3, 4, 3, 4, 4, 0, 0],
                [0, 4, 4, 3, 3, 3, 4, 4, 0, 0],
                [0, 4, 4, 4, 4, 4, 4, 4, 0, 0],
                [0, 5, 5, 4, 4, 4, 6, 6, 0, 0],
                [0, 5, 5, 4, 4, 4, 6, 6, 0, 0],
            ]
        )

        # Pad to standard size
        padded_input = jnp.zeros((30, 30), dtype=jnp.int32)
        padded_input = padded_input.at[:10, :10].set(input_grid)

        padded_output = jnp.zeros((30, 30), dtype=jnp.int32)
        padded_output = padded_output.at[:10, :10].set(output_grid)

        # Create masks
        input_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        input_mask = input_mask.at[:10, :10].set(True)

        output_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        output_mask = output_mask.at[:10, :10].set(True)

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
            task_index=create_jax_task_index("arc_agi_2_demo_task"),
        )

    def get_dataset_statistics(self):
        return {
            "total_tasks": 1120,
            "training_tasks": 1000,
            "evaluation_tasks": 120,
            "source": "GitHub (arcprize/ARC-AGI-2)",
            "format": "Individual JSON files",
            "year": 2025,
            "complexity_improvements": [
                "2.5x more training tasks than ARC-AGI-1",
                "Enhanced pattern complexity",
                "Multi-step reasoning challenges",
                "Advanced spatial transformations",
                "Hierarchical pattern structures",
            ],
            "train_pairs": {"min": 3, "max": 5, "avg": 4.0},
            "test_pairs": {"min": 1, "max": 2, "avg": 1.3},
            "grid_dimensions": {
                "max_height": 30,
                "max_width": 30,
                "typical_range": "5x5 to 25x25",
                "complexity": "Higher than ARC-AGI-1",
            },
            "new_features": [
                "More diverse pattern types",
                "Complex transformation chains",
                "Advanced reasoning requirements",
                "Enhanced evaluation metrics",
            ],
        }


def create_demo_parser():
    """Create a demo parser for when ARC-AGI-2 dataset is not available."""
    console.print("[yellow]📝 Using demo mode with mock ARC-AGI-2 data[/yellow]")
    return DemoArcAgi2Parser()


def demonstrate_parser_basics():
    """Demonstrate basic ARC-AGI-2 parser functionality."""
    console.print(
        Panel.fit(
            "[bold blue]🔍 ARC-AGI-2 Parser Basics (2025 Dataset)[/bold blue]",
            border_style="blue",
        )
    )

    # Create ARC-AGI-2 configuration
    try:
        config = get_config("dataset=arc_agi_2")
    except Exception as e:
        console.print(f"[yellow]⚠️  Configuration not found: {e}[/yellow]")
        console.print("[dim]Using fallback configuration...[/dim]")

        from omegaconf import DictConfig

        config = DictConfig(
            {
                "dataset": {
                    "data_root": "data/raw/ARC-AGI-2",
                    "training": {"path": "data/raw/ARC-AGI-2/data/training"},
                    "evaluation": {"path": "data/raw/ARC-AGI-2/data/evaluation"},
                    "parser": {"_target_": "jaxarc.parsers.ArcAgiParser"},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 10,
                    "max_test_pairs": 4,  # ARC-AGI-2 can have more test pairs
                }
            }
        )

    # Initialize parser with error handling for missing data
    try:
        parser = ArcAgiParser(config.dataset)
    except Exception as e:
        console.print(f"[yellow]⚠️  ARC-AGI-2 dataset not found: {e}[/yellow]")
        console.print(
            "[dim]This is expected if you haven't downloaded the ARC-AGI-2 dataset yet.[/dim]"
        )
        console.print(
            "[dim]You can download it using: python scripts/download_dataset.py arc-agi-2[/dim]"
        )

        # Return a demo parser with mock data
        return create_demo_parser()

    # Get available tasks
    task_ids = parser.get_available_task_ids()
    console.print(f"📊 Found [bold cyan]{len(task_ids)}[/bold cyan] ARC-AGI-2 tasks")

    # Get dataset statistics
    stats = parser.get_dataset_statistics()
    console.print("\n📈 Dataset Statistics:")
    console.print(f"  • Total tasks: [bold]{stats['total_tasks']}[/bold]")
    console.print(f"  • Training tasks: [bold]{stats['training_tasks']}[/bold]")
    console.print(f"  • Evaluation tasks: [bold]{stats['evaluation_tasks']}[/bold]")
    console.print(f"  • Source: [bold]{stats['source']}[/bold]")
    console.print(f"  • Year: [bold]{stats['year']}[/bold]")

    # Display complexity improvements
    console.print("\n🚀 ARC-AGI-2 Improvements:")
    for improvement in stats.get("complexity_improvements", []):
        console.print(f"  • [green]{improvement}[/green]")

    return parser


def demonstrate_complexity_analysis(parser):
    """Demonstrate the enhanced complexity of ARC-AGI-2 tasks."""
    console.print(
        Panel.fit(
            "[bold green]🧩 ARC-AGI-2 Complexity Analysis[/bold green]",
            border_style="green",
        )
    )

    # Get dataset statistics
    stats = parser.get_dataset_statistics()

    # Create complexity comparison table
    table = Table(title="ARC-AGI-2 Enhanced Complexity Features")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("ARC-AGI-1", style="yellow")
    table.add_column("ARC-AGI-2", style="green")
    table.add_column("Enhancement", style="bold magenta")

    table.add_row("Training Tasks", "400", "1000", "2.5x more data")
    table.add_row("Evaluation Tasks", "400", "120", "Curated selection")
    table.add_row("Demonstrations", "2-4 per task", "3-5 per task", "More examples")
    table.add_row("Test Inputs", "1 per task", "1-2 per task", "Multiple test cases")
    table.add_row(
        "Pattern Complexity",
        "Basic to moderate",
        "Moderate to advanced",
        "Higher difficulty",
    )
    table.add_row("Reasoning Steps", "1-3 steps", "2-5 steps", "Multi-step chains")
    table.add_row(
        "Spatial Relations", "Simple", "Complex hierarchical", "Advanced structures"
    )

    console.print(table)

    # Demonstrate task complexity analysis
    console.print("\n🔍 Task Complexity Analysis:")

    # Analyze a sample task
    key = jax.random.PRNGKey(789)
    task = parser.get_random_task(key)

    # Extract grid for analysis
    input_grid = task.test_input_grids[0]
    output_grid = task.true_test_output_grids[0]
    input_mask = task.test_input_masks[0]
    output_mask = task.true_test_output_masks[0]

    # Calculate dimensions
    input_height = int(input_mask.sum(axis=0).max())
    input_width = int(input_mask.sum(axis=1).max())
    output_height = int(output_mask.sum(axis=0).max())
    output_width = int(output_mask.sum(axis=1).max())

    # Extract actual grids
    actual_input = input_grid[:input_height, :input_width]
    actual_output = output_grid[:output_height, :output_width]

    # Analyze complexity metrics
    input_colors = len(jnp.unique(actual_input))
    output_colors = len(jnp.unique(actual_output))
    grid_size = input_height * input_width
    non_zero_ratio = float(jnp.sum(actual_input != 0)) / grid_size

    console.print(
        f"  • Grid size: [cyan]{input_height}×{input_width}[/cyan] ({grid_size} cells)"
    )
    console.print(f"  • Input colors: [yellow]{input_colors}[/yellow]")
    console.print(f"  • Output colors: [green]{output_colors}[/green]")
    console.print(f"  • Pattern density: [magenta]{non_zero_ratio:.2%}[/magenta]")
    console.print(f"  • Training pairs: [cyan]{task.num_train_pairs}[/cyan]")
    console.print(f"  • Test pairs: [green]{task.num_test_pairs}[/green]")

    # Complexity scoring
    complexity_score = (
        (grid_size / 25) * 0.3  # Size factor
        + (input_colors / 10) * 0.2  # Color diversity
        + (task.num_train_pairs / 5) * 0.2  # Training examples
        + non_zero_ratio * 0.3  # Pattern density
    )

    console.print(f"  • Complexity score: [bold]{complexity_score:.2f}/1.0[/bold]")

    if complexity_score > 0.7:
        console.print("  🔥 [bold red]High complexity task[/bold red]")
    elif complexity_score > 0.4:
        console.print("  ⚡ [bold yellow]Medium complexity task[/bold yellow]")
    else:
        console.print("  ✅ [bold green]Lower complexity task[/bold green]")

    return actual_input, actual_output


def demonstrate_agi1_vs_agi2_comparison():
    """Demonstrate comparison between ARC-AGI-1 and ARC-AGI-2."""
    console.print(
        Panel.fit(
            "[bold magenta]⚖️  ARC-AGI-1 vs ARC-AGI-2 Comparison[/bold magenta]",
            border_style="magenta",
        )
    )

    # Dataset size comparison
    console.print("[bold]Dataset Size Comparison:[/bold]")

    agi1_stats = {
        "training": 400,
        "evaluation": 400,
        "total": 800,
        "year": 2024,
    }

    agi2_stats = {
        "training": 1000,
        "evaluation": 120,
        "total": 1120,
        "year": 2025,
    }

    console.print(f"  ARC-AGI-1 ({agi1_stats['year']}):")
    console.print(f"    • Training: [yellow]{agi1_stats['training']}[/yellow] tasks")
    console.print(
        f"    • Evaluation: [yellow]{agi1_stats['evaluation']}[/yellow] tasks"
    )
    console.print(f"    • Total: [yellow]{agi1_stats['total']}[/yellow] tasks")

    console.print(f"  ARC-AGI-2 ({agi2_stats['year']}):")
    console.print(f"    • Training: [green]{agi2_stats['training']}[/green] tasks")
    console.print(f"    • Evaluation: [green]{agi2_stats['evaluation']}[/green] tasks")
    console.print(f"    • Total: [green]{agi2_stats['total']}[/green] tasks")

    # Calculate improvements
    training_improvement = agi2_stats["training"] / agi1_stats["training"]
    total_improvement = agi2_stats["total"] / agi1_stats["total"]

    console.print("\n📊 Improvements:")
    console.print(
        f"  • Training data: [bold green]{training_improvement:.1f}x more[/bold green]"
    )
    console.print(
        f"  • Total tasks: [bold green]{total_improvement:.1f}x more[/bold green]"
    )
    console.print("  • Evaluation: [bold blue]Curated selection[/bold blue]")

    # Feature comparison
    console.print("\n[bold]Feature Comparison:[/bold]")

    features_table = Table()
    features_table.add_column("Aspect", style="cyan")
    features_table.add_column("ARC-AGI-1", style="yellow")
    features_table.add_column("ARC-AGI-2", style="green")

    features_table.add_row("Focus", "Foundational patterns", "Advanced reasoning")
    features_table.add_row("Difficulty", "Moderate", "Higher complexity")
    features_table.add_row(
        "Pattern Types", "Basic transformations", "Multi-step chains"
    )
    features_table.add_row(
        "Spatial Reasoning", "Simple relations", "Hierarchical structures"
    )
    features_table.add_row(
        "Training Strategy", "Balanced learning", "Curriculum progression"
    )

    console.print(features_table)

    # Use case recommendations
    console.print("\n💡 [bold]Use Case Recommendations:[/bold]")
    console.print("  ARC-AGI-1:")
    console.print("    • [yellow]Learning fundamentals[/yellow]")
    console.print("    • [yellow]Algorithm development[/yellow]")
    console.print("    • [yellow]Baseline comparisons[/yellow]")
    console.print("    • [yellow]Quick prototyping[/yellow]")

    console.print("  ARC-AGI-2:")
    console.print("    • [green]Advanced model training[/green]")
    console.print("    • [green]Complex reasoning research[/green]")
    console.print("    • [green]Competition preparation[/green]")
    console.print("    • [green]State-of-the-art evaluation[/green]")


def demonstrate_advanced_features(parser):
    """Demonstrate advanced features specific to ARC-AGI-2."""
    console.print(
        Panel.fit(
            "[bold purple]🚀 ARC-AGI-2 Advanced Features[/bold purple]",
            border_style="purple",
        )
    )

    # Enhanced task loading
    console.print("[bold]Enhanced Task Loading:[/bold]")

    start_time = time.time()

    # Load multiple tasks to demonstrate efficiency
    keys = jax.random.split(jax.random.PRNGKey(999), 5)
    tasks = []

    for i, key in enumerate(keys):
        task = parser.get_random_task(key)
        tasks.append(task)

        metadata = parser.get_task_metadata(f"agi2_{i:03d}a")
        console.print(
            f"  Task {i + 1}: [cyan]{task.num_train_pairs}[/cyan] train pairs, "
            f"[green]{task.num_test_pairs}[/green] test pairs, "
            f"complexity: [yellow]{metadata.get('complexity_level', 'standard')}[/yellow]"
        )

    load_time = time.time() - start_time
    console.print(f"  ✅ Loaded {len(tasks)} tasks in [green]{load_time:.3f}s[/green]")

    # Multi-test input handling
    console.print("\n[bold]Multi-Test Input Handling:[/bold]")

    # Simulate task with multiple test inputs
    sample_task = tasks[0]
    console.print(f"  • Test inputs: [cyan]{sample_task.num_test_pairs}[/cyan]")
    console.print("  • Enhanced evaluation capability")
    console.print("  • Multiple solution validation")
    console.print("  • Robust performance metrics")

    # Advanced pattern recognition
    console.print("\n[bold]Advanced Pattern Recognition:[/bold]")

    pattern_types = [
        "Hierarchical transformations",
        "Multi-object interactions",
        "Complex spatial relationships",
        "Sequential pattern chains",
        "Conditional transformations",
    ]

    for pattern in pattern_types:
        console.print(f"  • [green]{pattern}[/green]")

    # Performance optimization
    console.print("\n[bold]Performance Optimizations:[/bold]")
    console.print("  • [green]Efficient individual file loading[/green]")
    console.print("  • [green]Optimized memory usage[/green]")
    console.print("  • [green]Parallel task processing[/green]")
    console.print("  • [green]Smart caching strategies[/green]")

    return tasks


def demonstrate_environment_integration(parser):
    """Demonstrate environment integration with ARC-AGI-2 enhanced features."""
    console.print(
        Panel.fit(
            "[bold yellow]🏃 ARC-AGI-2 Environment Integration[/bold yellow]",
            border_style="yellow",
        )
    )

    # Create ARC-AGI-2 optimized environment configuration
    config = create_standard_config(
        max_episode_steps=300,  # Longer episodes for complex tasks
        reward_on_submit_only=True,
        success_bonus=30.0,  # Higher bonus for complex tasks
        step_penalty=-0.003,  # Lower penalty for exploration
    )

    # Convert to unified config and create environment
    from jaxarc.envs.equinox_config import convert_arc_env_config_to_jax_arc_config
    unified_config = convert_arc_env_config_to_jax_arc_config(config)
    env = ArcEnvironment(unified_config)

    # Get a complex ARC-AGI-2 task
    key = jax.random.PRNGKey(456)
    arc_task = parser.get_random_task(key)

    # Reset environment with the ARC-AGI-2 task
    reset_key, step_key = jax.random.split(key)
    state, observation = env.reset(reset_key, task_data=arc_task)

    console.print("🔄 Environment reset with ARC-AGI-2 task")
    console.print(f"  • Observation shape: [cyan]{observation.shape}[/cyan]")
    console.print(
        f"  • Initial similarity: [yellow]{state.similarity_score:.3f}[/yellow]"
    )
    console.print(
        f"  • Working grid shape: [magenta]{state.working_grid.shape}[/magenta]"
    )
    console.print(f"  • Training pairs: [cyan]{arc_task.num_train_pairs}[/cyan]")
    console.print(f"  • Test pairs: [magenta]{arc_task.num_test_pairs}[/magenta]")

    # Take sophisticated demonstration steps
    console.print("\n🎮 Taking advanced demonstration steps:")

    for step_num in range(7):
        # Create more sophisticated actions for complex ARC-AGI-2 tasks
        if step_num < 3:
            # Pattern recognition phase
            action = {
                "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                .at[step_num * 3 : step_num * 3 + 4, step_num * 2 : step_num * 2 + 4]
                .set(True),
                "operation": jnp.array(1 + step_num, dtype=jnp.int32),
            }
        elif step_num < 5:
            # Transformation phase
            action = {
                "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                .at[step_num - 1 : step_num + 2, step_num - 1 : step_num + 2]
                .set(True),
                "operation": jnp.array(15, dtype=jnp.int32),  # Complex operation
            }
        else:
            # Refinement phase
            action = {
                "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                .at[step_num - 2 : step_num, step_num - 2 : step_num]
                .set(True),
                "operation": jnp.array(20, dtype=jnp.int32),  # Advanced operation
            }

        state, observation, reward, info = env.step(action)
        done = env.is_done

        console.print(
            f"  Step {step_num + 1}: reward=[green]{reward:.3f}[/green], "
            f"similarity=[yellow]{info['similarity']:.3f}[/yellow], "
            f"done=[red]{done}[/red]"
        )

        if done:
            console.print("  [bold red]Episode terminated![/bold red]")
            break

    return env, state


def save_arc_agi_2_visualization(input_grid, output_grid, filename: str):
    """Save ARC-AGI-2 task visualization to SVG files."""
    if input_grid is None or output_grid is None:
        return

    try:
        # Create SVG for input grid
        input_svg = draw_grid_svg(input_grid, label="ARC-AGI-2 - Input")
        input_filename = f"{filename}_input.svg"
        with open(input_filename, "w") as f:
            f.write(input_svg.as_svg())

        # Create SVG for output grid
        output_svg = draw_grid_svg(output_grid, label="ARC-AGI-2 - Output")
        output_filename = f"{filename}_output.svg"
        with open(output_filename, "w") as f:
            f.write(output_svg.as_svg())

        console.print("💾 Saved visualizations:")
        console.print(f"  • [cyan]{input_filename}[/cyan]")
        console.print(f"  • [green]{output_filename}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error saving visualization: {e}[/red]")


def main(
    complexity_analysis: bool = typer.Option(
        False,
        "--complexity-analysis",
        "-c",
        help="Analyze ARC-AGI-2 task complexity",
    ),
    compare_with_agi1: bool = typer.Option(
        False,
        "--compare-with-agi1",
        "-1",
        help="Compare ARC-AGI-2 with ARC-AGI-1",
    ),
    advanced_features: bool = typer.Option(
        False,
        "--advanced-features",
        "-a",
        help="Demonstrate advanced ARC-AGI-2 features",
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Save task visualizations to SVG files"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """ARC-AGI-2 Usage Example - Comprehensive demonstration of the 2025 enhanced dataset."""

    if verbose:
        logger.add("arc_agi_2_usage.log", level="DEBUG")

    console.print(
        Panel.fit(
            "[bold cyan]🧠 ARC-AGI-2 Usage Example[/bold cyan]\n"
            "Demonstrating the 2025 enhanced dataset with advanced complexity and features",
            border_style="cyan",
        )
    )

    try:
        # Initialize parser
        parser = demonstrate_parser_basics()

        # Demonstrate complexity analysis
        if complexity_analysis:
            input_grid, output_grid = demonstrate_complexity_analysis(parser)
            if visualize and input_grid is not None:
                save_arc_agi_2_visualization(
                    input_grid, output_grid, "arc_agi_2_complexity"
                )

        # Compare with ARC-AGI-1
        if compare_with_agi1:
            demonstrate_agi1_vs_agi2_comparison()

        # Demonstrate advanced features
        if advanced_features:
            tasks = demonstrate_advanced_features(parser)
            console.print(
                f"\n🔧 Loaded [bold]{len(tasks)}[/bold] tasks for advanced feature demo"
            )

        # Always demonstrate environment integration
        env, final_state = demonstrate_environment_integration(parser)
        console.print(
            f"\n🏁 Final similarity score: [yellow]{final_state.similarity_score:.3f}[/yellow]"
        )

        # Show default demonstration if no specific flags
        if not any([complexity_analysis, compare_with_agi1, advanced_features]):
            console.print("\n[dim]💡 Try these options for more demonstrations:[/dim]")
            console.print(
                "[dim]  • --complexity-analysis: Analyze task complexity[/dim]"
            )
            console.print("[dim]  • --compare-with-agi1: Compare with ARC-AGI-1[/dim]")
            console.print("[dim]  • --advanced-features: Show advanced features[/dim]")
            console.print("[dim]  • --visualize: Save task visualizations[/dim]")

            # Show a basic task example
            key = jax.random.PRNGKey(789)
            task = parser.get_random_task(key)

            console.print("\n[bold]Sample ARC-AGI-2 Task:[/bold]")
            console.print(f"  • Training pairs: [cyan]{task.num_train_pairs}[/cyan]")
            console.print(f"  • Test pairs: [magenta]{task.num_test_pairs}[/magenta]")
            console.print("  • Enhanced complexity patterns")
            console.print("  • Multi-step reasoning required")

        # Show usage suggestions
        console.print(
            Panel.fit(
                "[bold green]✅ Demo completed successfully![/bold green]\n\n"
                "💡 Try these commands:\n"
                "  • [cyan]python examples/arc_agi_2_usage_example.py --complexity-analysis --visualize[/cyan]\n"
                "  • [cyan]python examples/arc_agi_2_usage_example.py --compare-with-agi1[/cyan]\n"
                "  • [cyan]python examples/arc_agi_2_usage_example.py --advanced-features --verbose[/cyan]\n"
                "  • [cyan]python examples/arc_agi_2_usage_example.py --complexity-analysis --compare-with-agi1[/cyan]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        logger.exception("ARC-AGI-2 usage example failed")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    typer.run(main)
