"""Demo showing full task visualization with different display options.

This script demonstrates the full task visualization functionality with
the new display options for colored numbers vs blocks and improved aesthetics.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from rich.console import Console

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.utils.visualization import visualize_parsed_task_data_rich


def create_synthetic_task_data() -> dict:
    """Create a synthetic task for demonstration."""
    return {
        "demo_task": {
            "train": [
                {
                    "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                    "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                },
                {
                    "input": [[2, 3, 4], [5, 6, 7], [8, 9, 0]],
                    "output": [[3, 4, 5], [6, 7, 8], [9, 0, 1]],
                },
            ],
            "test": [
                {
                    "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "output": [[2, 3, 4], [5, 6, 7], [8, 9, 0]],
                }
            ],
        }
    }


def load_sample_task() -> dict:
    """Load a sample task from the ARC dataset or create synthetic one."""
    data_path = (
        Path(__file__).parent
        / "data"
        / "raw"
        / "arc-prize-2024"
        / "arc-agi_training_challenges.json"
    )

    if not data_path.exists():
        # Create a synthetic task if no data available
        return create_synthetic_task_data()

    try:
        with data_path.open() as f:
            data = json.load(f)

        # Get the first task
        first_task_id = next(iter(data.keys()))
        return {first_task_id: data[first_task_id]}
    except Exception:
        # Fall back to synthetic data if there's any issue
        return create_synthetic_task_data()


def demo_full_task_visualization():
    """Demonstrate full task visualization with different display options."""
    console = Console()

    console.print("\n[bold blue]═══ Full Task Visualization Demo ═══[/bold blue]\n")

    # Load sample task
    task_data_raw = load_sample_task()

    # Create a minimal config for the parser
    from omegaconf import DictConfig

    minimal_cfg = DictConfig(
        {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "max_train_pairs": 5,
            "max_test_pairs": 5,
            "training": {"challenges": "", "solutions": ""},
        }
    )

    # Parse the task - we'll use preprocess_task_data directly
    # since we have the raw data
    try:
        parser = ArcAgiParser(cfg=minimal_cfg)

        # Use preprocess_task_data directly with our raw data
        key = jax.random.PRNGKey(42)
        task_data = parser.preprocess_task_data(task_data_raw, key)

        console.print("[bold]Task Data Summary:[/bold]")
        console.print(f"• Task ID: {task_data.task_id}")
        console.print(f"• Training pairs: {task_data.num_train_pairs}")
        console.print(f"• Test pairs: {task_data.num_test_pairs}")

        # 1. Visualize with double-width blocks
        console.print("\n[bold]1. Full task visualization: Double-width blocks[/bold]")
        console.print("─" * 60)
        visualize_parsed_task_data_rich(
            task_data,
            show_test=True,
            show_coordinates=False,
            show_numbers=False,
            double_width=True,
        )

        # 2. Visualize with colored numbers
        console.print("\n[bold]2. Full task visualization: Colored numbers[/bold]")
        console.print("─" * 60)
        visualize_parsed_task_data_rich(
            task_data,
            show_test=True,
            show_coordinates=False,
            show_numbers=True,
            double_width=False,
        )

        # 3. Visualize with coordinates and numbers
        console.print("\n[bold]3. Full task with coordinates and numbers[/bold]")
        console.print("─" * 60)
        visualize_parsed_task_data_rich(
            task_data,
            show_test=True,
            show_coordinates=True,
            show_numbers=True,
            double_width=False,
        )

    except Exception as e:
        console.print(f"[red]Error parsing task: {e}[/red]")
        # Show stack trace for debugging
        import traceback

        console.print(traceback.format_exc())


def demo_comparison():
    """Show side-by-side comparison of different display modes."""
    console = Console()

    console.print("\n[bold blue]═══ Display Mode Comparison ═══[/bold blue]\n")

    # Create a simple example grid
    grid = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
        ]
    )

    from jaxarc.utils.visualization import visualize_grid_rich

    console.print("[bold]Comparing different display modes for the same grid:[/bold]")

    # Show all modes
    modes = [
        ("Single-width blocks", False, False),
        ("Double-width blocks", False, True),
        ("Colored numbers", True, False),
    ]

    for name, show_numbers, double_width in modes:
        console.print(f"\n[dim]{name}:[/dim]")
        table = visualize_grid_rich(
            grid,
            title=name,
            show_numbers=show_numbers,
            double_width=double_width,
        )
        console.print(table)

    console.print("\n[bold]Key observations:[/bold]")
    console.print("• Double-width blocks (██) create a more square appearance")
    console.print("• Single-width blocks (█) are more compact")
    console.print("• Colored numbers show exact values while maintaining color coding")
    console.print("• All modes support masking and coordinate display")


def main():
    """Run the full task visualization demo."""
    console = Console()

    console.print("[bold cyan]JaxARC Full Task Visualization Demo[/bold cyan]")
    console.print("This demo shows the complete task visualization functionality")
    console.print("with improved aesthetics and display options.")

    try:
        demo_comparison()
        demo_full_task_visualization()

        console.print(
            "\n[bold green]✓ Full task visualization demo completed![/bold green]"
        )
        console.print("\nThe visualization functions now support:")
        console.print("• Better square appearance with double-width blocks")
        console.print("• Colored numbers as an alternative to blocks")
        console.print("• Full task display with multiple examples")
        console.print("• Consistent styling across all visualization functions")

    except Exception as e:
        console.print(f"\n[bold red]✗ Demo failed: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
