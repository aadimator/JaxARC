#!/usr/bin/env python3
"""ConceptARC Parser Demo

This script demonstrates how to use the ConceptArcParser to work with
the ConceptARC dataset, which organizes ARC tasks into 16 concept groups
for systematic evaluation of abstraction and generalization abilities.

Usage:
    python examples/concept_arc_demo.py
    python examples/concept_arc_demo.py --concept Center
    python examples/concept_arc_demo.py --stats
"""

from __future__ import annotations

import jax
import typer
from loguru import logger
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from jaxarc.parsers import ConceptArcParser
from jaxarc.utils.visualization import log_grid_to_console

console = Console()


def create_concept_arc_config() -> DictConfig:
    """Create a ConceptARC configuration."""
    return DictConfig(
        {
            "corpus": {
                "path": "data/raw/ConceptARC/corpus",
                "concept_groups": [
                    "AboveBelow",
                    "Center",
                    "CleanUp",
                    "CompleteShape",
                    "Copy",
                    "Count",
                    "ExtendToBoundary",
                    "ExtractObjects",
                    "FilledNotFilled",
                    "HorizontalVertical",
                    "InsideOutside",
                    "MoveToBoundary",
                    "Order",
                    "SameDifferent",
                    "TopBottom2D",
                    "TopBottom3D",
                ],
            },
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 4,
            "max_test_pairs": 3,
        }
    )


def demonstrate_basic_usage():
    """Demonstrate basic ConceptARC parser usage."""
    console.print("\n[bold blue]🔍 ConceptARC Parser Basic Usage[/bold blue]")

    # Create parser
    config = create_concept_arc_config()
    parser = ConceptArcParser(config)

    # Get available concept groups
    concept_groups = parser.get_concept_groups()
    console.print(f"📊 Found {len(concept_groups)} concept groups:")
    for concept in concept_groups:
        tasks = parser.get_tasks_in_concept(concept)
        console.print(f"  • {concept}: {len(tasks)} tasks")

    # Get a random task
    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)

    console.print("\n🎯 Random task loaded:")
    console.print(f"  • Training pairs: {task.num_train_pairs}")
    console.print(f"  • Test pairs: {task.num_test_pairs}")
    console.print(f"  • Task index: {task.task_index}")


def demonstrate_concept_specific_usage(concept: str):
    """Demonstrate concept-specific task sampling."""
    console.print(f"\n[bold green]🎯 Concept-Specific Usage: {concept}[/bold green]")

    # Create parser
    config = create_concept_arc_config()
    parser = ConceptArcParser(config)

    try:
        # Get tasks from specific concept
        concept_tasks = parser.get_tasks_in_concept(concept)
        console.print(f"📋 {concept} concept has {len(concept_tasks)} tasks:")

        # Show first few task IDs
        for i, task_id in enumerate(concept_tasks[:5]):
            metadata = parser.get_task_metadata(task_id)
            console.print(
                f"  {i + 1}. {task_id} ({metadata['num_demonstrations']} demos, {metadata['num_test_inputs']} tests)"
            )

        if len(concept_tasks) > 5:
            console.print(f"  ... and {len(concept_tasks) - 5} more tasks")

        # Get a random task from this concept
        key = jax.random.PRNGKey(123)
        task = parser.get_random_task_from_concept(concept, key)

        console.print(f"\n🎲 Random task from {concept}:")
        console.print(f"  • Training pairs: {task.num_train_pairs}")
        console.print(f"  • Test pairs: {task.num_test_pairs}")

        # Show first training example
        if task.num_train_pairs > 0:
            console.print("\n📝 First training example:")
            input_grid = task.input_grids_examples[0]
            output_grid = task.output_grids_examples[0]
            input_mask = task.input_masks_examples[0]
            output_mask = task.output_masks_examples[0]

            # Get actual grid dimensions from mask
            input_height = int(input_mask.sum(axis=0).max())
            input_width = int(input_mask.sum(axis=1).max())
            output_height = int(output_mask.sum(axis=0).max())
            output_width = int(output_mask.sum(axis=1).max())

            console.print("Input:")
            log_grid_to_console(input_grid[:input_height, :input_width])
            console.print("Output:")
            log_grid_to_console(output_grid[:output_height, :output_width])

    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        available_concepts = parser.get_concept_groups()
        console.print(f"Available concepts: {', '.join(available_concepts)}")


def show_dataset_statistics():
    """Show comprehensive dataset statistics."""
    console.print("\n[bold magenta]📊 ConceptARC Dataset Statistics[/bold magenta]")

    # Create parser
    config = create_concept_arc_config()
    parser = ConceptArcParser(config)

    # Get comprehensive statistics
    stats = parser.get_dataset_statistics()

    console.print(f"🎯 Total tasks: {stats['total_tasks']}")
    console.print(f"🏷️  Total concept groups: {stats['total_concept_groups']}")

    # Create a detailed table
    table = Table(title="Concept Group Statistics")
    table.add_column("Concept Group", style="cyan", no_wrap=True)
    table.add_column("Tasks", justify="right", style="magenta")
    table.add_column("Avg Demos", justify="right", style="green")
    table.add_column("Avg Tests", justify="right", style="yellow")
    table.add_column("Demo Range", justify="center", style="blue")
    table.add_column("Test Range", justify="center", style="red")

    for concept, concept_stats in stats["concept_groups"].items():
        demo_range = f"{concept_stats.get('min_demonstrations', 'N/A')}-{concept_stats.get('max_demonstrations', 'N/A')}"
        test_range = f"{concept_stats.get('min_test_inputs', 'N/A')}-{concept_stats.get('max_test_inputs', 'N/A')}"

        table.add_row(
            concept,
            str(concept_stats["num_tasks"]),
            f"{concept_stats.get('avg_demonstrations', 0):.1f}",
            f"{concept_stats.get('avg_test_inputs', 0):.1f}",
            demo_range,
            test_range,
        )

    console.print(table)


def main(
    concept: str = typer.Option(
        None, "--concept", "-c", help="Specific concept group to demonstrate"
    ),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show dataset statistics"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """ConceptARC Parser Demonstration Script."""

    if verbose:
        logger.add("concept_arc_demo.log", level="DEBUG")

    console.print("[bold cyan]🧠 ConceptARC Parser Demo[/bold cyan]")
    console.print("Demonstrating concept-based ARC task organization and sampling\n")

    try:
        if stats:
            show_dataset_statistics()
        elif concept:
            demonstrate_concept_specific_usage(concept)
        else:
            demonstrate_basic_usage()

        console.print("\n[bold green]✅ Demo completed successfully![/bold green]")
        console.print("\n💡 Try these commands:")
        console.print("  • python examples/concept_arc_demo.py --concept Center")
        console.print("  • python examples/concept_arc_demo.py --stats")
        console.print(
            "  • python examples/concept_arc_demo.py --concept Copy --verbose"
        )

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        logger.exception("Demo failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
