#!/usr/bin/env python3
"""ConceptARC Usage Example

This comprehensive example demonstrates how to use the ConceptARC dataset with JaxARC,
including concept group exploration, task sampling, environment integration, and
visualization of concept-based tasks.

ConceptARC is a benchmark dataset organized around 16 concept groups with 10 tasks each,
designed to systematically assess abstraction and generalization abilities.

Usage:
    python examples/conceptarc_usage_example.py
    python examples/conceptarc_usage_example.py --concept Center --visualize
    python examples/conceptarc_usage_example.py --interactive
    python examples/conceptarc_usage_example.py --run-episode --concept Copy
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jaxarc.envs import ArcEnvironment, arc_reset, arc_step
from jaxarc.envs.factory import create_conceptarc_config
from jaxarc.parsers import ConceptArcParser
from jaxarc.utils.config import create_conceptarc_config as create_config_util
from jaxarc.utils.visualization import draw_grid_svg, log_grid_to_console

console = Console()


class DemoConceptArcParser:
    """Demo parser that simulates ConceptARC functionality without requiring the actual dataset."""
    
    def __init__(self):
        # Mock concept groups and tasks
        self._concept_groups = {
            "Center": ["Center/task1", "Center/task2", "Center/task3"],
            "Copy": ["Copy/task1", "Copy/task2", "Copy/task3"],
            "CleanUp": ["CleanUp/task1", "CleanUp/task2", "CleanUp/task3"],
            "Count": ["Count/task1", "Count/task2", "Count/task3"],
        }
        
        self._task_metadata = {}
        for concept, tasks in self._concept_groups.items():
            for task_id in tasks:
                self._task_metadata[task_id] = {
                    "concept_group": concept,
                    "task_name": task_id.split("/")[1],
                    "num_demonstrations": 2,
                    "num_test_inputs": 1,
                }
    
    def get_concept_groups(self):
        return list(self._concept_groups.keys())
    
    def get_tasks_in_concept(self, concept):
        return self._concept_groups.get(concept, [])
    
    def get_task_metadata(self, task_id):
        return self._task_metadata.get(task_id, {})
    
    def get_random_task_from_concept(self, concept, key):
        """Return a mock task for demonstration."""
        from jaxarc.types import JaxArcTask
        from jaxarc.utils.task_manager import create_jax_task_index
        
        # Create simple demo grids
        input_grid = jnp.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
        output_grid = jnp.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]])
        
        # Pad to standard size
        padded_input = jnp.zeros((30, 30), dtype=jnp.int32)
        padded_input = padded_input.at[:3, :3].set(input_grid)
        
        padded_output = jnp.zeros((30, 30), dtype=jnp.int32)
        padded_output = padded_output.at[:3, :3].set(output_grid)
        
        # Create masks
        input_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        input_mask = input_mask.at[:3, :3].set(True)
        
        output_mask = jnp.zeros((30, 30), dtype=jnp.bool_)
        output_mask = output_mask.at[:3, :3].set(True)
        
        return JaxArcTask(
            input_grids_examples=padded_input[None, ...],  # Add batch dimension
            input_masks_examples=input_mask[None, ...],
            output_grids_examples=padded_output[None, ...],
            output_masks_examples=output_mask[None, ...],
            num_train_pairs=1,
            test_input_grids=padded_input[None, ...],
            test_input_masks=input_mask[None, ...],
            true_test_output_grids=padded_output[None, ...],
            true_test_output_masks=output_mask[None, ...],
            num_test_pairs=1,
            task_index=create_jax_task_index(f"{concept}_demo_task"),
        )
    
    def get_dataset_statistics(self):
        return {
            "total_tasks": sum(len(tasks) for tasks in self._concept_groups.values()),
            "total_concept_groups": len(self._concept_groups),
            "concept_groups": {
                concept: {
                    "num_tasks": len(tasks),
                    "tasks": tasks,
                    "avg_demonstrations": 2.0,
                    "avg_test_inputs": 1.0,
                }
                for concept, tasks in self._concept_groups.items()
            }
        }


def create_demo_parser():
    """Create a demo parser for when ConceptARC dataset is not available."""
    console.print("[yellow]📝 Using demo mode with mock ConceptARC data[/yellow]")
    return DemoConceptArcParser()


def demonstrate_parser_basics():
    """Demonstrate basic ConceptARC parser functionality."""
    console.print(Panel.fit(
        "[bold blue]🔍 ConceptARC Parser Basics[/bold blue]",
        border_style="blue"
    ))

    # Create ConceptARC configuration
    config = create_config_util(
        max_episode_steps=100,
        task_split="corpus",
        success_bonus=15.0
    )
    
    # Initialize parser with error handling for missing data
    try:
        # Create proper configuration for ConceptARC parser
        from omegaconf import DictConfig
        parser_config = DictConfig({
            "corpus": {
                "path": "data/raw/ConceptARC/corpus",
                "concept_groups": [
                    "AboveBelow", "Center", "CleanUp", "CompleteShape", "Copy", "Count",
                    "ExtendToBoundary", "ExtractObjects", "FilledNotFilled", "HorizontalVertical",
                    "InsideOutside", "MoveToBoundary", "Order", "SameDifferent", "TopBottom2D", "TopBottom3D"
                ]
            },
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0
            },
            "max_train_pairs": 4,
            "max_test_pairs": 3
        })
        
        parser = ConceptArcParser(parser_config)
    except Exception as e:
        console.print(f"[yellow]⚠️  ConceptARC dataset not found: {e}[/yellow]")
        console.print("[dim]This is expected if you haven't downloaded the ConceptARC dataset yet.[/dim]")
        console.print("[dim]You can download it using: python scripts/download_kaggle_dataset.py download-conceptarc[/dim]")
        
        # Return a demo parser with mock data
        return create_demo_parser()
    
    # Get available concept groups
    concept_groups = parser.get_concept_groups()
    console.print(f"📊 Found [bold cyan]{len(concept_groups)}[/bold cyan] concept groups:")
    
    # Display concept groups in a table
    table = Table(title="ConceptARC Concept Groups")
    table.add_column("Concept Group", style="cyan", no_wrap=True)
    table.add_column("Tasks", justify="right", style="magenta")
    table.add_column("Description", style="green")
    
    concept_descriptions = {
        "AboveBelow": "Spatial relationships above/below",
        "Center": "Central positioning and symmetry",
        "CleanUp": "Removing noise and artifacts",
        "CompleteShape": "Shape completion tasks",
        "Copy": "Pattern copying and replication",
        "Count": "Counting and enumeration",
        "ExtendToBoundary": "Extension to boundaries",
        "ExtractObjects": "Object extraction and isolation",
        "FilledNotFilled": "Filled vs unfilled patterns",
        "HorizontalVertical": "Horizontal/vertical orientations",
        "InsideOutside": "Inside/outside relationships",
        "MoveToBoundary": "Movement to boundaries",
        "Order": "Ordering and sequencing",
        "SameDifferent": "Same/different comparisons",
        "TopBottom2D": "2D top/bottom relationships",
        "TopBottom3D": "3D top/bottom relationships"
    }
    
    for concept in sorted(concept_groups):
        tasks = parser.get_tasks_in_concept(concept)
        description = concept_descriptions.get(concept, "Concept-based reasoning")
        table.add_row(concept, str(len(tasks)), description)
    
    console.print(table)
    
    # Get dataset statistics
    stats = parser.get_dataset_statistics()
    console.print(f"\n📈 Dataset Statistics:")
    console.print(f"  • Total tasks: [bold]{stats['total_tasks']}[/bold]")
    console.print(f"  • Total concept groups: [bold]{stats['total_concept_groups']}[/bold]")
    
    return parser


def demonstrate_concept_exploration(parser: ConceptArcParser, concept: str):
    """Demonstrate concept-specific task exploration."""
    console.print(Panel.fit(
        f"[bold green]🎯 Exploring Concept: {concept}[/bold green]",
        border_style="green"
    ))
    
    try:
        # Get tasks from specific concept
        concept_tasks = parser.get_tasks_in_concept(concept)
        console.print(f"📋 [bold]{concept}[/bold] concept contains [cyan]{len(concept_tasks)}[/cyan] tasks:")
        
        # Show task metadata
        for i, task_id in enumerate(concept_tasks[:3]):  # Show first 3 tasks
            metadata = parser.get_task_metadata(task_id)
            console.print(
                f"  {i + 1}. [yellow]{task_id}[/yellow] "
                f"([cyan]{metadata['num_demonstrations']}[/cyan] demos, "
                f"[magenta]{metadata['num_test_inputs']}[/magenta] tests)"
            )
        
        if len(concept_tasks) > 3:
            console.print(f"  ... and [dim]{len(concept_tasks) - 3}[/dim] more tasks")
        
        # Sample a random task from this concept
        key = jax.random.PRNGKey(42)
        task = parser.get_random_task_from_concept(concept, key)
        
        console.print(f"\n🎲 Random task from [bold]{concept}[/bold]:")
        console.print(f"  • Training pairs: [cyan]{task.num_train_pairs}[/cyan]")
        console.print(f"  • Test pairs: [magenta]{task.num_test_pairs}[/magenta]")
        console.print(f"  • Task index: [yellow]{task.task_index}[/yellow]")
        
        return task
        
    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        available_concepts = parser.get_concept_groups()
        console.print(f"Available concepts: [dim]{', '.join(available_concepts)}[/dim]")
        return None


def demonstrate_task_visualization(task, concept: str):
    """Demonstrate task visualization capabilities."""
    if task is None:
        return
        
    console.print(Panel.fit(
        f"[bold magenta]🎨 Visualizing {concept} Task[/bold magenta]",
        border_style="magenta"
    ))
    
    # Show first training example
    if task.num_train_pairs > 0:
        console.print("\n📝 [bold]First Training Example:[/bold]")
        
        # Get the actual grid data (remove padding)
        input_grid = task.input_grids_examples[0]
        output_grid = task.output_grids_examples[0]
        input_mask = task.input_masks_examples[0]
        output_mask = task.output_masks_examples[0]
        
        # Calculate actual dimensions from masks
        input_height = int(input_mask.sum(axis=0).max())
        input_width = int(input_mask.sum(axis=1).max())
        output_height = int(output_mask.sum(axis=0).max())
        output_width = int(output_mask.sum(axis=1).max())
        
        # Extract actual grids
        actual_input = input_grid[:input_height, :input_width]
        actual_output = output_grid[:output_height, :output_width]
        
        console.print("\n[bold cyan]Input Grid:[/bold cyan]")
        log_grid_to_console(actual_input)
        
        console.print("\n[bold green]Output Grid:[/bold green]")
        log_grid_to_console(actual_output)
        
        # Show grid dimensions and colors
        input_colors = jnp.unique(actual_input)
        output_colors = jnp.unique(actual_output)
        
        console.print(f"\n📐 Grid Dimensions:")
        console.print(f"  • Input: [cyan]{input_height}×{input_width}[/cyan]")
        console.print(f"  • Output: [green]{output_height}×{output_width}[/green]")
        
        console.print(f"\n🎨 Colors Used:")
        console.print(f"  • Input: [cyan]{list(input_colors)}[/cyan]")
        console.print(f"  • Output: [green]{list(output_colors)}[/green]")
        
        return actual_input, actual_output
    
    return None, None


def demonstrate_environment_integration(parser: ConceptArcParser, concept: str):
    """Demonstrate integration with ARC environment."""
    console.print(Panel.fit(
        f"[bold yellow]🏃 Environment Integration: {concept}[/bold yellow]",
        border_style="yellow"
    ))
    
    # Create ConceptARC-optimized environment configuration
    config = create_conceptarc_config(
        max_episode_steps=150,
        task_split="corpus",
        reward_on_submit_only=True,
        success_bonus=20.0,
        step_penalty=-0.01
    )
    
    # Create environment with ConceptARC parser
    env = ArcEnvironment(config)
    
    # Get a task from the specific concept
    key = jax.random.PRNGKey(123)
    concept_task = parser.get_random_task_from_concept(concept, key)
    
    # Reset environment with the concept-specific task
    reset_key, step_key = jax.random.split(key)
    state, observation = env.reset(reset_key, task_data=concept_task)
    
    console.print(f"🔄 Environment reset with [bold]{concept}[/bold] task")
    console.print(f"  • Observation shape: [cyan]{observation.shape}[/cyan]")
    console.print(f"  • Initial similarity: [yellow]{state.similarity_score:.3f}[/yellow]")
    console.print(f"  • Working grid shape: [magenta]{state.working_grid.shape}[/magenta]")
    
    # Take a few demonstration steps
    console.print(f"\n🎮 Taking demonstration steps:")
    
    for step_num in range(3):
        # Create a simple action (fill a small region with color 1)
        action = {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
                        .at[step_num:step_num+2, step_num:step_num+2]
                        .set(True),
            "operation": jnp.array(1 + step_num, dtype=jnp.int32)  # Different colors
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


def demonstrate_batch_processing(parser: ConceptArcParser):
    """Demonstrate batch processing of ConceptARC tasks."""
    console.print(Panel.fit(
        "[bold purple]⚡ Batch Processing Demo[/bold purple]",
        border_style="purple"
    ))
    
    # Create configuration for batch processing
    config = create_conceptarc_config(max_episode_steps=50)
    
    # Sample multiple tasks from different concepts
    concept_groups = parser.get_concept_groups()[:4]  # Use first 4 concepts
    
    console.print(f"🔄 Processing tasks from [cyan]{len(concept_groups)}[/cyan] concepts:")
    
    def process_concept_task(key, concept):
        """Process a single task from a concept."""
        task = parser.get_random_task_from_concept(concept, key)
        # Simple processing: return task statistics
        return {
            "concept": concept,
            "train_pairs": task.num_train_pairs,
            "test_pairs": task.num_test_pairs,
            "task_index": task.task_index
        }
    
    # Use JAX for batch processing
    keys = jax.random.split(jax.random.PRNGKey(456), len(concept_groups))
    
    results = []
    for key, concept in zip(keys, concept_groups):
        result = process_concept_task(key, concept)
        results.append(result)
        console.print(f"  • [yellow]{concept}[/yellow]: {result['train_pairs']} train, {result['test_pairs']} test")
    
    console.print(f"\n✅ Processed [bold]{len(results)}[/bold] tasks successfully")
    return results


def run_interactive_exploration(parser: ConceptArcParser):
    """Run interactive concept exploration."""
    console.print(Panel.fit(
        "[bold cyan]🔍 Interactive ConceptARC Explorer[/bold cyan]",
        border_style="cyan"
    ))
    
    concept_groups = parser.get_concept_groups()
    
    while True:
        console.print(f"\n📋 Available concepts: [dim]{', '.join(concept_groups)}[/dim]")
        
        concept = typer.prompt("Enter concept name (or 'quit' to exit)")
        
        if concept.lower() in ['quit', 'exit', 'q']:
            console.print("👋 Goodbye!")
            break
        
        if concept not in concept_groups:
            console.print(f"[red]❌ Unknown concept: {concept}[/red]")
            continue
        
        # Explore the selected concept
        task = demonstrate_concept_exploration(parser, concept)
        if task:
            demonstrate_task_visualization(task, concept)
        
        continue_exploring = typer.confirm("Continue exploring?")
        if not continue_exploring:
            break


def save_task_visualization(input_grid, output_grid, concept: str, filename: str):
    """Save task visualization to SVG file."""
    if input_grid is None or output_grid is None:
        return
    
    try:
        # Create SVG for input grid
        input_svg = draw_grid_svg(input_grid, label=f"{concept} - Input")
        input_filename = f"{filename}_input.svg"
        with open(input_filename, 'w') as f:
            f.write(input_svg.as_svg())
        
        # Create SVG for output grid
        output_svg = draw_grid_svg(output_grid, label=f"{concept} - Output")
        output_filename = f"{filename}_output.svg"
        with open(output_filename, 'w') as f:
            f.write(output_svg.as_svg())
        
        console.print(f"💾 Saved visualizations:")
        console.print(f"  • [cyan]{input_filename}[/cyan]")
        console.print(f"  • [green]{output_filename}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error saving visualization: {e}[/red]")


def main(
    concept: str = typer.Option(
        None, "--concept", "-c", 
        help="Specific concept group to demonstrate"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", 
        help="Run interactive exploration mode"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", 
        help="Save task visualizations to SVG files"
    ),
    run_episode: bool = typer.Option(
        False, "--run-episode", "-e", 
        help="Run a complete environment episode"
    ),
    batch_demo: bool = typer.Option(
        False, "--batch", "-b", 
        help="Demonstrate batch processing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", 
        help="Enable verbose logging"
    ),
):
    """ConceptARC Usage Example - Comprehensive demonstration of ConceptARC dataset integration."""
    
    if verbose:
        logger.add("conceptarc_usage.log", level="DEBUG")
    
    console.print(Panel.fit(
        "[bold cyan]🧠 ConceptARC Usage Example[/bold cyan]\n"
        "Demonstrating concept-based ARC task organization, sampling, and environment integration",
        border_style="cyan"
    ))
    
    try:
        # Initialize parser
        parser = demonstrate_parser_basics()
        
        if interactive:
            run_interactive_exploration(parser)
            return
        
        # Select concept for demonstration
        if concept is None:
            available_concepts = parser.get_concept_groups()
            concept = available_concepts[2]  # Use "CleanUp" as default
            console.print(f"\n💡 Using default concept: [bold]{concept}[/bold]")
        
        # Demonstrate concept exploration
        task = demonstrate_concept_exploration(parser, concept)
        
        if task:
            # Demonstrate task visualization
            input_grid, output_grid = demonstrate_task_visualization(task, concept)
            
            # Save visualizations if requested
            if visualize and input_grid is not None:
                save_task_visualization(
                    input_grid, output_grid, concept, 
                    f"conceptarc_{concept.lower()}_example"
                )
            
            # Demonstrate environment integration
            if run_episode:
                env, final_state = demonstrate_environment_integration(parser, concept)
                console.print(f"\n🏁 Final similarity score: [yellow]{final_state.similarity_score:.3f}[/yellow]")
        
        # Demonstrate batch processing
        if batch_demo:
            demonstrate_batch_processing(parser)
        
        # Show usage suggestions
        console.print(Panel.fit(
            "[bold green]✅ Demo completed successfully![/bold green]\n\n"
            "💡 Try these commands:\n"
            "  • [cyan]python examples/conceptarc_usage_example.py --concept Center --visualize[/cyan]\n"
            "  • [cyan]python examples/conceptarc_usage_example.py --interactive[/cyan]\n"
            "  • [cyan]python examples/conceptarc_usage_example.py --run-episode --concept Copy[/cyan]\n"
            "  • [cyan]python examples/conceptarc_usage_example.py --batch --verbose[/cyan]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        logger.exception("ConceptARC usage example failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)