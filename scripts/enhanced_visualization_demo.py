"""Enhanced demonstration of JaxARC visualization features.

This script demonstrates the updated visualization functions with new options
for colored numbers vs blocks and double-width blocks for better aesthetics.
"""

from __future__ import annotations

import jax.numpy as jnp
from rich.console import Console

from jaxarc.utils.visualization import (
    draw_grid_svg,
    log_grid_to_console,
    save_svg_drawing,
    visualize_grid_rich,
    visualize_task_pair_rich,
)


def create_sample_grid() -> jnp.ndarray:
    """Create a sample grid for demonstration."""
    return jnp.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1],
        [2, 3, 4, 5],
    ])

def create_sample_grid_with_mask() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a sample grid with mask for demonstration."""
    grid = jnp.array([
        [0, 1, 2, 3, -1, -1],
        [4, 5, 6, 7, -1, -1],
        [8, 9, 0, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ])
    mask = jnp.array([
        [True, True, True, True, False, False],
        [True, True, True, True, False, False],
        [True, True, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ])
    return grid, mask

def demo_single_grid_visualization():
    """Demonstrate single grid visualization with different options."""
    console = Console()
    
    console.print("\n[bold blue]═══ Single Grid Visualization Demo ═══[/bold blue]\n")
    
    grid = create_sample_grid()
    
    # 1. Default visualization (double-width blocks)
    console.print("[bold]1. Default: Double-width colored blocks[/bold]")
    table = visualize_grid_rich(grid, title="Sample Grid", double_width=True, show_numbers=False)
    console.print(table)
    
    # 2. Single-width blocks
    console.print("\n[bold]2. Single-width colored blocks[/bold]")
    table = visualize_grid_rich(grid, title="Sample Grid", double_width=False, show_numbers=False)
    console.print(table)
    
    # 3. Colored numbers
    console.print("\n[bold]3. Colored numbers[/bold]")
    table = visualize_grid_rich(grid, title="Sample Grid", show_numbers=True)
    console.print(table)
    
    # 4. With coordinates
    console.print("\n[bold]4. Double-width blocks with coordinates[/bold]")
    table = visualize_grid_rich(grid, title="Sample Grid", show_coordinates=True, double_width=True)
    console.print(table)

def demo_masked_grid_visualization():
    """Demonstrate grid visualization with masking."""
    console = Console()
    
    console.print("\n[bold blue]═══ Masked Grid Visualization Demo ═══[/bold blue]\n")
    
    grid, mask = create_sample_grid_with_mask()
    
    # 1. Masked grid with double-width blocks
    console.print("[bold]1. Masked grid: Double-width blocks[/bold]")
    table = visualize_grid_rich(grid, mask, title="Masked Grid", double_width=True, show_numbers=False)
    console.print(table)
    
    # 2. Masked grid with numbers
    console.print("\n[bold]2. Masked grid: Colored numbers[/bold]")
    table = visualize_grid_rich(grid, mask, title="Masked Grid", show_numbers=True)
    console.print(table)

def demo_task_pair_visualization():
    """Demonstrate task pair visualization."""
    console = Console()
    
    console.print("\n[bold blue]═══ Task Pair Visualization Demo ═══[/bold blue]\n")
    
    input_grid = create_sample_grid()
    output_grid = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 1, 2],
        [3, 4, 5, 6],
    ])
    
    # 1. Task pair with double-width blocks
    console.print("[bold]1. Task pair: Double-width blocks[/bold]")
    input_table, output_table = visualize_task_pair_rich(
        input_grid, output_grid, title="Example", double_width=True, show_numbers=False
    )
    console.print(input_table)
    console.print(output_table)
    
    # 2. Task pair with numbers
    console.print("\n[bold]2. Task pair: Colored numbers[/bold]")
    input_table, output_table = visualize_task_pair_rich(
        input_grid, output_grid, title="Example", show_numbers=True
    )
    console.print(input_table)
    console.print(output_table)

def demo_log_function():
    """Demonstrate the log_grid_to_console function."""
    console = Console()
    
    console.print("\n[bold blue]═══ Log Function Demo ═══[/bold blue]\n")
    
    grid = create_sample_grid()
    
    console.print("[bold]Using log_grid_to_console with different options:[/bold]")
    
    # Double-width blocks
    console.print("\n[dim]Double-width blocks:[/dim]")
    log_grid_to_console(grid, title="Logged Grid", double_width=True, show_numbers=False)
    
    # Colored numbers
    console.print("\n[dim]Colored numbers:[/dim]")
    log_grid_to_console(grid, title="Logged Grid", show_numbers=True)

def demo_svg_generation():
    """Demonstrate SVG generation."""
    console = Console()
    
    console.print("\n[bold blue]═══ SVG Generation Demo ═══[/bold blue]\n")
    
    grid = create_sample_grid()
    
    # Generate SVG
    svg_drawing = draw_grid_svg(grid, label="Sample Grid")
    
    # Ensure we have a proper Drawing object (not a tuple)
    if isinstance(svg_drawing, tuple):
        # This shouldn't happen since as_group=False by default
        console.print("[red]✗[/red] Unexpected tuple result from draw_grid_svg")
        return
    
    # Save to file
    output_path = "sample_grid_enhanced.svg"
    save_svg_drawing(svg_drawing, output_path)
    
    console.print(f"[green]✓[/green] Generated SVG: {output_path}")

def main():
    """Run all demonstration functions."""
    console = Console()
    
    console.print("[bold cyan]JaxARC Enhanced Visualization Demo[/bold cyan]")
    console.print("This demo shows the new visualization options including:")
    console.print("• Double-width blocks for better square appearance")
    console.print("• Colored numbers as an alternative to blocks")
    console.print("• Improved aesthetics for terminal display")
    
    try:
        demo_single_grid_visualization()
        demo_masked_grid_visualization()
        demo_task_pair_visualization()
        demo_log_function()
        demo_svg_generation()
        
        console.print("\n[bold green]✓ All demos completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Demo failed: {e}[/bold red]")
        raise

if __name__ == "__main__":
    main()
