"""
Enhanced Visualization Demo for JaxARC.

This example demonstrates the new enhanced visualization capabilities including:
- Grid type support in all visualization functions
- Improved RL step visualization with better selection overlays
- Centralized operation names utility
- Different action formats and their visualizations
"""

from __future__ import annotations

import tempfile

import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from rich.console import Console

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.types import Grid
from jaxarc.utils.config import get_config
from jaxarc.envs.operations import (
    get_operation_display_text,
    get_operation_name,
    get_operations_by_category,
)
from jaxarc.utils.visualization import (
    draw_grid_svg,
    draw_rl_step_svg,
    log_grid_to_console,
    save_svg_drawing,
    visualize_grid_rich,
)


def demo_grid_type_support():
    """Demonstrate Grid type support in visualization functions."""
    console = Console()
    console.print("\n[bold blue]Demo 1: Grid Type Support[/bold blue]")

    # Create a Grid object
    data = jnp.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]], dtype=jnp.int32)

    mask = jnp.array(
        [[True, True, True], [True, True, False], [True, False, False]], dtype=jnp.bool_
    )

    grid = Grid(data=data, mask=mask)

    # Test console visualization
    console.print("\n[green]Console visualization with Grid object:[/green]")
    log_grid_to_console(grid, title="Grid Object Demo", show_numbers=True)

    # Test Rich visualization
    console.print("\n[green]Rich visualization with Grid object:[/green]")
    rich_table = visualize_grid_rich(grid, title="Grid with Mask")
    console.print(rich_table)

    # Test SVG generation
    console.print("\n[green]SVG generation with Grid object:[/green]")
    svg_drawing = draw_grid_svg(grid, label="Grid Object SVG")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        temp_path = f.name

    save_svg_drawing(svg_drawing, temp_path)
    console.print(f"SVG saved to: {temp_path}")

    return temp_path


def demo_operation_names():
    """Demonstrate the operation names utility."""
    console = Console()
    console.print("\n[bold blue]Demo 2: Operation Names Utility[/bold blue]")

    # Show operation categories
    console.print("\n[green]Operation categories:[/green]")
    categories = get_operations_by_category()
    for category, ops in categories.items():
        console.print(f"  {category}: {ops}")

    # Show some example operations
    console.print("\n[green]Example operations:[/green]")
    example_ops = [0, 5, 10, 15, 20, 24, 28, 32, 34]
    for op_id in example_ops:
        name = get_operation_name(op_id)
        display_text = get_operation_display_text(op_id)
        console.print(f"  {op_id}: {name} -> '{display_text}'")


def demo_rl_step_visualization():
    """Demonstrate the enhanced RL step visualization."""
    console = Console()
    console.print("\n[bold blue]Demo 3: Enhanced RL Step Visualization[/bold blue]")

    # Create before and after grids
    before_data = jnp.array(
        [[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 2, 2], [0, 2, 2, 0]], dtype=jnp.int32
    )

    after_data = jnp.array(
        [[0, 0, 3, 0], [0, 3, 3, 0], [3, 3, 2, 2], [0, 2, 2, 0]], dtype=jnp.int32
    )

    mask = jnp.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=jnp.bool_,
    )

    before_grid = Grid(data=before_data, mask=mask)
    after_grid = Grid(data=after_data, mask=mask)

    # Create a selection mask (bbox selection)
    selection_mask = jnp.array(
        [
            [False, False, True, False],
            [False, True, True, False],
            [True, True, False, False],
            [False, False, False, False],
        ],
        dtype=jnp.bool_,
    )

    # Generate RL step visualization
    console.print("\n[green]Generating RL step visualization...[/green]")
    svg_content = draw_rl_step_svg(
        before_grid=before_grid,
        after_grid=after_grid,
        selection_mask=selection_mask,
        operation_id=3,  # Fill 3
        step_number=1,
        label="Demo Episode",
    )

    # Save the visualization
    output_path = "enhanced_rl_step_demo.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    console.print(f"RL step visualization saved to: {output_path}")
    return output_path


def demo_different_action_formats():
    """Demonstrate visualizations with different action formats."""
    console = Console()
    console.print("\n[bold blue]Demo 4: Different Action Formats[/bold blue]")

    # Create a simple grid
    data = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=jnp.int32)

    mask = jnp.ones((3, 3), dtype=jnp.bool_)
    grid = Grid(data=data, mask=mask)

    # Point selection
    console.print("\n[green]Point selection visualization:[/green]")
    point_selection = jnp.array(
        [[False, True, False], [False, False, False], [False, False, False]],
        dtype=jnp.bool_,
    )

    point_svg = draw_rl_step_svg(
        before_grid=grid,
        after_grid=grid,
        selection_mask=point_selection,
        operation_id=20,  # Move Up
        step_number=1,
        label="Point Selection",
    )

    # Bbox selection
    console.print("\n[green]Bbox selection visualization:[/green]")
    bbox_selection = jnp.array(
        [[True, True, False], [True, True, False], [False, False, False]],
        dtype=jnp.bool_,
    )

    bbox_svg = draw_rl_step_svg(
        before_grid=grid,
        after_grid=grid,
        selection_mask=bbox_selection,
        operation_id=24,  # Rotate CW
        step_number=2,
        label="Bbox Selection",
    )

    # Save visualizations
    with open("point_selection_demo.svg", "w", encoding="utf-8") as f:
        f.write(point_svg)

    with open("bbox_selection_demo.svg", "w", encoding="utf-8") as f:
        f.write(bbox_svg)

    console.print("Action format visualizations saved to:")
    console.print("  - point_selection_demo.svg")
    console.print("  - bbox_selection_demo.svg")


def demo_environment_integration():
    """Demonstrate the environment integration with enhanced visualization."""
    console = Console()
    console.print("\n[bold blue]Demo 5: Enhanced Environment Integration[/bold blue]")

    # Create environment with enhanced visualization enabled
    config_overrides = [
        "debug/=on",
        "visualization/=debug_standard",
        "storage/=development",
        "action.selection_format=point",
        "max_episode_steps=8",
    ]
    
    hydra_config = get_config(overrides=config_overrides)
    
    # Add enhanced visualization config
    from omegaconf import OmegaConf
    enhanced_config = {
        "enhanced_visualization": {
            "enabled": True,
            "level": "standard",
        }
    }
    hydra_config = OmegaConf.merge(hydra_config, OmegaConf.create(enhanced_config))
    
    # Convert to unified config
    from jaxarc.envs.equinox_config import JaxArcConfig
    unified_config = JaxArcConfig.from_hydra(hydra_config)
    env = ArcEnvironment(unified_config)

    # Run a short episode
    console.print("\n[green]Running environment with enhanced visualization...[/green]")
    key = jr.PRNGKey(42)
    state, obs = env.reset(key)

    total_reward = 0.0
    for i in range(5):
        action_key, key = jr.split(key)

        # Create a simple point action
        row, col = jr.randint(action_key, shape=(2,), minval=0, maxval=5)
        action = {
            "point": jnp.array([row, col]),
            "operation": jr.randint(action_key, shape=(), minval=0, maxval=10),
        }

        state, obs, reward, info = env.step(action)
        total_reward += float(reward)
        
        console.print(f"  Step {i+1}: reward={float(reward):.3f}, similarity={float(info['similarity']):.3f}")

        if env.is_done:
            console.print(f"Episode finished early at step {i + 1}")
            break

    console.print(f"[green]Enhanced episode completed! Total reward: {total_reward:.3f}[/green]")
    
    # Check if enhanced visualizer was used
    if hasattr(env, '_enhanced_visualizer') and env._enhanced_visualizer:
        console.print("✓ Enhanced visualization system was active")
        console.print("✓ Episode management enabled")
        console.print("✓ Asynchronous logging enabled")
    else:
        console.print("⚠ Using legacy visualization system")
    
    # Clean up
    env.close()
    console.print("Environment closed and resources cleaned up")


def main():
    """Main demo function."""
    logger.info("Starting Enhanced Visualization Demo")

    console = Console()
    console.print("[bold yellow]JaxARC Enhanced Visualization Demo[/bold yellow]")
    console.print("This demo showcases the new visualization capabilities:")
    console.print("1. Grid type support")
    console.print("2. Operation names utility")
    console.print("3. Enhanced RL step visualization")
    console.print("4. Different action formats")
    console.print("5. Environment integration")

    try:
        # Run all demos
        demo_grid_type_support()
        demo_operation_names()
        demo_rl_step_visualization()
        demo_different_action_formats()
        demo_environment_integration()

        console.print("\n[bold green]All demos completed successfully![/bold green]")
        console.print("\nGenerated files:")
        console.print("- enhanced_rl_step_demo.svg")
        console.print("- point_selection_demo.svg")
        console.print("- bbox_selection_demo.svg")
        console.print("- outputs/enhanced_demo/step_*.svg")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        console.print(f"[bold red]Demo failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
