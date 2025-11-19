"""
Showcase of the unified render API in JaxARC.

This script demonstrates how to use the `env.render()` method with different modes:
- rgb_array: Returns a numpy array of the grid.
- ansi: Returns a string for terminal display.
- svg: Returns an SVG string.
"""

from __future__ import annotations

from pathlib import Path

import jax.random as jr
import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel

from jaxarc.configs import JaxArcConfig
from jaxarc.registration import make
from jaxarc.utils.core import get_config
from jaxarc.wrappers import StepVisualizationWrapper

console = Console()


def main():
    # 1. Setup Configuration
    console.rule("[bold yellow]JaxARC Render API Showcase")

    # Override config to ensure we have a task loaded
    config_overrides = [
        "dataset=mini_arc",
        "environment.render_mode=rgb_array",  # Default mode
    ]
    hydra_config = get_config(overrides=config_overrides)
    config = JaxArcConfig.from_hydra(hydra_config)

    # 2. Create Environment
    # We'll use a MiniARC task for demonstration
    from jaxarc.registration import available_task_ids

    available_ids = available_task_ids("Mini", config=config, auto_download=False)
    if not available_ids:
        console.print(
            "[red]No MiniARC tasks found. Please download dataset first.[/red]"
        )
        return

    task_id = available_ids[0]
    console.print(f"Creating environment for task: [cyan]{task_id}[/cyan]")

    env, env_params = make(f"Mini-{task_id}", config=config)

    # Wrap with StepVisualizationWrapper for detailed rendering
    env = StepVisualizationWrapper(env)

    # 3. Reset Environment
    key = jr.PRNGKey(42)
    state, _ = env.reset(key, env_params)

    # 4. Demonstrate Render Modes

    # --- ANSI Mode ---
    console.print(Panel("[bold]1. ANSI Render Mode[/bold]", style="blue"))
    ansi_output = env.render(state, mode="ansi")
    print(ansi_output)

    # --- RGB Array Mode ---
    console.print(Panel("[bold]2. RGB Array Render Mode[/bold]", style="green"))
    rgb_output = env.render(state, mode="rgb_array")
    console.print(f"RGB Output Shape: {rgb_output.shape}, Dtype: {rgb_output.dtype}")

    # Save RGB output as image
    try:
        img = Image.fromarray(rgb_output)
        img.save("render_rgb_example.png")
        console.print("Saved RGB render to [yellow]render_rgb_example.png[/yellow]")
    except ImportError:
        console.print("[red]PIL not installed. Skipping image save.[/red]")

    # --- SVG Mode ---
    console.print(Panel("[bold]3. SVG Render Mode[/bold]", style="magenta"))
    svg_output = env.render(state, mode="svg")
    console.print(f"SVG Output Length: {len(svg_output)} characters")

    # Save SVG output
    Path("render_svg_example.svg").write_text(svg_output, encoding="utf-8")
    console.print("Saved SVG render to [yellow]render_svg_example.svg[/yellow]")

    # --- Default Mode (from config) ---
    console.print(
        Panel("[bold]4. Default Render Mode (from config)[/bold]", style="white")
    )
    # Config was set to rgb_array
    default_output = env.render(state)
    console.print(f"Default Output Type: {type(default_output)}")
    if isinstance(default_output, np.ndarray):
        console.print(f"Default Output Shape: {default_output.shape}")

    # --- Detailed Mode (Wrapper) ---
    console.print(Panel("[bold]5. Detailed Render Mode (Wrapper)[/bold]", style="cyan"))

    # Perform a dummy step to generate a transition
    console.print("Performing a dummy step to generate transition...")
    grid_h, grid_w = state.working_grid.shape
    selection = np.zeros((grid_h, grid_w), dtype=bool)
    selection[:2, :2] = True

    # Simple action dict (assuming environment accepts it)
    action = {
        "operation": 0,  # FILL (usually)
        "selection": selection,
    }

    try:
        next_state, _ = env.step(state, action, env_params)

        detailed_svg = env.render(next_state, mode="detailed")
        console.print(f"Detailed SVG Output Length: {len(detailed_svg)} characters")
        Path("render_detailed_example.svg").write_text(detailed_svg, encoding="utf-8")
        console.print(
            "Saved Detailed SVG render to [yellow]render_detailed_example.svg[/yellow]"
        )
    except Exception as e:
        console.print(
            f"[red]Detailed render failed (step might have failed): {e}[/red]"
        )

    console.rule("[bold green]Showcase Complete")


if __name__ == "__main__":
    main()
