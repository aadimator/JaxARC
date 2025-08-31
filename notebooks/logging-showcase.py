#!/usr/bin/env python3
"""
JaxARC Logging Showcase

This script demonstrates the comprehensive logging capabilities of the JaxARC
environment. It runs a short RL episode with a random agent and showcases
the output from various logging handlers:

1.  **RichHandler**: Provides live, richly formatted output to the console,
    including task visualizations, step-by-step grid changes, and summary tables.
2.  **FileHandler**: Saves detailed, serialized episode data (states, actions, rewards)
    to JSON and Pickle files for later analysis.
3.  **SVGHandler**: Generates high-quality SVG visualizations for the task overview,
    each individual step, and a final episode summary.
4.  **WandbHandler**: (Setup included but disabled by default) Integrates with
    Weights & Biases for experiment tracking.

Usage:
    # Run the showcase with default settings
    pixi run python scripts/logging_showcase.py

    # To enable Weights & Biases logging (requires wandb account):
    # 1. Set up your wandb credentials (`wandb login`)
    # 2. Add the override:
    pixi run python scripts/logging_showcase.py --config-overrides "wandb.enabled=true" "wandb.project_name=JaxARC-Logging-Demo"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import jax
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.actions import StructuredAction, create_point_action
from jaxarc.state import State
from jaxarc.types import TimeStep
from jaxarc.utils.config import get_config
from jaxarc.utils.logging import ExperimentLogger
from jaxarc.registration import make
from jaxarc.utils.serialization_utils import serialize_log_step
from jaxarc.utils.logging.logging_utils import build_step_logging_payload, build_task_metadata_from_params, build_episode_summary_payload

# --- 1. Agent Definition (Pure Functional Style) ---
class AgentState(NamedTuple):
    """A simple state for our random agent."""

    key: jax.Array


def random_agent_policy(
    state: State, key: jax.Array, config: JaxArcConfig
) -> StructuredAction:
    """A pure function representing the policy of a random agent."""
    del state  # Unused for a random agent
    h, w = config.dataset.max_grid_height, config.dataset.max_grid_width
    k1, k2, k3 = jax.random.split(key, 3)
    op = jax.random.randint(k1, (), 0, 35)  # All grid operations
    r = jax.random.randint(k2, (), 0, h)
    c = jax.random.randint(k3, (), 0, w)
    return create_point_action(op, r, c)


# --- 2. Main Showcase Function ---
def run_logging_showcase(config_overrides: list[str]):
    """
    Sets up and runs a single episode to demonstrate logging.
    """
    console = Console()
    console.rule("[bold yellow]JaxARC Logging Showcase[/bold yellow]")

    # --- Configuration Setup ---
    logger.info("Setting up configuration for logging showcase...")
    # Start with a minimal config and enable all logging/visualization
    base_overrides = [
        "dataset=mini_arc",
        "action=full",  # Use all actions to see variety in logs
        "environment.debug_level=verbose",  # Standard debug level
        # Enable all logging and visualization features
        "logging.log_operations=true",
        "logging.log_rewards=true",
        "visualization.enabled=true",
        "visualization.episode_summaries=true",
        "visualization.step_visualizations=true",
        # Use a dedicated output directory for this showcase
        "storage.base_output_dir=outputs/logging_showcase",
        "storage.run_name=logging_demo_run",
        "storage.clear_output_on_start=true",  # Clear previous runs
        # Keep wandb disabled by default
        "wandb.enabled=false",
    ]
    all_overrides = base_overrides + config_overrides
    hydra_config = get_config(overrides=all_overrides)
    config = JaxArcConfig.from_hydra(hydra_config)

    console.print(
        Panel(
            f"[bold green]Configuration Loaded[/bold green]\n\n"
            f"Dataset: {config.dataset.dataset_name}\n"
            f"Action Format: {config.action.selection_format}\n"
            f"Logging Level: {config.logging.log_level}\n"
            f"Visualization Enabled: {config.visualization.enabled}\n"
            f"Output Directory: {config.storage.base_output_dir}/{config.storage.run_name}\n"
            f"WandB Enabled: {config.wandb.enabled}",
            title="Showcase Configuration",
            border_style="green",
        )
    )

    # --- Initialize Logger ---
    # The ExperimentLogger automatically detects the config and sets up handlers.
    exp_logger = ExperimentLogger(config)

    # --- Dataset and Environment Setup ---
    logger.info("Loading dataset and creating environment...")
    env, env_params = make("Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", config=config)
    key = jax.random.PRNGKey(1)

    # --- Run a Single Episode ---
    logger.info("Starting episode run...")
    start_time = time.time()
    timestep: TimeStep = env.reset(env_params, key=key)

    # Log task start
    metadata = build_task_metadata_from_params(env_params, timestep.state.task_idx)
    exp_logger.log_task_start(metadata)

    done = False
    step_count = 0
    total_reward = 0.0
    episode_steps_data = []

    while not done:
        key, action_key = jax.random.split(key)
        action = random_agent_policy(timestep.state, action_key, config)

        prev_state = timestep.state
        timestep = env.step(env_params, timestep, action)
        total_reward += timestep.reward
        step_count += 1

        # Prepare data for logging
        # Convert the (possibly JAX) step_count scalar to a host Python int where possible.
        try:
            step_num_val = int(timestep.state.step_count.item())
        except Exception:
            try:
                step_num_val = int(timestep.state.step_count)
            except Exception:
                step_num_val = None

        step_data_for_log = build_step_logging_payload(
            before_state=prev_state,
            after_state=timestep.state,
            action=action,
            reward=timestep.reward,
            info={},
            step_num=step_num_val,
            episode_num=42,
            params=env_params,  # optional; provide to include total_task_pairs
            include_task_meta=True,
            include_grids=False,  # don't attach full grids in this demo
        )
        # step_data_for_log = {
        #     "step_num": step_count,
        #     "episode_num": 0,
        #     "before_state": prev_state,
        #     "after_state": state,
        #     "action": action,
        #     "reward": reward,
        #     "info": info,
        #     "task_id": task_id,
        #     "task_pair_index": state.current_example_idx,
        #     "total_task_pairs": state.task_data.num_train_pairs,
        # }
        episode_steps_data.append(step_data_for_log)

        # Log the step
        exp_logger.log_step(step_data_for_log)

        # Update termination flag from the timestep (best-effort host-side conversion).
        try:
            # Prefer the TimeStep helper if available
            done = bool(timestep.last())
        except Exception:
            try:
                # Fallback: inspect step_type scalar (2 indicates LAST)
                st = getattr(timestep, "step_type", None)
                try:
                    done = bool(int(st.item() == 2))
                except Exception:
                    done = bool(int(st == 2))
            except Exception:
                done = False

        # Stop if the episode is done
        if done:
            break

    end_time = time.time()
    logger.info(f"Episode finished in {end_time - start_time:.2f} seconds.")

    # --- Log Episode Summary ---
    # Build a consistent episode summary payload using the logging utilities.
    try:
        summary_payload = build_episode_summary_payload(
            episode_num=0,
            step_data=episode_steps_data,
            total_reward=total_reward,
            params=env_params,
            include_serialized_steps=True,
        )
        exp_logger.log_episode_summary(summary_payload)
    except Exception as e:
        logger.warning(f"Failed to build or log episode summary: {e}")

    # --- Clean Shutdown ---
    exp_logger.close()

    # --- Final Output ---
    output_path = Path(config.storage.base_output_dir) / config.storage.run_name
    console.rule("[bold yellow]Logging Showcase Complete[/bold yellow]")
    console.print(
        Panel(
            f"The logging showcase has finished.\n\n"
            f"Check the console output above to see the [bold cyan]RichHandler[/bold cyan] in action.\n\n"
            f"Detailed logs and visualizations have been saved to:\n"
            f"[green]{output_path.resolve()}[/green]\n\n"
            f"Inside you will find:\n"
            f"  - [bold]logs/[/bold] (from [bold cyan]FileHandler[/bold cyan]):\n"
            f"    - `episode_0000_...json`: Detailed step-by-step data.\n"
            f"    - `episode_0000_...pkl`: Pickled version for easy reloading.\n"
            f"  - [bold]visualizations/[/bold] (from [bold cyan]SVGHandler[/bold cyan]):\n"
            f"    - `episode_0000/`: Directory for this episode.\n"
            f"      - `task_overview.svg`: Visualization of the ARC task.\n"
            f"      - `step_...svg`: A separate SVG for each step.\n"
            f"      - `summary.svg`: A final summary visualization.\n\n"
            f"If you enabled [bold cyan]WandbHandler[/bold cyan], check your project online.",
            title="Outputs Generated",
            border_style="yellow",
        )
    )


def main(
    config_overrides: list[str] = typer.Option(  # noqa: B008
        None,
        "--config-overrides",
        "-c",
        help="Hydra config overrides, e.g., 'wandb.enabled=true'",
    ),
):
    """
    CLI entry point for the logging showcase script.
    """
    run_logging_showcase(config_overrides or [])


if __name__ == "__main__":
    typer.run(main)
