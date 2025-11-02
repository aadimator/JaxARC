#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import jax
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from jaxarc.configs import JaxArcConfig
from jaxarc.envs import BboxActionWrapper, FlattenActionWrapper
from jaxarc.registration import make
from jaxarc.utils.core import get_config
from jaxarc.utils.logging import (
    ExperimentLogger,
    create_episode_summary,
    create_start_log,
    create_step_log,
)


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
    # env = PointActionWrapper(env)
    env = BboxActionWrapper(env)
    env = FlattenActionWrapper(env)

    # Get action space for the agent policy
    action_space = env.action_space(env_params)
    key = jax.random.PRNGKey(1)

    # --- Run Multiple Episodes ---
    logger.info("Starting 5-episode run...")

    for episode_num in range(5):
        logger.info(f"Starting episode {episode_num}...")
        start_time = time.time()
        state, timestep = env.reset(key, env_params=env_params)

        # Start episode logging with proper episode tracking
        metadata = create_start_log(
            params=env_params,
            state=state,
            episode_num=episode_num,
        )

        exp_logger.log_task_start(metadata)

        done = False
        step_count = 0
        total_reward = 0.0
        episode_steps_data = []

        while not done:
            key, action_key = jax.random.split(key)
            # Direct action sampling - much simpler!
            action = action_space.sample(action_key)

            # Store previous state for logging
            prev_state = state
            state, timestep = env.step(state, action, env_params=env_params)
            total_reward += timestep.reward
            step_count += 1

            # Prepare data for logging
            # Convert the (possibly JAX) step_count scalar to a host Python int where possible.
            try:
                step_num_val = int(step_count)
            except Exception:
                step_num_val = None

            step_data_for_log = create_step_log(
                timestep=timestep,
                state=state,
                action=action,
                step_num=step_num_val,
                episode_num=episode_num,
                prev_state=prev_state,
                env_params=env_params,
            )

            episode_steps_data.append(step_data_for_log)

            # Log the step
            exp_logger.log_step(step_data_for_log)

            # Update termination flag from the timestep (best-effort host-side conversion).
            try:
                # Use the TimeStep helper methods
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
        logger.info(
            f"Episode {episode_num} finished in {end_time - start_time:.2f} seconds."
        )

        # --- Log Episode Summary ---
        # Build a consistent episode summary payload using the logging utilities.
        try:
            summary_payload = create_episode_summary(
                episode_num=episode_num,
                step_logs=episode_steps_data,
                env_params=env_params,
            )
            exp_logger.log_episode_summary(summary_payload)
        except Exception as e:
            logger.warning(f"Failed to build or log episode summary: {e}")

        # Generate new key for next episode
        key = jax.random.split(key)[0]

    # --- Clean Shutdown ---
    exp_logger.close()

    # --- Final Output ---
    output_path = Path(config.storage.base_output_dir) / config.storage.run_name
    console.rule("[bold yellow]Logging Showcase Complete[/bold yellow]")
    console.print(
        Panel(
            f"The logging showcase has finished running [bold]5 episodes[/bold].\n\n"
            f"Check the console output above to see the [bold cyan]RichHandler[/bold cyan] in action.\n\n"
            f"Detailed logs and visualizations have been saved to:\n"
            f"[green]{output_path.resolve()}[/green]\n\n"
            f"Inside you will find:\n"
            f"  - [bold]logs/[/bold] (from [bold cyan]FileHandler[/bold cyan]):\n"
            f"    - `episode_0000_...json` through `episode_0004_...json`: Detailed step-by-step data.\n"
            f"    - `episode_0000_...pkl` through `episode_0004_...pkl`: Pickled versions for easy reloading.\n"
            f"  - [bold]visualizations/[/bold] (from [bold cyan]SVGHandler[/bold cyan]):\n"
            f"    - `episode_0000/` through `episode_0004/`: Directories for each episode.\n"
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
