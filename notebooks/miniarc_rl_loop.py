# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # JaxARC RL Loop with a Random Agent (Bbox Actions)
#
# This notebook implements a complete Reinforcement Learning (RL) loop for a Random Agent using the JaxARC environment. It is designed to test the full functionality of the JaxARC ecosystem, including:
# - **Configuration Loading**: Using Hydra with overrides for the MiniARC dataset.
# - **Bbox Action Format**: The agent will select actions using bounding boxes.
# - **JAX-compliant Agent**: A random agent implemented with JAX for performance.
# - **Full Integration**: Complete with visualization, logging, and Weights & Biases (wandb) for experiment tracking.
#
# This script is a foundational step for developing more sophisticated agents like PPO.

# %%
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

# JaxARC imports
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.action_space_controller import ActionSpaceController
from jaxarc.envs.config import JaxArcConfig
from jaxarc.parsers import MiniArcParser
from jaxarc.types import Grid
from jaxarc.utils.visualization import (
    EpisodeConfig,
    EpisodeManager,
    EpisodeSummaryData,
    StepVisualizationData,
    VisualizationConfig,
    Visualizer,
    WandbConfig,
    WandbIntegration,
)
from jaxarc.utils.config import get_config

console = Console()


# %% [markdown]
# ## 1. Configuration Setup
#
# Here, we set up the configuration for our RL loop. We'll use Hydra to load the base configuration and apply specific overrides for our experiment. We're selecting the `mini_arc` dataset for its small 5x5 grids (ideal for rapid testing) and the `raw` action set, but we'll override the selection format to `bbox`.

# %%
def setup_configuration() -> DictConfig:
    """
    Loads and sets up the configuration for the RL loop.

    This function uses hydra to load the base configuration and then applies
    overrides to select the MiniARC dataset and the 'raw' (minimal) action space
    with the 'bbox' selection format.

    Returns:
        DictConfig: The fully resolved hydra configuration object.
    """
    logger.info("Setting up configuration...")
    # We are using the raw action config group and bbox selection format as requested.
    config_overrides = [
        "dataset=mini_arc",
        "action=raw",
        "action.selection_format=bbox",  # Use bbox selection format
        "visualization=full",
        "logging=full",
        "storage=research",
        "wandb=research",
    ]

    # Load the configuration using the get_config utility from JaxARC
    hydra_config = get_config(overrides=config_overrides)

    console.print(
        Panel(
            f"[bold green]Configuration Loaded Successfully[/bold green]\n\n"
            f"Dataset: {hydra_config.dataset.dataset_name}\n"
            f"Action Selection Format: {hydra_config.action.selection_format}\n"
            f"Allowed Operations: {hydra_config.action.allowed_operations}",
            title="JaxARC Configuration",
            border_style="green",
        )
    )

    return hydra_config


# %% [markdown]
# ## 2. Visualization and Logging Setup
#
# We'll initialize the `Visualizer` for saving step-by-step visual logs of the agent's behavior and the `WandbIntegration` for comprehensive experiment tracking. For this demo, wandb will run in offline mode.

# %%
def setup_visualization_and_logging() -> tuple[Visualizer, WandbIntegration]:
    """
    Initializes the visualization, logging, and wandb integration components.

    Returns:
        Tuple[Visualizer, WandbIntegration]: The initialized visualizer and wandb integration objects.
    """
    logger.info("Setting up visualization and logging...")

    # --- Enhanced Visualizer Setup ---
    vis_config = VisualizationConfig(
        debug_level="full",  # Log every step
        output_formats=["svg"],
        image_quality="high",
    )

    # Correctly create an EpisodeConfig object
    episode_config = EpisodeConfig(
        base_output_dir="outputs/rl_loop_random_agent_bbox",
        run_name=f"run_{int(time.time())}",
    )
    # Correctly pass the config object to the EpisodeManager
    episode_manager = EpisodeManager(config=episode_config)

    # Correctly pass vis_config as the 'config' argument to Visualizer
    visualizer = Visualizer(
        config=vis_config, episode_manager=episode_manager
    )
    
    # --- Wandb Integration Setup ---
    wandb_config = WandbConfig(
        enabled=False,
        project_name="jaxarc-benchmarks",
        offline_mode=False,  # Use offline mode for this demo
        tags=["random-agent", "miniarc", "bbox-actions"],
    )

    wandb_integration = WandbIntegration(wandb_config)

    console.print(
        Panel(
            f"Project: {wandb_config.project_name}\n"
            f"Offline Mode: {wandb_config.offline_mode}",
            title="Wandb Integration",
            border_style="magenta",
        )
    )

    return visualizer, wandb_integration


# %% [markdown]
# ## 3. Random Agent Definition
#
# This section defines our `RandomAgent`. It's a simple, JAX-compliant agent that selects actions randomly. The `select_action` method is JIT-compiled for performance and has been updated to generate random bounding boxes (`bbox`).

# %%
class AgentState(NamedTuple):
    """JAX-compatible state for the agent."""

    key: jax.Array


class RandomAgent:
    """A JAX-compliant agent that takes random actions using the ActionSpaceController."""

    def __init__(self, grid_shape: Tuple[int, int], action_controller: ActionSpaceController):
        """
        Initializes the RandomAgent.

        Args:
            grid_shape (Tuple[int, int]): The shape of the grid (height, width).
            action_controller (ActionSpaceController): Controller for managing valid actions.
        """
        self.grid_height, self.grid_width = grid_shape
        self.action_controller = action_controller

    @staticmethod
    def init_agent(key: jax.Array) -> AgentState:
        """Initializes the agent's state."""
        return AgentState(key=key)

    def select_action(
        self, agent_state: AgentState, observation: jax.Array, env_state, config
    ) -> Tuple[dict, AgentState]:
        """
        Selects a random action using the ActionSpaceController to ensure validity.

        Args:
            agent_state (AgentState): The current state of the agent.
            observation (jax.Array): The environment observation (not used by this random agent).
            env_state: The current environment state for action validation.
            config: The action configuration.

        Returns:
            Tuple[dict, AgentState]: A tuple containing the action dictionary and the new agent state.
        """
        key, op_key, coord_key = jr.split(agent_state.key, 3)

        # Get allowed operations from the action space controller
        allowed_mask = self.action_controller.get_allowed_operations(env_state, config.action)
        allowed_operations = jnp.where(allowed_mask)[0]
        
        # If no operations are allowed, fallback to operation 0 (usually a safe no-op)
        num_allowed = len(allowed_operations)
        if num_allowed == 0:
            logger.warning("No allowed operations found, using operation 0 as fallback")
            operation = jnp.array(0, dtype=jnp.int32)
        else:
            # Select a random operation from the allowed list
            random_op_index = jr.randint(
                op_key, shape=(), minval=0, maxval=num_allowed
            )
            operation = allowed_operations[random_op_index]

        # Generate a random bounding box [r1, c1, r2, c2]
        coords = jr.randint(coord_key, shape=(4,), minval=0, maxval=max(self.grid_height, self.grid_width))
        r1 = coords[0] % self.grid_height
        c1 = coords[1] % self.grid_width
        r2 = coords[2] % self.grid_height
        c2 = coords[3] % self.grid_width
        
        # Ensure r1 <= r2 and c1 <= c2
        bbox = jnp.array([
            jnp.minimum(r1, r2),
            jnp.minimum(c1, c2),
            jnp.maximum(r1, r2),
            jnp.maximum(c1, c2)
        ])

        action = {"bbox": bbox, "operation": operation}

        new_agent_state = AgentState(key=key)

        return action, new_agent_state


# %% [markdown]
# ## 4. The Reinforcement Learning Loop
#
# This is the core of the script. The `run_rl_loop` function orchestrates the interaction between the agent and the environment over multiple episodes. It handles task loading, environment resets, action selection, stepping, and logging.

# %%
def run_rl_loop(
    hydra_config: DictConfig,
    visualizer: Visualizer,
    wandb_integration: WandbIntegration,
    num_episodes: int = 5,
    max_steps_per_episode: int = 20,
):
    """
    The main reinforcement learning loop.

    Args:
        hydra_config (DictConfig): The environment configuration from Hydra.
        visualizer (Visualizer): The visualization utility.
        wandb_integration (WandbIntegration): The wandb utility.
        num_episodes (int): The number of episodes to run.
        max_steps_per_episode (int): The maximum number of steps per episode.
    """
    logger.info("Starting RL Loop...")

    config = JaxArcConfig.from_hydra(hydra_config)

    # --- Dataset Loading ---
    parser = MiniArcParser(config.dataset)
    training_tasks = parser.get_available_task_ids()

    # --- Action Space Controller Initialization ---
    action_controller = ActionSpaceController()
    
    # --- Agent Initialization ---
    agent = RandomAgent(
        grid_shape=(config.dataset.max_grid_height, config.dataset.max_grid_width),
        action_controller=action_controller,
    )

    # --- Wandb Initialization ---
    experiment_config = {
        "agent": "RandomAgent",
        "dataset": config.dataset.dataset_name,
        "action_format": config.action.selection_format,
        "num_episodes": num_episodes,
        "max_steps": max_steps_per_episode,
    }
    wandb_integration.initialize_run(experiment_config=experiment_config)

    # --- Main Loop ---
    key = jr.PRNGKey(42)
    key, task_key = jr.split(key, 2)

    # Select a random task
    task_id_index = jr.randint(task_key, shape=(), minval=0, maxval=len(training_tasks))
    task_id = training_tasks[int(task_id_index)]
    task = parser.get_task_by_id(task_id)

    # Start a new run for the entire experiment
    visualizer.episode_manager.start_new_run()

    for episode_idx in range(num_episodes):
        console.rule(f"[bold cyan]Episode {episode_idx + 1}/{num_episodes}")

        # --- Episode Initialization ---
        key, env_key, agent_key = jr.split(key, 3)

        # Reset the environment
        state, observation = arc_reset(env_key, config, task_data=task)

        # Initialize the agent
        agent_state = agent.init_agent(agent_key)

        # Start visualization for the episode
        visualizer.start_episode(episode_idx, task_id=task_id)

        # Debug: Show allowed operations for this episode
        allowed_mask = action_controller.get_allowed_operations(state, config.action)
        allowed_ops = [i for i in range(len(allowed_mask)) if allowed_mask[i]]
        logger.info(f"Episode {episode_idx}: Allowed operations: {allowed_ops}")

        total_reward = 0.0
        reward_progression = []
        similarity_progression = []

        # --- Step Loop ---
        for step_num in range(max_steps_per_episode):
            # Select action using the action controller for validation
            action, agent_state = agent.select_action(agent_state, observation, state, config)

            # Store state before the step for visualization
            state_before = state

            # Step the environment
            state, observation, reward, done, info = arc_step(state, action, config)

            total_reward += reward
            reward_progression.append(float(total_reward))
            similarity_progression.append(float(info.get("similarity", 0.0)))

            # --- Convert JAX arrays in action and info to NumPy for logging ---
            action_log = {k: np.asarray(v) if hasattr(v, 'shape') else v for k, v in action.items()}
            info_log = {k: np.asarray(v) if hasattr(v, 'shape') else v for k, v in info.items()}

            # --- Logging and Visualization ---
            step_metrics = {
                "reward": float(reward),
                "total_reward": float(total_reward),
                "similarity": float(info_log.get("similarity", 0.0)),
                "episode": episode_idx,
                "operation": int(action_log["operation"]),
            }

            # Log to wandb
            wandb_integration.log_step(
                step_num=episode_idx * max_steps_per_episode + step_num,
                metrics=step_metrics,
            )

            # Create StepVisualizationData object and log to visualizer
            step_data = StepVisualizationData(
                step_num=step_num,
                before_grid=Grid(data=state_before.working_grid, mask=state_before.working_grid_mask),
                after_grid=Grid(data=state.working_grid, mask=state.working_grid_mask),
                action=action_log,
                reward=float(reward),
                info=info_log
            )
            visualizer.visualize_step(step_data)

            logger.info(
                f"  Step {step_num}: Op={action_log['operation']}, Reward={reward:.3f}, Similarity={info_log.get('similarity', 0.0):.3f}"
            )

            if done:
                logger.info(f"Episode finished at step {step_num}.")
                break

        # --- Episode Summary ---
        summary_data = EpisodeSummaryData(
            episode_num=episode_idx,
            total_steps=int(state.step_count),
            total_reward=float(total_reward),
            reward_progression=reward_progression,
            similarity_progression=similarity_progression,
            final_similarity=float(state.similarity_score),
            task_id=task_id,
            success=float(state.similarity_score) >= 0.99
        )
        
        visualizer.visualize_episode_summary(summary_data)
        wandb_integration.log_episode_summary(episode_idx, {
            "episode_reward": summary_data.total_reward,
            "episode_steps": summary_data.total_steps,
            "final_similarity": summary_data.final_similarity,
            "success": summary_data.success
        })

    # --- Cleanup ---
    wandb_integration.finish_run()
    logger.info("RL Loop finished.")


# %% [markdown]
# ## 5. Main Execution Block
#
# This final cell brings everything together. It calls the setup functions and then starts the RL loop.

# %%
def main():
    """Main function to run the RL loop script."""
    try:
        config = setup_configuration()
        visualizer, wandb_integration = setup_visualization_and_logging()
        run_rl_loop(config, visualizer, wandb_integration)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        console.print_exception()


if __name__ == "__main__":
    main()

# %%
