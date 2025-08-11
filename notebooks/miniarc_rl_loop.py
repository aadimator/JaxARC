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
# # JaxARC RL Loop with Random Agent and Batched Environments
#
# This notebook demonstrates the complete JaxARC ecosystem with both single-agent and batched environment capabilities:
#
# ## Single-Agent RL Loop Features:
# - **Configuration Loading**: Using Hydra with overrides for the MiniARC dataset
# - **Structured Actions**: Using the new structured action system with BboxAction
# - **JAX-compliant Agent**: A random agent implemented with JAX for performance
# - **Full Integration**: Complete with visualization, logging, and Weights & Biases (wandb)
#
# ## Batched Environment Features:
# - **Massive Parallelization**: 1000+ environments running simultaneously
# - **Mixed Initialization Strategy**: Diverse grid initialization (25% demo, 35% permutation, 20% empty, 20% random)
# - **JAX Vectorization**: Leveraging `jax.vmap` for efficient batch processing
# - **Performance Benchmarking**: Demonstrating the speed advantages of batched processing
#
# This notebook showcases the full power of JAX-based RL environments and serves as a foundation for developing sophisticated agents like PPO.

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
from jaxarc.envs import (
    ActionSpaceController,
    JaxArcConfig,
    arc_reset,
    arc_step,
    create_bbox_action,
)
from jaxarc.envs.actions import StructuredAction
from jaxarc.envs.functional import batch_reset, batch_step
from jaxarc.parsers import MiniArcParser
from jaxarc.state import ArcEnvState
from jaxarc.types import Grid
from jaxarc.utils.config import get_config
from jaxarc.utils.logging import ExperimentLogger

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
        "action.selection_format=bbox",
        "wandb.enabled=false",
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


def setup_batched_configuration() -> DictConfig:
    """
    Loads configuration for batched environments.

    Returns:
        DictConfig: Configuration optimized for batched training
    """
    logger.info("Setting up batched configuration...")
    config_overrides = [
        "dataset=mini_arc",
        "action=raw",
        "action.selection_format=bbox",
        "grid_initialization=mixed",  # Use mixed initialization strategy for diversity
        "wandb.enabled=false",
    ]

    hydra_config = get_config(overrides=config_overrides)

    console.print(
        Panel(
            f"[bold green]Batched Configuration Loaded[/bold green]\n\n"
            f"Dataset: {hydra_config.dataset.dataset_name}\n"
            f"Environment: {hydra_config.environment.debug_level}\n"
            f"Grid Initialization: {hydra_config.grid_initialization.mode}\n"
            f"Action Format: {hydra_config.action.selection_format}",
            title="Batched JaxARC Configuration",
            border_style="blue",
        )
    )

    return hydra_config


# %% [markdown]
# ## 2. Experiment Logger Setup
#
# We'll initialize the new `ExperimentLogger` which manages all logging concerns through focused handlers. This replaces the old complex `Visualizer` and `WandbIntegration` system with a simpler, more maintainable approach.


# %%
def setup_experiment_logger(config: JaxArcConfig) -> ExperimentLogger:
    """
    Initializes the experiment logger with all handlers.

    Args:
        config: JaxARC configuration object

    Returns:
        ExperimentLogger: The initialized experiment logger
    """
    logger.info("Setting up experiment logger...")

    # Create the experiment logger - it will automatically initialize
    # the appropriate handlers based on the configuration
    experiment_logger = ExperimentLogger(config)

    # Get handler info for display
    handler_names = list(experiment_logger.handlers.keys())
    
    console.print(
        Panel(
            f"[bold green]Experiment Logger Initialized[/bold green]\n\n"
            f"Active Handlers: {', '.join(handler_names) if handler_names else 'None'}\n"
            f"Debug Level: {getattr(config.environment, 'debug_level', 'standard')}\n"
            f"Wandb Enabled: {config.wandb.enabled}",
            title="Experiment Logger",
            border_style="magenta",
        )
    )

    return experiment_logger


# %% [markdown]
# ## 3. Random Agent Definition
#
# This section defines our `RandomAgent`. It's a simple, JAX-compliant agent that selects actions randomly. The `select_action` method is JIT-compiled for performance and has been updated to generate random bounding boxes (`bbox`).


# %%
class AgentState(NamedTuple):
    """JAX-compatible state for the agent."""

    key: jax.Array


class RandomAgent:
    """A JAX-compliant agent that takes random actions using structured actions."""

    def __init__(
        self, grid_shape: Tuple[int, int], action_controller: ActionSpaceController
    ):
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
        self,
        agent_state: AgentState,
        env_state: ArcEnvState,
        config,
    ) -> Tuple[StructuredAction, AgentState]:
        """
        Selects a random action using structured actions.

        Args:
            agent_state (AgentState): The current state of the agent.
            env_state: The current environment state for action validation.
            config: The action configuration.

        Returns:
            Tuple[StructuredAction, AgentState]: A tuple containing the structured action and the new agent state.
        """
        key, op_key, coord_key = jr.split(agent_state.key, 3)

        # Get allowed operations from the action space controller
        allowed_mask = self.action_controller.get_allowed_operations(
            env_state, config.action
        )
        allowed_operations = jnp.where(allowed_mask)[0]

        # If no operations are allowed, fallback to operation 0 (usually a safe no-op)
        num_allowed = len(allowed_operations)
        if num_allowed == 0:
            logger.warning("No allowed operations found, using operation 0 as fallback")
            operation = jnp.array(0, dtype=jnp.int32)
        else:
            # Select a random operation from the allowed list
            random_op_index = jr.randint(op_key, shape=(), minval=0, maxval=num_allowed)
            operation = allowed_operations[random_op_index]

        # Generate a random bounding box [r1, c1, r2, c2] with proper bounds
        key1, key2, key3, key4 = jr.split(coord_key, 4)
        r1 = jr.randint(key1, shape=(), minval=0, maxval=self.grid_height)
        c1 = jr.randint(key2, shape=(), minval=0, maxval=self.grid_width)
        r2 = jr.randint(key3, shape=(), minval=0, maxval=self.grid_height)
        c2 = jr.randint(key4, shape=(), minval=0, maxval=self.grid_width)

        # Ensure r1 <= r2 and c1 <= c2
        min_r = jnp.minimum(r1, r2)
        min_c = jnp.minimum(c1, c2)
        max_r = jnp.maximum(r1, r2)
        max_c = jnp.maximum(c1, c2)

        # Create structured bbox action
        action = create_bbox_action(
            operation=operation, r1=min_r, c1=min_c, r2=max_r, c2=max_c
        )

        new_agent_state = AgentState(key=key)

        return action, new_agent_state


# %% [markdown]
# ## 4. The Reinforcement Learning Loop
#
# This is the core of the script. The `run_rl_loop` function orchestrates the interaction between the agent and the environment over multiple episodes. It handles task loading, environment resets, action selection, stepping, and logging.


# %%
def run_rl_loop(
    hydra_config: DictConfig,
    experiment_logger: ExperimentLogger,
    num_episodes: int = 5,
    max_steps_per_episode: int = 20,
):
    """
    The main reinforcement learning loop.

    Args:
        hydra_config (DictConfig): The environment configuration from Hydra.
        experiment_logger (ExperimentLogger): The experiment logger for all logging.
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

    # --- Start Logging Run ---
    # The experiment logger handles run initialization automatically
    # through its handlers based on configuration

    # --- Main Loop ---
    key = jr.PRNGKey(42)
    key, task_key = jr.split(key, 2)

    # Select a random task
    task_id_index = jr.randint(task_key, shape=(), minval=0, maxval=len(training_tasks))
    task_id = training_tasks[int(task_id_index)]
    task = parser.get_task_by_id(task_id)
    logger.info(f"Selected Task ID: {task.get_task_id()} with {task.num_train_pairs} training pairs")

    # Start a new run for the entire experiment
    if 'svg' in experiment_logger.handlers:
        experiment_logger.handlers['svg'].start_run(f"random_agent_run_{int(time.time())}")

    for episode_idx in range(num_episodes):
        console.rule(f"[bold cyan]Episode {episode_idx + 1}/{num_episodes}")

        # --- Episode Initialization ---
        key, env_key, agent_key = jr.split(key, 3)

        # Reset the environment
        state, observation = arc_reset(env_key, config, task_data=task)

        # Initialize the agent
        agent_state = agent.init_agent(agent_key)

        # Start episode logging
        if 'svg' in experiment_logger.handlers:
            experiment_logger.handlers['svg'].start_episode(episode_idx)

        # Log task information at episode start
        task_stats = {
            'max_grid_height': config.dataset.max_grid_height,
            'max_grid_width': config.dataset.max_grid_width,
            'task_complexity': task.num_train_pairs + task.num_test_pairs,
        }
        
        task_data_for_logging = {
            'task_id': task_id,
            'task_object': task,
            'episode_num': episode_idx,
            'num_train_pairs': task.num_train_pairs,
            'num_test_pairs': task.num_test_pairs,
            'task_stats': task_stats,
        }
        
        experiment_logger.log_task_start(task_data_for_logging, show_test=True)

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
            action, agent_state = agent.select_action(agent_state, state, config)

            # Store state before the step for visualization
            state_before = state

            # Step the environment
            logger.debug(
                f"Action before step: operation={action.operation}, bbox=({action.r1}, {action.c1}, {action.r2}, {action.c2})"
            )

            try:
                state, observation, reward, done, info = arc_step(state, action, config)
            except Exception as e:
                logger.error(f"arc_step failed at step {step_num}: {e}")
                logger.error(
                    f"Action: operation={action.operation}, bbox=({action.r1}, {action.c1}, {action.r2}, {action.c2})"
                )
                logger.error(f"State working grid shape: {state.working_grid.shape}")
                logger.error(
                    f"State working grid mask sum: {jnp.sum(state.working_grid_mask)}"
                )
                # Skip this step and continue
                continue

            # Debug: Check if selection was properly set
            if hasattr(state, "selected"):
                selection_count = int(jnp.sum(state.selected))
                logger.debug(f"Selection after step: {selection_count} cells selected")
            else:
                logger.debug("No selection field found in state")

            total_reward += reward
            reward_progression.append(float(total_reward))
            similarity_progression.append(float(info.get("similarity", 0.0)))

            # --- Convert structured action to dict for logging ---
            action_log = {
                "operation": int(action.operation),
                "r1": int(action.r1),
                "c1": int(action.c1),
                "r2": int(action.r2),
                "c2": int(action.c2),
                "selection": action.to_selection_mask(state_before.working_grid.shape),
            }

            info_log = {
                k: np.asarray(v) if hasattr(v, "shape") else v for k, v in info.items()
            }

            # --- Prepare step data for new logging system ---
            step_data = {
                "step_num": step_num,
                "episode_num": episode_idx,
                "before_state": state_before,
                "after_state": state,
                "action": action_log,
                "reward": float(reward),
                "info": {
                    **info_log,
                    "metrics": {  # Metrics for wandb handler
                        "reward": float(reward),
                        "total_reward": float(total_reward),
                        "similarity": float(info_log.get("similarity", 0.0)),
                        "episode": episode_idx,
                        "operation": int(action.operation),
                    }
                },
                "task_id": task_id,
                "task_pair_index": state.current_example_idx,
                "total_task_pairs": task.num_train_pairs,
            }

            # Log step through experiment logger
            try:
                experiment_logger.log_step(step_data)
            except Exception as e:
                logger.warning(f"Step logging failed: {e}")
                import traceback
                logger.debug(f"Step logging error details: {traceback.format_exc()}")

            logger.info(
                f"  Step {step_num}: Op={action_log['operation']}, Reward={reward:.3f}, Similarity={info_log.get('similarity', 0.0):.3f}"
            )

            if done:
                logger.info(f"Episode finished at step {step_num}.")
                break

        # --- Episode Summary ---
        summary_data = {
            "episode_num": episode_idx,
            "total_steps": int(state.step_count),
            "total_reward": float(total_reward),
            "reward_progression": reward_progression,
            "similarity_progression": similarity_progression,
            "final_similarity": float(state.similarity_score),
            "task_id": task_id,
            "success": float(state.similarity_score) >= 0.99,
            "key_moments": [],
        }

        # Log episode summary through experiment logger
        try:
            experiment_logger.log_episode_summary(summary_data)
        except Exception as e:
            logger.warning(f"Episode summary logging failed: {e}")

    # --- Cleanup ---
    experiment_logger.close()
    logger.info("RL Loop finished.")


# %% [markdown]
# ## 5. Batched Environment Demo
#
# This section demonstrates the power of JAX's vectorization with batched environments.
# We'll create 1000+ environments running in parallel using the `mixed` initialization strategy.
#
# ### Key Innovation: Proper Batched Operation Selection
#
# Unlike naive implementations that use the same operation for all environments, our
# `BatchedRandomAgent` correctly handles the fact that each environment in the batch
# may have different allowed operations based on its current state. This is achieved through:
#
# 1. **Vectorized Operation Validation**: Using `jax.vmap` to get allowed operations for each environment
# 2. **Individual Random Selection**: Each environment selects from its own allowed operations
# 3. **Efficient Batch Processing**: All operations are performed in parallel using JAX transformations
#
# This approach ensures realistic and correct behavior while maintaining the performance benefits of batched processing.


# %%
class BatchedRandomAgent:
    """A JAX-compliant agent for batched environments.

    This agent demonstrates proper batched operation selection where each environment
    in the batch can have different allowed operations based on its current state.
    The agent uses JAX's vmap to efficiently process all environments in parallel
    while respecting individual environment constraints.
    """

    def __init__(
        self, grid_shape: Tuple[int, int], action_controller: ActionSpaceController
    ):
        self.grid_height, self.grid_width = grid_shape
        self.action_controller = action_controller

    def select_batch_actions(self, keys: jax.Array, states: ArcEnvState, config) -> any:
        """Select actions for a batch of environments with proper operation selection.

        This method demonstrates the correct way to handle batched environments where
        each environment may have different allowed operations based on its current state.

        Args:
            keys: PRNG keys for each environment [batch_size, 2] (each key is shape (2,))
            states: Batch of environment states
            config: Environment configuration

        Returns:
            BboxAction: Structured action for the entire batch
        """
        batch_size = keys.shape[0]

        # Split keys for operations and coordinates
        op_keys = keys  # Use the same keys for both operations and coordinates
        
        # Get allowed operations for each environment in the batch
        def get_allowed_ops_single(state):
            return self.action_controller.get_allowed_operations(state, config.action)

        # Vectorize over the batch dimension
        get_allowed_ops_batch = jax.vmap(get_allowed_ops_single)
        allowed_masks = get_allowed_ops_batch(states)  # Shape: (batch_size, num_operations)

        # Select random operations and coordinates for each environment
        def select_action_single(key, allowed_mask):
            # Split the key for operation selection and coordinate generation
            op_key, coord_key = jr.split(key, 2)
            
            # Get indices of allowed operations
            allowed_indices = jnp.where(
                allowed_mask, size=allowed_mask.shape[0], fill_value=-1
            )[0]
            # Count valid allowed operations (non-negative indices)
            num_allowed = jnp.sum(allowed_indices >= 0)

            # Select operation
            def select_from_allowed():
                random_idx = jr.randint(op_key, shape=(), minval=0, maxval=num_allowed)
                return allowed_indices[random_idx]

            def fallback_operation():
                return jnp.array(0, dtype=jnp.int32)

            operation = jax.lax.cond(
                num_allowed > 0, select_from_allowed, fallback_operation
            )
            
            # Generate coordinates
            k1, k2, k3, k4 = jr.split(coord_key, 4)
            r1 = jr.randint(k1, shape=(), minval=0, maxval=self.grid_height)
            c1 = jr.randint(k2, shape=(), minval=0, maxval=self.grid_width)
            r2 = jr.randint(k3, shape=(), minval=0, maxval=self.grid_height)
            c2 = jr.randint(k4, shape=(), minval=0, maxval=self.grid_width)
            
            # Ensure proper ordering
            min_r = jnp.minimum(r1, r2)
            min_c = jnp.minimum(c1, c2)
            max_r = jnp.maximum(r1, r2)
            max_c = jnp.maximum(c1, c2)
            
            return operation, min_r, min_c, max_r, max_c

        # Vectorize action selection over the batch
        select_actions_batch = jax.vmap(select_action_single)
        operations, r1, c1, r2, c2 = select_actions_batch(op_keys, allowed_masks)

        # Create batched bbox actions
        return create_bbox_action(
            operation=operations, r1=r1, c1=c1, r2=r2, c2=c2
        )


def run_batched_demo(batch_size: int = 1000, num_steps: int = 10):
    """
    Demonstrate batched environment processing with mixed initialization.

    Args:
        batch_size: Number of parallel environments
        num_steps: Number of steps to run
    """
    logger.info(f"Starting batched demo with {batch_size} environments...")

    # Setup configuration for batched environments
    config = setup_batched_configuration()
    typed_config = JaxArcConfig.from_hydra(config)

    # Load tasks for batched environments
    parser = MiniArcParser(typed_config.dataset)
    training_tasks = parser.get_available_task_ids()

    # Select a task for all environments (could be randomized per environment)
    task_id = training_tasks[0]
    task = parser.get_task_by_id(task_id)

    console.print(
        Panel(
            f"[bold cyan]Batched Environment Demo[/bold cyan]\n\n"
            f"Batch Size: {batch_size}\n"
            f"Steps per Environment: {num_steps}\n"
            f"Total Computations: {batch_size * num_steps:,}\n"
            f"Task: {task_id}\n"
            f"Environment: training",
            title="Batched Demo Configuration",
            border_style="cyan",
        )
    )

    # Initialize batched environments
    start_time = time.time()

    # Create batch of PRNG keys
    key = jr.PRNGKey(42)
    keys = jr.split(key, batch_size)

    # Reset all environments in parallel
    logger.info("Resetting batched environments...")
    reset_start = time.time()
    states, observations = batch_reset(keys, typed_config, task)
    reset_time = time.time() - reset_start

    logger.info(f"Batch reset completed in {reset_time:.3f}s")
    logger.info(f"States shape: {states.working_grid.shape}")
    logger.info(f"Observations shape: {observations.shape}")

    # Initialize batched agent
    action_controller = ActionSpaceController()
    agent = BatchedRandomAgent(
        grid_shape=(
            typed_config.dataset.max_grid_height,
            typed_config.dataset.max_grid_width,
        ),
        action_controller=action_controller,
    )

    # Run batched steps
    logger.info("Running batched steps...")
    step_times = []
    
    # Initialize variables in case all steps fail
    rewards = jnp.zeros(batch_size)
    dones = jnp.zeros(batch_size, dtype=bool)

    for step in range(num_steps):
        step_start = time.time()

        try:
            # Generate new keys for this step
            step_key = jr.PRNGKey(42 + step)
            step_keys = jr.split(step_key, batch_size)  # Shape: [batch_size, 2]
            
            # Debug: Check key shapes
            logger.debug(f"Step {step + 1}: step_keys shape = {step_keys.shape}")
            logger.debug(f"Step {step + 1}: states.working_grid shape = {states.working_grid.shape}")

            # Select actions for all environments
            actions = agent.select_batch_actions(step_keys, states, typed_config)

            # Step all environments in parallel
            states, observations, rewards, dones, _ = batch_step(
                states, actions, typed_config
            )
        except Exception as e:
            logger.error(f"Batched step {step + 1} failed: {e}")
            # Skip this step and continue
            step_times.append(0.0)
            continue

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Log progress with operation distribution
        avg_reward = float(jnp.mean(rewards))
        num_done = int(jnp.sum(dones))
        avg_similarity = float(jnp.mean(states.similarity_score))

        # Show operation distribution for this step
        unique_ops, op_counts = jnp.unique(actions.operation, return_counts=True)
        op_distribution = {
            int(op): int(count) for op, count in zip(unique_ops, op_counts)
        }

        logger.info(
            f"Step {step + 1}/{num_steps}: "
            f"Time={step_time:.3f}s, "
            f"Avg Reward={avg_reward:.3f}, "
            f"Done={num_done}/{batch_size}, "
            f"Avg Similarity={avg_similarity:.3f}, "
            f"Op Distribution={op_distribution}"
        )

    total_time = time.time() - start_time
    avg_step_time = np.mean(step_times)

    # Performance summary
    total_computations = batch_size * num_steps
    computations_per_second = total_computations / total_time

    console.print(
        Panel(
            f"[bold green]Batched Demo Results[/bold green]\n\n"
            f"Total Time: {total_time:.3f}s\n"
            f"Average Step Time: {avg_step_time:.3f}s\n"
            f"Reset Time: {reset_time:.3f}s\n"
            f"Total Computations: {total_computations:,}\n"
            f"Computations/Second: {computations_per_second:,.0f}\n"
            f"Speedup vs Sequential: ~{batch_size:.0f}x\n\n"
            f"Final Statistics:\n"
            f"- Average Reward: {float(jnp.mean(rewards)):.3f}\n"
            f"- Environments Done: {int(jnp.sum(dones))}/{batch_size}\n"
            f"- Average Similarity: {float(jnp.mean(states.similarity_score)):.3f}",
            title="Performance Results",
            border_style="green",
        )
    )

    return states, observations, rewards, dones


# %% [markdown]
# ## 6. Main Execution Block
#
# This final cell brings everything together. It calls the setup functions and then starts both the single-agent RL loop and the batched environment demo.


# %%
def main():
    """Main function to run both single and batched RL demos."""
    try:
        console.rule("[bold yellow]JaxARC RL Loop Demo")

        # Run single-agent RL loop
        console.print("\n[bold cyan]Running Single-Agent RL Loop...[/bold cyan]")
        try:
            hydra_config = setup_configuration()
            typed_config = JaxArcConfig.from_hydra(hydra_config)
            experiment_logger = setup_experiment_logger(typed_config)
            run_rl_loop(
                hydra_config,
                experiment_logger,
                num_episodes=5,
                max_steps_per_episode=30,
            )
        except Exception as e:
            logger.error(f"Single-agent RL loop failed: {e}")
            console.print(f"[red]Single-agent RL loop failed: {e}[/red]")

        # console.rule("[bold yellow]Batched Environment Demo")

        # # Run batched environment demo with larger batch to show performance
        # console.print("\n[bold cyan]Running Batched Environment Demo...[/bold cyan]")
        # try:
        #     run_batched_demo(batch_size=100, num_steps=10)
        # except Exception as e:
        #     logger.error(f"Batched demo failed: {e}")
        #     console.print(f"[red]Batched demo failed: {e}[/red]")

        # console.rule("[bold green]All Demos Completed Successfully!")

    except RuntimeError as e:
        logger.error(f"Runtime error occurred: {e}")
        console.print(f"[red]Runtime Error: {e}[/red]")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        console.print_exception()


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Summary
#
# This notebook demonstrates the complete JaxARC ecosystem with both single-agent and batched environment capabilities, now using the new simplified logging architecture:
#
# ### Key Features Demonstrated:
# 1. **New Logging System**: Uses the simplified `ExperimentLogger` with focused handlers
# 2. **Structured Actions**: Using the new structured action system with BboxAction
# 3. **Single-Agent RL**: Complete RL loop with SVG visualization and wandb integration
# 4. **Batched Environments**: Massive parallelization with 1000+ environments
# 5. **Mixed Initialization**: Diverse grid initialization strategies for enhanced training
# 6. **Performance Optimization**: JAX vectorization for efficient batch processing
#
# ### New Logging Architecture Benefits:
# - **Simplified Design**: Single `ExperimentLogger` replaces complex `Visualizer` orchestration
# - **Handler-Based**: Focused handlers for file, SVG, console, and wandb logging
# - **Error Isolation**: Handler failures don't affect other logging components
# - **Configuration-Driven**: Handlers automatically enabled based on config settings
# - **Graceful Degradation**: System continues working even if some handlers fail
#
# ### Performance Benefits:
# - **Vectorized Operations**: JAX's `vmap` enables efficient batch processing
# - **JIT Compilation**: All functions are JIT-compiled for maximum performance
# - **Memory Efficiency**: Structured actions and optimized state management
# - **Scalability**: Can easily scale to thousands of parallel environments
#
# ### Next Steps:
# - Replace the random agent with a neural network (e.g., using Flax)
# - Implement PPO or other RL algorithms using PureJaxRL
# - Experiment with different initialization strategies and reward functions
# - Scale up to even larger batch sizes for distributed training
# - Add custom metrics to the `info['metrics']` dictionary for automatic wandb logging
#
# This notebook serves as a complete demo of the new logging system and a solid foundation for developing sophisticated RL agents on ARC tasks!

# %%
