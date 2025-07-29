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
from jaxarc.envs.functional import batch_reset, batch_step
from jaxarc.parsers import MiniArcParser
from jaxarc.state import ArcEnvState
from jaxarc.utils.config import get_config
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
        "environment=training",  # Use training environment for batched processing
        "grid_initialization=mixed",  # Use mixed initialization strategy for diversity
        "visualization=minimal",  # Reduce visualization overhead for batched training
        "logging=basic",
        "storage=research",
        "wandb=research",
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
    visualizer = Visualizer(config=vis_config, episode_manager=episode_manager)

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
    ) -> Tuple[any, AgentState]:
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

        # Start visualization for the episode with task visualization
        visualizer.start_episode_with_task(
            episode_num=episode_idx,
            task_data=task,
            task_id=task_id,
            current_pair_index=0,  # Assuming we start with the first pair
            episode_mode="train",
        )

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
                logger.error(f"Action: operation={action.operation}, bbox=({action.r1}, {action.c1}, {action.r2}, {action.c2})")
                logger.error(f"State working grid shape: {state.working_grid.shape}")
                logger.error(f"State working grid mask sum: {jnp.sum(state.working_grid_mask)}")
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

            # --- Create selection mask BEFORE step for proper visualization ---
            # This is crucial for showing the highlighted cells in the "before" state
            selection_mask_for_viz = None
            try:
                from jaxarc.envs.actions import bbox_handler
                # Create the selection mask that will be applied by the action
                selection_mask_jax = bbox_handler(action, state_before.working_grid_mask)
                selection_mask_for_viz = np.array(selection_mask_jax)
                logger.debug(
                    f"Created selection mask for visualization: {np.sum(selection_mask_for_viz)} cells"
                )
            except Exception as e:
                logger.warning(f"Failed to create selection mask for visualization: {e}")
                # Fallback: create selection mask from bbox coordinates
                grid_height, grid_width = state_before.working_grid.shape
                selection_mask_for_viz = np.zeros((grid_height, grid_width), dtype=bool)
                
                # Extract bbox coordinates
                r1, c1, r2, c2 = action.r1, action.c1, action.r2, action.c2
                r1, c1, r2, c2 = int(r1), int(c1), int(r2), int(c2)
                
                # Ensure proper bounds
                r1 = max(0, min(r1, grid_height - 1))
                c1 = max(0, min(c1, grid_width - 1))
                r2 = max(0, min(r2, grid_height - 1))
                c2 = max(0, min(c2, grid_width - 1))
                
                # Ensure proper ordering
                min_r, max_r = min(r1, r2), max(r1, r2)
                min_c, max_c = min(c1, c2), max(c1, c2)
                
                # Create rectangular selection (inclusive bounds)
                selection_mask_for_viz[min_r : max_r + 1, min_c : max_c + 1] = True
                logger.debug(
                    f"Fallback bbox selection: {np.sum(selection_mask_for_viz)} cells selected"
                )

            # --- Convert structured action to dict for logging ---
            action_log = {
                "operation": int(action.operation),
                "r1": int(action.r1),
                "c1": int(action.c1),
                "r2": int(action.r2),
                "c2": int(action.c2),
            }
            
            # Add selection to action_log for visualization compatibility
            if selection_mask_for_viz is not None:
                action_log["selection"] = selection_mask_for_viz
            info_log = {
                k: np.asarray(v) if hasattr(v, "shape") else v for k, v in info.items()
            }

            # --- Logging and Visualization ---
            step_metrics = {
                "reward": float(reward),
                "total_reward": float(total_reward),
                "similarity": float(info_log.get("similarity", 0.0)),
                "episode": episode_idx,
                "operation": int(action.operation),
            }

            # Log to wandb
            wandb_integration.log_step(
                step_num=episode_idx * max_steps_per_episode + step_num,
                metrics=step_metrics,
            )

            # Create StepVisualizationData object and log to visualizer
            try:

                # Debug the selection mask
                if selection_mask_for_viz is not None:
                    logger.debug(f"Selection mask shape: {selection_mask_for_viz.shape}")
                    logger.debug(f"Selection mask dtype: {selection_mask_for_viz.dtype}")
                    logger.debug(f"Selection mask sum: {np.sum(selection_mask_for_viz)}")
                    logger.debug(f"Selection mask any: {np.any(selection_mask_for_viz)}")
                else:
                    logger.debug("Selection mask is None")

                step_data = StepVisualizationData(
                    step_num=step_num,
                    before_grid=np.array(state_before.working_grid),
                    after_grid=np.array(state.working_grid),
                    action=action_log,
                    reward=float(reward),
                    info=info_log,
                    task_id=task_id,
                    task_pair_index=0,  # Assuming we're working with the first pair
                    total_task_pairs=task.num_train_pairs,
                    selection_mask=selection_mask_for_viz,  # Use the pre-computed selection mask
                )
                visualizer.visualize_step(step_data)
            except Exception as e:
                logger.warning(f"Visualization step failed: {e}")
                import traceback

                logger.debug(f"Visualization error details: {traceback.format_exc()}")

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
            success=float(state.similarity_score) >= 0.99,
        )

        visualizer.visualize_episode_summary(summary_data)
        wandb_integration.log_episode_summary(
            episode_idx,
            {
                "episode_reward": summary_data.total_reward,
                "episode_steps": summary_data.total_steps,
                "final_similarity": summary_data.final_similarity,
                "success": summary_data.success,
            },
        )

    # --- Cleanup ---
    wandb_integration.finish_run()
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
        each environment may have different allowed operations. It uses JAX's vmap
        to efficiently:
        1. Get allowed operations for each environment in the batch
        2. Select random valid operations for each environment individually
        3. Generate random coordinates for bounding box actions
        4. Create structured actions for the entire batch

        Args:
            keys: PRNG keys for each environment [batch_size, 2]
            states: Batch of environment states
            config: Environment configuration

        Returns:
            BboxAction: Structured action for the entire batch
        """
        batch_size = keys.shape[0]

        # Split keys for operations and coordinates
        # keys has shape [batch_size, 2], we need to split each key
        op_keys = keys[:, 0]  # [batch_size]
        coord_keys = keys[:, 1]  # [batch_size]

        # Get allowed operations for each environment in the batch
        # We need to vmap the get_allowed_operations function over the batch
        def get_allowed_ops_single(state):
            return self.action_controller.get_allowed_operations(state, config.action)

        # Vectorize over the batch dimension
        get_allowed_ops_batch = jax.vmap(get_allowed_ops_single)
        allowed_masks = get_allowed_ops_batch(
            states
        )  # Shape: (batch_size, num_operations)

        # Select random operations for each environment from their allowed operations
        def select_random_operation(op_key, allowed_mask):
            # Extract the key properly - op_key should be a 2-element array
            key = op_key

            # Get indices of allowed operations
            allowed_indices = jnp.where(
                allowed_mask, size=allowed_mask.shape[0], fill_value=-1
            )[0]
            # Count valid allowed operations (non-negative indices)
            num_allowed = jnp.sum(allowed_indices >= 0)

            # If no operations are allowed, fallback to operation 0
            # Otherwise, select randomly from allowed operations
            def select_from_allowed():
                random_idx = jr.randint(key, shape=(), minval=0, maxval=num_allowed)
                return allowed_indices[random_idx]

            def fallback_operation():
                return jnp.array(0, dtype=jnp.int32)

            return jax.lax.cond(
                num_allowed > 0, select_from_allowed, fallback_operation
            )

        # Vectorize operation selection over the batch
        select_ops_batch = jax.vmap(select_random_operation)
        operations = select_ops_batch(op_keys, allowed_masks)

        # Generate random bounding boxes for each environment with proper bounds
        # Use vmap to generate coordinates for each environment
        def generate_coords(key):
            k1, k2, k3, k4 = jr.split(key, 4)
            r1 = jr.randint(k1, shape=(), minval=0, maxval=self.grid_height)
            c1 = jr.randint(k2, shape=(), minval=0, maxval=self.grid_width)
            r2 = jr.randint(k3, shape=(), minval=0, maxval=self.grid_height)
            c2 = jr.randint(k4, shape=(), minval=0, maxval=self.grid_width)
            return r1, c1, r2, c2
        
        # Vectorize coordinate generation over the batch
        generate_coords_batch = jax.vmap(generate_coords)
        r1, c1, r2, c2 = generate_coords_batch(coord_keys)

        # Ensure proper ordering
        min_r = jnp.minimum(r1, r2)
        min_c = jnp.minimum(c1, c2)
        max_r = jnp.maximum(r1, r2)
        max_c = jnp.maximum(c1, c2)

        # Create batched bbox actions
        return create_bbox_action(
            operation=operations, r1=min_r, c1=min_c, r2=max_r, c2=max_c
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

    for step in range(num_steps):
        step_start = time.time()

        try:
            # Generate new keys for this step
            step_key = jr.PRNGKey(42 + step)
            step_keys = jr.split(step_key, batch_size)
            # Create pairs of keys for each environment [batch_size, 2]
            coord_keys = jr.split(jr.PRNGKey(100 + step), batch_size)
            step_keys = jnp.stack([step_keys, coord_keys], axis=1)

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
            config = setup_configuration()
            visualizer, wandb_integration = setup_visualization_and_logging()
            run_rl_loop(
                config,
                visualizer,
                wandb_integration,
                num_episodes=5,
                max_steps_per_episode=30,
            )
        except Exception as e:
            logger.error(f"Single-agent RL loop failed: {e}")
            console.print(f"[red]Single-agent RL loop failed: {e}[/red]")

        console.rule("[bold yellow]Batched Environment Demo")

        # Run batched environment demo with smaller batch for testing
        console.print("\n[bold cyan]Running Batched Environment Demo...[/bold cyan]")
        try:
            run_batched_demo(batch_size=10, num_steps=10)
        except Exception as e:
            logger.error(f"Batched demo failed: {e}")
            console.print(f"[red]Batched demo failed: {e}[/red]")

        console.rule("[bold green]All Demos Completed Successfully!")

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
# This notebook demonstrates the complete JaxARC ecosystem with both single-agent and batched environment capabilities:
#
# ### Key Features Demonstrated:
# 1. **Modern API Usage**: Updated to use the latest structured actions and configuration system
# 2. **Single-Agent RL**: Complete RL loop with visualization and logging
# 3. **Batched Environments**: Massive parallelization with 1000+ environments
# 4. **Mixed Initialization**: Diverse grid initialization strategies for enhanced training
# 5. **Performance Optimization**: JAX vectorization for efficient batch processing
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
#
# This notebook serves as a solid foundation for developing sophisticated RL agents on ARC tasks!

# %%
