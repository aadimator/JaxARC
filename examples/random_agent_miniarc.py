"""
High-Performance JaxARC RL Loop using the PureJaxRL pattern.

This script demonstrates the optimal way to run RL experiments with JaxARC,
achieving maximum performance by JIT-compiling the entire training loop.
This serves as a blueprint for implementing advanced algorithms like PPO.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.actions import StructuredAction, create_bbox_action
from jaxarc.registration import make
from jaxarc.types import TimeStep
from jaxarc.utils.config import get_config

console = Console()


# ---
# 1. Configuration Setup (Same as before)
# ---
def setup_configuration() -> JaxArcConfig:
    """Loads and sets up the configuration for the RL loop."""
    logger.info("Setting up configuration for high-performance loop...")
    config_overrides = [
        "dataset=mini_arc",
        "action=raw",
        "action.selection_format=bbox",
        "wandb.enabled=false",
        "logging.log_operations=false",
        "logging.log_rewards=false",
        "visualization.enabled=false",
    ]
    hydra_config = get_config(overrides=config_overrides)

    console.print(
        Panel(
            f"[bold green]Configuration Loaded[/bold green]\n\n"
            f"Dataset: {hydra_config.dataset.dataset_name}\n"
            f"Action Format: {hydra_config.action.selection_format}",
            title="JaxARC Configuration",
            border_style="green",
        )
    )
    return JaxArcConfig.from_hydra(hydra_config)


# ---
# 2. Agent Definition (Pure Functional Style)
# ---
class AgentState(NamedTuple):
    """A simple state for our agent, just holding a PRNG key."""

    key: jax.Array


def random_agent_policy(
    params: None, obs: jax.Array, key: jax.Array, config: JaxArcConfig
) -> StructuredAction:
    """
    A pure function representing the policy of a random agent.
    This is JIT-compatible and can be used inside the main training loop.
    """
    # In a real agent, `params` would be the neural network weights.
    # `obs` would be the input to the network.
    _ = params, obs  # Unused for a random agent

    batch_size = obs.shape[0]
    h, w = config.dataset.max_grid_height, config.dataset.max_grid_width

    # Generate random parameters for the entire batch at once
    op_key, r1_key, c1_key, r2_key, c2_key = jr.split(key, 5)
    ops = jr.randint(op_key, shape=(batch_size,), minval=0, maxval=35)
    r1 = jr.randint(r1_key, shape=(batch_size,), minval=0, maxval=h)
    c1 = jr.randint(c1_key, shape=(batch_size,), minval=0, maxval=w)
    r2 = jr.randint(r2_key, shape=(batch_size,), minval=0, maxval=h)
    c2 = jr.randint(c2_key, shape=(batch_size,), minval=0, maxval=w)

    # Ensure r1 <= r2 and c1 <= c2 for the whole batch
    min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
    min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)

    # vmap the action creation function for efficiency
    return jax.vmap(create_bbox_action)(ops, min_r, min_c, max_r, max_c)


# ---
# 3. The PureJaxRL Training Loop Factory
# ---
def make_train(
    config: JaxArcConfig,
    env,
    env_params,
    num_envs: int,
    num_steps: int,
    num_updates: int,
):
    """
    A factory function that creates the single, JIT-compiled training function.
    This is the core of the PureJaxRL pattern.

    Notes:
    - The new registration-based API returns an (env, env_params) pair and the
      environment's core methods follow the TimeStep-based signature:
        timestep = env.reset(env_params, key)
        timestep = env.step(env_params, timestep, action)
    - We accept `env` and `env_params` here so the training factory simply closes
      over them and remains JIT-friendly.
    """
    # The environment and env_params are provided by the caller and closed over.

    def train(key: jax.Array):
        """
        The main training function. This entire function will be JIT-compiled.
        It contains the initialization and the main training loop (a scan).
        """

        # --- 1. INITIALIZATION ---
        # In a real agent, this is where you would initialize network parameters and optimizer state.
        # For our random agent, we don't have params or an optimizer.
        agent_params = None

        # Initialize the environments
        key, reset_key = jr.split(key)
        # New TimeStep-based API: reset returns a TimeStep object that embeds state+obs.
        # Support multiple parallel envs by vmapping reset when num_envs > 1.
        if num_envs > 1:
            reset_keys = jr.split(reset_key, num_envs)
            timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_keys)
        else:
            timesteps = env.reset(env_params, reset_key)
        # The `runner_state` is the collection of all states that change over the training loop.
        # We pack the whole TimeStep rather than separate env_state/obs tuples.
        runner_state = (agent_params, timesteps, key)

        # --- 2. THE TRAINING LOOP (as a scan) ---
        def _update_step(runner_state, _):
            """
            This function represents one update step of the RL algorithm (e.g., one PPO update).
            It contains the environment rollout and the agent learning step.
            """
            agent_params, timestep, key = runner_state

            # A. THE ROLLOUT PHASE
            def _env_step_body(carry, _):
                prev_timestep, key = carry
                key, action_key = jr.split(key)

                # Get actions from the agent's policy using the timestep.observation
                actions = random_agent_policy(
                    agent_params, prev_timestep.observation, action_key, config
                )

                # Step the environment using the TimeStep-based API (vectorized when requested)
                if num_envs > 1:
                    next_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, actions)
                else:
                    next_timestep = env.step(env_params, prev_timestep, actions)

                # In a real agent, you would store the full transition for learning.
                # For this random agent, we only care about the reward.
                return (next_timestep, key), next_timestep.reward

            # Run the rollout for a fixed number of steps using lax.scan
            key, rollout_key = jr.split(key)
            (final_timestep, _), collected_rewards = jax.lax.scan(
                _env_step_body, (timestep, rollout_key), None, length=num_steps
            )

            # B. THE AGENT UPDATE PHASE
            # In a real agent, you would use the `collected_transitions` to calculate the loss
            # and update the agent_params. For a random agent, this is a no-op.

            # Pack the state for the next update iteration (keep the final TimeStep)
            new_runner_state = (agent_params, final_timestep, key)

            # Return metrics from this update step
            metrics = {"mean_reward": jnp.mean(collected_rewards)}

            return new_runner_state, metrics

        # Run the entire training process using lax.scan
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=num_updates
        )

        return {"runner_state": runner_state, "metrics": metrics}

    # JIT-compile the entire train function. This is the magic!
    return jax.jit(train)


# ---
# 4. Main Execution Block
# ---
def main():
    """Main function to run the high-performance RL demo."""
    # Training parameters are now separate from the environment config
    num_envs = 4096
    num_steps = 128
    num_updates = 10

    config = setup_configuration()

    # --- Dataset Loading ---
    # Use the registration-based factory to construct an env and env_params for the chosen task.
    # Let the parser/registry handle buffering and EnvParams construction.
    # Pick a single available Mini task via the registry helper.
    from jaxarc.registration import available_task_ids

    available_ids = available_task_ids("Mini", config=config, auto_download=False)
    task_id = available_ids[0]
    env, env_params = make(f"Mini-{task_id}", config=config)

    console.rule("[bold yellow]JaxARC High-Performance Demo (PureJaxRL Style)")
    console.print(
        Panel(
            f"[bold cyan]Running with PureJaxRL pattern[/bold cyan]\n\n"
            f"Parallel Environments: {num_envs:,}\n"
            f"Steps per Rollout: {num_steps}\n"
            f"Total Training Steps: {num_envs * num_steps * num_updates:,}",
            title="Experiment Parameters",
            border_style="cyan",
        )
    )

    # --- Create and Compile the Training Function ---
    # Pass the constructed env and env_params into the training factory.
    train_fn = make_train(config, env, env_params, num_envs, num_steps, num_updates)

    # --- WARMUP (First call triggers JIT compilation) ---
    logger.info("Starting JIT compilation (this may take a moment)...")
    start_compile = time.time()
    key = jr.PRNGKey(42)
    output = train_fn(key)
    # Block until the compilation and first run are complete
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
    compile_time = time.time() - start_compile
    logger.info(f"JIT compilation finished in {compile_time:.2f}s")

    # --- TIMED RUN (Second call uses the compiled function) ---
    logger.info("Starting timed run...")
    start_run = time.time()
    key = jr.PRNGKey(43)  # Use a different key for the timed run
    output = train_fn(key)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
    run_time = time.time() - start_run

    # --- Performance Summary ---
    total_steps = num_envs * num_steps * num_updates
    sps = total_steps / run_time

    console.print(
        Panel(
            f"[bold green]Benchmark Results[/bold green]\n\n"
            f"Total Steps: {total_steps:,}\n"
            f"Execution Time: {run_time:.2f}s\n"
            f"Steps Per Second (SPS): [bold yellow]{sps:,.0f}[/bold yellow]",
            title="Performance Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
