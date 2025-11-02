from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from jaxarc.configs import JaxArcConfig
from jaxarc.registration import make
from jaxarc.utils.core import get_config

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
            f"Action Format: point wrapper (dict-based)",
            title="JaxARC Configuration",
            border_style="green",
        )
    )
    return JaxArcConfig.from_hydra(hydra_config)


# ---
# 3. The PureJaxRL Training Loop Factory
# ---
def make_train(
    env,
    env_params,
    num_envs: int,
    num_steps: int,
    num_updates: int,
):
    """
    A factory function that creates the single, JIT-compiled training function.
    This is the core of the PureJaxRL pattern.

    """
    # Get action space for sampling
    action_space = env.action_space(env_params)

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
        # New TimeStep-based API: reset returns a (State, TimeStep) tuple.
        # Support multiple parallel envs by vmapping reset when num_envs > 1.
        if num_envs > 1:
            reset_keys = jr.split(reset_key, num_envs)
            states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(
                reset_keys, env_params
            )
        else:
            states, timesteps = env.reset(reset_key, env_params=env_params)
        # The `runner_state` is the collection of all states that change over the training loop.
        runner_state = (agent_params, states, timesteps, key)

        # --- 2. THE TRAINING LOOP (as a scan) ---
        def _update_step(runner_state, _):
            """
            This function represents one update step of the RL algorithm (e.g., one PPO update).
            It contains the environment rollout and the agent learning step.
            """
            agent_params, states, timesteps, key = runner_state

            # A. THE ROLLOUT PHASE
            def _env_step_body(carry, _):
                prev_states, _, key = carry
                key, action_key = jr.split(key)

                # Get actions by directly sampling from action space
                if num_envs > 1:
                    # Batch action sampling
                    action_keys = jr.split(action_key, num_envs)
                    actions = jax.vmap(action_space.sample)(action_keys)
                else:
                    actions = action_space.sample(action_key)

                # Step the environment using the new API (vectorized when requested)
                if num_envs > 1:
                    next_states, next_timesteps = jax.vmap(
                        env.step, in_axes=(0, 0, None)
                    )(prev_states, actions, env_params)
                else:
                    next_states, next_timesteps = env.step(
                        prev_states, actions, env_params=env_params
                    )

                # In a real agent, you would store the full transition for learning.
                # For this random agent, we only care about the reward.
                return (next_states, next_timesteps, key), next_timesteps.reward

            # Run the rollout for a fixed number of steps using lax.scan
            key, rollout_key = jr.split(key)
            ((final_states, final_timesteps, _), collected_rewards) = jax.lax.scan(
                _env_step_body, (states, timesteps, rollout_key), None, length=num_steps
            )

            # B. THE AGENT UPDATE PHASE
            # In a real agent, you would use the `collected_transitions` to calculate the loss
            # and update the agent_params. For a random agent, this is a no-op.

            # Pack the state for the next update iteration
            new_runner_state = (agent_params, final_states, final_timesteps, key)

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

    # ---
    # Dataset Loading
    # ---
    # Use the registration-based factory to construct an env and env_params for the chosen task.
    # Let the parser/registry handle buffering and EnvParams construction.
    # Pick a single available Mini task via the registry helper.
    from jaxarc.envs import PointActionWrapper
    from jaxarc.registration import available_task_ids

    available_ids = available_task_ids("Mini", config=config, auto_download=False)
    task_id = available_ids[0]
    env, env_params = make(f"Mini-{task_id}", config=config)

    # Wrap with PointActionWrapper to handle dict<->Action conversion automatically
    env = PointActionWrapper(env)

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

    # ---
    # Create and Compile the Training Function
    # ---
    # Pass the constructed env and env_params into the training factory.
    train_fn = make_train(env, env_params, num_envs, num_steps, num_updates)

    # ---
    # WARMUP (First call triggers JIT compilation)
    # ---
    logger.info("Starting JIT compilation (this may take a moment)...")
    start_compile = time.time()
    key = jr.PRNGKey(42)
    output = train_fn(key)
    # Block until the compilation and first run are complete
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
    compile_time = time.time() - start_compile
    logger.info(f"JIT compilation finished in {compile_time:.2f}s")

    # ---
    # TIMED RUN (Second call uses the compiled function)
    # ---
    logger.info("Starting timed run...")
    start_run = time.time()
    key = jr.PRNGKey(43)  # Use a different key for the timed run
    output = train_fn(key)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
    run_time = time.time() - start_run

    # ---
    # Performance Summary
    # ---
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
