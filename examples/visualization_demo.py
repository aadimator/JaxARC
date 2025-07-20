"""
Demonstration of enhanced visualization capabilities.

This script runs a short episode of the ARC environment with random actions
and saves a visualization of each step to an SVG file.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.utils.config import get_config


def create_random_action(key: jr.PRNGKey, config: ArcEnvConfig) -> dict:
    """Creates a random action based on the environment configuration."""
    op_key, sel_key = jr.split(key)

    # Create a random operation
    op = jr.randint(op_key, shape=(), minval=0, maxval=config.action.num_operations)

    # Create a random selection based on the selection format
    if config.action.selection_format == "bbox":
        h, w = config.grid.max_grid_height, config.grid.max_grid_width
        r1, c1, r2, c2 = jr.randint(sel_key, shape=(4,), minval=0, maxval=max(h, w))
        selection = jnp.array([r1, c1, r2, c2])
    elif config.action.selection_format == "point":
        h, w = config.grid.max_grid_height, config.grid.max_grid_width
        r, c = jr.randint(sel_key, shape=(2,), minval=0, maxval=jnp.array([h, w]))
        selection = jnp.array([r, c])
    else:  # Default to mask
        h, w = config.grid.max_grid_height, config.grid.max_grid_width
        selection = jr.bernoulli(sel_key, p=0.1, shape=(h, w))

    return {"selection": selection, "operation": op}


def main():
    """Main function to run the visualization demo."""
    # Load configuration with visualization enabled and action format set to bbox
    # for simpler random action generation.
    config = ArcEnvConfig.from_hydra(
        get_config(
            overrides=["debug.log_rl_steps=true", "action.selection_format=bbox"]
        )
    )

    # Convert to unified config and create environment
    from jaxarc.envs.equinox_config import convert_arc_env_config_to_jax_arc_config
    unified_config = convert_arc_env_config_to_jax_arc_config(config)
    env = ArcEnvironment(unified_config)

    # Run a short episode with visualization
    # We don't pass task_data to reset, so it uses the internal demo task.
    key = jr.PRNGKey(0)
    state, obs = env.reset(key)

    print(
        f"Running episode and saving visualizations to {config.debug.rl_steps_output_dir}..."
    )

    for i in range(5):
        # Take a random action
        action_key, key = jr.split(key)
        action = create_random_action(action_key, config)
        state, obs, reward, info = env.step(action)

        if env.is_done:
            print(f"Episode finished early at step {i + 1}.")
            break

    print(
        f"\nEpisode completed! Check the directory '{config.debug.rl_steps_output_dir}' for the visualization files."
    )
    print(
        "You can open the 'step_*.svg' files in a web browser to see the visualizations."
    )


if __name__ == "__main__":
    main()
