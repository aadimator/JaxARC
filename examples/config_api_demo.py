#!/usr/bin/env python3
"""
Config-Based API Demo for JaxARC

This example demonstrates the new config-based architecture for JaxARC,
showing how to use typed configurations, functional API, and Hydra integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import OmegaConf

from jaxarc.envs import (
    ActionConfig,
    # Config classes
    ArcEnvConfig,
    GridConfig,
    RewardConfig,
    # Functional API
    arc_reset,
    arc_step,
    create_bbox_config,
    create_point_config,
    # Factory functions
    create_standard_config,
    create_training_config,
    get_preset_config,
)


def demo_basic_usage():
    """Demonstrate basic usage of the config-based API."""
    logger.info("=== Basic Usage Demo ===")

    # Create a standard configuration
    config = create_standard_config(
        max_episode_steps=50,
        reward_on_submit_only=False,
        success_bonus=15.0,
        log_operations=True,
    )

    logger.info(f"Created config with max steps: {config.max_episode_steps}")
    logger.info(f"Selection format: {config.action.selection_format}")

    # Initialize environment
    key = jax.random.PRNGKey(42)
    state, observation = arc_reset(key, config)

    logger.info(f"Initial grid shape: {observation.shape}")
    logger.info(f"Initial similarity: {state.similarity_score:.3f}")

    # Take a few steps
    for step in range(3):
        # Create a simple action (fill selection with color 1)
        action = {
            "selection": jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
            .at[2:5, 2:5]
            .set(True),
            "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
        }

        state, observation, reward, done, info = arc_step(state, action, config)

        logger.info(
            f"Step {step + 1}: reward={reward:.3f}, similarity={info['similarity']:.3f}"
        )

        if done:
            logger.info("Episode terminated!")
            break


def demo_hydra_integration():
    """Demonstrate Hydra configuration integration."""
    logger.info("\n=== Hydra Integration Demo ===")

    # Create a Hydra-style configuration
    hydra_config = OmegaConf.create(
        {
            "max_episode_steps": 75,
            "log_operations": True,
            "reward": {
                "reward_on_submit_only": True,
                "step_penalty": -0.02,
                "success_bonus": 20.0,
            },
            "grid": {
                "max_grid_height": 25,
                "max_grid_width": 25,
            },
            "action": {
                "selection_format": "mask",
                "num_operations": 30,
            },
        }
    )

    # Convert to typed config
    config = ArcEnvConfig.from_hydra(hydra_config)

    logger.info(f"Hydra config converted - max steps: {config.max_episode_steps}")
    logger.info(f"Grid size: {config.grid.max_grid_size}")

    # Use with functional API
    key = jax.random.PRNGKey(123)
    state, observation = arc_reset(key, hydra_config)  # Can pass DictConfig directly!

    logger.info(f"Reset with Hydra config - grid shape: {observation.shape}")


def demo_action_formats():
    """Demonstrate different action formats."""
    logger.info("\n=== Action Formats Demo ===")

    # 1. Selection-operation format (default)
    logger.info("1. Selection-Operation Format:")
    config = create_standard_config(max_episode_steps=20)
    key = jax.random.PRNGKey(456)
    state, obs = arc_reset(key, config)

    action = {
        "selection": jnp.ones_like(
            state.working_grid, dtype=jnp.bool_
        ),  # Select entire grid
        "operation": jnp.array(2, dtype=jnp.int32),  # Fill with color 2
    }
    state, obs, reward, done, info = arc_step(state, action, config)
    logger.info(f"  Reward: {reward:.3f}")

    # 2. Point-based format
    logger.info("2. Point-Based Format:")
    point_config = create_point_config(max_episode_steps=20)
    state, obs = arc_reset(key, point_config)

    point_action = {
        "point": (3, 4),  # Select single point at (3, 4)
        "operation": jnp.array(3, dtype=jnp.int32),  # Fill with color 3
    }
    state, obs, reward, done, info = arc_step(state, point_action, point_config)
    logger.info(f"  Reward: {reward:.3f}")

    # 3. Bounding box format
    logger.info("3. Bounding Box Format:")
    bbox_config = create_bbox_config(max_episode_steps=20)
    state, obs = arc_reset(key, bbox_config)

    bbox_action = {
        "bbox": (1, 1, 4, 4),  # Select rectangular region
        "operation": jnp.array(4, dtype=jnp.int32),  # Fill with color 4
    }
    state, obs, reward, done, info = arc_step(state, bbox_action, bbox_config)
    logger.info(f"  Reward: {reward:.3f}")


def demo_jax_compatibility():
    """Demonstrate JAX transformations with the new API."""
    logger.info("\n=== JAX Compatibility Demo ===")

    config = create_standard_config(max_episode_steps=10)

    # JIT compilation with static config
    @jax.jit
    def jitted_reset(key):
        return arc_reset(key, config)

    @jax.jit
    def jitted_step(state, action):
        return arc_step(state, action, config)

    # Use JIT-compiled functions
    key = jax.random.PRNGKey(789)
    state, obs = jitted_reset(key)
    logger.info(f"JIT reset successful - similarity: {state.similarity_score:.3f}")

    action = {
        "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
        "operation": jnp.array(5, dtype=jnp.int32),
    }

    state, obs, reward, done, info = jitted_step(state, action)
    logger.info(f"JIT step successful - reward: {reward:.3f}")

    # Batch processing with vmap
    def single_episode(key):
        state, obs = arc_reset(key, config)
        action = {
            "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
            "operation": jnp.array(0, dtype=jnp.int32),
        }
        state, obs, reward, done, info = arc_step(state, action, config)
        return reward

    # Process multiple episodes in parallel
    keys = jax.random.split(key, 5)
    batch_rewards = jax.vmap(single_episode)(keys)
    logger.info(f"Batch processing successful - rewards: {batch_rewards}")


def demo_preset_configs():
    """Demonstrate preset configuration usage."""
    logger.info("\n=== Preset Configs Demo ===")

    # Different preset configurations
    presets = ["raw", "standard", "full", "point", "bbox"]

    for preset_name in presets:
        config = get_preset_config(preset_name, max_episode_steps=30)
        logger.info(
            f"{preset_name.capitalize()} config: "
            f"selection_format={config.action.selection_format}, "
            f"validation={config.strict_validation}"
        )

    # Training presets for curriculum learning
    training_levels = ["basic", "standard", "advanced", "expert"]

    for level in training_levels:
        config = create_training_config(level, max_episode_steps=25)
        logger.info(
            f"{level.capitalize()} training: "
            f"max_steps={config.max_episode_steps}, "
            f"reward_on_submit={config.reward.reward_on_submit_only}"
        )


def demo_manual_config_creation():
    """Demonstrate manual configuration creation."""
    logger.info("\n=== Manual Config Creation Demo ===")

    # Create custom reward configuration
    custom_reward = RewardConfig(
        reward_on_submit_only=False,
        step_penalty=-0.005,
        success_bonus=25.0,
        similarity_weight=2.0,
        progress_bonus=0.5,
    )

    # Create custom grid configuration
    custom_grid = GridConfig(
        max_grid_height=20,
        max_grid_width=20,
        min_grid_height=5,
        min_grid_width=5,
        max_colors=8,
        background_color=0,
    )

    # Create custom action configuration
    custom_action = ActionConfig(
        selection_format="mask",
        selection_threshold=0.7,
        allow_partial_selection=False,
        num_operations=25,
        validate_actions=True,
        clip_invalid_actions=True,
    )

    # Combine into full configuration
    custom_config = ArcEnvConfig(
        max_episode_steps=60,
        auto_reset=True,
        log_operations=True,
        log_grid_changes=True,
        log_rewards=True,
        strict_validation=True,
        allow_invalid_actions=False,
        reward=custom_reward,
        grid=custom_grid,
        action=custom_action,
    )

    logger.info(f"Custom config created with {custom_config.grid.max_colors} colors")
    logger.info(f"Success bonus: {custom_config.reward.success_bonus}")

    # Test the custom configuration
    key = jax.random.PRNGKey(999)
    state, obs = arc_reset(key, custom_config)
    logger.info(f"Custom config works - grid shape: {obs.shape}")


def main():
    """Run all demos."""
    logger.info("Starting JaxARC Config-Based API Demo")

    try:
        demo_basic_usage()
        demo_hydra_integration()
        demo_action_formats()
        demo_jax_compatibility()
        demo_preset_configs()
        demo_manual_config_creation()

        logger.info("\n🎉 All demos completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
