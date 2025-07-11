#!/usr/bin/env python3
"""Demo script showing the clean class-based API for JaxARC environments.

This script demonstrates how to use the streamlined ArcEnvironment class
with the functional configuration system.
"""

import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs import ArcEnvironment, ArcEnvConfig, ActionConfig, GridConfig, RewardConfig


def main():
    """Demonstrate the clean class-based API."""
    logger.info("🎯 JaxARC Clean API Demo")

    # 1. Create configuration using the typed config system
    config = ArcEnvConfig(
        max_episode_steps=20,
        log_operations=True,
        log_grid_changes=True,
        reward=RewardConfig(
            success_bonus=100.0,
            step_penalty=-0.1,
            progress_bonus=5.0,
            reward_on_submit_only=False
        ),
        grid=GridConfig(
            max_grid_height=10,
            max_grid_width=10,
            max_colors=5
        ),
        action=ActionConfig(
            action_format="selection_operation",
            num_operations=5
        )
    )

    logger.info(f"📋 Configuration: {config}")

    # 2. Create environment with config
    env = ArcEnvironment(config)

    # 3. Get space information
    obs_info = env.get_observation_space_info()
    action_info = env.get_action_space_info()

    logger.info(f"👁️  Observation space: {obs_info}")
    logger.info(f"🎮 Action space: {action_info}")

    # 4. Reset environment
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    logger.info(f"🔄 Reset complete. State: step={state.step_count}, done={state.episode_done}")
    logger.info(f"📊 Initial observation shape: {obs.shape}")

    # 5. Take some steps
    for step in range(3):
        if env.is_done:
            logger.info("🏁 Episode terminated early")
            break

        # Create a simple action (select a 2x2 area and copy)
        action = {
            "selection": jnp.array([1, 1, 2, 2]),  # Select 2x2 area
            "operation": 0  # Copy operation
        }

        logger.info(f"🚀 Step {step + 1}: Taking action {action}")

        try:
            next_state, next_obs, reward, info = env.step(action)

            logger.info(f"✅ Step {step + 1} complete:")
            logger.info(f"   • Step count: {next_state.step_count}")
            logger.info(f"   • Reward: {reward}")
            logger.info(f"   • Done: {next_state.episode_done}")
            logger.info(f"   • Info: {info}")

        except Exception as e:
            logger.warning(f"⚠️  Step {step + 1} failed: {e}")
            break

    # 6. Show final state
    logger.info(f"🎯 Final state: step={env.state.step_count}, done={env.is_done}")
    logger.info("✨ Demo complete!")


def demo_different_action_formats():
    """Demo different action formats."""
    logger.info("\n🎮 Demo: Different Action Formats")

    formats = [
        ("selection_operation", {"selection": jnp.array([0, 0, 1, 1]), "operation": 0}),
        ("point", {"point": jnp.array([1, 1]), "operation": 0}),
        ("bbox", {"bbox": jnp.array([0, 0, 1, 1]), "operation": 0})
    ]

    for action_format, sample_action in formats:
        logger.info(f"\n🔧 Testing action format: {action_format}")

        config = ArcEnvConfig(
            max_episode_steps=5,
            action=ActionConfig(action_format=action_format, num_operations=3)
        )

        env = ArcEnvironment(config)
        action_info = env.get_action_space_info()

        logger.info(f"   Action space: {action_info}")
        logger.info(f"   Sample action: {sample_action}")

        # Quick test
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        try:
            next_state, next_obs, reward, info = env.step(sample_action)
            logger.info(f"   ✅ Action executed successfully, reward: {reward}")
        except Exception as e:
            logger.warning(f"   ⚠️  Action failed: {e}")


def demo_hydra_integration():
    """Demo integration with Hydra configs."""
    logger.info("\n⚙️  Demo: Hydra Integration")

    # You can also create configs from Hydra
    from jaxarc.envs.factory import create_standard_config

    config = create_standard_config()
    logger.info(f"📋 Standard config created: max_steps={config.max_episode_steps}")

    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    logger.info(f"✅ Environment with Hydra config works: {obs.shape}")


if __name__ == "__main__":
    main()
    demo_different_action_formats()
    demo_hydra_integration()
