#!/usr/bin/env python3
"""
Demo script for the MultiAgentPrimitiveArcEnv.

This script demonstrates the basic functionality of the primitive environment
including initialization, reset, step execution, and JAX compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from src.jaxarc.envs.primitive_env import MultiAgentPrimitiveArcEnv


def demo_basic_functionality():
    """Demonstrate basic environment functionality."""
    logger.info("🚀 Starting MultiAgentPrimitiveArcEnv demo")

    # Create environment configuration
    config = {
        "max_grid_size": [10, 10],
        "max_num_agents": 2,
        "max_episode_steps": 50,
        "max_program_length": 10,
        "max_action_params": 6,
        "reward": {
            "progress_weight": 1.0,
            "step_penalty": -0.01,
            "success_bonus": 10.0,
        }
    }

    # Initialize environment
    env = MultiAgentPrimitiveArcEnv(
        num_agents=2,
        config=config,
    )

    logger.info(f"📊 Environment initialized with {env.num_agents} agents")
    logger.info(f"🎯 Grid size: {env.max_grid_size}")
    logger.info(f"⏱️  Max episode steps: {env.max_episode_steps}")

    # Show action and observation space info
    agent_id = env.agents[0]
    action_space = env.action_spaces[agent_id]
    obs_space = env.observation_spaces[agent_id]

    logger.info(f"🎮 Action space shape: {action_space.shape}")
    logger.info(f"👁️  Observation space shape: {obs_space.shape}")

    return env


def demo_episode_execution(env: MultiAgentPrimitiveArcEnv):
    """Demonstrate running a complete episode."""
    logger.info("\n🎬 Starting episode execution demo")

    # Initialize PRNG key
    key = jax.random.PRNGKey(42)

    # Reset environment
    key, reset_key = jax.random.split(key)
    observations, state = env.reset(reset_key)

    logger.info(f"🔄 Environment reset. Initial step: {state.step}")
    logger.info(f"📍 Active training pair: {state.active_train_pair_idx}")
    logger.info(f"📚 Program length: {state.program_length}")

    # Run a few steps
    total_steps = 5
    for step_idx in range(total_steps):
        logger.info(f"\n--- Step {step_idx + 1}/{total_steps} ---")

        # Create random actions for all agents
        key, action_key = jax.random.split(key)
        actions = {}
        for agent_id in env.agents:
            action_shape = env.action_spaces[agent_id].shape
            # Generate random valid actions
            actions[agent_id] = jax.random.randint(
                action_key, action_shape, 0, 3, dtype=jnp.int32
            )

        # Execute step
        key, step_key = jax.random.split(key)
        next_obs, next_state, rewards, dones, info = env.step_env(
            step_key, state, actions
        )

        # Log step results
        logger.info(f"📈 Step: {next_state.step}")
        logger.info(f"🎯 Grid similarity: {float(info['grid_similarity']):.3f}")
        logger.info(f"🏃‍♂️ Active agents: {int(info['active_agents'])}")
        logger.info(f"📋 Program length: {int(info['program_length'])}")
        logger.info(f"🏁 Episode done: {bool(info['episode_done'])}")

        # Log rewards
        for agent_id, reward in rewards.items():
            logger.info(f"💰 {agent_id} reward: {float(reward):.3f}")

        # Update state for next iteration
        state = next_state

        # Check if episode is done
        if dones["__all__"]:
            logger.info("🎊 Episode completed!")
            break

    return state


def demo_jax_compatibility(env: MultiAgentPrimitiveArcEnv):
    """Demonstrate JAX transformations compatibility."""
    logger.info("\n⚡ Testing JAX compatibility")

    # Test JIT compilation
    logger.info("🔧 Testing JIT compilation...")
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step_env)

    key = jax.random.PRNGKey(123)

    # JIT compiled reset
    key, reset_key = jax.random.split(key)
    observations, state = jitted_reset(reset_key)
    logger.info("✅ JIT reset successful")

    # JIT compiled step
    actions = {}
    for agent_id in env.agents:
        action_shape = env.action_spaces[agent_id].shape
        actions[agent_id] = jnp.zeros(action_shape, dtype=jnp.int32)

    key, step_key = jax.random.split(key)
    next_obs, next_state, rewards, dones, info = jitted_step(step_key, state, actions)
    logger.info("✅ JIT step successful")

    # Test vectorization capability
    logger.info("🔄 Testing vectorization compatibility...")
    batch_size = 4
    keys = jax.random.split(key, batch_size)

    # Note: For full vmap, we'd need to modify the environment slightly
    # For now, just show that individual components are vmap-ready
    grid_similarities = jax.vmap(
        lambda s: env._calculate_grid_similarity(s.current_grid, s.target_grid)
    )

    # Create a batch of states (simplified)
    batched_grids = jnp.stack([state.working_grid] * batch_size)
    # Get target from task data
    target_grid = state.task_data.output_grids_examples[state.active_train_pair_idx]
    batched_targets = jnp.stack([target_grid] * batch_size)

    similarities = jax.vmap(env._calculate_grid_similarity)(batched_grids, batched_targets)
    logger.info(f"✅ Vectorized similarity calculation: {similarities}")


def demo_action_types():
    """Demonstrate different action types and their usage."""
    logger.info("\n🎮 Action Types Demo")

    from src.jaxarc.types import PrimitiveType, ControlType, ActionCategory

    logger.info("Available primitive types:")
    for prim_type in PrimitiveType:
        logger.info(f"  - {prim_type.name}: {prim_type.value}")

    logger.info("Available control types:")
    for ctrl_type in ControlType:
        logger.info(f"  - {ctrl_type.name}: {ctrl_type.value}")

    logger.info("Action categories:")
    for category in ActionCategory:
        logger.info(f"  - {category.name}: {category.value}")

    # Example action construction
    logger.info("\n📝 Example action format:")
    logger.info("Action array: [category, primitive_type, control_type, param1, param2, ...]")
    logger.info("- Draw pixel at (5,3) with color 2:")
    logger.info("  [PRIMITIVE, DRAW_PIXEL, 0, 5, 3, 2, 0, ...]")
    logger.info("- Submit solution:")
    logger.info("  [CONTROL, 0, SUBMIT, 0, 0, 0, 0, ...]")


def main():
    """Main demo function."""
    logger.info("🎯 JaxARC Primitive Environment Demo")
    logger.info("=" * 50)

    try:
        # Demo 1: Basic functionality
        env = demo_basic_functionality()

        # Demo 2: Episode execution
        final_state = demo_episode_execution(env)

        # Demo 3: JAX compatibility
        demo_jax_compatibility(env)

        # Demo 4: Action types
        demo_action_types()

        logger.info("\n🎉 Demo completed successfully!")
        logger.info("📚 Next steps:")
        logger.info("  1. Implement primitive operations (draw_pixel, draw_line, etc.)")
        logger.info("  2. Add agent collaboration mechanisms")
        logger.info("  3. Implement reward shaping")
        logger.info("  4. Add visualization capabilities")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
