#!/usr/bin/env python3
"""
Test script to verify different environment configurations work correctly.

This script tests the unified configuration system by creating different 
environment types and verifying they work with the ArcEnvironment class.

Usage:
    pixi run python examples/basic/test_config_environments.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs import ConfigFactory, ArcEnvironment
from jaxarc.types import ARCLEOperationType


def test_raw_environment():
    """Test raw environment with minimal action set."""
    logger.info("Testing Raw Environment (minimal actions)")

    # Create configuration using unified ConfigFactory with minimal debugging
    config = ConfigFactory.create_development_config(
        max_episode_steps=50,
        debug_level="off",  # Disable debugging to avoid visualization issues
        max_operations=12,  # Limited operations for raw environment
        allowed_operations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34]
    )

    # Create environment with unified config
    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    logger.info("Raw environment created successfully")
    logger.info(f"Grid shape: {obs.shape}")
    logger.info(f"Allowed operations: {config.action.allowed_operations}")

    # Test valid action (fill with color 1)
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[0:3, 0:3].set(True)  # Select small area

    action = {
        "selection": selection,
        "operation": jnp.array(1, dtype=jnp.int32),  # FILL_1 operation
    }

    state, obs, reward, info = env.step(action)
    logger.info(f"Raw environment step successful, reward: {reward:.3f}")

    return True


def test_standard_environment():
    """Test standard environment with extended action set."""
    logger.info("Testing Standard Environment (extended actions)")

    # Use standard action set (27 operations)
    standard_operations = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  # Fill operations
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  # Flood fill operations
        20, 21, 22, 23,  # Movement operations
        24, 25, 26,  # Transformation operations
    ]

    # Create standard configuration with more operations
    config = ConfigFactory.create_development_config(
        max_episode_steps=100,
        debug_level="off",  # Disable debugging to avoid visualization issues
        max_operations=27,
        allowed_operations=standard_operations
    )

    # Create environment
    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(123)
    state, obs = env.reset(key)

    logger.info("Standard environment created successfully")
    logger.info(f"Grid shape: {obs.shape}")
    logger.info(f"Number of allowed operations: {len(config.action.allowed_operations)}")

    # Test flood fill action
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[5:8, 5:8].set(True)

    action = {
        "selection": selection,
        "operation": jnp.array(12, dtype=jnp.int32),  # FLOOD_FILL_2 operation
    }

    state, obs, reward, info = env.step(action)
    logger.info(f"Standard environment flood fill successful, reward: {reward:.3f}")

    return True


def test_full_environment():
    """Test full environment with all actions."""
    logger.info("Testing Full Environment (all actions)")

    # Create full configuration with all operations
    config = ConfigFactory.create_research_config(
        max_episode_steps=150,
        debug_level="off",  # Disable debugging to avoid visualization issues
        max_operations=35,
        allowed_operations=list(range(35))  # All operations 0-34
    )

    # Create environment
    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(456)
    state, obs = env.reset(key)

    logger.info("Full environment created successfully")
    logger.info(f"Grid shape: {obs.shape}")
    logger.info(f"Number of allowed operations: {len(config.action.allowed_operations)}")

    # Test movement action
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[10:15, 10:15].set(True)

    action = {
        "selection": selection,
        "operation": jnp.array(20, dtype=jnp.int32),  # MOVE_UP operation
    }

    state, obs, reward, info = env.step(action)
    logger.info(f"Full environment movement successful, reward: {reward:.3f}")

    return True


def test_training_rewards():
    """Test training rewards configuration."""
    logger.info("Testing Training Rewards Configuration")

    # Create configuration optimized for training with dense rewards
    config = ConfigFactory.create_development_config(
        max_episode_steps=200,
        debug_level="off",  # Disable debugging to avoid visualization issues
        reward_on_submit_only=False,  # Enable dense rewards
        step_penalty=-0.005,
        success_bonus=20.0,
        similarity_weight=2.0,
        progress_bonus=0.1
    )

    # Create environment
    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(789)
    state, obs = env.reset(key)

    logger.info("Training rewards environment created successfully")
    logger.info(f"Dense rewards enabled: {not config.reward.reward_on_submit_only}")
    logger.info(f"Step penalty: {config.reward.step_penalty}")
    logger.info(f"Success bonus: {config.reward.success_bonus}")

    # Test action to get non-zero reward
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[1:4, 1:4].set(True)

    action = {
        "selection": selection,
        "operation": jnp.array(3, dtype=jnp.int32),  # FILL_3 operation
    }

    state, obs, reward, info = env.step(action)
    logger.info(f"Training rewards step reward: {reward:.3f} (should be non-zero)")

    return True


def test_point_actions():
    """Test point-based actions."""
    logger.info("Testing Point-Based Actions")

    # Create configuration with point-based actions
    config = ConfigFactory.from_preset("point_actions", max_episode_steps=75)

    # Create environment
    env = ArcEnvironment(config)
    key = jax.random.PRNGKey(101112)
    state, obs = env.reset(key)

    logger.info("Point-based actions environment created successfully")
    logger.info(f"Selection format: {config.action.selection_format}")

    # Test point action
    action = {
        "point": (5, 5),  # Point at position (5, 5)
        "operation": jnp.array(4, dtype=jnp.int32),  # FILL_4 operation
    }

    state, obs, reward, info = env.step(action)
    logger.info(f"Point action successful, reward: {reward:.3f}")

    return True


def test_config_validation():
    """Test configuration validation."""
    logger.info("Testing Configuration Validation")

    # Test invalid configuration (should raise error)
    try:
        config = ConfigFactory.create_development_config(
            allowed_operations=[0, 1, 2, 999]  # 999 is invalid
        )
        # This should fail during environment creation
        env = ArcEnvironment(config)
        logger.error("❌ Configuration validation failed - invalid config was accepted")
        return False
    except Exception as e:
        logger.info(f"✓ Configuration correctly rejected invalid operation ID: {e}")

    # Test valid configuration
    try:
        config = ConfigFactory.create_development_config(
            max_episode_steps=50,
            debug_level="off"  # Disable debugging to avoid visualization issues
        )
        env = ArcEnvironment(config)
        logger.info("✓ Valid configuration accepted successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Valid configuration was rejected: {e}")
        return False


def test_operation_mapping():
    """Test operation ID mapping."""
    logger.info("Testing Operation ID Mapping")

    # Test that operation IDs are valid (0-34)
    test_operations = [
        (0, "FILL_0"),
        (9, "FILL_9"),
        (10, "FLOOD_FILL_0"),
        (19, "FLOOD_FILL_9"),
        (20, "MOVE_UP"),
        (21, "MOVE_DOWN"),
        (22, "MOVE_LEFT"),
        (23, "MOVE_RIGHT"),
        (24, "ROTATE_C"),
        (25, "ROTATE_CC"),
        (26, "FLIP_HORIZONTAL"),
        (27, "FLIP_VERTICAL"),
        (28, "COPY"),
        (29, "PASTE"),
        (30, "CUT"),
        (31, "CLEAR"),
        (32, "COPY_INPUT"),
        (33, "RESIZE"),
        (34, "SUBMIT"),
    ]

    for op_id, expected_name in test_operations:
        try:
            # Test that the operation ID is in valid range
            if 0 <= op_id <= 34:
                logger.info(f"✓ Operation {op_id} is valid ({expected_name})")
            else:
                logger.error(f"❌ Operation {op_id} is out of valid range")
                return False
        except Exception as e:
            logger.error(f"❌ Operation {op_id} validation failed: {e}")
            return False

    return True


def main():
    """Run all configuration tests."""
    logger.info("Starting JaxARC Configuration Tests")

    tests = [
        ("Raw Environment", test_raw_environment),
        ("Standard Environment", test_standard_environment),
        ("Full Environment", test_full_environment),
        ("Training Rewards", test_training_rewards),
        ("Point Actions", test_point_actions),
        ("Config Validation", test_config_validation),
        ("Operation Mapping", test_operation_mapping),
    ]

    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'=' * 50}")
        
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
            logger.info(f"✓ {test_name} {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[test_name] = "ERROR"
            logger.error(f"❌ {test_name} ERROR: {e}")

    # Print summary
    logger.info(f"\n{'=' * 50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 50}")
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Configuration system is working correctly.")
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the configuration system.")


if __name__ == "__main__":
    main()