#!/usr/bin/env python3
"""
Test script to verify different environment configurations work correctly.

This script tests the new structured configuration system by creating
different environment types and verifying they work with the functional API.

Usage:
    python examples/test_config_environments.py
"""

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from loguru import logger

from jaxarc.envs.factory import create_config_from_hydra, create_complete_hydra_config
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.types import ARCLEOperationType


def test_raw_environment():
    """Test raw environment with minimal action set."""
    logger.info("Testing Raw Environment (minimal actions)")

    config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 50,
            "log_operations": False,
            "reward": {
                "reward_on_submit_only": True,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
                "similarity_weight": 1.0,
                "progress_bonus": 0.1,
                "invalid_action_penalty": -0.5
            },
            "action": {
                "action_format": "selection_operation",
                "selection_threshold": 0.5,
                "allow_partial_selection": True,
                "num_operations": 35,
                "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34],
                "validate_actions": True,
                "clip_invalid_actions": False
            }
        }
    })

    env_config = create_config_from_hydra(config.environment)

    # Test environment creation
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Raw environment created successfully")
    logger.info(f"Grid shape: {state.working_grid.shape}")
    logger.info(f"Allowed operations: {env_config.action.allowed_operations}")

    # Test valid action (fill with color 1)
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[0:3, 0:3].set(True)  # Select small area

    action = {
        "selection": selection,
        "operation": jnp.array(ARCLEOperationType.FILL_1, dtype=jnp.int32)
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Raw environment step successful, reward: {reward:.3f}")

    return True


def test_standard_environment():
    """Test standard environment with extended action set."""
    logger.info("Testing Standard Environment (extended actions)")

    config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 100,
            "log_operations": False,
            "reward": {
                "reward_on_submit_only": True,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
                "similarity_weight": 1.0,
                "progress_bonus": 0.1,
                "invalid_action_penalty": -0.5
            },
            "action": {
                "action_format": "selection_operation",
                "selection_threshold": 0.5,
                "allow_partial_selection": True,
                "num_operations": 35,
                "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 32, 33, 34],
                "validate_actions": True,
                "clip_invalid_actions": False
            }
        }
    })

    env_config = create_config_from_hydra(config.environment)

    # Test environment creation
    key = jax.random.PRNGKey(123)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Standard environment created successfully")
    logger.info(f"Grid shape: {state.working_grid.shape}")
    logger.info(f"Number of allowed operations: {len(env_config.action.allowed_operations)}")

    # Test flood fill operation (not available in raw)
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[2, 2].set(True)  # Select single cell for flood fill

    action = {
        "selection": selection,
        "operation": jnp.array(ARCLEOperationType.FLOOD_FILL_2, dtype=jnp.int32)
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Standard environment flood fill successful, reward: {reward:.3f}")

    return True


def test_full_environment():
    """Test full environment with all actions including object operations."""
    logger.info("Testing Full Environment (all actions)")

    config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 200,
            "log_operations": False,
            "reward": {
                "reward_on_submit_only": True,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
                "similarity_weight": 1.0,
                "progress_bonus": 0.1,
                "invalid_action_penalty": -0.5
            },
            "action": {
                "action_format": "selection_operation",
                "selection_threshold": 0.5,
                "allow_partial_selection": True,
                "num_operations": 35,
                "allowed_operations": list(range(35)),  # All operations 0-34
                "validate_actions": True,
                "clip_invalid_actions": False
            }
        }
    })

    env_config = create_config_from_hydra(config.environment)

    # Test environment creation
    key = jax.random.PRNGKey(456)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Full environment created successfully")
    logger.info(f"Grid shape: {state.working_grid.shape}")
    logger.info(f"Number of allowed operations: {len(env_config.action.allowed_operations)}")

    # Test movement operation (only available in full)
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[1:4, 1:4].set(True)  # Select area to move

    action = {
        "selection": selection,
        "operation": jnp.array(ARCLEOperationType.MOVE_UP, dtype=jnp.int32)
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Full environment movement successful, reward: {reward:.3f}")

    return True


def test_training_rewards():
    """Test training-specific reward configuration."""
    logger.info("Testing Training Rewards Configuration")

    config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 75,
            "log_operations": False,
            "reward": {
                "reward_on_submit_only": False,  # Dense rewards
                "step_penalty": -0.005,
                "success_bonus": 20.0,
                "similarity_weight": 2.0,
                "progress_bonus": 0.5,
                "invalid_action_penalty": -0.1
            },
            "action": {
                "action_format": "selection_operation",
                "selection_threshold": 0.5,
                "allow_partial_selection": True,
                "num_operations": 35,
                "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 32, 33, 34],
                "validate_actions": True,
                "clip_invalid_actions": False
            }
        }
    })

    env_config = create_config_from_hydra(config.environment)

    # Test environment creation
    key = jax.random.PRNGKey(789)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Training rewards environment created successfully")
    logger.info(f"Dense rewards enabled: {not env_config.reward.reward_on_submit_only}")
    logger.info(f"Step penalty: {env_config.reward.step_penalty}")
    logger.info(f"Success bonus: {env_config.reward.success_bonus}")

    # Test that we get rewards at each step (not just submit)
    selection = jnp.zeros_like(state.working_grid, dtype=jnp.bool_)
    selection = selection.at[0:2, 0:2].set(True)

    action = {
        "selection": selection,
        "operation": jnp.array(ARCLEOperationType.FILL_3, dtype=jnp.int32)
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Training rewards step reward: {reward:.3f} (should be non-zero)")

    return True


def test_point_actions():
    """Test point-based action configuration."""
    logger.info("Testing Point-Based Actions")

    config = OmegaConf.create({
        "environment": {
            "max_episode_steps": 100,
            "log_operations": False,
            "reward": {
                "reward_on_submit_only": True,
                "step_penalty": -0.01,
                "success_bonus": 10.0,
                "similarity_weight": 1.0,
                "progress_bonus": 0.1,
                "invalid_action_penalty": -0.5
            },
            "action": {
                "action_format": "point",
                "selection_threshold": 0.5,
                "allow_partial_selection": False,
                "num_operations": 35,
                "allowed_operations": list(range(35)),
                "validate_actions": True,
                "clip_invalid_actions": True
            }
        }
    })

    env_config = create_config_from_hydra(config.environment)

    # Test environment creation
    key = jax.random.PRNGKey(101)
    state, obs = arc_reset(key, env_config)

    logger.info(f"Point-based actions environment created successfully")
    logger.info(f"Action format: {env_config.action.action_format}")

    # Test point action
    action = {
        "point": (2, 3),
        "operation": jnp.array(ARCLEOperationType.FILL_5, dtype=jnp.int32)
    }

    state, obs, reward, done, info = arc_step(state, action, env_config)
    logger.info(f"Point action successful, reward: {reward:.3f}")

    return True


def test_config_validation():
    """Test configuration validation and error handling."""
    logger.info("Testing Configuration Validation")

    # Test that invalid operation IDs are rejected during config creation
    try:
        config = OmegaConf.create({
            "environment": {
                "action": {
                    "allowed_operations": [0, 1, 2, 999]  # Invalid operation ID
                }
            }
        })

        # This should fail during config creation due to invalid operation ID
        env_config = create_config_from_hydra(config.environment)
        logger.error("Expected validation error for invalid operation ID 999, but config was created successfully")
        return False

    except Exception as e:
        logger.info(f"✓ Configuration correctly rejected invalid operation ID: {e}")

    # Test that valid configuration works
    try:
        valid_config = OmegaConf.create({
            "environment": {
                "action": {
                    "allowed_operations": [0, 1, 2, 34]  # Valid operation IDs
                }
            }
        })

        env_config = create_config_from_hydra(valid_config.environment)
        logger.info("✓ Valid configuration accepted successfully")
        return True

    except Exception as e:
        logger.error(f"Valid configuration was unexpectedly rejected: {e}")
        return False


def test_operation_mapping():
    """Test that operation IDs map correctly to ARCLEOperationType."""
    logger.info("Testing Operation ID Mapping")

    # Test key operation mappings
    operation_tests = [
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
        (34, "SUBMIT")
    ]

    for op_id, op_name in operation_tests:
        arcle_op_value = getattr(ARCLEOperationType, op_name)
        if op_id == arcle_op_value:
            logger.info(f"✓ Operation {op_id} correctly maps to {op_name}")
        else:
            logger.error(f"✗ Operation {op_id} does not map to {op_name} (got {arcle_op_value})")
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

    results = []
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")

            result = test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")

        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Configuration system is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the configuration system.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
