"""
JAX-compatible validation and error handling utilities.

This module centralizes all validation logic, using Equinox for JAX-compatible
error checking. It provides functions for validating states, actions, and
individual values within JAX transformations.
"""

from __future__ import annotations

import os
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from loguru import logger

from ..envs.config import JaxArcConfig
from ..state import ArcEnvState

ErrorMode = Literal["raise", "nan", "breakpoint"]


def get_error_mode() -> ErrorMode:
    """Get the current error handling mode from the environment variable."""
    mode = os.environ.get("EQX_ON_ERROR", "raise").lower()
    if mode in ["raise", "nan", "breakpoint"]:
        return mode
    logger.warning(f"Unknown error mode '{mode}', defaulting to 'raise'")
    return "raise"


def set_error_mode(mode: ErrorMode) -> None:
    """Set the error handling mode via an environment variable."""
    if mode not in ["raise", "nan", "breakpoint"]:
        raise ValueError(f"Invalid error mode: {mode}. Must be 'raise', 'nan', or 'breakpoint'")
    os.environ["EQX_ON_ERROR"] = mode
    logger.info(f"Error handling mode set to: {mode}")


def configure_debugging(
    mode: ErrorMode = "raise",
    breakpoint_frames: int = 3,
    enable_nan_checks: bool = True,
) -> None:
    """Configure the debugging environment for error handling."""
    set_error_mode(mode)
    os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = str(breakpoint_frames)
    if enable_nan_checks:
        os.environ["JAX_DEBUG_NANS"] = "True"
    logger.info(f"Debugging configured: mode={mode}, frames={breakpoint_frames}, nan_checks={enable_nan_checks}")


@eqx.filter_jit
def validate_action(action: "StructuredAction", config: JaxArcConfig) -> "StructuredAction":
    """Validate a structured action with runtime error checking."""
    max_operations = 42
    action = eqx.error_if(
        action,
        (action.operation < 0) | (action.operation >= max_operations),
        f"Invalid operation ID: must be in [0, {max_operations-1}]",
    )
    action = action.validate(
        grid_shape=(config.dataset.max_grid_height, config.dataset.max_grid_width),
        max_operations=max_operations,
    )
    return action


def validate_batch_actions(
    actions: "StructuredAction", config: JaxArcConfig, batch_size: int
) -> "StructuredAction":
    """Validate a batch of structured actions with clear error messages."""
    try:
        validate_fn = lambda action: validate_action(action, config)
        return jax.vmap(validate_fn)(actions)
    except Exception as e:
        error_msg = f"Batch action validation failed (batch_size={batch_size}): {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


@eqx.filter_jit
def validate_state_consistency(state: ArcEnvState) -> ArcEnvState:
    """Perform comprehensive validation of the environment state."""
    working_shape = state.working_grid.shape
    target_shape = state.target_grid.shape
    shapes_match = jnp.array_equal(jnp.array(working_shape), jnp.array(target_shape))
    state = eqx.error_if(state, ~shapes_match, "Working and target grid shapes must match")

    mask_shape = state.working_grid_mask.shape
    grid_mask_matches = jnp.array_equal(jnp.array(working_shape), jnp.array(mask_shape))
    state = eqx.error_if(state, ~grid_mask_matches, "Working grid mask shape must match working grid")

    state = eqx.error_if(state, state.step_count < 0, "Step count cannot be negative")
    state = eqx.error_if(
        state,
        (state.similarity_score < 0.0) | (state.similarity_score > 1.0),
        "Similarity score must be in [0.0, 1.0]",
    )
    return state


@eqx.filter_jit
def assert_positive(value: jnp.ndarray, name: str = "value") -> jnp.ndarray:
    """Assert that a value is positive."""
    return eqx.error_if(value, value <= 0, f"{name} must be positive")


@eqx.filter_jit
def assert_in_range(
    value: jnp.ndarray, min_val: float, max_val: float, name: str = "value"
) -> jnp.ndarray:
    """Assert that a value is within a specified range."""
    return eqx.error_if(
        value,
        (value < min_val) | (value > max_val),
        f"{name} must be in range [{min_val}, {max_val}]",
    )


@eqx.filter_jit
def assert_shape_matches(
    array: jnp.ndarray, expected_shape: tuple[int, ...], name: str = "array"
) -> jnp.ndarray:
    """Assert that an array has the expected shape."""
    actual_shape = array.shape
    shapes_match = jnp.array_equal(jnp.array(actual_shape), jnp.array(expected_shape))
    return eqx.error_if(array, ~shapes_match, f"{name} shape mismatch: expected {expected_shape}")