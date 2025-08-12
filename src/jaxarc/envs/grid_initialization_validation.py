"""
Comprehensive validation and error handling for grid initialization.

This module provides validation functions for configuration parameters,
runtime validation for generated grids, and robust error recovery mechanisms
with informative error messages for debugging initialization issues.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from loguru import logger

from jaxarc.configs import GridInitializationConfig

from ..types import JaxArcTask
from ..utils.jax_types import GridArray, MaskArray


class GridInitializationError(Exception):
    """Base exception for grid initialization errors."""

    def __init__(self, message: str, error_code: str = "GRID_INIT_ERROR"):
        self.error_code = error_code
        super().__init__(message)


class ConfigurationValidationError(GridInitializationError):
    """Exception for configuration validation errors."""

    def __init__(self, message: str):
        super().__init__(message, "CONFIG_VALIDATION_ERROR")


class GridValidationError(GridInitializationError):
    """Exception for grid validation errors."""

    def __init__(self, message: str):
        super().__init__(message, "GRID_VALIDATION_ERROR")


# Fallback functionality removed


def validate_grid_initialization_config(config: GridInitializationConfig) -> list[str]:
    """
    Validate grid initialization configuration parameters.

    Args:
        config: GridInitializationConfig to validate

    Returns:
        List of validation error messages (empty if valid)

    Examples:
        ```python
        config = GridInitializationConfig(mode="mixed", demo_weight=0.5, permutation_weight=0.6)
        errors = validate_grid_initialization_config(config)
        if errors:
            print(f"Configuration errors: {errors}")
        ```
    """
    errors = []

    # Validate weights (must be non-negative and sum to 1.0 for mixed mode)
    weights = [
        ("demo_weight", config.demo_weight),
        ("permutation_weight", config.permutation_weight),
        ("empty_weight", config.empty_weight),
        ("random_weight", config.random_weight),
    ]

    for weight_name, weight_value in weights:
        if not isinstance(weight_value, (int, float)):
            errors.append(
                f"{weight_name} must be a number, got {type(weight_value).__name__}"
            )
            continue

        if weight_value < 0.0:
            errors.append(f"{weight_name} must be non-negative, got {weight_value}")
        elif weight_value > 1.0:
            errors.append(f"{weight_name} must be <= 1.0, got {weight_value}")

    # Weights should sum to 1.0 (always)
    total_weight = sum(weight for _, weight in weights)
    if abs(total_weight - 1.0) > 1e-6:
        errors.append(
            f"Initialization weights must sum to 1.0, got {total_weight:.6f}. "
            f"Individual weights: demo={config.demo_weight}, "
            f"permutation={config.permutation_weight}, empty={config.empty_weight}, "
            f"random={config.random_weight}"
        )

    # Validate permutation types
    valid_permutation_types = {"rotate", "reflect", "color_remap"}
    if hasattr(config.permutation_types, "__iter__"):
        invalid_types = set(config.permutation_types) - valid_permutation_types
        if invalid_types:
            errors.append(
                f"Invalid permutation types: {sorted(invalid_types)}. "
                f"Valid types: {sorted(valid_permutation_types)}"
            )

        # If permutation weight is positive, permutation_types must not be empty
        if config.permutation_weight > 0.0 and not config.permutation_types:
            errors.append(
                "permutation_types cannot be empty when permutation_weight > 0"
            )

    # Validate random density
    if not isinstance(config.random_density, (int, float)):
        errors.append(
            f"random_density must be a number, got {type(config.random_density).__name__}"
        )
    elif not 0.0 <= config.random_density <= 1.0:
        errors.append(
            f"random_density must be in range [0.0, 1.0], got {config.random_density}"
        )

    # Validate random pattern type
    valid_pattern_types = {"sparse", "dense", "structured", "noise"}
    if config.random_pattern_type not in valid_pattern_types:
        errors.append(
            f"Invalid random_pattern_type '{config.random_pattern_type}'. "
            f"Valid types: {sorted(valid_pattern_types)}"
        )

    # No mode cross-validation anymore

    return errors


def validate_generated_grid(
    grid: GridArray, mask: MaskArray, task: JaxArcTask, mode: str = "unknown"
) -> list[str]:
    """
    Validate a generated grid against ARC constraints.

    Args:
        grid: Generated grid to validate
        mask: Corresponding mask for the grid
        task: JaxArcTask for context and constraints
        mode: Initialization mode used (for error context)

    Returns:
        List of validation error messages (empty if valid)

    Examples:
        ```python
        errors = validate_generated_grid(grid, mask, task, mode="random")
        if errors:
            logger.error(f"Grid validation failed: {errors}")
        ```
    """
    errors = []

    # Validate grid shape
    if grid.ndim != 2:
        errors.append(f"Grid must be 2D, got {grid.ndim}D shape: {grid.shape}")
        return errors  # Can't continue validation with wrong dimensions

    if mask.ndim != 2:
        errors.append(f"Mask must be 2D, got {mask.ndim}D shape: {mask.shape}")
        return errors

    # Validate grid and mask have same shape
    if grid.shape != mask.shape:
        errors.append(f"Grid shape {grid.shape} doesn't match mask shape {mask.shape}")
        return errors

    # Validate grid dimensions are reasonable
    height, width = grid.shape
    if height <= 0 or width <= 0:
        errors.append(f"Grid dimensions must be positive, got {height}x{width}")

    if height > 100 or width > 100:
        logger.warning(f"Grid is very large: {height}x{width}")

    # Validate grid contains only valid ARC colors (0-9)
    unique_colors = jnp.unique(grid)
    invalid_colors = unique_colors[(unique_colors < 0) | (unique_colors > 9)]
    if len(invalid_colors) > 0:
        errors.append(
            f"Grid contains invalid colors: {invalid_colors.tolist()}. "
            "Valid ARC colors are 0-9"
        )

    # Validate mask contains only boolean values
    if mask.dtype != jnp.bool_:
        errors.append(f"Mask must be boolean type, got {mask.dtype}")

    # Validate mask has at least some valid cells
    valid_cells = jnp.sum(mask)
    if valid_cells == 0:
        errors.append("Mask has no valid cells (all False)")
    elif valid_cells == mask.size:
        # All cells valid - this is fine but worth noting
        pass

    # Context-specific validations
    if mode == "empty":
        # Empty grids should be all zeros
        non_zero_count = jnp.sum(grid != 0)
        if non_zero_count > 0:
            errors.append(
                f"Empty mode grid should be all zeros, but has {non_zero_count} non-zero cells"
            )

    elif mode == "demo":
        # Demo grids should match one of the available demo grids
        # This is more complex to validate in JAX, so we'll skip detailed validation
        # but ensure basic constraints are met
        pass

    elif mode == "random":
        # Random grids should have some variety (not all same color)
        if len(unique_colors) == 1 and unique_colors[0] == 0:
            logger.warning("Random grid is all background color (0)")

    return errors


def validate_task_compatibility(task: JaxArcTask) -> list[str]:
    """
    Validate that a task is compatible with grid initialization.

    Args:
        task: JaxArcTask to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate task has required attributes
    if not hasattr(task, "input_grids_examples"):
        errors.append("Task missing input_grids_examples attribute")
        return errors

    if not hasattr(task, "input_masks_examples"):
        errors.append("Task missing input_masks_examples attribute")
        return errors

    if not hasattr(task, "num_train_pairs"):
        errors.append("Task missing num_train_pairs attribute")
        return errors

    # Validate task has at least one training pair for demo mode
    if task.num_train_pairs <= 0:
        errors.append(
            f"Task has no training pairs (num_train_pairs={task.num_train_pairs}). "
            "At least one training pair is required for demo and permutation modes."
        )

    # Validate grid shapes are consistent
    if hasattr(task, "get_grid_shape"):
        try:
            grid_shape = task.get_grid_shape()
            if len(grid_shape) != 2:
                errors.append(f"Task grid shape must be 2D, got {grid_shape}")
            elif grid_shape[0] <= 0 or grid_shape[1] <= 0:
                errors.append(f"Task grid shape must be positive, got {grid_shape}")
        except Exception as e:
            errors.append(f"Failed to get task grid shape: {e}")

    return errors


# Fallback grid creation removed


def log_initialization_debug_info(
    config: GridInitializationConfig,
    task: JaxArcTask,
    mode_used: str,
    success: bool,
    error_message: str = "",
) -> None:
    """
    Log detailed debug information about grid initialization.

    Args:
        config: Configuration used for initialization
        task: Task being initialized
        mode_used: Actual initialization mode used
        success: Whether initialization succeeded
        error_message: Error message if initialization failed
    """
    debug_info = {
        "actual_mode": mode_used,
        "success": success,
        "task_train_pairs": getattr(task, "num_train_pairs", "unknown"),
        "config_weights": {
            "demo": config.demo_weight,
            "permutation": config.permutation_weight,
            "empty": config.empty_weight,
            "random": config.random_weight,
        },
        "permutation_types": list(config.permutation_types)
        if config.permutation_types
        else [],
        "random_density": config.random_density,
        "random_pattern_type": config.random_pattern_type,
    }

    if success:
        logger.debug(f"Grid initialization successful: {debug_info}")
    else:
        logger.error(
            f"Grid initialization failed: {debug_info}, error: {error_message}"
        )


def get_detailed_error_message(
    error: Exception, config: GridInitializationConfig, mode: str, context: str = ""
) -> str:
    """
    Generate detailed error message for debugging initialization issues.

    Args:
        error: The exception that occurred
        config: Configuration being used
        mode: Initialization mode that failed
        context: Additional context information

    Returns:
        Detailed error message with debugging information
    """
    error_details = [
        f"Grid initialization failed in {mode} mode",
        f"Error: {type(error).__name__}: {error}",
        (
            "Weights: demo={:.3f}, permutation={:.3f}, empty={:.3f}, random={:.3f}".format(
                config.demo_weight,
                config.permutation_weight,
                config.empty_weight,
                config.random_weight,
            )
        ),
    ]

    if mode in ("permutation", "mixed"):
        error_details.append(f"Permutation types: {list(config.permutation_types)}")

    if mode in ("random", "mixed"):
        error_details.append(
            f"Random config: density={config.random_density}, "
            f"pattern_type={config.random_pattern_type}"
        )

    # No fallback

    if context:
        error_details.append(f"Context: {context}")

    return " | ".join(error_details)
