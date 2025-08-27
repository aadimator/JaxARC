"""
Core grid initialization engine for diverse working grid initialization strategies.

This module implements the core functionality for initializing working grids
with multiple strategies including demo grids, permutations, empty grids,
and random patterns to enhance training diversity in batched environments.

The module includes comprehensive validation and error handling to ensure
robust initialization with informative error messages for debugging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from jaxarc.configs import GridInitializationConfig
from jaxarc.configs.validation import ConfigValidationError

from ..types import JaxArcTask
from ..utils.jax_types import GridArray, MaskArray, PRNGKey


class GridInitializationError(Exception):
    """Raised when grid initialization fails."""


def validate_task_compatibility(task: JaxArcTask) -> list[str]:
    """Validate that a task is compatible with grid initialization."""
    errors: list[str] = []

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

    # Validate task has at least one training pair for demo/permutation modes
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
        except (AttributeError, TypeError, ValueError) as e:
            errors.append(f"Failed to get task grid shape: {e}")

    return errors


def get_detailed_error_message(
    error: Exception, config: GridInitializationConfig, mode: str, context: str = ""
) -> str:
    """Build a detailed error message for initialization failures."""
    parts = [
        f"Grid initialization failed in {mode} mode",
        f"Error: {type(error).__name__}: {error}",
        (
            f"Weights: demo={config.demo_weight:.3f}, "
            f"permutation={config.permutation_weight:.3f}, "
            f"empty={config.empty_weight:.3f}, random={config.random_weight:.3f}"
        ),
    ]

    if config.permutation_weight > 0.0:
        parts.append(f"Permutation types: {list(config.permutation_types)}")

    parts.append(
        f"Random config: density={config.random_density}, pattern_type={config.random_pattern_type}"
    )

    if context:
        parts.append(f"Context: {context}")

    return " | ".join(parts)


def initialize_working_grids_with_validation(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    batch_size: int = 1,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize working grids with full validation and error handling.

    This is the main entry point that includes comprehensive validation
    and error handling. It calls the JAX-compatible core function and
    then performs validation outside of JAX compilation.

    Args:
        task: JaxArcTask containing available demonstration and test pairs
        config: GridInitializationConfig specifying initialization strategy
        key: JAX PRNG key for random operations
        batch_size: Number of grids to initialize (default: 1)
        initial_pair_idx: Optional specific pair index for demo-based modes.
                         If None, uses random selection. If specified, uses that pair.

    Returns:
        Tuple containing:
        - initialized_grids: JAX array of initialized grids [batch_size, height, width]
        - grid_masks: JAX array of corresponding masks [batch_size, height, width]

    Raises:
        ConfigValidationError: If configuration is invalid
        GridInitializationError: If initialization fails
    """
    # Validate configuration before proceeding
    # Prefer co-located validation on the config itself
    config_errors = list(config.validate())
    if config_errors:
        error_msg = f"Invalid grid initialization configuration: {config_errors}"
        logger.error(error_msg)
        raise ConfigValidationError(error_msg)

    # Validate task compatibility
    task_errors = validate_task_compatibility(task)
    if task_errors:
        error_msg = f"Task incompatible with grid initialization: {task_errors}"
        logger.error(error_msg)
        raise GridInitializationError(error_msg)

    try:
        # Call the JAX-compatible core function
        initialized_grids, grid_masks = initialize_working_grids(
            task, config, key, batch_size, initial_pair_idx
        )

        # Post-process validation (outside JAX compilation)
        # validation currently handled in core functions; skip here

        # Note: We can't do this validation inside JIT-compiled functions
        # So we skip it for now and rely on the core functions being correct

        logger.debug(f"Successfully initialized {batch_size} grids with validation")

        return initialized_grids, grid_masks

    except Exception as e:
        error_msg = get_detailed_error_message(e, config, "batch_initialization")
        logger.error(error_msg)
        raise GridInitializationError(error_msg) from e


def initialize_working_grids(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    batch_size: int = 1,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize working grids based on configuration strategy (JAX-compatible core).

    This is the JAX-compatible core grid initialization function that can be
    JIT-compiled. It does not include validation or error handling to maintain
    JAX compatibility.

    Args:
        task: JaxArcTask containing available demonstration and test pairs
        config: GridInitializationConfig specifying initialization strategy
        key: JAX PRNG key for random operations
        batch_size: Number of grids to initialize (default: 1)
        initial_pair_idx: Optional specific pair index for demo-based modes.
                         If None, uses random selection. If specified, uses that pair.

    Returns:
        Tuple containing:
        - initialized_grids: JAX array of initialized grids [batch_size, height, width]
        - grid_masks: JAX array of corresponding masks [batch_size, height, width]

    Examples:
        ```python
        # Single grid initialization with random demo selection
        grids, masks = initialize_working_grids(task, config, key)

        # Batch initialization with specific demo pair
        grids, masks = initialize_working_grids(
            task, config, key, batch_size=32, initial_pair_idx=2
        )
        ```

    Note:
        This function is designed to be JAX-compatible and can be JIT-compiled.
        For validation and error handling, use initialize_working_grids_with_validation.
    """
    # Split PRNG key for batch operations
    keys = jax.random.split(key, batch_size + 1)
    mode_key, init_keys = keys[0], keys[1:]

    # Select initialization modes for batch using probability weights
    mode_indices = _select_batch_modes(mode_key, config, batch_size)

    # Vectorize initialization across batch using vmap
    vectorized_init = jax.vmap(
        lambda single_key, mode_idx: _initialize_single_grid(
            task, config, single_key, mode_idx, initial_pair_idx
        ),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )

    # Initialize all grids in batch
    initialized_grids, grid_masks = vectorized_init(init_keys, mode_indices)

    return initialized_grids, grid_masks


def _select_batch_modes(
    key: PRNGKey, config: GridInitializationConfig, batch_size: int
) -> jnp.ndarray:
    """Select initialization modes for batch using JAX random choice with weights.

    Args:
        key: JAX PRNG key for random selection
        config: GridInitializationConfig containing mode weights
        batch_size: Number of modes to select

    Returns:
        JAX array of mode indices [batch_size] with values 0-3:
        - 0: demo
        - 1: permutation
        - 2: empty
        - 3: random
    """
    # Use probability weights (always-on)
    weights = jnp.array(
        [
            config.demo_weight,
            config.permutation_weight,
            config.empty_weight,
            config.random_weight,
        ],
        dtype=jnp.float32,
    )

    # Check for invalid weights (NaN, inf, negative)
    weights = jnp.where(jnp.isfinite(weights) & (weights >= 0), weights, 0.0)

    # Ensure weights sum to something positive
    weight_sum = jnp.sum(weights)
    weights = jnp.where(
        weight_sum > 1e-8,
        weights / weight_sum,
        jnp.array(
            [0.25, 0.25, 0.25, 0.25], dtype=jnp.float32
        ),  # Equal weights fallback
    )

    # Sample mode indices using JAX random choice
    return jax.random.choice(
        key,
        a=4,
        shape=(batch_size,),
        p=weights,
    )


def _initialize_single_grid(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    mode_idx: int,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize a single grid using JAX-compatible mode dispatching.

    This is the core initialization function that is JAX-compatible
    and can be JIT-compiled. It does not include validation or error
    handling to maintain JAX compatibility.

    Args:
        task: JaxArcTask containing available pairs
        config: GridInitializationConfig for initialization parameters
        key: JAX PRNG key for random operations
        mode_idx: Mode index (0=demo, 1=permutation, 2=empty, 3=random)
        initial_pair_idx: Optional specific pair index for demo-based modes

    Returns:
        Tuple of (initialized_grid, grid_mask)
    """
    # Use JAX switch for efficient mode dispatching
    return jax.lax.switch(
        mode_idx,
        [
            lambda: _init_demo_grid(task, key, initial_pair_idx),
            lambda: _init_permutation_grid(task, config, key, initial_pair_idx),
            lambda: _init_empty_grid(task),
            lambda: _init_random_grid(task, config, key),
        ],
    )


def _init_demo_grid(
    task: JaxArcTask, key: PRNGKey, initial_pair_idx: int | None = None
) -> tuple[GridArray, MaskArray]:
    """Initialize grid from demo input examples with support for manual pair selection.

    Args:
        task: JaxArcTask containing demonstration pairs
        key: JAX PRNG key for random demo selection (used only if initial_pair_idx is None)
        initial_pair_idx: Optional specific pair index. If None, selects randomly.

    Returns:
        Tuple of (demo_grid, demo_mask)

    Note:
        This function is designed to be fully JAX-compatible. If there are no
        training pairs, it will create a default empty grid.
    """

    # Handle case where there are no training pairs
    def create_default_grid():
        # Get the maximum grid dimensions from the task structure
        # Even with no training pairs, the task should have the correct padded dimensions
        max_height, max_width = task.get_grid_shape()

        # Create an empty grid with maximum dimensions (all padding)
        default_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        # Create a mask that indicates no valid region (all False)
        default_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        return default_grid, default_mask

    def use_demo_grid():
        # Choose demo index based on whether initial_pair_idx is specified
        if initial_pair_idx is not None:
            # Use specified pair index (with bounds checking)
            demo_idx = jnp.clip(
                initial_pair_idx, 0, jnp.maximum(task.num_train_pairs - 1, 0)
            )
        else:
            # Select random demo pair from available demonstrations
            # Ensure we don't divide by zero
            safe_num_pairs = jnp.maximum(task.num_train_pairs, 1)
            random_idx = jax.random.randint(key, (), 0, safe_num_pairs)
            demo_idx = random_idx % safe_num_pairs
            demo_idx = jnp.clip(demo_idx, 0, jnp.maximum(task.num_train_pairs - 1, 0))

        # Get demo grid and mask (this should only be called when num_train_pairs > 0)
        demo_grid = task.input_grids_examples[demo_idx]
        demo_mask = task.input_masks_examples[demo_idx]

        return demo_grid, demo_mask

    # Use conditional to handle empty task case
    return jax.lax.cond(task.num_train_pairs > 0, use_demo_grid, create_default_grid)


def _init_permutation_grid(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize grid with permuted versions of demo inputs.

    Args:
        task: JaxArcTask containing demonstration pairs
        config: GridInitializationConfig with permutation settings
        key: JAX PRNG key for random operations
        initial_pair_idx: Optional specific pair index for base demo selection

    Returns:
        Tuple of (permuted_grid, grid_mask)

    Note:
        If no valid permutations are available, falls back to the base demo grid.
        This function is designed to be JAX-compatible.
    """
    # Split key for demo selection and permutation
    demo_key, perm_key = jax.random.split(key)

    # Start with a demo grid (respecting initial_pair_idx)
    base_grid, base_mask = _init_demo_grid(task, demo_key, initial_pair_idx)

    # Check if permutation types are available
    has_permutations = len(config.permutation_types) > 0

    # Apply permutations if available, otherwise return base grid
    permuted_grid = jax.lax.cond(
        has_permutations,
        lambda: _apply_grid_permutations(base_grid, config, perm_key),
        lambda: base_grid,  # Fallback to base grid if no permutations
    )

    return permuted_grid, base_mask


def _init_empty_grid(task: JaxArcTask) -> tuple[GridArray, MaskArray]:
    """Initialize completely empty grids (all zeros).

    Args:
        task: JaxArcTask to get grid dimensions

    Returns:
        Tuple of (empty_grid, grid_mask)

    Note:
        This function is designed to be JAX-compatible. If there are no
        training pairs, it creates a default 10x10 grid.
    """

    def create_default_empty():
        # Get the maximum grid dimensions from the task structure
        max_height, max_width = task.get_grid_shape()

        # Create an empty grid with maximum dimensions
        empty_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        # For empty grids, we typically want a full valid region
        # But since there are no training pairs, create a minimal valid region
        grid_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        return empty_grid, grid_mask

    def use_template_shape():
        # Use the first demo grid as a template for shape
        template_grid = task.input_grids_examples[0]
        template_mask = task.input_masks_examples[0]

        # Create empty grid with same shape as template, filled with zeros
        empty_grid = jnp.zeros_like(template_grid, dtype=jnp.int32)

        # Use the same mask as the template to preserve the valid region
        grid_mask = template_mask

        return empty_grid, grid_mask

    # Use conditional to handle empty task case
    return jax.lax.cond(
        task.num_train_pairs > 0, use_template_shape, create_default_empty
    )


def _init_random_grid(
    task: JaxArcTask, config: GridInitializationConfig, key: PRNGKey
) -> tuple[GridArray, MaskArray]:
    """Initialize grids with random patterns.

    Args:
        task: JaxArcTask to get grid dimensions and constraints
        config: GridInitializationConfig with random pattern settings
        key: JAX PRNG key for random generation

    Returns:
        Tuple of (random_grid, grid_mask)

    Note:
        This function is designed to be JAX-compatible. If there are no
        training pairs, it creates a default 10x10 grid.
    """

    def create_default_random():
        # Get the maximum grid dimensions from the task structure
        max_height, max_width = task.get_grid_shape()
        grid_shape = (max_height, max_width)

        density = jnp.clip(config.random_density, 0.0, 1.0)

        # Generate random pattern (default to sparse)
        pattern_type_map = {"sparse": 0, "dense": 1, "structured": 2, "noise": 3}
        pattern_idx = pattern_type_map.get(config.random_pattern_type, 0)

        random_grid = jax.lax.switch(
            pattern_idx,
            [
                lambda: _generate_sparse_pattern(grid_shape, density, key),
                lambda: _generate_dense_pattern(grid_shape, density, key),
                lambda: _generate_structured_pattern(grid_shape, density, key),
                lambda: _generate_noise_pattern(grid_shape, density, key),
            ],
        )

        # For random grids with no training pairs, create a minimal valid region
        grid_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        return random_grid, grid_mask

    def use_template_shape():
        # Use the first demo grid as a template for shape
        template_grid = task.input_grids_examples[0]
        template_mask = task.input_masks_examples[0]
        grid_shape = template_grid.shape

        # Validate and clamp density to valid range
        density = jnp.clip(config.random_density, 0.0, 1.0)

        # Generate random patterns based on pattern type using JAX switch
        pattern_type_map = {"sparse": 0, "dense": 1, "structured": 2, "noise": 3}

        # Default to sparse if pattern type is invalid
        pattern_idx = pattern_type_map.get(config.random_pattern_type, 0)

        # Use JAX switch for pattern generation
        random_grid = jax.lax.switch(
            pattern_idx,
            [
                lambda: _generate_sparse_pattern(grid_shape, density, key),
                lambda: _generate_dense_pattern(grid_shape, density, key),
                lambda: _generate_structured_pattern(grid_shape, density, key),
                lambda: _generate_noise_pattern(grid_shape, density, key),
            ],
        )

        # Apply the template mask to ensure we only have content in valid regions
        # Set invalid regions to 0 (background)
        random_grid = jnp.where(template_mask, random_grid, 0)

        # Use the same mask as the template to preserve the valid region
        grid_mask = template_mask

        return random_grid, grid_mask

    # Use conditional to handle empty task case
    return jax.lax.cond(
        task.num_train_pairs > 0, use_template_shape, create_default_random
    )


def _apply_grid_permutations(
    grid: GridArray, config: GridInitializationConfig, key: PRNGKey
) -> GridArray:
    """Apply various transformations to create grid variations.

    Args:
        grid: Input grid to transform
        config: GridInitializationConfig with permutation types
        key: JAX PRNG key for random permutation selection

    Returns:
        Transformed grid (or original grid if no valid permutations)

    Note:
        This function is designed to be fully JAX-compatible and handles
        empty permutation types gracefully.
    """
    # Select random permutation type
    perm_key, apply_key = jax.random.split(key)

    # Create a mapping of permutation types to indices
    # We'll use a fixed set of permutations and select based on availability
    has_rotate = "rotate" in config.permutation_types
    has_reflect = "reflect" in config.permutation_types
    has_color_remap = "color_remap" in config.permutation_types

    # If no permutations available, return original grid
    has_any_perm = has_rotate | has_reflect | has_color_remap

    # Select a random permutation type (0=rotate, 1=reflect, 2=color_remap)
    perm_choice = jax.random.randint(perm_key, (), 0, 3)

    # Apply permutation based on availability and choice
    def apply_permutation():
        return jax.lax.switch(
            perm_choice,
            [
                lambda: jax.lax.cond(
                    has_rotate, lambda: _apply_rotation(grid, apply_key), lambda: grid
                ),
                lambda: jax.lax.cond(
                    has_reflect,
                    lambda: _apply_reflection(grid, apply_key),
                    lambda: grid,
                ),
                lambda: jax.lax.cond(
                    has_color_remap,
                    lambda: _apply_color_remap(grid, apply_key),
                    lambda: grid,
                ),
            ],
        )

    # Return permuted grid if any permutations available, otherwise original
    return jax.lax.cond(has_any_perm, apply_permutation, lambda: grid)


def _apply_rotation(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply random rotation (90°, 180°, 270°) to grid.

    Args:
        grid: Input grid to rotate
        key: JAX PRNG key for random rotation selection

    Returns:
        Rotated grid (or original grid if rotation fails)

    Note:
        This function includes basic validation and is designed to be
        JAX-compatible. For non-square grids, only 180° rotation is applied
        to maintain shape compatibility.
    """
    # Validate input grid
    if grid.ndim != 2:
        return grid  # Return original if not 2D

    height, width = grid.shape

    # For non-square grids, only apply 180° rotation to maintain shape
    if height != width:
        return jnp.rot90(grid, k=2)  # 180° rotation preserves shape

    # For square grids, apply random rotation
    rotation_idx = jax.random.choice(key, 3)

    # Apply rotation
    return jax.lax.switch(
        rotation_idx,
        [
            lambda: jnp.rot90(grid, k=1),  # 90° clockwise
            lambda: jnp.rot90(grid, k=2),  # 180°
            lambda: jnp.rot90(grid, k=3),  # 270° clockwise (90° counter-clockwise)
        ],
    )


def _apply_reflection(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply random reflection (horizontal or vertical) to grid.

    Args:
        grid: Input grid to reflect
        key: JAX PRNG key for random reflection selection

    Returns:
        Reflected grid (or original grid if reflection fails)

    Note:
        This function includes basic validation and is designed to be
        JAX-compatible. Invalid grids are returned unchanged.
    """
    # Validate input grid
    if grid.ndim != 2:
        return grid  # Return original if not 2D

    # Select reflection: 0=horizontal, 1=vertical
    reflection_idx = jax.random.choice(key, 2)

    # Apply reflection
    return jax.lax.switch(
        reflection_idx,
        [
            lambda: jnp.fliplr(grid),  # Horizontal flip
            lambda: jnp.flipud(grid),  # Vertical flip
        ],
    )


def _apply_color_remap(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply systematic color remapping while preserving structure.

    Args:
        grid: Input grid to remap colors
        key: JAX PRNG key for random color mapping

    Returns:
        Color-remapped grid (or original grid if remapping fails)

    Note:
        This function includes validation to ensure colors stay within
        valid ARC range (0-9). It is designed to be JAX-compatible.
    """
    # Validate input grid
    if grid.ndim != 2:
        return grid  # Return original if not 2D

    # Ensure all grid values are within valid ARC color range (0-9)
    grid_clamped = jnp.clip(grid, 0, 9)

    # Create a simple color mapping
    # Map each color 0-9 to a different color 0-9
    arc_colors = jnp.arange(10, dtype=jnp.int32)
    shuffled_colors = jax.random.permutation(key, arc_colors)

    # Use JAX-compatible vectorized color mapping
    # Create a lookup table and use advanced indexing
    remapped_grid = shuffled_colors[grid_clamped]

    # Ensure output is still within valid range (additional safety check)
    return jnp.clip(remapped_grid, 0, 9)


def _generate_sparse_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate sparse random pattern with isolated elements.

    Args:
        shape: Grid shape (height, width) - should be static
        density: Pattern density (0.0 to 1.0)
        key: JAX PRNG key for random generation

    Returns:
        Grid with sparse random pattern

    Note:
        This function is designed to be JAX-compatible and expects
        static shapes. The caller is responsible for masking the
        result to valid regions if needed.
    """
    # Validate and clamp density
    density = jnp.clip(density, 0.0, 1.0)

    # Generate random mask for pattern placement
    mask_key, color_key = jax.random.split(key)
    pattern_mask = jax.random.bernoulli(mask_key, density, shape)

    # Generate random colors for pattern elements (ARC colors 1-9, avoiding 0 for background)
    # Use 1-9 for non-background colors to ensure visibility
    random_colors = jax.random.randint(color_key, shape, 1, 10)

    # Apply pattern: use random colors where mask is True, 0 (background) elsewhere
    sparse_grid = jnp.where(pattern_mask, random_colors, 0)

    return sparse_grid.astype(jnp.int32)


def _generate_dense_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate dense random pattern with connected regions.

    Args:
        shape: Grid shape (height, width) - should be static
        density: Pattern density (0.0 to 1.0)
        key: JAX PRNG key for random generation

    Returns:
        Grid with dense random pattern

    Note:
        This function uses morphological operations to create connected regions
        and is designed to be JAX-compatible with static shapes.
    """
    # Validate and clamp density
    density = jnp.clip(density, 0.0, 1.0)

    # Start with sparse pattern as base
    sparse_key, dilation_key = jax.random.split(key)
    sparse_grid = _generate_sparse_pattern(shape, density, sparse_key)

    # Apply simple dilation to create connected regions
    # Use a 3x3 kernel for morphological dilation approximation
    kernel = jnp.ones((3, 3), dtype=jnp.float32) / 9.0  # Normalized kernel

    # Apply convolution to create dense regions
    # Pad the input to handle edge effects
    padded_sparse = jnp.pad(
        sparse_grid.astype(jnp.float32), 1, mode="constant", constant_values=0
    )
    dilated = jax.scipy.signal.convolve2d(padded_sparse, kernel, mode="valid")

    # Create dense mask based on convolution result
    dense_mask = dilated > 0.1  # Any neighboring non-zero cells (lowered threshold)

    # Generate new random colors for dense regions (1-9 to avoid background)
    color_key = jax.random.split(dilation_key)[0]
    random_colors = jax.random.randint(color_key, shape, 1, 10)

    # Combine original sparse pattern with dilated regions
    dense_grid = jnp.where(dense_mask, random_colors, sparse_grid)

    return dense_grid.astype(jnp.int32)


def _generate_structured_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate structured pattern with simple geometric shapes.

    Args:
        shape: Grid shape (height, width) - should be static
        density: Pattern density (0.0 to 1.0)
        key: JAX PRNG key for random generation

    Returns:
        Grid with structured geometric pattern

    Note:
        This function creates simple geometric patterns like lines.
        It is designed to be JAX-compatible with static shapes.
    """
    height, _ = shape

    # Validate and clamp density
    density = jnp.clip(density, 0.0, 1.0)

    # Create a simple structured pattern using geometric shapes
    # Start with a base sparse pattern
    base_key, struct_key = jax.random.split(key)
    base_grid = _generate_sparse_pattern(shape, density * 0.5, base_key)

    # Add simple line structures
    line_key, color_key = jax.random.split(struct_key)

    # Generate a few horizontal lines
    # Use fixed number of lines to avoid dynamic shapes
    num_lines = 3
    line_positions = jax.random.randint(line_key, (num_lines,), 0, height)
    line_colors = jax.random.randint(color_key, (num_lines,), 1, 10)

    # Create coordinate grids
    h_indices = jnp.arange(height)[:, None]

    # Initialize line mask
    line_mask = jnp.zeros(shape, dtype=jnp.bool_)

    # Add horizontal lines based on density (unroll the loop for JAX compatibility)
    # Line 1
    should_add_line_1 = density > 0.3
    line_at_pos_1 = h_indices == line_positions[0]
    line_mask = jnp.where(should_add_line_1, line_mask | line_at_pos_1, line_mask)

    # Line 2
    should_add_line_2 = density > 0.6
    line_at_pos_2 = h_indices == line_positions[1]
    line_mask = jnp.where(should_add_line_2, line_mask | line_at_pos_2, line_mask)

    # Line 3
    should_add_line_3 = density > 0.9
    line_at_pos_3 = h_indices == line_positions[2]
    line_mask = jnp.where(should_add_line_3, line_mask | line_at_pos_3, line_mask)

    # Create structured grid by combining base pattern with lines
    # Use the first line color for all lines
    structured_grid = jnp.where(line_mask, line_colors[0], base_grid)

    return structured_grid.astype(jnp.int32)


def _generate_noise_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate completely random noise pattern.

    Args:
        shape: Grid shape (height, width) - should be static
        density: Pattern density (0.0 to 1.0)
        key: JAX PRNG key for random generation

    Returns:
        Grid with random noise pattern

    Note:
        This function generates pure random noise with specified density
        and is designed to be JAX-compatible with static shapes.
    """
    # Validate and clamp density
    density = jnp.clip(density, 0.0, 1.0)

    # Generate random colors for all cells (ARC colors 1-9, avoiding 0 for background)
    color_key, mask_key = jax.random.split(key)
    random_colors = jax.random.randint(color_key, shape, 1, 10)

    # Apply density mask
    density_mask = jax.random.bernoulli(mask_key, density, shape)

    # Apply noise where mask is True, background (0) elsewhere
    noise_grid = jnp.where(density_mask, random_colors, 0)

    return noise_grid.astype(jnp.int32)
