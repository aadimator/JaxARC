"""
Grid initialization strategies for ARC tasks.

This module provides four initialization strategies for working grids:
1. Demo mode: Copy from training examples
2. Permutation mode: Apply transformations to demo grids
3. Empty mode: Start with blank grids
4. Random mode: Generate random patterns
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.configs import GridInitializationConfig

from ..types import GridArray, JaxArcTask, MaskArray, PRNGKey


def initialize_working_grids(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    batch_size: int = 1,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize working grids with configurable strategies.

    Args:
        task: JaxArcTask containing demonstration pairs
        config: GridInitializationConfig with mode weights
        key: JAX PRNG key for random operations
        batch_size: Number of grids to initialize
        initial_pair_idx: Optional specific pair index for demo-based modes

    Returns:
        Tuple of (initialized_grids, grid_masks) with shape [batch_size, height, width]
    """
    # Split keys for batch operations
    keys = jax.random.split(key, batch_size + 1)
    mode_key, init_keys = keys[0], keys[1:]

    # Select modes for each grid in batch
    mode_indices = _select_batch_modes(mode_key, config, batch_size)

    # Vectorize initialization across batch
    vectorized_init = jax.vmap(
        lambda single_key, mode_idx: _initialize_single_grid(
            task, config, single_key, mode_idx, initial_pair_idx
        ),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )

    return vectorized_init(init_keys, mode_indices)


def _select_batch_modes(
    key: PRNGKey, config: GridInitializationConfig, batch_size: int
) -> jnp.ndarray:
    """Select initialization modes for batch using weights."""
    # Create weights array and normalize
    weights = jnp.array(
        [
            config.demo_weight,
            config.permutation_weight,
            config.empty_weight,
            config.random_weight,
        ],
        dtype=jnp.float32,
    )

    # Normalize weights (handle zero sum case)
    weight_sum = jnp.sum(weights)
    weights = jnp.where(
        weight_sum > 1e-8,
        weights / weight_sum,
        jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32),
    )

    # Sample mode indices (0=demo, 1=permutation, 2=empty, 3=random)
    return jax.random.choice(key, a=4, shape=(batch_size,), p=weights)


def _initialize_single_grid(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    mode_idx: int,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize a single grid based on mode index."""
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
    """Initialize grid from demo input examples."""

    def use_demo_pair():
        # Select demo index
        if initial_pair_idx is not None:
            demo_idx = jnp.clip(initial_pair_idx, 0, task.num_train_pairs - 1)
        else:
            demo_idx = jax.random.randint(key, (), 0, task.num_train_pairs)

        return task.input_grids_examples[demo_idx], task.input_masks_examples[demo_idx]

    def create_empty_fallback():
        max_height, max_width = task.get_grid_shape()
        empty_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        empty_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        return empty_grid, empty_mask

    return jax.lax.cond(
        task.num_train_pairs > 0,
        use_demo_pair,
        create_empty_fallback,
    )


def _init_permutation_grid(
    task: JaxArcTask,
    config: GridInitializationConfig,
    key: PRNGKey,
    initial_pair_idx: int | None = None,
) -> tuple[GridArray, MaskArray]:
    """Initialize grid with permuted versions of demo inputs."""
    demo_key, perm_key = jax.random.split(key)

    # Start with a demo grid
    base_grid, base_mask = _init_demo_grid(task, demo_key, initial_pair_idx)

    # Apply random permutation
    permuted_grid = _apply_permutation(base_grid, config, perm_key)

    return permuted_grid, base_mask


def _init_empty_grid(task: JaxArcTask) -> tuple[GridArray, MaskArray]:
    """Initialize empty grid using task dimensions."""

    def use_template_shape():
        template_grid = task.input_grids_examples[0]
        template_mask = task.input_masks_examples[0]
        empty_grid = jnp.zeros_like(template_grid, dtype=jnp.int32)
        return empty_grid, template_mask

    def create_default_empty():
        max_height, max_width = task.get_grid_shape()
        empty_grid = jnp.zeros((max_height, max_width), dtype=jnp.int32)
        empty_mask = jnp.zeros((max_height, max_width), dtype=jnp.bool_)
        return empty_grid, empty_mask

    return jax.lax.cond(
        task.num_train_pairs > 0,
        use_template_shape,
        create_default_empty,
    )


def _init_random_grid(
    task: JaxArcTask, config: GridInitializationConfig, key: PRNGKey
) -> tuple[GridArray, MaskArray]:
    """Initialize grids with random patterns."""

    def use_template_shape():
        template_grid = task.input_grids_examples[0]
        template_mask = task.input_masks_examples[0]

        # Generate random pattern
        random_grid = _generate_random_pattern(
            template_grid.shape, config.random_density, config.random_pattern_type, key
        )

        # Apply mask to keep pattern only in valid regions
        random_grid = jnp.where(template_mask, random_grid, 0)

        return random_grid, template_mask

    def create_default_random():
        max_height, max_width = task.get_grid_shape()
        grid_shape = (max_height, max_width)

        # Generate random pattern
        random_grid = _generate_random_pattern(
            grid_shape, config.random_density, config.random_pattern_type, key
        )

        empty_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        return random_grid, empty_mask

    return jax.lax.cond(
        task.num_train_pairs > 0,
        use_template_shape,
        create_default_random,
    )


def _apply_permutation(
    grid: GridArray, config: GridInitializationConfig, key: PRNGKey
) -> GridArray:
    """Apply simple permutations to grid."""
    # Check if we have any permutation types
    has_rotate = "rotate" in config.permutation_types
    has_reflect = "reflect" in config.permutation_types
    has_color_remap = "color_remap" in config.permutation_types

    if not (has_rotate or has_reflect or has_color_remap):
        return grid

    # Select random permutation type
    perm_choice = jax.random.randint(key, (), 0, 3)

    return jax.lax.switch(
        perm_choice,
        [
            lambda: _apply_rotation(grid, key) if has_rotate else grid,
            lambda: _apply_reflection(grid, key) if has_reflect else grid,
            lambda: _apply_color_remap(grid, key) if has_color_remap else grid,
        ],
    )


def _apply_rotation(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply random rotation to grid."""
    if grid.ndim != 2:
        return grid

    height, width = grid.shape

    # For non-square grids, only apply 180° rotation
    if height != width:
        return jnp.rot90(grid, k=2)

    # For square grids, apply random rotation using switch for static k values
    rotation_choice = jax.random.randint(key, (), 0, 4)
    return jax.lax.switch(
        rotation_choice,
        [
            lambda: grid,  # 0° rotation (no change)
            lambda: jnp.rot90(grid, k=1),  # 90° clockwise
            lambda: jnp.rot90(grid, k=2),  # 180°
            lambda: jnp.rot90(grid, k=3),  # 270° clockwise
        ],
    )


def _apply_reflection(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply random reflection to grid."""
    if grid.ndim != 2:
        return grid

    reflection_choice = jax.random.randint(key, (), 0, 2)
    return jax.lax.switch(
        reflection_choice,
        [
            lambda: jnp.fliplr(grid),  # Horizontal flip
            lambda: jnp.flipud(grid),  # Vertical flip
        ],
    )


def _apply_color_remap(grid: GridArray, key: PRNGKey) -> GridArray:
    """Apply color remapping while preserving structure."""
    if grid.ndim != 2:
        return grid

    # Clamp to valid ARC color range
    grid_clamped = jnp.clip(grid, 0, 9)

    # Create simple color mapping
    arc_colors = jnp.arange(10, dtype=jnp.int32)
    shuffled_colors = jax.random.permutation(key, arc_colors)

    # Apply mapping
    remapped_grid = shuffled_colors[grid_clamped]
    return jnp.clip(remapped_grid, 0, 9)


def _generate_random_pattern(
    shape: tuple[int, int], density: float, pattern_type: str, key: PRNGKey
) -> GridArray:
    """Generate random patterns based on type."""
    density = jnp.clip(density, 0.0, 1.0)

    # Select pattern type
    if pattern_type == "dense":
        return _generate_dense_pattern(shape, density, key)
    # Default to sparse
    return _generate_sparse_pattern(shape, density, key)


def _generate_sparse_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate sparse random pattern."""
    mask_key, color_key = jax.random.split(key)

    # Generate random mask for pattern placement
    pattern_mask = jax.random.bernoulli(mask_key, density, shape)

    # Generate random colors (1-9, avoiding 0 for background)
    random_colors = jax.random.randint(color_key, shape, 1, 10)

    # Apply pattern
    sparse_grid = jnp.where(pattern_mask, random_colors, 0)
    return sparse_grid.astype(jnp.int32)


def _generate_dense_pattern(
    shape: tuple[int, int], density: float, key: PRNGKey
) -> GridArray:
    """Generate dense random pattern with some clustering."""
    # Start with sparse base
    sparse_key, cluster_key = jax.random.split(key)
    base_grid = _generate_sparse_pattern(shape, density * 0.7, sparse_key)

    # Add clustering by applying simple convolution
    kernel = jnp.ones((3, 3), dtype=jnp.float32)
    padded_base = jnp.pad(base_grid.astype(jnp.float32), 1, mode="constant")

    # Convolve to find neighboring patterns
    convolved = jax.scipy.signal.convolve2d(padded_base, kernel, mode="valid")

    # Create additional pattern where neighbors exist
    neighbor_mask = (convolved > 0) & (base_grid == 0)
    add_pattern = jax.random.bernoulli(cluster_key, 0.3, shape) & neighbor_mask

    # Generate colors for additional pattern
    color_key = jax.random.split(cluster_key)[0]
    additional_colors = jax.random.randint(color_key, shape, 1, 10)

    # Combine patterns
    dense_grid = jnp.where(add_pattern, additional_colors, base_grid)
    return dense_grid.astype(jnp.int32)
