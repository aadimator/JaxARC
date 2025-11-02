"""
Pytest configuration and common fixtures for JaxARC tests.

This module provides shared fixtures and configuration for the entire test suite,
focusing on JAX compatibility and common test data structures.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from jaxarc import JaxArcConfig
from jaxarc.registration import available_task_ids, make
from jaxarc.types import Grid, JaxArcTask, TaskPair


@pytest.fixture
def prng_key() -> PRNGKeyArray:
    """Provide a JAX PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_grid() -> Grid:
    """Provide a standard test grid with known properties."""
    # Create a simple 3x3 grid with some pattern
    data = jnp.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=jnp.int32)

    # Create mask (all cells are valid)
    mask = jnp.ones((3, 3), dtype=jnp.bool_)

    return Grid(data=data, mask=mask)


@pytest.fixture
def sample_task_pair(sample_grid) -> TaskPair:
    """Provide a sample TaskPair for testing."""
    # Create input and output grids (output is a simple transformation)
    input_grid = sample_grid

    # Output grid is input with colors incremented by 1 (mod 10)
    output_data = (sample_grid.data + 1) % 10
    output_grid = Grid(data=output_data, mask=sample_grid.mask)

    return TaskPair(input=input_grid, output=output_grid)


@pytest.fixture
def sample_task(sample_task_pair) -> JaxArcTask:
    """Provide a complete JaxArcTask for testing."""
    # Create training and test pairs
    train_pairs = [sample_task_pair]  # Single training pair for simplicity
    test_pairs = [sample_task_pair]  # Single test pair for simplicity

    return JaxArcTask(task_id="test_task_001", train=train_pairs, test=test_pairs)


@pytest.fixture
def default_config() -> JaxArcConfig:
    """Provide a valid JaxArcConfig for testing."""
    return JaxArcConfig()


@pytest.fixture
def large_grid() -> Grid:
    """Provide a larger grid for testing edge cases."""
    # Create a 10x10 grid with checkerboard pattern
    data = jnp.zeros((10, 10), dtype=jnp.int32)
    data = data.at[::2, ::2].set(1)  # Set even positions to 1
    data = data.at[1::2, 1::2].set(1)  # Set odd positions to 1

    mask = jnp.ones((10, 10), dtype=jnp.bool_)

    return Grid(data=data, mask=mask)


@pytest.fixture
def empty_grid() -> Grid:
    """Provide an empty grid for testing."""
    data = jnp.zeros((5, 5), dtype=jnp.int32)
    mask = jnp.ones((5, 5), dtype=jnp.bool_)

    return Grid(data=data, mask=mask)


@pytest.fixture
def masked_grid() -> Grid:
    """Provide a grid with some masked (invalid) cells."""
    data = jnp.ones((4, 4), dtype=jnp.int32)

    # Create mask with some invalid cells
    mask = jnp.ones((4, 4), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(False)  # Top-left invalid
    mask = mask.at[3, 3].set(False)  # Bottom-right invalid

    return Grid(data=data, mask=mask)


@pytest.fixture
def sample_env_and_params(default_config):
    """Create sample environment and parameters for testing."""
    task_ids = available_task_ids("Mini", config=default_config)
    task_id = task_ids[0] if task_ids else "all"
    env, _ = make(f"Mini-{task_id}", config=default_config)
    return env, env.params


@pytest.fixture
def sample_state(sample_env_and_params):
    """Create sample state for testing."""
    env, env_params = sample_env_and_params
    key = jax.random.PRNGKey(42)
    state, timestep = env.reset(key)
    return state


# Configure pytest settings
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "fast: marks tests as fast/unit-level (select with '-m fast')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "jax_transform: marks tests that verify JAX transformations"
    )


# Configure JAX for testing
def pytest_sessionstart(session):
    """Configure JAX settings for the test session."""
    # Disable JIT compilation for easier debugging during tests
    # Individual tests can re-enable JIT as needed
    jax.config.update("jax_disable_jit", True)

    # Use CPU platform for consistent test results
    jax.config.update("jax_platform_name", "cpu")
