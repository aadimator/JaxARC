"""
Pytest configuration and shared fixtures for JaxARC testing.

This module provides common fixtures, utilities, and configuration for all tests
in the JaxARC project, with focus on JAX-compatible testing patterns.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from hypothesis import strategies as st

from .equinox_test_utils import EquinoxMockFactory
from .jax_test_framework import JaxTransformationTester
from .jax_testing_utils import (
    JaxEquinoxTester,
    JaxShapeChecker,
    JaxTransformationValidator,
)
from .test_utils import MockDataGenerator

# Set JAX to use CPU for testing to ensure reproducibility
jax.config.update("jax_platform_name", "cpu")

# Configure Hypothesis for reasonable test performance
from hypothesis import settings

settings.register_profile("ci", max_examples=50, deadline=5000)
settings.register_profile("dev", max_examples=10, deadline=1000)
settings.load_profile("dev")


@pytest.fixture(scope="session")
def jax_key():
    """Provide a consistent JAX PRNG key for all tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def split_key(jax_key):
    """Provide a function to split JAX keys consistently."""

    def _split_key(num_keys: int = 2):
        return jax.random.split(jax_key, num_keys)

    return _split_key


@pytest.fixture
def small_grid_shape():
    """Standard small grid shape for testing."""
    return (5, 5)


@pytest.fixture
def medium_grid_shape():
    """Standard medium grid shape for testing."""
    return (10, 10)


@pytest.fixture
def large_grid_shape():
    """Standard large grid shape for testing."""
    return (30, 30)


# Hypothesis strategies for JAX arrays
@st.composite
def jax_arrays(draw, shape=None, dtype=jnp.int32, min_value=0, max_value=9):
    """Generate JAX arrays with specified constraints."""
    if shape is None:
        height = draw(st.integers(min_value=1, max_value=10))
        width = draw(st.integers(min_value=1, max_value=10))
        shape = (height, width)

    if dtype == jnp.int32:
        elements = st.integers(min_value=min_value, max_value=max_value)
    elif dtype == jnp.float32:
        elements = st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    elif dtype == jnp.bool_:
        elements = st.booleans()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    array_data = draw(
        st.lists(elements, min_size=shape[0] * shape[1], max_size=shape[0] * shape[1])
    )
    return jnp.array(array_data, dtype=dtype).reshape(shape)


@st.composite
def grid_arrays(draw, max_height=10, max_width=10):
    """Generate valid ARC grid arrays (values 0-9)."""
    height = draw(st.integers(min_value=1, max_value=max_height))
    width = draw(st.integers(min_value=1, max_value=max_width))
    return draw(
        jax_arrays(shape=(height, width), dtype=jnp.int32, min_value=0, max_value=9)
    )


@st.composite
def mask_arrays(draw, shape=None):
    """Generate boolean mask arrays."""
    if shape is None:
        height = draw(st.integers(min_value=1, max_value=10))
        width = draw(st.integers(min_value=1, max_value=10))
        shape = (height, width)
    return draw(jax_arrays(shape=shape, dtype=jnp.bool_, min_value=0, max_value=1))


@st.composite
def selection_arrays(draw, shape=None):
    """Generate continuous selection arrays (values 0.0-1.0)."""
    if shape is None:
        height = draw(st.integers(min_value=1, max_value=10))
        width = draw(st.integers(min_value=1, max_value=10))
        shape = (height, width)
    return draw(
        jax_arrays(shape=shape, dtype=jnp.float32, min_value=0.0, max_value=1.0)
    )


@pytest.fixture
def mock_grid():
    """Provide a mock Grid instance for testing."""
    return EquinoxMockFactory.create_mock_grid()


@pytest.fixture
def mock_task():
    """Provide a mock JaxArcTask instance for testing."""
    return EquinoxMockFactory.create_mock_jax_arc_task()


@pytest.fixture
def mock_action():
    """Provide a mock ARCLEAction instance for testing."""
    return EquinoxMockFactory.create_mock_arcle_action()


@pytest.fixture
def mock_data_generator():
    """Provide MockDataGenerator instance for testing."""
    return MockDataGenerator()


@pytest.fixture
def jax_transformation_tester():
    """Provide a factory for JaxTransformationTester."""

    def _create_tester(func, test_inputs, test_kwargs=None):
        return JaxTransformationTester(func, test_inputs, test_kwargs)

    return _create_tester


@pytest.fixture
def jax_transformation_validator():
    """Provide JaxTransformationValidator for testing."""
    return JaxTransformationValidator


@pytest.fixture
def jax_shape_checker():
    """Provide JaxShapeChecker for testing."""
    return JaxShapeChecker


@pytest.fixture
def jax_equinox_tester():
    """Provide JaxEquinoxTester for testing."""
    return JaxEquinoxTester


@pytest.fixture
def equinox_module_tester():
    """Provide a factory for testing Equinox modules."""

    def _create_tester(module_class, valid_init_args):
        from .equinox_test_utils import EquinoxModuleTester

        return EquinoxModuleTester(module_class, valid_init_args)

    return _create_tester


@pytest.fixture
def chex_assert():
    """Provide chex assertion utilities."""
    return chex


# Export strategies for use in tests
__all__ = [
    "grid_arrays",
    "jax_arrays",
    "mask_arrays",
    "selection_arrays",
]
