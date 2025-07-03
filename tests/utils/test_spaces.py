from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.utils.spaces import MultiBinary


class TestMultiBinary:
    """Test suite for the MultiBinary space."""

    def test_creation(self):
        """Test that the space can be created with different n values."""
        space1 = MultiBinary(10)
        assert space1.n == (10,)

        space2 = MultiBinary((2, 3))
        assert space2.n == (2, 3)

    def test_sample(self):
        """Test that sampling returns an array of the correct shape and type."""
        space = MultiBinary(5)
        key = jax.random.PRNGKey(0)
        sample = space.sample(key)

        assert isinstance(sample, jnp.ndarray)
        assert sample.shape == (5,)
        assert sample.dtype == jnp.int32
        assert jnp.all((sample == 0) | (sample == 1))

    def test_contains(self):
        """Test that the contains method correctly identifies valid elements."""
        space = MultiBinary(4)

        assert space.contains(jnp.array([0, 1, 0, 1]))
        assert not space.contains(jnp.array([0, 1, 2, 1]))
        assert not space.contains(jnp.array([0, 1, 0]))

    def test_jit_compatibility(self):
        """Test that the sample and contains methods are JAX-jittable."""
        space = MultiBinary(8)
        key = jax.random.PRNGKey(0)

        # JIT the sample method
        jit_sample = jax.jit(space.sample)
        sample = jit_sample(key)
        assert sample.shape == (8,)

        # JIT the contains method
        jit_contains = jax.jit(space.contains)
        assert jit_contains(sample)
        assert not jit_contains(sample + 2)
