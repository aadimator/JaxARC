from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.spaces import Space


class MultiBinary(Space):
    """
    A JAX-jittable MultiBinary space.

    This space represents a multi-dimensional binary space.
    """

    def __init__(self, n: int | tuple[int, ...]):
        """Initialize the space.

        Args:
            n: The shape of the space.
        """
        if isinstance(n, int):
            self.n = (n,)
        else:
            self.n = n

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample a random element from the space."""
        return jax.random.bernoulli(key, 0.5, shape=self.n).astype(jnp.int32)

    def contains(self, x: chex.Array) -> chex.Array:
        """Check if an element is a member of the space."""
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)

        return x.shape == self.n and jnp.all((x == 0) | (x == 1))
