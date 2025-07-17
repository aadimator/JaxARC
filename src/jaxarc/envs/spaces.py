"""JAX-compatible action and observation spaces for ARC environment.

This module provides JAX-jittable space definitions for the ARC environment,
including action spaces and observation spaces. These are core environment
components that define the structure of actions and observations.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp


class Space:
    """Minimal jittable class for abstract JAX space.
    
    This is a base class for defining action and observation spaces
    that are compatible with JAX transformations.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample a random element from the space.
        
        Args:
            rng: JAX PRNG key for random sampling
            
        Returns:
            Random sample from the space
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def contains(self, x: jnp.ndarray) -> bool:
        """Check if an element is a member of the space.
        
        Args:
            x: Element to check
            
        Returns:
            True if element is in the space, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class MultiBinary(Space):
    """A JAX-jittable MultiBinary space.

    This space represents a multi-dimensional binary space, commonly used
    for action spaces where multiple binary actions can be taken simultaneously.
    
    Examples:
        ```python
        from jaxarc.envs.spaces import MultiBinary
        import jax
        
        # Create a 2D binary space
        space = MultiBinary((10, 10))
        
        # Sample from the space
        key = jax.random.PRNGKey(42)
        sample = space.sample(key)
        
        # Check if an element is in the space
        valid = space.contains(sample)
        ```
    """

    def __init__(self, n: int | tuple[int, ...]):
        """Initialize the MultiBinary space.

        Args:
            n: The shape of the space. If int, creates a 1D space of that size.
               If tuple, creates a multi-dimensional space with that shape.
               
        Example:
            ```python
            # 1D binary space with 5 elements
            space1 = MultiBinary(5)
            
            # 2D binary space with shape (10, 10)
            space2 = MultiBinary((10, 10))
            ```
        """
        if isinstance(n, int):
            self.n = (n,)
        else:
            self.n = tuple(n)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample a random element from the space.
        
        Args:
            key: JAX PRNG key for random sampling
            
        Returns:
            Random binary array with shape self.n and dtype int32
        """
        return jax.random.bernoulli(key, 0.5, shape=self.n).astype(jnp.int32)

    def contains(self, x: chex.Array) -> chex.Array:
        """Check if an element is a member of the space.
        
        Args:
            x: Array to check
            
        Returns:
            Boolean indicating if x is in the space (correct shape and binary values)
        """
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)

        # Check shape and that all values are 0 or 1
        shape_match = x.shape == self.n
        binary_values = jnp.all((x == 0) | (x == 1))
        
        return shape_match & binary_values

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the space."""
        return self.n
        
    def __repr__(self) -> str:
        """String representation of the space."""
        return f"MultiBinary({self.n})"