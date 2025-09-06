"""JAX-compatible action and observation spaces for ARC (Stoa-based).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from jaxarc.types import NUM_COLORS, NUM_OPERATIONS

T = TypeVar("T")

# === CORE STOA SPACE HIERARCHY (Essential classes only) ===

class Space(abc.ABC, Generic[T]):
    """Abstract base class for RL environment spaces."""
    
    @property
    @abc.abstractmethod 
    def shape(self) -> tuple[int, ...] | None:
        """Shape of values in this space."""
        ...
    
    @property
    @abc.abstractmethod
    def dtype(self) -> jnp.dtype | None:
        """Dtype of values in this space."""
        ...
        
    @abc.abstractmethod
    def sample(self, key: PRNGKey) -> T:
        """Sample a random value from this space."""
        ...
        
    @abc.abstractmethod
    def contains(self, value: Any) -> Array:
        """Check if value belongs to this space."""
        ...

    def tree_flatten(self) -> tuple[Sequence[Any], dict[str, Any]]:
        """Flatten this space for JAX tree operations."""
        return [], self.__dict__

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: Sequence[Any]) -> Space:
        """Unflatten this space from JAX tree operations."""
        return cls(**aux_data)


class BoundedArraySpace(Space[Array]):
    """Array space with min/max bounds."""
    
    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = jnp.float32, 
                 minimum: ArrayLike = -jnp.inf, maximum: ArrayLike = jnp.inf, name: str = ""):
        self._shape = tuple(shape)
        self._dtype = jnp.dtype(dtype)
        self._minimum = jnp.asarray(minimum, dtype=self._dtype)
        self._maximum = jnp.asarray(maximum, dtype=self._dtype)
        self._name = name
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    @property 
    def dtype(self) -> jnp.dtype:
        return self._dtype
        
    def sample(self, key: PRNGKey) -> Array:
        return jax.random.uniform(key, self.shape, minval=self._minimum, maxval=self._maximum)
        
    def contains(self, value: Any) -> Array:
        if not isinstance(value, (jnp.ndarray, Array)):
            return jnp.array(False)
        shape_ok = value.shape == self.shape
        bounds_ok = jnp.all((value >= self._minimum) & (value <= self._maximum))
        return jnp.array(shape_ok) & bounds_ok


class DiscreteSpace(BoundedArraySpace):
    """Discrete scalar space with values 0 to num_values-1."""
    
    def __init__(self, num_values: int, name: str = ""):
        super().__init__(shape=(), dtype=jnp.int32, minimum=0, maximum=num_values-1, name=name)
        self.num_values = num_values
        
    def sample(self, key: PRNGKey) -> Array:
        return jax.random.randint(key, (), 0, self.num_values)


class DictSpace(Space[dict[str, Any]]):
    """Dictionary of spaces."""
    
    def __init__(self, spaces: dict[str, Space], name: str = ""):
        self._spaces = spaces
        self._name = name
    
    @property
    def shape(self) -> None:
        return None
    
    @property
    def dtype(self) -> None:
        return None
        
    def sample(self, key: PRNGKey) -> dict[str, Any]:
        keys = jax.random.split(key, len(self._spaces))
        return {name: space.sample(keys[i]) for i, (name, space) in enumerate(self._spaces.items())}
        
    def contains(self, value: Any) -> Array:
        if not isinstance(value, dict):
            return jnp.array(False)
        return jnp.array(all(space.contains(value[name]) for name, space in self._spaces.items()))


# === ARC-SPECIFIC SPACES ===

class GridSpace(BoundedArraySpace):
    """ARC grid space with proper color bounds."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        super().__init__(
            shape=(max_height, max_width),
            dtype=jnp.int32,
            minimum=-1,  # Background/padding
            maximum=NUM_COLORS - 1,  # ARC colors 0-9
            name="arc_grid"
        )


class SelectionSpace(BoundedArraySpace):
    """Binary selection mask space."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        super().__init__(
            shape=(max_height, max_width),
            dtype=jnp.bool_,
            minimum=False,
            maximum=True,
            name="selection_mask"
        )


class ARCActionSpace(DictSpace):
    """Complete ARC action space (operation + selection)."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        spaces = {
            "operation": DiscreteSpace(NUM_OPERATIONS, "operation"),
            "selection": SelectionSpace(max_height, max_width)
        }
        super().__init__(spaces, "arc_action")


# === LEGACY COMPATIBILITY (minimal) ===

class MultiBinary(BoundedArraySpace):
    """Legacy compatibility for existing MultiBinary usage."""
    
    def __init__(self, n: int | tuple[int, ...]):
        shape = (n,) if isinstance(n, int) else tuple(n)
        super().__init__(shape=shape, dtype=jnp.int32, minimum=0, maximum=1, name="multibinary")
        self.n = shape
    
    def sample(self, key: PRNGKey) -> Array:
        return jax.random.bernoulli(key, 0.5, shape=self.shape).astype(jnp.int32)


# Register spaces as JAX pytrees  
jax.tree_util.register_pytree_node(BoundedArraySpace, BoundedArraySpace.tree_flatten, BoundedArraySpace.tree_unflatten)
jax.tree_util.register_pytree_node(DiscreteSpace, DiscreteSpace.tree_flatten, DiscreteSpace.tree_unflatten)
jax.tree_util.register_pytree_node(DictSpace, DictSpace.tree_flatten, DictSpace.tree_unflatten)


__all__ = [
    "ARCActionSpace",
    "BoundedArraySpace", 
    "DictSpace",
    "DiscreteSpace", 
    "GridSpace", 
    "MultiBinary",  # Legacy compatibility
    "SelectionSpace", 
    "Space", 
]
