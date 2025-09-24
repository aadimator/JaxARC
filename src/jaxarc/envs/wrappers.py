"""
Wrappers for JaxARC environments (simplified with clean delegation).

This module implements clean wrappers following Stoa delegation patterns:
- Core Environment: Only knows about Action objects (mask-based selections)
- Wrappers: Convert user-friendly formats to masks or reshape observations

- PointActionWrapper: Converts {"operation": op, "row": r, "col": c} dicts to mask actions
- BboxActionWrapper: Converts {"operation": op, "r1": r1, "c1": c1, "r2": r2, "c2": c2} dicts to mask actions
- FlattenDictActionWrapper: Flattens a DictSpace of Discrete sub-spaces into a single Discrete action space
- AddChannelDimWrapper: Adds a trailing channel dimension to observations

Usage:
    ```python
    from jaxarc.registration import make
    from jaxarc.envs.wrappers import BboxActionWrapper

    # Create base environment (handles Action only)
    env, env_params = make("Mini")

    # Wrap with action wrapper (converts bbox to mask)
    env = BboxActionWrapper(env)

    # Use normal environment API with bbox actions
    state, timestep = env.reset(key, env_params=env_params)
    action = {"operation": 15, "r1": 2, "c1": 3, "r2": 7, "c2": 8}
    state, timestep = env.step(state, action, env_params=env_params)

    # The wrapper handles the conversion:
    # bbox dict -> Action with rectangular mask -> core environment
    ```
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from stoa.core_wrappers.wrapper import Wrapper

from ..state import State
from ..types import EnvParams, TimeStep
from .actions import Action, create_action
from .spaces import BoundedArraySpace, DictSpace, DiscreteSpace


def _point_to_mask(point_action: dict, grid_shape: tuple[int, int]) -> Action:
    """Convert point action dict to mask action.

    Args:
        point_action: Dict with keys 'operation', 'row', 'col'
        grid_shape: Shape of the grid (height, width)

    Returns:
        Action with single point selected
    """
    operation = point_action["operation"]
    row = point_action["row"]
    col = point_action["col"]
    height, width = grid_shape

    # Create mask with single point
    mask = jnp.zeros((height, width), dtype=jnp.bool_)

    # Clip coordinates to valid range
    valid_row = jnp.clip(row, 0, height - 1)
    valid_col = jnp.clip(col, 0, width - 1)

    # Set the point in the mask
    mask = mask.at[valid_row, valid_col].set(True)

    return create_action(operation, mask)


def _bbox_to_mask(bbox_action: dict, grid_shape: tuple[int, int]) -> Action:
    """Convert bounding box action dict to mask action.

    Args:
        bbox_action: Dict with keys 'operation', 'r1', 'c1', 'r2', 'c2'
        grid_shape: Shape of the grid (height, width)

    Returns:
        Action with rectangular region selected
    """
    operation = bbox_action["operation"]
    r1, c1 = bbox_action["r1"], bbox_action["c1"]
    r2, c2 = bbox_action["r2"], bbox_action["c2"]
    height, width = grid_shape

    # Clip coordinates to valid range
    r1 = jnp.clip(r1, 0, height - 1)
    c1 = jnp.clip(c1, 0, width - 1)
    r2 = jnp.clip(r2, 0, height - 1)
    c2 = jnp.clip(c2, 0, width - 1)

    # Ensure proper ordering (min, max)
    min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
    min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)

    # Create coordinate meshes
    rows = jnp.arange(height)
    cols = jnp.arange(width)
    row_mesh, col_mesh = jnp.meshgrid(rows, cols, indexing="ij")

    # Create bbox mask (inclusive bounds)
    mask = (
        (row_mesh >= min_r)
        & (row_mesh <= max_r)
        & (col_mesh >= min_c)
        & (col_mesh <= max_c)
    )

    return create_action(operation, mask)


# Generic JIT versions using static_argnums for better performance
_jit_point_to_mask = jax.jit(_point_to_mask, static_argnums=1)
_jit_bbox_to_mask = jax.jit(_bbox_to_mask, static_argnums=1)


class PointActionWrapper(Wrapper):
    """Point action wrapper with custom action space."""

    def action_space(self, env_params: EnvParams | None = None) -> DictSpace:
        """Custom action space for point actions: (operation, row, col)."""
        # Use provided params or fall back to the environment's default params.
        p = self._env.params if env_params is None else env_params
        
        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(p)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = p.dataset.max_grid_height
        width = p.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "row": DiscreteSpace(height, dtype=jnp.int32),
                "col": DiscreteSpace(width, dtype=jnp.int32),
            },
            name="point_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert point to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_point_to_mask(action, grid_shape)

        # Delegate to underlying env using mask-based Action
        next_state, timestep = self._env.step(state, mask_action, env_params)

        # Core Environment now guarantees canonical_action/operation_id in extras.
        return next_state, timestep

class BboxActionWrapper(Wrapper):
    """Bbox action wrapper with custom action space."""

    def action_space(self, env_params: EnvParams | None = None) -> DictSpace:
        """Custom action space for bbox actions: (operation, r1, c1, r2, c2)."""
        # Use provided params or fall back to the environment's default params.
        p = self._env.params if env_params is None else env_params
        
        # Get the underlying action space to extract operation count
        base_action_space = self._env.action_space(p)
        operation_space = base_action_space.spaces["operation"]

        # Get grid dimensions from env params
        height = p.dataset.max_grid_height
        width = p.dataset.max_grid_width

        return DictSpace(
            {
                "operation": operation_space,
                "r1": DiscreteSpace(height, dtype=jnp.int32),
                "c1": DiscreteSpace(width, dtype=jnp.int32),
                "r2": DiscreteSpace(height, dtype=jnp.int32),
                "c2": DiscreteSpace(width, dtype=jnp.int32),
            },
            name="bbox_action",
        )

    def step(
        self, state: State, action: dict, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert bbox to mask and delegate."""
        grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])
        mask_action = _jit_bbox_to_mask(action, grid_shape)

        # Delegate to underlying env using mask-based Action
        next_state, timestep = self._env.step(state, mask_action, env_params)

        # Core Environment now guarantees canonical_action/operation_id in extras.
        return next_state, timestep

class FlattenDictActionWrapper(Wrapper):
    """Flatten a dictionary action space of Discrete sub-spaces into a single Discrete.

    Notes:
    - Works when the underlying action_space is a DictSpace whose sub-spaces are all
      DiscreteSpace (e.g., PointActionWrapper or BboxActionWrapper outputs).
    - This will NOT work directly with the core ARCActionSpace (it contains a mask),
      so wrap the env with a dict-discrete wrapper first.
    """

    def __init__(self, env):
        super().__init__(env)
        # Lazy-init fields; computed on first access with env or provided env_params
        self._cached_params = None
        self._action_space = None
        self.action_dims: list[int] = []
        self.num_actions: int | None = None

    def _ensure_initialized(self, env_params: EnvParams | None = None) -> None:
        p = self._env.params if env_params is None else env_params
        if (self._cached_params is p) and (self._action_space is not None):
            return

        base_space = self._env.action_space(p)
        # Accept protocol: any object with a 'spaces' mapping behaves like DictSpace
        if not hasattr(base_space, "spaces"):
            msg = "FlattenDictActionWrapper requires an action_space with a 'spaces' attribute (DictSpace-like)."
            raise ValueError(msg)

        dims: list[int] = []
        for sub in base_space.spaces.values():
            # Accept DiscreteSpace or any object exposing a category count
            if isinstance(sub, DiscreteSpace):
                n = getattr(sub, "num_values", None)
                if n is None:
                    n = getattr(sub, "n", None)
            else:
                n = getattr(sub, "num_values", None)
                if n is None:
                    n = getattr(sub, "n", None)
            if n is None:
                msg = (
                    "All sub-spaces must be discrete (provide 'num_values' or 'n'). Wrap with point/bbox wrapper first."
                )
                raise ValueError(msg)
            dims.append(int(n))

        self._cached_params = p
        self._action_space = base_space
        self.action_dims = dims
        self.num_actions = int(np.prod(self.action_dims).item()) if dims else 0

    def _unflatten_action(self, action: jax.Array, env_params: EnvParams | None = None) -> dict[str, jax.Array]:
        """Convert a flat discrete action index into a dict of discrete components."""
        self._ensure_initialized(env_params)
        assert self._action_space is not None
        assert self.num_actions is not None

        action_sizes = list(self._action_space.spaces.values())
        action_keys = list(self._action_space.spaces.keys())

        unflattened: dict[str, jax.Array] = {}
        remainder = action
        for i, (key, _space) in enumerate(zip(action_keys, action_sizes)):
            # Divisor is product of remaining dimensions
            divisor = int(np.prod(self.action_dims[i + 1 :]).item()) if (i + 1) < len(self.action_dims) else 1
            # Use JAX-friendly integer ops
            unflattened[key] = (remainder // divisor).astype(jnp.int32)
            remainder = remainder % divisor

        return unflattened

    def step(
        self, state: State, action: jax.Array, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Convert flat discrete action to dict action, then delegate to env."""
        dict_action = self._unflatten_action(action, env_params)
        next_state, timestep = self._env.step(state, dict_action, env_params)

        # Core Environment guarantees canonical_action/operation_id; avoid extras mutation
        return next_state, timestep
    def action_space(self, env_params: EnvParams | None = None) -> DiscreteSpace:
        """Return a single DiscreteSpace of size prod of dict sub-spaces."""
        self._ensure_initialized(env_params)
        assert self.num_actions is not None
        return DiscreteSpace(self.num_actions, dtype=jnp.int32, name="flattened_action")


class AddChannelDimWrapper(Wrapper):
    """Add a trailing channel dimension to observations (H, W) -> (H, W, 1)."""

    def _process_obs(self, obs: jax.Array) -> jax.Array:
        return jnp.expand_dims(obs, axis=-1)

    def observation_space(self, env_params: EnvParams | None = None) -> BoundedArraySpace:
        obs_space = self._env.observation_space(env_params)
        return BoundedArraySpace(
            minimum=obs_space.minimum,
            maximum=obs_space.maximum,
            shape=(*obs_space.shape, 1),
            dtype=obs_space.dtype,
            name=getattr(obs_space, "name", None),
        )

    def step(
        self, state: State, action, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        # Accept both canonical Action and dict-form actions; convert dict to Action here
        if isinstance(action, dict) and ("operation" in action) and ("selection" in action):
            op = jnp.asarray(action["operation"], dtype=jnp.int32)
            sel = jnp.asarray(action["selection"], dtype=jnp.bool_)
            action = create_action(op, sel)

        next_state, timestep = self._env.step(state, action, env_params)

        # Safely rebuild TimeStep with modified observation and extras
        new_obs = self._process_obs(timestep.observation)
        extras = timestep.extras
        if isinstance(extras, dict) and ("next_obs" in extras):
            # Avoid in-place mutation under JAX transforms by copying dict
            extras = dict(extras)
            extras["next_obs"] = self._process_obs(extras["next_obs"])  # type: ignore[index]

        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=new_obs,
            extras=extras,
        )
        return next_state, new_timestep

    def reset(self, rng_key: jax.Array, env_params: EnvParams | None = None) -> tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)

        new_obs = self._process_obs(timestep.observation)
        extras = timestep.extras
        if isinstance(extras, dict) and ("next_obs" in extras):
            extras = dict(extras)
            extras["next_obs"] = self._process_obs(extras["next_obs"])  # type: ignore[index]

        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=new_obs,
            extras=extras,
        )
        return state, new_timestep


__all__ = [
    "AddChannelDimWrapper",
    "BboxActionWrapper",
    "FlattenDictActionWrapper",
    "PointActionWrapper",
]
