"""Simplified Action Space Controller (operations 0-34 only).

This module manages a static set of ARC grid operations (IDs 0-34). All prior
control / pair-switching operations (>=35) have been removed.

Features:
- Allowed-operation mask construction from config/state
- Operation ID validation (range + mask membership)
- Invalid-operation handling policies: clip | reject | penalize | passthrough
- Logit masking, valid-op sampling, probability filtering (JAX compatible)

Example:
    controller = ActionSpaceController()
    mask = controller.get_allowed_operations(state, params.action)
    ok, msg = controller.validate_operation(12, state, params.action)
    filtered = controller.filter_invalid_operation(99, state, params.action)  # clipped
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.configs.action_config import ActionConfig

from ..state import State
from ..utils.jax_types import NUM_OPERATIONS, OperationMask


class ActionSpaceController:
    """Controller for ARC operation availability (0-34)."""

    def get_allowed_operations(
        self, state: State, config: ActionConfig
    ) -> OperationMask:
        allowed_ops = getattr(config, "allowed_operations", None)
        if isinstance(allowed_ops, tuple) and len(allowed_ops) > 0:
            idx = jnp.asarray(allowed_ops, dtype=jnp.int32)
            idx = jnp.clip(idx, 0, NUM_OPERATIONS - 1)
            base = jnp.zeros((NUM_OPERATIONS,), dtype=jnp.bool_).at[idx].set(True)
        else:
            base = jnp.ones((NUM_OPERATIONS,), dtype=jnp.bool_)
        if hasattr(state, "allowed_operations_mask"):
            base = jnp.logical_and(base, state.allowed_operations_mask)
        return base

    def validate_operation(
        self, operation_id: int, state: State, config: ActionConfig
    ) -> tuple[bool, str | None]:
        if not (0 <= operation_id < NUM_OPERATIONS):
            return (
                False,
                f"Operation ID {operation_id} out of range [0,{NUM_OPERATIONS - 1}]",
            )
        mask = self.get_allowed_operations(state, config)
        if not bool(mask[operation_id]):
            return False, f"Operation {operation_id} not allowed by current mask"
        return True, None

    def validate_operation_jax(
        self, operation_id: jnp.ndarray, state: State, config: ActionConfig
    ) -> jnp.ndarray:
        mask = self.get_allowed_operations(state, config)
        in_range = (operation_id >= 0) & (operation_id < NUM_OPERATIONS)
        safe = jnp.clip(operation_id, 0, NUM_OPERATIONS - 1)
        return in_range & mask[safe]

    def validate_operation_range_jax(self, operation_id: jnp.ndarray) -> jnp.ndarray:
        return (operation_id >= 0) & (operation_id < NUM_OPERATIONS)

    def _find_nearest_valid_operation(
        self, op_id: jnp.ndarray, mask: OperationMask
    ) -> jnp.ndarray:
        ids = jnp.arange(NUM_OPERATIONS)
        dists = jnp.abs(ids - op_id)
        dists = jnp.where(mask, dists, jnp.inf)
        idx = jnp.argmin(dists)
        return jnp.where(jnp.any(mask), idx, jnp.array(0, dtype=jnp.int32))

    def filter_invalid_operation(
        self, operation_id: int | jnp.ndarray, state: State, config: ActionConfig
    ) -> int | jnp.ndarray:
        if isinstance(operation_id, int):
            arr = jnp.array(operation_id, dtype=jnp.int32)
            single = True
        else:
            arr = operation_id.astype(jnp.int32)
            single = False
        mask = self.get_allowed_operations(state, config)
        valid = self.validate_operation_jax(arr, state, config)
        policy = getattr(config, "invalid_operation_policy", "clip")
        clipped = jnp.clip(arr, 0, NUM_OPERATIONS - 1)
        if policy in ("clip", "penalize"):
            repl = self._find_nearest_valid_operation(clipped, mask)
            out = jnp.where(valid, arr, repl)
        elif policy == "reject":
            out = jnp.where(valid, arr, jnp.array(-1, dtype=jnp.int32))
        elif policy == "passthrough":
            out = arr
        else:
            repl = self._find_nearest_valid_operation(clipped, mask)
            out = jnp.where(valid, arr, repl)
        return int(out) if single else out

    def filter_invalid_operation_jax(
        self, operation_id: jnp.ndarray, state: State, config: ActionConfig
    ) -> jnp.ndarray:
        arr = operation_id.astype(jnp.int32)
        mask = self.get_allowed_operations(state, config)
        valid = self.validate_operation_jax(arr, state, config)
        policy = getattr(config, "invalid_operation_policy", "clip")
        clipped = jnp.clip(arr, 0, NUM_OPERATIONS - 1)
        if policy in ("clip", "penalize"):
            repl = self._find_nearest_valid_operation(clipped, mask)
            out = jnp.where(valid, arr, repl)
        elif policy == "reject":
            out = jnp.where(valid, arr, jnp.array(-1, dtype=jnp.int32))
        elif policy == "passthrough":
            out = arr
        else:
            repl = self._find_nearest_valid_operation(clipped, mask)
            out = jnp.where(valid, arr, repl)
        return out

    def handle_invalid_operation_jax(
        self, operation_id: jnp.ndarray, state: State, config: ActionConfig
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        valid = self.validate_operation_jax(operation_id, state, config)
        filtered = self.filter_invalid_operation_jax(operation_id, state, config)
        return filtered, jnp.logical_not(valid)

    def get_validation_penalty_jax(
        self,
        operation_id: jnp.ndarray,
        state: State,
        config: ActionConfig,
        penalty_value: float = -1.0,
    ) -> jnp.ndarray:
        if getattr(config, "invalid_operation_policy", "clip") != "penalize":
            return jnp.array(0.0)
        valid = self.validate_operation_jax(operation_id, state, config)
        return jnp.where(valid, 0.0, penalty_value)

    def apply_operation_mask_jax(
        self,
        action_logits: jnp.ndarray,
        state: State,
        config: ActionConfig,
        mask_value: float = -jnp.inf,
    ) -> jnp.ndarray:
        mask = self.get_allowed_operations(state, config)
        bmask = jnp.broadcast_to(mask, action_logits.shape)
        return jnp.where(bmask, action_logits, mask_value)

    def get_valid_operations_indices_jax(
        self, state: State, config: ActionConfig
    ) -> jnp.ndarray:
        mask = self.get_allowed_operations(state, config)
        ids = jnp.arange(NUM_OPERATIONS, dtype=jnp.int32)
        return jnp.where(mask, ids, jnp.array(-1, dtype=jnp.int32))

    def sample_valid_operation_jax(
        self, key: jnp.ndarray, state: State, config: ActionConfig
    ) -> jnp.ndarray:
        mask = self.get_allowed_operations(state, config)
        probs = jnp.where(mask, 1.0, 0.0)
        total = jnp.sum(mask)
        probs = jnp.where(
            total > 0, probs / total, jnp.ones(NUM_OPERATIONS) / NUM_OPERATIONS
        )
        return jax.random.categorical(key, jnp.log(probs + 1e-8))

    def get_next_valid_operation(
        self,
        operation_id: int,
        state: State,
        config: ActionConfig,
        direction: str = "forward",
    ) -> int:
        mask = self.get_allowed_operations(state, config)
        step = 1 if direction == "forward" else -1
        start = jnp.array(operation_id, dtype=jnp.int32)
        offsets = jnp.arange(1, NUM_OPERATIONS) * step
        candidates = (start + offsets) % NUM_OPERATIONS
        valid = mask[candidates]
        first = jnp.argmax(valid)
        has_any = jnp.any(valid)
        choice = jnp.where(has_any, candidates[first], jnp.array(-1, dtype=jnp.int32))
        return int(choice)

    def batch_validate_operations_jax(
        self, operation_ids: jnp.ndarray, state: State, config: ActionConfig
    ) -> jnp.ndarray:
        return jax.vmap(lambda op: self.validate_operation_jax(op, state, config))(
            operation_ids
        )

    def create_operation_filter_mask_jax(
        self,
        operation_logits: jnp.ndarray,
        state: State,
        config: ActionConfig,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        masked = self.apply_operation_mask_jax(
            operation_logits, state, config, mask_value=-jnp.inf
        )
        scaled = masked / temperature
        return jax.nn.softmax(scaled, axis=-1)

    def create_action_mask_for_agent(self, state: State, config: ActionConfig) -> dict:
        mask = self.get_allowed_operations(state, config)
        categories = {
            "fill": list(range(10)),
            "flood_fill": list(range(10, 20)),
            "movement": list(range(20, 24)),
            "transformation": list(range(24, 28)),
            "editing": list(range(28, 32)),
            "special": list(range(32, 35)),
        }
        breakdown = {}
        for name, ids in categories.items():
            arr = mask[jnp.array(ids)]
            breakdown[name] = {
                "allowed": int(jnp.sum(arr)),
                "total": len(ids),
                "mask": arr.tolist(),
            }
        return {
            "operation_mask": mask.tolist(),
            "total_allowed": int(jnp.sum(mask)),
            "total_operations": NUM_OPERATIONS,
            "has_restrictions": int(jnp.sum(mask)) < NUM_OPERATIONS,
            "category_breakdown": breakdown,
            "context_info": {
                "step_count": int(state.step_count),
            },
            "validation_policy": getattr(config, "invalid_operation_policy", "clip"),
        }

    def is_grid_operation(self, operation_id: int) -> bool:
        return 0 <= operation_id < NUM_OPERATIONS
