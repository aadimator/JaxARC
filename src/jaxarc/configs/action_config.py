from __future__ import annotations

import equinox as eqx
from loguru import logger
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_float_range,
    validate_positive_int,
    validate_string_choice,
)


class ActionConfig(eqx.Module):
    """Configuration for action space and validation.

    This config contains all settings related to action handling,
    validation, and operation constraints, including dynamic action space control.
    """

    # Operation parameters
    max_operations: int = 35
    allowed_operations: tuple[int, ...] | None = None

    # Validation settings
    validate_actions: bool = True
    allow_invalid_actions: bool = False

    # Dynamic action space control settings
    dynamic_action_filtering: bool = False
    context_dependent_operations: bool = False
    invalid_operation_policy: str = "clip"

    # Compatibility attribute used by validate method in monolith
    selection_threshold: float = 1.0

    def __init__(self, **kwargs):
        allowed_operations = kwargs.get("allowed_operations")
        if allowed_operations is not None and not isinstance(allowed_operations, tuple):
            allowed_operations = (
                tuple(allowed_operations) if allowed_operations else None
            )

        self.max_operations = kwargs.get("max_operations", 35)
        self.allowed_operations = allowed_operations
        self.validate_actions = kwargs.get("validate_actions", True)
        self.allow_invalid_actions = kwargs.get("allow_invalid_actions", False)
        self.dynamic_action_filtering = kwargs.get("dynamic_action_filtering", False)
        self.context_dependent_operations = kwargs.get(
            "context_dependent_operations", False
        )
        self.invalid_operation_policy = kwargs.get("invalid_operation_policy", "clip")
        self.selection_threshold = kwargs.get("selection_threshold", 1.0)

    def validate(self) -> tuple[str, ...]:
        """Validate action configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            validate_float_range(
                self.selection_threshold, "selection_threshold", 0.0, 1.0
            )

            validate_positive_int(self.max_operations, "max_operations")
            if self.max_operations > 100:
                logger.warning(f"max_operations is very large: {self.max_operations}")

            if self.allowed_operations is not None:
                if not isinstance(self.allowed_operations, tuple):
                    errors.append("allowed_operations must be a tuple or None")
                elif not self.allowed_operations:
                    errors.append("allowed_operations cannot be empty if specified")
                else:
                    for i, op in enumerate(self.allowed_operations):
                        if not isinstance(op, int):
                            errors.append(f"allowed_operations[{i}] must be an integer")
                        elif not 0 <= op < self.max_operations:
                            errors.append(
                                f"allowed_operations[{i}] must be in range [0, {self.max_operations})"
                            )

                    if len(set(self.allowed_operations)) != len(
                        self.allowed_operations
                    ):
                        duplicates = [
                            op
                            for op in set(self.allowed_operations)
                            if self.allowed_operations.count(op) > 1
                        ]
                        errors.append(
                            f"allowed_operations contains duplicate operations: {duplicates}"
                        )

            valid_policies = ("clip", "reject", "passthrough", "penalize")
            validate_string_choice(
                self.invalid_operation_policy,
                "invalid_operation_policy",
                valid_policies,
            )

            if not self.validate_actions and not self.allow_invalid_actions:
                logger.warning(
                    "allow_invalid_actions has no effect when validate_actions=False"
                )

            if not self.dynamic_action_filtering and self.context_dependent_operations:
                logger.warning(
                    "context_dependent_operations has no effect when dynamic_action_filtering=False"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"ActionConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> ActionConfig:
        """Create action config from Hydra DictConfig."""
        allowed_ops = cfg.get("allowed_operations")
        if allowed_ops is not None and not isinstance(allowed_ops, tuple):
            allowed_ops = tuple(allowed_ops) if allowed_ops else None

        return cls(
            allowed_operations=allowed_ops,
            max_operations=cfg.get("num_operations", 35),
            validate_actions=cfg.get("validate_actions", True),
            allow_invalid_actions=not cfg.get("clip_invalid_actions", True),
            dynamic_action_filtering=cfg.get("dynamic_action_filtering", False),
            context_dependent_operations=cfg.get("context_dependent_operations", False),
            invalid_operation_policy=cfg.get("invalid_operation_policy", "clip"),
            selection_threshold=cfg.get("selection_threshold", 1.0),
        )
