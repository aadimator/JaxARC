from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_float_range,
)


class RewardConfig(eqx.Module):
    """Configuration for reward calculation.

    This config contains all settings related to reward computation,
    penalties, bonuses, and reward shaping with mode-aware enhancements.
    """

    # Basic reward settings
    step_penalty: float = -0.01
    success_bonus: float = 10.0
    similarity_weight: float = 1.0
    unsolved_submission_penalty: float = 0.0

    def validate(self) -> tuple[str, ...]:
        """Validate reward configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            validate_float_range(self.step_penalty, "step_penalty", -10.0, 1.0)
            validate_float_range(self.success_bonus, "success_bonus", -100.0, 1000.0)
            validate_float_range(
                self.similarity_weight, "similarity_weight", 0.0, 100.0
            )
            validate_float_range(
                self.unsolved_submission_penalty,
                "unsolved_submission_penalty",
                -1000.0,
                0.0,
            )
        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"RewardConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> RewardConfig:
        """Create reward config from Hydra DictConfig."""
        return cls(
            step_penalty=cfg.get("step_penalty", -0.01),
            success_bonus=cfg.get("success_bonus", 10.0),
            similarity_weight=cfg.get("similarity_weight", 1.0),
            unsolved_submission_penalty=cfg.get("unsolved_submission_penalty", 0.0),
        )
