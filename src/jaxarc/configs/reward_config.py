from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_float_range,
    validate_non_negative_int,
)


class RewardConfig(eqx.Module):
    """Configuration for reward calculation.

    This config contains all settings related to reward computation,
    penalties, bonuses, and reward shaping with mode-aware enhancements.
    """

    # Basic reward settings
    reward_on_submit_only: bool = True
    step_penalty: float = -0.01
    success_bonus: float = 10.0
    similarity_weight: float = 1.0

    efficiency_bonus_threshold: int = 50
    efficiency_bonus: float = 1.0
    unsolved_submission_penalty: float = 0.0

    # Reward clipping for stability
    clip_similarity_delta: bool = False
    similarity_delta_min: float = -0.2
    similarity_delta_max: float = 0.2

    def validate(self) -> tuple[str, ...]:
        """Validate reward configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            validate_float_range(self.step_penalty, "step_penalty", -10.0, 1.0)
            validate_float_range(self.success_bonus, "success_bonus", -100.0, 1000.0)
            validate_float_range(self.similarity_weight, "similarity_weight", 0.0, 100.0)
            validate_non_negative_int(
                self.efficiency_bonus_threshold, "efficiency_bonus_threshold"
            )
            validate_float_range(self.efficiency_bonus, "efficiency_bonus", -10.0, 10.0)
            validate_float_range(
                self.unsolved_submission_penalty,
                "unsolved_submission_penalty",
                -1000.0,
                0.0,
            )
            validate_float_range(self.similarity_delta_min, "similarity_delta_min", -1.0, 0.0)
            validate_float_range(self.similarity_delta_max, "similarity_delta_max", 0.0, 1.0)
        except ConfigValidationError as e:
            errors.append(str(e))
        
        # Validate clipping range
        if self.clip_similarity_delta and self.similarity_delta_min >= self.similarity_delta_max:
            errors.append("similarity_delta_min must be less than similarity_delta_max")

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
            reward_on_submit_only=cfg.get("reward_on_submit_only", True),
            step_penalty=cfg.get("step_penalty", -0.01),
            success_bonus=cfg.get("success_bonus", 10.0),
            similarity_weight=cfg.get("similarity_weight", 1.0),
            efficiency_bonus_threshold=cfg.get("efficiency_bonus_threshold", 50),
            efficiency_bonus=cfg.get("efficiency_bonus", 1.0),
            unsolved_submission_penalty=cfg.get("unsolved_submission_penalty", 0.0),
            clip_similarity_delta=cfg.get("clip_similarity_delta", False),
            similarity_delta_min=cfg.get("similarity_delta_min", -0.2),
            similarity_delta_max=cfg.get("similarity_delta_max", 0.2),
        )
