from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    check_hashable,
    ensure_tuple,
    validate_float_range,
    validate_tuple_elements,
)


class GridInitializationConfig(eqx.Module):
    """Configuration for grid initialization strategies.

    This config controls how working grids are initialized in the environment.
    Supports four modes for research flexibility:
    - Demo mode: Copy from training examples
    - Permutation mode: Apply transformations to demo grids
    - Empty mode: Start with blank grids
    - Random mode: Generate random patterns
    """

    # Mode weights (normalized automatically, don't need to sum to 1.0)
    demo_weight: float = 0.4
    permutation_weight: float = 0.3
    empty_weight: float = 0.2
    random_weight: float = 0.1

    # Permutation configuration (simplified)
    permutation_types: tuple[str, ...] = ("rotate", "reflect", "color_remap")

    # Random pattern configuration (simplified)
    random_density: float = 0.3
    random_pattern_type: str = "sparse"  # "sparse" or "dense"

    def __init__(self, **kwargs):
        self.demo_weight = kwargs.get("demo_weight", 0.4)
        self.permutation_weight = kwargs.get("permutation_weight", 0.3)
        self.empty_weight = kwargs.get("empty_weight", 0.2)
        self.random_weight = kwargs.get("random_weight", 0.1)
        self.permutation_types = ensure_tuple(
            kwargs.get("permutation_types", ("rotate", "reflect", "color_remap")),
            default=("rotate", "reflect", "color_remap"),
        )
        self.random_density = kwargs.get("random_density", 0.3)
        self.random_pattern_type = kwargs.get("random_pattern_type", "sparse")

    def validate(self) -> tuple[str, ...]:
        """Validate grid initialization configuration."""
        errors: list[str] = []

        try:
            # Validate weights (they will be normalized, so just need to be non-negative)
            validate_float_range(self.demo_weight, "demo_weight", 0.0, float("inf"))
            validate_float_range(
                self.permutation_weight, "permutation_weight", 0.0, float("inf")
            )
            validate_float_range(self.empty_weight, "empty_weight", 0.0, float("inf"))
            validate_float_range(self.random_weight, "random_weight", 0.0, float("inf"))

            # At least one weight must be positive
            total_weight = (
                self.demo_weight
                + self.permutation_weight
                + self.empty_weight
                + self.random_weight
            )
            if total_weight <= 0:
                errors.append("At least one initialization weight must be positive")

            # Validate random configuration
            validate_float_range(self.random_density, "random_density", 0.0, 1.0)

            if self.random_pattern_type not in ("sparse", "dense"):
                errors.append(
                    f"Invalid random_pattern_type: {self.random_pattern_type}. Must be 'sparse' or 'dense'"
                )

            # Validate permutation types
            _valid_perm_types = {"rotate", "reflect", "color_remap"}
            if hasattr(self.permutation_types, "__iter__"):
                errors.extend(
                    validate_tuple_elements(
                        self.permutation_types,
                        "permutation_types",
                        element_type=str,
                        allowed=_valid_perm_types,
                    )
                )

            # If permutation weight is positive, require non-empty permutation_types
            if self.permutation_weight > 0.0 and not self.permutation_types:
                errors.append(
                    "permutation_types cannot be empty when permutation_weight > 0"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        check_hashable(self, "GridInitializationConfig")

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> GridInitializationConfig:
        """Create grid initialization config from Hydra DictConfig."""
        return cls(
            demo_weight=cfg.get("demo_weight", 0.4),
            permutation_weight=cfg.get("permutation_weight", 0.3),
            empty_weight=cfg.get("empty_weight", 0.2),
            random_weight=cfg.get("random_weight", 0.1),
            permutation_types=cfg.get(
                "permutation_types", ["rotate", "reflect", "color_remap"]
            ),
            random_density=cfg.get("random_density", 0.3),
            random_pattern_type=cfg.get("random_pattern_type", "sparse"),
        )
