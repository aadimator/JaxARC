from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import ConfigValidationError, validate_float_range


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
        # Handle permutation_types conversion
        permutation_types = kwargs.get(
            "permutation_types", ("rotate", "reflect", "color_remap")
        )
        if isinstance(permutation_types, str):
            permutation_types = (permutation_types,)
        elif hasattr(permutation_types, "__iter__") and not isinstance(
            permutation_types, (str, tuple)
        ):
            permutation_types = (
                tuple(permutation_types) if permutation_types else ("rotate",)
            )
        elif not isinstance(permutation_types, tuple):
            permutation_types = ("rotate", "reflect", "color_remap")

        self.demo_weight = kwargs.get("demo_weight", 0.4)
        self.permutation_weight = kwargs.get("permutation_weight", 0.3)
        self.empty_weight = kwargs.get("empty_weight", 0.2)
        self.random_weight = kwargs.get("random_weight", 0.1)
        self.permutation_types = permutation_types
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
            valid_permutation_types = {"rotate", "reflect", "color_remap"}
            if hasattr(self.permutation_types, "__iter__"):
                for ptype in self.permutation_types:
                    if ptype not in valid_permutation_types:
                        errors.append(
                            f"Invalid permutation type: {ptype}. Valid types: {valid_permutation_types}"
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
        """Validate hashability for JAX compatibility."""
        try:
            hash(self)
        except TypeError as e:
            msg = (
                f"GridInitializationConfig must be hashable for JAX compatibility: {e}"
            )
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> GridInitializationConfig:
        """Create grid initialization config from Hydra DictConfig."""
        permutation_types = cfg.get(
            "permutation_types", ["rotate", "reflect", "color_remap"]
        )
        if isinstance(permutation_types, str):
            permutation_types = (permutation_types,)
        elif hasattr(permutation_types, "__iter__") and not isinstance(
            permutation_types, (str, tuple)
        ):
            permutation_types = (
                tuple(permutation_types) if permutation_types else ("rotate",)
            )
        elif not isinstance(permutation_types, tuple):
            permutation_types = ("rotate", "reflect", "color_remap")

        return cls(
            demo_weight=cfg.get("demo_weight", 0.4),
            permutation_weight=cfg.get("permutation_weight", 0.3),
            empty_weight=cfg.get("empty_weight", 0.2),
            random_weight=cfg.get("random_weight", 0.1),
            permutation_types=permutation_types,
            random_density=cfg.get("random_density", 0.3),
            random_pattern_type=cfg.get("random_pattern_type", "sparse"),
        )
