from __future__ import annotations

import equinox as eqx
from loguru import logger
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_float_range,
    validate_string_choice,
)


class GridInitializationConfig(eqx.Module):
    """Configuration for diverse grid initialization strategies.

    This config controls how working grids are initialized in the environment,
    supporting multiple modes including demo grids, permutations, empty grids,
    and random patterns for enhanced training diversity.
    """

    # Initialization mode selection
    mode: str = "demo"

    # Probability weights for mixed mode (must sum to 1.0)
    demo_weight: float = 0.25
    permutation_weight: float = 0.25
    empty_weight: float = 0.25
    random_weight: float = 0.25

    # Permutation configuration
    permutation_types: tuple[str, ...] = ("rotate", "reflect", "color_remap")

    # Random initialization configuration
    random_density: float = 0.3
    random_pattern_type: str = "sparse"

    # Fallback and error handling
    enable_fallback: bool = True

    def __init__(self, **kwargs):
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

        self.mode = kwargs.get("mode", "demo")
        self.demo_weight = kwargs.get("demo_weight", 0.25)
        self.permutation_weight = kwargs.get("permutation_weight", 0.25)
        self.empty_weight = kwargs.get("empty_weight", 0.25)
        self.random_weight = kwargs.get("random_weight", 0.25)
        self.permutation_types = permutation_types
        self.random_density = kwargs.get("random_density", 0.3)
        self.random_pattern_type = kwargs.get("random_pattern_type", "sparse")
        self.enable_fallback = kwargs.get("enable_fallback", True)

    def validate(self) -> tuple[str, ...]:
        """Validate grid initialization configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            valid_modes = ("demo", "permutation", "empty", "random", "mixed")
            validate_string_choice(self.mode, "mode", valid_modes)

            valid_pattern_types = ("sparse", "dense", "structured", "noise")
            validate_string_choice(
                self.random_pattern_type, "random_pattern_type", valid_pattern_types
            )

            validate_float_range(self.demo_weight, "demo_weight", 0.0, 1.0)
            validate_float_range(
                self.permutation_weight, "permutation_weight", 0.0, 1.0
            )
            validate_float_range(self.empty_weight, "empty_weight", 0.0, 1.0)
            validate_float_range(self.random_weight, "random_weight", 0.0, 1.0)
            validate_float_range(self.random_density, "random_density", 0.0, 1.0)

            total_weight = (
                self.demo_weight
                + self.permutation_weight
                + self.empty_weight
                + self.random_weight
            )
            if abs(total_weight - 1.0) > 1e-6:
                errors.append(
                    f"Initialization weights must sum to 1.0, got {total_weight:.6f} "
                    f"(demo: {self.demo_weight}, permutation: {self.permutation_weight}, "
                    f"empty: {self.empty_weight}, random: {self.random_weight})"
                )

            valid_permutation_types = ("rotate", "reflect", "color_remap", "translate")
            if hasattr(self.permutation_types, "__iter__"):
                for ptype in self.permutation_types:
                    if ptype not in valid_permutation_types:
                        errors.append(
                            f"Invalid permutation type: {ptype}. "
                            f"Valid types: {valid_permutation_types}"
                        )

            if self.mode != "mixed" and any(
                w != 0.25
                for w in [
                    self.demo_weight,
                    self.permutation_weight,
                    self.empty_weight,
                    self.random_weight,
                ]
            ):
                logger.warning(
                    f"Mode is '{self.mode}' but weights are specified. "
                    "Weights are only used in 'mixed' mode."
                )

            if self.mode == "permutation" and not self.permutation_types:
                errors.append(
                    "permutation_types cannot be empty when mode is 'permutation'"
                )

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"GridInitializationConfig must be hashable for JAX compatibility: {e}"
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
            mode=cfg.get("mode", "demo"),
            demo_weight=cfg.get("demo_weight", 0.25),
            permutation_weight=cfg.get("permutation_weight", 0.25),
            empty_weight=cfg.get("empty_weight", 0.25),
            random_weight=cfg.get("random_weight", 0.25),
            permutation_types=permutation_types,
            random_density=cfg.get("random_density", 0.3),
            random_pattern_type=cfg.get("random_pattern_type", "sparse"),
            enable_fallback=cfg.get("enable_fallback", True),
        )
