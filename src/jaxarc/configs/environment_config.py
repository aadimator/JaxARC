from __future__ import annotations

from typing import Literal

import equinox as eqx
from loguru import logger
from omegaconf import DictConfig

from .validation import (
    ConfigValidationError,
    validate_positive_int,
    validate_string_choice,
)


class EnvironmentConfig(eqx.Module):
    """Core environment behavior and runtime settings.

    This config only contains settings that directly affect environment behavior,
    not dataset constraints, logging, visualization, or storage settings.
    """

    # Episode settings
    max_episode_steps: int = 100
    auto_reset: bool = True

    # Debug level (simplified: off|minimal|verbose)
    debug_level: Literal["off", "minimal", "verbose"] = "minimal"

    # Render mode (rgb_array|ansi|svg)
    render_mode: Literal["rgb_array", "ansi", "svg"] = "rgb_array"

    def validate(self) -> tuple[str, ...]:
        """Validate environment configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            # Validate episode settings
            validate_positive_int(self.max_episode_steps, "max_episode_steps")
            if self.max_episode_steps > 10000:
                logger.warning(
                    f"max_episode_steps is very large: {self.max_episode_steps}"
                )

            # Validate debug level
            valid_levels = ("off", "minimal", "verbose")
            validate_string_choice(self.debug_level, "debug_level", valid_levels)

            # Validate render mode
            valid_render_modes = ("rgb_array", "ansi", "svg")
            validate_string_choice(self.render_mode, "render_mode", valid_render_modes)

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"EnvironmentConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> EnvironmentConfig:
        """Create environment config from Hydra DictConfig."""
        return cls(
            max_episode_steps=cfg.get("max_episode_steps", 100),
            auto_reset=cfg.get("auto_reset", True),
            debug_level=cfg.get("debug_level", "minimal"),
            render_mode=cfg.get("render_mode", "rgb_array"),
        )
