from __future__ import annotations

import equinox as eqx
from loguru import logger
from omegaconf import DictConfig

from .validation import ConfigValidationError, validate_string_choice


class VisualizationConfig(eqx.Module):
    """All visualization and rendering settings.

    This config contains everything related to visual output, rendering,
    and visualization behavior. No logging or storage settings here.
    """

    # Core settings
    enabled: bool = True
    level: str = "standard"

    def __init__(self, **kwargs):
        # Set all fields
        self.enabled = kwargs.get("enabled", True)
        self.level = kwargs.get("level", "standard")
        self.episode_summaries = kwargs.get("episode_summaries", True)
        self.step_visualizations = kwargs.get("step_visualizations", True)

    # Episode visualization
    episode_summaries: bool = True
    step_visualizations: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate visualization configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            # Validate level choices
            valid_levels = ("off", "minimal", "standard", "verbose", "full")
            validate_string_choice(self.level, "level", valid_levels)

            # Cross-field validation warnings
            if not self.enabled and self.level != "off":
                logger.warning("Visualization disabled but level is not 'off'")

            if self.level == "off" and self.enabled:
                logger.warning("Visualization level is 'off' but enabled=True")

        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = (
                f"VisualizationConfig must be hashable for JAX compatibility: {e}"
            )
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> VisualizationConfig:
        """Create visualization config from Hydra DictConfig."""
        return cls(
            enabled=cfg.get("enabled", True),
            level=cfg.get("level", "standard"),
            episode_summaries=cfg.get("episode_summaries", True),
            step_visualizations=cfg.get("step_visualizations", True),
        )
