from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig


class VisualizationConfig(eqx.Module):
    """All visualization and rendering settings.

    This config contains everything related to visual output, rendering,
    and visualization behavior. No logging or storage settings here.
    """

    # Core settings
    enabled: bool = True

    def __init__(self, **kwargs):
        # Set all fields
        self.enabled = kwargs.get("enabled", True)
        self.episode_summaries = kwargs.get("episode_summaries", True)
        self.step_visualizations = kwargs.get("step_visualizations", True)

    # Episode visualization
    episode_summaries: bool = True
    step_visualizations: bool = True

    def validate(self) -> tuple[str, ...]:
        """Validate visualization configuration and return tuple of errors."""
        errors: list[str] = []

        # Minimal validation
        if not isinstance(self.enabled, bool):
            errors.append("enabled must be a boolean for VisualizationConfig")
        if not isinstance(self.episode_summaries, bool):
            errors.append("episode_summaries must be a boolean")
        if not isinstance(self.step_visualizations, bool):
            errors.append("step_visualizations must be a boolean")

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"VisualizationConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> VisualizationConfig:
        """Create visualization config from Hydra DictConfig."""
        return cls(
            enabled=cfg.get("enabled", True),
            episode_summaries=cfg.get("episode_summaries", True),
            step_visualizations=cfg.get("step_visualizations", True),
        )
