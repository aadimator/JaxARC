from __future__ import annotations

import equinox as eqx
from omegaconf import DictConfig

from .validation import ConfigValidationError


class WandbConfig(eqx.Module):
    """Weights & Biases integration settings.

    This config contains everything related to W&B logging and tracking.
    No local logging or storage settings here.
    """

    # Core wandb settings
    enabled: bool = False
    project_name: str = "jaxarc-experiments"
    entity: str | None = None
    tags: tuple[str, ...] = ("jaxarc",)
    notes: str = "JaxARC experiment"
    group: str | None = None
    job_type: str = "training"

    # Error handling
    offline_mode: bool = False

    # Storage
    save_code: bool = True

    def __init__(self, **kwargs):
        """Initialize with automatic list-to-tuple conversion."""
        tags = kwargs.get("tags", ("jaxarc",))
        if isinstance(tags, str):
            tags = (tags,)
        elif hasattr(tags, "__iter__") and not isinstance(tags, (str, tuple)):
            tags = tuple(tags)
        elif not isinstance(tags, tuple):
            tags = ("jaxarc",)

        self.enabled = kwargs.get("enabled", False)
        self.project_name = kwargs.get("project_name", "jaxarc-experiments")
        self.entity = kwargs.get("entity")
        self.tags = tags
        self.notes = kwargs.get("notes", "JaxARC experiment")
        self.group = kwargs.get("group")
        self.job_type = kwargs.get("job_type", "training")
        self.offline_mode = kwargs.get("offline_mode", False)
        self.save_code = kwargs.get("save_code", True)

    def validate(self) -> tuple[str, ...]:
        """Validate wandb configuration and return tuple of errors."""
        errors: list[str] = []

        try:
            if not self.project_name.strip():
                errors.append("project_name cannot be empty")
        except ConfigValidationError as e:
            errors.append(str(e))

        return tuple(errors)

    def __check_init__(self):
        """Validate hashability after initialization."""
        try:
            hash(self)
        except TypeError as e:
            msg = f"WandbConfig must be hashable for JAX compatibility: {e}"
            raise ValueError(msg) from e

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> WandbConfig:
        """Create wandb config from Hydra DictConfig."""
        tags = cfg.get("tags", ["jaxarc"])
        if isinstance(tags, str):
            tags = (tags,)
        elif hasattr(tags, "__iter__") and not isinstance(tags, (str, tuple)):
            tags = tuple(tags)
        elif not isinstance(tags, tuple):
            tags = ("jaxarc",)

        return cls(
            tags=tags,
            enabled=cfg.get("enabled", False),
            project_name=cfg.get("project_name", "jaxarc-experiments"),
            entity=cfg.get("entity"),
            notes=cfg.get("notes", "JaxARC experiment"),
            group=cfg.get("group"),
            job_type=cfg.get("job_type", "training"),
            offline_mode=cfg.get("offline_mode", False),
            save_code=cfg.get("save_code", True),
        )
