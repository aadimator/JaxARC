"""Configuration preparation functions for JaxARC registration.

Standalone functions extracted from EnvRegistry static methods.
These handle config creation, dataset loading, episode mode resolution,
and dataset availability checks.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from jaxarc.utils import DatasetError


def prepare_config(
    config: Optional[Any], max_episode_steps: int, dataset_key: str
) -> Any:
    """Prepare a JaxArcConfig, applying overrides when possible.

    - If no config provided, instantiate a default JaxArcConfig.
    - Ensure environment.max_episode_steps matches the spec if possible.
    - Normalize dataset configuration based on spec.dataset_key.
    """
    # Import locally to avoid hard dependency at module import time
    try:
        from jaxarc.configs.environment_config import EnvironmentConfig
        from jaxarc.configs.main_config import JaxArcConfig
        from jaxarc.utils.core import get_config
    except Exception as e:
        raise ValueError(
            "Could not import configuration types. Ensure configurations "
            "are available or provide a ready 'config' object."
        ) from e
    # If config not provided, prefer a safe construction path that avoids Hydra re-init
    if config is None:
        try:
            # Detect if a Hydra app is already initialized in this process
            from hydra.core.global_hydra import GlobalHydra  # type: ignore

            gh = GlobalHydra.instance()
            hydra_active = gh.is_initialized()
        except Exception:
            hydra_active = False

        if hydra_active:
            # Avoid re-initializing Hydra: build a default config directly
            cfg = JaxArcConfig()
        else:
            # Standalone usage: use Hydra defaults
            cfg = JaxArcConfig.from_hydra(get_config())
    else:
        cfg = config

    # Enforce max_episode_steps
    try:
        cfg.environment = EnvironmentConfig(max_episode_steps=max_episode_steps)
    except Exception:
        pass

    # Best-effort dataset normalization is handled later by ensure_dataset_available

    return cfg


def resolve_episode_mode(episode_mode: int | None, selector: str | None) -> int:
    """Resolve episode mode using explicit value or selector token (train/eval)."""
    if episode_mode is not None:
        return int(episode_mode)
    if not selector:
        return 0
    sel = selector.lower()
    if sel in ("train", "training"):
        return 0
    if sel in ("eval", "evaluation", "test", "corpus"):
        return 1
    return 0


def load_dataset_config(dataset_key: str) -> Any:
    """Load dataset config from packaged YAML without initializing Hydra.

    This avoids conflicts when JaxARC is embedded inside an existing Hydra app.
    Returns a DatasetConfig instance for the requested dataset key.
    """
    try:
        # Map normalized key -> dataset YAML file name inside jaxarc/conf/dataset
        key_lower = dataset_key.lower()
        file_name: Optional[str] = None
        if key_lower in ("mini", "miniarc", "mini-arc"):
            file_name = "mini_arc.yaml"
        elif key_lower in ("concept", "conceptarc", "concept-arc"):
            file_name = "concept_arc.yaml"
        elif key_lower in ("agi1", "arc-agi-1", "agi-1", "agi_1"):
            file_name = "arc_agi_1.yaml"
        elif key_lower in ("agi2", "arc-agi-2", "agi-2", "agi_2"):
            file_name = "arc_agi_2.yaml"
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        # Load YAML via importlib.resources to avoid file path issues
        import importlib.resources as pkg_resources
        import io

        import yaml
        from omegaconf import DictConfig, OmegaConf

        from jaxarc.configs.dataset_config import DatasetConfig

        dataset_dir = pkg_resources.files("jaxarc") / "conf" / "dataset"
        yaml_path = dataset_dir / file_name
        # Read text and convert to DictConfig
        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_text = f.read()
        data = yaml.safe_load(io.StringIO(yaml_text)) or {}
        cfg: DictConfig = OmegaConf.create(data)
        return DatasetConfig.from_hydra(cfg)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset config for '{dataset_key}': {e}"
        ) from e


def ensure_dataset_available(config: Any, dataset_key: str, auto_download: bool) -> Any:
    """Ensure dataset exists and matches the requested dataset key.

    - If config.dataset exists but doesn't match the requested key, replace it.
    - Load dataset config directly from packaged YAML (no Hydra init).
    - Ensure files are present via DatasetManager and fix dataset_path.
    """
    import equinox as eqx

    from jaxarc.configs.dataset_config import DatasetConfig
    from jaxarc.utils import DatasetManager

    manager = DatasetManager()

    # Load the desired dataset configuration from YAML
    desired_ds: DatasetConfig = load_dataset_config(dataset_key)

    # Preserve task_split from current config if it was modified (e.g., by maybe_adjust_task_split)
    # This must happen BEFORE replacing the dataset config
    current_ds = getattr(config, "dataset", None)
    if isinstance(current_ds, DatasetConfig):
        same = (
            str(current_ds.dataset_name).strip().lower()
            == str(desired_ds.dataset_name).strip().lower()
        )

        # Preserve task_split if it differs from default (was modified by maybe_adjust_task_split)
        if hasattr(current_ds, "task_split") and hasattr(desired_ds, "task_split"):
            if current_ds.task_split != desired_ds.task_split:
                logger.debug(
                    f"Preserving task_split='{current_ds.task_split}' (was modified by selector)"
                )
                desired_ds = eqx.tree_at(
                    lambda d: d.task_split, desired_ds, current_ds.task_split
                )

        if not same:
            logger.debug(
                f"Overriding provided DatasetConfig '{current_ds.dataset_name}' with '{desired_ds.dataset_name}' from key '{dataset_key}'."
            )

        config = eqx.tree_at(lambda c: c.dataset, config, desired_ds)
    else:
        # No valid dataset found in config, set to desired
        config = eqx.tree_at(lambda c: c.dataset, config, desired_ds)

    # Now ensure the dataset files are on disk and update dataset_path
    try:
        dataset_path = manager.ensure_dataset_available(
            config, auto_download=auto_download
        )
        ds = config.dataset
        ds = eqx.tree_at(lambda d: d.dataset_path, ds, str(dataset_path))
        config = eqx.tree_at(lambda c: c.dataset, config, ds)
        return config
    except DatasetError as e:
        logger.error(f"Dataset management failed: {e}")
        raise ValueError(f"Dataset not available: {e}") from e


def maybe_adjust_task_split(
    config: Any, dataset_key: str, selector: Optional[str]
) -> Any:
    """Adjust config.dataset.task_split based on selector for AGI datasets.

    Returns the modified config (necessary because equinox objects are immutable).
    """
    try:
        import equinox as eqx

        sel = (selector or "").lower()
        if dataset_key.lower() in (
            "agi1",
            "arc-agi-1",
            "agi-1",
            "agi_1",
            "agi2",
            "arc-agi-2",
            "agi-2",
            "agi_2",
        ):
            new_split = None
            if sel in ("train", "training"):
                logger.debug(f"Adjusting task_split to 'train' for selector '{sel}'")
                new_split = "train"
            elif sel in ("eval", "evaluation", "test", "corpus"):
                logger.debug(
                    f"Adjusting task_split to 'evaluation' for selector '{sel}'"
                )
                new_split = "evaluation"

            if new_split is not None:
                # Use eqx.tree_at to properly modify immutable config
                ds = eqx.tree_at(lambda d: d.task_split, config.dataset, new_split)
                config = eqx.tree_at(lambda c: c.dataset, config, ds)

        return config
    except Exception:
        # Best-effort only
        return config
