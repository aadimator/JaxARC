# NOTE: Moved from utils/config.py as part of Phase 1 consolidation.

"""Hydra configuration utilities for jaxarc project.

This module provides utilities for working with Hydra configurations,
including loading configs and managing data paths. It focuses purely
on Hydra integration without mixing in factory functions.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here


def get_config(overrides: list[str] | None = None) -> DictConfig:
    """Load the default Hydra configuration.

    Args:
        overrides: List of configuration overrides in Hydra format

    Returns:
        Loaded Hydra configuration

    Example:
        ```python
        from jaxarc.utils.core import get_config

        # Load default config
        cfg = get_config()

        # Load with overrides
        cfg = get_config(["dataset.dataset_name=ConceptARC", "action.selection_format=point"])
        ```
    """
    # Find the path to the 'conf' directory within the 'jaxarc' package
    config_dir = importlib.resources.files("jaxarc") / "conf"

    with initialize_config_dir(
        config_dir=str(config_dir.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=overrides or [])


def get_path(path_type: str, create: bool = False) -> Path:
    """Get a configured path by type.

    Args:
        path_type: Type of path ('data_raw', 'data_processed', 'data_interim', 'data_external')
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path object for the requested path type

    Raises:
        KeyError: If path_type is not found in configuration

    Example:
        ```python
        from jaxarc.utils.core import get_path

        # Get raw data path
        raw_path = get_path("data_raw", create=True)
        ```
    """
    cfg = get_config()

    if path_type not in cfg.paths:
        available_paths = list(cfg.paths.keys())
        msg = f"Path type '{path_type}' not found. Available: {available_paths}"
        raise KeyError(msg)

    path_str = cfg.paths[path_type]
    path: Path = here(path_str)

    if create:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created path: {path}")

    return path


def get_raw_path(create: bool = False) -> Path:
    """Get the raw data path.

    Args:
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to raw data directory
    """
    return get_path("data_raw", create=create)
