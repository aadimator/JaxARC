"""Simple Hydra configuration utilities for jaxarc project."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pyprojroot import here


def get_config() -> DictConfig:
    """Load the default Hydra configuration."""
    config_dir = here("conf")

    with initialize_config_dir(
        config_dir=str(config_dir.absolute()), version_base=None
    ):
        return compose(config_name="config")


def get_path(path_type: str, create: bool = False) -> Path:
    """Get a configured path by type.

    Args:
        path_type: Type of path ('raw', 'processed', 'interim', 'external')
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path object for the requested path type
    """
    cfg = get_config()
    print(cfg)
    path_str = cfg.paths[path_type]
    path: Path = here(path_str)

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_raw_path(create: bool = False) -> Path:
    """Get the raw data path."""
    return get_path("data_raw", create=create)


def get_processed_path(create: bool = False) -> Path:
    """Get the processed data path."""
    return get_path("data_processed", create=create)


def get_interim_path(create: bool = False) -> Path:
    """Get the interim data path."""
    return get_path("data_interim", create=create)


def get_external_path(create: bool = False) -> Path:
    """Get the external data path."""
    return get_path("data_external", create=create)
