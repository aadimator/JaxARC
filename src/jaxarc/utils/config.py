"""Simple Hydra configuration utilities for jaxarc project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here


def get_config(overrides: list[str] | None = None) -> DictConfig:
    """Load the default Hydra configuration."""
    config_dir = here("conf")

    with initialize_config_dir(
        config_dir=str(config_dir.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=overrides or [])


def get_path(path_type: str, create: bool = False) -> Path:
    """Get a configured path by type.

    Args:
        path_type: Type of path ('raw', 'processed', 'interim', 'external')
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path object for the requested path type
    """
    cfg = get_config()
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


def create_conceptarc_config(**kwargs: Any) -> "ArcEnvConfig":
    """
    Create configuration factory function for ConceptARC dataset.

    This is a convenience function that delegates to the main factory function
    in the envs.factory module. ConceptARC is organized around 16 concept groups
    with systematic evaluation of abstraction and generalization abilities.

    Args:
        **kwargs: Configuration overrides passed to the factory function

    Returns:
        ArcEnvConfig optimized for ConceptARC dataset

    Raises:
        ValueError: If configuration validation fails
        ImportError: If required dependencies are not available

    Example:
        ```python
        from jaxarc.utils.config import create_conceptarc_config
        
        config = create_conceptarc_config(
            max_episode_steps=150,
            task_split="corpus",
            success_bonus=15.0
        )
        ```
    """
    try:
        from jaxarc.envs.factory import create_conceptarc_config as _create_conceptarc_config
        
        logger.debug("Creating ConceptARC configuration via factory function")
        return _create_conceptarc_config(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import ConceptARC factory function: {e}")
        raise ImportError(
            "ConceptARC factory function not available. "
            "Ensure jaxarc.envs.factory module is properly installed."
        ) from e
    except Exception as e:
        logger.error(f"Error creating ConceptARC configuration: {e}")
        raise ValueError(f"ConceptARC configuration creation failed: {e}") from e


def create_miniarc_config(**kwargs: Any) -> "ArcEnvConfig":
    """
    Create configuration factory function for MiniARC dataset.

    This is a convenience function that delegates to the main factory function
    in the envs.factory module. MiniARC is a 5x5 compact version of ARC
    designed for faster experimentation and prototyping.

    Args:
        **kwargs: Configuration overrides passed to the factory function

    Returns:
        ArcEnvConfig optimized for MiniARC dataset

    Raises:
        ValueError: If configuration validation fails
        ImportError: If required dependencies are not available

    Example:
        ```python
        from jaxarc.utils.config import create_miniarc_config
        
        config = create_miniarc_config(
            max_episode_steps=60,
            task_split="training",
            step_penalty=-0.002
        )
        ```
    """
    try:
        from jaxarc.envs.factory import create_miniarc_config as _create_miniarc_config
        
        logger.debug("Creating MiniARC configuration via factory function")
        return _create_miniarc_config(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import MiniARC factory function: {e}")
        raise ImportError(
            "MiniARC factory function not available. "
            "Ensure jaxarc.envs.factory module is properly installed."
        ) from e
    except Exception as e:
        logger.error(f"Error creating MiniARC configuration: {e}")
        raise ValueError(f"MiniARC configuration creation failed: {e}") from e


def validate_dataset_config(config: "ArcEnvConfig", dataset_name: str) -> None:
    """
    Validate configuration for specific dataset requirements.

    Args:
        config: ArcEnvConfig to validate
        dataset_name: Name of the dataset ("ConceptARC", "MiniARC", etc.)

    Raises:
        ValueError: If configuration is invalid for the specified dataset
    """
    try:
        from jaxarc.envs.config import validate_config
        
        # First run general validation
        validate_config(config)
        
        # Dataset-specific validation
        if dataset_name.lower() == "conceptarc":
            _validate_conceptarc_config(config)
        elif dataset_name.lower() == "miniarc":
            _validate_miniarc_config(config)
        else:
            logger.warning(f"No specific validation available for dataset: {dataset_name}")
            
        logger.debug(f"Configuration validated for {dataset_name}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed for {dataset_name}: {e}")
        raise ValueError(f"Invalid configuration for {dataset_name}: {e}") from e


def _validate_conceptarc_config(config: "ArcEnvConfig") -> None:
    """Validate ConceptARC-specific configuration requirements."""
    # ConceptARC uses standard ARC dimensions
    if config.grid.max_grid_height < 15 or config.grid.max_grid_width < 15:
        logger.warning(
            f"ConceptARC typically uses larger grids. Current max: "
            f"{config.grid.max_grid_height}x{config.grid.max_grid_width}"
        )
    
    # ConceptARC works well with mask-based actions for concept reasoning
    if config.action.selection_format != "mask":
        logger.info(
            f"ConceptARC typically works best with mask-based actions. "
            f"Current: {config.action.selection_format}"
        )
    
    # Check dataset name consistency
    if config.dataset.dataset_name != "ConceptARC":
        logger.warning(
            f"Dataset name mismatch: expected 'ConceptARC', got '{config.dataset.dataset_name}'"
        )


def _validate_miniarc_config(config: "ArcEnvConfig") -> None:
    """Validate MiniARC-specific configuration requirements."""
    # MiniARC should use 5x5 grid constraints
    if config.grid.max_grid_height > 5 or config.grid.max_grid_width > 5:
        logger.warning(
            f"MiniARC is optimized for 5x5 grids. Current max: "
            f"{config.grid.max_grid_height}x{config.grid.max_grid_width}. "
            f"Consider using max_grid_height=5 and max_grid_width=5."
        )
    
    # MiniARC works well with point-based actions for small grids
    if config.action.selection_format == "mask":
        logger.info(
            "MiniARC typically works well with point-based actions for 5x5 grids. "
            "Consider using selection_format='point' for optimal performance."
        )
    
    # Check dataset name consistency
    if config.dataset.dataset_name != "MiniARC":
        logger.warning(
            f"Dataset name mismatch: expected 'MiniARC', got '{config.dataset.dataset_name}'"
        )
