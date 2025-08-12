"""Dataset-specific configuration validation utilities.

This module provides validation functions for different ARC datasets,
ensuring configurations are appropriate for specific dataset requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from jaxarc.configs import JaxArcConfig


def validate_dataset_config(config: JaxArcConfig, dataset_name: str) -> None:
    """Validate configuration for specific dataset requirements.

    Args:
        config: JaxArcConfig to validate
        dataset_name: Name of the dataset ("ConceptARC", "MiniARC", etc.)

    Raises:
        ValueError: If configuration is invalid for the specified dataset

    Example:
        ```python
        from jaxarc.utils.dataset_validation import validate_dataset_config
        from jaxarc.envs import create_conceptarc_config

        config = create_conceptarc_config()
        validate_dataset_config(config, "ConceptARC")
        ```
    """
    try:
        # First run general validation using the config's validate method
        validation_errors = config.validate()
        if validation_errors:
            raise ValueError(f"Config validation errors: {validation_errors}")

        # Dataset-specific validation
        if dataset_name.lower() == "conceptarc":
            _validate_conceptarc_config(config)
        elif dataset_name.lower() == "miniarc":
            _validate_miniarc_config(config)
        else:
            logger.warning(
                f"No specific validation available for dataset: {dataset_name}"
            )

        logger.debug(f"Configuration validated for {dataset_name}")

    except Exception as e:
        logger.error(f"Configuration validation failed for {dataset_name}: {e}")
        raise ValueError(f"Invalid configuration for {dataset_name}: {e}") from e


def _validate_conceptarc_config(config: JaxArcConfig) -> None:
    """Validate ConceptARC-specific configuration requirements."""
    # ConceptARC uses standard ARC dimensions
    if config.dataset.max_grid_height < 15 or config.dataset.max_grid_width < 15:
        logger.warning(
            f"ConceptARC typically uses larger grids. Current max: "
            f"{config.dataset.max_grid_height}x{config.dataset.max_grid_width}"
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


def _validate_miniarc_config(config: JaxArcConfig) -> None:
    """Validate MiniARC-specific configuration requirements."""
    # MiniARC should use 5x5 grid constraints
    if config.dataset.max_grid_height > 5 or config.dataset.max_grid_width > 5:
        logger.warning(
            f"MiniARC is optimized for 5x5 grids. Current max: "
            f"{config.dataset.max_grid_height}x{config.dataset.max_grid_width}. "
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


def get_dataset_recommendations(dataset_name: str) -> dict[str, str]:
    """Get recommended configuration settings for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary of recommended configuration overrides

    Example:
        ```python
        from jaxarc.utils.dataset_validation import get_dataset_recommendations

        recommendations = get_dataset_recommendations("MiniARC")
        print(recommendations)
        # {'grid.max_grid_height': '5', 'grid.max_grid_width': '5', 'action.selection_format': 'point'}
        ```
    """
    recommendations = {}

    if dataset_name.lower() == "conceptarc":
        recommendations.update(
            {
                "dataset.max_grid_height": "30",
                "dataset.max_grid_width": "30",
                "action.selection_format": "mask",
                "dataset.dataset_name": "ConceptARC",
            }
        )
    elif dataset_name.lower() == "miniarc":
        recommendations.update(
            {
                "dataset.max_grid_height": "5",
                "dataset.max_grid_width": "5",
                "action.selection_format": "point",
                "dataset.dataset_name": "MiniARC",
            }
        )
    else:
        logger.warning(f"No recommendations available for dataset: {dataset_name}")

    return recommendations
