"""
Unified dataset manager for JaxARC datasets.

Provides a simple, configuration-driven interface for downloading and validating
datasets. Replaces the previous separate downloader and validation utilities
with a single, KISS-compliant solution.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from pyprojroot import here

if TYPE_CHECKING:
    from jaxarc.configs import DatasetConfig, JaxArcConfig


class DatasetError(Exception):
    """Exception raised when dataset operations fail."""


class DatasetManager:
    """Simple, unified dataset manager supporting all JaxARC datasets."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the dataset manager.

        Args:
            output_dir: Base directory for downloads. If None, uses project root/data.
        """
        self.output_dir = output_dir or (here() / "data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dataset_available(
        self, config: JaxArcConfig | DatasetConfig, auto_download: bool = False
    ) -> Path:
        """
        Ensure dataset exists at the configured path, optionally downloading if missing.

        Args:
            config: Dataset configuration (JaxArcConfig or DatasetConfig)
            auto_download: Whether to download if dataset is missing

        Returns:
            Path to the dataset directory

        Raises:
            DatasetError: If dataset is not available and auto_download is False,
                         or if download/validation fails

        Example:
            ```python
            from jaxarc.utils.dataset_manager import DatasetManager
            from jaxarc.configs import JaxArcConfig

            config = JaxArcConfig()
            manager = DatasetManager()
            dataset_path = manager.ensure_dataset_available(config, auto_download=True)
            ```
        """
        # Extract dataset config if we got a JaxArcConfig
        dataset_config = getattr(config, "dataset", config)

        # Get configured dataset path
        dataset_path_str = getattr(dataset_config, "dataset_path", "")
        if not dataset_path_str:
            raise DatasetError("Dataset path not configured")

        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_absolute():
            dataset_path = here() / dataset_path

        # Check if dataset already exists and is valid
        if self.validate_dataset(dataset_config, dataset_path):
            logger.debug(
                f"Dataset '{dataset_config.dataset_name}' found at {dataset_path}"
            )
            return dataset_path

        # Dataset missing or invalid
        if not auto_download:
            raise DatasetError(
                f"Dataset '{dataset_config.dataset_name}' not found at {dataset_path}. "
                "Set auto_download=True to download automatically."
            )

        # Download the dataset
        return self.download_dataset(dataset_config, dataset_path)

    def download_dataset(
        self, dataset_config: DatasetConfig, target_path: Optional[Path] = None
    ) -> Path:
        """
        Download dataset from its configured repository.

        Args:
            dataset_config: Dataset configuration with repo URL and metadata
            target_path: Target directory. If None, uses config.dataset_path

        Returns:
            Path to downloaded dataset directory

        Raises:
            DatasetError: If download fails
        """
        repo_url = getattr(dataset_config, "dataset_repo", "")
        if not repo_url:
            raise DatasetError(
                f"No repository URL configured for {dataset_config.dataset_name}"
            )

        if target_path is None:
            dataset_path_str = getattr(dataset_config, "dataset_path", "")
            if not dataset_path_str:
                raise DatasetError(
                    "No target path provided and dataset_path not configured"
                )
            target_path = Path(dataset_path_str)
            if not target_path.is_absolute():
                target_path = here() / target_path

        logger.info(
            f"Downloading {dataset_config.dataset_name} from {repo_url} to {target_path}"
        )

        # Remove existing directory if it exists
        if target_path.exists():
            logger.info(f"Removing existing directory: {target_path}")
            shutil.rmtree(target_path)

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone the repository
        try:
            cmd = ["git", "clone", repo_url, str(target_path)]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            logger.success(f"Successfully downloaded {dataset_config.dataset_name}")
            if result.stdout:
                logger.debug(f"Git output: {result.stdout}")

        except FileNotFoundError as e:
            raise DatasetError(
                "Git is not installed or not available in PATH. "
                "Please install git to download datasets."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise DatasetError(
                "Download timed out after 5 minutes. Check your network connection."
            ) from e
        except subprocess.CalledProcessError as e:
            error_msg = f"Git clone failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            raise DatasetError(error_msg) from e

        # Validate the download
        if not self.validate_dataset(dataset_config, target_path):
            raise DatasetError(f"Downloaded dataset failed validation: {target_path}")

        return target_path

    def validate_dataset(
        self, dataset_config: DatasetConfig, dataset_path: Path
    ) -> bool:
        """
        Validate that dataset directory exists and has expected structure.

        Args:
            dataset_config: Dataset configuration with validation metadata
            dataset_path: Path to dataset directory

        Returns:
            True if dataset is valid, False otherwise
        """
        if not dataset_path.exists() or not dataset_path.is_dir():
            logger.debug(
                f"Dataset path does not exist or is not a directory: {dataset_path}"
            )
            return False

        # Check if directory is not empty
        try:
            contents = list(dataset_path.iterdir())
            if not contents:
                logger.debug(f"Dataset directory is empty: {dataset_path}")
                return False
        except OSError:
            logger.debug(f"Cannot access dataset directory: {dataset_path}")
            return False

        # Check for expected subdirectories
        expected_subdirs = getattr(dataset_config, "expected_subdirs", [])
        for subdir_name in expected_subdirs:
            subdir_path = dataset_path / subdir_name
            if not subdir_path.exists() or not subdir_path.is_dir():
                logger.debug(f"Expected subdirectory not found: {subdir_path}")
                return False

        logger.debug(f"Dataset validation passed: {dataset_path}")
        return True

    def validate_config(self, config: JaxArcConfig, dataset_name: str) -> None:
        """
        Validate configuration for specific dataset requirements.

        Args:
            config: JaxArcConfig to validate
            dataset_name: Name of the dataset for context

        Raises:
            ValueError: If configuration is invalid for the specified dataset

        Example:
            ```python
            from jaxarc.utils.dataset_manager import DatasetManager
            from jaxarc.configs import JaxArcConfig

            config = JaxArcConfig()
            manager = DatasetManager()
            manager.validate_config(config, "MiniARC")
            ```
        """
        try:
            # Run general config validation first
            validation_errors = config.validate()
            if validation_errors:
                raise ValueError(f"Config validation errors: {validation_errors}")

            # Dataset-specific validation (simplified)
            dataset_config = config.dataset
            dataset_name_lower = dataset_name.lower()

            if dataset_name_lower in ("miniarc", "mini-arc", "mini"):
                if (
                    dataset_config.max_grid_height > 5
                    or dataset_config.max_grid_width > 5
                ):
                    logger.warning(
                        f"MiniARC is optimized for 5x5 grids. Current max: "
                        f"{dataset_config.max_grid_height}x{dataset_config.max_grid_width}. "
                        f"Consider using smaller grid sizes."
                    )

            elif dataset_name_lower in ("conceptarc", "concept-arc", "concept"):
                if (
                    dataset_config.max_grid_height < 15
                    or dataset_config.max_grid_width < 15
                ):
                    logger.warning(
                        f"ConceptARC typically uses larger grids. Current max: "
                        f"{dataset_config.max_grid_height}x{dataset_config.max_grid_width}"
                    )

            # Check dataset name consistency
            if dataset_config.dataset_name != dataset_name:
                logger.warning(
                    f"Dataset name mismatch: expected '{dataset_name}', "
                    f"got '{dataset_config.dataset_name}'"
                )

            logger.debug(f"Configuration validated for {dataset_name}")

        except Exception as e:
            logger.error(f"Configuration validation failed for {dataset_name}: {e}")
            raise ValueError(f"Invalid configuration for {dataset_name}: {e}") from e

    @staticmethod
    def get_dataset_recommendations(dataset_name: str) -> dict[str, Any]:
        """
        Get recommended configuration settings for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary of recommended configuration overrides

        Example:
            ```python
            from jaxarc.utils.dataset_manager import DatasetManager

            recommendations = DatasetManager.get_dataset_recommendations("MiniARC")
            print(recommendations)
            # {'dataset.max_grid_height': 5, 'dataset.max_grid_width': 5, ...}
            ```
        """
        recommendations: dict[str, Any] = {}

        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower in ("conceptarc", "concept-arc", "concept"):
            recommendations.update(
                {
                    "dataset.max_grid_height": 30,
                    "dataset.max_grid_width": 30,
                    "dataset.dataset_name": "ConceptARC",
                }
            )
        elif dataset_name_lower in ("miniarc", "mini-arc", "mini"):
            recommendations.update(
                {
                    "dataset.max_grid_height": 5,
                    "dataset.max_grid_width": 5,
                    "dataset.dataset_name": "MiniARC",
                }
            )
        elif dataset_name_lower in ("arc-agi-1", "agi1", "agi-1"):
            recommendations.update(
                {
                    "dataset.max_grid_height": 30,
                    "dataset.max_grid_width": 30,
                    "dataset.dataset_name": "ARC-AGI-1",
                }
            )
        elif dataset_name_lower in ("arc-agi-2", "agi2", "agi-2"):
            recommendations.update(
                {
                    "dataset.max_grid_height": 30,
                    "dataset.max_grid_width": 30,
                    "dataset.dataset_name": "ARC-AGI-2",
                }
            )
        else:
            logger.warning(f"No recommendations available for dataset: {dataset_name}")

        return recommendations
