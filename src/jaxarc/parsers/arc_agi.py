from __future__ import annotations

from typing import Any

from jaxarc.configs import DatasetConfig

from .base_parser import ArcDataParserBase


class ArcAgiParser(ArcDataParserBase):
    """Parses ARC-AGI task files into JaxArcTask objects.

    This parser supports ARC-AGI datasets downloaded from GitHub repositories, including:
    - ARC-AGI-1 (fchollet/ARC-AGI repository)
    - ARC-AGI-2 (arcprize/ARC-AGI-2 repository)

    Both datasets follow the GitHub format with individual JSON files per task.
    Each task file contains complete task data including training pairs and test pairs
    with outputs when available.

    The parser outputs JAX-compatible JaxArcTask structures with padded
    arrays and boolean masks for efficient processing in the SARL environment.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the ArcAgiParser with typed configuration.

        This parser accepts a typed DatasetConfig object for better type safety
        and validation. For backward compatibility with Hydra configurations,
        use the from_hydra() class method.

        Args:
            config: Typed dataset configuration containing paths and parser settings,
                   including max_grid_height, max_grid_width, max_train_pairs, max_test_pairs,
                   dataset_path, and task_split ("train" or "evaluation").

        Examples:
            ```python
            # Direct typed config usage (preferred)
            from jaxarc.configs import DatasetConfig
            from omegaconf import DictConfig

            hydra_config = DictConfig({...})
            dataset_config = DatasetConfig.from_hydra(hydra_config)
            parser = ArcAgiParser(dataset_config)

            # Alternative: use from_hydra class method
            parser = ArcAgiParser.from_hydra(hydra_config)
            ```

        Raises:
            ValueError: If configuration is invalid or missing required fields
            RuntimeError: If data directory is not found or contains no JSON files
        """
        super().__init__(config)
        self._scan_available_tasks()

    def get_data_path(self) -> str:
        """Get the actual data path for ARC-AGI based on split.

        ARC-AGI structure: {base_path}/data/{split}
        where split can be 'training' or 'evaluation'

        Returns:
            str: The resolved path to the ARC-AGI data directory
        """
        base_path = self.config.dataset_path
        split = "training" if self.config.task_split == "train" else "evaluation"
        return f"{base_path}/data/{split}"

    def _extract_task_content(self, raw_task_data: Any) -> dict:
        """Extract and validate task content from raw ARC-AGI data.

        Validates that the raw data follows the GitHub format with
        'train' and 'test' keys.

        Args:
            raw_task_data: Raw task data dictionary

        Returns:
            Task content dictionary with 'train' and 'test' keys

        Raises:
            ValueError: If the task data format is invalid
        """
        if not isinstance(raw_task_data, dict):
            msg = f"Expected dict, got {type(raw_task_data)}"
            raise ValueError(msg)

        if "train" in raw_task_data and "test" in raw_task_data:
            return raw_task_data

        msg = "Invalid task data format. Expected GitHub format with 'train' and 'test' keys"
        raise ValueError(msg)
