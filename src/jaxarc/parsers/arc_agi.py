from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import jax
from loguru import logger
from pyprojroot import here

from jaxarc.configs import DatasetConfig
from jaxarc.types import JaxArcTask
from jaxarc.utils.task_manager import create_jax_task_index

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

        # Lazy loading: only scan for task IDs, don't load data yet
        self._task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}  # Lazy cache: load on first access
        self._data_dir: Path | None = None

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

    def _scan_available_tasks(self) -> None:
        """Scan directory for available task IDs without loading task data.

        This is much faster than loading all tasks - we only read filenames,
        not file contents. Tasks are loaded on-demand when requested.
        """
        try:
            # Get resolved data path based on split
            data_dir_path = self.get_data_path()
            self._data_dir = here(data_dir_path)

            if not self._data_dir.exists() or not self._data_dir.is_dir():
                msg = f"Data directory not found: {self._data_dir}"
                raise RuntimeError(msg)

            # Scan for JSON files - just get filenames, don't load content
            json_files = list(self._data_dir.glob("*.json"))
            if not json_files:
                msg = f"No JSON files found in {self._data_dir}"
                raise RuntimeError(msg)

            # Extract task IDs from filenames
            self._task_ids = [f.stem for f in json_files]

            logger.info(
                f"Found {len(self._task_ids)} tasks in {self._data_dir} "
                f"(lazy loading - tasks loaded on-demand)"
            )

        except Exception as e:
            logger.error(f"Error scanning tasks: {e}")
            raise

    def load_task_file(self, task_file_path: str) -> Any:
        """Load raw task data from a JSON file.

        Args:
            task_file_path: Path to the JSON file containing task data

        Returns:
            Dictionary containing the raw task data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON is invalid
        """
        file_path = Path(task_file_path)

        if not file_path.exists():
            msg = f"Task file not found: {file_path}"
            raise FileNotFoundError(msg)

        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {file_path}: {e}"
            raise ValueError(msg) from e

    def preprocess_task_data(
        self,
        raw_task_data: Any,
        key: chex.PRNGKey,  # noqa: ARG002
        task_id: str | None = None,
    ) -> JaxArcTask:
        """Convert raw task data into JaxArcTask structure.

        Args:
            raw_task_data: Raw task data dictionary
            key: JAX PRNG key (unused in this deterministic preprocessing)

        Returns:
            JaxArcTask: JAX-compatible task data with padded arrays

        Raises:
            ValueError: If the task data format is invalid
        """
        # Extract task ID and content
        extracted_task_id, task_content = self._extract_task_id_and_content(
            raw_task_data
        )

        # Use provided task_id if available, otherwise use extracted one
        final_task_id = task_id if task_id is not None else extracted_task_id

        # Process training and test pairs
        train_input_grids, train_output_grids = self._process_training_pairs(
            task_content
        )
        test_input_grids, test_output_grids = self._process_test_pairs(task_content)

        # Pad arrays and create masks
        padded_arrays = self._pad_and_create_masks(
            train_input_grids, train_output_grids, test_input_grids, test_output_grids
        )

        # Log parsing statistics
        self._log_parsing_stats(
            train_input_grids,
            train_output_grids,
            test_input_grids,
            test_output_grids,
            final_task_id,
        )

        # Create JaxArcTask structure with JAX-compatible task index
        return JaxArcTask(
            input_grids_examples=padded_arrays["train_inputs"],
            input_masks_examples=padded_arrays["train_input_masks"],
            output_grids_examples=padded_arrays["train_outputs"],
            output_masks_examples=padded_arrays["train_output_masks"],
            num_train_pairs=len(train_input_grids),
            test_input_grids=padded_arrays["test_inputs"],
            test_input_masks=padded_arrays["test_input_masks"],
            true_test_output_grids=padded_arrays["test_outputs"],
            true_test_output_masks=padded_arrays["test_output_masks"],
            num_test_pairs=len(test_input_grids),
            task_index=create_jax_task_index(final_task_id),
        )

    def _extract_task_id_and_content(self, raw_task_data: Any) -> tuple[str, dict]:
        """Extract task ID and content from raw task data.

        For GitHub format, the raw_task_data is expected to be direct task content:
        {"train": [...], "test": [...]}

        Args:
            raw_task_data: Raw task data dictionary

        Returns:
            Tuple of (task_id, task_content)

        Raises:
            ValueError: If the task data format is invalid
        """
        if not isinstance(raw_task_data, dict):
            msg = f"Expected dict, got {type(raw_task_data)}"
            raise ValueError(msg)

        # GitHub format: direct task content
        if "train" in raw_task_data and "test" in raw_task_data:
            # Task ID will be determined from filename during loading
            return "unknown", raw_task_data

        msg = "Invalid task data format. Expected GitHub format with 'train' and 'test' keys"
        raise ValueError(msg)

    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset with lazy loading.

        Args:
            key: JAX PRNG key for random selection

        Returns:
            JaxArcTask: A randomly selected and preprocessed task

        Raises:
            RuntimeError: If no tasks are available
        """
        if not self._task_ids:
            msg = "No tasks available in dataset"
            raise RuntimeError(msg)

        # Randomly select a task ID
        task_index = jax.random.randint(key, (), 0, len(self._task_ids))
        task_id = self._task_ids[int(task_index)]

        # Use get_task_by_id which handles lazy loading
        return self.get_task_by_id(task_id)

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get a specific task by its ID with lazy loading.

        Tasks are loaded from disk on first access and cached for subsequent calls.
        This is much more efficient than loading all tasks upfront.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            JaxArcTask: The preprocessed task data

        Raises:
            ValueError: If the task ID is not found
        """
        if task_id not in self._task_ids:
            msg = f"Task ID '{task_id}' not found in dataset"
            raise ValueError(msg)

        # Lazy load: check cache first, load from disk if not cached
        if task_id not in self._cached_tasks:
            self._load_task_from_disk(task_id)

        # Get the cached task data (GitHub format: direct task content)
        task_data = self._cached_tasks[task_id]

        # Create a dummy key for preprocessing (deterministic)
        key = jax.random.PRNGKey(0)

        # Preprocess and return
        return self.preprocess_task_data(task_data, key, task_id)

    def _load_task_from_disk(self, task_id: str) -> None:
        """Load a single task from disk and add to cache.

        Args:
            task_id: ID of the task to load

        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If JSON is invalid
        """
        if self._data_dir is None:
            msg = "Data directory not initialized"
            raise RuntimeError(msg)

        task_file = self._data_dir / f"{task_id}.json"

        if not task_file.exists():
            msg = f"Task file not found: {task_file}"
            raise FileNotFoundError(msg)

        try:
            with task_file.open("r", encoding="utf-8") as f:
                task_data = json.load(f)
            self._cached_tasks[task_id] = task_data
            logger.debug(f"Loaded task '{task_id}' from disk")
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {task_file}: {e}"
            raise ValueError(msg) from e

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Returns:
            List of task IDs available in the dataset
        """
        return self._task_ids.copy()
