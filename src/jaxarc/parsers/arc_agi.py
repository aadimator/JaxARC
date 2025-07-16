from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from jaxarc.types import JaxArcTask
from jaxarc.utils.task_manager import create_jax_task_index

from .base_parser import ArcDataParserBase
from .utils import convert_grid_to_jax, log_parsing_stats, pad_array_sequence


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

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """Initialize the ArcAgiParser with configuration.

        Args:
            cfg: Configuration object containing dataset paths and parser settings,
                 including max_grid_height, max_grid_width, max_train_pairs, and max_test_pairs
        """
        super().__init__(cfg)

        # Load and cache all tasks in memory
        self._task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}

        self._load_and_cache_tasks()

    def _load_and_cache_tasks(self) -> None:
        """Load and cache all tasks from individual JSON files in GitHub format."""
        try:
            # Load from default split (usually 'training')
            default_split = self.cfg.get("default_split", "training")
            split_config = self.cfg.get(default_split, {})

            # GitHub format uses directory paths instead of file paths
            data_dir_path = split_config.get("path")
            if not data_dir_path:
                # Check if this is legacy Kaggle format configuration
                if "challenges" in split_config:
                    raise RuntimeError(
                        "Legacy Kaggle format detected. Please update configuration to use GitHub format with 'path' instead of 'challenges'/'solutions'"
                    )
                raise RuntimeError("No data path specified in configuration")

            data_dir = here(data_dir_path)
            if not data_dir.exists() or not data_dir.is_dir():
                raise RuntimeError(f"Data directory not found: {data_dir}")

            # Load individual JSON files
            json_files = list(data_dir.glob("*.json"))
            if not json_files:
                raise RuntimeError(f"No JSON files found in {data_dir}")

            self._cached_tasks = {}
            for json_file in json_files:
                task_id = json_file.stem  # filename without extension
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        task_data = json.load(f)
                    self._cached_tasks[task_id] = task_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error loading task file {json_file}: {e}")
                    continue

            self._task_ids = list(self._cached_tasks.keys())
            logger.info(
                f"Loaded {len(self._cached_tasks)} tasks from GitHub format in {data_dir}"
            )

        except Exception as e:
            logger.error(f"Error loading and caching tasks: {e}")
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

    def _process_training_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process training pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (train_input_grids, train_output_grids)

        Raises:
            ValueError: If training data is invalid
        """
        train_pairs_data = task_content.get("train", [])

        if not train_pairs_data:
            msg = "Task must have at least one training pair"
            raise ValueError(msg)

        train_input_grids = []
        train_output_grids = []

        for i, pair in enumerate(train_pairs_data):
            if "input" not in pair or "output" not in pair:
                msg = f"Training pair {i} missing input or output"
                raise ValueError(msg)

            input_grid = convert_grid_to_jax(pair["input"])
            output_grid = convert_grid_to_jax(pair["output"])

            # Validate grid dimensions
            self.validate_grid_dimensions(*input_grid.shape)
            self.validate_grid_dimensions(*output_grid.shape)

            # Validate color values
            self._validate_grid_colors(input_grid)
            self._validate_grid_colors(output_grid)

            train_input_grids.append(input_grid)
            train_output_grids.append(output_grid)

        return train_input_grids, train_output_grids

    def _process_test_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process test pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (test_input_grids, test_output_grids)

        Raises:
            ValueError: If test data is invalid
        """
        test_pairs_data = task_content.get("test", [])

        if not test_pairs_data:
            msg = "Task must have at least one test pair"
            raise ValueError(msg)

        test_input_grids = []
        test_output_grids = []

        for i, pair in enumerate(test_pairs_data):
            if "input" not in pair:
                msg = f"Test pair {i} missing input"
                raise ValueError(msg)

            input_grid = convert_grid_to_jax(pair["input"])
            self.validate_grid_dimensions(*input_grid.shape)
            self._validate_grid_colors(input_grid)
            test_input_grids.append(input_grid)

            # For test pairs, output might be None or provided depending on the dataset
            if "output" in pair and pair["output"] is not None:
                output_grid = convert_grid_to_jax(pair["output"])
                self.validate_grid_dimensions(*output_grid.shape)
                self._validate_grid_colors(output_grid)
                test_output_grids.append(output_grid)
            else:
                # Create dummy output grid (will be masked as invalid)
                dummy_output = jnp.zeros_like(input_grid)
                test_output_grids.append(dummy_output)

        return test_input_grids, test_output_grids

    def _pad_and_create_masks(
        self,
        train_input_grids: list,
        train_output_grids: list,
        test_input_grids: list,
        test_output_grids: list,
    ) -> dict:
        """Pad arrays and create validity masks.

        Args:
            train_input_grids: List of training input grids
            train_output_grids: List of training output grids
            test_input_grids: List of test input grids
            test_output_grids: List of test output grids

        Returns:
            Dictionary containing padded arrays and masks
        """
        # Pad all arrays to maximum dimensions
        padded_train_inputs, train_input_masks = pad_array_sequence(
            train_input_grids,
            self.max_train_pairs,
            self.max_grid_height,
            self.max_grid_width,
            fill_value=-1,  # Use -1 as fill value for inputs
        )

        padded_train_outputs, train_output_masks = pad_array_sequence(
            train_output_grids,
            self.max_train_pairs,
            self.max_grid_height,
            self.max_grid_width,
            fill_value=-1,
        )

        padded_test_inputs, test_input_masks = pad_array_sequence(
            test_input_grids,
            self.max_test_pairs,
            self.max_grid_height,
            self.max_grid_width,
            fill_value=-1,
        )

        padded_test_outputs, test_output_masks = pad_array_sequence(
            test_output_grids,
            self.max_test_pairs,
            self.max_grid_height,
            self.max_grid_width,
            fill_value=-1,
        )

        return {
            "train_inputs": padded_train_inputs,
            "train_input_masks": train_input_masks,
            "train_outputs": padded_train_outputs,
            "train_output_masks": train_output_masks,
            "test_inputs": padded_test_inputs,
            "test_input_masks": test_input_masks,
            "test_outputs": padded_test_outputs,
            "test_output_masks": test_output_masks,
        }

    def _log_parsing_stats(
        self,
        train_input_grids: list,
        train_output_grids: list,
        test_input_grids: list,
        test_output_grids: list,
        task_id: str,
    ) -> None:
        """Log parsing statistics.

        Args:
            train_input_grids: List of training input grids
            train_output_grids: List of training output grids
            test_input_grids: List of test input grids
            test_output_grids: List of test output grids
            task_id: Task identifier
        """
        max_train_dims = max(
            (grid.shape for grid in train_input_grids + train_output_grids),
            default=(0, 0),
        )
        max_test_dims = max(
            (grid.shape for grid in test_input_grids + test_output_grids),
            default=(0, 0),
        )
        max_dims = (
            max(max_train_dims[0], max_test_dims[0]),
            max(max_train_dims[1], max_test_dims[1]),
        )

        log_parsing_stats(
            len(train_input_grids), len(test_input_grids), max_dims, task_id
        )

    def _validate_grid_colors(self, grid: jnp.ndarray) -> None:
        """Validate that all colors in a grid are within the allowed range.

        Args:
            grid: JAX array representing the grid to validate

        Raises:
            ValueError: If any color value is outside the valid range
        """
        # Get unique color values in the grid
        unique_colors = jnp.unique(grid)

        # Check each color value
        for color in unique_colors:
            # Convert to Python int for validation
            color_val = int(color)
            try:
                self.validate_color_value(color_val)
            except ValueError as e:
                msg = f"Invalid color in grid: {e}"
                raise ValueError(msg) from e

    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset.

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

        # Get the cached task data (GitHub format: direct task content)
        task_data = self._cached_tasks[task_id]

        # Preprocess and return
        return self.preprocess_task_data(task_data, key, task_id)

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get a specific task by its ID.

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

        # Get the cached task data (GitHub format: direct task content)
        task_data = self._cached_tasks[task_id]

        # Create a dummy key for preprocessing (deterministic)
        key = jax.random.PRNGKey(0)

        # Preprocess and return
        return self.preprocess_task_data(task_data, key, task_id)

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Returns:
            List of task IDs available in the dataset
        """
        return self._task_ids.copy()
