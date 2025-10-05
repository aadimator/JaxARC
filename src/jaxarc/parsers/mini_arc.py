"""MiniARC dataset parser implementation.

This module provides a parser for the MiniARC dataset, which is a 5x5 compact version
of ARC with 400 training and 400 evaluation tasks. The parser is optimized for smaller
grid dimensions and faster processing, making it ideal for rapid prototyping and testing.
"""

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


class MiniArcParser(ArcDataParserBase):
    """Parser for MiniARC dataset optimized for 5x5 grids.

    MiniARC is a compact version of ARC with 400+ individual task files designed
    for faster experimentation and prototyping. All grids are constrained to a
    maximum size of 5x5, enabling rapid iteration and testing.

    The dataset follows a flat directory structure with individual JSON files
    for each task, using descriptive filenames that indicate task purpose.

    This parser provides optimizations specific to the smaller grid constraints
    and maintains compatibility with the existing JaxArcTask interface.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the MiniArcParser with typed configuration.

        This parser accepts a typed DatasetConfig object for better type safety
        and validation. For backward compatibility with Hydra configurations,
        use the from_hydra() class method.

        Args:
            config: Typed dataset configuration containing paths and parser settings,
                   optimized for 5x5 grid constraints. Must include dataset_path,
                   max_grid_height, max_grid_width, and other required fields.

        Examples:
            ```python
            # Direct typed config usage (preferred)
            from jaxarc.configs import DatasetConfig
            from omegaconf import DictConfig

            hydra_config = DictConfig({...})
            dataset_config = DatasetConfig.from_hydra(hydra_config)
            parser = MiniArcParser(dataset_config)

            # Alternative: use from_hydra class method
            parser = MiniArcParser.from_hydra(hydra_config)
            ```

        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        super().__init__(config)

        # Validate and warn about grid constraints for MiniARC optimization
        self._validate_grid_constraints()

        # Lazy loading: only scan for task IDs, don't load data yet
        self._task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}  # Lazy cache: load on first access
        self._data_dir: Path | None = None

        # Scan available tasks (lazy loading)
        self._scan_available_tasks()

    def get_data_path(self) -> str:
        """Get the actual data path for MiniARC based on split.

        MiniARC structure: {base_path}/data/MiniARC (only one dataset)

        Returns:
            str: The resolved path to the MiniARC data directory
        """
        base_path = self.config.dataset_path
        return f"{base_path}/data/MiniARC"

    def _validate_grid_constraints(self) -> None:
        """Validate configuration is optimized for 5x5 grids."""
        if self.max_grid_height > 5 or self.max_grid_width > 5:
            logger.warning(
                f"MiniARC is optimized for 5x5 grids, but configuration allows "
                f"{self.max_grid_height}x{self.max_grid_width}. Consider using "
                f"max_grid_height=5 and max_grid_width=5 for optimal performance."
            )

        # Log optimization status
        if self.max_grid_height == 5 and self.max_grid_width == 5:
            logger.info("MiniARC parser configured with optimal 5x5 grid constraints")

    def _scan_available_tasks(self) -> None:
        """Scan directory for available task IDs without loading task data.

        This is much faster than loading all tasks - we only read filenames,
        not file contents. Tasks are loaded on-demand when requested.
        """
        try:
            # Get resolved data path
            tasks_path = self.get_data_path()
            self._data_dir = here(tasks_path)

            if not self._data_dir.exists():
                msg = f"MiniARC tasks directory not found: {self._data_dir}"
                raise RuntimeError(msg)

            # Scan for JSON files - just get filenames, don't load content
            task_files = list(self._data_dir.glob("*.json"))

            if not task_files:
                msg = f"No JSON task files found in {self._data_dir}"
                raise RuntimeError(msg)

            # Extract task IDs from filenames
            self._task_ids = [f.stem for f in task_files]

            logger.info(
                f"Found {len(self._task_ids)} tasks in MiniARC dataset "
                f"(lazy loading - tasks loaded on-demand, optimized for 5x5 grids)"
            )

        except Exception as e:
            logger.error(f"Error scanning MiniARC tasks: {e}")
            raise

    def _load_task_from_disk(self, task_id: str) -> None:
        """Load a single task from disk and add to cache.

        Args:
            task_id: ID of the task to load

        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If JSON is invalid or violates MiniARC constraints
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

            # Validate task structure
            self._validate_task_structure(task_data, task_id)

            # Validate MiniARC-specific constraints (5x5 optimization)
            self._validate_miniarc_constraints(task_data, task_id)

            self._cached_tasks[task_id] = task_data
            logger.debug(f"Loaded MiniARC task '{task_id}' from disk")
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {task_file}: {e}"
            raise ValueError(msg) from e

    def _validate_task_structure(self, task_data: dict, task_id: str) -> None:
        """Validate that task data has the required ARC structure.

        Args:
            task_data: Task data dictionary
            task_id: Task identifier for error reporting

        Raises:
            ValueError: If task structure is invalid
        """
        if not isinstance(task_data, dict):
            msg = f"Task {task_id}: Expected dict, got {type(task_data)}"
            raise ValueError(msg)

        # Check for required sections
        if "train" not in task_data:
            msg = f"Task {task_id}: Missing 'train' section"
            raise ValueError(msg)

        if "test" not in task_data:
            msg = f"Task {task_id}: Missing 'test' section"
            raise ValueError(msg)

        # Validate training pairs
        train_pairs = task_data["train"]
        if not isinstance(train_pairs, list) or not train_pairs:
            msg = f"Task {task_id}: 'train' must be a non-empty list"
            raise ValueError(msg)

        # Validate test pairs
        test_pairs = task_data["test"]
        if not isinstance(test_pairs, list) or not test_pairs:
            msg = f"Task {task_id}: 'test' must be a non-empty list"
            raise ValueError(msg)

    def _validate_miniarc_constraints(self, task_data: dict, task_id: str) -> None:
        """Validate that task data meets MiniARC 5x5 constraints.

        Args:
            task_data: Task data dictionary
            task_id: Task identifier for error reporting

        Raises:
            ValueError: If grids exceed 5x5 constraints
        """
        # Check all grids in training pairs
        for i, pair in enumerate(task_data["train"]):
            if "input" in pair:
                self._validate_grid_size(pair["input"], f"{task_id} train[{i}].input")
            if "output" in pair:
                self._validate_grid_size(pair["output"], f"{task_id} train[{i}].output")

        # Check all grids in test pairs
        for i, pair in enumerate(task_data["test"]):
            if "input" in pair:
                self._validate_grid_size(pair["input"], f"{task_id} test[{i}].input")
            if "output" in pair:
                self._validate_grid_size(pair["output"], f"{task_id} test[{i}].output")

    def _validate_grid_size(self, grid_data: list[list[int]], grid_name: str) -> None:
        """Validate that a grid meets the 5x5 size constraint.

        Args:
            grid_data: Grid as list of lists
            grid_name: Grid identifier for error reporting

        Raises:
            ValueError: If grid exceeds 5x5 dimensions
        """
        if not grid_data:
            return

        height = len(grid_data)
        width = len(grid_data[0]) if grid_data else 0

        # For MiniARC, we enforce strict 5x5 constraint
        if height > 5 or width > 5:
            msg = (
                f"Grid {grid_name} has dimensions {height}x{width}, "
                f"which exceeds MiniARC 5x5 constraint"
            )
            raise ValueError(msg)

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
    ) -> JaxArcTask:
        """Convert raw task data into JaxArcTask structure.

        Args:
            raw_task_data: Raw task data dictionary or tuple (task_id, task_content)
            key: JAX PRNG key (unused in this deterministic preprocessing)

        Returns:
            JaxArcTask: JAX-compatible task data with padded arrays

        Raises:
            ValueError: If the task data format is invalid
        """
        # Handle both direct task data and (task_id, task_content) tuple
        if isinstance(raw_task_data, tuple) and len(raw_task_data) == 2:
            task_id, task_content = raw_task_data
        else:
            # Assume it's direct task content, create a dummy task ID
            task_content = raw_task_data
            task_id = "unknown_miniarc_task"

        # Process training and test pairs
        train_input_grids, train_output_grids = self._process_training_pairs(
            task_content
        )
        test_input_grids, test_output_grids = self._process_test_pairs(task_content)

        # Pad arrays and create masks (optimized for smaller grids)
        padded_arrays = self._pad_and_create_masks(
            train_input_grids, train_output_grids, test_input_grids, test_output_grids
        )

        # Log parsing statistics
        self._log_parsing_stats(
            train_input_grids,
            train_output_grids,
            test_input_grids,
            test_output_grids,
            task_id,
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
            task_index=create_jax_task_index(task_id),
        )

    def _process_training_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process training pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (train_input_grids, train_output_grids)

        Raises:
            ValueError: If training data is invalid
        """
        # Call parent implementation but customize error message for MiniARC
        try:
            return super()._process_training_pairs(task_content)
        except ValueError as e:
            if "Task must have at least one training pair" in str(e):
                msg = "MiniARC task must have at least one training pair"
                raise ValueError(msg) from e
            raise

    def _process_test_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process test pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (test_input_grids, test_output_grids)

        Raises:
            ValueError: If test data is invalid
        """
        # Call parent implementation but customize error message for MiniARC
        try:
            return super()._process_test_pairs(task_content)
        except ValueError as e:
            if "Task must have at least one test pair" in str(e):
                msg = "MiniARC task must have at least one test pair"
                raise ValueError(msg) from e
            raise

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
            msg = "No tasks available in MiniARC dataset"
            raise RuntimeError(msg)

        # Randomly select a task ID
        task_index = jax.random.randint(key, (), 0, len(self._task_ids))
        task_id = self._task_ids[int(task_index)]

        # Use get_task_by_id which handles lazy loading
        return self.get_task_by_id(task_id)

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
            msg = f"Task ID '{task_id}' not found in MiniARC dataset"
            raise ValueError(msg)

        # Lazy loading: load task from disk if not in cache
        if task_id not in self._cached_tasks:
            self._load_task_from_disk(task_id)

        # Get the cached task data
        task_data = self._cached_tasks[task_id]

        # Create a dummy key for preprocessing (deterministic)
        key = jax.random.PRNGKey(0)

        # Preprocess and return
        return self.preprocess_task_data((task_id, task_data), key)

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Returns:
            List of task IDs available in the dataset
        """
        return self._task_ids.copy()

    def get_dataset_statistics(self) -> dict:
        """Get statistics about the MiniARC dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        if not self._task_ids:
            return {
                "total_tasks": 0,
                "optimization": "5x5 grids",
                "max_configured_dimensions": f"{self.max_grid_height}x{self.max_grid_width}",
            }

        # Calculate grid size statistics
        grid_sizes = []
        train_pair_counts = []
        test_pair_counts = []

        for task_id in self._task_ids:
            # Lazy loading: load task if not in cache
            if task_id not in self._cached_tasks:
                self._load_task_from_disk(task_id)
            task_data = self._cached_tasks[task_id]

            # Count training and test pairs
            train_pairs = len(task_data.get("train", []))
            test_pairs = len(task_data.get("test", []))
            train_pair_counts.append(train_pairs)
            test_pair_counts.append(test_pairs)

            # Find maximum grid size in this task
            max_height, max_width = 0, 0

            for pair in task_data.get("train", []):
                if "input" in pair:
                    h, w = (
                        len(pair["input"]),
                        len(pair["input"][0]) if pair["input"] else 0,
                    )
                    max_height, max_width = max(max_height, h), max(max_width, w)
                if "output" in pair:
                    h, w = (
                        len(pair["output"]),
                        len(pair["output"][0]) if pair["output"] else 0,
                    )
                    max_height, max_width = max(max_height, h), max(max_width, w)

            for pair in task_data.get("test", []):
                if "input" in pair:
                    h, w = (
                        len(pair["input"]),
                        len(pair["input"][0]) if pair["input"] else 0,
                    )
                    max_height, max_width = max(max_height, h), max(max_width, w)
                if "output" in pair:
                    h, w = (
                        len(pair["output"]),
                        len(pair["output"][0]) if pair["output"] else 0,
                    )
                    max_height, max_width = max(max_height, h), max(max_width, w)

            grid_sizes.append((max_height, max_width))

        # Calculate statistics
        stats = {
            "total_tasks": len(self._task_ids),
            "optimization": "5x5 grids",
            "max_configured_dimensions": f"{self.max_grid_height}x{self.max_grid_width}",
            "train_pairs": {
                "min": min(train_pair_counts),
                "max": max(train_pair_counts),
                "avg": sum(train_pair_counts) / len(train_pair_counts),
            },
            "test_pairs": {
                "min": min(test_pair_counts),
                "max": max(test_pair_counts),
                "avg": sum(test_pair_counts) / len(test_pair_counts),
            },
            "grid_dimensions": {
                "max_height": max(size[0] for size in grid_sizes),
                "max_width": max(size[1] for size in grid_sizes),
                "avg_height": sum(size[0] for size in grid_sizes) / len(grid_sizes),
                "avg_width": sum(size[1] for size in grid_sizes) / len(grid_sizes),
            },
        }

        # Check if dataset is truly optimized for 5x5
        max_actual_height = stats["grid_dimensions"]["max_height"]
        max_actual_width = stats["grid_dimensions"]["max_width"]
        if max_actual_height <= 5 and max_actual_width <= 5:
            stats["is_5x5_optimized"] = True
        else:
            stats["is_5x5_optimized"] = False
            stats["warning"] = (
                f"Some grids exceed 5x5: max actual size is {max_actual_height}x{max_actual_width}"
            )

        return stats
