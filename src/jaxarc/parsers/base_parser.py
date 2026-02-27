"""Abstract base class for ARC data parsers.

This module defines the standard interface that all ARC data parsers must implement.
It provides a contract for loading, preprocessing, and serving ARC task data in a
JAX-compatible format for the MARL environment.
"""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from loguru import logger
from pyprojroot import here

from jaxarc.configs import DatasetConfig
from jaxarc.types import (
    ColorValue,
    GridArray,
    GridHeight,
    GridWidth,
    JaxArcTask,
    MaskArray,
    PRNGKey,
)
from jaxarc.utils.task_manager import create_jax_task_index

# Type aliases for parser functions
GridList = list[GridArray]
MaskList = list[MaskArray]


class ArcDataParserBase(ABC):
    """Abstract base class for all ARC data parsers.

    This class defines the standard interface for parsers that convert raw ARC
    dataset files into JAX-compatible JaxArcTask structures. Concrete
    implementations should handle dataset-specific formats while maintaining
    a consistent API.

    The parser is designed to work with typed DatasetConfig objects,
    ensuring all JAX arrays have static shapes required for efficient JIT compilation.

    Attributes:
        config: Typed dataset configuration containing parser settings
        max_grid_height: Maximum height for grid padding
        max_grid_width: Maximum width for grid padding
        min_grid_height: Minimum height for valid grids
        min_grid_width: Minimum width for valid grids
        max_colors: Maximum number of colors in the ARC color palette
        background_color: Default background color value
        max_train_pairs: Maximum number of training pairs per task
        max_test_pairs: Maximum number of test pairs per task
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the parser with typed configuration.

        Args:
            config: Typed dataset configuration containing parser settings such as
                   dataset paths, max dimensions, and other parser-specific parameters

        Raises:
            ValueError: If configuration validation fails
        """
        # Validate the configuration
        validation_errors = config.validate()
        if validation_errors:
            msg = f"Configuration validation failed: {'; '.join(validation_errors)}"
            raise ValueError(msg)

        # Store the typed configuration
        self.config = config

        # Common state for lazy loading (concrete parsers call _scan_available_tasks)
        self._task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}
        self._data_dir: Path | None = None

        # Extract commonly used values for convenience
        self.max_grid_height = config.max_grid_height
        self.max_grid_width = config.max_grid_width
        self.min_grid_height = config.min_grid_height
        self.min_grid_width = config.min_grid_width
        self.max_colors = config.max_colors
        self.background_color = config.background_color
        self.max_train_pairs = config.max_train_pairs
        self.max_test_pairs = config.max_test_pairs

    def get_data_path(self) -> str:
        """Get the actual data path based on dataset type and split.

        This method should be overridden by subclasses to handle dataset-specific
        path resolution based on the task_split.

        Returns:
            str: The resolved path to the data directory
        """
        # Default implementation just returns the configured path
        return self.config.dataset_path

    @classmethod
    def from_hydra(cls, hydra_config):
        """Create parser from Hydra configuration for backward compatibility.

        This class method provides backward compatibility with existing Hydra-based
        configurations while internally using typed DatasetConfig objects for
        better type safety and validation.

        Args:
            hydra_config: Raw Hydra DictConfig for dataset configuration containing
                         fields like dataset_path, max_grid_height, max_grid_width,
                         and other dataset-specific settings.

        Returns:
            Parser instance initialized with typed DatasetConfig converted from
            the provided Hydra configuration.

        Examples:
            ```python
            from omegaconf import DictConfig

            # Hydra configuration
            hydra_config = DictConfig(
                {
                    "dataset_path": "data/MiniARC",
                    "max_grid_height": 5,
                    "max_grid_width": 5,
                    # ... other fields
                }
            )

            # Create parser using from_hydra
            parser = MiniArcParser.from_hydra(hydra_config)
            ```

        Note:
            This method is provided for backward compatibility. For new code,
            prefer creating DatasetConfig objects directly and using the
            standard __init__ method.
        """
        dataset_config = DatasetConfig.from_hydra(hydra_config)
        return cls(dataset_config)

    @property
    def _dataset_name(self) -> str:
        """Name of the dataset for error messages. Override in subclasses."""
        return "dataset"

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

    def _extract_task_content(self, raw_task_data: Any) -> dict:
        """Extract task content dict from raw task data.

        Override in subclasses that need custom extraction logic (e.g., ArcAgi
        validates GitHub format). Default returns raw_task_data as-is.

        Args:
            raw_task_data: Raw task data as loaded from disk

        Returns:
            Task content dictionary with 'train' and 'test' keys
        """
        return raw_task_data

    def preprocess_task_data(
        self,
        raw_task_data: Any,
        key: PRNGKey,  # noqa: ARG002
        task_id: str | None = None,
    ) -> JaxArcTask:
        """Convert raw task data into JaxArcTask structure.

        Supports both direct task content dicts and (task_id, task_content) tuples
        for backward compatibility.

        Args:
            raw_task_data: Raw task data dictionary or (task_id, task_content) tuple
            key: JAX PRNG key (unused in this deterministic preprocessing)
            task_id: Optional task ID. If not provided, extracted from raw_task_data.

        Returns:
            JaxArcTask: JAX-compatible task data with padded arrays

        Raises:
            ValueError: If the task data format is invalid
        """
        # Handle (task_id, task_content) tuple format for backward compatibility
        if isinstance(raw_task_data, tuple) and len(raw_task_data) == 2:
            tuple_id, task_content = raw_task_data
            if task_id is None:
                task_id = tuple_id
        else:
            task_content = self._extract_task_content(raw_task_data)
            if task_id is None:
                task_id = "unknown"

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

    def get_random_task(self, key: PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset with lazy loading.

        Args:
            key: JAX PRNG key for random selection

        Returns:
            JaxArcTask: A randomly selected and preprocessed task

        Raises:
            RuntimeError: If no tasks are available
        """
        if not self._task_ids:
            msg = f"No tasks available in {self._dataset_name}"
            raise RuntimeError(msg)

        # Randomly select a task ID
        task_index = jax.random.randint(key, (), 0, len(self._task_ids))
        task_id = self._task_ids[int(task_index)]

        # Use get_task_by_id which handles lazy loading
        return self.get_task_by_id(task_id)

    def get_max_dimensions(self) -> tuple[int, int, int, int]:
        """Get the maximum dimensions used by this parser.

        Returns:
            Tuple of (max_grid_height, max_grid_width, max_train_pairs, max_test_pairs)
        """
        return (
            self.max_grid_height,
            self.max_grid_width,
            self.max_train_pairs,
            self.max_test_pairs,
        )

    def get_grid_config(self) -> dict[str, int]:
        """Get the grid configuration settings.

        Returns:
            Dictionary containing grid configuration values
        """
        return {
            "max_grid_height": self.max_grid_height,
            "max_grid_width": self.max_grid_width,
            "min_grid_height": self.min_grid_height,
            "min_grid_width": self.min_grid_width,
            "max_colors": self.max_colors,
            "background_color": self.background_color,
        }

    def validate_grid_dimensions(self, height: GridHeight, width: GridWidth) -> None:
        """Validate that grid dimensions are within the configured bounds.

        Args:
            height: Grid height to validate
            width: Grid width to validate

        Raises:
            ValueError: If dimensions are outside the configured bounds
        """
        if height < self.min_grid_height or width < self.min_grid_width:
            msg = (
                f"Grid dimensions ({height}x{width}) are below minimum "
                f"({self.min_grid_height}x{self.min_grid_width})"
            )
            raise ValueError(msg)
        if height > self.max_grid_height or width > self.max_grid_width:
            msg = (
                f"Grid dimensions ({height}x{width}) exceed maximum "
                f"({self.max_grid_height}x{self.max_grid_width})"
            )
            raise ValueError(msg)

    def validate_color_value(self, color: ColorValue) -> None:
        """Validate that a color value is within the allowed range.

        Args:
            color: Color value to validate

        Raises:
            ValueError: If color is outside the valid range
        """
        if color < 0 or color >= self.max_colors:
            msg = f"Color value ({color}) must be in range [0, {self.max_colors})"
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

            input_grid = self._convert_grid_to_jax(pair["input"])
            output_grid = self._convert_grid_to_jax(pair["output"])

            # Validate grid dimensions
            self.validate_grid_dimensions(*input_grid.shape)
            self.validate_grid_dimensions(*output_grid.shape)

            # Validate color values
            self._validate_grid_colors(input_grid)
            self._validate_grid_colors(output_grid)

            train_input_grids.append(input_grid)
            train_output_grids.append(output_grid)

        return train_input_grids, train_output_grids

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
        from ..utils.grid_utils import pad_array_sequence

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

    def _validate_grid_colors(self, grid) -> None:
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

    def _validate_arc_grid_data(self, grid_data: list[list[int]]) -> None:
        """Validate that grid data is in the correct ARC format.

        Args:
            grid_data: Grid as list of lists of integers

        Raises:
            ValueError: If grid format is invalid
        """
        if not grid_data:
            raise ValueError("Grid data cannot be empty")

        if not isinstance(grid_data, list):
            raise ValueError("Grid data must be a list")

        if not all(isinstance(row, list) for row in grid_data):
            raise ValueError("Grid data must be a list of lists")

        # Check consistent row lengths
        if grid_data:
            row_length = len(grid_data[0])
            if not all(len(row) == row_length for row in grid_data):
                raise ValueError("All rows in grid must have the same length")

        # Check that all cells are integers in valid range
        for i, row in enumerate(grid_data):
            for j, cell in enumerate(row):
                if not isinstance(cell, int):
                    raise ValueError(
                        f"Grid cell at ({i}, {j}) must be an integer, got {type(cell)}"
                    )
                if not (0 <= cell <= 9):
                    raise ValueError(
                        f"Grid cell at ({i}, {j}) has value {cell}, must be 0-9"
                    )

    def _convert_grid_to_jax(self, grid_data: list[list[int]]) -> jnp.ndarray:
        """Convert grid data from list format to JAX array.

        Args:
            grid_data: Grid as list of lists of integers

        Returns:
            JAX array of shape (height, width) with int32 dtype

        Raises:
            ValueError: If grid format is invalid
        """
        self._validate_arc_grid_data(grid_data)
        return jnp.array(grid_data, dtype=jnp.int32)

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

        task_info = f"Task {task_id}" if task_id else "Task"
        logger.debug(
            f"{task_info}: {len(train_input_grids)} train pairs, {len(test_input_grids)} test pairs, "
            f"max grid size: {max_dims[0]}x{max_dims[1]}"
        )

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

            input_grid = self._convert_grid_to_jax(pair["input"])
            self.validate_grid_dimensions(*input_grid.shape)
            self._validate_grid_colors(input_grid)
            test_input_grids.append(input_grid)

            # For test pairs, output might be provided or missing
            if "output" in pair and pair["output"] is not None:
                output_grid = self._convert_grid_to_jax(pair["output"])
                self.validate_grid_dimensions(*output_grid.shape)
                self._validate_grid_colors(output_grid)
                test_output_grids.append(output_grid)
            else:
                # Create dummy output grid (will be masked as invalid)
                dummy_output = jnp.zeros_like(input_grid)
                test_output_grids.append(dummy_output)

        return test_input_grids, test_output_grids

    # =========================================================================
    # Task Index to Task ID Mapping System
    # =========================================================================

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get a specific task by its ID with lazy loading.

        Tasks are loaded from disk on first access and cached for subsequent calls.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            JaxArcTask: The preprocessed task data

        Raises:
            ValueError: If the task ID is not found
        """
        if task_id not in self._task_ids:
            msg = f"Task ID '{task_id}' not found in {self._dataset_name}"
            raise ValueError(msg)

        # Lazy load: check cache first, load from disk if not cached
        if task_id not in self._cached_tasks:
            self._load_task_from_disk(task_id)

        # Get the cached task data
        task_data = self._cached_tasks[task_id]

        # Create a dummy key for preprocessing (deterministic)
        key = jax.random.PRNGKey(0)

        # Preprocess and return
        return self.preprocess_task_data(task_data, key, task_id=task_id)

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Returns:
            List of task IDs available in the dataset
        """
        return self._task_ids.copy()

    def _scan_available_tasks(self) -> None:
        """Scan directory for available task IDs without loading task data.

        Override in subclasses with non-standard directory structures
        (e.g., ConceptARC's nested concept group directories).
        """
        try:
            data_dir_path = self.get_data_path()
            self._data_dir = here(data_dir_path)

            if not self._data_dir.exists() or not self._data_dir.is_dir():
                msg = f"Data directory not found: {self._data_dir}"
                raise RuntimeError(msg)

            json_files = list(self._data_dir.glob("*.json"))
            if not json_files:
                msg = f"No JSON files found in {self._data_dir}"
                raise RuntimeError(msg)

            self._task_ids = [f.stem for f in json_files]

            logger.info(
                f"Found {len(self._task_ids)} tasks in {self._data_dir} "
                f"(lazy loading - tasks loaded on-demand)"
            )

        except Exception as e:
            logger.error(f"Error scanning tasks: {e}")
            raise

    def _load_task_from_disk(self, task_id: str) -> None:
        """Load a single task from disk and add to cache.

        Override in subclasses that need additional validation or metadata
        (e.g., MiniARC constraint validation, ConceptARC metadata).

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

    def validate_task_index_mapping(self, task_index: int) -> bool:
        """Validate that a task_index can be resolved to a valid task.

        This method checks if a given task_index corresponds to a task
        that exists in the current dataset.

        Args:
            task_index: Integer task index to validate

        Returns:
            True if the task_index can be resolved, False otherwise
        """
        from jaxarc.utils.task_manager import get_task_id_globally

        # Get task_id from global task manager
        task_id = get_task_id_globally(task_index)
        if task_id is None:
            return False

        # Check if this parser has the task
        available_ids = self.get_available_task_ids()
        return task_id in available_ids

    def reconstruct_task_from_index(self, task_index: int) -> JaxArcTask:
        """Reconstruct task_data from task_index.

        This method is used during deserialization to reconstruct the full
        task_data from a stored task_index.

        Args:
            task_index: Integer task index to reconstruct

        Returns:
            JaxArcTask: Reconstructed task data

        Raises:
            ValueError: If task_index cannot be resolved or task not found
        """
        from jaxarc.utils.task_manager import get_task_id_globally

        # Get task_id from global task manager
        task_id = get_task_id_globally(task_index)
        if task_id is None:
            raise ValueError(
                f"Task index {task_index} not found in global task manager"
            )

        # Get the task using the task_id
        try:
            return self.get_task_by_id(task_id)
        except ValueError as e:
            raise ValueError(
                f"Cannot reconstruct task from index {task_index}: {e}"
            ) from e

    def get_task_index_for_id(self, task_id: str) -> int:
        """Get the task_index for a given task_id.

        This method looks up the task_index for a task_id, registering
        the task if it's not already in the global task manager.

        Args:
            task_id: String task ID to look up

        Returns:
            Integer task index

        Raises:
            ValueError: If task_id is not available in this parser
        """
        from jaxarc.utils.task_manager import register_task_globally

        # Validate that this parser has the task
        available_ids = self.get_available_task_ids()
        if task_id not in available_ids:
            raise ValueError(f"Task ID '{task_id}' not available in this parser")

        # Register/get the task index
        return register_task_globally(task_id)
