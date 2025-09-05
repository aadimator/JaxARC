"""Abstract base class for ARC data parsers.

This module defines the standard interface that all ARC data parsers must implement.
It provides a contract for loading, preprocessing, and serving ARC task data in a
JAX-compatible format for the MARL environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from loguru import logger

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

    @abstractmethod
    def load_task_file(self, task_file_path: str) -> Any:
        """Load the raw content of a single task file.

        This method should handle the dataset-specific file format and return
        the raw data structure (e.g., dict for JSON files). Error handling
        for file access and format parsing should be implemented here.

        Args:
            task_file_path: Path to the task file to load

        Returns:
            Raw task data in dataset-specific format (e.g., dict for JSON)

        Raises:
            FileNotFoundError: If the task file doesn't exist
            ValueError: If the file format is invalid or corrupted
        """

    @abstractmethod
    def preprocess_task_data(self, raw_task_data: Any, key: PRNGKey) -> JaxArcTask:
        """Convert raw task data into a JAX-compatible JaxArcTask structure.

        This method performs the core transformation from dataset-specific format
        to the standardized JaxArcTask pytree. It should handle:
        - Converting grids to JAX arrays with proper dtypes
        - Padding grids to maximum dimensions
        - Creating boolean masks for valid data regions
        - Validating data integrity

        Args:
            raw_task_data: Raw data as returned by load_task_file
            key: JAX PRNG key for any stochastic preprocessing steps

        Returns:
            JaxArcTask: JAX-compatible task data with padded arrays and masks

        Raises:
            ValueError: If the raw data format is invalid or incompatible
        """

    @abstractmethod
    def get_random_task(self, key: PRNGKey) -> JaxArcTask:
        """Get a random task from the dataset.

        This method orchestrates the complete pipeline from task selection to
        preprocessing. It should:
        1. Use the PRNG key to randomly select a task from the dataset
        2. Load the raw task data using load_task_file
        3. Preprocess it using preprocess_task_data
        4. Return the final JaxArcTask

        Args:
            key: JAX PRNG key for random task selection and preprocessing

        Returns:
            JaxArcTask: A randomly selected and preprocessed task

        Raises:
            RuntimeError: If no tasks are available or dataset is empty
            ValueError: If task selection or preprocessing fails
        """

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

    @abstractmethod
    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get a specific task by its ID.

        This method must be implemented by concrete parsers to support
        task_data reconstruction during deserialization.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            JaxArcTask: The preprocessed task data

        Raises:
            ValueError: If the task ID is not found
        """

    @abstractmethod
    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        This method must be implemented by concrete parsers to support
        task index mapping validation.

        Returns:
            List of task IDs available in the dataset
        """

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
