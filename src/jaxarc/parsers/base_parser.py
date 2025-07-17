"""Abstract base class for ARC data parsers.

This module defines the standard interface that all ARC data parsers must implement.
It provides a contract for loading, preprocessing, and serving ARC task data in a
JAX-compatible format for the MARL environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import chex
from omegaconf import DictConfig

from jaxarc.types import JaxArcTask


class ArcDataParserBase(ABC):
    """Abstract base class for all ARC data parsers.

    This class defines the standard interface for parsers that convert raw ARC
    dataset files into JAX-compatible JaxArcTask structures. Concrete
    implementations should handle dataset-specific formats while maintaining
    a consistent API.

    The parser is designed to work with configuration-driven maximum dimensions
    for grid sizes, ensuring all JAX arrays have static shapes required for
    efficient JIT compilation.

    Attributes:
        cfg: Hydra configuration object containing parser-specific settings
        max_grid_height: Maximum height for grid padding
        max_grid_width: Maximum width for grid padding
        min_grid_height: Minimum height for valid grids
        min_grid_width: Minimum width for valid grids
        max_colors: Maximum number of colors in the ARC color palette
        background_color: Default background color value
        max_train_pairs: Maximum number of training pairs per task
        max_test_pairs: Maximum number of test pairs per task
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """Initialize the parser with configuration.

        Args:
            cfg: Hydra configuration object containing parser settings such as
                dataset paths, max dimensions, and other parser-specific parameters

        Raises:
            ValueError: If any of the maximum dimensions are non-positive
            KeyError: If required configuration fields are missing
        """
        # Extract configuration settings
        try:
            # Grid configuration
            max_grid_height = cfg.grid.max_grid_height
            max_grid_width = cfg.grid.max_grid_width
            min_grid_height = cfg.grid.min_grid_height
            min_grid_width = cfg.grid.min_grid_width
            max_colors = cfg.grid.max_colors
            background_color = cfg.grid.background_color

            # Task configuration
            max_train_pairs = cfg.max_train_pairs
            max_test_pairs = cfg.max_test_pairs
        except (AttributeError, KeyError) as e:
            msg = f"Missing required configuration field: {e}"
            raise KeyError(msg) from e

        # Validate grid dimensions
        if max_grid_height <= 0 or max_grid_width <= 0:
            msg = f"Grid dimensions must be positive, got {max_grid_height}x{max_grid_width}"
            raise ValueError(msg)
        if min_grid_height <= 0 or min_grid_width <= 0:
            msg = f"Minimum grid dimensions must be positive, got {min_grid_height}x{min_grid_width}"
            raise ValueError(msg)
        if min_grid_height > max_grid_height or min_grid_width > max_grid_width:
            msg = f"Minimum dimensions ({min_grid_height}x{min_grid_width}) cannot exceed maximum ({max_grid_height}x{max_grid_width})"
            raise ValueError(msg)

        # Validate task configuration
        if max_train_pairs <= 0 or max_test_pairs <= 0:
            msg = f"Max pairs must be positive, got train={max_train_pairs}, test={max_test_pairs}"
            raise ValueError(msg)

        # Validate color configuration
        if max_colors <= 0:
            msg = f"Max colors must be positive, got {max_colors}"
            raise ValueError(msg)
        if background_color < 0 or background_color >= max_colors:
            msg = f"Background color ({background_color}) must be in range [0, {max_colors})"
            raise ValueError(msg)

        self.cfg = cfg
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.min_grid_height = min_grid_height
        self.min_grid_width = min_grid_width
        self.max_colors = max_colors
        self.background_color = background_color
        self.max_train_pairs = max_train_pairs
        self.max_test_pairs = max_test_pairs

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
    def preprocess_task_data(self, raw_task_data: Any, key: chex.PRNGKey) -> JaxArcTask:
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
    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
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

    def validate_grid_dimensions(self, height: int, width: int) -> None:
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

    def validate_color_value(self, color: int) -> None:
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
        from .utils import convert_grid_to_jax
        
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
        from .utils import pad_array_sequence
        
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
        import jax.numpy as jnp
        
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
        from .utils import log_parsing_stats
        
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

    def _process_test_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process test pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (test_input_grids, test_output_grids)

        Raises:
            ValueError: If test data is invalid
        """
        from .utils import convert_grid_to_jax
        import jax.numpy as jnp
        
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

            # For test pairs, output might be provided or missing
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
