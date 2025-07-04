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
        # Extract maximum dimensions from configuration
        try:
            max_grid_height = cfg.max_grid_height
            max_grid_width = cfg.max_grid_width
            max_train_pairs = cfg.max_train_pairs
            max_test_pairs = cfg.max_test_pairs
        except (AttributeError, KeyError) as e:
            msg = f"Missing required configuration field: {e}"
            raise KeyError(msg) from e

        if max_grid_height <= 0 or max_grid_width <= 0:
            msg = f"Grid dimensions must be positive, got {max_grid_height}x{max_grid_width}"
            raise ValueError(msg)
        if max_train_pairs <= 0 or max_test_pairs <= 0:
            msg = f"Max pairs must be positive, got train={max_train_pairs}, test={max_test_pairs}"
            raise ValueError(msg)

        self.cfg = cfg
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
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

    def validate_grid_dimensions(self, height: int, width: int) -> None:
        """Validate that grid dimensions are within the configured maximums.

        Args:
            height: Grid height to validate
            width: Grid width to validate

        Raises:
            ValueError: If dimensions exceed the configured maximums
        """
        if height > self.max_grid_height or width > self.max_grid_width:
            msg = (
                f"Grid dimensions ({height}x{width}) exceed maximum "
                f"({self.max_grid_height}x{self.max_grid_width})"
            )
            raise ValueError(msg)
