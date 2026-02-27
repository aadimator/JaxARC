"""MiniARC dataset parser implementation.

This module provides a parser for the MiniARC dataset, which is a 5x5 compact version
of ARC with 400 training and 400 evaluation tasks. The parser is optimized for smaller
grid dimensions and faster processing, making it ideal for rapid prototyping and testing.
"""

from __future__ import annotations

import json

from loguru import logger

from jaxarc.configs import DatasetConfig

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

        # Scan available tasks (lazy loading)
        self._scan_available_tasks()

    @property
    def _dataset_name(self) -> str:
        """Dataset name for error messages."""
        return "MiniARC dataset"

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
