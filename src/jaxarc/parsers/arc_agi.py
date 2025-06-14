from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from jaxarc.base import ArcDataParserBase
from jaxarc.types import ParsedTaskData

from .utils import convert_grid_to_jax, log_parsing_stats, pad_array_sequence


class ArcAgiParser(ArcDataParserBase):
    """Parses ARC-AGI task files into ParsedTaskData objects.

    This parser supports ARC-AGI datasets downloaded from Kaggle, including:
    - ARC-AGI-1 (2024 dataset)
    - ARC-AGI-2 (2025 dataset)

    Both datasets follow the same JSON structure format and can be parsed
    with this implementation. It handles challenge files (containing training
    pairs and test inputs) and optional solution files (containing test outputs).

    The parser outputs JAX-compatible ParsedTaskData structures with padded
    arrays and boolean masks for efficient processing in the MARL environment.
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
        """Load and cache all tasks from challenges and solutions files."""
        try:
            # Load from default split (usually 'training')
            default_split = self.cfg.get("default_split", "training")
            split_config = self.cfg.get(default_split, {})

            challenges_path = split_config.get("challenges")
            solutions_path = split_config.get("solutions")

            # Load challenges data
            challenges_data = {}
            if challenges_path:
                challenges_file = Path(challenges_path)
                if challenges_file.exists():
                    with challenges_file.open("r", encoding="utf-8") as f:
                        challenges_data = json.load(f)
                    logger.info(
                        f"Loaded {len(challenges_data)} tasks from {challenges_file}"
                    )
                else:
                    logger.warning(f"Challenges file not found: {challenges_file}")
                    return

            # Load solutions data
            solutions_data = {}
            if solutions_path:
                solutions_file = Path(solutions_path)
                if solutions_file.exists():
                    with solutions_file.open("r", encoding="utf-8") as f:
                        solutions_data = json.load(f)
                    logger.info(f"Loaded solutions from {solutions_file}")
                else:
                    logger.warning(f"Solutions file not found: {solutions_file}")

            # Merge challenges and solutions data
            self._cached_tasks = {}
            for task_id, task_data in challenges_data.items():
                # Start with the challenge data
                merged_task = task_data.copy()

                # Add solutions if available
                if solutions_data.get(task_id):
                    # Solutions are stored as a list of outputs for test pairs
                    test_outputs = solutions_data[task_id]

                    # Merge solutions into test pairs
                    if "test" in merged_task:
                        for i, test_pair in enumerate(merged_task["test"]):
                            if i < len(test_outputs):
                                test_pair["output"] = test_outputs[i]

                self._cached_tasks[task_id] = merged_task

            self._task_ids = list(self._cached_tasks.keys())
            logger.info(f"Cached {len(self._cached_tasks)} tasks in memory")

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
    ) -> ParsedTaskData:
        """Convert raw task data into ParsedTaskData structure.

        Args:
            raw_task_data: Raw task data dictionary
            key: JAX PRNG key (unused in this deterministic preprocessing)

        Returns:
            ParsedTaskData: JAX-compatible task data with padded arrays

        Raises:
            ValueError: If the task data format is invalid
        """
        if not isinstance(raw_task_data, dict):
            msg = f"Expected dict, got {type(raw_task_data)}"
            raise ValueError(msg)

        # Extract task ID and content
        if len(raw_task_data) != 1:
            msg = f"Expected single task, got {len(raw_task_data)} tasks"
            raise ValueError(msg)

        task_id, task_content = next(iter(raw_task_data.items()))

        # Process training pairs
        train_pairs_data = task_content.get("train", [])
        test_pairs_data = task_content.get("test", [])

        if not train_pairs_data:
            msg = "Task must have at least one training pair"
            raise ValueError(msg)

        if not test_pairs_data:
            msg = "Task must have at least one test pair"
            raise ValueError(msg)

        # Convert training pairs to JAX arrays
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

            train_input_grids.append(input_grid)
            train_output_grids.append(output_grid)

        # Convert test pairs to JAX arrays
        test_input_grids = []
        test_output_grids = []

        for i, pair in enumerate(test_pairs_data):
            if "input" not in pair:
                msg = f"Test pair {i} missing input"
                raise ValueError(msg)

            input_grid = convert_grid_to_jax(pair["input"])
            self.validate_grid_dimensions(*input_grid.shape)
            test_input_grids.append(input_grid)

            # For test pairs, output might be None (challenge files)
            # or provided (solution files). We'll handle both cases.
            if "output" in pair and pair["output"] is not None:
                output_grid = convert_grid_to_jax(pair["output"])
                self.validate_grid_dimensions(*output_grid.shape)
                test_output_grids.append(output_grid)
            else:
                # Create dummy output grid (will be masked as invalid)
                dummy_output = jnp.zeros_like(input_grid)
                test_output_grids.append(dummy_output)

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

        # Log parsing statistics
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
            len(train_pairs_data), len(test_pairs_data), max_dims, task_id
        )

        # Create ParsedTaskData structure
        return ParsedTaskData(
            input_grids_examples=padded_train_inputs,
            input_masks_examples=train_input_masks,
            output_grids_examples=padded_train_outputs,
            output_masks_examples=train_output_masks,
            num_train_pairs=len(train_pairs_data),
            test_input_grids=padded_test_inputs,
            test_input_masks=test_input_masks,
            true_test_output_grids=padded_test_outputs,
            true_test_output_masks=test_output_masks,
            num_test_pairs=len(test_pairs_data),
            task_id=task_id,
        )

    def get_random_task(self, key: chex.PRNGKey) -> ParsedTaskData:
        """Get a random task from the dataset.

        Args:
            key: JAX PRNG key for random selection

        Returns:
            ParsedTaskData: A randomly selected and preprocessed task

        Raises:
            RuntimeError: If no tasks are available
        """
        if not self._task_ids:
            msg = "No tasks available in dataset"
            raise RuntimeError(msg)

        # Randomly select a task ID
        task_index = jax.random.randint(key, (), 0, len(self._task_ids))
        task_id = self._task_ids[int(task_index)]

        # Get the cached task data
        task_data = {task_id: self._cached_tasks[task_id]}

        # Preprocess and return
        return self.preprocess_task_data(task_data, key)

    def get_task_by_id(self, task_id: str) -> ParsedTaskData:
        """Get a specific task by its ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            ParsedTaskData: The preprocessed task data

        Raises:
            ValueError: If the task ID is not found
        """
        if task_id not in self._task_ids:
            msg = f"Task ID '{task_id}' not found in dataset"
            raise ValueError(msg)

        # Get the cached task data
        task_data = {task_id: self._cached_tasks[task_id]}

        # Create a dummy key for preprocessing (deterministic)
        key = jax.random.PRNGKey(0)

        # Preprocess and return
        return self.preprocess_task_data(task_data, key)

    def get_available_task_ids(self) -> list[str]:
        """Get list of all available task IDs.

        Returns:
            List of task IDs available in the dataset
        """
        return self._task_ids.copy()
