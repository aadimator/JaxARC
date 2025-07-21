"""ConceptARC dataset parser implementation.

This module provides a parser for the ConceptARC dataset, which organizes ARC tasks
into 16 concept groups for systematic evaluation of abstraction and generalization
abilities. The parser handles the hierarchical directory structure and provides
concept-based task sampling functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import jax
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from jaxarc.types import JaxArcTask
from jaxarc.utils.task_manager import create_jax_task_index

from .base_parser import ArcDataParserBase


class ConceptArcParser(ArcDataParserBase):
    """Parser for ConceptARC dataset with concept group organization.

    ConceptARC is a benchmark dataset organized around 16 concept groups with 10 tasks each,
    designed to systematically assess abstraction and generalization abilities. Each task
    contains 1-4 demonstration pairs and 3 test inputs per task.

    The dataset follows a hierarchical directory structure:
    corpus/{ConceptGroup}/{TaskName}.json

    Concept groups include:
    - AboveBelow, Center, CleanUp, CompleteShape, Copy, Count
    - ExtendToBoundary, ExtractObjects, FilledNotFilled, HorizontalVertical
    - InsideOutside, MoveToBoundary, Order, SameDifferent, TopBottom2D, TopBottom3D

    This parser provides concept-based task sampling and maintains compatibility
    with the existing JaxArcTask interface.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the ConceptArcParser with configuration.

        Args:
            cfg: Configuration object containing dataset paths and parser settings,
                 including concept group definitions and corpus path
        """
        super().__init__(cfg)

        # Initialize concept group storage
        self._concept_groups: dict[str, list[str]] = {}
        self._task_metadata: dict[str, dict] = {}
        self._all_task_ids: list[str] = []
        self._cached_tasks: dict[str, dict] = {}

        # Load and cache all tasks
        self._load_and_cache_tasks()

    def _load_and_cache_tasks(self) -> None:
        """Load and cache all tasks from the ConceptARC corpus directory structure."""
        try:
            # Get corpus path from configuration
            corpus_config = self.cfg.get("corpus", {})
            corpus_path = corpus_config.get("path")

            if not corpus_path:
                msg = "ConceptARC corpus path not specified in configuration"
                raise ValueError(msg)

            corpus_dir = here(corpus_path)
            if not corpus_dir.exists():
                logger.warning(f"ConceptARC corpus directory not found: {corpus_dir}")
                return

            # Get expected concept groups from configuration
            expected_concept_groups = corpus_config.get("concept_groups", [])

            # Discover concept groups from directory structure
            self._discover_concept_groups(corpus_dir, expected_concept_groups)

            # Load tasks from each concept group
            self._load_tasks_from_concept_groups(corpus_dir)

            logger.info(
                f"Loaded {len(self._all_task_ids)} tasks from {len(self._concept_groups)} "
                f"concept groups in ConceptARC dataset"
            )

        except Exception as e:
            logger.error(f"Error loading and caching ConceptARC tasks: {e}")
            raise

    def _discover_concept_groups(
        self, corpus_dir: Path, expected_groups: list[str]
    ) -> None:
        """Discover concept groups from the corpus directory structure.

        Args:
            corpus_dir: Path to the ConceptARC corpus directory
            expected_groups: List of expected concept group names from configuration
        """
        self._concept_groups = {}

        # Scan for concept group directories
        for item in corpus_dir.iterdir():
            if item.is_dir():
                concept_name = item.name

                # Find JSON task files in this concept directory
                task_files = list(item.glob("*.json"))
                if task_files:
                    task_ids = []
                    for task_file in task_files:
                        # Create unique task ID: concept_group/task_name
                        task_name = task_file.stem
                        task_id = f"{concept_name}/{task_name}"
                        task_ids.append(task_id)

                        # Store task metadata
                        self._task_metadata[task_id] = {
                            "concept_group": concept_name,
                            "task_name": task_name,
                            "file_path": str(task_file),
                        }

                    self._concept_groups[concept_name] = task_ids
                    logger.debug(
                        f"Found concept group '{concept_name}' with {len(task_ids)} tasks"
                    )

        # Validate against expected concept groups
        found_groups = set(self._concept_groups.keys())
        expected_groups_set = set(expected_groups) if expected_groups else set()

        if expected_groups_set:
            missing_groups = expected_groups_set - found_groups
            extra_groups = found_groups - expected_groups_set

            if missing_groups:
                logger.warning(
                    f"Missing expected concept groups: {sorted(missing_groups)}"
                )
            if extra_groups:
                logger.info(f"Found additional concept groups: {sorted(extra_groups)}")

        if not self._concept_groups:
            msg = f"No concept groups found in {corpus_dir}"
            raise ValueError(msg)

    def _load_tasks_from_concept_groups(self, corpus_dir: Path) -> None:
        """Load task data from all concept groups.

        Args:
            corpus_dir: Path to the ConceptARC corpus directory
        """
        self._cached_tasks = {}
        self._all_task_ids = []

        for concept_group, task_ids in self._concept_groups.items():
            for task_id in task_ids:
                metadata = self._task_metadata[task_id]
                task_file_path = Path(metadata["file_path"])

                try:
                    # Load task data from JSON file
                    with task_file_path.open("r", encoding="utf-8") as f:
                        task_data = json.load(f)

                    # Cache the task data
                    self._cached_tasks[task_id] = task_data
                    self._all_task_ids.append(task_id)

                    # Update metadata with task statistics
                    train_pairs = len(task_data.get("train", []))
                    test_pairs = len(task_data.get("test", []))
                    self._task_metadata[task_id].update(
                        {
                            "num_demonstrations": train_pairs,
                            "num_test_inputs": test_pairs,
                        }
                    )

                except Exception as e:
                    logger.error(
                        f"Error loading task {task_id} from {task_file_path}: {e}"
                    )
                    continue

        logger.info(f"Successfully cached {len(self._cached_tasks)} ConceptARC tasks")

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
            task_id = "unknown_task"

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

    def _process_training_pairs(self, task_content: dict) -> tuple[list, list]:
        """Process training pairs and convert them to JAX arrays.

        Args:
            task_content: Task content dictionary

        Returns:
            Tuple of (train_input_grids, train_output_grids)

        Raises:
            ValueError: If training data is invalid
        """
        # Call parent implementation but customize error message for ConceptARC
        try:
            return super()._process_training_pairs(task_content)
        except ValueError as e:
            if "Task must have at least one training pair" in str(e):
                msg = "ConceptARC task must have at least one training pair"
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
        # Call parent implementation but customize error message for ConceptARC
        try:
            return super()._process_test_pairs(task_content)
        except ValueError as e:
            if "Task must have at least one test pair" in str(e):
                msg = "ConceptARC task must have at least one test pair"
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
        if not self._all_task_ids:
            msg = "No tasks available in ConceptARC dataset"
            raise RuntimeError(msg)

        # Randomly select a task ID
        task_index = jax.random.randint(key, (), 0, len(self._all_task_ids))
        task_id = self._all_task_ids[int(task_index)]

        # Get the cached task data
        task_data = self._cached_tasks[task_id]

        # Preprocess and return
        return self.preprocess_task_data((task_id, task_data), key)

    def get_random_task_from_concept(
        self, concept: str, key: chex.PRNGKey
    ) -> JaxArcTask:
        """Get random task from specific concept group.

        Args:
            concept: Name of the concept group
            key: JAX PRNG key for random selection

        Returns:
            JaxArcTask: A randomly selected task from the specified concept group

        Raises:
            ValueError: If the concept group is not found
            RuntimeError: If no tasks are available in the concept group
        """
        if concept not in self._concept_groups:
            available_concepts = sorted(self._concept_groups.keys())
            msg = (
                f"Concept group '{concept}' not found. Available: {available_concepts}"
            )
            raise ValueError(msg)

        concept_task_ids = self._concept_groups[concept]
        if not concept_task_ids:
            msg = f"No tasks available in concept group '{concept}'"
            raise RuntimeError(msg)

        # Randomly select a task from this concept group
        task_index = jax.random.randint(key, (), 0, len(concept_task_ids))
        task_id = concept_task_ids[int(task_index)]

        # Get the cached task data
        task_data = self._cached_tasks[task_id]

        # Preprocess and return
        return self.preprocess_task_data((task_id, task_data), key)

    def get_concept_groups(self) -> list[str]:
        """Get list of available concept groups.

        Returns:
            List of concept group names available in the dataset
        """
        return sorted(self._concept_groups.keys())

    def get_tasks_in_concept(self, concept: str) -> list[str]:
        """Get list of task IDs in a specific concept group.

        Args:
            concept: Name of the concept group

        Returns:
            List of task IDs in the specified concept group

        Raises:
            ValueError: If the concept group is not found
        """
        if concept not in self._concept_groups:
            available_concepts = sorted(self._concept_groups.keys())
            msg = (
                f"Concept group '{concept}' not found. Available: {available_concepts}"
            )
            raise ValueError(msg)

        return self._concept_groups[concept].copy()

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Get a specific task by its ID.

        Args:
            task_id: ID of the task to retrieve (format: concept_group/task_name)

        Returns:
            JaxArcTask: The preprocessed task data

        Raises:
            ValueError: If the task ID is not found
        """
        if task_id not in self._all_task_ids:
            msg = f"Task ID '{task_id}' not found in ConceptARC dataset"
            raise ValueError(msg)

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
        return self._all_task_ids.copy()

    def get_task_metadata(self, task_id: str) -> dict:
        """Get metadata for a specific task.

        Args:
            task_id: ID of the task

        Returns:
            Dictionary containing task metadata

        Raises:
            ValueError: If the task ID is not found
        """
        if task_id not in self._task_metadata:
            msg = f"Task ID '{task_id}' not found in ConceptARC dataset"
            raise ValueError(msg)

        return self._task_metadata[task_id].copy()

    def get_dataset_statistics(self) -> dict:
        """Get statistics about the ConceptARC dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_tasks": len(self._all_task_ids),
            "total_concept_groups": len(self._concept_groups),
            "concept_groups": {},
        }

        for concept, task_ids in self._concept_groups.items():
            concept_stats = {
                "num_tasks": len(task_ids),
                "tasks": task_ids.copy(),
            }

            # Calculate demonstration and test input statistics for this concept
            demonstrations = []
            test_inputs = []
            for task_id in task_ids:
                metadata = self._task_metadata.get(task_id, {})
                if "num_demonstrations" in metadata:
                    demonstrations.append(metadata["num_demonstrations"])
                if "num_test_inputs" in metadata:
                    test_inputs.append(metadata["num_test_inputs"])

            if demonstrations:
                concept_stats["avg_demonstrations"] = sum(demonstrations) / len(
                    demonstrations
                )
                concept_stats["min_demonstrations"] = min(demonstrations)
                concept_stats["max_demonstrations"] = max(demonstrations)

            if test_inputs:
                concept_stats["avg_test_inputs"] = sum(test_inputs) / len(test_inputs)
                concept_stats["min_test_inputs"] = min(test_inputs)
                concept_stats["max_test_inputs"] = max(test_inputs)

            stats["concept_groups"][concept] = concept_stats

        return stats
