from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from loguru import logger

from jaxarc.base.types import ArcTask, Grid, TaskPair


class ArcAgiParser:
    """Parses ARC-AGI task files into ArcTask objects.

    This parser supports ARC-AGI datasets downloaded from Kaggle, including:
    - ARC-AGI-1 (2024 dataset)
    - ARC-AGI-2 (2025 dataset)

    Both datasets follow the same JSON structure format and can be parsed
    with this implementation. It handles challenge files (containing training
    pairs and test inputs) and optional solution files (containing test outputs).
    """

    def _parse_grid_json(self, grid_json: list[list[int]]) -> Grid:
        """Converts a JSON representation of a grid to a Grid object."""
        if (
            not grid_json
            or not isinstance(grid_json, list)
            or not all(isinstance(row, list) for row in grid_json)
            or (grid_json and any(len(row) != len(grid_json[0]) for row in grid_json))
        ):
            msg = "Grid JSON must be a list of lists with consistent row lengths."
            raise ValueError(msg)

        if not all(all(isinstance(cell, int) for cell in row) for row in grid_json):
            msg = "Grid cells must be integers."
            raise ValueError(msg)
        return Grid(array=jnp.array(grid_json, dtype=jnp.int32))

    def _parse_pair_json(
        self, pair_json: dict[str, Any], is_train_pair: bool
    ) -> TaskPair:
        """Converts a JSON representation of an input-output pair to a TaskPair object.

        For training pairs, 'output' is expected in pair_json.
        For test inputs (from challenge files), 'output' is not expected here.
        """
        if "input" not in pair_json:
            msg = "Task pair JSON must contain an 'input' key."
            raise ValueError(msg)

        input_grid = self._parse_grid_json(pair_json["input"])
        output_grid: Grid | None = None

        if is_train_pair:
            if "output" not in pair_json:
                msg = "Training task pair JSON must contain an 'output' key."
                raise ValueError(msg)
            output_grid = self._parse_grid_json(pair_json["output"])
        # If it's a test pair from the challenge file, output_grid remains None.
        # The 'output' key in test pairs from challenge files is ignored,
        # as solutions are loaded separately.
        elif "output" in pair_json and not is_train_pair:
            logger.warning(
                "Test input pair in challenge file contains an 'output' key. "
                "This is unexpected and will be ignored. Test outputs are loaded "
                "from the solutions file if available."
            )

        return TaskPair(input=input_grid, output=output_grid)

    def parse_task_json(self, task_json: dict[str, Any], task_id: str) -> ArcTask:
        """Converts a JSON representation of a single task to an ArcTask object.

        Assumes task_json is the content for a specific task_id.
        Test pairs parsed from here will have output=None.
        """
        train_pairs_json = task_json.get("train", [])
        test_inputs_json = task_json.get("test", [])

        train_pairs = [
            self._parse_pair_json(pair, is_train_pair=True) for pair in train_pairs_json
        ]

        test_pairs_with_inputs_only = [
            self._parse_pair_json(pair, is_train_pair=False)
            for pair in test_inputs_json
        ]

        return ArcTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs_with_inputs_only,
        )

    def parse_task_file(self, file_path: str | Path, task_id: str) -> ArcTask:
        """Parses a single task from a JSON challenge file by its ID.

        The resulting ArcTask's test_pairs will have output=None, as this
        method does not handle solution files.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            msg = f"Task file not found: {file_path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if task_id not in data:
                msg = f"Task ID '{task_id}' not found in {file_path}"
                raise KeyError(msg)
            task_json_content = data[task_id]
            return self.parse_task_json(task_json_content, task_id)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {file_path}: {e}"
            raise ValueError(msg) from e
        except Exception as e:
            logger.error("Error parsing task %s from %s: %s", task_id, file_path, e)
            raise

    def parse_all_tasks_from_file(
        self,
        challenges_file_path: str | Path,
        solutions_file_path: str | Path | None = None,
    ) -> dict[str, ArcTask]:
        """Parses all tasks from a JSON challenge file.

        If a solutions_file_path is provided, it attempts to load test outputs
        and populate them into the corresponding ArcTask objects.
        """
        challenges_file = Path(challenges_file_path)
        if not challenges_file.exists():
            msg = f"Challenge file not found: {challenges_file}"
            raise FileNotFoundError(msg)

        try:
            challenges_data = json.loads(challenges_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in challenge file {challenges_file}: {e}"
            raise ValueError(msg) from e

        tasks: dict[str, ArcTask] = {}
        for task_id, task_json_content in challenges_data.items():
            try:
                tasks[task_id] = self.parse_task_json(task_json_content, task_id)
            except Exception as e:
                logger.error(
                    "Skipping task %s from %s due to parsing error: %s",
                    task_id,
                    challenges_file,
                    e,
                )
                continue

        if solutions_file_path:
            solutions_file = Path(solutions_file_path)
            if solutions_file.exists():
                logger.info("Loading solutions from: %s", solutions_file)
                try:
                    solutions_data = json.loads(
                        solutions_file.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid JSON in solutions file %s: %s. Solutions will not be loaded.",
                        solutions_file,
                        e,
                    )
                    solutions_data = {}

                for task_id, task_obj in tasks.items():
                    if task_id in solutions_data:
                        solution_outputs_json = solutions_data[task_id]

                        if not isinstance(solution_outputs_json, list):
                            logger.warning(
                                "Task %s: Solutions data is not a list. Skipping solutions for this task.",
                                task_id,
                            )
                            continue  # Skip to the next task_id if solutions format is wrong

                        if len(solution_outputs_json) != len(task_obj.test_pairs):
                            logger.warning(
                                "Task %s: Mismatch between number of test inputs (%d) and solution outputs (%d). "
                                "Some test outputs may not be loaded or may be incorrect if lists are misaligned.",
                                task_id,
                                len(task_obj.test_pairs),
                                len(solution_outputs_json),
                            )

                        updated_test_pairs: list[TaskPair] = []
                        for i, test_pair_input_only in enumerate(task_obj.test_pairs):
                            current_output_grid: Grid | None = None  # Default to None
                            if i < len(solution_outputs_json):
                                try:
                                    # Ensure the specific solution output exists and is a list of lists
                                    output_grid_json = solution_outputs_json[i]
                                    if isinstance(output_grid_json, list):
                                        current_output_grid = self._parse_grid_json(
                                            output_grid_json
                                        )
                                    else:
                                        logger.warning(
                                            "Task %s, test pair %d: Expected solution output to be a grid (list of lists), got %s. Output will be None.",
                                            task_id,
                                            i,
                                            type(output_grid_json),
                                        )
                                except IndexError:
                                    # This case is covered by the length check above, but good for safety
                                    logger.warning(
                                        "Task %s, test pair %d: No corresponding solution output found. Output will be None.",
                                        task_id,
                                        i,
                                    )
                                except (
                                    ValueError
                                ) as e_parse:  # Catch errors from _parse_grid_json
                                    logger.warning(
                                        "Task %s, test pair %d: Error parsing solution grid: %s. Output will be None.",
                                        task_id,
                                        i,
                                        e_parse,
                                    )
                                except (
                                    Exception
                                ) as e_sol:  # Catch any other unexpected errors
                                    logger.error(
                                        "Task %s, test pair %d: Unexpected error processing solution: %s. Output will be None.",
                                        task_id,
                                        i,
                                        e_sol,
                                    )
                            else:  # Not enough solutions provided for the number of test inputs
                                logger.warning(
                                    "Task %s, test pair %d: Missing solution output. Output will be None.",
                                    task_id,
                                    i,
                                )

                            updated_test_pairs.append(
                                TaskPair(
                                    input=test_pair_input_only.input,
                                    output=current_output_grid,
                                )
                            )
                        task_obj.test_pairs = updated_test_pairs
                    else:
                        logger.warning(
                            "Task %s: No solutions found in solutions file. Test outputs will be None.",
                            task_id,
                        )
            else:
                logger.warning(
                    "Solutions file specified but not found: %s. Test outputs will be None.",
                    solutions_file,
                )

        return tasks
