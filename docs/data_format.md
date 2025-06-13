# Data Description

The format is different from the previous competition, so please read this
information carefully, and refer to supplementary documentation as needed.

When looking at a task, a "test-taker" has access to inputs and outputs of the
demonstration pairs (train pairs), plus the input(s) of the test pair(s). The
goal is to construct the output grid(s) corresponding to the test input grid(s),
using **2 trials** for each test input. "Constructing the output grid" involves
picking the height and width of the output grid, then filling each cell in the
grid with a symbol (integer between 0 and 9, which are visualized as colors).
Only **exact** solutions (all cells match the expected answer) can be said to be
correct.

Any additional information, as well as an interactive app to explore the
objective of this competition is found at the
[ARCPrize.org](http://arcprize.org/play). **\*It is highly recommended that you
explore the interactive app, as the best way to understand the objective of the
competition**.\*

## Task files

The information is stored in two files:

- **arc-agi_training-challenges.json**: contains input/output pairs that
  demonstrate reasoning pattern to be applied to the "test" input for each task.
  This file and the corresponding **solutions** file can be used as training for
  your models.
- **arc-agi_training-solutions.json**: contains the corresponding task "test"
  outputs (ground truth).
- **arc-agi_evaluation-challenges.json**: contains input/output pairs that
  demonstrate reasoning pattern to be applied to the "test" input for each task.
  This file and the corresponding **solutions** file can be used as validation
  data for your models.
- **arc-agi_evaluation-solutions.json**: contains the corresponding task "test"
  outputs (ground truth).
- **arc-agi_test-challenges.json**: this file contains the tasks that will be
  used for the leaderboard evaluation, and contains "train" input/output pairs
  as well as the "test" input for each task. Your task is to predict the "test"
  output. **Note:** The file shown on this page is a placeholder using tasks
  from **arc-agi_evaluation-challenges.json**. When you submit your notebook to
  be rerun, this file is swapped with the actual test challenges.
- **sample_submission.json**: a submission file in the correct format

Each task contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs"
  (typically 3 pairs).
- `"test"`: test input - your model should predict the output.

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9
(inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

The data on this page should be used to develop and evaluate your models. When
notebooks are submitted for rerun, they are scored using 240 unseen tasks found
in the rerun file named **arc-agi_test_challenges.json**. The rerun tasks will
contain `train` pairs of inputs and outputs as well as the tasks `test` input.
Your algorithm must predict the `test` output. The majority of the 240 tasks
used for leaderboard score only have one `test` input that will require a
corresponding output prediction, although for a small number of tasks, you will
be asked to make predictions for two `test` inputs.

## Previous Implementation of Data Loader

Though we'll have to modify it so it works with our JAX based code.

```python
"""
Data loading and task management for ARC-AGI dataset.

This module provides functionality to load, process, and manage ARC tasks,
integrating with Hydra for configuration management.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from rich import print as rprint
from rich.table import Table
from rich.text import Text

from arc25.utils import idx2chr, load_json_file


def fmt_grid(
    grid: np.ndarray, colour: bool = True, spaces: bool | str = True
) -> str | Text:
    """Format grid for display with optional coloring."""
    # Ensure grid is numpy array
    grid_array = np.array(grid) if not isinstance(grid, np.ndarray) else grid

    grid_str_parts = []
    if not colour:
        for row in grid_array:
            if spaces == "gpt":
                grid_str_parts.append("".join([" " + str(x) for x in row]))
            elif spaces:
                grid_str_parts.append(" ".join([str(x) for x in row]))
            else:
                grid_str_parts.append("".join([str(x) for x in row]))
        return "\n".join(grid_str_parts)

    # Define color map for rich Text
    # Using integers as keys directly as grid contains integers
    color_map: dict[int, tuple[str, str]] = {}
    if spaces == "gpt":
        color_map = {i: (str(i) + " ", f"color({i})") for i in range(10)}
        color_map.update({-1: (". ", "grey")})  # Example for potential empty/padding
    elif spaces:
        color_map = {i: (str(i) + " ", f"color({i})") for i in range(10)}
        color_map.update({-1: (". ", "grey")})
    else:
        color_map = {i: (str(i), f"color({i})") for i in range(10)}
        color_map.update({-1: (".", "grey")})

    text_parts: list[str | tuple[str, str]] = []
    for row in grid_array:
        for digit in row:
            text_parts.append(
                color_map.get(int(digit), (str(digit), ""))
            )  # Handle potential non-int digits gracefully
        text_parts.append(("\n", ""))  # Newline style for Text.assemble

    # Remove the last newline
    if text_parts:
        text_parts.pop()

    return Text.assemble(*text_parts)


class Task:
    """Represents a single ARC task with training and test examples."""

    def __init__(
        self,
        task_id: str,
        train_pairs: list[dict[str, Any]],
        test_pairs: list[dict[str, Any]],
        dataset_name: str | None = None,
        task_file_path: str | Path | None = None,
    ):
        """
        Initializes an ARC Task.

        Args:
            task_id: The unique identifier for the task.
            train_pairs: A list of training pairs, where each pair is a dict
                         with "input" and "output" keys, and values are grids.
            test_pairs: A list of test pairs. For challenges, "output" might be missing.
                        For solutions, "output" will be present.
            dataset_name: Optional name of the dataset this task belongs to.
            task_file_path: Optional path to the source JSON file for this task.
        """
        self.id = task_id
        self.dataset_name = dataset_name
        self.task_file_path = Path(task_file_path) if task_file_path else None

        self.train: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in train_pairs:
            self.train.append(
                (
                    np.array(pair["input"], dtype=int),
                    np.array(pair["output"], dtype=int),
                )
            )

        self.test: list[tuple[np.ndarray, np.ndarray | None]] = []
        for pair in test_pairs:
            input_grid = np.array(pair["input"], dtype=int)
            output_grid = (
                np.array(pair["output"], dtype=int)
                if "output" in pair and pair["output"] is not None
                else None
            )
            self.test.append((input_grid, output_grid))

    def __lt__(self, other: Task) -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return (
            f"<Task id={self.id} dataset={self.dataset_name} | "
            f"{len(self.train)} train | {len(self.test)} test>"
        )

    @classmethod
    def from_json_data(
        cls,
        task_id: str,
        task_data: dict[str, Any],
        dataset_name: str | None = None,
        task_file_path: str | Path | None = None,
    ) -> Task:
        """Loads a task from a dictionary (parsed JSON data)."""
        train_pairs = task_data.get("train", [])
        test_pairs = task_data.get("test", [])
        return cls(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            dataset_name=dataset_name,
            task_file_path=task_file_path,
        )

    def show(self, answer: bool = False, rich_print: bool = True) -> Table | str:
        """
        Display the task in a rich table format or as a string.

        Args:
            answer: Whether to show the test output (if available).
            rich_print: If True, prints the table to console using rich.
                        If False, returns the Table object or string representation.

        Returns:
            A rich Table object if rich_print is False and rich is available,
            otherwise a string representation.
        """
        table = Table(title=repr(self), show_lines=True, padding=(0, 1))

        header_styles = {
            "train_in": "cyan",
            "train_out": "magenta",
            "test_in": "bold green",
            "test_out": "bold red",
        }

        num_train = len(self.train)
        num_test = len(self.test)

        # Add columns for training pairs
        for i in range(num_train):
            in_grid, out_grid = self.train[i]
            table.add_column(
                f"{idx2chr(i)}-In ({in_grid.shape[0]}x{in_grid.shape[1]})",
                justify="center",
                no_wrap=True,
                header_style=header_styles["train_in"],
            )
            table.add_column(
                f"{idx2chr(i)}-Out ({out_grid.shape[0]}x{out_grid.shape[1]})",
                justify="center",
                no_wrap=True,
                header_style=header_styles["train_out"],
            )

        # Add columns for test pairs
        for i in range(num_test):
            in_grid, out_grid = self.test[i]
            table.add_column(
                f"T{idx2chr(i)}-In ({in_grid.shape[0]}x{in_grid.shape[1]})",
                justify="center",
                no_wrap=True,
                header_style=header_styles["test_in"],
            )
            if answer and out_grid is not None:
                table.add_column(
                    f"T{idx2chr(i)}-Out ({out_grid.shape[0]}x{out_grid.shape[1]})",
                    justify="center",
                    no_wrap=True,
                    header_style=header_styles["test_out"],
                )
            elif (
                answer and out_grid is None
            ):  # Case where answer is requested but not available
                table.add_column(
                    f"T{idx2chr(i)}-Out (N/A)",
                    justify="center",
                    no_wrap=True,
                    header_style=header_styles["test_out"],
                )

        row_data: list[str | Text] = []
        # Training data
        for in_grid, out_grid in self.train:
            row_data.append(fmt_grid(in_grid))
            row_data.append(fmt_grid(out_grid))

        # Test data
        for in_grid, out_grid in self.test:
            row_data.append(fmt_grid(in_grid))
            if answer and out_grid is not None:
                row_data.append(fmt_grid(out_grid))
            elif answer and out_grid is None:
                row_data.append(Text("N/A", style="italic grey"))

        # Pad row_data if necessary due to some test outputs not being shown
        max_cols = len(table.columns)
        if len(row_data) < max_cols:
            row_data.extend([Text("") for _ in range(max_cols - len(row_data))])

        table.add_row(*row_data)

        if rich_print:
            rprint(table)
            return (
                table  # Technically returns None if rprint is used, but for consistency
            )
        return table

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary format, compatible with JSON."""
        return {
            "task_id": self.id,
            "dataset_name": self.dataset_name,
            "task_file_path": str(self.task_file_path) if self.task_file_path else None,
            "train": [
                {"input": input_grid.tolist(), "output": output_grid.tolist()}
                for input_grid, output_grid in self.train
            ],
            "test": [
                {
                    "input": input_grid.tolist(),
                    "output": output_grid.tolist() if output_grid is not None else None,
                }
                for input_grid, output_grid in self.test
            ],
        }

    def score_prediction(
        self, predicted_outputs: list[Union[np.ndarray, list[list[int]]]]
    ) -> list[bool]:
        """
        Scores a list of predicted output grids against the test examples.
        Each task can have multiple test inputs. This method expects a prediction
        for each test input.

        Args:
            predicted_outputs: A list of predicted grids. Each grid can be a
                               numpy array or a list of lists. The order must
                               correspond to the order of test inputs in self.test.

        Returns:
            A list of booleans, where each boolean indicates if the corresponding
            prediction was correct.
        """
        results = []
        if len(predicted_outputs) != len(self.test):
            # Consider raising an error or logging a warning
            # For now, score based on the minimum length
            # This handles cases where not all test outputs are predicted
            # Or more are predicted than test inputs (though this is less likely)
            min_len = min(len(predicted_outputs), len(self.test))
            print(
                f"Warning: Number of predictions ({len(predicted_outputs)}) does not match number of test cases ({len(self.test)}) for task {self.id}. Scoring based on {min_len} pairs."
            )

        for i in range(min(len(predicted_outputs), len(self.test))):
            _true_input_grid, true_output_grid = self.test[
                i
            ]  # _true_input_grid to mark as unused

            if true_output_grid is None:
                results.append(False)
                continue

            predicted_grid_raw = predicted_outputs[i]
            if not isinstance(predicted_grid_raw, np.ndarray):
                predicted_grid = np.array(predicted_grid_raw, dtype=int)
            else:
                predicted_grid = predicted_grid_raw.astype(int)

            if predicted_grid.shape != true_output_grid.shape:
                results.append(False)
            else:
                results.append(np.array_equal(predicted_grid, true_output_grid))
        return results


class TaskSet:
    """A collection of tasks with indexing and scoring capabilities."""

    def __init__(self, tasks: list[Task]):
        self.tasks = sorted(tasks)
        self.task_dict: dict[str, Task] = {task.id: task for task in self.tasks}

    def __getitem__(self, key: Union[int, str, slice]) -> Union[Task, TaskSet]:
        if isinstance(key, slice):
            return TaskSet(self.tasks[key])
        if isinstance(key, str):
            task = self.task_dict.get(key)
            if task is None:
                msg = f"Task with id '{key}' not found in TaskSet."
                raise KeyError(msg)
            return task
        if isinstance(key, int):
            if 0 <= key < len(self.tasks):
                return self.tasks[key]
            msg = f"Index {key} out of bounds for TaskSet of length {len(self.tasks)}."
            raise IndexError(msg)
        msg = f"TaskSet indices must be integers, slices, or task_id strings, not {type(key).__name__}"
        raise TypeError(msg)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self) -> str:
        return f"<TaskSet: {len(self.tasks)} tasks>"

    def score_submission_file(
        self,
        submission_file_path: Union[str, Path],
        top_n_predictions: int = 3,
        return_correct_ids: bool = False,
    ) -> Union[int, tuple[int, set[str]]]:
        """
        Scores a submission file in the ARC Kaggle CSV format.
        The submission format is: output_id (taskid_testidx), output (pipe-separated grid rows)

        Args:
            submission_file_path: Path to the CSV submission file.
            top_n_predictions: How many predictions per test case in the submission to consider.
            return_correct_ids: If True, returns a tuple of (total_score, set_of_correct_task_ids).

        Returns:
            The total number of correctly solved tasks, or a tuple if return_correct_ids is True.
        """
        predictions_by_task_output_id: defaultdict[str, list[np.ndarray]] = defaultdict(
            list
        )

        with Path(submission_file_path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                output_id = row["output_id"]
                output_str_predictions = (
                    row["output"].strip().split(" ")[:top_n_predictions]
                )

                parsed_attempt_grids: list[np.ndarray] = []
                for pred_str in output_str_predictions:
                    if not pred_str:
                        continue
                    try:
                        grid_rows_str = pred_str.strip("|").split("|")
                        grid_parsed = [
                            [int(digit) for digit in r_str]
                            for r_str in grid_rows_str
                            if r_str
                        ]
                        if grid_parsed and all(
                            len(r) == len(grid_parsed[0]) for r in grid_parsed
                        ):
                            parsed_attempt_grids.append(
                                np.array(grid_parsed, dtype=int)
                            )
                        else:
                            pass
                    except ValueError:
                        pass

                if parsed_attempt_grids:
                    predictions_by_task_output_id[output_id] = parsed_attempt_grids

        total_tasks_correct = 0
        correctly_solved_task_ids: set[str] = set()

        for task in self.tasks:
            task_fully_correct = True
            if not task.test:
                task_fully_correct = False
                continue

            for test_idx, (_test_input_grid, true_output_grid) in enumerate(task.test):
                if true_output_grid is None:
                    task_fully_correct = False
                    break

                current_output_id = f"{task.id}_{test_idx}"
                submitted_predictions_for_this_test_case = (
                    predictions_by_task_output_id.get(current_output_id)
                )

                if not submitted_predictions_for_this_test_case:
                    task_fully_correct = False
                    break

                attempted_grids_for_this_test_case = (
                    submitted_predictions_for_this_test_case
                )

                is_this_test_case_correct = False
                for predicted_grid in attempted_grids_for_this_test_case:
                    if (
                        predicted_grid.shape == true_output_grid.shape
                        and np.array_equal(predicted_grid, true_output_grid)
                    ):
                        is_this_test_case_correct = True
                        break

                if not is_this_test_case_correct:
                    task_fully_correct = False
                    break

            if task_fully_correct:
                total_tasks_correct += 1
                correctly_solved_task_ids.add(task.id)

        if return_correct_ids:
            return total_tasks_correct, correctly_solved_task_ids
        return total_tasks_correct


class ARCDataLoader:
    """
    Data loader for ARC datasets, configured via Hydra or direct instantiation.
    Handles loading challenges and their corresponding solutions.
    """

    def __init__(
        self,
        challenges_file_path: Union[str, Path],
        solutions_file_path: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = "arc_dataset",
    ):
        """
        Initializes the ARCDataLoader.

        Args:
            challenges_file_path: Path to the JSON file containing task challenges.
            solutions_file_path: Optional path to the JSON file containing task solutions.
                                 If None, tasks will be loaded without test outputs.
            dataset_name: A name for this dataset instance.
        """
        self.challenges_path = Path(challenges_file_path)
        self.solutions_path = Path(solutions_file_path) if solutions_file_path else None
        self.dataset_name = dataset_name

        if not self.challenges_path.is_file():
            msg = f"Challenges file not found or is not a file: {self.challenges_path}"
            raise FileNotFoundError(msg)
        if self.solutions_path and not self.solutions_path.is_file():
            msg = f"Solutions file not found or is not a file: {self.solutions_path}"
            raise FileNotFoundError(msg)

        self._challenges_data: dict[str, Any] | None = None
        self._solutions_data: dict[str, Any] | None = None
        self._task_ids: list[str] | None = None

    def _load_challenges(self) -> dict[str, Any]:
        if self._challenges_data is None:
            self._challenges_data = load_json_file(self.challenges_path)
            if not isinstance(self._challenges_data, dict):
                msg = f"Challenges file {self.challenges_path} did not load as a dictionary."
                raise ValueError(msg)
        return self._challenges_data

    def _load_solutions(self) -> dict[str, Any] | None:
        if self._solutions_data is None and self.solutions_path:
            self._solutions_data = load_json_file(self.solutions_path)
            if not isinstance(self._solutions_data, dict):
                msg = f"Solutions file {self.solutions_path} did not load as a dictionary."
                raise ValueError(msg)
        return self._solutions_data

    def get_task_ids(self) -> list[str]:
        """Returns a sorted list of all task IDs in the challenges file."""
        if self._task_ids is None:
            challenge_data = self._load_challenges()
            self._task_ids = sorted(challenge_data.keys())
        return self._task_ids

    def load_task(self, task_id: str) -> Task:
        """
        Loads a specific ARC task by its ID.

        Args:
            task_id: The unique identifier for the task.

        Returns:
            An ARC Task object.

        Raises:
            KeyError: If the task_id is not found in the challenges file.
        """
        challenges = self._load_challenges()
        if task_id not in challenges:
            msg = f"Task ID '{task_id}' not found in challenges file: {self.challenges_path}"
            raise KeyError(msg)

        task_challenge_data = challenges[task_id]

        raw_test_pairs = task_challenge_data.get("test", [])
        final_test_pairs = []

        solutions = None
        if self.solutions_path:
            solutions = self._load_solutions()

        if solutions and task_id in solutions:
            solution_outputs_for_task = solutions[task_id]

            for i, test_challenge_pair in enumerate(raw_test_pairs):
                current_input = test_challenge_pair["input"]
                current_output = None
                if i < len(solution_outputs_for_task):
                    current_output = solution_outputs_for_task[i]
                final_test_pairs.append(
                    {"input": current_input, "output": current_output}
                )
        else:
            for test_challenge_pair in raw_test_pairs:
                final_test_pairs.append(
                    {
                        "input": test_challenge_pair["input"],
                        "output": test_challenge_pair.get("output"),
                    }
                )

        return Task.from_json_data(
            task_id=task_id,
            task_data={
                "train": task_challenge_data.get("train", []),
                "test": final_test_pairs,
            },
            dataset_name=self.dataset_name,
            task_file_path=self.challenges_path,
        )

    def load_all_tasks(self) -> TaskSet:
        """Loads all tasks from the challenges and solutions files into a TaskSet."""
        task_ids = self.get_task_ids()
        all_tasks = [self.load_task(task_id) for task_id in task_ids]
        return TaskSet(all_tasks)

    def get_task_count(self) -> int:
        """Returns the total number of tasks."""
        return len(self.get_task_ids())

    def sample_tasks(self, n: int = 5, random_seed: int | None = None) -> TaskSet:
        """
        Returns a TaskSet containing a random sample of n tasks.
        If n is larger than total tasks, returns all tasks.
        """
        if random_seed is not None:
            random.seed(random_seed)

        task_ids = self.get_task_ids()
        if n >= len(task_ids):
            return self.load_all_tasks()

        sampled_ids = random.sample(task_ids, n)
        sampled_tasks = [self.load_task(task_id) for task_id in sampled_ids]
        return TaskSet(sampled_tasks)


# Example of how to use with Hydra (in a script, not this library file)
#
# import hydra
# from omegaconf import DictConfig
# from arc25.data import ARCDataLoader
#
# @hydra.main(config_path="../conf", config_name="config")
# def my_app(cfg: DictConfig) -> None:
#     print("Hydra Config:")
#     print(cfg.pretty())
#
#     training_data_cfg = cfg.data_sets.arc_prize_2025_training
#
#     data_loader = ARCDataLoader(
#         challenges_file_path=training_data_cfg.challenges_file,
#         solutions_file_path=training_data_cfg.solutions_file,
#         dataset_name=training_data_cfg.dataset_name
#     )
#
#     print(f"Initialized ARCDataLoader for: {data_loader.dataset_name}")
#     print(f"Found {data_loader.get_task_count()} tasks.")
#
#     if data_loader.get_task_count() > 0:
#         task_ids = data_loader.get_task_ids()
#         print(f"First 5 task IDs: {task_ids[:5]}")
#
#         first_task = data_loader.load_task(task_ids[0])
#         print(f"Loaded first task: {first_task.task_id}")
#         first_task.show()
#
# if __name__ == '__main__':
#     print("Running data.py directly for testing (not using Hydra context here).")
#
#     project_root = Path(__file__).resolve().parent.parent.parent
#
#     test_challenges_path = project_root / "data" / "raw" / "arc-prize-2025" / "arc-agi_training_challenges.json"
#     test_solutions_path = project_root / "data" / "raw" / "arc-prize-2025" / "arc-agi_training_solutions.json"
#     sample_submission_path = project_root / "data" / "raw" / "arc-prize-2025" / "sample_submission.json"
#
#     if not test_challenges_path.exists():
#         print(f"Test challenges file not found at: {test_challenges_path}")
#         print("Skipping direct execution tests.")
#     else:
#         print(f"Using challenges: {test_challenges_path}")
#         print(f"Using solutions: {test_solutions_path}")
#
#         loader = ARCDataLoader(
#             challenges_file_path=test_challenges_path,
#             solutions_file_path=test_solutions_path,
#             dataset_name="direct_test_training_set"
#         )
#         print(f"Loader initialized for {loader.dataset_name}, found {loader.get_task_count()} tasks.")
#
#         if loader.get_task_count() > 0:
#             task_ids = loader.get_task_ids()
#             task1 = loader.load_task(task_ids[0])
#             task1.show(answer=True)
#
#             task2_id = task_ids[1] if len(task_ids) > 1 else task_ids[0]
#             task2 = loader.load_task(task2_id)
#
#             if task2.test and task2.test[0][1] is not None:
#                 correct_pred = [np.copy(task2.test[0][1])]
#                 incorrect_pred_shape = [np.array([[1,2],[3,4]]) if task2.test[0][1].shape != (2,2) else np.array([[0]])]
#                 incorrect_pred_values = [np.zeros_like(task2.test[0][1])]
#
#                 print(f"Scoring task {task2.task_id} test case 0:")
#                 print(f"  Correct prediction: {task2.score_prediction(correct_pred)}")
#                 print(f"  Incorrect shape: {task2.score_prediction(incorrect_pred_shape)}")
#                 print(f"  Incorrect values: {task2.score_prediction(incorrect_pred_values)}")
#
#                 if len(task2.test) > 1 and task2.test[1][1] is not None:
#                     multi_preds = [np.copy(task2.test[0][1]), np.zeros_like(task2.test[1][1])]
#                     print(f"  Multiple predictions (correct, incorrect): {task2.score_prediction(multi_preds)}")
#
#             # Commented out submission scoring test as it might require specific eval files
#             # if sample_submission_path.exists():
#             #     print(f"\nTesting TaskSet score_submission_file with {sample_submission_path}")
#             #     eval_challenges_path = project_root / "data" / "raw" / "arc-prize-2025" / "arc-agi_evaluation_challenges.json"
#             #     eval_solutions_path = project_root / "data" / "raw" / "arc-prize-2025" / "arc-agi_evaluation_solutions.json"
#             #     if eval_challenges_path.exists() and eval_solutions_path.exists():
#             #         eval_loader = ARCDataLoader(eval_challenges_path, eval_solutions_path, "eval_for_submission_test")
#             #         eval_task_set = eval_loader.load_all_tasks()
#             #         score, correct_ids = eval_task_set.score_submission_file(sample_submission_path, return_correct_ids=True)
#             #         print(f"Sample submission score: {score} / {len(eval_task_set)}")
#             #         print(f"Correctly solved task IDs from sample: {correct_ids}")
#             #     else:
#             #         print("Evaluation files for sample submission not found, skipping submission scoring test.")
#             # else:
#             #     print(f"Sample submission file not found at {sample_submission_path}, skipping submission scoring test.")
```
