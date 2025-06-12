"""
Demonstration script showing how to use the ArcAgiParser with different datasets.

This script shows how to:
1. Load configuration for different ARC-AGI datasets
2. Use the parser to load and parse tasks
3. Switch between ARC-AGI-1 and ARC-AGI-2 datasets

Usage:
    pixi run python scripts/demo_parser.py environment=arc_agi_1
    pixi run python scripts/demo_parser.py environment=arc_agi_2
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from jaxarc.parsers import ArcAgiParser


def load_and_parse_sample_task(cfg: DictConfig) -> None:
    """Load and parse a sample task from the configured dataset."""
    parser = ArcAgiParser()

    # Determine which dataset split to use (e.g., training, evaluation)
    # For this demo, we'll default to 'training' if not specified,
    # but you might want to make this configurable via command line.
    dataset_type = cfg.environment.get("default_split", "training")  # Example: default to training

    if dataset_type not in cfg.environment:
        logger.error(f"Dataset type '{dataset_type}' not found in configuration.")
        return

    current_config = cfg.environment[dataset_type]
    base_dataset_name = cfg.environment.dataset_name

    logger.info(
        f"Dataset: {base_dataset_name}_{dataset_type} ({cfg.environment.dataset_year})"
    )
    logger.info(f"Description: {cfg.environment.description}")
    logger.info(f"Data root: {cfg.environment.data_root}")

    challenges_path = Path(current_config.challenges)
    solutions_path = Path(current_config.solutions) if "solutions" in current_config else None

    if challenges_path.exists():
        logger.info(f"Loading {dataset_type} tasks from: {challenges_path}")
        if solutions_path:
            logger.info(f"Using solutions from: {solutions_path}")
        else:
            logger.info(f"No solutions file specified for {dataset_type}.")

        try:
            tasks = parser.parse_all_tasks_from_file(challenges_path, solutions_path)
            logger.info(f"Found {len(tasks)} tasks in the {dataset_type} file")

            if tasks:
                # Show details of the first task
                first_task_id = next(iter(tasks.keys()))
                first_task = tasks[first_task_id]

                logger.info(f"Sample task: {first_task_id}")
                logger.info(f"  Training pairs: {len(first_task.train_pairs)}")
                logger.info(f"  Test pairs: {len(first_task.test_pairs)}")

                if first_task.train_pairs:
                    first_train = first_task.train_pairs[0]
                    logger.info(f"  First training input shape: {first_train.input.array.shape}")
                    if first_train.output:
                        logger.info(f"  First training output shape: {first_train.output.array.shape}")
                    else:
                        logger.info("  First training output: Not available")

                if first_task.test_pairs:
                    first_test = first_task.test_pairs[0]
                    logger.info(f"  First test input shape: {first_test.input.array.shape}")
                    if first_test.output:
                        logger.info(f"  First test output shape: {first_test.output.array.shape}")
                    else:
                        logger.info("  First test output: Not available (or not loaded)")

        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.error(f"Error parsing tasks: {e}")

    else:
        logger.warning(f"Training challenges file not found at {challenges_path}")
        logger.info(
            "Make sure to download the dataset and adjust the paths in the config files."
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function demonstrating parser usage."""

    logger.info("ARC-AGI Parser Demonstration")
    logger.info("=" * 40)

    load_and_parse_sample_task(cfg)

    logger.info("=" * 40)
    logger.info("Demonstration complete!")
    logger.info("To switch datasets, use environment argument, e.g.:")
    logger.info("  pixi run python scripts/demo_parser.py environment=arc_agi_1")
    logger.info("  pixi run python scripts/demo_parser.py environment=arc_agi_2")
    logger.info("You can also specify a split, e.g., by modifying the config or script to select 'evaluation' or 'testing'.")


if __name__ == "__main__":
    # Hydra decorator automatically provides cfg parameter when script is run
    main()  # # pylint: disable=E1120
