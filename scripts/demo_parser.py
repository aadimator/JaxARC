"""
Demonstration script showing how to use the ArcAgiParser with different datasets.

This script shows how to:
1. Load configuration for different ARC-AGI datasets
2. Use the parser to load and parse tasks into JAX-compatible structures
3. Switch between ARC-AGI-1 and ARC-AGI-2 datasets

Usage:
    pixi run python scripts/demo_parser.py environment=arc_agi_1
    pixi run python scripts/demo_parser.py environment=arc_agi_2
"""

from __future__ import annotations

import hydra
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser


def load_and_parse_sample_task(cfg: DictConfig) -> None:
    """Load and parse a sample task from the configured dataset."""
    # Create parser with configuration and max dimensions
    parser = ArcAgiParser(
        cfg=cfg.environment,
    )

    dataset_name = cfg.environment.dataset_name
    dataset_year = cfg.environment.dataset_year

    logger.info(f"Dataset: {dataset_name} ({dataset_year})")
    logger.info(f"Description: {cfg.environment.description}")
    logger.info(f"Available tasks: {len(parser.get_available_task_ids())}")

    # Get a random task
    key = jax.random.PRNGKey(42)
    try:
        parsed_task = parser.get_random_task(key)

        logger.info(f"Successfully parsed task: {parsed_task.task_id}")
        logger.info(f"Number of training pairs: {parsed_task.num_train_pairs}")
        logger.info(f"Number of test pairs: {parsed_task.num_test_pairs}")
        logger.info(f"Input grids shape: {parsed_task.input_grids_examples.shape}")
        logger.info(f"Output grids shape: {parsed_task.output_grids_examples.shape}")
        logger.info(f"Test input grids shape: {parsed_task.test_input_grids.shape}")

        # Show some details about the first training pair
        if parsed_task.num_train_pairs > 0:
            first_input_mask = parsed_task.input_masks_examples[0]
            first_output_mask = parsed_task.output_masks_examples[0]

            # Count valid (non-padded) cells
            input_valid_cells = int(first_input_mask.sum())
            output_valid_cells = int(first_output_mask.sum())

            logger.info(f"First training pair - input valid cells: {input_valid_cells}")
            logger.info(
                f"First training pair - output valid cells: {output_valid_cells}"
            )

            # Show actual grid content (only valid region)
            if input_valid_cells > 0:
                valid_rows = first_input_mask.any(axis=1)
                valid_cols = first_input_mask.any(axis=0)

                # Find the actual grid bounds
                row_indices = jnp.where(valid_rows)[0]
                col_indices = jnp.where(valid_cols)[0]

                if len(row_indices) > 0 and len(col_indices) > 0:
                    min_row, max_row = int(row_indices[0]), int(row_indices[-1]) + 1
                    min_col, max_col = int(col_indices[0]), int(col_indices[-1]) + 1

                    actual_input_grid = parsed_task.input_grids_examples[
                        0, min_row:max_row, min_col:max_col
                    ]
                    actual_output_grid = parsed_task.output_grids_examples[
                        0, min_row:max_row, min_col:max_col
                    ]

                    logger.info(
                        f"First training input grid ({actual_input_grid.shape}):"
                    )
                    logger.info(f"\n{actual_input_grid}")
                    logger.info(
                        f"First training output grid ({actual_output_grid.shape}):"
                    )
                    logger.info(f"\n{actual_output_grid}")

        # Test specific task retrieval
        available_ids = parser.get_available_task_ids()
        if available_ids:
            specific_task = parser.get_task_by_id(available_ids[0])
            logger.info(
                f"Successfully retrieved specific task: {specific_task.task_id}"
            )

        # Demonstrate JAX compatibility
        logger.info("Testing JAX compatibility...")

        def sum_valid_pixels(grids: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(grids * masks)

        jitted_sum = jax.jit(sum_valid_pixels)
        total_input_pixels = jitted_sum(  # pylint: disable=E1102
            parsed_task.input_grids_examples, parsed_task.input_masks_examples
        )
        logger.info(
            f"Total valid input pixels across all training pairs: {total_input_pixels}"
        )

    except Exception as e:
        logger.error(f"Error parsing task: {e}")
        raise


def analyze_dataset_statistics(cfg: DictConfig) -> None:
    """Analyze the dataset to find actual maximum training and test pairs."""
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS ANALYSIS")
    logger.info("=" * 50)

    # Create parser with configuration
    parser = ArcAgiParser(cfg=cfg.environment)

    # Access the cached tasks to analyze them
    cached_tasks = parser._cached_tasks

    max_train_pairs = 0
    max_test_pairs = 0
    max_grid_height = 0
    max_grid_width = 0

    train_pair_counts = []
    test_pair_counts = []

    logger.info(f"Analyzing {len(cached_tasks)} tasks...")

    for _task_id, task_data in cached_tasks.items():
        # Count training pairs
        train_pairs = len(task_data.get("train", []))
        test_pairs = len(task_data.get("test", []))

        train_pair_counts.append(train_pairs)
        test_pair_counts.append(test_pairs)

        max_train_pairs = max(max_train_pairs, train_pairs)
        max_test_pairs = max(max_test_pairs, test_pairs)

        # Also check grid dimensions
        for pair in task_data.get("train", []):
            if pair.get("input"):
                height = len(pair["input"])
                width = len(pair["input"][0]) if height > 0 else 0
                max_grid_height = max(max_grid_height, height)
                max_grid_width = max(max_grid_width, width)
            if pair.get("output"):
                height = len(pair["output"])
                width = len(pair["output"][0]) if height > 0 else 0
                max_grid_height = max(max_grid_height, height)
                max_grid_width = max(max_grid_width, width)

        for pair in task_data.get("test", []):
            if pair.get("input"):
                height = len(pair["input"])
                width = len(pair["input"][0]) if height > 0 else 0
                max_grid_height = max(max_grid_height, height)
                max_grid_width = max(max_grid_width, width)
            if pair.get("output"):
                height = len(pair["output"])
                width = len(pair["output"][0]) if height > 0 else 0
                max_grid_height = max(max_grid_height, height)
                max_grid_width = max(max_grid_width, width)

    # Report findings
    logger.info("ACTUAL DATASET MAXIMUMS:")
    logger.info(f"  Maximum training pairs per task: {max_train_pairs}")
    logger.info(f"  Maximum test pairs per task: {max_test_pairs}")
    logger.info(f"  Maximum grid height: {max_grid_height}")
    logger.info(f"  Maximum grid width: {max_grid_width}")
    logger.info("")
    logger.info("CURRENT CONFIG SETTINGS:")
    logger.info(f"  Configured max_train_pairs: {cfg.environment.max_train_pairs}")
    logger.info(f"  Configured max_test_pairs: {cfg.environment.max_test_pairs}")
    logger.info(f"  Configured max_grid_height: {cfg.environment.max_grid_height}")
    logger.info(f"  Configured max_grid_width: {cfg.environment.max_grid_width}")
    logger.info("")

    # Provide recommendations
    if max_train_pairs <= cfg.environment.max_train_pairs:
        logger.info("✅ Current max_train_pairs setting is sufficient")
    else:
        logger.warning(
            f"⚠️  Recommended to increase max_train_pairs to {max_train_pairs}"
        )

    if max_test_pairs <= cfg.environment.max_test_pairs:
        logger.info("✅ Current max_test_pairs setting is sufficient")
    else:
        logger.warning(f"⚠️  Recommended to increase max_test_pairs to {max_test_pairs}")

    if max_grid_height <= cfg.environment.max_grid_height:
        logger.info("✅ Current max_grid_height setting is sufficient")
    else:
        logger.warning(
            f"⚠️  Recommended to increase max_grid_height to {max_grid_height}"
        )

    if max_grid_width <= cfg.environment.max_grid_width:
        logger.info("✅ Current max_grid_width setting is sufficient")
    else:
        logger.warning(f"⚠️  Recommended to increase max_grid_width to {max_grid_width}")

    logger.info("=" * 50)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the parser demo."""
    logger.info("Starting ARC-AGI Parser Demo")
    logger.info(f"Configuration: {cfg.environment.dataset_name}")

    try:
        # First analyze the dataset statistics
        analyze_dataset_statistics(cfg)

        # Then run the sample task demonstration
        load_and_parse_sample_task(cfg)
        logger.info("Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return


if __name__ == "__main__":
    main()  # pylint: disable=E1120
