"""
Simple dataset download script for JaxARC.

Uses the unified DatasetManager to download datasets from their configured repositories.
All dataset metadata is now configuration-driven via YAML files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

from jaxarc.configs import JaxArcConfig
from jaxarc.utils import DatasetError, DatasetManager
from jaxarc.utils.config import get_config

app = typer.Typer(
    help="Download ARC datasets from configured repositories",
    rich_markup_mode="rich",
)


def _get_dataset_config(dataset_name: str) -> JaxArcConfig:
    """Load dataset configuration from YAML files."""
    try:
        # Map dataset names to config files
        config_mapping = {
            "arc-agi-1": "arc_agi_1",
            "arc-agi-2": "arc_agi_2",
            "conceptarc": "concept_arc",
            "miniarc": "mini_arc",
        }

        if dataset_name not in config_mapping:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config_name = config_mapping[dataset_name]
        config = get_config(overrides=[f"dataset={config_name}"])
        return JaxArcConfig.from_hydra(config)

    except Exception as e:
        logger.error(f"Failed to load config for {dataset_name}: {e}")
        sys.exit(1)


def _download_dataset(
    dataset_name: str, output_dir: Path | None = None, force: bool = False
) -> None:
    """Download a dataset using the unified DatasetManager."""
    try:
        # Load dataset configuration
        config = _get_dataset_config(dataset_name)

        # Override output directory if specified
        if output_dir is not None:
            import equinox as eqx

            dataset_config = config.dataset
            dataset_config = eqx.tree_at(
                lambda d: d.dataset_path,
                dataset_config,
                str(output_dir / config.dataset.dataset_name),
            )
            config = eqx.tree_at(lambda c: c.dataset, config, dataset_config)

        # Check if dataset already exists
        manager = DatasetManager()
        dataset_path = Path(config.dataset.dataset_path)
        if not dataset_path.is_absolute():
            from pyprojroot import here

            dataset_path = here() / dataset_path

        if (
            dataset_path.exists()
            and manager.validate_dataset(config.dataset, dataset_path)
            and not force
        ):
            logger.info(
                f"Dataset {config.dataset.dataset_name} already exists at {dataset_path}"
            )
            logger.info("Use --force to re-download")
            return

        # Download the dataset
        logger.info(f"Downloading {config.dataset.dataset_name}...")
        downloaded_path = manager.ensure_dataset_available(config, auto_download=True)

        logger.success(
            f"Successfully downloaded {config.dataset.dataset_name} to {downloaded_path}"
        )
        logger.info(f"Repository: {config.dataset.dataset_repo}")
        logger.info(f"Parser: {config.dataset.parser_entry_point}")

    except DatasetError as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error downloading {dataset_name}: {e}")
        sys.exit(1)


@app.command()
def arc_agi_1(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-1 dataset."""
    _download_dataset("arc-agi-1", output_dir, force)


@app.command()
def arc_agi_2(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-2 dataset."""
    _download_dataset("arc-agi-2", output_dir, force)


@app.command()
def conceptarc(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ConceptARC dataset."""
    _download_dataset("conceptarc", output_dir, force)


@app.command()
def miniarc(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download MiniARC dataset."""
    _download_dataset("miniarc", output_dir, force)


@app.command(name="all")
def download_all(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download all ARC datasets."""
    datasets = ["arc-agi-1", "arc-agi-2", "conceptarc", "miniarc"]

    logger.info("Downloading all ARC datasets...")

    for dataset_name in datasets:
        try:
            _download_dataset(dataset_name, output_dir, force)
        except SystemExit:
            logger.error(
                f"Failed to download {dataset_name}, continuing with others..."
            )
            continue

    logger.success("Dataset download process completed!")


if __name__ == "__main__":
    app()
