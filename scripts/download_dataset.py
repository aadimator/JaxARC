"""
Streamlined dataset download script for JaxARC.

Provides simple commands to download all supported ARC datasets from GitHub repositories.
Replaces Kaggle-based downloading with unified GitHub-based approach.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import typer
from loguru import logger

from jaxarc.utils.config import get_raw_path
from jaxarc.utils.dataset_downloader import DatasetDownloader, DatasetDownloadError

# Dataset configuration constants
@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset download."""
    name: str
    display_name: str
    downloader_method: str
    expected_structure: dict[str, str | int]
    next_steps: list[str]

# Dataset configurations
DATASETS = {
    "arc-agi-1": DatasetConfig(
        name="arc-agi-1",
        display_name="ARC-AGI-1",
        downloader_method="download_arc_agi_1",
        expected_structure={
            "training_tasks": 400,
            "evaluation_tasks": 400,
            "structure": "Individual JSON files per task",
            "training_dir": "data/training/",
            "evaluation_dir": "data/evaluation/"
        },
        next_steps=[
            "Configure dataset path in conf/dataset/arc_agi_1.yaml",
            "Use ArcAgiParser to load tasks from GitHub format"
        ]
    ),
    "arc-agi-2": DatasetConfig(
        name="arc-agi-2",
        display_name="ARC-AGI-2",
        downloader_method="download_arc_agi_2",
        expected_structure={
            "training_tasks": 1000,
            "evaluation_tasks": 120,
            "structure": "Individual JSON files per task",
            "training_dir": "data/training/",
            "evaluation_dir": "data/evaluation/"
        },
        next_steps=[
            "Configure dataset path in conf/dataset/arc_agi_2.yaml",
            "Use ArcAgiParser to load tasks from GitHub format"
        ]
    ),
    "conceptarc": DatasetConfig(
        name="conceptarc",
        display_name="ConceptARC",
        downloader_method="download_conceptarc",
        expected_structure={
            "concept_groups": 16,
            "tasks_per_group": 10,
            "structure": "Standard ARC JSON format",
            "data_dir": "corpus/ directory"
        },
        next_steps=[
            "Configure dataset path in conf/dataset/concept_arc.yaml",
            "Use ConceptArcParser to load tasks by concept group"
        ]
    ),
    "miniarc": DatasetConfig(
        name="miniarc",
        display_name="MiniARC",
        downloader_method="download_miniarc",
        expected_structure={
            "total_tasks": "400+",
            "optimization": "5x5 grids",
            "structure": "Standard ARC JSON format",
            "data_dir": "data/ directory"
        },
        next_steps=[
            "Configure dataset path in conf/dataset/mini_arc.yaml",
            "Use MiniArcParser for optimized 5x5 grid processing"
        ]
    )
}

app = typer.Typer(
    help="Download ARC datasets from GitHub repositories",
    epilog="""
Examples:
  # Download specific datasets
  python scripts/download_dataset.py arc-agi-1
  python scripts/download_dataset.py arc-agi-2
  python scripts/download_dataset.py conceptarc
  python scripts/download_dataset.py miniarc

  # Download all datasets
  python scripts/download_dataset.py all

  # Download with custom options
  python scripts/download_dataset.py arc-agi-1 --output /custom/path --force
    """,
    rich_markup_mode="rich",
)


def _setup_output_directory(output_dir: Path | None) -> Path:
    """Setup and validate output directory."""
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory {output_dir}: {e}")
            sys.exit(1)
    return output_dir


def _check_existing_dataset(target_dir: Path, dataset_name: str, force: bool) -> bool:
    """Check if dataset already exists and handle force option."""
    if target_dir.exists() and not force:
        logger.info(f"{dataset_name} already exists at {target_dir}")
        logger.info("Use --force to re-download or specify a different output directory")
        return True
    return False


def _log_dataset_info(config: DatasetConfig, downloaded_path: Path) -> None:
    """Log dataset information and next steps."""
    logger.success(f"{config.display_name} dataset downloaded to: {downloaded_path}")
    logger.info("Dataset structure:")

    # Log structure information
    structure = config.expected_structure
    if "training_tasks" in structure and "evaluation_tasks" in structure:
        logger.info(f"  - {structure['training_tasks']} training tasks in {structure['training_dir']}")
        logger.info(f"  - {structure['evaluation_tasks']} evaluation tasks in {structure['evaluation_dir']}")
    elif "concept_groups" in structure:
        logger.info(f"  - {structure['concept_groups']} concept groups in {structure['data_dir']}")
        logger.info(f"  - {structure['tasks_per_group']} tasks per concept group")
    elif "total_tasks" in structure:
        logger.info(f"  - {structure['total_tasks']} task files in {structure['data_dir']}")
        if "optimization" in structure:
            logger.info(f"  - {structure['optimization']}")
    
    logger.info(f"  - {structure['structure']}")
    logger.info("")
    logger.info("Next steps:")
    for i, step in enumerate(config.next_steps, 1):
        logger.info(f"  {i}. {step}")


def _download_single_dataset(
    dataset_key: str,
    output_dir: Path | None = None,
    force: bool = False
) -> None:
    """Download a single dataset using configuration."""
    if dataset_key not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        sys.exit(1)
    
    config = DATASETS[dataset_key]
    output_dir = _setup_output_directory(output_dir)
    # Use display_name directly for consistent directory naming
    target_dir = output_dir / config.display_name
    
    if _check_existing_dataset(target_dir, config.display_name, force):
        return
    
    try:
        downloader = DatasetDownloader(output_dir)
        downloader_method = getattr(downloader, config.downloader_method)
        downloaded_path = downloader_method()
        _log_dataset_info(config, downloaded_path)
        
    except DatasetDownloadError as e:
        logger.error(f"Failed to download {config.display_name}: {e}")
        sys.exit(1)


@app.command()
def arc_agi_1(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-1 dataset from GitHub (fchollet/ARC-AGI)."""
    _download_single_dataset("arc-agi-1", output_dir, force)


@app.command()
def arc_agi_2(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ARC-AGI-2 dataset from GitHub (arcprize/ARC-AGI-2)."""
    _download_single_dataset("arc-agi-2", output_dir, force)


@app.command()
def conceptarc(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download ConceptARC dataset from GitHub."""
    _download_single_dataset("conceptarc", output_dir, force)


@app.command()
def miniarc(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download MiniARC dataset from GitHub."""
    _download_single_dataset("miniarc", output_dir, force)


def _download_all_datasets(output_dir: Path, force: bool) -> None:
    """Download all datasets with consistent error handling."""
    logger.info("Downloading all ARC datasets from GitHub...")
    
    downloader = DatasetDownloader(output_dir)
    downloaded_datasets = []
    
    for config in DATASETS.values():
        target_dir = output_dir / config.display_name
        
        if target_dir.exists() and not force:
            logger.info(f"{config.display_name} already exists, skipping (use --force to re-download)")
            continue
            
        logger.info(f"Downloading {config.display_name}...")
        try:
            downloader_method = getattr(downloader, config.downloader_method)
            downloader_method()
            downloaded_datasets.append(config.display_name)
            logger.success(f"{config.display_name} downloaded successfully!")
        except DatasetDownloadError as e:
            logger.error(f"Failed to download {config.display_name}: {e}")
            raise
    
    # Summary
    if downloaded_datasets:
        logger.success(f"Successfully downloaded {len(downloaded_datasets)} datasets!")
    else:
        logger.info("All datasets already exist. Use --force to re-download.")
    
    _log_all_datasets_summary(output_dir)


def _log_all_datasets_summary(output_dir: Path) -> None:
    """Log summary information for all datasets."""
    logger.info("")
    logger.info("Dataset summary:")
    logger.info(f"  - Data directory: {output_dir}")
    logger.info("  - ARC-AGI-1: 400 training + 400 evaluation tasks")
    logger.info("  - ARC-AGI-2: 1000 training + 120 evaluation tasks")
    logger.info("  - ConceptARC: 16 concept groups, 10 tasks each")
    logger.info("  - MiniARC: 400+ tasks optimized for 5x5 grids")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Configure dataset paths in conf/dataset/ files")
    logger.info("  2. Use appropriate parsers for each dataset type")
    logger.info("  3. Run examples to test dataset loading")


@app.command(name="all")
def download_all(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
) -> None:
    """Download all ARC datasets from GitHub."""
    output_dir = _setup_output_directory(output_dir)
    
    try:
        _download_all_datasets(output_dir, force)
    except DatasetDownloadError as e:
        logger.error(f"Failed to download datasets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
