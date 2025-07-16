"""
Script to download ARC datasets from various sources.

Supports:
- Kaggle competition datasets (ARC-AGI-1, ARC-AGI-2)
- GitHub repository datasets (ConceptARC, MiniARC)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

from jaxarc.utils.config import get_raw_path
from jaxarc.utils.dataset_downloader import DatasetDownloader, DatasetDownloadError


def download_kaggle_data(competition_name: str, output_dir: Path) -> None:
    """
    Download data from Kaggle competition.

    Args:
        competition_name: Name of the Kaggle competition
        output_dir: Directory to save the downloaded data
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download competition data using kaggle CLI
    cmd = [
        "kaggle",
        "competitions",
        "download",
        competition_name,
        "--path",
        str(output_dir),
    ]

    logger.info(f"Downloading {competition_name} data to {output_dir}...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.success("Download completed successfully!")
        if result.stdout:
            logger.info(result.stdout)

        # Unzip files if they exist
        zip_files = list(output_dir.glob("*.zip"))
        for zip_file in zip_files:
            # Create extraction directory named after the competition
            extraction_dir = output_dir / competition_name
            extraction_dir.mkdir(exist_ok=True)

            logger.info(f"Extracting {zip_file} to {extraction_dir}...")
            subprocess.run(
                ["unzip", "-o", str(zip_file), "-d", str(extraction_dir)], check=True
            )
            # Optionally remove zip file after extraction
            zip_file.unlink()

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading data: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(
            "Error: kaggle CLI not found. Please install it with: pip install kaggle"
        )
        sys.exit(1)


def download_github_repository(repo_url: str, output_dir: Path, repo_name: str) -> None:
    """
    Download dataset from GitHub repository.

    Args:
        repo_url: GitHub repository URL
        output_dir: Directory to save the downloaded data
        repo_name: Name for the local directory
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = output_dir / repo_name

    # Remove existing directory if it exists
    if target_dir.exists():
        logger.info(f"Removing existing directory: {target_dir}")
        subprocess.run(["rm", "-rf", str(target_dir)], check=True)

    logger.info(f"Cloning {repo_url} to {target_dir}...")

    try:
        # Clone the repository
        cmd = ["git", "clone", repo_url, str(target_dir)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        logger.success(f"Successfully cloned {repo_name} dataset!")
        if result.stdout:
            logger.info(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(
            "Error: git not found. Please install git to download GitHub repositories."
        )
        sys.exit(1)


def download_conceptarc(output_dir: Path) -> None:
    """Download ConceptARC dataset from GitHub using DatasetDownloader."""
    try:
        downloader = DatasetDownloader(output_dir)
        downloader.download_conceptarc()
    except DatasetDownloadError as e:
        logger.error(f"Failed to download ConceptARC: {e}")
        sys.exit(1)


def download_miniarc(output_dir: Path) -> None:
    """Download MiniARC dataset from GitHub using DatasetDownloader."""
    try:
        downloader = DatasetDownloader(output_dir)
        downloader.download_miniarc()
    except DatasetDownloadError as e:
        logger.error(f"Failed to download MiniARC: {e}")
        sys.exit(1)


app = typer.Typer(
    help="Download ARC datasets from various sources",
    epilog="""
Examples:
  # Download specific datasets
  python scripts/download_kaggle_dataset.py download-conceptarc
  python scripts/download_kaggle_dataset.py download-miniarc

  # Download with custom options
  python scripts/download_kaggle_dataset.py download-conceptarc --output /custom/path --force
  python scripts/download_kaggle_dataset.py download-miniarc --no-validate

  # Download Kaggle datasets (requires kaggle CLI)
  python scripts/download_kaggle_dataset.py kaggle arc-prize-2024
  python scripts/download_kaggle_dataset.py kaggle arc-prize-2025

  # Download all datasets
  python scripts/download_kaggle_dataset.py all-datasets
    """,
    rich_markup_mode="rich",
)


@app.command()
def kaggle(
    competition: str = typer.Argument(
        default="arc-prize-2025", help="Name of the Kaggle competition"
    ),
) -> None:
    """Download Kaggle competition data (ARC-AGI-1, ARC-AGI-2)."""
    raw_path = get_raw_path(create=True)
    download_kaggle_data(competition, raw_path)


@app.command()
def conceptarc(
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: configured raw data path)",
    ),
) -> None:
    """Download ConceptARC dataset from GitHub."""
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    download_conceptarc(output_dir)
    logger.info(f"ConceptARC dataset downloaded to: {output_dir / 'ConceptARC'}")
    logger.info("Configure your dataset path in conf/dataset/concept_arc.yaml")


@app.command(name="download-conceptarc")
def download_conceptarc_cmd(
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: configured raw data path)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force download even if directory exists"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate dataset structure after download",
    ),
) -> None:
    """
    Download ConceptARC dataset from GitHub repository.

    ConceptARC is a benchmark dataset organized around 16 concept groups with 10 tasks each,
    designed to systematically assess abstraction and generalization abilities.

    Examples:
        # Download to default location
        python scripts/download_kaggle_dataset.py download-conceptarc

        # Download to custom directory
        python scripts/download_kaggle_dataset.py download-conceptarc -o /path/to/data

        # Force re-download
        python scripts/download_kaggle_dataset.py download-conceptarc --force
    """
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = output_dir / "ConceptARC"

    # Check if already exists and handle force option
    if target_dir.exists() and not force:
        logger.info(f"ConceptARC already exists at {target_dir}")
        logger.info(
            "Use --force to re-download or specify a different output directory"
        )
        return

    try:
        downloader = DatasetDownloader(output_dir)
        downloaded_path = downloader.download_conceptarc()

        logger.success(f"ConceptARC dataset downloaded to: {downloaded_path}")
        logger.info("Dataset structure:")
        logger.info("  - 16 concept groups in corpus/ directory")
        logger.info("  - 10 tasks per concept group")
        logger.info("  - Standard ARC JSON format")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Configure dataset path in conf/dataset/concept_arc.yaml")
        logger.info("  2. Use ConceptArcParser to load tasks by concept group")

    except DatasetDownloadError as e:
        logger.error(f"Failed to download ConceptARC: {e}")
        sys.exit(1)


@app.command()
def miniarc(
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: configured raw data path)",
    ),
) -> None:
    """Download MiniARC dataset from GitHub."""
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    download_miniarc(output_dir)
    logger.info(f"MiniARC dataset downloaded to: {output_dir / 'MiniARC'}")
    logger.info("Configure your dataset path in conf/dataset/mini_arc.yaml")


@app.command(name="download-miniarc")
def download_miniarc_cmd(
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: configured raw data path)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force download even if directory exists"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate dataset structure after download",
    ),
) -> None:
    """
    Download MiniARC dataset from GitHub repository.

    MiniARC is a 5x5 compact version of ARC with 400 training and 400 evaluation tasks,
    designed for faster experimentation and prototyping.

    Examples:
        # Download to default location
        python scripts/download_kaggle_dataset.py download-miniarc

        # Download to custom directory
        python scripts/download_kaggle_dataset.py download-miniarc -o /path/to/data

        # Force re-download without validation
        python scripts/download_kaggle_dataset.py download-miniarc --force --no-validate
    """
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = output_dir / "MiniARC"

    # Check if already exists and handle force option
    if target_dir.exists() and not force:
        logger.info(f"MiniARC already exists at {target_dir}")
        logger.info(
            "Use --force to re-download or specify a different output directory"
        )
        return

    try:
        downloader = DatasetDownloader(output_dir)
        downloaded_path = downloader.download_miniarc()

        logger.success(f"MiniARC dataset downloaded to: {downloaded_path}")
        logger.info("Dataset structure:")
        logger.info("  - 400+ task files in data/ directory")
        logger.info("  - Optimized for 5x5 grids")
        logger.info("  - Standard ARC JSON format")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Configure dataset path in conf/dataset/mini_arc.yaml")
        logger.info("  2. Use MiniArcParser for optimized 5x5 grid processing")

    except DatasetDownloadError as e:
        logger.error(f"Failed to download MiniARC: {e}")
        sys.exit(1)


@app.command(name="all-datasets")
def all_datasets(
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: configured raw data path)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force download even if directories exist"
    ),
    skip_kaggle: bool = typer.Option(
        False,
        "--skip-kaggle",
        help="Skip Kaggle datasets (useful if kaggle CLI not available)",
    ),
) -> None:
    """
    Download all available ARC datasets.

    This command downloads:
    - ARC-AGI 2024 and 2025 datasets from Kaggle (if kaggle CLI available)
    - ConceptARC dataset from GitHub
    - MiniARC dataset from GitHub

    Examples:
        # Download all datasets to default location
        python scripts/download_kaggle_dataset.py all-datasets

        # Skip Kaggle datasets if CLI not available
        python scripts/download_kaggle_dataset.py all-datasets --skip-kaggle

        # Force re-download all datasets
        python scripts/download_kaggle_dataset.py all-datasets --force
    """
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading all ARC datasets...")

    # Download Kaggle datasets
    if not skip_kaggle:
        try:
            logger.info("Downloading Kaggle datasets...")
            download_kaggle_data("arc-prize-2024", output_dir)
            download_kaggle_data("arc-prize-2025", output_dir)
            logger.success("Kaggle datasets downloaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to download Kaggle datasets: {e}")
            logger.info("You can download them manually or install kaggle CLI")
            logger.info("Use --skip-kaggle to skip Kaggle datasets in future runs")
    else:
        logger.info("Skipping Kaggle datasets as requested")

    # Download GitHub datasets using enhanced commands
    logger.info("Downloading GitHub datasets...")

    try:
        # Use the enhanced download functions with force option
        downloader = DatasetDownloader(output_dir)

        # ConceptARC
        conceptarc_dir = output_dir / "ConceptARC"
        if conceptarc_dir.exists() and not force:
            logger.info(
                "ConceptARC already exists, skipping (use --force to re-download)"
            )
        else:
            downloader.download_conceptarc()
            logger.success("ConceptARC downloaded successfully!")

        # MiniARC
        miniarc_dir = output_dir / "MiniARC"
        if miniarc_dir.exists() and not force:
            logger.info("MiniARC already exists, skipping (use --force to re-download)")
        else:
            downloader.download_miniarc()
            logger.success("MiniARC downloaded successfully!")

    except DatasetDownloadError as e:
        logger.error(f"Failed to download GitHub datasets: {e}")
        sys.exit(1)

    logger.success("All requested datasets downloaded successfully!")
    logger.info("")
    logger.info("Dataset summary:")
    logger.info(f"  - Data directory: {output_dir}")
    if not skip_kaggle:
        logger.info("  - ARC-AGI 2024: Kaggle competition dataset")
        logger.info("  - ARC-AGI 2025: Kaggle competition dataset")
    logger.info("  - ConceptARC: 16 concept groups, 10 tasks each")
    logger.info("  - MiniARC: 400+ tasks optimized for 5x5 grids")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Configure dataset paths in conf/dataset/ files")
    logger.info("  2. Use appropriate parsers for each dataset type")
    logger.info("  3. Run examples to test dataset loading")


def main(
    competition: str = typer.Argument(
        default="arc-prize-2025",
        help="Name of the Kaggle competition (deprecated, use 'kaggle' command)",
    ),
) -> None:
    """Download Kaggle competition data (deprecated - use specific commands)."""
    logger.warning("Direct execution is deprecated. Use specific commands:")
    logger.info("  python scripts/download_kaggle_dataset.py kaggle arc-prize-2025")
    logger.info("  python scripts/download_kaggle_dataset.py download-conceptarc")
    logger.info("  python scripts/download_kaggle_dataset.py download-miniarc")
    logger.info("  python scripts/download_kaggle_dataset.py all-datasets")

    raw_path = get_raw_path(create=True)
    download_kaggle_data(competition, raw_path)


if __name__ == "__main__":
    app()
