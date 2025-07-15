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
    """Download ConceptARC dataset from GitHub."""
    repo_url = "https://github.com/victorvikram/ConceptARC.git"
    download_github_repository(repo_url, output_dir, "ConceptARC")


def download_miniarc(output_dir: Path) -> None:
    """Download MiniARC dataset from GitHub."""
    repo_url = "https://github.com/KSB21ST/MINI-ARC.git"
    download_github_repository(repo_url, output_dir, "MiniARC")


app = typer.Typer(help="Download ARC datasets from various sources")


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
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
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


@app.command()
def miniarc(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
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


@app.command()
def all_datasets(
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: configured raw data path)"
    ),
) -> None:
    """Download all available datasets."""
    if output_dir is None:
        output_dir = get_raw_path(create=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading all ARC datasets...")
    
    # Download Kaggle datasets
    try:
        download_kaggle_data("arc-prize-2024", output_dir)
        download_kaggle_data("arc-prize-2025", output_dir)
    except Exception as e:
        logger.warning(f"Failed to download Kaggle datasets: {e}")
        logger.info("You can download them manually or install kaggle CLI")
    
    # Download GitHub datasets
    download_conceptarc(output_dir)
    download_miniarc(output_dir)
    
    logger.success("All datasets downloaded successfully!")


def main(
    competition: str = typer.Argument(
        default="arc-prize-2025", help="Name of the Kaggle competition (deprecated, use 'kaggle' command)"
    ),
) -> None:
    """Download Kaggle competition data (deprecated - use specific commands)."""
    logger.warning("Direct execution is deprecated. Use specific commands:")
    logger.info("  python scripts/download_kaggle_dataset.py kaggle arc-prize-2025")
    logger.info("  python scripts/download_kaggle_dataset.py conceptarc")
    logger.info("  python scripts/download_kaggle_dataset.py miniarc")
    logger.info("  python scripts/download_kaggle_dataset.py all-datasets")
    
    raw_path = get_raw_path(create=True)
    download_kaggle_data(competition, raw_path)


if __name__ == "__main__":
    app()
