"""
Script to download data from Kaggle competition.
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


def main(
    competition: str = typer.Argument(
        default="arc-prize-2025", help="Name of the Kaggle competition"
    ),
) -> None:
    """Download Kaggle competition data to the configured raw data directory."""
    raw_path = get_raw_path(create=True)
    download_kaggle_data(competition, raw_path)


if __name__ == "__main__":
    typer.run(main)
