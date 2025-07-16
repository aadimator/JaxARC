"""
Dataset downloader utility for JaxARC datasets.

Provides a unified interface for downloading datasets from various sources
including GitHub repositories and Kaggle competitions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


class DatasetDownloadError(Exception):
    """Exception raised when dataset download fails."""


class DatasetDownloader:
    """Unified dataset downloader supporting multiple sources."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the dataset downloader.

        Args:
            output_dir: Base directory for downloads. If None, uses current directory.
        """
        self.output_dir = output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_conceptarc(self, target_dir: Optional[Path] = None) -> Path:
        """
        Download ConceptARC dataset from GitHub repository.

        Args:
            target_dir: Specific target directory. If None, uses output_dir/ConceptARC.

        Returns:
            Path to the downloaded dataset directory.

        Raises:
            DatasetDownloadError: If download fails.
        """
        repo_url = "https://github.com/victorvikram/ConceptARC.git"
        repo_name = "ConceptARC"

        if target_dir is None:
            target_dir = self.output_dir / repo_name

        logger.info(f"Downloading ConceptARC dataset to {target_dir}")

        try:
            return self._clone_repository(repo_url, target_dir, repo_name)
        except Exception as e:
            raise DatasetDownloadError(f"Failed to download ConceptARC: {e}") from e

    def download_miniarc(self, target_dir: Optional[Path] = None) -> Path:
        """
        Download MiniARC dataset from GitHub repository.

        Args:
            target_dir: Specific target directory. If None, uses output_dir/MiniARC.

        Returns:
            Path to the downloaded dataset directory.

        Raises:
            DatasetDownloadError: If download fails.
        """
        repo_url = "https://github.com/KSB21ST/MINI-ARC.git"
        repo_name = "MiniARC"

        if target_dir is None:
            target_dir = self.output_dir / repo_name

        logger.info(f"Downloading MiniARC dataset to {target_dir}")

        try:
            return self._clone_repository(repo_url, target_dir, repo_name)
        except Exception as e:
            raise DatasetDownloadError(f"Failed to download MiniARC: {e}") from e

    def _clone_repository(
        self, repo_url: str, target_dir: Path, repo_name: str
    ) -> Path:
        """
        Clone Git repository with comprehensive error handling.

        Args:
            repo_url: GitHub repository URL
            target_dir: Target directory for cloning
            repo_name: Name of the repository for logging

        Returns:
            Path to the cloned repository directory.

        Raises:
            DatasetDownloadError: If cloning fails.
        """
        # Validate prerequisites
        self._check_git_availability()
        self._check_network_connectivity(repo_url)
        self._check_disk_space(target_dir.parent)
        self._check_write_permissions(target_dir.parent)

        # Remove existing directory if it exists
        if target_dir.exists():
            logger.info(f"Removing existing directory: {target_dir}")
            try:
                shutil.rmtree(target_dir)
            except OSError as e:
                raise DatasetDownloadError(
                    f"Failed to remove existing directory {target_dir}: {e}"
                ) from e

        # Clone the repository
        logger.info(f"Cloning {repo_url} to {target_dir}...")

        try:
            cmd = ["git", "clone", repo_url, str(target_dir)]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            logger.success(f"Successfully cloned {repo_name} dataset!")
            if result.stdout:
                logger.debug(f"Git output: {result.stdout}")

            # Validate the download
            self._validate_download(target_dir, repo_name)

            return target_dir

        except subprocess.TimeoutExpired as e:
            raise DatasetDownloadError(
                "Git clone timed out after 5 minutes. Check your network connection."
            ) from e
        except subprocess.CalledProcessError as e:
            error_msg = f"Git clone failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            raise DatasetDownloadError(error_msg) from e

    def _check_git_availability(self) -> None:
        """Check if git is available on the system."""
        try:
            subprocess.run(
                ["git", "--version"], check=True, capture_output=True, text=True
            )
        except FileNotFoundError as e:
            raise DatasetDownloadError(
                "Git is not installed or not available in PATH. "
                "Please install git to download GitHub repositories."
            ) from e
        except subprocess.CalledProcessError as e:
            raise DatasetDownloadError(f"Git is not working properly: {e}") from e

    def _check_network_connectivity(self, repo_url: str) -> None:
        """Check basic network connectivity to the repository host."""
        try:
            # Extract hostname from URL
            if repo_url.startswith("https://"):
                hostname = repo_url.split("/")[2]
            else:
                hostname = "github.com"

            # Simple ping test (works on most systems)
            result = subprocess.run(
                ["ping", "-c", "1", hostname],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(f"Network connectivity test to {hostname} failed")
                logger.info(
                    "Proceeding anyway - git clone will provide detailed error if needed"
                )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Ping might not be available or might timeout
            logger.debug("Network connectivity test skipped")

    def _check_disk_space(self, directory: Path, min_space_mb: int = 100) -> None:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(directory)
            available_mb = stat.free // (1024 * 1024)

            if available_mb < min_space_mb:
                raise DatasetDownloadError(
                    f"Insufficient disk space. Available: {available_mb}MB, "
                    f"Required: {min_space_mb}MB"
                )

        except OSError as e:
            logger.warning(f"Could not check disk space: {e}")

    def _check_write_permissions(self, directory: Path) -> None:
        """Check write permissions for the target directory."""
        try:
            # Try to create the directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)

            # Test write access by creating a temporary file
            test_file = directory / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except OSError as e:
                raise DatasetDownloadError(
                    f"No write permission for directory {directory}: {e}"
                ) from e

        except OSError as e:
            raise DatasetDownloadError(
                f"Cannot create or access directory {directory}: {e}"
            ) from e

    def _validate_download(self, target_dir: Path, repo_name: str) -> None:
        """Validate that the download was successful."""
        if not target_dir.exists():
            raise DatasetDownloadError(
                f"Download directory {target_dir} does not exist"
            )

        if not target_dir.is_dir():
            raise DatasetDownloadError(f"Download path {target_dir} is not a directory")

        # Check if directory is not empty
        try:
            contents = list(target_dir.iterdir())
            if not contents:
                raise DatasetDownloadError(
                    f"Downloaded directory {target_dir} is empty"
                )
        except OSError as e:
            raise DatasetDownloadError(
                f"Cannot access downloaded directory: {e}"
            ) from e

        # Dataset-specific validation
        if repo_name == "ConceptARC":
            self._validate_conceptarc_structure(target_dir)
        elif repo_name == "MiniARC":
            self._validate_miniarc_structure(target_dir)

    def _validate_conceptarc_structure(self, target_dir: Path) -> None:
        """Validate ConceptARC dataset structure."""
        corpus_dir = target_dir / "corpus"
        if not corpus_dir.exists():
            raise DatasetDownloadError(
                f"ConceptARC corpus directory not found at {corpus_dir}"
            )

        # Check for at least some concept groups
        concept_groups = [d for d in corpus_dir.iterdir() if d.is_dir()]
        if len(concept_groups) < 5:  # Expect at least 5 concept groups
            logger.warning(
                f"ConceptARC: Found only {len(concept_groups)} concept groups, "
                "expected around 16"
            )

    def _validate_miniarc_structure(self, target_dir: Path) -> None:
        """Validate MiniARC dataset structure."""
        data_dir = target_dir / "data"
        if not data_dir.exists():
            # Try alternative structure
            miniarc_dir = target_dir / "MiniARC"
            if miniarc_dir.exists():
                logger.info("Found MiniARC data in MiniARC subdirectory")
                return

            raise DatasetDownloadError(
                f"MiniARC data directory not found at {data_dir} or {miniarc_dir}"
            )

        # Check for JSON files
        json_files = list(data_dir.glob("*.json"))
        if len(json_files) < 100:  # Expect hundreds of task files
            logger.warning(
                f"MiniARC: Found only {len(json_files)} JSON files, "
                "expected around 400+"
            )
