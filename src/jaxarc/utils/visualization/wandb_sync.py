"""Utilities for syncing offline wandb data and managing wandb connectivity."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WandbSyncManager:
    """Manager for syncing offline wandb data and handling connectivity issues."""

    def __init__(self, cache_dir: Path, project_name: str = "jaxarc-experiments"):
        """Initialize sync manager.

        Args:
            cache_dir: Directory containing cached wandb data
            project_name: Wandb project name for syncing
        """
        self.cache_dir = Path(cache_dir)
        self.project_name = project_name
        self.metadata_file = self.cache_dir / "cache_metadata.json"

    def sync_all_cached_data(self, entity: Optional[str] = None) -> Dict[str, Any]:
        """Sync all cached data using wandb CLI.

        Args:
            entity: Optional wandb entity (username or team)

        Returns:
            Dictionary with sync results
        """
        if not self.cache_dir.exists():
            return {
                "success": False,
                "error": f"Cache directory does not exist: {self.cache_dir}",
                "synced_runs": 0,
                "failed_runs": 0,
            }

        # Find all wandb offline runs in cache
        offline_runs = self._find_offline_runs()

        if not offline_runs:
            return {
                "success": True,
                "message": "No offline runs found to sync",
                "synced_runs": 0,
                "failed_runs": 0,
            }

        logger.info(f"Found {len(offline_runs)} offline runs to sync")

        synced_count = 0
        failed_count = 0
        errors = []

        for run_dir in offline_runs:
            try:
                success = self._sync_single_run(run_dir, entity)
                if success:
                    synced_count += 1
                    logger.info(f"Successfully synced run: {run_dir.name}")
                else:
                    failed_count += 1
                    error_msg = f"Failed to sync run: {run_dir.name}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            except Exception as e:
                failed_count += 1
                error_msg = f"Error syncing run {run_dir.name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return {
            "success": failed_count == 0,
            "synced_runs": synced_count,
            "failed_runs": failed_count,
            "total_runs": len(offline_runs),
            "errors": errors,
        }

    def _find_offline_runs(self) -> List[Path]:
        """Find all offline wandb runs in the cache directory.

        Returns:
            List of paths to offline run directories
        """
        offline_runs = []

        # Look for wandb offline run directories
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                # Check if it's a wandb run directory
                if (item / "wandb-metadata.json").exists() or (item / "files").exists():
                    offline_runs.append(item)

                # Also check subdirectories for nested runs
                for subitem in item.iterdir():
                    if subitem.is_dir() and (
                        (subitem / "wandb-metadata.json").exists()
                        or (subitem / "files").exists()
                    ):
                        offline_runs.append(subitem)

        return offline_runs

    def _sync_single_run(self, run_dir: Path, entity: Optional[str] = None) -> bool:
        """Sync a single offline run using wandb CLI.

        Args:
            run_dir: Path to the offline run directory
            entity: Optional wandb entity

        Returns:
            True if sync was successful, False otherwise
        """
        try:
            # Build wandb sync command
            cmd = ["wandb", "sync", str(run_dir)]

            if entity:
                cmd.extend(["--entity", entity])

            # Add project if specified
            if self.project_name:
                cmd.extend(["--project", self.project_name])

            # Run the sync command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.debug(f"Wandb sync output: {result.stdout}")
                return True
            logger.error(f"Wandb sync failed: {result.stderr}")
            return False

        except subprocess.TimeoutExpired:
            logger.error(f"Wandb sync timed out for run: {run_dir.name}")
            return False
        except Exception as e:
            logger.error(f"Error running wandb sync: {e}")
            return False

    def check_wandb_connectivity(self) -> Tuple[bool, Optional[str]]:
        """Check if wandb is accessible and user is logged in.

        Returns:
            Tuple of (is_connected, error_message)
        """
        try:
            # Check if wandb CLI is available
            result = subprocess.run(
                ["wandb", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode != 0:
                return False, "wandb CLI not available"

            # Check if user is logged in
            result = subprocess.run(
                ["wandb", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                return True, None
            return False, "Not logged in to wandb"

        except subprocess.TimeoutExpired:
            return False, "wandb CLI timeout"
        except FileNotFoundError:
            return False, "wandb CLI not installed"
        except Exception as e:
            return False, f"Error checking wandb status: {e}"

    def get_cache_status(self) -> Dict[str, Any]:
        """Get status of cached data.

        Returns:
            Dictionary with cache status information
        """
        if not self.cache_dir.exists():
            return {
                "cache_exists": False,
                "total_size_mb": 0,
                "offline_runs": 0,
                "cached_entries": 0,
            }

        # Calculate total cache size
        total_size = 0
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        # Count offline runs
        offline_runs = len(self._find_offline_runs())

        # Count cached entries from metadata
        cached_entries = 0
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                    cached_entries = len(metadata.get("entries", []))
            except Exception as e:
                logger.warning(f"Error reading cache metadata: {e}")

        return {
            "cache_exists": True,
            "cache_directory": str(self.cache_dir),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024**2),
            "offline_runs": offline_runs,
            "cached_entries": cached_entries,
        }

    def cleanup_synced_data(self, confirm: bool = False) -> Dict[str, Any]:
        """Clean up successfully synced data.

        Args:
            confirm: Must be True to actually perform cleanup

        Returns:
            Dictionary with cleanup results
        """
        if not confirm:
            return {
                "success": False,
                "error": "Cleanup not confirmed. Set confirm=True to proceed.",
                "cleaned_files": 0,
            }

        if not self.cache_dir.exists():
            return {
                "success": True,
                "message": "Cache directory does not exist",
                "cleaned_files": 0,
            }

        cleaned_count = 0
        errors = []

        try:
            # Find and remove synced run directories
            offline_runs = self._find_offline_runs()

            for run_dir in offline_runs:
                try:
                    # Check if run appears to be synced (this is heuristic)
                    if self._is_run_synced(run_dir):
                        import shutil

                        shutil.rmtree(run_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned synced run: {run_dir.name}")

                except Exception as e:
                    error_msg = f"Error cleaning run {run_dir.name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Clean up empty directories
            self._cleanup_empty_directories()

            return {
                "success": len(errors) == 0,
                "cleaned_files": cleaned_count,
                "errors": errors,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during cleanup: {e}",
                "cleaned_files": cleaned_count,
            }

    def _is_run_synced(self, run_dir: Path) -> bool:
        """Check if a run appears to have been synced.

        This is a heuristic check - in practice, you might want to maintain
        a more reliable record of synced runs.

        Args:
            run_dir: Path to the run directory

        Returns:
            True if the run appears to be synced
        """
        # Look for sync markers or check modification times
        sync_marker = run_dir / ".synced"
        if sync_marker.exists():
            return True

        # Check if run is older than a certain threshold (e.g., 24 hours)
        # and assume it's been synced if it's old enough
        try:
            run_time = run_dir.stat().st_mtime
            current_time = time.time()
            age_hours = (current_time - run_time) / 3600

            # If run is older than 24 hours, assume it's synced
            # This is a heuristic and may not be reliable in all cases
            return age_hours > 24

        except Exception:
            return False

    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories in the cache."""
        try:
            for item in self.cache_dir.rglob("*"):
                if item.is_dir() and not any(item.iterdir()):
                    item.rmdir()
                    logger.debug(f"Removed empty directory: {item}")
        except Exception as e:
            logger.warning(f"Error cleaning empty directories: {e}")


def create_sync_manager(
    cache_dir: Optional[Path] = None, project_name: str = "jaxarc-experiments"
) -> WandbSyncManager:
    """Create a WandbSyncManager with default settings.

    Args:
        cache_dir: Optional cache directory (defaults to ~/.jaxarc/wandb_cache)
        project_name: Wandb project name

    Returns:
        Configured WandbSyncManager instance
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".jaxarc" / "wandb_cache"

    return WandbSyncManager(cache_dir, project_name)


def sync_offline_wandb_data(
    cache_dir: Optional[Path] = None,
    project_name: str = "jaxarc-experiments",
    entity: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to sync all offline wandb data.

    Args:
        cache_dir: Optional cache directory
        project_name: Wandb project name
        entity: Optional wandb entity

    Returns:
        Dictionary with sync results
    """
    sync_manager = create_sync_manager(cache_dir, project_name)
    return sync_manager.sync_all_cached_data(entity)


def check_wandb_status() -> Dict[str, Any]:
    """Check wandb connectivity and installation status.

    Returns:
        Dictionary with wandb status information
    """
    sync_manager = create_sync_manager()
    is_connected, error = sync_manager.check_wandb_connectivity()

    return {
        "connected": is_connected,
        "error": error,
        "cache_status": sync_manager.get_cache_status(),
    }
