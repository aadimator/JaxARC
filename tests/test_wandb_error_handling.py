"""Tests for wandb error handling and offline support."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.jaxarc.utils.visualization.wandb_integration import (
    WandbConfig,
    WandbIntegration,
)
from src.jaxarc.utils.visualization.wandb_sync import (
    WandbSyncManager,
    create_sync_manager,
)


class TestWandbErrorHandling:
    """Test wandb error handling and recovery mechanisms."""

    def test_network_error_detection(self):
        """Test detection of network-related errors."""
        config = WandbConfig(enabled=True, auto_offline_on_error=True)

        with patch(
            "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
        ):
            integration = WandbIntegration(config)

            # Test various network error types
            network_errors = [
                ConnectionError("Connection failed"),
                TimeoutError("Request timed out"),
                OSError("Network is unreachable"),
                Exception("DNS resolution failed"),
                Exception("Connection refused"),
                Exception("Broken pipe"),
            ]

            for error in network_errors:
                assert integration._is_network_error(error), (
                    f"Should detect {type(error).__name__} as network error"
                )

            # Test non-network errors
            non_network_errors = [
                ValueError("Invalid value"),
                KeyError("Missing key"),
                TypeError("Wrong type"),
            ]

            for error in non_network_errors:
                assert not integration._is_network_error(error), (
                    f"Should not detect {type(error).__name__} as network error"
                )

    def test_automatic_offline_mode_switch(self):
        """Test automatic switch to offline mode on network errors."""
        config = WandbConfig(enabled=True, auto_offline_on_error=True, retry_attempts=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.offline_cache_dir = temp_dir

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._wandb_available = True
                integration._wandb = Mock()  # Add the missing _wandb attribute
                integration.run = Mock()
                integration.run.id = "test-run-id"

                # Mock network error on log attempt
                integration.run.log.side_effect = ConnectionError("Network unreachable")

                # Attempt to log data
                result = integration._log_with_retry({"test": "data"}, step=1)

                # Should succeed by switching to offline mode
                assert result is True
                assert integration._offline_mode_active is True
                assert len(integration._cached_entries) == 1

    def test_retry_with_exponential_backoff(self):
        """Test retry logic with exponential backoff."""
        config = WandbConfig(
            enabled=True,
            retry_attempts=3,
            retry_delay=0.1,
            max_retry_delay=1.0,
            auto_offline_on_error=False,
        )

        with patch(
            "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
        ):
            with patch("time.sleep") as mock_sleep:
                integration = WandbIntegration(config)
                integration._wandb_available = True
                integration._wandb = Mock()  # Add the missing _wandb attribute
                integration.run = Mock()

                # Mock temporary failures followed by success
                integration.run.log.side_effect = [
                    Exception("Temporary error"),
                    Exception("Another temporary error"),
                    None,  # Success on third attempt
                ]

                start_time = time.time()
                result = integration._log_with_retry({"test": "data"}, step=1)

                # Should succeed after retries
                assert result is True
                assert integration.run.log.call_count == 3

                # Check exponential backoff was used
                assert mock_sleep.call_count == 2
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert sleep_calls[0] < sleep_calls[1]  # Increasing delay

    def test_offline_cache_management(self):
        """Test offline cache creation and management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(
                enabled=True,
                offline_cache_dir=temp_dir,
                max_cache_size_gb=0.001,  # Very small limit for testing
                cache_compression=True,
            )

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._offline_mode_active = True

                # Cache multiple entries
                for i in range(10):
                    integration._cache_log_entry(
                        {"step": i, "data": f"test_data_{i}"}, step=i
                    )

                # Check cache was created
                assert len(integration._cached_entries) > 0
                cache_dir = Path(temp_dir)
                assert cache_dir.exists()
                assert (cache_dir / "cache_metadata.json").exists()

                # Check cache files exist
                for entry in integration._cached_entries:
                    cache_file = cache_dir / entry["filename"]
                    assert cache_file.exists()

    def test_cache_cleanup_on_size_limit(self):
        """Test cache cleanup when size limit is exceeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(
                enabled=True,
                offline_cache_dir=temp_dir,
                max_cache_size_gb=0.000001,  # Extremely small limit
                cache_compression=False,
            )

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._offline_mode_active = True

                # Cache many entries to exceed limit
                initial_count = 20
                for i in range(initial_count):
                    large_data = {
                        "step": i,
                        "large_data": "x" * 1000,
                    }  # Large data to trigger cleanup
                    integration._cache_log_entry(large_data, step=i)

                # Should have fewer entries due to cleanup
                assert len(integration._cached_entries) < initial_count

    def test_offline_data_sync(self):
        """Test syncing of cached offline data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(
                enabled=True, offline_cache_dir=temp_dir, sync_batch_size=3
            )

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._wandb_available = True
                integration.run = Mock()
                integration.run.id = "test-run-id"  # Add run ID for caching
                integration._offline_mode_active = True

                # Cache some entries manually to avoid Mock serialization issues
                test_data = [
                    {"step": 1, "metric": 0.5},
                    {"step": 2, "metric": 0.7},
                    {"step": 3, "metric": 0.9},
                ]

                # Manually create cache entries to avoid Mock serialization
                for i, data in enumerate(test_data):
                    cache_entry = {
                        "timestamp": time.time(),
                        "data": data,
                        "step": i + 1,
                        "run_id": "test-run-id",
                    }

                    # Save to cache file
                    cache_filename = f"entry_{i:06d}.json"
                    cache_file = Path(temp_dir) / cache_filename

                    with open(cache_file, "w") as f:
                        json.dump(cache_entry, f)

                    # Add to metadata
                    integration._cached_entries.append(
                        {
                            "filename": cache_filename,
                            "timestamp": cache_entry["timestamp"],
                            "step": i + 1,
                            "run_id": "test-run-id",
                        }
                    )

                # Save metadata
                integration._save_cache_metadata()

                # Sync the data
                result = integration.sync_offline_data(force=True)

                # Check sync results
                assert result["success"] is True
                assert result["synced_count"] == len(test_data)
                assert result["failed_count"] == 0

                # Check wandb.log was called for each entry
                assert integration.run.log.call_count == len(test_data)

                # Check offline mode was reset
                assert integration._offline_mode_active is False

    def test_sync_with_failures(self):
        """Test sync behavior when some entries fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(enabled=True, offline_cache_dir=temp_dir)

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._wandb_available = True
                integration.run = Mock()
                integration.run.id = "test-run-id"
                integration._offline_mode_active = True

                # Manually create cache entries to avoid Mock serialization
                test_data = [
                    {"step": 0, "metric": 0.1},
                    {"step": 1, "metric": 0.2},
                    {"step": 2, "metric": 0.3},
                ]

                for i, data in enumerate(test_data):
                    cache_entry = {
                        "timestamp": time.time(),
                        "data": data,
                        "step": i,
                        "run_id": "test-run-id",
                    }

                    # Save to cache file
                    cache_filename = f"entry_{i:06d}.json"
                    cache_file = Path(temp_dir) / cache_filename

                    with open(cache_file, "w") as f:
                        json.dump(cache_entry, f)

                    # Add to metadata
                    integration._cached_entries.append(
                        {
                            "filename": cache_filename,
                            "timestamp": cache_entry["timestamp"],
                            "step": i,
                            "run_id": "test-run-id",
                        }
                    )

                # Save metadata
                integration._save_cache_metadata()

                # Mock partial failure during sync
                integration.run.log.side_effect = [
                    None,  # Success
                    Exception("Sync error"),  # Failure
                    None,  # Success
                ]

                result = integration.sync_offline_data(force=True)

                # Should report partial success
                assert result["success"] is False
                assert result["synced_count"] == 2
                assert result["failed_count"] == 1
                assert len(result["errors"]) == 1

    def test_offline_status_reporting(self):
        """Test offline status and cache information reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(enabled=True, offline_cache_dir=temp_dir)

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._offline_mode_active = True

                # Cache some data
                integration._cache_log_entry({"test": "data"}, step=1)

                # Get offline status
                status = integration.get_offline_status()

                assert status["offline_mode_active"] is True
                assert status["cached_entries_count"] == 1
                assert status["cache_size_bytes"] > 0
                assert status["cache_directory"] == temp_dir

                # Get sync status
                sync_status = integration.get_sync_status()
                assert "auto_sync_enabled" in sync_status
                assert "wandb_available" in sync_status

    def test_force_offline_online_modes(self):
        """Test manual switching between offline and online modes."""
        config = WandbConfig(enabled=True, auto_sync_on_reconnect=True)

        with patch(
            "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
        ):
            integration = WandbIntegration(config)
            integration._wandb_available = True
            integration.run = Mock()

            # Test force offline
            integration.force_offline_mode()
            assert integration._offline_mode_active is True

            # Test force online (should attempt sync)
            with patch.object(integration, "sync_offline_data") as mock_sync:
                mock_sync.return_value = {"success": True, "synced_count": 0}
                integration.force_online_mode()
                assert integration._offline_mode_active is False

    def test_clear_offline_cache(self):
        """Test clearing of offline cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WandbConfig(enabled=True, offline_cache_dir=temp_dir)

            with patch(
                "src.jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
            ):
                integration = WandbIntegration(config)
                integration._offline_mode_active = True

                # Cache some data
                integration._cache_log_entry({"test": "data"}, step=1)
                assert len(integration._cached_entries) == 1

                # Clear cache without confirmation
                result = integration.clear_offline_cache(confirm=False)
                assert result is False
                assert len(integration._cached_entries) == 1

                # Clear cache with confirmation
                result = integration.clear_offline_cache(confirm=True)
                assert result is True
                assert len(integration._cached_entries) == 0


class TestWandbSyncManager:
    """Test wandb sync manager functionality."""

    def test_sync_manager_creation(self):
        """Test creation of sync manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = WandbSyncManager(Path(temp_dir), "test-project")

            assert sync_manager.cache_dir == Path(temp_dir)
            assert sync_manager.project_name == "test-project"

    def test_cache_status_reporting(self):
        """Test cache status reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = WandbSyncManager(Path(temp_dir), "test-project")

            # Test with empty cache
            status = sync_manager.get_cache_status()
            assert status["cache_exists"] is True
            assert status["total_size_mb"] == 0
            assert status["offline_runs"] == 0

            # Create some test files
            test_file = Path(temp_dir) / "test.json"
            test_file.write_text('{"test": "data"}')

            status = sync_manager.get_cache_status()
            assert status["total_size_mb"] > 0

    def test_find_offline_runs(self):
        """Test finding offline wandb runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = WandbSyncManager(Path(temp_dir), "test-project")

            # Create mock wandb run directories
            run1_dir = Path(temp_dir) / "run1"
            run1_dir.mkdir()
            (run1_dir / "wandb-metadata.json").write_text('{"test": "metadata"}')

            run2_dir = Path(temp_dir) / "run2"
            run2_dir.mkdir()
            (run2_dir / "files").mkdir()

            # Find runs
            offline_runs = sync_manager._find_offline_runs()
            assert len(offline_runs) == 2
            assert run1_dir in offline_runs
            assert run2_dir in offline_runs

    @patch("subprocess.run")
    def test_wandb_connectivity_check(self, mock_run):
        """Test wandb connectivity checking."""
        sync_manager = create_sync_manager()

        # Test successful connectivity
        mock_run.side_effect = [
            Mock(returncode=0, stdout="wandb 0.12.0"),  # version check
            Mock(returncode=0, stdout="Logged in"),  # status check
        ]

        is_connected, error = sync_manager.check_wandb_connectivity()
        assert is_connected is True
        assert error is None

        # Test failed connectivity
        mock_run.side_effect = [
            Mock(returncode=0, stdout="wandb 0.12.0"),  # version check
            Mock(returncode=1, stderr="Not logged in"),  # status check fails
        ]

        is_connected, error = sync_manager.check_wandb_connectivity()
        assert is_connected is False
        assert "Not logged in" in error

    @patch("subprocess.run")
    def test_single_run_sync(self, mock_run):
        """Test syncing a single offline run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = WandbSyncManager(Path(temp_dir), "test-project")

            # Create mock run directory
            run_dir = Path(temp_dir) / "test_run"
            run_dir.mkdir()
            (run_dir / "wandb-metadata.json").write_text('{"test": "metadata"}')

            # Test successful sync
            mock_run.return_value = Mock(returncode=0, stdout="Sync successful")

            result = sync_manager._sync_single_run(run_dir, entity="test-entity")
            assert result is True

            # Verify correct command was called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "wandb" in args
            assert "sync" in args
            assert str(run_dir) in args
            assert "--entity" in args
            assert "test-entity" in args
            assert "--project" in args
            assert "test-project" in args

    def test_cleanup_synced_data(self):
        """Test cleanup of synced data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = WandbSyncManager(Path(temp_dir), "test-project")

            # Create mock old run directory
            old_run_dir = Path(temp_dir) / "old_run"
            old_run_dir.mkdir()
            (old_run_dir / "wandb-metadata.json").write_text('{"test": "metadata"}')

            # Make it appear old
            import os

            old_time = time.time() - (25 * 3600)  # 25 hours ago
            os.utime(old_run_dir, (old_time, old_time))

            # Test cleanup without confirmation
            result = sync_manager.cleanup_synced_data(confirm=False)
            assert result["success"] is False
            assert old_run_dir.exists()

            # Test cleanup with confirmation
            result = sync_manager.cleanup_synced_data(confirm=True)
            assert result["success"] is True
            assert result["cleaned_files"] == 1
            assert not old_run_dir.exists()


class TestWandbConfigValidation:
    """Test wandb configuration validation."""

    def test_valid_config_creation(self):
        """Test creation of valid wandb config."""
        config = WandbConfig(
            enabled=True,
            project_name="test-project",
            image_format="png",
            log_frequency=10,
            max_image_size=(800, 600),
        )

        assert config.enabled is True
        assert config.project_name == "test-project"
        assert config.image_format == "png"

    def test_invalid_image_format(self):
        """Test validation of invalid image format."""
        with pytest.raises(ValueError, match="Invalid image_format"):
            WandbConfig(image_format="invalid")

    def test_invalid_log_frequency(self):
        """Test validation of invalid log frequency."""
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            WandbConfig(log_frequency=0)

    def test_invalid_image_size(self):
        """Test validation of invalid image size."""
        with pytest.raises(ValueError, match="max_image_size must be tuple"):
            WandbConfig(max_image_size=(800,))  # Only one dimension

        with pytest.raises(ValueError, match="max_image_size must be tuple"):
            WandbConfig(max_image_size=(0, 600))  # Zero dimension


if __name__ == "__main__":
    pytest.main([__file__])
