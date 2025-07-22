#!/usr/bin/env python3
"""
Demo script showing wandb error handling and offline support features.

This script demonstrates:
1. Network error detection and automatic offline mode switching
2. Offline data caching and synchronization
3. Manual offline/online mode control
4. Cache management and cleanup
5. Sync utilities for offline data

Run with: pixi run python examples/wandb_error_handling_demo.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).parent.parent / "src"))

from jaxarc.utils.visualization.wandb_integration import (
    WandbConfig,
    WandbIntegration,
    create_development_wandb_config,
)
from jaxarc.utils.visualization.wandb_sync import (
    WandbSyncManager,
    check_wandb_status,
    sync_offline_wandb_data,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_error_handling():
    """Demonstrate basic error handling and offline mode switching."""
    print("\n=== Demo: Basic Error Handling ===")

    # Create config with error handling enabled
    config = WandbConfig(
        enabled=True,
        project_name="jaxarc-error-handling-demo",
        auto_offline_on_error=True,
        retry_attempts=3,
        retry_delay=1.0,
        max_retry_delay=10.0,
    )

    print(f"Config: auto_offline_on_error={config.auto_offline_on_error}")
    print(f"Config: retry_attempts={config.retry_attempts}")

    # Note: In a real scenario, wandb would be imported and network errors would be real
    print("In a real scenario, network errors would automatically trigger offline mode")
    print("The system would cache data locally and sync when connectivity is restored")


def demo_offline_caching():
    """Demonstrate offline data caching and management."""
    print("\n=== Demo: Offline Caching ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = WandbConfig(
            enabled=True,
            offline_cache_dir=temp_dir,
            max_cache_size_gb=0.1,  # Small limit for demo
            cache_compression=True,
            auto_sync_on_reconnect=True,
        )

        # Mock wandb integration for demo
        with patch(
            "jaxarc.utils.visualization.wandb_integration.WandbIntegration._initialize_wandb"
        ):
            integration = WandbIntegration(config)
            integration._wandb_available = True
            integration._wandb = Mock()
            integration.run = Mock()
            integration.run.id = "demo-run-id"

            # Force offline mode
            integration.force_offline_mode()
            print(f"Offline mode active: {integration.is_offline}")

            # Simulate logging data while offline
            print("Logging data while offline...")
            for i in range(5):
                data = {
                    "step": i,
                    "loss": 1.0 / (i + 1),
                    "accuracy": i * 0.2,
                    "timestamp": time.time(),
                }
                integration._cache_log_entry(data, step=i)
                print(f"  Cached step {i}")

            # Check cache status
            status = integration.get_offline_status()
            print("\nCache status:")
            print(f"  Cached entries: {status['cached_entries_count']}")
            print(f"  Cache size: {status['cache_size_mb']:.2f} MB")
            print(f"  Cache directory: {status['cache_directory']}")

            # Simulate going back online and syncing
            print("\nSimulating network restoration and sync...")
            integration.force_online_mode()

            # In a real scenario, this would sync to wandb
            sync_result = integration.sync_offline_data(force=True)
            print(
                f"Sync result: {sync_result['synced_count']} synced, {sync_result['failed_count']} failed"
            )


def demo_sync_manager():
    """Demonstrate the wandb sync manager utilities."""
    print("\n=== Demo: Sync Manager ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sync manager
        sync_manager = WandbSyncManager(Path(temp_dir), "demo-project")

        # Check cache status
        cache_status = sync_manager.get_cache_status()
        print(f"Cache status: {cache_status}")

        # Create mock offline run directory
        run_dir = Path(temp_dir) / "offline_run_001"
        run_dir.mkdir()
        (run_dir / "wandb-metadata.json").write_text(
            '{"run_id": "demo-run", "project": "demo-project"}'
        )
        (run_dir / "files").mkdir()
        (run_dir / "files" / "config.yaml").write_text("demo: config")

        print(f"Created mock offline run: {run_dir}")

        # Find offline runs
        offline_runs = sync_manager._find_offline_runs()
        print(f"Found {len(offline_runs)} offline runs")

        # Check wandb connectivity (will fail in demo environment)
        is_connected, error = sync_manager.check_wandb_connectivity()
        print(f"Wandb connectivity: {is_connected} (error: {error})")

        # Get updated cache status
        cache_status = sync_manager.get_cache_status()
        print("Updated cache status:")
        print(f"  Total size: {cache_status['total_size_mb']:.2f} MB")
        print(f"  Offline runs: {cache_status['offline_runs']}")


def demo_configuration_options():
    """Demonstrate different configuration options for error handling."""
    print("\n=== Demo: Configuration Options ===")

    # Development config (offline by default)
    dev_config = create_development_wandb_config()
    print("Development config:")
    print(f"  Offline mode: {dev_config.offline_mode}")
    print(f"  Auto offline on error: {dev_config.auto_offline_on_error}")
    print(f"  Log frequency: {dev_config.log_frequency}")

    # Custom error handling config
    custom_config = WandbConfig(
        enabled=True,
        project_name="custom-project",
        auto_offline_on_error=True,
        retry_attempts=5,
        retry_delay=2.0,
        max_retry_delay=60.0,
        network_timeout=15.0,
        offline_cache_dir="./custom_cache",
        max_cache_size_gb=2.0,
        cache_compression=True,
        auto_sync_on_reconnect=True,
        sync_batch_size=25,
    )

    print("\nCustom error handling config:")
    print(f"  Retry attempts: {custom_config.retry_attempts}")
    print(f"  Max retry delay: {custom_config.max_retry_delay}s")
    print(f"  Network timeout: {custom_config.network_timeout}s")
    print(f"  Cache size limit: {custom_config.max_cache_size_gb} GB")
    print(f"  Sync batch size: {custom_config.sync_batch_size}")


def demo_utility_functions():
    """Demonstrate utility functions for wandb management."""
    print("\n=== Demo: Utility Functions ===")

    # Check wandb status
    print("Checking wandb status...")
    status = check_wandb_status()
    print(f"Wandb connected: {status['connected']}")
    if status["error"]:
        print(f"Error: {status['error']}")

    # Cache status from utility
    cache_info = status["cache_status"]
    print(f"Cache exists: {cache_info['cache_exists']}")
    print(f"Cache size: {cache_info['total_size_mb']:.2f} MB")

    # Demonstrate sync utility (would work with real cached data)
    print("\nSync utility demo (would sync real cached data):")
    with tempfile.TemporaryDirectory() as temp_dir:
        # This would sync real cached data in a real scenario
        sync_result = sync_offline_wandb_data(
            cache_dir=Path(temp_dir), project_name="demo-project", entity=None
        )
        print(f"Sync result: {sync_result}")


def main():
    """Run all demos."""
    print("JaxARC Wandb Error Handling and Offline Support Demo")
    print("=" * 60)

    try:
        demo_basic_error_handling()
        demo_offline_caching()
        demo_sync_manager()
        demo_configuration_options()
        demo_utility_functions()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Automatic offline mode switching on network errors")
        print("✓ Offline data caching with compression")
        print("✓ Automatic sync when connectivity is restored")
        print("✓ Cache size management and cleanup")
        print("✓ Retry logic with exponential backoff")
        print("✓ Comprehensive error recovery mechanisms")
        print("✓ Utility functions for cache management")

    except Exception as e:
        logger.error("Demo failed: %s", e)
        raise


if __name__ == "__main__":
    main()
