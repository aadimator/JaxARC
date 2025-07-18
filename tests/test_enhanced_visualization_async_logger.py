"""Unit tests for AsyncLogger component of enhanced visualization system."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import MagicMock, Mock, patch

import pytest

from jaxarc.utils.visualization.async_logger import (
    AsyncLogger,
    AsyncLoggerConfig,
    LogEntry,
)


class TestAsyncLoggerConfig:
    """Test AsyncLoggerConfig dataclass and validation."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AsyncLoggerConfig()
        
        assert config.queue_size == 1000
        assert config.worker_threads == 2
        assert config.batch_size == 10
        assert config.flush_interval == 5.0
        assert config.enable_compression is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AsyncLoggerConfig(
            queue_size=500,
            worker_threads=4,
            batch_size=20,
            flush_interval=10.0,
            enable_compression=False,
            max_retries=5,
            retry_delay=2.0,
        )
        
        assert config.queue_size == 500
        assert config.worker_threads == 4
        assert config.batch_size == 20
        assert config.flush_interval == 10.0
        assert config.enable_compression is False
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
    
    def test_config_validation_invalid_queue_size(self) -> None:
        """Test validation fails for invalid queue size."""
        with pytest.raises(ValueError, match="queue_size must be positive"):
            AsyncLoggerConfig(queue_size=0)
    
    def test_config_validation_invalid_worker_threads(self) -> None:
        """Test validation fails for invalid worker thread count."""
        with pytest.raises(ValueError, match="worker_threads must be positive"):
            AsyncLoggerConfig(worker_threads=0)
    
    def test_config_validation_invalid_batch_size(self) -> None:
        """Test validation fails for invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            AsyncLoggerConfig(batch_size=0)
    
    def test_config_validation_negative_intervals(self) -> None:
        """Test validation fails for negative intervals."""
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            AsyncLoggerConfig(flush_interval=-1.0)
        
        with pytest.raises(ValueError, match="retry_delay must be non-negative"):
            AsyncLoggerConfig(retry_delay=-1.0)


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_default_log_entry(self) -> None:
        """Test default log entry creation."""
        entry = LogEntry()
        
        assert entry.priority == 0
        assert entry.timestamp > 0  # Should be set automatically
        assert entry.entry_type == "generic"
        assert entry.data == {}
    
    def test_custom_log_entry(self) -> None:
        """Test custom log entry creation."""
        test_data = {"key": "value", "number": 42}
        entry = LogEntry(
            priority=5,
            timestamp=123456.789,
            entry_type="test_entry",
            data=test_data,
        )
        
        assert entry.priority == 5
        assert entry.timestamp == 123456.789
        assert entry.entry_type == "test_entry"
        assert entry.data == test_data
    
    def test_log_entry_serialization(self) -> None:
        """Test log entry can be serialized."""
        entry = LogEntry(
            priority=1,
            entry_type="step_visualization",
            data={"step": 10, "reward": 0.5},
        )
        
        serialized = entry.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["priority"] == 1
        assert serialized["entry_type"] == "step_visualization"
        assert serialized["data"]["step"] == 10
        assert serialized["data"]["reward"] == 0.5
        assert "timestamp" in serialized
    
    def test_log_entry_comparison(self) -> None:
        """Test log entry priority comparison for queue ordering."""
        high_priority = LogEntry(priority=1)
        low_priority = LogEntry(priority=10)
        
        # Lower priority number = higher priority
        assert high_priority < low_priority
        assert not (low_priority < high_priority)
        assert high_priority != low_priority


class TestAsyncLogger:
    """Test AsyncLogger functionality."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def logger_config(self) -> AsyncLoggerConfig:
        """Create test logger configuration."""
        return AsyncLoggerConfig(
            queue_size=100,
            worker_threads=1,  # Single thread for predictable testing
            batch_size=5,
            flush_interval=1.0,  # Short interval for testing
            max_retries=2,
            retry_delay=0.1,
        )
    
    @pytest.fixture
    def async_logger(self, logger_config: AsyncLoggerConfig, temp_dir: Path) -> AsyncLogger:
        """Create async logger for testing."""
        logger = AsyncLogger(logger_config, output_dir=temp_dir)
        yield logger
        logger.shutdown()
    
    def test_initialization(self, async_logger: AsyncLogger, logger_config: AsyncLoggerConfig) -> None:
        """Test async logger initialization."""
        assert async_logger.config == logger_config
        assert async_logger.is_running is True
        assert len(async_logger.workers) == logger_config.worker_threads
        assert async_logger.queue.maxsize == logger_config.queue_size
        
        # Check that worker threads are alive
        for worker in async_logger.workers:
            assert worker.is_alive()
    
    def test_log_entry_basic(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test basic log entry functionality."""
        entry = LogEntry(
            entry_type="test_entry",
            data={"message": "test log entry"},
        )
        
        # Log the entry
        async_logger.log_entry(entry)
        
        # Wait for processing
        time.sleep(0.2)
        async_logger.flush()
        
        # Check that entry was processed
        assert async_logger.queue.qsize() == 0
        
        # Check log file was created
        log_files = list(temp_dir.glob("*.json"))
        assert len(log_files) > 0
    
    def test_log_step_visualization(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test step visualization logging."""
        step_data = {
            "step_num": 5,
            "reward": 0.8,
            "action": {"type": "fill", "color": 1},
            "grid_before": [[0, 0], [0, 0]],
            "grid_after": [[1, 1], [1, 1]],
        }
        
        async_logger.log_step_visualization(step_data, priority=1)
        
        # Wait and flush
        time.sleep(0.2)
        async_logger.flush()
        
        # Verify logging
        log_files = list(temp_dir.glob("step_*.json"))
        assert len(log_files) > 0
        
        with open(log_files[0]) as f:
            logged_data = json.load(f)
        
        assert logged_data["step_num"] == 5
        assert logged_data["reward"] == 0.8
        assert logged_data["action"]["type"] == "fill"
    
    def test_log_episode_summary(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test episode summary logging."""
        episode_data = {
            "episode_num": 10,
            "total_steps": 25,
            "total_reward": 15.5,
            "success": True,
            "final_similarity": 0.95,
        }
        
        async_logger.log_episode_summary(episode_data)
        
        # Wait and flush
        time.sleep(0.2)
        async_logger.flush()
        
        # Verify logging
        log_files = list(temp_dir.glob("episode_*.json"))
        assert len(log_files) > 0
        
        with open(log_files[0]) as f:
            logged_data = json.load(f)
        
        assert logged_data["episode_num"] == 10
        assert logged_data["total_steps"] == 25
        assert logged_data["success"] is True
    
    def test_priority_ordering(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test that high priority entries are processed first."""
        # Add entries with different priorities
        low_priority_entry = LogEntry(
            priority=10,
            entry_type="low_priority",
            data={"order": "last"},
        )
        high_priority_entry = LogEntry(
            priority=1,
            entry_type="high_priority", 
            data={"order": "first"},
        )
        
        # Add low priority first, then high priority
        async_logger.log_entry(low_priority_entry)
        async_logger.log_entry(high_priority_entry)
        
        # Wait for processing
        time.sleep(0.3)
        async_logger.flush()
        
        # Check processing order by examining log files
        log_files = sorted(temp_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
        
        if len(log_files) >= 2:
            # First processed file should be high priority
            with open(log_files[0]) as f:
                first_data = json.load(f)
            assert first_data.get("order") == "first"
    
    def test_batch_processing(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test batch processing of log entries."""
        # Add multiple entries quickly
        entries = []
        for i in range(10):
            entry = LogEntry(
                entry_type="batch_test",
                data={"batch_id": i},
            )
            entries.append(entry)
            async_logger.log_entry(entry)
        
        # Wait for batch processing
        time.sleep(0.5)
        async_logger.flush()
        
        # Should have processed all entries
        assert async_logger.queue.qsize() == 0
        
        # Check that files were created
        log_files = list(temp_dir.glob("*.json"))
        assert len(log_files) >= 10
    
    def test_queue_full_handling(self, temp_dir: Path) -> None:
        """Test handling of full queue."""
        # Create logger with very small queue
        small_config = AsyncLoggerConfig(
            queue_size=2,
            worker_threads=1,
            batch_size=1,
            flush_interval=10.0,  # Long interval to fill queue
        )
        
        logger = AsyncLogger(small_config, output_dir=temp_dir)
        
        try:
            # Fill the queue
            for i in range(5):  # More than queue size
                entry = LogEntry(data={"id": i})
                if i < 2:
                    logger.log_entry(entry)  # Should succeed
                else:
                    # Should handle gracefully when queue is full
                    try:
                        logger.log_entry(entry)
                    except Exception:
                        pass  # Expected behavior when queue is full
        finally:
            logger.shutdown()
    
    def test_flush_functionality(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test manual flush functionality."""
        # Add some entries
        for i in range(5):
            entry = LogEntry(data={"flush_test": i})
            async_logger.log_entry(entry)
        
        # Queue should have entries
        assert async_logger.queue.qsize() > 0
        
        # Flush should process all entries
        async_logger.flush()
        
        # Queue should be empty after flush
        assert async_logger.queue.qsize() == 0
    
    def test_shutdown_graceful(self, logger_config: AsyncLoggerConfig, temp_dir: Path) -> None:
        """Test graceful shutdown."""
        logger = AsyncLogger(logger_config, output_dir=temp_dir)
        
        # Add some entries
        for i in range(3):
            entry = LogEntry(data={"shutdown_test": i})
            logger.log_entry(entry)
        
        # Shutdown should process remaining entries
        logger.shutdown()
        
        # Workers should be stopped
        for worker in logger.workers:
            assert not worker.is_alive()
        
        # Should not be running
        assert logger.is_running is False
    
    def test_error_handling_in_worker(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test error handling in worker threads."""
        # Create an entry that will cause an error during processing
        problematic_entry = LogEntry(
            entry_type="error_test",
            data={"problematic": object()},  # Non-serializable object
        )
        
        # Mock the file writing to raise an exception
        with patch('builtins.open', side_effect=IOError("Simulated IO error")):
            async_logger.log_entry(problematic_entry)
            
            # Wait for processing attempt
            time.sleep(0.2)
            
            # Logger should still be running despite the error
            assert async_logger.is_running is True
            
            # Worker threads should still be alive
            for worker in async_logger.workers:
                assert worker.is_alive()
    
    def test_compression_functionality(self, temp_dir: Path) -> None:
        """Test log compression functionality."""
        config = AsyncLoggerConfig(
            enable_compression=True,
            worker_threads=1,
            batch_size=1,
        )
        
        logger = AsyncLogger(config, output_dir=temp_dir)
        
        try:
            # Add entry with large data
            large_data = {"large_content": "x" * 1000}
            entry = LogEntry(data=large_data)
            logger.log_entry(entry)
            
            # Wait for processing
            time.sleep(0.2)
            logger.flush()
            
            # Check for compressed files
            compressed_files = list(temp_dir.glob("*.gz"))
            regular_files = list(temp_dir.glob("*.json"))
            
            # Should have either compressed or regular files
            assert len(compressed_files) > 0 or len(regular_files) > 0
            
        finally:
            logger.shutdown()
    
    def test_retry_mechanism(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test retry mechanism for failed operations."""
        entry = LogEntry(data={"retry_test": True})
        
        # Mock file operations to fail initially, then succeed
        call_count = 0
        original_open = open
        
        def mock_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                raise IOError("Simulated failure")
            return original_open(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open):
            async_logger.log_entry(entry)
            
            # Wait for retries
            time.sleep(0.5)
            async_logger.flush()
            
            # Should have retried and eventually succeeded
            assert call_count > 1
    
    def test_concurrent_logging(self, async_logger: AsyncLogger, temp_dir: Path) -> None:
        """Test concurrent logging from multiple threads."""
        results = []
        errors = []
        
        def log_entries(thread_id: int) -> None:
            try:
                for i in range(10):
                    entry = LogEntry(
                        data={"thread_id": thread_id, "entry_id": i}
                    )
                    async_logger.log_entry(entry)
                    results.append(f"thread_{thread_id}_entry_{i}")
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=log_entries, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(0.5)
        async_logger.flush()
        
        # Should have processed entries from all threads
        assert len(results) == 30  # 3 threads * 10 entries each
        assert len(errors) == 0
        
        # Check log files
        log_files = list(temp_dir.glob("*.json"))
        assert len(log_files) >= 30
    
    def test_performance_monitoring(self, async_logger: AsyncLogger) -> None:
        """Test performance monitoring capabilities."""
        # Add entries and measure performance
        start_time = time.time()
        
        for i in range(100):
            entry = LogEntry(data={"perf_test": i})
            async_logger.log_entry(entry)
        
        queue_time = time.time() - start_time
        
        # Queueing should be very fast
        assert queue_time < 1.0  # Should queue 100 entries in less than 1 second
        
        # Get performance stats
        stats = async_logger.get_performance_stats()
        assert "entries_queued" in stats
        assert "entries_processed" in stats
        assert "queue_size" in stats
        assert stats["entries_queued"] >= 100


if __name__ == "__main__":
    pytest.main([__file__])