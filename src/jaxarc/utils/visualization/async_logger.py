"""Asynchronous logging system for visualization data.

This module provides asynchronous logging capabilities to minimize JAX performance
impact during visualization and logging operations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, List, Optional

import chex
from loguru import logger


@chex.dataclass
class AsyncLoggerConfig:
    """Configuration for asynchronous logging."""

    queue_size: int = 1000
    worker_threads: int = 2
    batch_size: int = 10
    flush_interval: float = 5.0  # seconds
    enable_compression: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds


@chex.dataclass
class LogEntry:
    """Base log entry with priority and metadata."""

    priority: int = 0  # Lower numbers = higher priority
    timestamp: float = 0.0
    entry_type: str = "generic"
    data: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())
        if self.data is None:
            object.__setattr__(self, "data", {})


class AsyncLogger:
    """Asynchronous logger for visualization data.

    This logger uses a priority queue and worker threads to handle logging
    operations without blocking the main JAX computation thread.
    """

    def __init__(self, config: AsyncLoggerConfig, output_dir: Optional[Path] = None):
        """Initialize the async logger.

        Args:
            config: Configuration for the async logger
            output_dir: Base directory for log outputs
        """
        self.config = config
        self.output_dir = output_dir or Path("outputs/logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe queue for log entries
        self.queue: Queue[LogEntry] = Queue(maxsize=config.queue_size)

        # Worker threads and control
        self.workers: List[Thread] = []
        self.shutdown_event = Event()
        self.flush_event = Event()

        # Statistics
        self.stats = {
            "entries_queued": 0,
            "entries_processed": 0,
            "entries_failed": 0,
            "queue_full_count": 0,
        }

        # Start worker threads
        self._start_workers()

        logger.info(f"AsyncLogger initialized with {config.worker_threads} workers")

    def _start_workers(self) -> None:
        """Start worker threads for processing log entries."""
        for i in range(self.config.worker_threads):
            worker = Thread(
                target=self._worker_loop, name=f"AsyncLogger-Worker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self) -> None:
        """Main worker loop for processing log entries."""
        batch: List[LogEntry] = []
        last_flush = time.time()

        while not self.shutdown_event.is_set():
            try:
                # Try to get entries for batch processing
                timeout = max(
                    0.1, self.config.flush_interval - (time.time() - last_flush)
                )

                try:
                    entry = self.queue.get(timeout=timeout)
                    batch.append(entry)
                    self.queue.task_done()
                except Empty:
                    pass

                # Process batch if it's full or flush interval reached
                should_flush = (
                    len(batch) >= self.config.batch_size
                    or (
                        batch and time.time() - last_flush >= self.config.flush_interval
                    )
                    or self.flush_event.is_set()
                )

                if should_flush and batch:
                    self._process_batch(batch)
                    batch.clear()
                    last_flush = time.time()

                    if self.flush_event.is_set():
                        self.flush_event.clear()

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self.stats["entries_failed"] += len(batch)
                batch.clear()

    def _process_batch(self, batch: List[LogEntry]) -> None:
        """Process a batch of log entries.

        Args:
            batch: List of log entries to process
        """
        # Sort batch by priority (lower number = higher priority)
        batch.sort(key=lambda x: x.priority)

        for entry in batch:
            try:
                self._process_single_entry(entry)
                self.stats["entries_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process log entry: {e}")
                self.stats["entries_failed"] += 1

    def _process_single_entry(self, entry: LogEntry) -> None:
        """Process a single log entry.

        Args:
            entry: Log entry to process
        """
        # Create entry-type specific directory
        entry_dir = self.output_dir / entry.entry_type
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(entry.timestamp))
        filename = f"{timestamp_str}_{entry.priority:03d}.json"
        filepath = entry_dir / filename

        # Prepare data for serialization
        log_data = {
            "timestamp": entry.timestamp,
            "priority": entry.priority,
            "entry_type": entry.entry_type,
            "data": entry.data,
        }

        # Write to file with retries
        for attempt in range(self.config.max_retries):
            try:
                with open(filepath, "w") as f:
                    json.dump(log_data, f, indent=2, default=self._json_serializer)
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                time.sleep(self.config.retry_delay)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation of the object
        """
        if hasattr(obj, "tolist"):  # JAX/NumPy arrays
            return obj.tolist()
        if hasattr(obj, "__dict__"):  # Custom objects
            return obj.__dict__
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    def log_entry(
        self, entry_type: str, data: Dict[str, Any], priority: int = 0
    ) -> bool:
        """Queue a log entry for async processing.

        Args:
            entry_type: Type of log entry (e.g., "step", "episode")
            data: Data to log
            priority: Priority level (lower = higher priority)

        Returns:
            True if entry was queued successfully, False if queue is full
        """
        entry = LogEntry(priority=priority, entry_type=entry_type, data=data)

        try:
            self.queue.put_nowait(entry)
            self.stats["entries_queued"] += 1
            return True
        except:
            self.stats["queue_full_count"] += 1
            logger.warning(f"Async logger queue full, dropping {entry_type} entry")
            return False

    def log_step_visualization(
        self, step_data: Dict[str, Any], priority: int = 0
    ) -> bool:
        """Queue step visualization data for async processing.

        Args:
            step_data: Step visualization data
            priority: Priority level

        Returns:
            True if queued successfully
        """
        return self.log_entry("step_visualization", step_data, priority)

    def log_episode_summary(
        self, episode_data: Dict[str, Any], priority: int = 0
    ) -> bool:
        """Queue episode summary data for async processing.

        Args:
            episode_data: Episode summary data
            priority: Priority level

        Returns:
            True if queued successfully
        """
        return self.log_entry("episode_summary", episode_data, priority)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Force flush all pending logs.

        Args:
            timeout: Maximum time to wait for flush completion

        Returns:
            True if flush completed successfully
        """
        if self.queue.empty():
            return True

        # Signal workers to flush
        self.flush_event.set()

        # Wait for queue to be empty
        start_time = time.time()
        while not self.queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(
                    "Flush timeout reached, some entries may not be processed"
                )
                return False
            time.sleep(0.1)

        # Wait for all tasks to be done
        try:
            if timeout:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout > 0:
                    self.queue.join()  # This might not respect timeout perfectly
            else:
                self.queue.join()
            return True
        except:
            return False

    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shutdown the async logger.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down AsyncLogger...")

        # First try to flush remaining entries
        self.flush(timeout=timeout / 2)

        # Signal shutdown to workers
        self.shutdown_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(
                timeout=timeout / len(self.workers) if self.workers else timeout
            )
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not shutdown gracefully")

        logger.info(f"AsyncLogger shutdown complete. Stats: {self.stats}")

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary with logging statistics
        """
        stats = self.stats.copy()
        stats["queue_size"] = self.queue.qsize()
        stats["workers_alive"] = sum(1 for w in self.workers if w.is_alive())
        return stats

    def is_healthy(self) -> bool:
        """Check if the logger is healthy and operational.

        Returns:
            True if logger is healthy
        """
        return (
            not self.shutdown_event.is_set()
            and all(w.is_alive() for w in self.workers)
            and self.queue.qsize() < self.config.queue_size * 0.9  # Not too full
        )


# Context manager for automatic cleanup
class AsyncLoggerContext:
    """Context manager for AsyncLogger with automatic cleanup."""

    def __init__(self, config: AsyncLoggerConfig, output_dir: Optional[Path] = None):
        self.config = config
        self.output_dir = output_dir
        self.logger: Optional[AsyncLogger] = None

    def __enter__(self) -> AsyncLogger:
        self.logger = AsyncLogger(self.config, self.output_dir)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.shutdown()
