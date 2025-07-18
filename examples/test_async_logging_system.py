#!/usr/bin/env python3
"""Test script for the asynchronous logging system implementation."""

import time
import tempfile
from pathlib import Path

# Test the async logger
from src.jaxarc.utils.visualization.async_logger import AsyncLogger, AsyncLoggerConfig

# Test the structured logger
from src.jaxarc.utils.logging.structured_logger import (
    StructuredLogger, 
    LoggingConfig,
    StepLogEntry,
    EpisodeLogEntry
)

# Test the performance monitor
from src.jaxarc.utils.logging.performance_monitor import (
    PerformanceMonitor,
    PerformanceConfig,
    monitor_performance
)


def test_async_logger():
    """Test the AsyncLogger functionality."""
    print("Testing AsyncLogger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AsyncLoggerConfig(
            queue_size=100,
            worker_threads=1,
            batch_size=5,
            flush_interval=1.0
        )
        
        logger = AsyncLogger(config, Path(temp_dir))
        
        # Test logging entries
        for i in range(10):
            success = logger.log_entry(
                "test_entry",
                {"step": i, "value": i * 2},
                priority=i % 3
            )
            assert success, f"Failed to log entry {i}"
        
        # Test step visualization logging
        step_data = {
            "step_num": 1,
            "before_grid": [[0, 1], [1, 0]],
            "after_grid": [[1, 0], [0, 1]],
            "action": {"type": "flip"},
            "reward": 1.0
        }
        success = logger.log_step_visualization(step_data)
        assert success, "Failed to log step visualization"
        
        # Test episode summary logging
        episode_data = {
            "episode_num": 1,
            "total_steps": 10,
            "total_reward": 5.0,
            "final_similarity": 0.8
        }
        success = logger.log_episode_summary(episode_data)
        assert success, "Failed to log episode summary"
        
        # Test flush and stats
        logger.flush(timeout=5.0)
        stats = logger.get_stats()
        print(f"Logger stats: {stats}")
        
        assert stats["entries_queued"] >= 12, "Not all entries were queued"
        assert logger.is_healthy(), "Logger is not healthy"
        
        logger.shutdown()
        print("AsyncLogger test passed!")


def test_structured_logger():
    """Test the StructuredLogger functionality."""
    print("Testing StructuredLogger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggingConfig(
            output_dir=temp_dir,
            async_logging=False,  # Use sync for testing
            compression=False
        )
        
        logger = StructuredLogger(config)
        
        # Test episode lifecycle
        logger.start_episode(
            episode_num=1,
            task_id="test_task_001",
            config_hash="abc123",
            metadata={"test": True}
        )
        
        # Log some steps
        for step in range(5):
            logger.log_step(
                step_num=step,
                before_state={"grid": [[0, 1], [1, 0]], "step_count": step},
                action={"type": "flip", "params": {}},
                after_state={"grid": [[1, 0], [0, 1]], "step_count": step + 1},
                reward=0.2,
                info={"similarity": 0.5 + step * 0.1}
            )
        
        logger.end_episode()
        
        # Test loading episode
        loaded_episode = logger.load_episode(1)
        assert loaded_episode is not None, "Failed to load episode"
        assert loaded_episode.episode_num == 1, "Wrong episode number"
        assert loaded_episode.total_steps == 5, "Wrong step count"
        assert len(loaded_episode.steps) == 5, "Wrong number of steps"
        
        # Test episode listing
        episodes = logger.list_episodes()
        assert 1 in episodes, "Episode 1 not found in list"
        
        # Test episode summary
        summary = logger.get_episode_summary(1)
        assert summary is not None, "Failed to get episode summary"
        assert summary["total_steps"] == 5, "Wrong step count in summary"
        
        logger.shutdown()
        print("StructuredLogger test passed!")


def test_performance_monitor():
    """Test the PerformanceMonitor functionality."""
    print("Testing PerformanceMonitor...")
    
    config = PerformanceConfig(
        enabled=True,
        max_samples=100,
        alert_threshold=0.1,
        adaptive_logging=True
    )
    
    monitor = PerformanceMonitor(config)
    
    # Test function monitoring
    @monitor_performance(monitor, "test_function")
    def test_function(duration=0.01):
        time.sleep(duration)
        return "result"
    
    # Run function multiple times
    for i in range(10):
        result = test_function(0.001)  # 1ms sleep
        assert result == "result", "Function result incorrect"
    
    # Test visualization impact monitoring
    def main_computation():
        time.sleep(0.005)  # 5ms
        return "computed"
    
    def visualization():
        time.sleep(0.001)  # 1ms
        return "visualized"
    
    combined_func = monitor.measure_visualization_impact(
        main_computation,
        visualization,
        "test_combined"
    )
    
    # Run combined function
    for i in range(5):
        result = combined_func()
        assert result == "computed", "Combined function result incorrect"
    
    # Test performance report
    report = monitor.get_performance_report()
    print(f"Performance report: {report}")
    
    assert "test_function" in report["functions"], "test_function not in report"
    assert "test_combined" in report["functions"], "test_combined not in report"
    
    # Test function stats
    stats = monitor.get_function_stats("test_function")
    assert stats is not None, "Failed to get function stats"
    assert stats["count"] > 0, "No samples recorded"
    
    # Test should_log functionality
    should_log = monitor.should_log()
    assert isinstance(should_log, bool), "should_log should return boolean"
    
    print("PerformanceMonitor test passed!")


def main():
    """Run all tests."""
    print("Testing asynchronous logging system implementation...")
    print("=" * 60)
    
    try:
        test_async_logger()
        print()
        test_structured_logger()
        print()
        test_performance_monitor()
        print()
        print("=" * 60)
        print("All tests passed! ✅")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())