"""Performance and scalability tests for enhanced visualization system."""

from __future__ import annotations

import gc
import psutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxarc.utils.visualization.enhanced_visualizer import EnhancedVisualizer, VisualizationConfig
from jaxarc.utils.visualization.episode_manager import EpisodeManager, EpisodeConfig
from jaxarc.utils.visualization.async_logger import AsyncLogger, AsyncLoggerConfig
from jaxarc.utils.visualization.wandb_integration import WandbIntegration, WandbConfig


class PerformanceProfiler:
    """Helper class for performance profiling."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start profiling."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def end(self):
        """End profiling."""
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, self.end_memory)
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    @property
    def memory_delta(self) -> float:
        """Get memory delta in MB."""
        if self.start_memory and self.end_memory:
            return self.end_memory - self.start_memory
        return 0.0
        
    @property
    def peak_memory_delta(self) -> float:
        """Get peak memory delta in MB."""
        if self.start_memory and self.peak_memory:
            return self.peak_memory - self.start_memory
        return 0.0


class TestLargeEpisodeHandling:
    """Test handling of large episodes (1000+ steps)."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def performance_visualizer(self, temp_dir: Path) -> EnhancedVisualizer:
        """Create visualizer optimized for performance testing."""
        episode_config = EpisodeConfig(
            base_output_dir=str(temp_dir / "episodes"),
            max_episodes_per_run=10,
            cleanup_policy="size_based",
            max_storage_gb=2.0,
        )
        
        async_config = AsyncLoggerConfig(
            queue_size=2000,  # Large queue for performance
            worker_threads=4,
            batch_size=20,
            flush_interval=2.0,
            enable_compression=True,
        )
        
        wandb_config = WandbConfig(enabled=False)  # Disabled for performance
        
        vis_config = VisualizationConfig(
            debug_level="minimal",  # Minimal overhead
            output_formats=["svg"],  # Single format
            show_coordinates=False,
            show_operation_names=False,
            include_metrics=False,
            episode_config=episode_config,
            async_logger_config=async_config,
            wandb_config=wandb_config,
        )
        
        episode_manager = EpisodeManager(episode_config)
        async_logger = AsyncLogger(async_config, output_dir=temp_dir / "logs")
        wandb_integration = WandbIntegration(wandb_config)
        
        visualizer = EnhancedVisualizer(
            config=vis_config,
            episode_manager=episode_manager,
            async_logger=async_logger,
            wandb_integration=wandb_integration,
        )
        
        yield visualizer
        visualizer.shutdown()
    
    def test_large_episode_1000_steps(self, performance_visualizer: EnhancedVisualizer) -> None:
        """Test handling of 1000-step episode."""
        profiler = PerformanceProfiler()
        profiler.start()
        
        performance_visualizer.episode_manager.start_new_run("large_episode_test")
        performance_visualizer.start_episode(episode_num=1, task_id="large_task_001")
        
        # Mock visualization to avoid actual SVG generation overhead
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
            mock_viz.return_value = "<svg>step</svg>"
            
            # Simulate 1000 steps
            for step in range(1000):
                if step % 100 == 0:
                    profiler.update_peak_memory()
                
                before_grid = jnp.ones((10, 10), dtype=jnp.int32) * (step % 10)
                after_grid = jnp.ones((10, 10), dtype=jnp.int32) * ((step + 1) % 10)
                action = {"type": "fill", "color": step % 10, "step": step}
                reward = 0.001 * step  # Small rewards
                
                performance_visualizer.visualize_step(
                    before_grid=before_grid,
                    action=action,
                    after_grid=after_grid,
                    reward=reward,
                    info={"step": step},
                    step_num=step,
                )
        
        # Wait for async processing
        performance_visualizer.async_logger.flush()
        profiler.end()
        
        # Performance assertions
        assert profiler.duration < 30.0, f"1000 steps took too long: {profiler.duration:.2f}s"
        assert profiler.peak_memory_delta < 500.0, f"Memory usage too high: {profiler.peak_memory_delta:.2f}MB"
        
        # Verify all steps were processed
        assert performance_visualizer.step_count == 1000
        
        # Check that visualization was called for appropriate steps
        # With minimal debug level, not all steps may be visualized
        assert mock_viz.call_count <= 1000
        
        # Check performance stats
        stats = performance_visualizer.get_performance_stats()
        assert stats["steps_visualized"] == 1000
        assert stats["avg_step_time"] < 0.1  # Less than 100ms per step on average
    
    def test_multiple_large_episodes(self, performance_visualizer: EnhancedVisualizer) -> None:
        """Test handling of multiple large episodes."""
        profiler = PerformanceProfiler()
        profiler.start()
        
        performance_visualizer.episode_manager.start_new_run("multiple_large_episodes")
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz, \
             patch('jaxarc.utils.visualization.enhanced_visualizer.create_episode_summary_svg') as mock_summary:
            
            mock_viz.return_value = "<svg>step</svg>"
            mock_summary.return_value = "<svg>summary</svg>"
            
            # Run 5 episodes of 500 steps each
            for episode_num in range(1, 6):
                performance_visualizer.start_episode(episode_num=episode_num, task_id=f"large_task_{episode_num:03d}")
                
                episode_rewards = []
                for step in range(500):
                    if step % 50 == 0:
                        profiler.update_peak_memory()
                    
                    before_grid = jnp.zeros((8, 8), dtype=jnp.int32)
                    after_grid = jnp.ones((8, 8), dtype=jnp.int32) * episode_num
                    reward = 0.01 * step * episode_num
                    episode_rewards.append(reward)
                    
                    performance_visualizer.visualize_step(
                        before_grid=before_grid,
                        action={"type": "multi_large", "episode": episode_num, "step": step},
                        after_grid=after_grid,
                        reward=reward,
                        info={"episode": episode_num},
                        step_num=step,
                    )
                
                # Create episode summary
                from jaxarc.utils.visualization.enhanced_visualizer import EpisodeSummaryData
                episode_data = EpisodeSummaryData(
                    episode_num=episode_num,
                    total_steps=500,
                    total_reward=sum(episode_rewards),
                    reward_progression=episode_rewards[-10:],  # Last 10 rewards to save memory
                    final_similarity=0.8 + 0.02 * episode_num,
                    task_id=f"large_task_{episode_num:03d}",
                    success=episode_num > 2,
                )
                
                performance_visualizer.visualize_episode_summary(episode_data)
        
        # Wait for all async processing
        performance_visualizer.async_logger.flush()
        profiler.end()
        
        # Performance assertions for 2500 total steps
        assert profiler.duration < 60.0, f"Multiple large episodes took too long: {profiler.duration:.2f}s"
        assert profiler.peak_memory_delta < 1000.0, f"Memory usage too high: {profiler.peak_memory_delta:.2f}MB"
        
        # Verify processing
        assert performance_visualizer.step_count == 2500  # 5 episodes * 500 steps
        
        # Check episode summaries were created
        assert mock_summary.call_count == 5
        
        # Check storage usage is reasonable
        storage_usage = performance_visualizer.episode_manager.get_storage_usage_gb()
        assert storage_usage < 2.0, f"Storage usage too high: {storage_usage:.2f}GB"
    
    def test_memory_leak_detection(self, performance_visualizer: EnhancedVisualizer) -> None:
        """Test for memory leaks during long-running visualization."""
        performance_visualizer.episode_manager.start_new_run("memory_leak_test")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
            mock_viz.return_value = "<svg>leak test</svg>"
            
            # Run multiple episodes to detect leaks
            for episode_num in range(1, 11):  # 10 episodes
                performance_visualizer.start_episode(episode_num=episode_num, task_id=f"leak_task_{episode_num}")
                
                # 100 steps per episode
                for step in range(100):
                    large_grid = jnp.ones((20, 20), dtype=jnp.int32) * step
                    
                    performance_visualizer.visualize_step(
                        before_grid=large_grid,
                        action={"type": "leak_test", "episode": episode_num, "step": step},
                        after_grid=large_grid + 1,
                        reward=0.01,
                        info={"memory_test": True},
                        step_num=step,
                    )
                
                # Force garbage collection and measure memory
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Flush async logger to prevent queue buildup
                performance_visualizer.async_logger.flush()
        
        # Analyze memory trend
        memory_growth = memory_samples[-1] - memory_samples[0]
        
        # Allow some memory growth but detect significant leaks
        max_acceptable_growth = 200.0  # 200MB max growth
        assert memory_growth < max_acceptable_growth, f"Potential memory leak detected: {memory_growth:.2f}MB growth"
        
        # Check that memory doesn't grow linearly (which would indicate a leak)
        if len(memory_samples) > 5:
            # Calculate correlation between episode number and memory usage
            episodes = list(range(len(memory_samples)))
            correlation = np.corrcoef(episodes, memory_samples)[0, 1]
            
            # Strong positive correlation (> 0.8) might indicate a leak
            assert correlation < 0.8, f"Strong memory growth correlation detected: {correlation:.3f}"


class TestConcurrentEpisodeProcessing:
    """Test concurrent episode processing capabilities."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_concurrent_visualizer(self, temp_dir: Path, visualizer_id: int) -> EnhancedVisualizer:
        """Create visualizer for concurrent testing."""
        episode_config = EpisodeConfig(
            base_output_dir=str(temp_dir / f"concurrent_{visualizer_id}"),
            max_episodes_per_run=20,
        )
        
        async_config = AsyncLoggerConfig(
            queue_size=500,
            worker_threads=2,
            batch_size=10,
        )
        
        wandb_config = WandbConfig(enabled=False)
        
        vis_config = VisualizationConfig(
            debug_level="standard",
            output_formats=["svg"],
            episode_config=episode_config,
            async_logger_config=async_config,
            wandb_config=wandb_config,
        )
        
        episode_manager = EpisodeManager(episode_config)
        async_logger = AsyncLogger(async_config, output_dir=temp_dir / f"logs_{visualizer_id}")
        wandb_integration = WandbIntegration(wandb_config)
        
        return EnhancedVisualizer(
            config=vis_config,
            episode_manager=episode_manager,
            async_logger=async_logger,
            wandb_integration=wandb_integration,
        )
    
    def test_concurrent_visualizers(self, temp_dir: Path) -> None:
        """Test multiple visualizers running concurrently."""
        num_visualizers = 4
        steps_per_episode = 50
        episodes_per_visualizer = 3
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        def run_visualizer(visualizer_id: int) -> Dict[str, Any]:
            """Run a single visualizer."""
            visualizer = self.create_concurrent_visualizer(temp_dir, visualizer_id)
            
            try:
                visualizer.episode_manager.start_new_run(f"concurrent_run_{visualizer_id}")
                
                results = {
                    "visualizer_id": visualizer_id,
                    "episodes_completed": 0,
                    "steps_completed": 0,
                    "errors": [],
                    "start_time": time.time(),
                }
                
                with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
                    mock_viz.return_value = f"<svg>concurrent_{visualizer_id}</svg>"
                    
                    for episode_num in range(1, episodes_per_visualizer + 1):
                        visualizer.start_episode(
                            episode_num=episode_num,
                            task_id=f"concurrent_task_{visualizer_id}_{episode_num}"
                        )
                        
                        for step in range(steps_per_episode):
                            try:
                                before_grid = jnp.ones((5, 5), dtype=jnp.int32) * visualizer_id
                                after_grid = jnp.ones((5, 5), dtype=jnp.int32) * (visualizer_id + step)
                                
                                visualizer.visualize_step(
                                    before_grid=before_grid,
                                    action={"type": "concurrent", "visualizer": visualizer_id, "step": step},
                                    after_grid=after_grid,
                                    reward=0.1 * step,
                                    info={"concurrent": True},
                                    step_num=step,
                                )
                                
                                results["steps_completed"] += 1
                                
                            except Exception as e:
                                results["errors"].append(str(e))
                        
                        results["episodes_completed"] += 1
                
                # Flush and wait
                visualizer.async_logger.flush()
                results["end_time"] = time.time()
                results["duration"] = results["end_time"] - results["start_time"]
                
                return results
                
            finally:
                visualizer.shutdown()
        
        # Run visualizers concurrently
        with ThreadPoolExecutor(max_workers=num_visualizers) as executor:
            futures = [
                executor.submit(run_visualizer, i)
                for i in range(num_visualizers)
            ]
            
            results = []
            for future in as_completed(futures, timeout=60.0):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        profiler.end()
        
        # Analyze results
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) == num_visualizers, f"Only {len(successful_results)}/{num_visualizers} visualizers succeeded"
        
        # Check that all visualizers completed their work
        for result in successful_results:
            assert result["episodes_completed"] == episodes_per_visualizer
            assert result["steps_completed"] == steps_per_episode * episodes_per_visualizer
            assert len(result["errors"]) == 0, f"Visualizer {result['visualizer_id']} had errors: {result['errors']}"
        
        # Check performance
        total_steps = num_visualizers * episodes_per_visualizer * steps_per_episode
        avg_duration = np.mean([r["duration"] for r in successful_results])
        
        assert profiler.duration < 30.0, f"Concurrent processing took too long: {profiler.duration:.2f}s"
        assert avg_duration < 15.0, f"Average visualizer duration too long: {avg_duration:.2f}s"
        
        # Check that concurrent processing was actually faster than sequential
        estimated_sequential_time = avg_duration * num_visualizers
        speedup = estimated_sequential_time / profiler.duration
        assert speedup > 1.5, f"Insufficient speedup from concurrency: {speedup:.2f}x"
        
        # Verify separate outputs
        for i in range(num_visualizers):
            concurrent_dir = temp_dir / f"concurrent_{i}"
            assert concurrent_dir.exists(), f"Output directory for visualizer {i} not found"
            
            run_dirs = list(concurrent_dir.glob("*run_*"))
            assert len(run_dirs) >= 1, f"No run directories found for visualizer {i}"
    
    def test_thread_safety_stress_test(self, temp_dir: Path) -> None:
        """Stress test thread safety with high concurrency."""
        visualizer = self.create_concurrent_visualizer(temp_dir, 0)
        
        try:
            visualizer.episode_manager.start_new_run("thread_safety_test")
            visualizer.start_episode(episode_num=1, task_id="thread_safety_task")
            
            results = []
            errors = []
            
            def stress_visualize(thread_id: int) -> None:
                """Stress test visualization from multiple threads."""
                try:
                    for i in range(20):  # 20 steps per thread
                        before_grid = jnp.ones((3, 3), dtype=jnp.int32) * thread_id
                        after_grid = jnp.ones((3, 3), dtype=jnp.int32) * (thread_id + i)
                        
                        visualizer.visualize_step(
                            before_grid=before_grid,
                            action={"type": "stress", "thread": thread_id, "step": i},
                            after_grid=after_grid,
                            reward=0.01 * i,
                            info={"thread_id": thread_id},
                            step_num=thread_id * 20 + i,
                        )
                        
                        # Small delay to increase chance of race conditions
                        time.sleep(0.001)
                    
                    results.append(f"thread_{thread_id}_success")
                    
                except Exception as e:
                    errors.append(f"thread_{thread_id}_error: {e}")
            
            # Create many threads for stress testing
            num_threads = 10
            threads = []
            
            with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
                mock_viz.return_value = "<svg>stress test</svg>"
                
                for thread_id in range(num_threads):
                    thread = threading.Thread(target=stress_visualize, args=(thread_id,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads
                for thread in threads:
                    thread.join(timeout=10.0)
            
            # Wait for async processing
            time.sleep(1.0)
            visualizer.async_logger.flush()
            
            # Check results
            assert len(results) == num_threads, f"Only {len(results)}/{num_threads} threads succeeded"
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            
            # Check that all steps were processed
            expected_steps = num_threads * 20
            assert visualizer.step_count == expected_steps, f"Expected {expected_steps} steps, got {visualizer.step_count}"
            
        finally:
            visualizer.shutdown()


class TestStorageCleanupEfficiency:
    """Test storage cleanup efficiency and performance."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_cleanup_performance_large_dataset(self, temp_dir: Path) -> None:
        """Test cleanup performance with large datasets."""
        episode_config = EpisodeConfig(
            base_output_dir=str(temp_dir),
            max_episodes_per_run=50,
            cleanup_policy="size_based",
            max_storage_gb=0.1,  # Small limit to trigger cleanup
        )
        
        episode_manager = EpisodeManager(episode_config)
        episode_manager.start_new_run("cleanup_performance_test")
        
        profiler = PerformanceProfiler()
        
        # Create many episodes with large files
        episode_dirs = []
        for episode_num in range(100):  # More than max_episodes_per_run
            episode_dir = episode_manager.start_new_episode(episode_num + 1)
            episode_dirs.append(episode_dir)
            
            # Create multiple large files per episode
            for file_num in range(5):
                large_content = "x" * 10000  # 10KB per file
                (episode_dir / f"large_file_{file_num}.txt").write_text(large_content)
            
            # Create metadata
            metadata = {
                "episode_num": episode_num + 1,
                "files_created": 5,
                "total_size_kb": 50,
            }
            (episode_dir / "metadata.json").write_text(json.dumps(metadata))
        
        # Measure cleanup performance
        profiler.start()
        episode_manager.cleanup_old_data()
        profiler.end()
        
        # Check cleanup efficiency
        assert profiler.duration < 5.0, f"Cleanup took too long: {profiler.duration:.2f}s"
        
        # Verify cleanup occurred
        remaining_dirs = [d for d in episode_dirs if d.exists()]
        assert len(remaining_dirs) < 100, "Cleanup should have removed some episodes"
        assert len(remaining_dirs) <= episode_config.max_episodes_per_run, "Too many episodes remaining"
        
        # Check storage usage is within limits
        storage_usage = episode_manager.get_storage_usage_gb()
        assert storage_usage <= episode_config.max_storage_gb * 1.1, f"Storage usage {storage_usage:.3f}GB exceeds limit"
    
    def test_cleanup_different_policies_performance(self, temp_dir: Path) -> None:
        """Test performance of different cleanup policies."""
        policies = ["oldest_first", "size_based"]
        policy_results = {}
        
        for policy in policies:
            policy_temp_dir = temp_dir / policy
            policy_temp_dir.mkdir()
            
            episode_config = EpisodeConfig(
                base_output_dir=str(policy_temp_dir),
                max_episodes_per_run=20,
                cleanup_policy=policy,
                max_storage_gb=0.05,
            )
            
            episode_manager = EpisodeManager(episode_config)
            episode_manager.start_new_run(f"cleanup_test_{policy}")
            
            # Create episodes with varying sizes and timestamps
            for episode_num in range(50):  # More than limit
                episode_dir = episode_manager.start_new_episode(episode_num + 1)
                
                # Vary file sizes for size_based testing
                file_size = 1000 * (episode_num % 10 + 1)  # 1KB to 10KB
                content = "x" * file_size
                (episode_dir / "variable_file.txt").write_text(content)
                
                # Add small delay for timestamp variation
                time.sleep(0.01)
            
            # Measure cleanup performance
            profiler = PerformanceProfiler()
            profiler.start()
            episode_manager.cleanup_old_data()
            profiler.end()
            
            policy_results[policy] = {
                "duration": profiler.duration,
                "remaining_episodes": len([d for d in episode_manager.list_episodes() if d.exists()]),
                "storage_usage": episode_manager.get_storage_usage_gb(),
            }
        
        # Compare policy performance
        for policy, results in policy_results.items():
            assert results["duration"] < 3.0, f"{policy} cleanup took too long: {results['duration']:.2f}s"
            assert results["remaining_episodes"] <= 20, f"{policy} left too many episodes: {results['remaining_episodes']}"
            assert results["storage_usage"] <= 0.1, f"{policy} storage usage too high: {results['storage_usage']:.3f}GB"
        
        # Both policies should be reasonably efficient
        duration_diff = abs(policy_results["oldest_first"]["duration"] - policy_results["size_based"]["duration"])
        assert duration_diff < 2.0, f"Large performance difference between policies: {duration_diff:.2f}s"
    
    def test_concurrent_cleanup_safety(self, temp_dir: Path) -> None:
        """Test cleanup safety during concurrent operations."""
        episode_config = EpisodeConfig(
            base_output_dir=str(temp_dir),
            max_episodes_per_run=10,
            cleanup_policy="oldest_first",
            max_storage_gb=0.05,
        )
        
        episode_manager = EpisodeManager(episode_config)
        episode_manager.start_new_run("concurrent_cleanup_test")
        
        results = []
        errors = []
        
        def create_episodes():
            """Create episodes concurrently."""
            try:
                for i in range(20):
                    episode_dir = episode_manager.start_new_episode(i + 1)
                    (episode_dir / "test_file.txt").write_text("test content" * 100)
                    time.sleep(0.05)  # Small delay
                results.append("creation_success")
            except Exception as e:
                errors.append(f"creation_error: {e}")
        
        def trigger_cleanup():
            """Trigger cleanup concurrently."""
            try:
                time.sleep(0.5)  # Let some episodes be created first
                for _ in range(5):  # Multiple cleanup attempts
                    episode_manager.cleanup_old_data()
                    time.sleep(0.2)
                results.append("cleanup_success")
            except Exception as e:
                errors.append(f"cleanup_error: {e}")
        
        # Run concurrent operations
        creation_thread = threading.Thread(target=create_episodes)
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        
        creation_thread.start()
        cleanup_thread.start()
        
        creation_thread.join(timeout=10.0)
        cleanup_thread.join(timeout=10.0)
        
        # Check results
        assert len(errors) == 0, f"Concurrent cleanup errors: {errors}"
        assert "creation_success" in results, "Episode creation failed"
        assert "cleanup_success" in results, "Cleanup failed"
        
        # Verify final state is consistent
        remaining_episodes = episode_manager.list_episodes()
        assert len(remaining_episodes) <= episode_config.max_episodes_per_run
        
        # Check that remaining episodes are valid
        for episode_dir in remaining_episodes:
            assert episode_dir.exists()
            assert (episode_dir / "episode_metadata.json").exists()


class TestMemoryUsageAndLeakDetection:
    """Test memory usage patterns and leak detection."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_memory_usage_scaling(self, temp_dir: Path) -> None:
        """Test memory usage scaling with episode size."""
        episode_sizes = [10, 50, 100, 500]  # Different episode sizes
        memory_usage = {}
        
        for episode_size in episode_sizes:
            # Create fresh visualizer for each test
            episode_config = EpisodeConfig(base_output_dir=str(temp_dir / f"memory_test_{episode_size}"))
            async_config = AsyncLoggerConfig(queue_size=1000, worker_threads=1)
            wandb_config = WandbConfig(enabled=False)
            
            vis_config = VisualizationConfig(
                debug_level="minimal",
                episode_config=episode_config,
                async_logger_config=async_config,
                wandb_config=wandb_config,
            )
            
            episode_manager = EpisodeManager(episode_config)
            async_logger = AsyncLogger(async_config, output_dir=temp_dir / f"logs_{episode_size}")
            wandb_integration = WandbIntegration(wandb_config)
            
            visualizer = EnhancedVisualizer(
                config=vis_config,
                episode_manager=episode_manager,
                async_logger=async_logger,
                wandb_integration=wandb_integration,
            )
            
            try:
                # Measure memory before episode
                gc.collect()
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                visualizer.episode_manager.start_new_run(f"memory_test_{episode_size}")
                visualizer.start_episode(episode_num=1, task_id=f"memory_task_{episode_size}")
                
                with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
                    mock_viz.return_value = "<svg>memory test</svg>"
                    
                    # Run episode
                    for step in range(episode_size):
                        grid_size = min(10, step // 10 + 3)  # Gradually increase grid size
                        before_grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32) * step
                        after_grid = before_grid + 1
                        
                        visualizer.visualize_step(
                            before_grid=before_grid,
                            action={"type": "memory_test", "step": step},
                            after_grid=after_grid,
                            reward=0.01,
                            info={"memory_test": True},
                            step_num=step,
                        )
                
                # Flush and measure final memory
                visualizer.async_logger.flush()
                gc.collect()
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                memory_usage[episode_size] = {
                    "initial": initial_memory,
                    "final": final_memory,
                    "delta": final_memory - initial_memory,
                }
                
            finally:
                visualizer.shutdown()
        
        # Analyze memory scaling
        episode_sizes_sorted = sorted(episode_sizes)
        memory_deltas = [memory_usage[size]["delta"] for size in episode_sizes_sorted]
        
        # Memory usage should scale reasonably with episode size
        for i in range(1, len(episode_sizes_sorted)):
            size_ratio = episode_sizes_sorted[i] / episode_sizes_sorted[i-1]
            memory_ratio = memory_deltas[i] / max(memory_deltas[i-1], 1.0)  # Avoid division by zero
            
            # Memory growth should be sub-linear (better than O(n))
            assert memory_ratio < size_ratio * 1.5, f"Memory scaling too aggressive: {memory_ratio:.2f}x for {size_ratio:.2f}x size increase"
        
        # Absolute memory usage should be reasonable
        max_memory_delta = max(memory_deltas)
        assert max_memory_delta < 200.0, f"Maximum memory usage too high: {max_memory_delta:.2f}MB"
    
    def test_async_logger_memory_management(self, temp_dir: Path) -> None:
        """Test async logger memory management under load."""
        async_config = AsyncLoggerConfig(
            queue_size=1000,
            worker_threads=2,
            batch_size=50,
            flush_interval=0.5,
        )
        
        async_logger = AsyncLogger(async_config, output_dir=temp_dir)
        
        try:
            # Measure initial memory
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = initial_memory
            
            # Generate high load
            for batch in range(20):  # 20 batches
                for i in range(100):  # 100 entries per batch
                    large_data = {
                        "batch": batch,
                        "entry": i,
                        "large_content": "x" * 1000,  # 1KB per entry
                        "grid_data": np.random.rand(10, 10).tolist(),
                    }
                    
                    from jaxarc.utils.visualization.async_logger import LogEntry
                    entry = LogEntry(
                        entry_type="memory_test",
                        data=large_data,
                        priority=i % 10,
                    )
                    
                    async_logger.log_entry(entry)
                
                # Check memory periodically
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Partial flush to prevent unbounded growth
                if batch % 5 == 0:
                    async_logger.flush()
                    gc.collect()
            
            # Final flush and memory check
            async_logger.flush()
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Memory analysis
            memory_growth = final_memory - initial_memory
            peak_memory_growth = peak_memory - initial_memory
            
            # Memory growth should be bounded
            assert memory_growth < 100.0, f"Final memory growth too high: {memory_growth:.2f}MB"
            assert peak_memory_growth < 200.0, f"Peak memory growth too high: {peak_memory_growth:.2f}MB"
            
            # Check that queue doesn't grow unbounded
            queue_size = async_logger.queue.qsize()
            assert queue_size < async_config.queue_size * 0.8, f"Queue size too high: {queue_size}"
            
        finally:
            async_logger.shutdown()
    
    def test_garbage_collection_effectiveness(self, temp_dir: Path) -> None:
        """Test garbage collection effectiveness."""
        episode_config = EpisodeConfig(base_output_dir=str(temp_dir))
        async_config = AsyncLoggerConfig(queue_size=500, worker_threads=1)
        wandb_config = WandbConfig(enabled=False)
        
        vis_config = VisualizationConfig(
            debug_level="standard",
            episode_config=episode_config,
            async_logger_config=async_config,
            wandb_config=wandb_config,
        )
        
        episode_manager = EpisodeManager(episode_config)
        async_logger = AsyncLogger(async_config, output_dir=temp_dir)
        wandb_integration = WandbIntegration(wandb_config)
        
        visualizer = EnhancedVisualizer(
            config=vis_config,
            episode_manager=episode_manager,
            async_logger=async_logger,
            wandb_integration=wandb_integration,
        )
        
        try:
            visualizer.episode_manager.start_new_run("gc_test")
            
            # Create and destroy many objects
            memory_samples = []
            
            for cycle in range(10):
                # Measure memory before cycle
                gc.collect()
                pre_cycle_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Create episode with many large objects
                visualizer.start_episode(episode_num=cycle + 1, task_id=f"gc_task_{cycle}")
                
                large_objects = []
                for step in range(50):
                    # Create large grids
                    large_grid = jnp.ones((15, 15), dtype=jnp.int32) * step
                    large_objects.append(large_grid)
                    
                    with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_viz:
                        mock_viz.return_value = "<svg>gc test</svg>"
                        
                        visualizer.visualize_step(
                            before_grid=large_grid,
                            action={"type": "gc_test", "cycle": cycle, "step": step},
                            after_grid=large_grid + 1,
                            reward=0.01,
                            info={"gc_test": True},
                            step_num=step,
                        )
                
                # Explicitly delete large objects
                del large_objects
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory after cleanup
                post_cycle_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append({
                    "cycle": cycle,
                    "pre_cycle": pre_cycle_memory,
                    "post_cycle": post_cycle_memory,
                    "cycle_growth": post_cycle_memory - pre_cycle_memory,
                })
            
            # Analyze garbage collection effectiveness
            cycle_growths = [sample["cycle_growth"] for sample in memory_samples]
            avg_cycle_growth = np.mean(cycle_growths)
            
            # Average cycle growth should be small (effective GC)
            assert avg_cycle_growth < 20.0, f"Average cycle memory growth too high: {avg_cycle_growth:.2f}MB"
            
            # Total memory growth should be bounded
            total_growth = memory_samples[-1]["post_cycle"] - memory_samples[0]["pre_cycle"]
            assert total_growth < 100.0, f"Total memory growth too high: {total_growth:.2f}MB"
            
            # Check that memory doesn't grow monotonically (indicating effective GC)
            decreasing_cycles = sum(1 for i in range(1, len(cycle_growths)) if cycle_growths[i] < cycle_growths[i-1])
            assert decreasing_cycles >= 2, "Memory should decrease in some cycles due to GC"
            
        finally:
            visualizer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])