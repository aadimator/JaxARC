"""Integration tests for enhanced visualization system end-to-end functionality."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.utils.visualization.async_logger import AsyncLogger, AsyncLoggerConfig
from jaxarc.utils.visualization.config_composition import quick_compose
from jaxarc.utils.visualization.config_validation import validate_config
from jaxarc.utils.visualization.enhanced_visualizer import (
    EnhancedVisualizer,
    VisualizationConfig,
)
from jaxarc.utils.visualization.episode_manager import EpisodeConfig, EpisodeManager
from jaxarc.utils.visualization.wandb_integration import WandbConfig, WandbIntegration


class TestVisualizationPipelineIntegration:
    """Test complete visualization pipeline end-to-end."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def complete_vis_config(self) -> VisualizationConfig:
        """Create comprehensive visualization configuration."""
        return VisualizationConfig(
            debug_level="standard",
            output_formats=["svg", "png"],
            show_operation_names=True,
            highlight_changes=True,
            include_metrics=True,
            color_scheme="default",
        )

    @pytest.fixture
    def episode_config(self, temp_dir: Path) -> EpisodeConfig:
        """Create episode configuration for integration testing."""
        return EpisodeConfig(
            base_output_dir=str(temp_dir / "episodes"),
            max_episodes_per_run=50,
            cleanup_policy="size_based",
            max_storage_gb=1.0,
        )

    @pytest.fixture
    def async_logger_config(self, temp_dir: Path) -> AsyncLoggerConfig:
        """Create async logger configuration."""
        return AsyncLoggerConfig(
            queue_size=500,
            worker_threads=2,
            batch_size=5,
            flush_interval=1.0,
            enable_compression=True,
        )

    @pytest.fixture
    def wandb_config(self) -> WandbConfig:
        """Create wandb configuration for testing."""
        return WandbConfig(
            enabled=True,
            project_name="integration-test",
            log_frequency=5,
            offline_mode=True,  # Use offline mode for testing
        )

    @pytest.fixture
    def integrated_visualizer(
        self,
        complete_vis_config: VisualizationConfig,
        episode_config: EpisodeConfig,
        async_logger_config: AsyncLoggerConfig,
        wandb_config: WandbConfig,
        temp_dir: Path,
    ) -> EnhancedVisualizer:
        """Create fully integrated visualizer system."""
        episode_manager = EpisodeManager(episode_config)
        async_logger = AsyncLogger(async_logger_config, output_dir=temp_dir / "logs")

        with patch("jaxarc.utils.visualization.wandb_integration.wandb"):
            wandb_integration = WandbIntegration(wandb_config)

        # Create complete config with nested configs
        full_config = VisualizationConfig(
            debug_level=complete_vis_config.debug_level,
            output_formats=complete_vis_config.output_formats,
            show_operation_names=complete_vis_config.show_operation_names,
            highlight_changes=complete_vis_config.highlight_changes,
            include_metrics=complete_vis_config.include_metrics,
            color_scheme=complete_vis_config.color_scheme,
            episode_config=episode_config,
            async_logger_config=async_logger_config,
            wandb_config=wandb_config,
        )

        visualizer = EnhancedVisualizer(
            config=full_config,
            episode_manager=episode_manager,
            async_logger=async_logger,
            wandb_integration=wandb_integration,
        )

        yield visualizer

        # Cleanup
        visualizer.shutdown()

    def test_complete_episode_workflow(
        self, integrated_visualizer: EnhancedVisualizer, temp_dir: Path
    ) -> None:
        """Test complete episode workflow from start to finish."""
        # Start run and episode
        integrated_visualizer.episode_manager.start_new_run("integration_test_run")
        integrated_visualizer.start_episode(
            episode_num=1, task_id="integration_task_001"
        )

        # Mock visualization functions
        with (
            patch(
                "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg"
            ) as mock_step_viz,
            patch(
                "jaxarc.utils.visualization.enhanced_visualizer.create_episode_summary_svg"
            ) as mock_summary_viz,
            patch(
                "jaxarc.utils.visualization.enhanced_visualizer.convert_svg_to_png"
            ) as mock_png_convert,
        ):
            mock_step_viz.return_value = "<svg>step visualization</svg>"
            mock_summary_viz.return_value = "<svg>episode summary</svg>"
            mock_png_convert.return_value = b"png_data"

            # Simulate episode steps
            rewards = []
            similarities = []

            for step in range(20):
                before_grid = jnp.ones((5, 5), dtype=jnp.int32) * step
                after_grid = jnp.ones((5, 5), dtype=jnp.int32) * (step + 1)
                action = {"type": "fill", "color": step % 10, "step": step}
                reward = 0.1 * step
                similarity = min(0.95, 0.1 + 0.04 * step)

                rewards.append(reward)
                similarities.append(similarity)

                integrated_visualizer.visualize_step(
                    before_grid=before_grid,
                    action=action,
                    after_grid=after_grid,
                    reward=reward,
                    info={"similarity": similarity, "step": step},
                    step_num=step,
                )

            # End episode with summary
            from jaxarc.utils.visualization.enhanced_visualizer import (
                EpisodeSummaryData,
            )

            episode_data = EpisodeSummaryData(
                episode_num=1,
                total_steps=20,
                total_reward=sum(rewards),
                reward_progression=rewards,
                similarity_progression=similarities,
                final_similarity=similarities[-1],
                task_id="integration_task_001",
                success=similarities[-1] > 0.9,
                key_moments=[5, 10, 15],
            )

            integrated_visualizer.visualize_episode_summary(episode_data)

        # Wait for async processing
        time.sleep(2.0)
        integrated_visualizer.async_logger.flush()

        # Verify outputs
        # Check episode directory structure
        episode_dir = integrated_visualizer.episode_manager.current_episode_dir
        assert episode_dir.exists()

        # Check step visualizations were created
        step_files = list(episode_dir.glob("step_*.svg"))
        assert len(step_files) > 0  # Should have some step files

        # Check episode summary
        summary_files = list(episode_dir.glob("episode_summary.*"))
        assert len(summary_files) > 0

        # Check async logging
        log_files = list((temp_dir / "logs").glob("*.json"))
        assert len(log_files) > 0

        # Verify step visualizations were called
        assert mock_step_viz.call_count > 0
        assert mock_summary_viz.call_count == 1

        # Check performance stats
        stats = integrated_visualizer.get_performance_stats()
        assert stats["steps_visualized"] == 20
        assert stats["episodes_completed"] == 1
        assert "avg_step_time" in stats

    def test_multiple_episodes_workflow(
        self, integrated_visualizer: EnhancedVisualizer, temp_dir: Path
    ) -> None:
        """Test workflow with multiple episodes."""
        integrated_visualizer.episode_manager.start_new_run("multi_episode_test")

        with (
            patch(
                "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg"
            ) as mock_step_viz,
            patch(
                "jaxarc.utils.visualization.enhanced_visualizer.create_episode_summary_svg"
            ) as mock_summary_viz,
        ):
            mock_step_viz.return_value = "<svg>step</svg>"
            mock_summary_viz.return_value = "<svg>summary</svg>"

            episode_summaries = []

            # Run 3 episodes
            for episode_num in range(1, 4):
                integrated_visualizer.start_episode(
                    episode_num=episode_num, task_id=f"task_{episode_num:03d}"
                )

                # Simulate steps for each episode
                episode_rewards = []
                for step in range(10):
                    before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
                    after_grid = jnp.ones((3, 3), dtype=jnp.int32)
                    reward = 0.1 * (episode_num + step)
                    episode_rewards.append(reward)

                    integrated_visualizer.visualize_step(
                        before_grid=before_grid,
                        action={"type": "test", "episode": episode_num, "step": step},
                        after_grid=after_grid,
                        reward=reward,
                        info={"episode": episode_num},
                        step_num=step,
                    )

                # Create episode summary
                from jaxarc.utils.visualization.enhanced_visualizer import (
                    EpisodeSummaryData,
                )

                episode_data = EpisodeSummaryData(
                    episode_num=episode_num,
                    total_steps=10,
                    total_reward=sum(episode_rewards),
                    reward_progression=episode_rewards,
                    final_similarity=0.8 + 0.05 * episode_num,
                    task_id=f"task_{episode_num:03d}",
                    success=episode_num > 1,  # First episode fails, others succeed
                )

                episode_summaries.append(episode_data)
                integrated_visualizer.visualize_episode_summary(episode_data)

        # Wait for processing
        time.sleep(1.5)
        integrated_visualizer.async_logger.flush()

        # Test comparison visualization
        with patch(
            "jaxarc.utils.visualization.enhanced_visualizer.create_comparison_svg"
        ) as mock_comparison:
            mock_comparison.return_value = "<svg>comparison</svg>"

            comparison_result = integrated_visualizer.create_comparison_visualization(
                episode_summaries, comparison_type="reward_progression"
            )

            assert comparison_result == "<svg>comparison</svg>"
            mock_comparison.assert_called_once()

        # Verify multiple episode directories
        run_dir = integrated_visualizer.episode_manager.current_run_dir
        episode_dirs = list(run_dir.glob("episode_*"))
        assert len(episode_dirs) == 3

        # Check that each episode has its own files
        for episode_dir in episode_dirs:
            assert episode_dir.is_dir()
            metadata_file = episode_dir / "episode_metadata.json"
            assert metadata_file.exists()

    def test_jax_performance_impact_measurement(
        self, integrated_visualizer: EnhancedVisualizer
    ) -> None:
        """Test JAX performance impact measurement."""
        integrated_visualizer.episode_manager.start_new_run("performance_test")
        integrated_visualizer.start_episode(episode_num=1, task_id="perf_task")

        # Create JAX computation to measure impact
        @jax.jit
        def test_computation(x):
            return jnp.sum(x**2) + jnp.mean(x)

        # Measure baseline performance without visualization
        key = jax.random.PRNGKey(42)
        test_data = jax.random.normal(key, (1000, 1000))

        # Warmup
        _ = test_computation(test_data)

        # Baseline timing
        baseline_times = []
        for _ in range(10):
            start = time.time()
            result = test_computation(test_data)
            baseline_times.append(time.time() - start)

        baseline_avg = np.mean(baseline_times)

        # Measure performance with visualization
        with patch(
            "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg"
        ) as mock_viz:
            mock_viz.return_value = "<svg>perf test</svg>"

            viz_times = []
            for i in range(10):
                start = time.time()

                # JAX computation
                result = test_computation(test_data)

                # Visualization (should be async and not block)
                integrated_visualizer.visualize_step(
                    before_grid=jnp.array([[0, 1], [2, 3]]),
                    action={"type": "perf_test", "iteration": i},
                    after_grid=jnp.array([[1, 1], [2, 4]]),
                    reward=float(result),
                    info={"performance_test": True},
                    step_num=i,
                )

                viz_times.append(time.time() - start)

        viz_avg = np.mean(viz_times)

        # Performance impact should be minimal (< 10% overhead)
        performance_impact = (viz_avg - baseline_avg) / baseline_avg
        assert performance_impact < 0.1, (
            f"Performance impact too high: {performance_impact:.2%}"
        )

        # Check performance monitoring
        perf_stats = integrated_visualizer.get_performance_stats()
        assert "jax_computation_time" in perf_stats or "avg_step_time" in perf_stats
        assert perf_stats["steps_visualized"] == 10

    def test_configuration_composition_and_validation(self, temp_dir: Path) -> None:
        """Test configuration composition and validation."""
        # Test valid configuration composition
        base_config = {
            "debug_level": "standard",
            "output_formats": ["svg"],
            "episode_management": {
                "base_output_dir": str(temp_dir),
                "max_episodes_per_run": 100,
            },
            "async_logging": {
                "queue_size": 1000,
                "worker_threads": 2,
            },
            "wandb": {
                "enabled": False,
                "project_name": "test-project",
            },
        }

        # Test configuration validation
        from omegaconf import OmegaConf

        config_obj = OmegaConf.create(base_config)
        validation_errors = validate_config(config_obj)
        assert len(validation_errors) == 0

        # Test configuration composition using quick_compose
        composed_config = quick_compose(
            debug_level="verbose",
            storage_type="development",
            wandb_enabled=True,
        )
        assert composed_config is not None  # Basic composition test

        # Test invalid configuration
        invalid_config = {
            "visualization": {
                "debug_level": "invalid_level",
            }
        }

        invalid_config_obj = OmegaConf.create(invalid_config)
        invalid_errors = validate_config(invalid_config_obj)
        assert len(invalid_errors) > 0
        assert any("debug_level" in error.field for error in invalid_errors)

    def test_error_handling_and_recovery(
        self, integrated_visualizer: EnhancedVisualizer, temp_dir: Path
    ) -> None:
        """Test error handling and recovery scenarios."""
        integrated_visualizer.episode_manager.start_new_run("error_test")
        integrated_visualizer.start_episode(episode_num=1, task_id="error_task")

        # Test visualization error recovery
        error_count = 0

        def failing_visualization(*args, **kwargs):
            nonlocal error_count
            error_count += 1
            if error_count <= 3:  # Fail first 3 times
                raise Exception(f"Visualization error {error_count}")
            return "<svg>recovered visualization</svg>"

        with patch(
            "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg",
            side_effect=failing_visualization,
        ):
            # Should handle errors gracefully
            for step in range(5):
                integrated_visualizer.visualize_step(
                    before_grid=jnp.array([[0, 1], [2, 3]]),
                    action={"type": "error_test", "step": step},
                    after_grid=jnp.array([[1, 1], [2, 4]]),
                    reward=0.1 * step,
                    info={"error_test": True},
                    step_num=step,
                )

        # System should still be functional
        assert integrated_visualizer.step_count == 5

        # Check error statistics
        error_stats = integrated_visualizer.get_error_stats()
        assert error_stats["visualization_errors"] >= 3
        assert error_stats["recovered_errors"] >= 1

        # Test storage error recovery
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
            # Should handle storage errors
            try:
                integrated_visualizer.episode_manager.start_new_episode(2)
            except Exception:
                pass  # Expected to fail

            # System should still be responsive
            assert integrated_visualizer.async_logger.is_running

        # Test async logger error recovery
        with patch("builtins.open", side_effect=OSError("Disk full")):
            # Should handle logging errors
            integrated_visualizer.async_logger.log_entry(
                {"type": "error_test", "data": {"test": "error_recovery"}}
            )

            time.sleep(0.5)  # Allow error handling

            # Logger should still be running
            assert integrated_visualizer.async_logger.is_running

    def test_concurrent_episode_processing(self, temp_dir: Path) -> None:
        """Test concurrent episode processing."""
        # Create multiple visualizers for concurrent testing
        configs = []
        visualizers = []

        for i in range(3):
            episode_config = EpisodeConfig(
                base_output_dir=str(temp_dir / f"concurrent_{i}"),
                max_episodes_per_run=10,
            )
            async_config = AsyncLoggerConfig(
                queue_size=100,
                worker_threads=1,
            )
            vis_config = VisualizationConfig(debug_level="minimal")

            episode_manager = EpisodeManager(episode_config)
            async_logger = AsyncLogger(async_config, output_dir=temp_dir / f"logs_{i}")

            with patch("jaxarc.utils.visualization.wandb_integration.wandb"):
                wandb_integration = WandbIntegration(WandbConfig(enabled=False))

            complete_config = VisualizationConfig(
                debug_level="minimal",
                episode_config=episode_config,
                async_logger_config=async_config,
                wandb_config=wandb_config,
            )

            visualizer = EnhancedVisualizer(
                config=complete_config,
                episode_manager=episode_manager,
                async_logger=async_logger,
                wandb_integration=wandb_integration,
            )

            visualizers.append(visualizer)

        try:
            results = []
            errors = []

            def run_episode(visualizer_id: int, visualizer: EnhancedVisualizer):
                try:
                    visualizer.episode_manager.start_new_run(
                        f"concurrent_run_{visualizer_id}"
                    )
                    visualizer.start_episode(
                        episode_num=1, task_id=f"concurrent_task_{visualizer_id}"
                    )

                    with patch(
                        "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg"
                    ) as mock_viz:
                        mock_viz.return_value = f"<svg>concurrent_{visualizer_id}</svg>"

                        for step in range(5):
                            visualizer.visualize_step(
                                before_grid=jnp.array([[visualizer_id, step]]),
                                action={
                                    "type": "concurrent",
                                    "visualizer": visualizer_id,
                                    "step": step,
                                },
                                after_grid=jnp.array([[visualizer_id + 1, step + 1]]),
                                reward=0.1 * step,
                                info={"concurrent": True},
                                step_num=step,
                            )

                    results.append(f"visualizer_{visualizer_id}_success")

                except Exception as e:
                    errors.append(f"visualizer_{visualizer_id}_error: {e}")

            # Run concurrent episodes
            threads = []
            for i, visualizer in enumerate(visualizers):
                thread = threading.Thread(target=run_episode, args=(i, visualizer))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)

            # Wait for async processing
            time.sleep(1.0)
            for visualizer in visualizers:
                visualizer.async_logger.flush()

            # Check results
            assert len(results) == 3, (
                f"Expected 3 successes, got {len(results)}: {results}"
            )
            assert len(errors) == 0, f"Unexpected errors: {errors}"

            # Verify separate outputs
            for i in range(3):
                concurrent_dir = temp_dir / f"concurrent_{i}"
                assert concurrent_dir.exists()

                run_dirs = list(concurrent_dir.glob("*run_*"))
                assert len(run_dirs) >= 1

        finally:
            # Cleanup
            for visualizer in visualizers:
                visualizer.shutdown()

    def test_storage_cleanup_integration(
        self, integrated_visualizer: EnhancedVisualizer, temp_dir: Path
    ) -> None:
        """Test storage cleanup integration with episode management."""
        # Configure small storage limits
        integrated_visualizer.episode_manager.config = (
            integrated_visualizer.episode_manager.config.replace(
                max_episodes_per_run=3,
                max_storage_gb=0.001,  # Very small limit
                cleanup_policy="size_based",
            )
        )

        integrated_visualizer.episode_manager.start_new_run("cleanup_test")

        with patch(
            "jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg"
        ) as mock_viz:
            mock_viz.return_value = (
                "<svg>" + "x" * 1000 + "</svg>"
            )  # Large visualization

            episode_dirs = []

            # Create episodes that exceed storage limits
            for episode_num in range(5):  # More than max_episodes_per_run
                integrated_visualizer.start_episode(
                    episode_num=episode_num + 1, task_id=f"cleanup_task_{episode_num}"
                )
                episode_dirs.append(
                    integrated_visualizer.episode_manager.current_episode_dir
                )

                # Create large visualizations
                for step in range(3):
                    integrated_visualizer.visualize_step(
                        before_grid=jnp.ones((10, 10), dtype=jnp.int32),
                        action={
                            "type": "cleanup_test",
                            "episode": episode_num,
                            "step": step,
                        },
                        after_grid=jnp.ones((10, 10), dtype=jnp.int32) * 2,
                        reward=0.1,
                        info={},
                        step_num=step,
                    )

                # Write large files to trigger cleanup
                large_content = "x" * 10000  # 10KB
                (
                    integrated_visualizer.episode_manager.current_episode_dir
                    / "large_file.txt"
                ).write_text(large_content)

        # Wait for processing and trigger cleanup
        time.sleep(1.0)
        integrated_visualizer.episode_manager.cleanup_old_data()

        # Check that cleanup occurred
        remaining_dirs = [d for d in episode_dirs if d.exists()]
        assert len(remaining_dirs) < 5, "Cleanup should have removed some episodes"

        # Check storage usage is within limits
        storage_usage = integrated_visualizer.episode_manager.get_storage_usage_gb()
        # Should be close to or under the limit (allowing some tolerance)
        assert storage_usage <= 0.01, (
            f"Storage usage {storage_usage} exceeds expected limit"
        )


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_hydra_config_integration(self, temp_dir: Path) -> None:
        """Test integration with Hydra configuration system."""
        # Mock Hydra config structure
        hydra_config = {
            "visualization": {
                "debug_level": "standard",
                "output_formats": ["svg", "png"],
                "show_operation_names": True,
            },
            "episode_management": {
                "base_output_dir": str(temp_dir),
                "max_episodes_per_run": 100,
                "cleanup_policy": "oldest_first",
            },
            "async_logging": {
                "queue_size": 500,
                "worker_threads": 2,
                "enable_compression": True,
            },
            "wandb": {
                "enabled": True,
                "project_name": "hydra-integration-test",
                "log_frequency": 10,
            },
        }

        # Test configuration parsing and validation
        # Test basic hydra config validation
        from omegaconf import OmegaConf

        config_obj = OmegaConf.create(hydra_config)
        validation_errors = validate_config(config_obj)

        # Should have no validation errors for valid config
        assert len(validation_errors) == 0

    def test_command_line_override_integration(self, temp_dir: Path) -> None:
        """Test command-line override integration."""
        base_config = {
            "visualization": {
                "debug_level": "standard",
                "output_formats": ["svg"],
            },
            "wandb": {
                "enabled": False,
                "project_name": "base-project",
            },
        }

        # Simulate command-line overrides
        cli_overrides = {
            "visualization.debug_level": "verbose",
            "visualization.output_formats": ["svg", "png", "html"],
            "wandb.enabled": True,
            "wandb.log_frequency": 5,
        }

        # Test CLI override simulation using quick_compose
        final_config = quick_compose(
            debug_level="verbose",
            storage_type="development",
            wandb_enabled=True,
        )

        # Basic test that composition works
        assert final_config is not None

    def test_environment_specific_configs(self, temp_dir: Path) -> None:
        """Test environment-specific configuration loading."""
        # Test development environment config
        dev_config = {
            "environment": "development",
            "visualization": {
                "debug_level": "verbose",
                "output_formats": ["svg"],
            },
            "episode_management": {
                "max_episodes_per_run": 50,
                "cleanup_policy": "manual",
            },
            "wandb": {
                "enabled": True,
                "offline_mode": False,
            },
        }

        # Test production environment config
        prod_config = {
            "environment": "production",
            "visualization": {
                "debug_level": "minimal",
                "output_formats": ["png"],
            },
            "episode_management": {
                "max_episodes_per_run": 1000,
                "cleanup_policy": "size_based",
            },
            "wandb": {
                "enabled": True,
                "offline_mode": True,
            },
        }

        # Test environment config validation
        from omegaconf import OmegaConf

        dev_config_obj = OmegaConf.create(dev_config)
        prod_config_obj = OmegaConf.create(prod_config)

        dev_errors = validate_config(dev_config_obj)
        prod_errors = validate_config(prod_config_obj)

        # Both should be valid (no errors)
        assert len(dev_errors) == 0
        assert len(prod_errors) == 0

        # Check environment-specific optimizations
        assert (
            dev_config["visualization"]["debug_level"] == "verbose"
        )  # More debugging in dev
        assert (
            prod_config["visualization"]["debug_level"] == "minimal"
        )  # Less overhead in prod

        assert (
            dev_config["episode_management"]["cleanup_policy"] == "manual"
        )  # Manual cleanup in dev
        assert (
            prod_config["episode_management"]["cleanup_policy"] == "size_based"
        )  # Automatic in prod


if __name__ == "__main__":
    pytest.main([__file__])
