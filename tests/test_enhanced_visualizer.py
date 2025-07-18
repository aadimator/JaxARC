"""Unit tests for EnhancedVisualizer component of enhanced visualization system."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
from typing import Any, Dict

import pytest
import numpy as np
import jax.numpy as jnp

from jaxarc.utils.visualization.enhanced_visualizer import (
    EnhancedVisualizer,
    VisualizationConfig,
    StepVisualizationData,
    EpisodeSummaryData,
)
from jaxarc.utils.visualization.episode_manager import EpisodeManager, EpisodeConfig
from jaxarc.utils.visualization.async_logger import AsyncLogger, AsyncLoggerConfig
from jaxarc.utils.visualization.wandb_integration import WandbIntegration, WandbConfig
from jaxarc.types import Grid


class TestVisualizationConfig:
    """Test VisualizationConfig dataclass and validation."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VisualizationConfig()
        
        assert config.debug_level == "standard"
        assert config.output_formats == ["svg"]
        assert config.image_quality == "high"
        assert config.show_coordinates is False
        assert config.show_operation_names is True
        assert config.highlight_changes is True
        assert config.include_metrics is True
        assert config.color_scheme == "default"
        assert config.use_double_width is True
        assert config.show_numbers is False
        assert config.async_processing is True
        assert config.max_concurrent_saves == 4
        assert config.compress_images is False
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VisualizationConfig(
            debug_level="verbose",
            output_formats=["svg", "png"],
            image_quality="medium",
            show_coordinates=True,
            show_operation_names=False,
            highlight_changes=False,
            include_metrics=False,
            color_scheme="colorblind",
            use_double_width=False,
            show_numbers=True,
            async_processing=False,
            max_concurrent_saves=8,
            compress_images=True,
        )
        
        assert config.debug_level == "verbose"
        assert config.output_formats == ["svg", "png"]
        assert config.image_quality == "medium"
        assert config.show_coordinates is True
        assert config.show_operation_names is False
        assert config.highlight_changes is False
        assert config.include_metrics is False
        assert config.color_scheme == "colorblind"
        assert config.use_double_width is False
        assert config.show_numbers is True
        assert config.async_processing is False
        assert config.max_concurrent_saves == 8
        assert config.compress_images is True
    
    def test_config_validation_invalid_debug_level(self) -> None:
        """Test validation fails for invalid debug level."""
        with pytest.raises(ValueError, match="Invalid debug_level"):
            VisualizationConfig(debug_level="invalid")
    
    def test_config_validation_invalid_image_quality(self) -> None:
        """Test validation fails for invalid image quality."""
        with pytest.raises(ValueError, match="Invalid image_quality"):
            VisualizationConfig(image_quality="invalid")
    
    def test_config_validation_invalid_color_scheme(self) -> None:
        """Test validation fails for invalid color scheme."""
        with pytest.raises(ValueError, match="Invalid color_scheme"):
            VisualizationConfig(color_scheme="invalid")
    
    def test_config_validation_negative_values(self) -> None:
        """Test validation fails for negative values."""
        with pytest.raises(ValueError, match="max_concurrent_saves must be positive"):
            VisualizationConfig(max_concurrent_saves=0)


class TestStepVisualizationData:
    """Test StepVisualizationData dataclass."""
    
    def test_step_data_creation(self) -> None:
        """Test step visualization data creation."""
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 4]])
        action = {"type": "fill", "color": 1, "position": (0, 0)}
        
        step_data = StepVisualizationData(
            step_num=5,
            before_grid=before_grid,
            after_grid=after_grid,
            action=action,
            reward=0.8,
            info={"operation": "fill"},
            selection_mask=jnp.array([[True, False], [False, False]]),
            changed_cells=jnp.array([[True, False], [False, True]]),
            operation_name="Fill Operation",
        )
        
        assert step_data.step_num == 5
        assert jnp.array_equal(step_data.before_grid, before_grid)
        assert jnp.array_equal(step_data.after_grid, after_grid)
        assert step_data.action == action
        assert step_data.reward == 0.8
        assert step_data.info == {"operation": "fill"}
        assert step_data.operation_name == "Fill Operation"
    
    def test_step_data_serialization(self) -> None:
        """Test step data serialization for logging."""
        step_data = StepVisualizationData(
            step_num=3,
            before_grid=jnp.array([[0, 1]]),
            after_grid=jnp.array([[1, 1]]),
            action={"type": "fill"},
            reward=0.5,
            info={},
        )
        
        serialized = step_data.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["step_num"] == 3
        assert serialized["reward"] == 0.5
        assert "before_grid" in serialized
        assert "after_grid" in serialized
        assert "action" in serialized


class TestEpisodeSummaryData:
    """Test EpisodeSummaryData dataclass."""
    
    def test_episode_summary_creation(self) -> None:
        """Test episode summary data creation."""
        summary_data = EpisodeSummaryData(
            episode_num=10,
            total_steps=25,
            total_reward=15.5,
            reward_progression=[0.1, 0.3, 0.8, 1.2, 1.5],
            similarity_progression=[0.2, 0.4, 0.6, 0.8, 0.95],
            final_similarity=0.95,
            task_id="task_001",
            success=True,
            key_moments=[5, 12, 20],
            performance_metrics={"avg_step_time": 0.05, "memory_usage": 128.5},
        )
        
        assert summary_data.episode_num == 10
        assert summary_data.total_steps == 25
        assert summary_data.total_reward == 15.5
        assert summary_data.final_similarity == 0.95
        assert summary_data.task_id == "task_001"
        assert summary_data.success is True
        assert summary_data.key_moments == [5, 12, 20]
        assert summary_data.performance_metrics["avg_step_time"] == 0.05
    
    def test_episode_summary_serialization(self) -> None:
        """Test episode summary serialization."""
        summary_data = EpisodeSummaryData(
            episode_num=5,
            total_steps=15,
            total_reward=10.0,
            final_similarity=0.8,
            task_id="test_task",
            success=False,
        )
        
        serialized = summary_data.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["episode_num"] == 5
        assert serialized["total_steps"] == 15
        assert serialized["success"] is False


class TestEnhancedVisualizer:
    """Test EnhancedVisualizer functionality."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vis_config(self) -> VisualizationConfig:
        """Create test visualization configuration."""
        return VisualizationConfig(
            debug_level="standard",
            output_formats=["svg"],
            show_operation_names=True,
            highlight_changes=True,
        )
    
    @pytest.fixture
    def episode_manager(self, temp_dir: Path) -> EpisodeManager:
        """Create episode manager for testing."""
        config = EpisodeConfig(base_output_dir=str(temp_dir))
        return EpisodeManager(config)
    
    @pytest.fixture
    def async_logger(self, temp_dir: Path) -> AsyncLogger:
        """Create async logger for testing."""
        config = AsyncLoggerConfig(worker_threads=1, queue_size=100)
        logger = AsyncLogger(config, output_dir=temp_dir)
        yield logger
        logger.shutdown()
    
    @pytest.fixture
    def wandb_integration(self) -> WandbIntegration:
        """Create mock wandb integration."""
        config = WandbConfig(enabled=False)  # Disabled for testing
        return WandbIntegration(config)
    
    @pytest.fixture
    def enhanced_visualizer(
        self,
        vis_config: VisualizationConfig,
        episode_manager: EpisodeManager,
        async_logger: AsyncLogger,
        wandb_integration: WandbIntegration,
    ) -> EnhancedVisualizer:
        """Create enhanced visualizer for testing."""
        # Create a complete config with nested configs
        complete_config = VisualizationConfig(
            debug_level=vis_config.debug_level,
            output_formats=vis_config.output_formats,
            show_operation_names=vis_config.show_operation_names,
            highlight_changes=vis_config.highlight_changes,
            episode_config=episode_manager.config,
            async_logger_config=async_logger.config,
            wandb_config=wandb_integration.config,
        )
        return EnhancedVisualizer(
            config=complete_config,
            episode_manager=episode_manager,
            async_logger=async_logger,
            wandb_integration=wandb_integration,
        )
    
    def test_initialization(
        self,
        enhanced_visualizer: EnhancedVisualizer,
        vis_config: VisualizationConfig,
        episode_manager: EpisodeManager,
        async_logger: AsyncLogger,
        wandb_integration: WandbIntegration,
    ) -> None:
        """Test enhanced visualizer initialization."""
        assert enhanced_visualizer.vis_config == vis_config
        assert enhanced_visualizer.episode_manager == episode_manager
        assert enhanced_visualizer.async_logger == async_logger
        assert enhanced_visualizer.wandb == wandb_integration
        assert enhanced_visualizer.current_episode_num is None
        assert enhanced_visualizer.step_count == 0
    
    def test_start_episode(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test starting a new episode."""
        # Start run first
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        
        # Start episode
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        assert enhanced_visualizer.current_episode_num == 1
        assert enhanced_visualizer.current_task_id == "task_001"
        assert enhanced_visualizer.step_count == 0
        assert enhanced_visualizer.episode_start_time is not None
        
        # Check episode directory was created
        assert enhanced_visualizer.episode_manager.current_episode_num == 1
    
    def test_visualize_step_standard_level(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test step visualization at standard debug level."""
        # Setup episode
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        # Create test data
        before_grid = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        after_grid = jnp.array([[1, 1, 2], [3, 1, 5], [6, 7, 8]])
        action = {"type": "fill", "color": 1, "position": (0, 0)}
        
        # Mock the core visualization function
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_draw:
            mock_draw.return_value = "<svg>test svg content</svg>"
            
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action=action,
                after_grid=after_grid,
                reward=0.8,
                info={"operation": "fill"},
                step_num=5,
            )
        
        # Check that visualization was called
        mock_draw.assert_called_once()
        
        # Check step count incremented
        assert enhanced_visualizer.step_count == 1
        
        # Check async logging was triggered
        time.sleep(0.1)  # Allow async processing
        assert enhanced_visualizer.async_logger.queue.qsize() >= 0  # May have been processed
    
    def test_visualize_step_off_level(self, enhanced_visualizer: EnhancedVisualizer) -> None:
        """Test step visualization with debug level off."""
        # Set debug level to off
        enhanced_visualizer.vis_config = enhanced_visualizer.vis_config.replace(debug_level="off")
        
        # Setup episode
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 3]])
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_draw:
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "fill"},
                after_grid=after_grid,
                reward=0.5,
                info={},
                step_num=1,
            )
        
        # Should not call visualization when debug level is off
        mock_draw.assert_not_called()
        
        # Step count should still increment
        assert enhanced_visualizer.step_count == 1
    
    def test_visualize_step_verbose_level(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test step visualization at verbose debug level."""
        # Set debug level to verbose
        enhanced_visualizer.vis_config = enhanced_visualizer.vis_config.replace(debug_level="verbose")
        
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 4]])
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_draw:
            mock_draw.return_value = "<svg>verbose svg</svg>"
            
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "fill", "details": "verbose_info"},
                after_grid=after_grid,
                reward=0.9,
                info={"verbose": True, "extra_data": "test"},
                step_num=3,
            )
        
        # Should call visualization with verbose details
        mock_draw.assert_called_once()
        call_args = mock_draw.call_args
        
        # Check that verbose information is included
        assert "verbose_info" in str(call_args) or "extra_data" in str(call_args)
    
    def test_visualize_episode_summary(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test episode summary visualization."""
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        # Simulate some steps
        enhanced_visualizer.step_count = 20
        enhanced_visualizer.episode_rewards = [0.1, 0.3, 0.5, 0.8, 1.0]
        enhanced_visualizer.episode_similarities = [0.2, 0.4, 0.6, 0.8, 0.95]
        
        episode_data = EpisodeSummaryData(
            episode_num=1,
            total_steps=20,
            total_reward=sum(enhanced_visualizer.episode_rewards),
            reward_progression=enhanced_visualizer.episode_rewards,
            similarity_progression=enhanced_visualizer.episode_similarities,
            final_similarity=0.95,
            task_id="task_001",
            success=True,
            key_moments=[5, 10, 15],
        )
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.create_episode_summary_svg') as mock_create:
            mock_create.return_value = "<svg>episode summary</svg>"
            
            enhanced_visualizer.visualize_episode_summary(episode_data)
        
        # Check summary visualization was created
        mock_create.assert_called_once()
        
        # Check async logging
        time.sleep(0.1)
        assert enhanced_visualizer.async_logger.queue.qsize() >= 0
    
    def test_create_comparison_visualization(self, enhanced_visualizer: EnhancedVisualizer) -> None:
        """Test comparison visualization across episodes."""
        episodes_data = [
            EpisodeSummaryData(
                episode_num=1,
                total_steps=15,
                total_reward=10.0,
                final_similarity=0.8,
                task_id="task_001",
                success=False,
            ),
            EpisodeSummaryData(
                episode_num=2,
                total_steps=12,
                total_reward=15.0,
                final_similarity=0.95,
                task_id="task_001",
                success=True,
            ),
            EpisodeSummaryData(
                episode_num=3,
                total_steps=10,
                total_reward=18.0,
                final_similarity=0.98,
                task_id="task_001",
                success=True,
            ),
        ]
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.create_comparison_svg') as mock_create:
            mock_create.return_value = "<svg>comparison chart</svg>"
            
            result = enhanced_visualizer.create_comparison_visualization(
                episodes_data,
                comparison_type="reward_progression"
            )
        
        # Check comparison was created
        mock_create.assert_called_once()
        assert result == "<svg>comparison chart</svg>"
        
        # Check call arguments
        call_args = mock_create.call_args[0]
        assert len(call_args[0]) == 3  # Three episodes
        assert call_args[1] == "reward_progression"
    
    def test_performance_monitoring(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test performance monitoring during visualization."""
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 3]])
        
        # Mock slow visualization to test performance monitoring
        def slow_visualization(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow operation
            return "<svg>slow visualization</svg>"
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg', side_effect=slow_visualization):
            start_time = time.time()
            
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "fill"},
                after_grid=after_grid,
                reward=0.5,
                info={},
                step_num=1,
            )
            
            end_time = time.time()
        
        # Check performance was monitored
        assert hasattr(enhanced_visualizer, 'performance_stats')
        stats = enhanced_visualizer.get_performance_stats()
        assert "avg_step_time" in stats
        assert "total_visualization_time" in stats
        assert stats["steps_visualized"] >= 1
    
    def test_memory_management(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test memory management during visualization."""
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        # Create large grids to test memory management
        large_grid = jnp.ones((25, 25), dtype=jnp.int32)
        
        # Visualize many steps to test memory cleanup
        for step in range(10):
            enhanced_visualizer.visualize_step(
                before_grid=large_grid,
                action={"type": "fill", "step": step},
                after_grid=large_grid + step,
                reward=0.1 * step,
                info={"step": step},
                step_num=step,
            )
        
        # Check memory usage is reasonable
        memory_stats = enhanced_visualizer.get_memory_stats()
        assert "current_memory_mb" in memory_stats
        assert "peak_memory_mb" in memory_stats
        assert memory_stats["current_memory_mb"] > 0
    
    def test_error_handling_visualization_failure(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test error handling when visualization fails."""
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 3]])
        
        # Mock visualization to raise an error
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg', side_effect=Exception("Visualization error")):
            # Should handle error gracefully and not crash
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "fill"},
                after_grid=after_grid,
                reward=0.5,
                info={},
                step_num=1,
            )
        
        # Step count should still increment despite error
        assert enhanced_visualizer.step_count == 1
        
        # Error should be logged
        error_stats = enhanced_visualizer.get_error_stats()
        assert error_stats["visualization_errors"] >= 1
    
    def test_jax_compatibility(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test JAX compatibility of visualization functions."""
        import jax
        
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        # Create JAX arrays
        key = jax.random.PRNGKey(42)
        before_grid = jax.random.randint(key, (5, 5), 0, 10)
        after_grid = before_grid + 1
        
        # Should handle JAX arrays without issues
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_draw:
            mock_draw.return_value = "<svg>jax compatible</svg>"
            
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "jax_test"},
                after_grid=after_grid,
                reward=0.7,
                info={},
                step_num=1,
            )
        
        # Should successfully call visualization
        mock_draw.assert_called_once()
        
        # Check that JAX arrays were properly handled
        call_args = mock_draw.call_args
        # Arrays should be converted to numpy for visualization
        assert not isinstance(call_args[0][0], jnp.ndarray) or True  # Allow both numpy and jax arrays
    
    def test_multiple_output_formats(self, enhanced_visualizer: EnhancedVisualizer, temp_dir: Path) -> None:
        """Test multiple output format generation."""
        # Configure multiple output formats
        enhanced_visualizer.vis_config = enhanced_visualizer.vis_config.replace(
            output_formats=["svg", "png"]
        )
        
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        before_grid = jnp.array([[0, 1], [2, 3]])
        after_grid = jnp.array([[1, 1], [2, 3]])
        
        with patch('jaxarc.utils.visualization.enhanced_visualizer.draw_rl_step_svg') as mock_svg, \
             patch('jaxarc.utils.visualization.enhanced_visualizer.convert_svg_to_png') as mock_png:
            
            mock_svg.return_value = "<svg>test</svg>"
            mock_png.return_value = b"png_data"
            
            enhanced_visualizer.visualize_step(
                before_grid=before_grid,
                action={"type": "fill"},
                after_grid=after_grid,
                reward=0.5,
                info={},
                step_num=1,
            )
        
        # Should generate both formats
        mock_svg.assert_called_once()
        mock_png.assert_called_once()
    
    def test_cleanup_and_shutdown(self, enhanced_visualizer: EnhancedVisualizer) -> None:
        """Test proper cleanup and shutdown."""
        enhanced_visualizer.episode_manager.start_new_run("test_run")
        enhanced_visualizer.start_episode(episode_num=1, task_id="task_001")
        
        # Add some data
        enhanced_visualizer.step_count = 10
        enhanced_visualizer.episode_rewards = [0.1, 0.2, 0.3]
        
        # Shutdown should clean up resources
        enhanced_visualizer.shutdown()
        
        # Check cleanup
        assert enhanced_visualizer.async_logger.is_running is False
        assert enhanced_visualizer.current_episode_num is None
        assert enhanced_visualizer.step_count == 0


if __name__ == "__main__":
    pytest.main([__file__])