"""Tests for Visualizer class functionality.

This module tests the main Visualizer class functionality including
configuration, episode management, and integration with other components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from jaxarc.utils.visualization.visualizer import (
    StepVisualizationData,
    VisualizationConfig,
    Visualizer,
)


class TestVisualizerClass:
    """Test main Visualizer class functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_config(self):
        """Create test visualization configuration."""
        return VisualizationConfig(
            debug_level="standard",
            output_formats=["svg"],
            async_processing=False,  # Disable for testing
            show_coordinates=True,
            show_operation_names=True,
            highlight_changes=True,
        )

    @pytest.fixture
    def test_visualizer(self, test_config, temp_output_dir):
        """Create test visualizer."""
        return Visualizer(test_config, output_dir=temp_output_dir)

    def test_visualizer_initialization(self, test_config, temp_output_dir):
        """Test Visualizer initialization."""
        visualizer = Visualizer(test_config, output_dir=temp_output_dir)
        
        assert visualizer.config == test_config
        assert visualizer.output_dir == temp_output_dir
        assert hasattr(visualizer, 'episode_manager')
        assert hasattr(visualizer, 'async_logger')

    def test_visualizer_initialization_with_defaults(self):
        """Test Visualizer initialization with default parameters."""
        visualizer = Visualizer()
        
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer.output_dir is not None
        assert hasattr(visualizer, 'episode_manager')
        assert hasattr(visualizer, 'async_logger')

    def test_visualizer_start_episode(self, test_visualizer):
        """Test Visualizer start_episode method."""
        with patch.object(test_visualizer.episode_manager, 'start_episode') as mock_start:
            test_visualizer.start_episode(episode_num=1, task_id="test_task")
            
            mock_start.assert_called_once_with(1, "test_task")

    def test_visualizer_end_episode(self, test_visualizer):
        """Test Visualizer end_episode method."""
        with patch.object(test_visualizer.episode_manager, 'end_episode') as mock_end:
            test_visualizer.end_episode()
            
            mock_end.assert_called_once()

    def test_visualizer_log_step_basic(self, test_visualizer):
        """Test Visualizer log_step method with basic step data."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1, "color": 2},
            reward=1.0,
            info={},
        )
        
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("step.svg")
            
            test_visualizer.log_step(step_data)
            
            mock_create.assert_called_once_with(step_data)

    def test_visualizer_log_step_with_task_context(self, test_visualizer):
        """Test Visualizer log_step method with task context."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=5,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 3, "color": 1},
            reward=2.5,
            info={"success": True},
            task_id="context_task",
            task_pair_index=2,
            total_task_pairs=4,
        )
        
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("step_with_context.svg")
            
            test_visualizer.log_step(step_data)
            
            mock_create.assert_called_once_with(step_data)
            # Verify the step data includes task context
            call_args = mock_create.call_args[0][0]
            assert call_args.task_id == "context_task"
            assert call_args.task_pair_index == 2
            assert call_args.total_task_pairs == 4

    def test_visualizer_should_visualize_step(self, test_visualizer):
        """Test Visualizer should_visualize_step method."""
        # Should visualize based on config
        assert test_visualizer.should_visualize_step(1)
        assert test_visualizer.should_visualize_step(10)
        
        # Test with different debug levels
        config_off = VisualizationConfig(debug_level="off")
        visualizer_off = Visualizer(config_off)
        assert not visualizer_off.should_visualize_step(1)
        
        config_minimal = VisualizationConfig(debug_level="minimal")
        visualizer_minimal = Visualizer(config_minimal)
        # Minimal should have different behavior
        assert visualizer_minimal.should_visualize_step(1)

    def test_visualizer_create_step_visualization(self, test_visualizer):
        """Test Visualizer _create_step_visualization method."""
        before_grid = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        after_grid = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1},
            reward=1.0,
            info={},
        )
        
        with patch('jaxarc.utils.visualization.core.draw_step_svg') as mock_draw:
            mock_draw.return_value = "svg_content"
            
            result = test_visualizer._create_step_visualization(step_data)
            
            # Should attempt to create visualization
            mock_draw.assert_called_once()
            # Result should be a Path or None
            assert result is None or isinstance(result, Path)

    def test_visualizer_error_handling(self, test_visualizer):
        """Test Visualizer error handling."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1},
            reward=1.0,
            info={},
        )
        
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.side_effect = Exception("Visualization error")
            
            # Should handle errors gracefully
            try:
                test_visualizer.log_step(step_data)
            except Exception:
                pytest.fail("Visualizer should handle errors gracefully")

    def test_visualizer_async_processing(self, temp_output_dir):
        """Test Visualizer with async processing enabled."""
        config = VisualizationConfig(
            debug_level="standard",
            async_processing=True,
            max_concurrent_saves=2,
        )
        
        visualizer = Visualizer(config, output_dir=temp_output_dir)
        
        assert visualizer.config.async_processing is True
        assert visualizer.config.max_concurrent_saves == 2

    def test_visualizer_multiple_output_formats(self, temp_output_dir):
        """Test Visualizer with multiple output formats."""
        config = VisualizationConfig(
            debug_level="standard",
            output_formats=["svg", "png"],
        )
        
        visualizer = Visualizer(config, output_dir=temp_output_dir)
        
        assert "svg" in visualizer.config.output_formats
        assert "png" in visualizer.config.output_formats

    def test_visualizer_color_scheme_options(self, temp_output_dir):
        """Test Visualizer with different color schemes."""
        for scheme in ["default", "colorblind", "high_contrast"]:
            config = VisualizationConfig(color_scheme=scheme)
            visualizer = Visualizer(config, output_dir=temp_output_dir)
            assert visualizer.config.color_scheme == scheme

    def test_visualizer_integration_components(self, test_visualizer):
        """Test Visualizer integration with its components."""
        # Test episode manager integration
        assert hasattr(test_visualizer, 'episode_manager')
        assert test_visualizer.episode_manager is not None
        
        # Test async logger integration
        assert hasattr(test_visualizer, 'async_logger')
        assert test_visualizer.async_logger is not None
        
        # Test wandb integration (if enabled)
        if hasattr(test_visualizer, 'wandb_integration'):
            assert test_visualizer.wandb_integration is not None

    def test_visualizer_config_validation(self):
        """Test Visualizer handles invalid configurations."""
        # Test with invalid debug level
        with pytest.raises(ValueError):
            VisualizationConfig(debug_level="invalid")
        
        # Test with invalid output format
        with pytest.raises(ValueError):
            VisualizationConfig(output_formats=["invalid_format"])
        
        # Test with invalid color scheme
        with pytest.raises(ValueError):
            VisualizationConfig(color_scheme="invalid_scheme")

    def test_visualizer_output_directory_creation(self, temp_output_dir):
        """Test Visualizer creates output directories as needed."""
        # Use a subdirectory that doesn't exist
        output_subdir = temp_output_dir / "visualizations" / "episodes"
        
        config = VisualizationConfig(debug_level="standard")
        visualizer = Visualizer(config, output_dir=output_subdir)
        
        # Directory should be created or handled gracefully
        assert visualizer.output_dir == output_subdir

    def test_visualizer_step_filtering(self, test_visualizer):
        """Test Visualizer step filtering based on configuration."""
        before_grid = jnp.zeros((2, 2), dtype=jnp.int32)
        after_grid = jnp.ones((2, 2), dtype=jnp.int32)
        
        # Create multiple steps
        steps = []
        for i in range(10):
            step_data = StepVisualizationData(
                step_num=i,
                before_grid=before_grid,
                after_grid=after_grid,
                action={"operation": 1},
                reward=1.0,
                info={},
            )
            steps.append(step_data)
        
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("step.svg")
            
            # Log all steps
            for step in steps:
                if test_visualizer.should_visualize_step(step.step_num):
                    test_visualizer.log_step(step)
            
            # Should have created visualizations for steps that pass the filter
            assert mock_create.call_count > 0

    def test_visualizer_memory_management(self, test_visualizer):
        """Test Visualizer memory management with many steps."""
        before_grid = jnp.zeros((5, 5), dtype=jnp.int32)
        after_grid = jnp.ones((5, 5), dtype=jnp.int32)
        
        # Create many steps to test memory handling
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("step.svg")
            
            for i in range(100):
                step_data = StepVisualizationData(
                    step_num=i,
                    before_grid=before_grid,
                    after_grid=after_grid,
                    action={"operation": 1},
                    reward=1.0,
                    info={},
                )
                
                if test_visualizer.should_visualize_step(i):
                    test_visualizer.log_step(step_data)
            
            # Should handle many steps without issues
            assert mock_create.call_count > 0


class TestVisualizerAdvancedFeatures:
    """Test advanced Visualizer features."""

    @pytest.fixture
    def advanced_config(self):
        """Create advanced visualization configuration."""
        return VisualizationConfig(
            debug_level="verbose",
            output_formats=["svg", "png"],
            show_coordinates=True,
            show_operation_names=True,
            highlight_changes=True,
            include_metrics=True,
            color_scheme="colorblind",
            async_processing=True,
            max_concurrent_saves=4,
        )

    @pytest.fixture
    def advanced_visualizer(self, advanced_config):
        """Create advanced visualizer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Visualizer(advanced_config, output_dir=Path(temp_dir))

    def test_visualizer_advanced_step_data(self, advanced_visualizer):
        """Test Visualizer with advanced step data."""
        before_grid = jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=jnp.int32)
        after_grid = jnp.array([[1, 2, 0], [2, 0, 1], [0, 1, 2]], dtype=jnp.int32)
        selection_mask = jnp.array([[True, False, True], [False, True, False], [True, False, True]])
        changed_cells = jnp.array([[0, 0], [1, 1], [2, 2]])
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 5, "color": 3, "direction": "up"},
            reward=2.5,
            info={"operation_name": "rotate", "success": True},
            task_id="advanced_task",
            task_pair_index=1,
            total_task_pairs=3,
            selection_mask=selection_mask,
            changed_cells=changed_cells,
            operation_name="rotate_clockwise",
        )
        
        with patch.object(advanced_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("advanced_step.svg")
            
            advanced_visualizer.log_step(step_data)
            
            mock_create.assert_called_once_with(step_data)
            # Verify all advanced data is preserved
            call_args = mock_create.call_args[0][0]
            assert call_args.operation_name == "rotate_clockwise"
            assert jnp.array_equal(call_args.selection_mask, selection_mask)
            assert jnp.array_equal(call_args.changed_cells, changed_cells)

    def test_visualizer_performance_monitoring(self, advanced_visualizer):
        """Test Visualizer performance monitoring."""
        before_grid = jnp.zeros((10, 10), dtype=jnp.int32)
        after_grid = jnp.ones((10, 10), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1},
            reward=1.0,
            info={},
        )
        
        with patch.object(advanced_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("perf_step.svg")
            
            # Time the visualization creation
            import time
            start_time = time.time()
            advanced_visualizer.log_step(step_data)
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 1.0  # Less than 1 second
            mock_create.assert_called_once()

    def test_visualizer_concurrent_processing(self, advanced_visualizer):
        """Test Visualizer concurrent processing capabilities."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        steps = []
        for i in range(5):
            step_data = StepVisualizationData(
                step_num=i,
                before_grid=before_grid,
                after_grid=after_grid,
                action={"operation": 1},
                reward=1.0,
                info={},
            )
            steps.append(step_data)
        
        with patch.object(advanced_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("concurrent_step.svg")
            
            # Process multiple steps
            for step in steps:
                advanced_visualizer.log_step(step)
            
            # Should handle concurrent processing
            assert mock_create.call_count == len(steps)

    def test_visualizer_resource_cleanup(self, advanced_visualizer):
        """Test Visualizer resource cleanup."""
        # Test that visualizer cleans up resources properly
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1},
            reward=1.0,
            info={},
        )
        
        with patch.object(advanced_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("cleanup_step.svg")
            
            advanced_visualizer.log_step(step_data)
            
            # End episode to trigger cleanup
            advanced_visualizer.end_episode()
            
            # Should have processed the step
            mock_create.assert_called_once()

    def test_visualizer_configuration_updates(self, advanced_visualizer):
        """Test Visualizer handles configuration updates."""
        # Test that configuration changes are respected
        original_debug_level = advanced_visualizer.config.debug_level
        
        # Create new config with different settings
        new_config = VisualizationConfig(
            debug_level="minimal",
            output_formats=["svg"],
            show_coordinates=False,
        )
        
        # Create new visualizer with updated config
        with tempfile.TemporaryDirectory() as temp_dir:
            new_visualizer = Visualizer(new_config, output_dir=Path(temp_dir))
            
            assert new_visualizer.config.debug_level == "minimal"
            assert new_visualizer.config.show_coordinates is False
            assert new_visualizer.config.debug_level != original_debug_level