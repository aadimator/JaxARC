"""Tests for task visualization functionality.

This module tests the task visualization generation and step visualization
with task context functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from jaxarc.utils.visualization.visualizer import (
    StepVisualizationData,
    TaskVisualizationData,
    VisualizationConfig,
    Visualizer,
)


class TestTaskVisualizationData:
    """Test TaskVisualizationData dataclass."""

    def test_task_visualization_data_creation(self):
        """Test creating TaskVisualizationData with required fields."""
        task_data = TaskVisualizationData(
            task_id="test_task_001",
            task_data={"train": [], "test": []},
            current_pair_index=0,
            episode_mode="train",
        )
        
        assert task_data.task_id == "test_task_001"
        assert task_data.current_pair_index == 0
        assert task_data.episode_mode == "train"
        assert task_data.metadata == {}

    def test_task_visualization_data_with_metadata(self):
        """Test TaskVisualizationData with custom metadata."""
        metadata = {"concept": "pattern_completion", "difficulty": "medium"}
        
        task_data = TaskVisualizationData(
            task_id="concept_task_001",
            task_data={"train": [], "test": []},
            current_pair_index=1,
            episode_mode="test",
            metadata=metadata,
        )
        
        assert task_data.metadata == metadata
        assert task_data.episode_mode == "test"
        assert task_data.current_pair_index == 1

    def test_task_visualization_data_episode_modes(self):
        """Test TaskVisualizationData with different episode modes."""
        for mode in ["train", "test"]:
            task_data = TaskVisualizationData(
                task_id=f"task_{mode}",
                task_data={"train": [], "test": []},
                current_pair_index=0,
                episode_mode=mode,
            )
            assert task_data.episode_mode == mode


class TestStepVisualizationDataWithTaskContext:
    """Test StepVisualizationData with task context fields."""

    def test_step_visualization_data_with_task_context(self):
        """Test StepVisualizationData with task context fields."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=5,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1, "color": 2},
            reward=1.5,
            info={"success": True},
            task_id="test_task_001",
            task_pair_index=2,
            total_task_pairs=5,
        )
        
        assert step_data.task_id == "test_task_001"
        assert step_data.task_pair_index == 2
        assert step_data.total_task_pairs == 5
        assert step_data.step_num == 5
        assert jnp.array_equal(step_data.before_grid, before_grid)
        assert jnp.array_equal(step_data.after_grid, after_grid)

    def test_step_visualization_data_default_task_context(self):
        """Test StepVisualizationData with default task context values."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 0},
            reward=0.0,
            info={},
        )
        
        # Check default values
        assert step_data.task_id == ""
        assert step_data.task_pair_index == 0
        assert step_data.total_task_pairs == 1

    def test_step_visualization_data_with_optional_fields(self):
        """Test StepVisualizationData with all optional fields."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        selection_mask = jnp.ones((3, 3), dtype=bool)
        changed_cells = jnp.array([[1, 1], [2, 2]])
        
        step_data = StepVisualizationData(
            step_num=10,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 5, "color": 3},
            reward=2.0,
            info={"operation_name": "fill"},
            task_id="complex_task",
            task_pair_index=3,
            total_task_pairs=8,
            selection_mask=selection_mask,
            changed_cells=changed_cells,
            operation_name="fill_operation",
        )
        
        assert step_data.operation_name == "fill_operation"
        assert jnp.array_equal(step_data.selection_mask, selection_mask)
        assert jnp.array_equal(step_data.changed_cells, changed_cells)
        assert step_data.task_pair_index == 3
        assert step_data.total_task_pairs == 8


class TestVisualizerTaskMethods:
    """Test Visualizer methods for task visualization."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def test_config(self, temp_output_dir):
        """Create test visualization configuration."""
        return VisualizationConfig(
            debug_level="standard",
            output_formats=["svg"],
            async_processing=False,  # Disable for testing
        )

    @pytest.fixture
    def test_visualizer(self, test_config, temp_output_dir):
        """Create test visualizer."""
        return Visualizer(test_config, output_dir=temp_output_dir)

    @pytest.fixture
    def sample_task_data(self):
        """Create sample task data for testing."""
        return {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
                },
                {
                    "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                    "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                }
            ],
            "test": [
                {
                    "input": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                    "output": [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
                }
            ]
        }

    def test_start_episode_with_task(self, test_visualizer, sample_task_data):
        """Test start_episode_with_task method."""
        with patch.object(test_visualizer, '_create_task_visualization') as mock_create:
            mock_create.return_value = Path("test_task.svg")
            
            test_visualizer.start_episode_with_task(
                episode_num=1,
                task_data=sample_task_data,
                task_id="test_task_001",
                current_pair_index=0,
                episode_mode="train"
            )
            
            # Verify that task visualization was created
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert isinstance(call_args, TaskVisualizationData)
            assert call_args.task_id == "test_task_001"
            assert call_args.current_pair_index == 0
            assert call_args.episode_mode == "train"

    def test_start_episode_with_task_default_values(self, test_visualizer, sample_task_data):
        """Test start_episode_with_task with default values."""
        with patch.object(test_visualizer, '_create_task_visualization') as mock_create:
            mock_create.return_value = Path("test_task.svg")
            
            test_visualizer.start_episode_with_task(
                episode_num=2,
                task_data=sample_task_data
            )
            
            # Verify default values were used
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert call_args.task_id == ""
            assert call_args.current_pair_index == 0
            assert call_args.episode_mode == "train"

    def test_create_task_visualization_basic(self, test_visualizer, sample_task_data):
        """Test _create_task_visualization basic functionality."""
        task_viz_data = TaskVisualizationData(
            task_id="test_task_001",
            task_data=sample_task_data,
            current_pair_index=0,
            episode_mode="train",
        )
        
        with patch('jaxarc.utils.visualization.task_visualization.draw_parsed_task_data_svg') as mock_draw:
            mock_draw.return_value = "svg_content"
            
            result = test_visualizer._create_task_visualization(task_viz_data)
            
            # Should attempt to create visualization
            mock_draw.assert_called_once()
            # Result should be a Path or None
            assert result is None or isinstance(result, Path)

    def test_create_task_visualization_with_invalid_data(self, test_visualizer):
        """Test _create_task_visualization with invalid task data."""
        task_viz_data = TaskVisualizationData(
            task_id="invalid_task",
            task_data=None,  # Invalid data
            current_pair_index=0,
            episode_mode="train",
        )
        
        result = test_visualizer._create_task_visualization(task_viz_data)
        
        # Should handle invalid data gracefully
        assert result is None

    def test_create_task_visualization_error_handling(self, test_visualizer, sample_task_data):
        """Test _create_task_visualization error handling."""
        task_viz_data = TaskVisualizationData(
            task_id="error_task",
            task_data=sample_task_data,
            current_pair_index=0,
            episode_mode="train",
        )
        
        with patch('jaxarc.utils.visualization.task_visualization.draw_parsed_task_data_svg') as mock_draw:
            mock_draw.side_effect = Exception("Drawing error")
            
            result = test_visualizer._create_task_visualization(task_viz_data)
            
            # Should handle errors gracefully
            assert result is None

    def test_task_visualization_integration_with_episode_manager(self, test_visualizer, sample_task_data):
        """Test task visualization integration with episode manager."""
        with patch.object(test_visualizer.episode_manager, 'start_episode') as mock_start:
            with patch.object(test_visualizer, '_create_task_visualization') as mock_create:
                mock_create.return_value = Path("task.svg")
                
                test_visualizer.start_episode_with_task(
                    episode_num=3,
                    task_data=sample_task_data,
                    task_id="integration_test"
                )
                
                # Verify episode manager was called
                mock_start.assert_called_once_with(3, "integration_test")
                
                # Verify task visualization was created
                mock_create.assert_called_once()

    def test_step_visualization_with_task_context(self, test_visualizer):
        """Test step visualization includes task context."""
        before_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        after_grid = jnp.ones((3, 3), dtype=jnp.int32)
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1, "color": 2},
            reward=1.0,
            info={},
            task_id="context_test_task",
            task_pair_index=2,
            total_task_pairs=5,
        )
        
        with patch.object(test_visualizer, '_create_step_visualization') as mock_create:
            mock_create.return_value = Path("step.svg")
            
            test_visualizer.log_step(step_data)
            
            # Verify step visualization was created with task context
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert call_args.task_id == "context_test_task"
            assert call_args.task_pair_index == 2
            assert call_args.total_task_pairs == 5


class TestTaskVisualizationIntegration:
    """Test integration of task visualization with the overall system."""

    def test_task_visualization_data_serialization(self):
        """Test that TaskVisualizationData can be serialized/deserialized."""
        original_data = TaskVisualizationData(
            task_id="serialization_test",
            task_data={"train": [{"input": [[1, 0]], "output": [[0, 1]]}], "test": []},
            current_pair_index=1,
            episode_mode="test",
            metadata={"test": True},
        )
        
        # Test that all fields are accessible
        assert original_data.task_id == "serialization_test"
        assert original_data.current_pair_index == 1
        assert original_data.episode_mode == "test"
        assert original_data.metadata["test"] is True

    def test_step_visualization_data_with_jax_arrays(self):
        """Test StepVisualizationData works with JAX arrays."""
        before_grid = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.int32)
        after_grid = jnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
        selection_mask = jnp.array([[True, False, True], [False, True, False], [True, False, True]])
        
        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=after_grid,
            action={"operation": 1},
            reward=1.0,
            info={},
            task_id="jax_test",
            task_pair_index=0,
            total_task_pairs=3,
            selection_mask=selection_mask,
        )
        
        # Verify JAX arrays are preserved
        assert isinstance(step_data.before_grid, jnp.ndarray)
        assert isinstance(step_data.after_grid, jnp.ndarray)
        assert isinstance(step_data.selection_mask, jnp.ndarray)
        assert step_data.before_grid.dtype == jnp.int32
        assert step_data.selection_mask.dtype == bool

    def test_visualization_config_affects_task_visualization(self):
        """Test that visualization config affects task visualization behavior."""
        # Test with debug_level "off"
        config_off = VisualizationConfig(debug_level="off")
        assert not config_off.should_visualize_step(1)
        
        # Test with debug_level "standard"
        config_standard = VisualizationConfig(debug_level="standard")
        assert config_standard.should_visualize_step(1)
        
        # Test with different output formats
        config_multiple = VisualizationConfig(output_formats=["svg", "png"])
        assert "svg" in config_multiple.output_formats
        assert "png" in config_multiple.output_formats

    def test_task_visualization_with_empty_task_data(self):
        """Test task visualization handles empty task data gracefully."""
        empty_task_data = TaskVisualizationData(
            task_id="empty_task",
            task_data={"train": [], "test": []},
            current_pair_index=0,
            episode_mode="train",
        )
        
        # Should not raise errors
        assert empty_task_data.task_id == "empty_task"
        assert empty_task_data.task_data["train"] == []
        assert empty_task_data.task_data["test"] == []

    def test_task_visualization_with_large_task_data(self):
        """Test task visualization handles large task data."""
        # Create task with many pairs
        large_task_data = {
            "train": [
                {"input": [[i % 2, (i+1) % 2]], "output": [[(i+1) % 2, i % 2]]}
                for i in range(10)
            ],
            "test": [
                {"input": [[i % 3, (i+1) % 3, (i+2) % 3]], "output": [[(i+2) % 3, (i+1) % 3, i % 3]]}
                for i in range(5)
            ]
        }
        
        task_viz_data = TaskVisualizationData(
            task_id="large_task",
            task_data=large_task_data,
            current_pair_index=0,
            episode_mode="train",
        )
        
        # Should handle large data without issues
        assert len(task_viz_data.task_data["train"]) == 10
        assert len(task_viz_data.task_data["test"]) == 5