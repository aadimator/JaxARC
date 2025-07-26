"""Tests for the enhanced visualization system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from jaxarc.types import Grid
from jaxarc.utils.visualization.visualizer import (
    EpisodeSummaryData,
    StepVisualizationData,
    VisualizationConfig,
    Visualizer,
)


class TestVisualizationConfig:
    """Test VisualizationConfig dataclass."""

    def test_default_config(self):
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

    def test_config_validation_debug_level(self):
        """Test debug level validation."""
        # Valid debug levels
        for level in ["off", "minimal", "standard", "verbose", "full"]:
            config = VisualizationConfig(debug_level=level)
            assert config.debug_level == level

        # Invalid debug level
        with pytest.raises(ValueError, match="debug_level must be one of"):
            VisualizationConfig(debug_level="invalid")

    def test_config_validation_output_formats(self):
        """Test output format validation."""
        # Valid formats
        config = VisualizationConfig(output_formats=["svg", "png", "html"])
        assert config.output_formats == ["svg", "png", "html"]

        # Invalid format
        with pytest.raises(ValueError, match="Invalid output format"):
            VisualizationConfig(output_formats=["invalid"])

    def test_config_validation_image_quality(self):
        """Test image quality validation."""
        # Valid qualities
        for quality in ["low", "medium", "high"]:
            config = VisualizationConfig(image_quality=quality)
            assert config.image_quality == quality

        # Invalid quality
        with pytest.raises(ValueError, match="image_quality must be one of"):
            VisualizationConfig(image_quality="invalid")

    def test_config_validation_color_scheme(self):
        """Test color scheme validation."""
        # Valid schemes
        for scheme in ["default", "colorblind", "high_contrast"]:
            config = VisualizationConfig(color_scheme=scheme)
            assert config.color_scheme == scheme

        # Invalid scheme
        with pytest.raises(ValueError, match="color_scheme must be one of"):
            VisualizationConfig(color_scheme="invalid")

    def test_config_validation_max_concurrent_saves(self):
        """Test max concurrent saves validation."""
        # Valid value
        config = VisualizationConfig(max_concurrent_saves=8)
        assert config.max_concurrent_saves == 8

        # Invalid value
        with pytest.raises(ValueError, match="max_concurrent_saves must be at least 1"):
            VisualizationConfig(max_concurrent_saves=0)

    def test_should_visualize_step(self):
        """Test step visualization decision logic."""
        # Off - never visualize
        config = VisualizationConfig(debug_level="off")
        assert config.should_visualize_step(0) is False
        assert config.should_visualize_step(10) is False

        # Minimal - never visualize steps
        config = VisualizationConfig(debug_level="minimal")
        assert config.should_visualize_step(0) is False
        assert config.should_visualize_step(10) is False

        # Standard - every 10th step
        config = VisualizationConfig(debug_level="standard")
        assert config.should_visualize_step(0) is True
        assert config.should_visualize_step(10) is True
        assert config.should_visualize_step(5) is False

        # Verbose - every 5th step
        config = VisualizationConfig(debug_level="verbose")
        assert config.should_visualize_step(0) is True
        assert config.should_visualize_step(5) is True
        assert config.should_visualize_step(3) is False

        # Full - every step
        config = VisualizationConfig(debug_level="full")
        assert config.should_visualize_step(0) is True
        assert config.should_visualize_step(1) is True
        assert config.should_visualize_step(100) is True

    def test_should_visualize_episode_summary(self):
        """Test episode summary visualization decision logic."""
        # Off - no summaries
        config = VisualizationConfig(debug_level="off")
        assert config.should_visualize_episode_summary() is False

        # All other levels - yes summaries
        for level in ["minimal", "standard", "verbose", "full"]:
            config = VisualizationConfig(debug_level=level)
            assert config.should_visualize_episode_summary() is True

    def test_get_color_palette(self):
        """Test color palette retrieval."""
        # Default palette
        config = VisualizationConfig(color_scheme="default")
        palette = config.get_color_palette()
        assert len(palette) == 11  # Colors 0-10
        assert all(isinstance(color, str) for color in palette.values())
        assert all(color.startswith("#") for color in palette.values())

        # Colorblind palette
        config = VisualizationConfig(color_scheme="colorblind")
        colorblind_palette = config.get_color_palette()
        assert len(colorblind_palette) == 11
        assert colorblind_palette != palette  # Should be different

        # High contrast palette
        config = VisualizationConfig(color_scheme="high_contrast")
        high_contrast_palette = config.get_color_palette()
        assert len(high_contrast_palette) == 11
        assert high_contrast_palette != palette  # Should be different


class TestStepVisualizationData:
    """Test StepVisualizationData dataclass."""

    def test_step_data_creation(self):
        """Test creating step visualization data."""
        before_grid = Grid(
            data=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
            mask=jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
        )
        after_grid = Grid(
            data=jnp.array([[1, 1], [2, 3]], dtype=jnp.int32),
            mask=jnp.array([[True, True], [True, True]], dtype=jnp.bool_),
        )

        step_data = StepVisualizationData(
            step_num=5,
            before_grid=before_grid,
            after_grid=after_grid,
            action={
                "operation": 1,
                "selection": jnp.array([[True, False], [False, False]]),
            },
            reward=0.5,
            info={"similarity": 0.8},
            operation_name="Fill 1",
            task_id="test_task",
            task_pair_index=0,
            total_task_pairs=1,
        )

        assert step_data.step_num == 5
        assert step_data.reward == 0.5
        assert step_data.operation_name == "Fill 1"
        assert step_data.timestamp > 0  # Should be set automatically

    def test_step_data_optional_fields(self):
        """Test step data with optional fields."""
        before_grid = Grid(
            data=jnp.array([[0, 1]], dtype=jnp.int32),
            mask=jnp.array([[True, True]], dtype=jnp.bool_),
        )

        step_data = StepVisualizationData(
            step_num=1,
            before_grid=before_grid,
            after_grid=before_grid,
            action={},
            reward=0.0,
            info={},
            task_id="",
            task_pair_index=0,
            total_task_pairs=1,
        )

        assert step_data.selection_mask is None
        assert step_data.changed_cells is None
        assert step_data.operation_name == ""


class TestEpisodeSummaryData:
    """Test EpisodeSummaryData dataclass."""

    def test_episode_summary_creation(self):
        """Test creating episode summary data."""
        summary = EpisodeSummaryData(
            episode_num=10,
            total_steps=50,
            total_reward=25.5,
            reward_progression=[0.0, 0.5, 1.0, 0.8, 1.2],
            similarity_progression=[0.1, 0.3, 0.5, 0.7, 0.9],
            final_similarity=0.9,
            task_id="task_001",
            success=True,
            key_moments=[10, 25, 40],
            start_time=1000.0,
        )

        assert summary.episode_num == 10
        assert summary.total_steps == 50
        assert summary.success is True
        assert len(summary.key_moments) == 3
        assert summary.end_time > summary.start_time  # Should be set automatically

    def test_episode_summary_auto_end_time(self):
        """Test automatic end time setting."""
        summary = EpisodeSummaryData(
            episode_num=1,
            total_steps=10,
            total_reward=5.0,
            reward_progression=[],
            similarity_progression=[],
            final_similarity=0.5,
            task_id="test",
            success=False,
        )

        assert summary.end_time > 0  # Should be set automatically


class TestVisualizer:
    """Test Visualizer class."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        config = VisualizationConfig()

        with patch(
            "jaxarc.utils.visualization.visualizer.EpisodeManager"
        ) as mock_em:
            with patch(
                "jaxarc.utils.visualization.visualizer.AsyncLogger"
            ) as mock_al:
                visualizer = Visualizer(config)

                assert visualizer.config == config
                assert visualizer.current_episode_num is None
                assert len(visualizer.current_episode_data) == 0

                # Should create episode manager and async logger
                mock_em.assert_called_once()
                mock_al.assert_called_once()

    def test_visualizer_with_wandb_disabled(self):
        """Test visualizer with wandb disabled."""
        config = VisualizationConfig()
        config.wandb_config.enabled = False

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)

                assert visualizer.wandb_integration is None

    def test_visualizer_with_wandb_enabled(self):
        """Test visualizer with wandb enabled."""
        config = VisualizationConfig()
        config.wandb_config.enabled = True

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                with patch(
                    "jaxarc.utils.visualization.visualizer.WandbIntegration"
                ) as mock_wandb:
                    visualizer = Visualizer(config)

                    mock_wandb.assert_called_once()
                    assert visualizer.wandb_integration is not None

    def test_start_episode(self):
        """Test starting a new episode."""
        config = VisualizationConfig(debug_level="standard")

        with patch(
            "jaxarc.utils.visualization.visualizer.EpisodeManager"
        ) as mock_em_class:
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                mock_em = MagicMock()
                mock_em.start_new_episode.return_value = Path("/test/episode/1")
                mock_em_class.return_value = mock_em

                visualizer = Visualizer(config)
                visualizer.start_episode(1, "task_001")

                assert visualizer.current_episode_num == 1
                assert len(visualizer.current_episode_data) == 0
                mock_em.start_new_episode.assert_called_once_with(1)

    def test_start_episode_debug_off(self):
        """Test starting episode with debug off."""
        config = VisualizationConfig(debug_level="off")

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)
                visualizer.start_episode(1, "task_001")

                # Should not set episode data when debug is off
                assert visualizer.current_episode_num is None

    def test_visualize_step_skip_based_on_config(self):
        """Test step visualization skipping based on config."""
        config = VisualizationConfig(debug_level="standard")  # Every 10th step

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)

                step_data = StepVisualizationData(
                    step_num=5,  # Not divisible by 10
                    before_grid=MagicMock(),
                    after_grid=MagicMock(),
                    action={},
                    reward=0.0,
                    info={},
                    task_id="",
                    task_pair_index=0,
                    total_task_pairs=1,
                )

                result = visualizer.visualize_step(step_data)

                # Should return None (skipped)
                assert result is None

    def test_visualize_step_create_svg(self):
        """Test step visualization SVG creation."""
        config = VisualizationConfig(debug_level="full", output_formats=["svg"])

        with patch(
            "jaxarc.utils.visualization.visualizer.EpisodeManager"
        ) as mock_em_class:
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                mock_em = MagicMock()
                mock_em.get_step_path.return_value = Path("/test/step.svg")
                mock_em_class.return_value = mock_em

                visualizer = Visualizer(config)
                visualizer.current_episode_num = 1

                step_data = StepVisualizationData(
                    step_num=1,
                    before_grid=MagicMock(),
                    after_grid=MagicMock(),
                    action={"operation": 1},
                    reward=0.5,
                    info={},
                    task_id="",
                    task_pair_index=0,
                    total_task_pairs=1,
                )

                with patch.object(
                    visualizer, "_create_step_svg", return_value=Path("/test/step.svg")
                ):
                    result = visualizer.visualize_step(step_data)

                    assert result == Path("/test/step.svg")
                    assert len(visualizer.current_episode_data) == 1

    def test_visualize_episode_summary(self):
        """Test episode summary visualization."""
        config = VisualizationConfig(debug_level="standard")

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)

                summary_data = EpisodeSummaryData(
                    episode_num=1,
                    total_steps=10,
                    total_reward=5.0,
                    reward_progression=[],
                    similarity_progression=[],
                    final_similarity=0.5,
                    task_id="test",
                    success=False,
                )

                with patch.object(
                    visualizer,
                    "_create_episode_summary",
                    return_value=Path("/test/summary.svg"),
                ):
                    result = visualizer.visualize_episode_summary(summary_data)

                    assert result == Path("/test/summary.svg")
                    # Should clear episode data
                    assert len(visualizer.current_episode_data) == 0
                    assert visualizer.current_episode_num is None

    def test_visualize_episode_summary_debug_off(self):
        """Test episode summary with debug off."""
        config = VisualizationConfig(debug_level="off")

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)

                summary_data = EpisodeSummaryData(
                    episode_num=1,
                    total_steps=10,
                    total_reward=5.0,
                    reward_progression=[],
                    similarity_progression=[],
                    final_similarity=0.5,
                    task_id="test",
                    success=False,
                )

                result = visualizer.visualize_episode_summary(summary_data)

                # Should return None when debug is off
                assert result is None

    def test_create_step_svg_error_handling(self):
        """Test error handling in step SVG creation."""
        config = VisualizationConfig()

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)
                visualizer.current_episode_num = 1

                step_data = StepVisualizationData(
                    step_num=1,
                    before_grid=MagicMock(),
                    after_grid=MagicMock(),
                    action={},
                    reward=0.0,
                    info={},
                    task_id="",
                    task_pair_index=0,
                    total_task_pairs=1,
                )

                # Test that the method handles errors gracefully
                # Since _create_step_svg is a real method, let's test error handling in visualize_step
                with patch.object(visualizer, "_create_step_svg", return_value=None):
                    result = visualizer.visualize_step(step_data)

                    # Should return None when step creation fails
                    assert result is None

    def test_get_performance_report(self):
        """Test performance report generation."""
        config = VisualizationConfig()

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                visualizer = Visualizer(config)

                # Update some performance stats
                visualizer._update_performance_stats(0.1)
                visualizer._update_performance_stats(0.2)

                report = visualizer.get_performance_report()

                assert "total_visualizations" in report
                assert "total_time" in report
                assert "avg_time_per_viz" in report
                assert "debug_level" in report
                assert report["total_visualizations"] == 2
                assert report["debug_level"] == "standard"

    def test_cleanup(self):
        """Test visualizer cleanup."""
        config = VisualizationConfig()

        with patch(
            "jaxarc.utils.visualization.visualizer.EpisodeManager"
        ) as mock_em_class:
            with patch(
                "jaxarc.utils.visualization.visualizer.AsyncLogger"
            ) as mock_al_class:
                mock_em = MagicMock()
                mock_al = MagicMock()
                mock_em_class.return_value = mock_em
                mock_al_class.return_value = mock_al

                visualizer = Visualizer(config)
                visualizer.cleanup()

                # Should call cleanup methods
                mock_al.flush.assert_called_once()
                mock_em.cleanup_old_data.assert_called_once()

    def test_cleanup_with_wandb(self):
        """Test cleanup with wandb integration."""
        config = VisualizationConfig()
        config.wandb_config.enabled = True

        with patch("jaxarc.utils.visualization.visualizer.EpisodeManager"):
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                with patch(
                    "jaxarc.utils.visualization.visualizer.WandbIntegration"
                ) as mock_wandb_class:
                    mock_wandb = MagicMock()
                    mock_wandb_class.return_value = mock_wandb

                    visualizer = Visualizer(config)
                    visualizer.cleanup()

                    # Should finish wandb run
                    mock_wandb.finish_run.assert_called_once()

    def test_create_comparison_visualization(self):
        """Test comparison visualization creation."""
        config = VisualizationConfig()

        with patch(
            "jaxarc.utils.visualization.visualizer.EpisodeManager"
        ) as mock_em_class:
            with patch("jaxarc.utils.visualization.visualizer.AsyncLogger"):
                mock_em = MagicMock()
                mock_em.config.get_base_path.return_value = Path("/test/base")
                mock_em_class.return_value = mock_em

                visualizer = Visualizer(config)

                episodes_data = [
                    EpisodeSummaryData(
                        episode_num=1,
                        total_steps=10,
                        total_reward=5.0,
                        reward_progression=[],
                        similarity_progression=[],
                        final_similarity=0.5,
                        task_id="test",
                        success=False,
                    )
                ]

                # Test that the method works correctly
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / "comparison.svg"

                    result = visualizer.create_comparison_visualization(
                        episodes_data, "reward_progression", output_path
                    )

                    # Should return the output path when successful
                    assert result == output_path
                    assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
