"""Enhanced visualization system for JaxARC.

This module provides the main VisualizationConfig and EnhancedVisualizer classes
that integrate all visualization components including episode management,
async logging, and wandb integration.
"""

from __future__ import annotations

import time
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import chex
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jaxarc.types import Grid

from .async_logger import AsyncLogger, AsyncLoggerConfig
from .episode_manager import EpisodeConfig, EpisodeManager
from .wandb_integration import WandbConfig, WandbIntegration


@chex.dataclass
class VisualizationConfig:
    """Configuration for visualization generation and management.

    This configuration controls all aspects of visualization including debug levels,
    output formats, color schemes, and integration with storage and logging systems.
    """

    # Debug level controls what gets visualized
    debug_level: Literal["off", "minimal", "standard", "verbose", "full"] = "standard"

    # Output format settings
    output_formats: List[str] = field(default_factory=lambda: ["svg"])
    image_quality: Literal["low", "medium", "high"] = "high"

    # Visual appearance settings
    show_coordinates: bool = False
    show_operation_names: bool = True
    highlight_changes: bool = True
    include_metrics: bool = True

    # Color scheme and accessibility
    color_scheme: Literal["default", "colorblind", "high_contrast"] = "default"
    use_double_width: bool = True
    show_numbers: bool = False

    # Performance and storage settings
    async_processing: bool = True
    max_concurrent_saves: int = 4
    compress_images: bool = False

    # Integration configurations
    episode_config: EpisodeConfig = field(default_factory=EpisodeConfig)
    async_logger_config: AsyncLoggerConfig = field(default_factory=AsyncLoggerConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        # Validate debug level
        valid_debug_levels = {"off", "minimal", "standard", "verbose", "full"}
        if self.debug_level not in valid_debug_levels:
            raise ValueError(f"debug_level must be one of {valid_debug_levels}")

        # Validate output formats
        valid_formats = {"svg", "png", "html"}
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid output format '{fmt}'. Must be one of {valid_formats}"
                )

        # Validate image quality
        valid_qualities = {"low", "medium", "high"}
        if self.image_quality not in valid_qualities:
            raise ValueError(f"image_quality must be one of {valid_qualities}")

        # Validate color scheme
        valid_schemes = {"default", "colorblind", "high_contrast"}
        if self.color_scheme not in valid_schemes:
            raise ValueError(f"color_scheme must be one of {valid_schemes}")

        # Validate performance settings
        if self.max_concurrent_saves < 1:
            raise ValueError("max_concurrent_saves must be at least 1")

    def should_visualize_step(self, step_num: int) -> bool:
        """Determine if a step should be visualized based on debug level."""
        if self.debug_level == "off":
            return False
        if self.debug_level == "minimal":
            return False  # Only episode summaries
        if self.debug_level == "standard":
            return step_num % 10 == 0  # Every 10th step
        if self.debug_level == "verbose":
            return step_num % 5 == 0  # Every 5th step
        if self.debug_level == "full":
            return True  # Every step
        return False

    def should_visualize_episode_summary(self) -> bool:
        """Determine if episode summaries should be created."""
        return self.debug_level != "off"

    def get_color_palette(self) -> Dict[int, str]:
        """Get color palette based on color scheme setting."""
        from .core import ARC_COLOR_PALETTE

        if self.color_scheme == "default":
            return ARC_COLOR_PALETTE
        if self.color_scheme == "colorblind":
            # Colorblind-friendly palette
            return {
                0: "#000000",  # black
                1: "#0173B2",  # blue
                2: "#DE8F05",  # orange
                3: "#029E73",  # green
                4: "#CC78BC",  # pink
                5: "#949494",  # grey
                6: "#D55E00",  # red-orange
                7: "#F0E442",  # yellow
                8: "#56B4E9",  # light blue
                9: "#8B4513",  # brown
                10: "#FFFFFF",  # white
            }
        if self.color_scheme == "high_contrast":
            # High contrast palette
            return {
                0: "#000000",  # black
                1: "#0000FF",  # pure blue
                2: "#FF0000",  # pure red
                3: "#00FF00",  # pure green
                4: "#FFFF00",  # pure yellow
                5: "#808080",  # grey
                6: "#FF00FF",  # magenta
                7: "#FFA500",  # orange
                8: "#00FFFF",  # cyan
                9: "#800000",  # maroon
                10: "#FFFFFF",  # white
            }
        return ARC_COLOR_PALETTE


@chex.dataclass
class StepVisualizationData:
    """Data structure for step visualization information."""

    step_num: int
    before_grid: Grid
    after_grid: Grid
    action: Dict[str, Any]
    reward: float
    info: Dict[str, Any]
    selection_mask: Optional[jnp.ndarray] = None
    changed_cells: Optional[jnp.ndarray] = None
    operation_name: str = ""
    timestamp: float = field(default_factory=time.time)


@chex.dataclass
class EpisodeSummaryData:
    """Data structure for episode summary information."""

    episode_num: int
    total_steps: int
    total_reward: float
    reward_progression: List[float]
    similarity_progression: List[float]
    final_similarity: float
    task_id: str
    success: bool
    key_moments: List[int] = field(default_factory=list)  # Important step numbers
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self) -> None:
        """Set end time if not provided."""
        if self.end_time == 0.0:
            object.__setattr__(self, "end_time", time.time())


class EnhancedVisualizer:
    """Enhanced visualization system integrating all components.

    This class provides a unified interface for visualization that integrates
    episode management, async logging, wandb integration, and performance
    optimization while maintaining JAX compatibility.
    """

    def __init__(
        self,
        config: VisualizationConfig,
        episode_manager: Optional[EpisodeManager] = None,
        async_logger: Optional[AsyncLogger] = None,
        wandb_integration: Optional[WandbIntegration] = None,
    ):
        """Initialize the enhanced visualizer.

        Args:
            config: Visualization configuration
            episode_manager: Optional episode manager (created if None)
            async_logger: Optional async logger (created if None)
            wandb_integration: Optional wandb integration (created if None)
        """
        self.config = config

        # Initialize components
        self.episode_manager = episode_manager or EpisodeManager(config.episode_config)
        self.async_logger = async_logger or AsyncLogger(config.async_logger_config)

        # Initialize wandb integration if enabled
        self.wandb_integration = None
        if config.wandb_config.enabled:
            self.wandb_integration = wandb_integration or WandbIntegration(
                config.wandb_config
            )

        # Internal state
        self.current_episode_num: Optional[int] = None
        self.current_episode_data: List[StepVisualizationData] = []
        self.performance_stats = {
            "total_visualizations": 0,
            "total_time": 0.0,
            "avg_time_per_viz": 0.0,
        }

        logger.info(
            f"Enhanced visualizer initialized with debug level: {config.debug_level}"
        )

    def start_episode(self, episode_num: int, task_id: str = "") -> None:
        """Start a new episode for visualization tracking.

        Args:
            episode_num: Episode number
            task_id: Optional task identifier
        """
        if self.config.debug_level == "off":
            return

        self.current_episode_num = episode_num
        self.current_episode_data = []

        # Create episode directory
        episode_dir = self.episode_manager.start_new_episode(episode_num)

        # Log episode start
        if self.config.debug_level in ["verbose", "full"]:
            logger.info(f"Started episode {episode_num} (task: {task_id})")
            logger.info(f"Episode directory: {episode_dir}")

        # Initialize wandb logging for episode
        if self.wandb_integration:
            self.wandb_integration.log_episode_start(episode_num, task_id)

    def visualize_step(
        self,
        step_data: StepVisualizationData,
    ) -> Optional[Path]:
        """Create and save step visualization.

        Args:
            step_data: Step visualization data

        Returns:
            Path to saved visualization file, or None if not saved
        """
        if not self.config.should_visualize_step(step_data.step_num):
            return None

        start_time = time.time()

        try:
            # Store step data for episode summary
            self.current_episode_data.append(step_data)

            # Generate visualization based on formats
            saved_paths = []
            for output_format in self.config.output_formats:
                if output_format == "svg":
                    path = self._create_step_svg(step_data)
                    if path:
                        saved_paths.append(path)
                elif output_format == "png":
                    path = self._create_step_png(step_data)
                    if path:
                        saved_paths.append(path)
                elif output_format == "html":
                    path = self._create_step_html(step_data)
                    if path:
                        saved_paths.append(path)

            # Log to wandb if enabled
            if self.wandb_integration and saved_paths:
                self.wandb_integration.log_step_visualization(
                    step_data.step_num,
                    {
                        "reward": step_data.reward,
                        "operation": step_data.operation_name,
                    },
                    saved_paths[0] if saved_paths else None,
                )

            # Update performance stats
            elapsed = time.time() - start_time
            self._update_performance_stats(elapsed)

            return saved_paths[0] if saved_paths else None

        except Exception as e:
            logger.error(f"Error creating step visualization: {e}")
            return None

    def visualize_episode_summary(
        self,
        summary_data: EpisodeSummaryData,
    ) -> Optional[Path]:
        """Create episode summary visualization.

        Args:
            summary_data: Episode summary data

        Returns:
            Path to saved summary file, or None if not created
        """
        if not self.config.should_visualize_episode_summary():
            return None

        try:
            # Generate summary visualization
            summary_path = self._create_episode_summary(summary_data)

            # Log to wandb if enabled
            if self.wandb_integration and summary_path:
                self.wandb_integration.log_episode_summary(
                    summary_data.episode_num,
                    {
                        "total_reward": summary_data.total_reward,
                        "total_steps": summary_data.total_steps,
                        "final_similarity": summary_data.final_similarity,
                        "success": summary_data.success,
                    },
                    summary_path,
                )

            # Clear episode data
            self.current_episode_data = []
            self.current_episode_num = None

            return summary_path

        except Exception as e:
            logger.error(f"Error creating episode summary: {e}")
            return None

    def _create_step_svg(self, step_data: StepVisualizationData) -> Optional[Path]:
        """Create SVG visualization for a step."""
        if self.current_episode_num is None:
            return None

        try:
            from .core import (
                detect_changed_cells,
                draw_rl_step_svg_enhanced,
                get_operation_display_name,
                infer_fill_color_from_grids,
            )

            # Get step file path
            step_path = self.episode_manager.get_step_path(
                step_data.step_num, file_type="svg"
            )

            # Detect changed cells if not provided
            changed_cells = step_data.changed_cells
            if changed_cells is None:
                changed_cells = detect_changed_cells(
                    step_data.before_grid, step_data.after_grid
                )

            # Get operation name if not provided
            operation_name = step_data.operation_name
            if not operation_name and "operation" in step_data.action:
                # Try to infer fill color from grids for better description
                fill_color = -1
                if step_data.action["operation"] == 1:  # Fill Selected
                    fill_color = infer_fill_color_from_grids(
                        step_data.before_grid,
                        step_data.after_grid,
                        np.asarray(step_data.action.get("selection", np.array([]))),
                    )
                    if fill_color >= 0:
                        step_data.action["fill_color"] = fill_color

                operation_name = get_operation_display_name(
                    step_data.action["operation"], step_data.action
                )

            # Create enhanced step visualization
            svg_content = draw_rl_step_svg_enhanced(
                before_grid=step_data.before_grid,
                after_grid=step_data.after_grid,
                action=step_data.action,
                reward=step_data.reward,
                info=step_data.info,
                step_num=step_data.step_num,
                operation_name=operation_name,
                changed_cells=changed_cells,
                config=self.config,
            )

            # Save SVG file
            step_path.parent.mkdir(parents=True, exist_ok=True)
            with open(step_path, "w") as f:
                f.write(svg_content)

            return step_path

        except Exception as e:
            logger.error(f"Error creating step SVG: {e}")
            return None

    def _create_step_png(self, step_data: StepVisualizationData) -> Optional[Path]:
        """Create PNG visualization for a step."""
        # For now, convert from SVG - could be optimized later
        svg_path = self._create_step_svg(step_data)
        if not svg_path:
            return None

        try:
            import cairosvg

            png_path = svg_path.with_suffix(".png")
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=800,
                output_height=600,
            )

            return png_path

        except ImportError:
            logger.warning("cairosvg not available for PNG conversion")
            return None
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            return None

    def _create_step_html(self, step_data: StepVisualizationData) -> Optional[Path]:
        """Create HTML visualization for a step."""
        if self.current_episode_num is None:
            return None

        try:
            # Get step file path
            step_path = self.episode_manager.get_step_path(
                step_data.step_num, file_type="html"
            )

            # Create HTML content with embedded SVG
            svg_path = self._create_step_svg(step_data)
            if not svg_path:
                return None

            with open(svg_path) as f:
                svg_content = f.read()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Step {step_data.step_num} - Episode {self.current_episode_num}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metadata {{ background: #f5f5f5; padding: 10px; margin-bottom: 20px; }}
                    .svg-container {{ text-align: center; }}
                </style>
            </head>
            <body>
                <div class="metadata">
                    <h2>Step {step_data.step_num}</h2>
                    <p><strong>Reward:</strong> {step_data.reward}</p>
                    <p><strong>Operation:</strong> {step_data.operation_name}</p>
                    <p><strong>Timestamp:</strong> {step_data.timestamp}</p>
                </div>
                <div class="svg-container">
                    {svg_content}
                </div>
            </body>
            </html>
            """

            # Save HTML file
            step_path.parent.mkdir(parents=True, exist_ok=True)
            with open(step_path, "w") as f:
                f.write(html_content)

            return step_path

        except Exception as e:
            logger.error(f"Error creating step HTML: {e}")
            return None

    def _create_episode_summary(
        self, summary_data: EpisodeSummaryData
    ) -> Optional[Path]:
        """Create episode summary visualization."""
        try:
            # Get summary file path
            summary_path = self.episode_manager.get_episode_summary_path("svg")

            # Create enhanced summary visualization
            from .core import draw_enhanced_episode_summary_svg

            svg_content = draw_enhanced_episode_summary_svg(
                summary_data=summary_data,
                step_data=self.current_episode_data,
                config=self.config,
            )

            # Save summary file
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                f.write(svg_content)

            return summary_path

        except Exception as e:
            logger.error(f"Error creating episode summary: {e}")
            return None

    def _update_performance_stats(self, elapsed_time: float) -> None:
        """Update performance statistics."""
        self.performance_stats["total_visualizations"] += 1
        self.performance_stats["total_time"] += elapsed_time
        self.performance_stats["avg_time_per_viz"] = (
            self.performance_stats["total_time"]
            / self.performance_stats["total_visualizations"]
        )

        # Log performance warning if visualization is taking too long
        if elapsed_time > 1.0:  # More than 1 second
            logger.warning(f"Slow visualization: {elapsed_time:.2f}s")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics report."""
        return {
            **self.performance_stats,
            "debug_level": self.config.debug_level,
            "output_formats": self.config.output_formats,
            "async_processing": self.config.async_processing,
        }

    def create_comparison_visualization(
        self,
        episodes_data: List[EpisodeSummaryData],
        comparison_type: str = "reward_progression",
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Create comparison visualization across multiple episodes.

        Args:
            episodes_data: List of episode summary data to compare
            comparison_type: Type of comparison ("reward_progression", "similarity", "performance")
            output_path: Optional output path (auto-generated if None)

        Returns:
            Path to saved comparison file, or None if not created
        """
        try:
            from .core import create_episode_comparison_visualization

            # Generate comparison visualization
            svg_content = create_episode_comparison_visualization(
                episodes_data=episodes_data,
                comparison_type=comparison_type,
            )

            # Determine output path
            if output_path is None:
                base_dir = self.episode_manager.config.get_base_path()
                output_path = base_dir / f"comparison_{comparison_type}.svg"

            # Save comparison file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(svg_content)

            logger.info(f"Created comparison visualization: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up resources and flush pending operations."""
        try:
            # Flush async logger
            if self.async_logger:
                self.async_logger.flush()

            # Cleanup episode manager
            if self.episode_manager:
                self.episode_manager.cleanup_old_data()

            # Finish wandb run
            if self.wandb_integration:
                self.wandb_integration.finish_run()

            logger.info("Enhanced visualizer cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
