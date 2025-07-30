"""Episode visualization functions for JaxARC.

This module provides functions for visualizing complete episodes, episode summaries,
and comparisons between multiple episodes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

import drawsvg  # type: ignore[import-untyped]
import numpy as np

from .constants import ARC_COLOR_PALETTE

if TYPE_CHECKING:
    pass


def draw_enhanced_episode_summary_svg(
    summary_data: Any,
    step_data: List[Any],
    config: Optional[Any] = None,
    width: float = 1400.0,
    height: float = 1000.0,
) -> str:
    """Generate enhanced SVG visualization of episode summary with comprehensive metrics.

    This enhanced version includes:
    - Reward progression chart with key moments highlighted
    - Similarity progression chart
    - Grid state thumbnails at key moments
    - Performance metrics panel
    - Success/failure analysis

    Args:
        summary_data: Episode summary data
        step_data: List of step visualization data
        config: Optional visualization configuration
        width: Width of the visualization
        height: Height of the visualization

    Returns:
        SVG string containing the enhanced episode summary
    """
    import drawsvg as draw

    # Create main drawing
    drawing = draw.Drawing(width, height)
    drawing.append(draw.Rectangle(0, 0, width, height, fill="#f8f9fa"))

    # Layout parameters
    padding = 40
    title_height = 100
    metrics_height = 80
    chart_height = 250
    thumbnails_height = 200
    remaining_height = (
        height
        - title_height
        - metrics_height
        - 2 * chart_height
        - thumbnails_height
        - 6 * padding
    )

    # Add enhanced title section
    title_bg_height = title_height - 20
    drawing.append(
        draw.Rectangle(
            padding,
            padding,
            width - 2 * padding,
            title_bg_height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Main title
    title_text = f"Episode {summary_data.episode_num} Summary"
    drawing.append(
        draw.Text(
            title_text,
            font_size=32,
            x=width / 2,
            y=padding + 40,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Success indicator
    success_color = "#27ae60" if summary_data.success else "#e74c3c"
    success_text = "SUCCESS" if summary_data.success else "FAILED"
    drawing.append(
        draw.Text(
            success_text,
            font_size=18,
            x=width - padding - 20,
            y=padding + 30,
            text_anchor="end",
            font_family="Anuphan",
            font_weight="700",
            fill=success_color,
        )
    )

    # Task ID
    drawing.append(
        draw.Text(
            f"Task: {summary_data.task_id}",
            font_size=16,
            x=padding + 20,
            y=padding + 70,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="400",
            fill="#6c757d",
        )
    )

    # Metrics panel
    metrics_y = title_height + 2 * padding
    drawing.append(
        draw.Rectangle(
            padding,
            metrics_y,
            width - 2 * padding,
            metrics_height,
            fill="#ffffff",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Metrics grid
    metrics = [
        ("Total Steps", summary_data.total_steps, ""),
        ("Total Reward", summary_data.total_reward, ".3f"),
        ("Final Similarity", summary_data.final_similarity, ".3f"),
        (
            "Avg Reward/Step",
            summary_data.total_reward / max(summary_data.total_steps, 1),
            ".3f",
        ),
    ]

    metric_width = (width - 2 * padding - 60) / len(metrics)
    for i, (name, value, fmt) in enumerate(metrics):
        x_pos = padding + 20 + i * metric_width

        # Metric name
        drawing.append(
            draw.Text(
                name,
                font_size=14,
                x=x_pos,
                y=metrics_y + 25,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#495057",
            )
        )

        # Metric value
        value_text = f"{value:{fmt}}" if fmt else str(value)
        drawing.append(
            draw.Text(
                value_text,
                font_size=18,
                x=x_pos,
                y=metrics_y + 50,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="500",
                fill="#2c3e50",
            )
        )

    # Reward progression chart
    chart1_y = metrics_y + metrics_height + padding
    chart_width = (width - 3 * padding) / 2

    drawing.append(
        draw.Rectangle(
            padding,
            chart1_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Chart title
    drawing.append(
        draw.Text(
            "Reward Progression",
            font_size=18,
            x=padding + 20,
            y=chart1_y + 30,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Draw reward progression line
    if summary_data.reward_progression and len(summary_data.reward_progression) > 1:
        rewards = summary_data.reward_progression
        chart_inner_width = chart_width - 40
        chart_inner_height = chart_height - 80

        max_reward = max(rewards) if max(rewards) > 0 else 1
        min_reward = min(rewards) if min(rewards) < 0 else 0
        reward_range = max_reward - min_reward if max_reward != min_reward else 1

        # Draw grid lines
        for i in range(5):
            y_grid = chart1_y + 50 + i * (chart_inner_height / 4)
            drawing.append(
                draw.Line(
                    padding + 20,
                    y_grid,
                    padding + 20 + chart_inner_width,
                    y_grid,
                    stroke="#e9ecef",
                    stroke_width=1,
                )
            )

        # Draw reward line
        points = []
        for i, reward in enumerate(rewards):
            x = padding + 20 + (i / (len(rewards) - 1)) * chart_inner_width
            y = (
                chart1_y
                + 50
                + chart_inner_height
                - ((reward - min_reward) / reward_range) * chart_inner_height
            )
            points.append((x, y))

        if len(points) > 1:
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"

            drawing.append(
                draw.Path(
                    d=path_data,
                    stroke="#3498db",
                    stroke_width=3,
                    fill="none",
                )
            )

            # Add points
            for i, (x, y) in enumerate(points):
                # Highlight key moments
                if i in summary_data.key_moments:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            6,
                            fill="#e74c3c",
                            stroke="white",
                            stroke_width=2,
                        )
                    )
                else:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            4,
                            fill="#3498db",
                            stroke="white",
                            stroke_width=1,
                        )
                    )

    # Similarity progression chart
    chart2_x = padding + chart_width + padding

    drawing.append(
        draw.Rectangle(
            chart2_x,
            chart1_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Chart title
    drawing.append(
        draw.Text(
            "Similarity Progression",
            font_size=18,
            x=chart2_x + 20,
            y=chart1_y + 30,
            text_anchor="start",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Draw similarity progression line
    if (
        summary_data.similarity_progression
        and len(summary_data.similarity_progression) > 1
    ):
        similarities = summary_data.similarity_progression
        chart_inner_width = chart_width - 40
        chart_inner_height = chart_height - 80

        # Draw grid lines
        for i in range(5):
            y_grid = chart1_y + 50 + i * (chart_inner_height / 4)
            drawing.append(
                draw.Line(
                    chart2_x + 20,
                    y_grid,
                    chart2_x + 20 + chart_inner_width,
                    y_grid,
                    stroke="#e9ecef",
                    stroke_width=1,
                )
            )

        # Draw similarity line
        points = []
        for i, similarity in enumerate(similarities):
            x = chart2_x + 20 + (i / (len(similarities) - 1)) * chart_inner_width
            y = chart1_y + 50 + chart_inner_height - (similarity * chart_inner_height)
            points.append((x, y))

        if len(points) > 1:
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"

            drawing.append(
                draw.Path(
                    d=path_data,
                    stroke="#27ae60",
                    stroke_width=3,
                    fill="none",
                )
            )

            # Add points
            for i, (x, y) in enumerate(points):
                # Highlight key moments
                if i in summary_data.key_moments:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            6,
                            fill="#e74c3c",
                            stroke="white",
                            stroke_width=2,
                        )
                    )
                else:
                    drawing.append(
                        draw.Circle(
                            x,
                            y,
                            4,
                            fill="#27ae60",
                            stroke="white",
                            stroke_width=1,
                        )
                    )

    # Key moments thumbnails section
    thumbnails_y = chart1_y + chart_height + padding

    if summary_data.key_moments and step_data:
        drawing.append(
            draw.Rectangle(
                padding,
                thumbnails_y,
                width - 2 * padding,
                thumbnails_height,
                fill="white",
                stroke="#dee2e6",
                stroke_width=1,
                rx=8,
            )
        )

        # Section title
        drawing.append(
            draw.Text(
                "Key Moments",
                font_size=18,
                x=padding + 20,
                y=thumbnails_y + 30,
                text_anchor="start",
                font_family="Anuphan",
                font_weight="600",
                fill="#2c3e50",
            )
        )

        # Draw thumbnails for key moments
        thumbnail_size = 120
        thumbnail_spacing = 20
        thumbnails_per_row = min(len(summary_data.key_moments), 8)

        for i, step_idx in enumerate(summary_data.key_moments[:thumbnails_per_row]):
            if step_idx < len(step_data):
                step = step_data[step_idx]

                thumb_x = padding + 20 + i * (thumbnail_size + thumbnail_spacing)
                thumb_y = thumbnails_y + 50

                # Draw thumbnail background
                drawing.append(
                    draw.Rectangle(
                        thumb_x,
                        thumb_y,
                        thumbnail_size,
                        thumbnail_size,
                        fill="#f8f9fa",
                        stroke="#dee2e6",
                        stroke_width=1,
                        rx=4,
                    )
                )

                # Draw simplified grid representation
                if hasattr(step, "after_grid"):
                    grid_data = np.asarray(step.after_grid.data)
                    grid_size = min(thumbnail_size - 20, 80)
                    cell_size = grid_size / max(grid_data.shape)

                    for row in range(min(grid_data.shape[0], 8)):
                        for col in range(min(grid_data.shape[1], 8)):
                            color_val = int(grid_data[row, col])
                            if config and hasattr(config, "get_color_palette"):
                                color_palette = config.get_color_palette()
                            else:
                                color_palette = ARC_COLOR_PALETTE

                            fill_color = color_palette.get(color_val, "#CCCCCC")

                            drawing.append(
                                draw.Rectangle(
                                    thumb_x + 10 + col * cell_size,
                                    thumb_y + 10 + row * cell_size,
                                    cell_size,
                                    cell_size,
                                    fill=fill_color,
                                    stroke="#6c757d",
                                    stroke_width=0.5,
                                )
                            )

                # Add step label
                drawing.append(
                    draw.Text(
                        f"Step {step_idx}",
                        font_size=12,
                        x=thumb_x + thumbnail_size / 2,
                        y=thumb_y + thumbnail_size + 15,
                        text_anchor="middle",
                        font_family="Anuphan",
                        font_weight="500",
                        fill="#495057",
                    )
                )

    # Add footer with timing information
    footer_y = height - 40
    if hasattr(summary_data, "start_time") and hasattr(summary_data, "end_time"):
        duration = summary_data.end_time - summary_data.start_time
        footer_text = f"Duration: {duration:.1f}s | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        footer_text = f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    drawing.append(
        draw.Text(
            footer_text,
            font_size=12,
            x=width / 2,
            y=footer_y,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="300",
            fill="#adb5bd",
        )
    )

    return drawing.as_svg()


def draw_episode_summary_svg(
    summary_data: Any,
    step_data: List[Any],
    config: Optional[Any] = None,
    width: float = 1400.0,
    height: float = 1000.0,
) -> str:
    """Generate episode summary visualization (enhanced version)."""
    return draw_enhanced_episode_summary_svg(
        summary_data=summary_data,
        step_data=step_data,
        config=config,
        width=width,
        height=height,
    )


def create_episode_comparison_visualization(
    episodes_data: List[Any],
    comparison_type: str = "reward_progression",
    width: float = 1200.0,
    height: float = 600.0,
) -> str:
    """Create comparison visualization across multiple episodes.

    Args:
        episodes_data: List of episode summary data
        comparison_type: Type of comparison ("reward_progression", "similarity", "performance")
        width: Width of the visualization
        height: Height of the visualization

    Returns:
        SVG string containing the comparison visualization
    """
    import drawsvg as draw

    # Create main drawing
    drawing = draw.Drawing(width, height)
    drawing.append(draw.Rectangle(0, 0, width, height, fill="#f8f9fa"))

    # Layout parameters
    padding = 40
    title_height = 80
    chart_height = height - title_height - 2 * padding - 60

    # Add title
    title_text = f"Episode Comparison - {comparison_type.replace('_', ' ').title()}"
    drawing.append(
        draw.Text(
            title_text,
            font_size=28,
            x=width / 2,
            y=50,
            text_anchor="middle",
            font_family="Anuphan",
            font_weight="600",
            fill="#2c3e50",
        )
    )

    # Chart area
    chart_y = title_height + padding
    chart_width = width - 2 * padding

    drawing.append(
        draw.Rectangle(
            padding,
            chart_y,
            chart_width,
            chart_height,
            fill="white",
            stroke="#dee2e6",
            stroke_width=1,
            rx=8,
        )
    )

    # Colors for different episodes
    episode_colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

    if comparison_type == "reward_progression":
        # Draw reward progression for each episode
        chart_inner_width = chart_width - 60
        chart_inner_height = chart_height - 60

        # Find global min/max for scaling
        all_rewards = []
        for episode in episodes_data:
            if hasattr(episode, "reward_progression") and episode.reward_progression:
                all_rewards.extend(episode.reward_progression)

        if all_rewards:
            max_reward = max(all_rewards)
            min_reward = min(all_rewards)
            reward_range = max_reward - min_reward if max_reward != min_reward else 1

            # Draw grid lines
            for i in range(5):
                y_grid = chart_y + 30 + i * (chart_inner_height / 4)
                drawing.append(
                    draw.Line(
                        padding + 30,
                        y_grid,
                        padding + 30 + chart_inner_width,
                        y_grid,
                        stroke="#e9ecef",
                        stroke_width=1,
                    )
                )

            # Draw each episode's progression
            for ep_idx, episode in enumerate(episodes_data[: len(episode_colors)]):
                if (
                    hasattr(episode, "reward_progression")
                    and episode.reward_progression
                ):
                    rewards = episode.reward_progression
                    color = episode_colors[ep_idx]

                    points = []
                    for i, reward in enumerate(rewards):
                        x = padding + 30 + (i / (len(rewards) - 1)) * chart_inner_width
                        y = (
                            chart_y
                            + 30
                            + chart_inner_height
                            - ((reward - min_reward) / reward_range)
                            * chart_inner_height
                        )
                        points.append((x, y))

                    if len(points) > 1:
                        path_data = f"M {points[0][0]} {points[0][1]}"
                        for x, y in points[1:]:
                            path_data += f" L {x} {y}"

                        drawing.append(
                            draw.Path(
                                d=path_data,
                                stroke=color,
                                stroke_width=2,
                                fill="none",
                            )
                        )

                        # Add points
                        for x, y in points:
                            drawing.append(
                                draw.Circle(
                                    x,
                                    y,
                                    3,
                                    fill=color,
                                    stroke="white",
                                    stroke_width=1,
                                )
                            )

                    # Add legend entry
                    legend_y = chart_y + chart_height + 20 + ep_idx * 20
                    drawing.append(
                        draw.Line(
                            padding + 20,
                            legend_y,
                            padding + 40,
                            legend_y,
                            stroke=color,
                            stroke_width=3,
                        )
                    )
                    drawing.append(
                        draw.Text(
                            f"Episode {episode.episode_num}",
                            font_size=14,
                            x=padding + 50,
                            y=legend_y + 5,
                            text_anchor="start",
                            font_family="Anuphan",
                            font_weight="400",
                            fill="#495057",
                        )
                    )

    elif comparison_type == "performance":
        # Create bar chart comparing final performance metrics
        metrics = ["total_reward", "final_similarity", "total_steps"]
        metric_labels = ["Total Reward", "Final Similarity", "Steps"]

        chart_inner_width = chart_width - 60
        chart_inner_height = chart_height - 60

        bar_width = (chart_width - 100) / (
            len(episodes_data) * len(metrics) + len(metrics)
        )
        group_spacing = bar_width * 0.5

        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Get values for this metric
            values = []
            for episode in episodes_data:
                if hasattr(episode, metric):
                    values.append(getattr(episode, metric))
                else:
                    values.append(0)

            if values:
                max_val = max(values) if max(values) > 0 else 1

                # Draw bars for this metric
                for ep_idx, (episode, value) in enumerate(zip(episodes_data, values)):
                    x = (
                        padding
                        + 30
                        + metric_idx * (len(episodes_data) * bar_width + group_spacing)
                        + ep_idx * bar_width
                    )
                    bar_height = (value / max_val) * (chart_inner_height - 40)
                    y = chart_y + chart_height - 30 - bar_height

                    color = episode_colors[ep_idx % len(episode_colors)]

                    drawing.append(
                        draw.Rectangle(
                            x,
                            y,
                            bar_width * 0.8,
                            bar_height,
                            fill=color,
                            stroke="white",
                            stroke_width=1,
                        )
                    )

                # Add metric label
                label_x = (
                    padding
                    + 30
                    + metric_idx * (len(episodes_data) * bar_width + group_spacing)
                    + (len(episodes_data) * bar_width) / 2
                )
                drawing.append(
                    draw.Text(
                        label,
                        font_size=12,
                        x=label_x,
                        y=chart_y + chart_height - 10,
                        text_anchor="middle",
                        font_family="Anuphan",
                        font_weight="500",
                        fill="#495057",
                    )
                )

    return drawing.as_svg()