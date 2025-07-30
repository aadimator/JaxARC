"""Task visualization functions for JaxARC.

This module provides functions for visualizing complete ARC tasks and task pairs
in both Rich terminal format and SVG format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import drawsvg  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np

from .constants import ARC_COLOR_PALETTE
from .svg_core import _draw_dotted_squircle, draw_grid_svg
from .utils import _extract_grid_data, _extract_valid_region

if TYPE_CHECKING:
    from jaxarc.types import Grid, JaxArcTask


def draw_task_pair_svg(
    input_grid: jnp.ndarray | np.ndarray | Grid,
    output_grid: jnp.ndarray | np.ndarray | Grid | None = None,
    input_mask: jnp.ndarray | np.ndarray | None = None,
    output_mask: jnp.ndarray | np.ndarray | None = None,
    width: float = 15.0,
    height: float = 8.0,
    label: str = "",
    show_unknown_output: bool = True,
) -> drawsvg.Drawing:
    """Draw an input-output task pair as SVG with strict height and flexible width.

    Args:
        input_grid: Input grid data
        output_grid: Output grid data (optional)
        input_mask: Optional mask for input grid
        output_mask: Optional mask for output grid
        width: Maximum width for the drawing (actual width may be less)
        height: Strict height constraint - all content must fit within this height
        label: Label for the pair
        show_unknown_output: Whether to show "?" for missing output

    Returns:
        SVG Drawing object
    """
    padding = 0.5
    extra_bottom_padding = 0.25
    io_gap = 0.4

    # Calculate available space for grids - height is STRICT
    ymax = (height - padding - extra_bottom_padding - io_gap) / 2

    # Calculate aspect ratios to determine width requirements
    input_grid_data, input_mask_data = _extract_grid_data(input_grid)
    if input_mask is not None:
        input_mask_data = np.asarray(input_mask)

    _, _, (input_h, input_w) = _extract_valid_region(input_grid_data, input_mask_data)

    input_ratio = input_w / input_h if input_h > 0 else 1.0
    max_ratio = input_ratio

    if output_grid is not None:
        output_grid_data, output_mask_data = _extract_grid_data(output_grid)
        if output_mask is not None:
            output_mask_data = np.asarray(output_mask)
        _, _, (output_h, output_w) = _extract_valid_region(
            output_grid_data, output_mask_data
        )

        output_ratio = output_w / output_h if output_h > 0 else 1.0
        max_ratio = max(input_ratio, output_ratio)

    # Calculate required width based on height constraint and aspect ratio
    required_width = ymax * max_ratio + padding * 2
    final_width = max(required_width, padding * 2 + 1.0)  # Minimum width

    # Don't exceed specified width constraint
    final_width = min(final_width, width)

    max_grid_width = final_width - padding * 2

    # Draw elements following two-pass approach
    drawlist = []
    x_ptr = 0.0
    y_ptr = 0.0

    # First pass: Draw input grid and determine dimensions
    input_result = draw_grid_svg(
        input_grid,
        input_mask,
        max_width=max_grid_width,
        max_height=ymax,
        label=f"{label} Input" if label else "Input",
        padding=padding,
        extra_bottom_padding=extra_bottom_padding,
        as_group=True,
    )

    if isinstance(input_result, tuple):
        input_group, input_origin, input_size = input_result
    else:
        msg = "Expected tuple result when as_group=True"
        raise ValueError(msg)

    # Calculate output dimensions for spacing
    actual_output_width = 0.0
    output_y_total_height = 0.0
    output_g = None
    output_origin_out = (-padding / 2, -padding / 2)

    if output_grid is not None:
        output_result = draw_grid_svg(
            output_grid,
            output_mask,
            max_width=max_grid_width,
            max_height=ymax,
            label=f"{label} Output" if label else "Output",
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(output_result, tuple):
            output_g, output_origin_out, output_size = output_result
            actual_output_width = output_size[0]
            output_y_total_height = output_size[1]
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)
    else:
        # Approximate height for '?' slot
        output_y_total_height = ymax + padding + extra_bottom_padding

    # Position input grid
    drawlist.append(
        drawsvg.Use(
            input_group,
            x=(max_grid_width + padding - input_size[0]) / 2 - input_origin[0],
            y=-input_origin[1],
        )
    )

    x_ptr += max(input_size[0], actual_output_width)
    y_ptr = max(y_ptr, input_size[1])

    # Second pass: Draw arrow and output
    arrow_x_center = input_size[0] / 2
    arrow_top_y = y_ptr + padding - 0.6
    arrow_bottom_y = y_ptr + padding + io_gap - 0.6

    drawlist.append(
        drawsvg.Line(
            arrow_x_center,
            arrow_top_y,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )
    drawlist.append(
        drawsvg.Line(
            arrow_x_center - 0.15,
            arrow_bottom_y - 0.2,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )
    drawlist.append(
        drawsvg.Line(
            arrow_x_center + 0.15,
            arrow_bottom_y - 0.2,
            arrow_x_center,
            arrow_bottom_y,
            stroke_width=0.05,
            stroke="#888888",
        )
    )

    # Position output
    y_content_top_output_area = y_ptr + io_gap

    if output_g is not None:
        drawlist.append(
            drawsvg.Use(
                output_g,
                x=(max_grid_width + padding - actual_output_width) / 2
                - output_origin_out[0],
                y=y_ptr - output_origin_out[1] + io_gap,
            )
        )
    elif show_unknown_output:
        # Draw question mark for unknown output
        q_text_y_center = (
            y_content_top_output_area + (ymax / 2) + extra_bottom_padding / 2
        )
        drawlist.append(
            drawsvg.Text(
                "?",
                x=(max_grid_width + padding) / 2,
                y=q_text_y_center,
                font_size=1.0,
                font_family="Anuphan",
                font_weight="700",
                fill="#333333",
                text_anchor="middle",
                alignment_baseline="middle",
            )
        )

    y_ptr2 = y_ptr + io_gap + output_y_total_height

    # Calculate final drawing dimensions
    final_drawing_width = max(x_ptr, final_width)
    final_drawing_height = max(y_ptr2, height)  # Height is strict

    # Create final drawing
    drawing = drawsvg.Drawing(
        final_drawing_width, final_drawing_height + 0.3, origin=(0, 0)
    )
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))

    # Add all draw elements
    for item in drawlist:
        drawing.append(item)

    # Embed font and set scale
    drawing.embed_google_font(
        "Anuphan:wght@400;600;700",
        text=set(
            "Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
    )
    drawing.set_pixel_scale(40)

    return drawing


def draw_parsed_task_data_svg(
    task_data: JaxArcTask,
    width: float = 30.0,
    height: float = 20.0,
    include_test: bool | str = False,
    border_colors: list[str] | None = None,
) -> drawsvg.Drawing:
    """Draw a complete JaxArcTask as an SVG with strict height and flexible width.

    Args:
        task_data: The parsed task data to visualize
        width: Maximum width for the drawing (actual width may be less)
        height: Strict height constraint - all content must fit within this height
        include_test: Whether to include test examples. If 'all', show test outputs too.
        border_colors: Custom border colors [input_color, output_color]

    Returns:
        SVG Drawing object
    """
    from jaxarc.utils.task_manager import extract_task_id_from_index

    if border_colors is None:
        border_colors = ["#111111ff", "#111111ff"]

    padding = 0.5
    extra_bottom_padding = 0.25
    io_gap = 0.4

    # Calculate available space for grids - height is STRICT
    ymax = (height - padding - extra_bottom_padding - io_gap) / 2

    # Prepare examples list
    examples = []

    # Add training examples
    for i in range(task_data.num_train_pairs):
        examples.append(
            (
                task_data.input_grids_examples[i],
                task_data.output_grids_examples[i],
                task_data.input_masks_examples[i],
                task_data.output_masks_examples[i],
                f"{i + 1}",
                False,  # is_test
            )
        )

    # Add test examples
    if include_test:
        for i in range(task_data.num_test_pairs):
            show_test_output = include_test == "all"
            output_grid = (
                task_data.true_test_output_grids[i] if show_test_output else None
            )
            output_mask = (
                task_data.true_test_output_masks[i] if show_test_output else None
            )

            examples.append(
                (
                    task_data.test_input_grids[i],
                    output_grid,
                    task_data.test_input_masks[i],
                    output_mask,
                    f"{i + 1}",
                    True,  # is_test
                )
            )

    if not examples:
        # Empty task
        drawing = drawsvg.Drawing(width, height, origin=(0, 0))
        drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))
        drawing.append(
            drawsvg.Text(
                f"Task {extract_task_id_from_index(task_data.task_index)} (No examples)",
                x=width / 2,
                y=height / 2,
                font_size=0.5,
                text_anchor="middle",
                fill="black",
            )
        )
        drawing.set_pixel_scale(40)
        return drawing

    # Prepare training examples
    train_examples = []
    for i in range(task_data.num_train_pairs):
        train_examples.append(
            (
                task_data.input_grids_examples[i],
                task_data.output_grids_examples[i],
                task_data.input_masks_examples[i],
                task_data.output_masks_examples[i],
                f"{i + 1}",
                False,  # is_test
            )
        )

    # Prepare test examples
    test_examples = []
    if include_test:
        for i in range(task_data.num_test_pairs):
            show_test_output = include_test == "all"
            output_grid = (
                task_data.true_test_output_grids[i] if show_test_output else None
            )
            output_mask = (
                task_data.true_test_output_masks[i] if show_test_output else None
            )

            test_examples.append(
                (
                    task_data.test_input_grids[i],
                    output_grid,
                    task_data.test_input_masks[i],
                    output_mask,
                    f"{i + 1}",
                    True,  # is_test
                )
            )

    # Combine all examples
    examples = train_examples + test_examples

    # Calculate ideal width for each example based on aspect ratio and height constraint
    max_widths = np.zeros(len(examples))

    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        _label,
        _is_test,
    ) in enumerate(examples):
        input_grid_data, _ = _extract_grid_data(input_grid)
        input_mask_data = np.asarray(input_mask) if input_mask is not None else None
        _, _, (input_h, input_w) = _extract_valid_region(
            input_grid_data, input_mask_data
        )

        input_ratio = input_w / input_h if input_h > 0 else 1.0
        max_ratio = input_ratio

        if output_grid is not None:
            output_grid_data, _ = _extract_grid_data(output_grid)
            output_mask_data = (
                np.asarray(output_mask) if output_mask is not None else None
            )
            _, _, (output_h, output_w) = _extract_valid_region(
                output_grid_data, output_mask_data
            )

            output_ratio = output_w / output_h if output_h > 0 else 1.0
            max_ratio = max(input_ratio, output_ratio)

        # Calculate ideal width based on height constraint and aspect ratio
        xmax_for_pair = ymax * max_ratio
        max_widths[i] = xmax_for_pair

    # Add extra spacing between training and test groups
    group_spacing = 0.5 if len(train_examples) > 0 and len(test_examples) > 0 else 0.0

    # Proportional allocation algorithm - distribute width based on needs
    paddingless_width = width - padding * len(examples) - group_spacing
    allocation = np.zeros_like(max_widths)
    increment = 0.01

    if paddingless_width > 0 and len(examples) > 0:
        if np.any(max_widths > 0):
            for _ in range(int(paddingless_width // increment)):
                incr_mask = (allocation + increment) <= max_widths
                if incr_mask.sum() > 0:
                    allocation[incr_mask] += increment / incr_mask.sum()
                else:
                    break

        # Fallback: equal distribution if no progress made
        if np.sum(allocation) == 0:
            allocation[:] = paddingless_width / len(examples)

    # Two-pass rendering following reference implementation pattern
    drawlist = []

    # Account for squircle margins in positioning if we have grouping
    squircle_margin = 0.15
    has_grouping = len(train_examples) > 0 and len(test_examples) > 0
    x_offset = squircle_margin if has_grouping else 0.0
    y_offset = squircle_margin if has_grouping else 0.0

    # Calculate group boundaries
    train_width = (
        sum(allocation[: len(train_examples)]) + padding * len(train_examples)
        if train_examples
        else 0
    )
    test_start_x = x_offset + train_width + (group_spacing if has_grouping else 0)

    x_ptr = x_offset
    y_ptr = y_offset

    # First pass: Draw input grids and calculate input row height
    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        label,
        is_test,
    ) in enumerate(examples):
        input_result = draw_grid_svg(
            input_grid,
            input_mask,
            max_width=allocation[i],
            max_height=ymax,
            label=f"In #{label}",
            border_color=border_colors[0],
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(input_result, tuple):
            input_group, input_origin, input_size = input_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)

        # Calculate actual output width for spacing
        actual_output_width = 0.0
        if output_grid is not None:
            output_result_for_spacing = draw_grid_svg(
                output_grid,
                output_mask,
                max_width=allocation[i],
                max_height=ymax,
                label=f"Out #{label}",
                border_color=border_colors[1],
                padding=padding,
                extra_bottom_padding=extra_bottom_padding,
                as_group=True,
            )
            if isinstance(output_result_for_spacing, tuple):
                _, _, (actual_output_width, _) = output_result_for_spacing

        # Determine x position based on whether this is a test example
        if is_test and has_grouping:
            # For test examples, position relative to test start
            test_index = i - len(train_examples)
            test_x_offset = (
                sum(allocation[len(train_examples) : len(train_examples) + test_index])
                + padding * test_index
            )
            current_x_ptr = test_start_x + test_x_offset
        else:
            # For training examples, use current x_ptr
            current_x_ptr = x_ptr

        # Position input grid
        drawlist.append(
            drawsvg.Use(
                input_group,
                x=current_x_ptr
                + (allocation[i] + padding - input_size[0]) / 2
                - input_origin[0],
                y=y_offset - input_origin[1],
            )
        )

        # Only advance x_ptr for training examples or when not grouping
        if not is_test or not has_grouping:
            x_ptr += max(input_size[0], actual_output_width)

        y_ptr = max(y_ptr, input_size[1])

    # Second pass: Draw arrows and outputs
    y_ptr2 = y_offset

    for i, (
        input_grid,
        output_grid,
        input_mask,
        output_mask,
        label,
        is_test,
    ) in enumerate(examples):
        # Recalculate input for positioning
        input_result = draw_grid_svg(
            input_grid,
            input_mask,
            max_width=allocation[i],
            max_height=ymax,
            label=f"In #{label}",
            border_color=border_colors[0],
            padding=padding,
            extra_bottom_padding=extra_bottom_padding,
            as_group=True,
        )

        if isinstance(input_result, tuple):
            input_group, input_origin, input_size = input_result
        else:
            msg = "Expected tuple result when as_group=True"
            raise ValueError(msg)

        output_g = None
        output_x_recalc = 0.0
        output_y_total_height = 0.0
        output_origin_recalc = (-padding / 2, -padding / 2)

        show_output = (not is_test) or (include_test == "all")

        if show_output and output_grid is not None:
            output_result = draw_grid_svg(
                output_grid,
                output_mask,
                max_width=allocation[i],
                max_height=ymax,
                label=f"Out #{label}",
                border_color=border_colors[1],
                padding=padding,
                extra_bottom_padding=extra_bottom_padding,
                as_group=True,
            )

            if isinstance(output_result, tuple):
                output_g, output_origin_recalc, output_size = output_result
                output_x_recalc = output_size[0]
                output_y_total_height = output_size[1]
            else:
                msg = "Expected tuple result when as_group=True"
                raise ValueError(msg)
        else:
            # Approximate height for '?' slot
            output_y_total_height = ymax + padding + extra_bottom_padding

        # Determine x position based on whether this is a test example
        if is_test and has_grouping:
            # For test examples, position relative to test start
            test_index = i - len(train_examples)
            test_x_offset = (
                sum(allocation[len(train_examples) : len(train_examples) + test_index])
                + padding * test_index
            )
            current_x_ptr = test_start_x + test_x_offset
        else:
            # For training examples, calculate position from start
            train_x_offset = sum(allocation[:i]) + padding * i
            current_x_ptr = x_offset + train_x_offset

        # Draw arrow
        arrow_x_center = current_x_ptr + input_size[0] / 2
        arrow_top_y = y_ptr + padding - 0.6
        arrow_bottom_y = y_ptr + padding + io_gap - 0.6

        drawlist.append(
            drawsvg.Line(
                arrow_x_center,
                arrow_top_y,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )
        drawlist.append(
            drawsvg.Line(
                arrow_x_center - 0.15,
                arrow_bottom_y - 0.2,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )
        drawlist.append(
            drawsvg.Line(
                arrow_x_center + 0.15,
                arrow_bottom_y - 0.2,
                arrow_x_center,
                arrow_bottom_y,
                stroke_width=0.05,
                stroke="#888888",
            )
        )

        # Position output
        y_content_top_output_area = y_ptr + io_gap

        if show_output and output_g is not None:
            drawlist.append(
                drawsvg.Use(
                    output_g,
                    x=current_x_ptr
                    + (allocation[i] + padding - output_x_recalc) / 2
                    - output_origin_recalc[0],
                    y=y_ptr - output_origin_recalc[1] + io_gap,
                )
            )
        else:
            # Draw question mark
            q_text_y_center = (
                y_content_top_output_area + (ymax / 2) + extra_bottom_padding / 2
            )
            drawlist.append(
                drawsvg.Text(
                    "?",
                    x=current_x_ptr + (allocation[i] + padding) / 2,
                    y=q_text_y_center,
                    font_size=1.0,
                    font_family="Anuphan",
                    font_weight="700",
                    fill="#333333",
                    text_anchor="middle",
                    alignment_baseline="middle",
                )
            )

        y_ptr2 = max(y_ptr2, y_ptr + io_gap + output_y_total_height)

    # Calculate final drawing dimensions accounting for squircle margins
    if has_grouping:
        test_width = (
            sum(allocation[len(train_examples) :]) + padding * len(test_examples)
            if test_examples
            else 0
        )
        final_drawing_width = round(
            x_offset + train_width + group_spacing + test_width + squircle_margin, 1
        )
    else:
        final_drawing_width = round(x_ptr, 1)
    final_drawing_height = round(y_ptr2 + (squircle_margin if has_grouping else 0), 1)

    # Ensure dimensions are not negative or too small
    final_drawing_width = max(final_drawing_width, 1.0)
    final_drawing_height = max(final_drawing_height, height)  # Height is strict

    # Create final drawing with calculated dimensions
    drawing = drawsvg.Drawing(
        final_drawing_width, final_drawing_height + 0.3, origin=(0, 0)
    )
    drawing.append(drawsvg.Rectangle(0, 0, "100%", "100%", fill="#eeeff6"))

    # Add all draw elements
    for item in drawlist:
        drawing.append(item)

    # Add grouping squircles if we have both training and test examples
    if len(train_examples) > 0 and len(test_examples) > 0:
        # Calculate training group bounds
        train_width = sum(allocation[: len(train_examples)]) + padding * len(
            train_examples
        )

        # Training group squircle
        train_squircle_elements = _draw_dotted_squircle(
            x=0,
            y=0,
            width=train_width + squircle_margin * 2,
            height=y_ptr2 - y_offset + squircle_margin,
            label="Train",
            stroke_color="#4A90E2",
        )
        for element in train_squircle_elements:
            drawing.append(element)

        # Test group squircle
        test_start_x = train_width + group_spacing + squircle_margin
        test_width = sum(allocation[len(train_examples) :]) + padding * len(
            test_examples
        )
        test_squircle_elements = _draw_dotted_squircle(
            x=test_start_x,
            y=0,
            width=test_width + squircle_margin,
            height=y_ptr2 - y_offset + squircle_margin,
            label="Test",
            stroke_color="#E94B3C",
        )
        for element in test_squircle_elements:
            drawing.append(element)

    # Add title
    font_size = 0.3
    title_text = f"Task: {extract_task_id_from_index(task_data.task_index)}"
    drawing.append(
        drawsvg.Text(
            title_text,
            x=final_drawing_width - 0.1,
            y=final_drawing_height + 0.2,
            font_size=font_size,
            font_family="Anuphan",
            font_weight="600",
            fill="#666666",
            text_anchor="end",
            alignment_baseline="bottom",
        )
    )

    # Embed font and set scale
    drawing.embed_google_font(
        "Anuphan:wght@400;600;700",
        text=set(
            "Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
    )
    drawing.set_pixel_scale(40)

    return drawing