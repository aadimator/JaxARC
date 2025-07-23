# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # JaxARC Action Handler System Demo
#
# This notebook demonstrates the JaxARC action handler system with professional SVG-based visualizations.
# We'll explore three different action formats with high-quality grid renderings:
# - **Point Actions**: Individual cell selection and operations
# - **Bbox Actions**: Rectangular region selection and operations
# - **Mask Actions**: Arbitrary region selection using boolean masks
#
# Each format is demonstrated with custom SVG visualizations composed using matplotlib.

# %% [markdown]
# ## Setup and Imports

# %%
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from jaxarc.envs.actions import get_action_handler

# JaxARC imports
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.environment import ArcEnvironment
from jaxarc.utils.visualization import (
    create_action_demonstration_figure,
    create_grid_matplotlib,
    draw_grid_svg,
    draw_task_pair_svg,
    save_svg_drawing,
    setup_matplotlib_style,
)

# Set up matplotlib styling using JaxARC utilities
setup_matplotlib_style()

print("🎨 JaxARC Action Handler Demo - SVG Visualization Setup Complete! 🚀")

# %% [markdown]
# ## Configuration and Environment Setup

# %%
from hydra import compose, initialize_config_dir
from pyprojroot import here

from jaxarc.parsers import ArcAgiParser

# Get config directory
project_root = here()
config_dir = project_root / "conf"
output_dir = project_root / "notebooks" / "output"
output_dir.mkdir(exist_ok=True)

print(f"📁 Config directory: {config_dir}")
print(f"📁 Output directory: {output_dir}")

# Initialize hydra and load configurations
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    point_cfg = compose(config_name="config", overrides=["action=point"])
    bbox_cfg = compose(config_name="config", overrides=["action=bbox"])
    mask_cfg = compose(config_name="config", overrides=["action=standard"])

# Dataset Parser
parser = ArcAgiParser(point_cfg.dataset)

# Create unified config objects directly
unified_point_config = JaxArcConfig.from_hydra(point_cfg)
unified_bbox_config = JaxArcConfig.from_hydra(bbox_cfg)
unified_mask_config = JaxArcConfig.from_hydra(mask_cfg)

point_env = ArcEnvironment(unified_point_config)
bbox_env = ArcEnvironment(unified_bbox_config)
mask_env = ArcEnvironment(unified_mask_config)

# Initialize environments
key = jax.random.PRNGKey(42)
point_state, point_obs = point_env.reset(key)
bbox_state, bbox_obs = bbox_env.reset(key)
mask_state, mask_obs = mask_env.reset(key)

print("✅ All environments initialized successfully!")

# %% [markdown]
# ## Helper Functions for Mask Creation


# %% [markdown]
# ## Point Action Format Demonstration

# %%
print("🎯 Demonstrating Point Action Format")
print("=" * 50)

# Define point actions
point_demo_actions = [
    (
        {"point": jnp.array([5, 5]), "operation": jnp.array(1)},
        "Fill cell (5,5) with blue",
    ),
    (
        {"point": jnp.array([5, 6]), "operation": jnp.array(2)},
        "Fill cell (5,6) with red",
    ),
    (
        {"selection": jnp.array([6, 5]), "operation": jnp.array(3)},
        "Fill cell (6,5) with green",
    ),
    (
        {"selection": jnp.array([6, 6]), "operation": jnp.array(4)},
        "Fill cell (6,6) with yellow",
    ),
]

# Create demonstration figure
fig_point, point_results = create_action_demonstration_figure(
    point_env, point_state, "point", point_demo_actions
)

# Save the figure
fig_point.savefig(output_dir / "point_action_demo.png", dpi=300, bbox_inches="tight")
fig_point.savefig(output_dir / "point_action_demo.svg", bbox_inches="tight")

plt.show()

# Print results summary
print("\n📊 Point Action Results:")
for i, (state, reward, selected) in enumerate(point_results):
    if state is not None:
        print(
            f"  Step {i + 1}: Reward={reward:.3f}, Selected={selected}, Similarity={state.similarity_score:.3f}"
        )

# %% [markdown]
# ## Bbox Action Format Demonstration

# %%
print("\n🎯 Demonstrating Bbox Action Format")
print("=" * 50)

# Define bbox actions
bbox_demo_actions = [
    (
        {"bbox": jnp.array([3, 3, 5, 5]), "operation": jnp.array(1)},
        "Fill 3x3 rectangle with blue",
    ),
    (
        {"bbox": jnp.array([7, 7, 9, 9]), "operation": jnp.array(2)},
        "Fill 3x3 rectangle with red",
    ),
    (
        {"selection": jnp.array([3, 7, 5, 9]), "operation": jnp.array(3)},
        "Fill 3x3 rectangle with green",
    ),
]

# Create demonstration figure
fig_bbox, bbox_results = create_action_demonstration_figure(
    bbox_env, bbox_state, "bbox", bbox_demo_actions
)

# Save the figure
fig_bbox.savefig(output_dir / "bbox_action_demo.png", dpi=300, bbox_inches="tight")
fig_bbox.savefig(output_dir / "bbox_action_demo.svg", bbox_inches="tight")

plt.show()

# Print results summary
print("\n📊 Bbox Action Results:")
for i, (state, reward, selected) in enumerate(bbox_results):
    if state is not None:
        print(
            f"  Step {i + 1}: Reward={reward:.3f}, Selected={selected}, Similarity={state.similarity_score:.3f}"
        )

# %% [markdown]
# ## Mask Action Format Demonstration

# %%
print("\n🎯 Demonstrating Mask Action Format")
print("=" * 50)


# Helper functions for creating custom masks
def create_circle_mask(center_row, center_col, radius, grid_shape):
    """Create a circular mask."""
    rows, cols = jnp.meshgrid(
        jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]), indexing="ij"
    )
    distance = jnp.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    return distance <= radius


def create_line_mask(start_row, start_col, end_row, end_col, grid_shape):
    """Create a line mask."""
    mask = jnp.zeros(grid_shape, dtype=jnp.bool_)

    # Simple line approximation
    steps = max(abs(end_row - start_row), abs(end_col - start_col))
    if steps == 0:
        return mask.at[start_row, start_col].set(True)

    row_step = (end_row - start_row) / steps
    col_step = (end_col - start_col) / steps

    for i in range(steps + 1):
        row = int(start_row + i * row_step)
        col = int(start_col + i * col_step)
        if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
            mask = mask.at[row, col].set(True)

    return mask


def create_cross_mask(center_row, center_col, size, grid_shape):
    """Create a cross pattern mask."""
    mask = jnp.zeros(grid_shape, dtype=jnp.bool_)

    # Horizontal line
    start_col = max(0, center_col - size)
    end_col = min(grid_shape[1], center_col + size + 1)
    mask = mask.at[center_row, start_col:end_col].set(True)

    # Vertical line
    start_row = max(0, center_row - size)
    end_row = min(grid_shape[0], center_row + size + 1)
    mask = mask.at[start_row:end_row, center_col].set(True)

    return mask


# Create masks for demonstration
grid_shape = mask_state.working_grid.shape
circle_mask = create_circle_mask(8, 8, 3, grid_shape)
line_mask = create_line_mask(5, 5, 15, 15, grid_shape)
cross_mask = create_cross_mask(10, 12, 4, grid_shape)

# Define mask actions
mask_demo_actions = [
    (
        {"selection": circle_mask.flatten(), "operation": jnp.array(1)},
        "Fill circular region with blue",
    ),
    (
        {"selection": line_mask.flatten(), "operation": jnp.array(2)},
        "Fill diagonal line with red",
    ),
    (
        {"selection": cross_mask.flatten(), "operation": jnp.array(3)},
        "Fill cross pattern with green",
    ),
]

# Create demonstration figure
fig_mask, mask_results = create_action_demonstration_figure(
    mask_env, mask_state, "mask", mask_demo_actions
)

# Save the figure
fig_mask.savefig(output_dir / "mask_action_demo.png", dpi=300, bbox_inches="tight")
fig_mask.savefig(output_dir / "mask_action_demo.svg", bbox_inches="tight")

plt.show()

# Print results summary
print("\n📊 Mask Action Results:")
for i, (state, reward, selected) in enumerate(mask_results):
    if state is not None:
        print(
            f"  Step {i + 1}: Reward={reward:.3f}, Selected={selected}, Similarity={state.similarity_score:.3f}"
        )

# %% [markdown]
# ## Performance Analysis and Comparison

# %%
print("\n⚡ Performance Analysis")
print("=" * 50)


# Performance testing
def benchmark_action_handler(handler, action_data, working_mask, iterations=1000):
    """Benchmark an action handler."""
    # Warm-up
    for _ in range(10):
        _ = handler(action_data, working_mask)

    # Timing
    start_time = time.time()
    for _ in range(iterations):
        result = handler(action_data, working_mask)
    end_time = time.time()

    return (end_time - start_time) / iterations, result


# Test setup
test_grid_shape = (20, 20)
working_mask = jnp.ones(test_grid_shape, dtype=jnp.bool_)
iterations = 1000

# Get handlers
point_handler = get_action_handler("point")
bbox_handler = get_action_handler("bbox")
mask_handler = get_action_handler("mask")

# Test data
point_data = jnp.array([10, 10])
bbox_data = jnp.array([5, 5, 15, 15])
mask_data = jnp.ones(test_grid_shape[0] * test_grid_shape[1], dtype=jnp.bool_) * 0.3

# Run benchmarks
point_time, point_result = benchmark_action_handler(
    point_handler, point_data, working_mask, iterations
)
bbox_time, bbox_result = benchmark_action_handler(
    bbox_handler, bbox_data, working_mask, iterations
)
mask_time, mask_result = benchmark_action_handler(
    mask_handler, mask_data, working_mask, iterations
)

# Create performance comparison figure
fig_perf, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_perf.suptitle("Action Handler Performance Analysis", fontsize=16, fontweight="bold")

# Performance bar chart
ax_perf = axes[0, 0]
handlers = ["Point", "Bbox", "Mask"]
times = [
    point_time * 1000,
    bbox_time * 1000,
    mask_time * 1000,
]  # Convert to milliseconds
colors = ["skyblue", "lightgreen", "lightcoral"]

bars = ax_perf.bar(handlers, times, color=colors, alpha=0.7, edgecolor="black")
ax_perf.set_ylabel("Time per call (ms)")
ax_perf.set_title("Handler Performance Comparison")
ax_perf.grid(True, alpha=0.3)

# Add value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax_perf.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.001,
        f"{time_val:.3f}ms",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Selection results visualization
results = [point_result, bbox_result, mask_result]
selected_counts = [jnp.sum(result) for result in results]

ax_sel = axes[0, 1]
bars_sel = ax_sel.bar(
    handlers, selected_counts, color=colors, alpha=0.7, edgecolor="black"
)
ax_sel.set_ylabel("Selected Cells")
ax_sel.set_title("Selection Results")
ax_sel.grid(True, alpha=0.3)

for bar, count in zip(bars_sel, selected_counts):
    height = bar.get_height()
    ax_sel.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 5,
        f"{int(count)}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Visualize handler results
ax_point = axes[1, 0]
create_grid_matplotlib(
    point_result.astype(int), title="Point Handler Result", ax=ax_point
)

ax_bbox = axes[1, 1]
create_grid_matplotlib(bbox_result.astype(int), title="Bbox Handler Result", ax=ax_bbox)

plt.tight_layout()
fig_perf.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches="tight")
fig_perf.savefig(output_dir / "performance_analysis.svg", bbox_inches="tight")
plt.show()

print(f"\n📊 Performance Results ({iterations} iterations):")
print(f"  Point Handler: {point_time * 1000:.3f}ms per call")
print(f"  Bbox Handler:  {bbox_time * 1000:.3f}ms per call")
print(f"  Mask Handler:  {mask_time * 1000:.3f}ms per call")

# %% [markdown]
# ## Batch Processing Demonstration

# %%
print("\n🚀 Batch Processing Demonstration")
print("=" * 50)

# Create batch processing example
batch_size = 6
grid_shape = (12, 12)
working_masks = jnp.ones((batch_size,) + grid_shape, dtype=jnp.bool_)

# Point batch processing
point_batch = jnp.array([[2, 2], [3, 8], [6, 4], [8, 9], [4, 6], [9, 3]])

batch_point_handler = jax.vmap(point_handler, in_axes=(0, 0))
point_batch_results = batch_point_handler(point_batch, working_masks)

# Bbox batch processing
bbox_batch = jnp.array(
    [
        [1, 1, 3, 3],
        [5, 5, 7, 7],
        [2, 6, 4, 8],
        [6, 2, 8, 4],
        [8, 8, 10, 10],
        [4, 9, 6, 11],
    ]
)

batch_bbox_handler = jax.vmap(bbox_handler, in_axes=(0, 0))
bbox_batch_results = batch_bbox_handler(bbox_batch, working_masks)

# Create batch processing visualization
fig_batch, axes = plt.subplots(2, batch_size, figsize=(24, 8))
fig_batch.suptitle("Batch Processing Results", fontsize=16, fontweight="bold")

# Point batch results
for i in range(batch_size):
    ax = axes[0, i]
    create_grid_matplotlib(
        point_batch_results[i].astype(int),
        title=f"Point {i + 1}: {point_batch[i]}",
        ax=ax,
    )

# Bbox batch results
for i in range(batch_size):
    ax = axes[1, i]
    create_grid_matplotlib(
        bbox_batch_results[i].astype(int),
        title=f"Bbox {i + 1}: {bbox_batch[i][:2]}-{bbox_batch[i][2:]}",
        ax=ax,
    )

plt.tight_layout()
fig_batch.savefig(
    output_dir / "batch_processing_demo.png", dpi=300, bbox_inches="tight"
)
fig_batch.savefig(output_dir / "batch_processing_demo.svg", bbox_inches="tight")
plt.show()

print("\n📊 Batch Processing Results:")
print(f"  Point batch shape: {point_batch_results.shape}")
print(f"  Bbox batch shape: {bbox_batch_results.shape}")
print(f"  Point selections: {jnp.sum(point_batch_results, axis=(1, 2))}")
print(f"  Bbox selections: {jnp.sum(bbox_batch_results, axis=(1, 2))}")

# %% [markdown]
# ## Summary and Comparison

# %%
print("\n🎉 Final Summary and Comparison")
print("=" * 50)

# Create comprehensive summary figure
fig_summary = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig_summary, hspace=0.4, wspace=0.3)

fig_summary.suptitle(
    "JaxARC Action Handler System - Complete Overview", fontsize=20, fontweight="bold"
)

# Action format comparison (top row)
formats = ["Point", "Bbox", "Mask"]
sample_results = [
    point_results[-1][0] if point_results[-1][0] is not None else point_state,
    bbox_results[-1][0] if bbox_results[-1][0] is not None else bbox_state,
    mask_results[-1][0] if mask_results[-1][0] is not None else mask_state,
]

for i, (fmt, result) in enumerate(zip(formats, sample_results)):
    ax = fig_summary.add_subplot(gs[0, i])
    create_grid_matplotlib(
        result.working_grid, result.selected, title=f"{fmt} Action Result", ax=ax
    )

# Performance comparison (second row, left)
ax_perf_summary = fig_summary.add_subplot(gs[1, :2])
ax_perf_summary.bar(handlers, times, color=colors, alpha=0.7, edgecolor="black")
ax_perf_summary.set_ylabel("Time per call (ms)")
ax_perf_summary.set_title("Performance Comparison")
ax_perf_summary.grid(True, alpha=0.3)

# Selection efficiency (second row, right)
ax_sel_summary = fig_summary.add_subplot(gs[1, 2:])
efficiency_data = {
    "Point": [1, 1, 1, 1],  # Always selects 1 cell
    "Bbox": [9, 9, 9],  # Selects rectangle area
    "Mask": [jnp.sum(circle_mask), jnp.sum(line_mask), jnp.sum(cross_mask)],  # Variable
}

x_pos = np.arange(len(formats))
for i, fmt in enumerate(formats):
    ax_sel_summary.bar(
        i, np.mean(efficiency_data[fmt]), color=colors[i], alpha=0.7, edgecolor="black"
    )

ax_sel_summary.set_xticks(x_pos)
ax_sel_summary.set_xticklabels(formats)
ax_sel_summary.set_ylabel("Average Selected Cells")
ax_sel_summary.set_title("Selection Efficiency")
ax_sel_summary.grid(True, alpha=0.3)

# Feature comparison table (bottom half)
ax_table = fig_summary.add_subplot(gs[2:, :])
ax_table.axis("off")

# Create feature comparison data
features = ["Action Format", "Precision", "Flexibility", "Performance", "Use Case"]
point_features = ["[row, col]", "Single Cell", "Low", "Fast", "Precise Selection"]
bbox_features = ["[r1,c1,r2,c2]", "Rectangle", "Medium", "Fast", "Region Selection"]
mask_features = ["Boolean Mask", "Arbitrary", "High", "Fast", "Complex Patterns"]

# Create table
table_data = [
    ["Feature", "Point Actions", "Bbox Actions", "Mask Actions"],
    ["Format", "[row, col]", "[r1,c1,r2,c2]", "Boolean Mask"],
    ["Precision", "Single Cell", "Rectangle", "Arbitrary Shape"],
    ["Flexibility", "Low", "Medium", "High"],
    [
        "Performance",
        f"{point_time * 1000:.2f}ms",
        f"{bbox_time * 1000:.2f}ms",
        f"{mask_time * 1000:.2f}ms",
    ],
    ["Best Use Case", "Precise Selection", "Region Selection", "Complex Patterns"],
]

# Create table visualization
table = ax_table.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style the table
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8E8E8")
        elif j == 0:  # Feature column
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor("#FFFFFF")

ax_table.set_title(
    "Action Handler Feature Comparison", fontsize=16, fontweight="bold", pad=20
)

plt.tight_layout()
fig_summary.savefig(output_dir / "complete_summary.png", dpi=300, bbox_inches="tight")
fig_summary.savefig(output_dir / "complete_summary.svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Output Files Generated

# %%
print(f"\n📁 Generated Output Files in {output_dir}:")
output_files = [
    "point_action_demo.png",
    "point_action_demo.svg",
    "bbox_action_demo.png",
    "bbox_action_demo.svg",
    "mask_action_demo.png",
    "mask_action_demo.svg",
    "performance_analysis.png",
    "performance_analysis.svg",
    "batch_processing_demo.png",
    "batch_processing_demo.svg",
    "complete_summary.png",
    "complete_summary.svg",
]

for file in output_files:
    if (output_dir / file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} (not found)")

# %% [markdown]
# ## SVG Integration with Existing JaxARC Utilities

# %%
print("\n🎨 Creating SVG outputs using JaxARC utilities...")

# Create SVG visualizations using the existing JaxARC SVG utilities
if "point_results" in locals() and point_results and point_results[-1][0] is not None:
    final_point_state = point_results[-1][0]

    # Individual SVG grids
    point_svg = draw_grid_svg(
        final_point_state.working_grid,
        final_point_state.selected,
        label="Point Actions Result",
    )
    save_svg_drawing(point_svg, str(output_dir / "jaxarc_point_result.svg"))

    # Before/after comparison
    comparison_svg = draw_task_pair_svg(
        point_state.working_grid,
        final_point_state.working_grid,
        label="Point Actions: Before → After",
    )
    save_svg_drawing(comparison_svg, str(output_dir / "jaxarc_point_comparison.svg"))

    print("  ✅ jaxarc_point_result.svg")
    print("  ✅ jaxarc_point_comparison.svg")

if "bbox_results" in locals() and bbox_results and bbox_results[-1][0] is not None:
    final_bbox_state = bbox_results[-1][0]

    bbox_svg = draw_grid_svg(
        final_bbox_state.working_grid,
        final_bbox_state.selected,
        label="Bbox Actions Result",
    )
    save_svg_drawing(bbox_svg, str(output_dir / "jaxarc_bbox_result.svg"))
    print("  ✅ jaxarc_bbox_result.svg")

if "mask_results" in locals() and mask_results and mask_results[-1][0] is not None:
    final_mask_state = mask_results[-1][0]

    mask_svg = draw_grid_svg(
        final_mask_state.working_grid,
        final_mask_state.selected,
        label="Mask Actions Result",
    )
    save_svg_drawing(mask_svg, str(output_dir / "jaxarc_mask_result.svg"))
    print("  ✅ jaxarc_mask_result.svg")

# %% [markdown]
# ## Summary and Conclusions

# %%
print("\n" + "=" * 60)
print("🎉 JAXARC ACTION HANDLER DEMO COMPLETE!")
print("=" * 60)

print("\n✅ Successfully demonstrated:")
print("  🎯 Point Actions: Individual cell selection")
print("  📦 Bbox Actions: Rectangular region selection")
print("  🎭 Mask Actions: Arbitrary shape selection")
print("  🚀 Batch Processing: JAX vmap integration")
print("  ⚡ Performance: JIT-compiled handlers")
print("  🎨 Visualization: Matplotlib + SVG output")
print("  📊 Analysis: Performance comparison")

print("\n📊 Key Performance Metrics:")
print(f"  • Point Handler: {point_time * 1000:.3f}ms per call")
print(f"  • Bbox Handler: {bbox_time * 1000:.3f}ms per call")
print(f"  • Mask Handler: {mask_time * 1000:.3f}ms per call")
print("  • All handlers are JIT-compiled for optimal performance")

print("\n🎨 Generated Visualizations:")
print(f"  • {len(output_files)} output files in {output_dir}")
print("  • High-resolution PNG and SVG formats")
print("  • Comprehensive performance analysis")
print("  • Batch processing demonstrations")

print("\n🔧 Technical Achievements:")
print("  • Standardized action format: {'selection': data, 'operation': op}")
print("  • JAX-native implementation with vmap support")
print("  • Professional SVG-based visualizations")
print("  • Complete integration with JaxARC environment")

print("\n🚀 Next Steps:")
print("  • Use action handlers for RL agent development")
print("  • Explore real ARC task datasets")
print("  • Implement custom action strategies")
print("  • Build multi-agent systems")

print("\n💡 The JaxARC action handler system provides a robust,")
print("   efficient, and flexible foundation for ARC-solving agents!")
