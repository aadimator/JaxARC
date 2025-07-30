# JaxARC Visualization Module

This directory contains the modular visualization system for JaxARC, providing comprehensive visualization capabilities for ARC grids, tasks, and RL training episodes.

## Module Structure

The visualization system has been refactored from a single large `core.py` file into smaller, focused modules:

### Core Modules

- **`constants.py`** - Color palettes and visualization constants
  - `ARC_COLOR_PALETTE` - Standard ARC color mapping

- **`utils.py`** - Helper functions and data extraction utilities
  - `_extract_grid_data()` - Extract numpy arrays from various grid formats
  - `_extract_valid_region()` - Extract valid (non-padded) regions from grids
  - `detect_changed_cells()` - Detect changes between grid states
  - `get_color_name()` - Human-readable color names
  - `infer_fill_color_from_grids()` - Infer fill colors from grid changes

### Display Modules

- **`rich_display.py`** - Rich terminal visualization functions
  - `visualize_grid_rich()` - Rich table visualization for single grids
  - `log_grid_to_console()` - Console logging with Rich formatting
  - `visualize_task_pair_rich()` - Rich visualization for input-output pairs
  - `visualize_parsed_task_data_rich()` - Complete task visualization

- **`svg_core.py`** - Core SVG drawing utilities
  - `draw_grid_svg()` - SVG generation for single grids
  - `save_svg_drawing()` - Save SVG drawings to files
  - `add_selection_visualization_overlay()` - Add selection overlays
  - `add_change_highlighting()` - Add change highlighting

### Specialized Modules

- **`task_visualization.py`** - Task and task pair visualization
  - `draw_task_pair_svg()` - SVG generation for task pairs
  - `draw_parsed_task_data_svg()` - SVG generation for complete tasks

- **`rl_visualization.py`** - RL-specific visualization functions
  - `draw_rl_step_svg()` - Visualization of RL step transitions
  - `save_rl_step_visualization()` - Save step visualizations to disk
  - `get_operation_display_name()` - Human-readable operation names
  - `create_action_summary_panel()` - Action summary panels
  - `create_metrics_visualization()` - Metrics visualization panels

- **`episode_visualization.py`** - Episode summary and comparison functions
  - `draw_episode_summary_svg()` - Episode summary visualization
  - `draw_enhanced_episode_summary_svg()` - Enhanced episode summaries
  - `create_episode_comparison_visualization()` - Multi-episode comparisons

- **`matplotlib_utils.py`** - Matplotlib integration utilities
  - `setup_matplotlib_style()` - Configure matplotlib styling

### Integration Modules

- **`analysis_tools.py`** - Analysis and metrics tools
- **`async_logger.py`** - Asynchronous logging functionality
- **`episode_manager.py`** - Episode management functionality
- **`jax_callbacks.py`** - JAX integration and performance optimization
- **`memory_manager.py`** - Memory management and caching
- **`replay_system.py`** - Replay and analysis functionality
- **`visualizer.py`** - Main visualization system
- **`wandb_sync.py`** - Weights & Biases integration

## Dependency Hierarchy

The modules are organized in a clear dependency hierarchy to avoid circular imports:

```
constants.py (no dependencies)
    ↓
utils.py (imports constants)
    ↓
svg_core.py, rich_display.py (import constants, utils)
    ↓
task_visualization.py (imports constants, utils, svg_core)
    ↓
rl_visualization.py (imports constants, utils, svg_core, task_visualization)
    ↓
episode_visualization.py (imports constants, utils, svg_core, rl_visualization)
    ↓
matplotlib_utils.py (imports constants)
```

## Public API

The `__init__.py` file maintains the same public API as before the refactoring, ensuring backward compatibility. All functions that were previously available from `jaxarc.utils.visualization` continue to work exactly as before.

### Key Functions

- **Grid Visualization**: `visualize_grid_rich()`, `draw_grid_svg()`, `log_grid_to_console()`
- **Task Visualization**: `visualize_task_pair_rich()`, `draw_task_pair_svg()`, `visualize_parsed_task_data_rich()`, `draw_parsed_task_data_svg()`
- **RL Visualization**: `draw_rl_step_svg()`, `save_rl_step_visualization()`
- **Episode Visualization**: `draw_episode_summary_svg()`
- **Utilities**: `save_svg_drawing()`, `setup_matplotlib_style()`
- **Constants**: `ARC_COLOR_PALETTE`

## Benefits of Refactoring

1. **Modularity**: Each module has a clear, focused responsibility
2. **Maintainability**: Easier to find, understand, and modify specific functionality
3. **Testability**: Smaller modules are easier to test in isolation
4. **Reusability**: Individual modules can be imported and used independently
5. **Scalability**: New visualization features can be added to appropriate modules
6. **Performance**: Reduced import overhead for specific functionality
7. **Backward Compatibility**: Existing code continues to work without changes

## Usage Examples

```python
# Import specific functionality
from jaxarc.utils.visualization.constants import ARC_COLOR_PALETTE
from jaxarc.utils.visualization.rich_display import visualize_grid_rich
from jaxarc.utils.visualization.svg_core import draw_grid_svg

# Or use the unified API (recommended for backward compatibility)
from jaxarc.utils.visualization import (
    ARC_COLOR_PALETTE,
    visualize_grid_rich,
    draw_grid_svg,
    draw_rl_step_svg,
    save_svg_drawing,
)
```

## Testing

The refactoring maintains full backward compatibility. All existing tests continue to pass, and the module structure supports better unit testing of individual components.