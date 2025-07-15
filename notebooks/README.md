# JaxARC Notebooks

This directory contains interactive notebooks demonstrating the JaxARC system
capabilities and providing hands-on tutorials for working with the codebase.

## 📚 Available Notebooks

### `action_handler_demo.py`

A comprehensive demonstration of the JaxARC action handler system showcasing:

- **Point Actions**: Individual cell selection and operations
  (`{"point": [row, col], "operation": op}`)
- **Bbox Actions**: Rectangular region selection and operations
  (`{"bbox": [r1, c1, r2, c2], "operation": op}`)
- **Mask Actions**: Arbitrary region selection using boolean masks
  (`{"mask": flattened_mask, "operation": op}`)
- **Batch Processing**: JAX vmap integration for efficient batch operations
- **Performance Comparison**: Benchmarking different action formats (all
  JIT-compiled)
- **Professional SVG Visualization**: High-quality matplotlib-based grid
  visualizations with selection overlays, saved as PNG and SVG formats
- **Custom Patterns**: Creating checkerboard, border, cross, and circular
  selection patterns
- **Low-Level API**: Direct access to action handler functions for advanced
  usage
- **Comprehensive Analysis**: Before/after comparisons, performance charts, and
  batch processing demonstrations

### `test_quick_integration.py`

A validation script that verifies all action handler functionality:

- Tests point, bbox, and mask action formats with the environment
- Validates low-level handler API functionality
- Confirms batch processing with JAX vmap works correctly
- Ensures all visualizations render properly

## 🚀 Getting Started

### Prerequisites

Make sure you have the JaxARC environment set up:

```bash
cd JaxARC
pixi shell
```

### Running the Notebooks

#### Option 1: Jupytext (Recommended)

The notebooks are in jupytext format (`.py` files with special markup). To run
them:

```bash
# Install jupytext if not already installed
pip install jupytext

# Convert to Jupyter notebook
jupytext --to notebook action_handler_demo.py

# Launch Jupyter
jupyter notebook action_handler_demo.ipynb
```

#### Option 2: Direct Python Execution

You can also run the notebook as a Python script to generate SVG visualizations:

```bash
pixi run python action_handler_demo.py
```

This will generate high-quality PNG and SVG visualizations in the
`notebooks/output/` directory.

#### Option 3: VS Code Integration

If you're using VS Code with the Python extension:

1. Install the Jupyter extension
2. Open the `.py` file
3. VS Code will automatically detect it as a notebook
4. Click "Run Cell" buttons or use Ctrl+Enter

## 🎯 What You'll Learn

### Action Handler System

- How to configure different action formats (point, bbox, mask)
- Loading real ARC task data using Hydra configurations
- Creating and executing actions in different formats
- Understanding the low-level action handler API

### JAX Integration

- How action handlers are JIT-compiled for performance
- Using `vmap` for batch processing
- Performance characteristics of different action formats

### Professional Visualization

- High-quality matplotlib-based grid visualizations
- SVG and PNG output for documentation and analysis
- Selection overlays with color-coded highlights
- Performance comparison charts and batch processing demos
- Before/after action comparisons

## 📋 Dependencies

The notebooks require the following packages (included in JaxARC environment):

- `jax` and `jax.numpy` - Core JAX functionality
- `matplotlib` - Plotting and visualization
- `rich` - Terminal formatting and visualization
- `hydra-core` - Configuration management
- `jupytext` - Notebook format conversion (optional)
- `jupyter` - Notebook interface (optional)

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'jaxarc'" Error**

- Make sure you're in the pixi environment: `pixi shell`
- Verify the package is installed: `pip list | grep jaxarc`

**Config Loading Errors**

- Ensure you're running from the JaxARC project root
- Check that `conf/` directory exists with the required config files

**Visualization Issues**

- For matplotlib output, ensure you have a display backend:
  `pip install matplotlib[tk]`
- SVG files are saved to `notebooks/output/` directory
- If PNG generation fails, SVG files will still be created

**JAX Compilation Warnings**

- First-time JIT compilation may show warnings - this is normal
- Subsequent runs will be faster due to caching

### Performance Tips

1. **First Run**: Initial execution may be slow due to JAX compilation
2. **Batch Processing**: Use vmap for processing multiple actions efficiently
3. **Memory**: Large grids may require more memory - adjust batch sizes
   accordingly

## 🎨 Customization

### Adding New Demonstrations

To add your own action demonstrations:

1. Create new action dictionaries following the format patterns
2. Add them to the `demonstrate_action_format` function calls
3. Implement custom visualization functions as needed

### Custom Action Patterns

Use the helper functions in the notebook to create custom selection patterns:

- `create_custom_action_pattern()` - Generate geometric patterns
- `create_circle_mask()` - Create circular selections
- `create_line_mask()` - Create linear selections

### Configuration Variants

Experiment with different configurations by modifying the Hydra overrides:

```python
# Example: Custom action configuration
custom_cfg = compose(
    config_name="config",
    overrides=[
        "action=point",
        "action.clip_invalid_actions=false",
        "action.num_operations=25",
    ],
)

# Or create ActionConfig directly:
from jaxarc.envs.config import ArcEnvConfig, ActionConfig

config = ArcEnvConfig(action=ActionConfig(selection_format="point"))
```

## 📖 Next Steps

After exploring the action handler demo, consider:

1. **Environment Integration**: Run `test_quick_integration.py` to verify your
   setup
2. **Custom Tasks**: Load real ARC tasks using the dataset configuration
3. **Agent Development**: Use the action handlers as building blocks for RL
   agents
4. **Performance Analysis**: Benchmark different action formats on various task
   types
5. **Advanced Visualizations**: Create custom SVG visualizations for your tasks
6. **Batch Training**: Leverage the vmap compatibility for efficient agent
   training

## 🎯 Key Insights from the Demo

- **Action Format Consistency**: All actions use
  `{"selection": data, "operation": op}` format
- **JIT Compilation**: All handlers are JIT-compiled for optimal performance
  (~0.02-0.03ms per call)
- **Batch Processing**: Seamless integration with JAX vmap for parallel
  processing
- **Professional Visualization**: High-quality matplotlib + SVG output for
  documentation and analysis
- **Grid Constraints**: Actions are automatically constrained to valid grid
  regions
- **Configuration Flexibility**: Easy switching between action formats via Hydra
  configs
- **Multi-format Output**: Generates both PNG (for viewing) and SVG (for
  scalable documentation)

## 🤝 Contributing

Found an issue or have suggestions for new notebooks? Please:

1. Check existing issues in the main repository
2. Create a new issue with the `notebook` label
3. Submit a pull request with your improvements

## 🔧 Validation

Before diving into the notebooks, you can verify everything works by running:

```bash
pixi run python notebooks/test_quick_integration.py
```

This will test all action formats, visualization utilities, and confirm the
system is working correctly.

You can also test the matplotlib visualization system separately:

```bash
pixi run python notebooks/test_matplotlib.py
```

## 📄 License

These notebooks are part of the JaxARC project and follow the same license
terms.
