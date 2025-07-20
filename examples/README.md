# JaxARC Examples

This directory contains comprehensive examples demonstrating JaxARC features, organized by complexity and purpose.

## Directory Structure

```
examples/
├── basic/           # Fundamental concepts and getting started
├── advanced/        # Complex features and optimization
├── integration/     # External tool integrations
└── deprecated/      # Outdated examples (do not use)
```

## Quick Start

If you're new to JaxARC, follow this learning path:

### 1. Basic Examples (`basic/`)
Start here to understand core concepts:
- **Configuration System**: Learn the unified configuration approach
- **Environment Setup**: Basic environment initialization and testing

### 2. Advanced Examples (`advanced/`)
Explore sophisticated features:
- **JAX Integration**: Advanced JAX callback patterns
- **Performance Optimization**: Memory management and optimization
- **Analysis Tools**: Episode replay and analysis systems

### 3. Integration Examples (`integration/`)
Connect with external tools:
- **Weights & Biases**: Experiment tracking and visualization
- **Error Handling**: Robust integration patterns

## Running Examples

All examples use the pixi environment manager:

```bash
# Basic configuration demo
pixi run python examples/basic/config_factory_demo.py

# Advanced JAX callbacks
pixi run python examples/advanced/jax_callbacks_demo.py

# WandB integration
pixi run python examples/integration/wandb_integration_demo.py
```

## Example Categories

### Basic Examples
Perfect for beginners and understanding fundamentals:
- ✅ **Working with current API**
- 🎯 **Focus on core concepts**
- 📚 **Well-documented with clear explanations**
- 🚀 **Quick to run and understand**

### Advanced Examples
For users ready to explore sophisticated features:
- ⚡ **Performance optimization techniques**
- 🔧 **Complex JAX integration patterns**
- 📊 **Analysis and debugging tools**
- 🏗️ **Production-ready patterns**

### Integration Examples
For connecting JaxARC with external tools:
- 🔗 **External service integration**
- 🛡️ **Robust error handling**
- 📡 **Offline/online mode support**
- ⚙️ **Configuration-driven setup**

## Key Features Demonstrated

### Configuration System
- **Unified Configuration**: Single JaxArcConfig replaces dual config patterns
- **Factory Functions**: Easy creation of common configurations
- **Preset System**: Pre-built configurations for different use cases
- **Validation**: Comprehensive configuration validation

### Environment Features
- **Action Formats**: Multiple ways to specify actions (mask, point, bbox)
- **Reward Systems**: Different reward structures for various training scenarios
- **Grid Operations**: 35 different operations for grid manipulation
- **Task Management**: Loading and managing ARC tasks

### Advanced Features
- **JAX Integration**: Proper use of JAX transformations with side effects
- **Memory Management**: Efficient handling of large datasets
- **Performance Monitoring**: Real-time performance tracking
- **Analysis Tools**: Comprehensive episode analysis and replay

### External Integrations
- **Experiment Tracking**: WandB integration with automatic organization
- **Offline Support**: Robust offline mode with automatic sync
- **Error Recovery**: Comprehensive error handling and retry logic
- **Image Optimization**: Efficient image processing for logging

## Deprecated Examples

The `deprecated/` directory contains examples that:
- ❌ Use outdated API patterns
- ❌ Have import errors or compatibility issues
- ❌ Are redundant with newer examples
- ❌ Use deprecated configuration systems

**Do not use deprecated examples.** They are kept for reference only and will be removed in future versions.

## Migration from Deprecated Examples

If you were using deprecated examples, here are the modern equivalents:

| Deprecated Example | Modern Equivalent | Location |
|-------------------|-------------------|----------|
| `config_api_demo.py` | `config_factory_demo.py` | `basic/` |
| `enhanced_config_demo.py` | `config_factory_demo.py` | `basic/` |
| `visualization_demo.py` | `jax_callbacks_demo.py` | `advanced/` |
| `arc_agi_*_usage_example.py` | Use ConfigFactory presets | `basic/` |

## Contributing Examples

When adding new examples:

1. **Choose the right category**:
   - `basic/` for fundamental concepts
   - `advanced/` for complex features
   - `integration/` for external tools

2. **Follow naming conventions**:
   - Use descriptive names ending in `_demo.py`
   - Include the main feature in the filename

3. **Include comprehensive documentation**:
   - Clear docstrings explaining the purpose
   - Step-by-step comments in the code
   - Usage instructions in README files

4. **Ensure compatibility**:
   - Use the unified configuration system
   - Test with current API
   - Handle errors gracefully

5. **Add to README**:
   - Update the appropriate category README
   - Include run instructions
   - Explain key concepts demonstrated

## Getting Help

- **Configuration Issues**: Start with `basic/config_factory_demo.py`
- **Environment Problems**: Check `basic/test_config_environments.py`
- **Performance Questions**: Explore `advanced/memory_optimization_demo.py`
- **Integration Help**: Review examples in `integration/`
- **API Changes**: Check migration guide in main documentation

## Requirements

All examples require:
- Python 3.13+
- Pixi package manager
- JaxARC dependencies (installed via `pixi install`)

Additional requirements for specific examples:
- **WandB Integration**: WandB account (optional for offline mode)
- **Advanced Examples**: Understanding of JAX fundamentals