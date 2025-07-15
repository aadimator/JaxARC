# JaxARC Documentation

JaxARC is a JAX-based Single-Agent Reinforcement Learning (SARL) environment for solving ARC (Abstraction and Reasoning Corpus) tasks. It provides a high-performance, functionally-pure environment designed for training AI agents on abstract reasoning puzzles.

## Key Features

- **JAX-Native**: Pure functional API with full JIT compilation support
- **Multiple Datasets**: Support for ARC-AGI, ConceptARC, and MiniARC datasets
- **Type Safety**: Comprehensive type checking with `chex` dataclasses
- **Modular Design**: Composable configuration system with Hydra
- **Rich Visualization**: Terminal and SVG grid rendering utilities

## Quick Start

This project uses [Pixi](https://pixi.js.org) to manage the project structure
and build process, and [Jupyter Book](https://jupyterbook.org) for
documentation. Below are the steps to get started with the project.

```bash
# Clone the repository
git clone git@github.com:aadimator/jaxarc.git
cd jaxarc

# Install Pixi globally
curl -fsSL https://pixi.sh/install.sh | sh

# Install project dependencies
pixi install
```

## Useful Pixi Commands

Here are some useful Pixi commands to manage the project:

```bash
# Add dependencies
pixi add <package_name>

# Remove dependencies
pixi remove <package_name>

# Run linting
pixi run lint

# Serve the documentation
pixi run docs-serve

# Run tests
pixi run test
```

## Documentation

### Core Documentation

- **[Parser Usage Guide](parser_usage.md)**: Comprehensive guide for using ARC dataset parsers
- **[Data Format Documentation](data_format.md)**: Detailed information about supported dataset formats
- **[Configuration API Guide](CONFIG_API_README.md)**: Complete configuration system documentation

### Dataset Support

JaxARC supports multiple ARC dataset variants:

- **ARC-AGI-1/2**: Original Kaggle competition datasets
- **ConceptARC**: 16 concept groups for systematic evaluation
- **MiniARC**: Compact 5x5 grid version for rapid prototyping

### Examples and Demos

- **Basic Usage**: `examples/config_api_demo.py`
- **ConceptARC Demo**: `examples/concept_arc_demo.py`
- **Hydra Integration**: `examples/hydra_integration_example.py`
- **Visualization**: `examples/visualization_demo.py`

### Architecture

- **[Project Architecture](../planning-docs/PROJECT_ARCHITECTURE.md)**: Technical architecture overview
- **[Implementation Guide](../planning-docs/guides/technical_implementation_guide.md)**: Detailed implementation guide

For detailed documentation on how the project was set up from scratch, refer to
the [setup guide](./setup.md). For more information on how to use Pixi, refer to the
[Pixi Documentation](https://pixi.js.org/docs/).
