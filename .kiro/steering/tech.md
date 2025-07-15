# Technology Stack

## Core Technologies

- **JAX**: Primary framework for high-performance numerical computing with JIT compilation
- **Python 3.13+**: Base language with type hints and modern features (updated from 3.9+)
- **Chex**: JAX testing and validation utilities for dataclasses and array shapes
- **Hydra**: Configuration management system for complex parameter hierarchies
- **Pixi**: Modern Python package and environment manager (replaces conda/pip)

## Key Libraries

- **JAX Ecosystem**: `jax`, `jax.numpy`, `chex` for functional programming and testing
- **Configuration**: `hydra-core`, `omegaconf` for typed configuration management  
- **Visualization**: `rich` (terminal), `drawsvg`, `cairosvg`, `seaborn` for grid rendering
- **Data Processing**: `tqdm`, `loguru` for progress tracking and logging
- **CLI**: `typer` for command-line interfaces
- **Utilities**: `pyprojroot` for project root detection, `kaggle` for dataset access

## Development Tools

- **Code Quality**: `ruff` (linting + formatting), `mypy` (type checking), `pylint`
- **Testing**: `pytest`, `pytest-cov` for unit tests and coverage (targeting 100% coverage)
- **Pre-commit**: Automated code quality checks with hooks for ruff, mypy, prettier
- **Documentation**: `jupyter-book` for documentation generation
- **Notebooks**: `jupyter` for exploration and experimentation

## Build System

- **Package Manager**: Pixi (defined in `pixi.toml`)
- **Build Backend**: Hatchling with `hatch-vcs` for version management
- **Project Config**: `pyproject.toml` following modern Python standards
- **Environments**: Separate pixi environments for dev, test, docs

## Common Commands

```bash
# Environment setup
pixi shell                              # Activate project environment
pixi run -e dev pre-commit install     # Setup pre-commit hooks

# Development
pixi run -e test test                   # Run tests
pixi run lint                           # Run linting (pre-commit)
pixi run pylint                         # Run pylint specifically

# Documentation
pixi run docs-serve                     # Serve documentation locally

# Examples
pixi run python examples/config_api_demo.py      # Basic API demo
pixi run python examples/hydra_integration_example.py  # Hydra config demo
pixi run python examples/visualization_demo.py   # Visualization utilities demo
```

## JAX-Specific Considerations

- **Pure Functions**: All core functions must be pure (no side effects) for JIT compilation
- **PRNG Management**: Use explicit PRNG key management (`jax.random.PRNGKey`, `jax.random.split`)
- **Static Shapes**: Maintain static array shapes using padding and masks for efficient transformations
- **Batch Processing**: Leverage `jax.vmap` for batch processing and `jax.pmap` for multi-device
- **Immutable State**: Use `chex.dataclass` for immutable state structures with `.replace()` for updates
- **Debug Integration**: Use `jax.debug.callback` for logging during JAX transformations
- **Testing**: Use `chex.assert_*` functions for JAX-specific assertions and validation

## RL Integration

- **PureJaxRL Compatible**: Environment designed to integrate with PureJaxRL for agent training
- **Single-Agent Focus**: Current implementation optimized for SARL with extensible architecture
- **Future Extensions**: Architecture supports HRL, Meta-RL, Multi-Task RL, and MARL extensions