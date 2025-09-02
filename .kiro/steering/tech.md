# Technology Stack

## Core Technologies

- **JAX**: Primary framework for high-performance numerical computing with JIT
  compilation
- **Python 3.13+**: Base language with type hints and modern features (>=3.9 supported)
- **Equinox**: Primary framework for JAX-compatible modules and PyTree
  registration (preferred over chex.dataclass)
- **JAXTyping**: Precise type annotations for JAX arrays with shape information
- **Hydra**: Configuration management system for complex parameter hierarchies
- **Pixi**: Modern Python package and environment manager (replaces conda/pip)

## Key Libraries

- **JAX Ecosystem**: `jax`, `jax.numpy`, `equinox`, `jaxtyping` for functional
  programming and type safety
- **Validation**: `chex` for runtime assertions and array validation
- **Configuration**: `hydra-core`, `omegaconf` for typed configuration
  management
- **Visualization**: `rich` (terminal), `drawsvg`, `cairosvg`, `seaborn` for
  grid rendering
- **Data Processing**: `tqdm`, `loguru` for progress tracking and logging
- **CLI**: `typer` for command-line interfaces
- **Utilities**: `pyprojroot` for project root detection

## Development Tools

- **Code Quality**: `ruff` (linting + formatting), `mypy` (type checking),
  `pylint`
- **Testing**: `pytest`, `pytest-cov`, `hypothesis` for unit tests and coverage
- **Pre-commit**: Automated code quality checks with hooks for ruff, mypy
- **Documentation**: `jupyter-book` for documentation generation
- **Notebooks**: `jupyter`, `marimo` for exploration and experimentation
- **Experiment Tracking**: `wandb` for experiment logging and visualization

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

# Examples - use pixi run python to select the relevant environment
pixi run python examples/basic_usage.py           # Basic environment usage
pixi run python examples/functional_api.py       # Functional API demo
pixi run python examples/configuration_demo.py   # Configuration system demo
```

## JAX-Specific Considerations

- **Pure Functions**: All core functions must be pure (no side effects) for JIT
  compilation
- **PRNG Management**: Use explicit PRNG key management (`jax.random.PRNGKey`,
  `jax.random.split`)
- **Static Shapes**: Maintain static array shapes using padding and masks for
  efficient transformations
- **Batch Processing**: Leverage `jax.vmap` for batch processing and `jax.pmap`
  for multi-device
- **Immutable State**: Use `equinox.Module` for immutable state structures with
  PyTree registration
- **Type Safety**: Use `jaxtyping` annotations for precise array shape
  documentation
- **Debug Integration**: Use `jax.debug.callback` for logging during JAX
  transformations
- **Testing**: Use `chex.assert_*` functions for JAX-specific assertions and
  validation

## RL Integration

- **Single-Agent Focus**: Current implementation optimized for SARL with
  extensible architecture
- **Functional API**: Pure functional environment operations (`reset`,
  `step`) for maximum JAX compatibility
- **Future Extensions**: Architecture supports HRL, Meta-RL, Multi-Task RL, and
  MARL extensions
