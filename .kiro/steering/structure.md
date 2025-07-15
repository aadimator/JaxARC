# Project Structure

## Source Code Organization

```
src/jaxarc/                    # Main package
├── __init__.py               # Package exports (ArcEnvironment, __version__)
├── types.py                  # Core data structures (Grid, JaxArcTask, ARCLEAction)
├── envs/                     # Environment implementations
│   ├── __init__.py          # Environment exports and factory functions
│   ├── environment.py       # Main ArcEnvironment class
│   ├── functional.py        # Pure functional API (arc_reset, arc_step)
│   ├── config.py           # Configuration dataclasses
│   ├── factory.py          # Configuration factory functions
│   ├── actions.py          # Action handlers (point, bbox, mask)
│   ├── grid_operations.py  # Grid transformation operations
│   └── arc_base.py         # Base SARL environment implementation
├── parsers/                 # Task data parsers (ARC dataset loading)
├── utils/                   # Utility functions
│   ├── visualization.py    # Grid rendering (terminal, SVG)
│   ├── grid_utils.py      # Grid manipulation utilities
│   └── config.py          # Configuration utilities
└── py.typed                # Type checking marker
```

## Configuration System

```
conf/                         # Hydra configuration hierarchy
├── config.yaml              # Main configuration with defaults
├── action/                   # Action format configurations
│   ├── standard.yaml        # Default mask-based actions
│   ├── point.yaml          # Point-based actions
│   ├── bbox.yaml           # Bounding box actions
│   ├── full.yaml           # All operations enabled
│   └── raw.yaml            # Minimal operations
├── dataset/                  # Dataset-specific configurations
│   ├── arc_agi_1.yaml      # ARC-AGI 2024 dataset
│   ├── arc_agi_2.yaml      # ARC-AGI 2025 dataset
│   └── mini_arc.yaml       # Smaller dataset for testing
├── environment/              # Environment behavior configurations
├── reward/                   # Reward function configurations
│   ├── standard.yaml       # Balanced reward structure
│   ├── training.yaml       # Training-optimized rewards
│   └── evaluation.yaml     # Evaluation-focused rewards
└── debug/                    # Debug mode configurations
```

## Key Directories

- **`data/`**: Raw and processed ARC datasets (arc-prize-2024, arc-prize-2025)
- **`tests/`**: Comprehensive test suite targeting 100% coverage
- **`examples/`**: Usage examples and demos (config API, Hydra integration, visualization)
- **`notebooks/`**: Jupyter notebooks for exploration and experimentation
- **`docs/`**: Documentation and guides
- **`planning-docs/`**: Architecture and design documents (PROJECT_ARCHITECTURE.md)
- **`scripts/`**: Utility scripts (dataset download, demo parser)

## Architecture Patterns

### Single-Agent RL Focus
- **SARL Environment**: Clean single-agent implementation optimized for learning
- **Extensible Design**: Architecture supports future HRL, Meta-RL, Multi-Task RL extensions
- **PureJaxRL Integration**: Designed to work with PureJaxRL for agent training

### Functional Core Design
- **Pure Functions**: Core environment operations (`arc_reset`, `arc_step`) are pure functions
- **Immutable State**: All state updates return new state objects using `chex.dataclass`
- **Explicit Dependencies**: Configuration and PRNG keys passed explicitly
- **JAX Compatibility**: Full support for `jax.jit`, `jax.vmap`, `jax.pmap`

### Configuration-Driven Architecture
- **Typed Configs**: All configuration uses `@chex.dataclass` with validation
- **Hierarchical Composition**: Hydra manages complex configuration hierarchies
- **Factory Functions**: `create_*_config()` functions for common configurations
- **Preset System**: Predefined configurations for different use cases

### Action System Design
- **Handler Pattern**: Different action formats (point, bbox, mask) use dedicated handlers
- **Validation Pipeline**: Actions validated and transformed through consistent pipeline
- **ARCLE Operations**: 35 operations (fill, flood fill, movement, rotation, clipboard, etc.)

## File Naming Conventions

- **Tests**: `test_*.py` following pytest conventions with comprehensive coverage
- **Configs**: `*.yaml` files in hierarchical structure under `conf/`
- **Examples**: Descriptive names like `config_api_demo.py`, `visualization_demo.py`
- **Utilities**: Grouped by functionality (`visualization.py`, `grid_utils.py`)

## Import Patterns

```python
# Core environment and configuration
from jaxarc.envs import ArcEnvironment, create_standard_config, arc_reset, arc_step
from jaxarc.types import Grid, JaxArcTask, ArcEnvState, ARCLEAction

# Parsers for data loading
from jaxarc.parsers import ArcAgiParser

# Utilities
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg
from jaxarc.utils.config import get_config
```

## Data Flow Architecture

1. **Task Loading**: `parsers/` → `JaxArcTask` data structures with static shapes
2. **Environment Setup**: `config/` → typed configuration objects via factory functions
3. **Environment Execution**: Pure functions in `envs/functional.py` with immutable state
4. **Action Processing**: Action handlers transform inputs to grid operations (35 ARCLE ops)
5. **Visualization**: `utils/visualization.py` for debugging with JAX debug callbacks
6. **Testing**: Comprehensive test suite with `chex` assertions for JAX compatibility

## Development Workflow

- **TDD Approach**: Test-driven development with 100% coverage goal
- **Planning First**: New features require planning documents in `planning-docs/`
- **JAX Validation**: All functions tested for JIT compilation and batch processing
- **Configuration Management**: Use Hydra for complex parameter hierarchies
- **Visualization**: Rich terminal and SVG rendering for debugging grid transformations