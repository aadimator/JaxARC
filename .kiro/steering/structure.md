# Project Structure

## Source Code Organization

```
src/jaxarc/                    # Main package
├── __init__.py               # Package exports (JaxArcConfig, State, ARCAction, EnvParams, TimeStep, __version__)
├── types.py                  # Core data structures (Grid, JaxArcTask, ARCAction, EnvParams, TimeStep)
├── state.py                  # Centralized State definition using Equinox
├── registration.py           # Environment registration utilities
├── configs/                  # Configuration system (Equinox-based)
│   ├── __init__.py          # Configuration exports
│   ├── main_config.py       # JaxArcConfig (unified configuration)
│   ├── action_config.py     # Action format configurations
│   ├── dataset_config.py    # Dataset-specific configurations
│   ├── environment_config.py # Environment behavior configurations
│   ├── reward_config.py     # Reward function configurations
│   ├── grid_initialization_config.py # Grid initialization configurations
│   ├── logging_config.py    # Logging configurations
│   ├── storage_config.py    # Storage configurations
│   ├── visualization_config.py # Visualization configurations
│   ├── wandb_config.py      # WandB experiment tracking
│   └── validation.py        # Configuration validation utilities
├── conf/                     # Hydra configuration hierarchy
│   ├── config.yaml          # Main configuration with defaults
│   ├── action/              # Action format configurations
│   ├── dataset/             # Dataset-specific configurations
│   └── reward/              # Reward function configurations
├── envs/                     # Environment implementations
│   ├── __init__.py          # Environment exports and functional API
│   ├── functional.py        # Pure functional API (reset, step)
│   ├── environment.py       # Simple environment interface
│   ├── actions.py           # Action handlers (mask-based)
│   ├── action_space_controller.py # Action space management
│   ├── action_wrappers.py   # Action wrapper utilities
│   ├── grid_operations.py   # Grid transformation operations
│   ├── grid_initialization.py # Grid initialization utilities
│   ├── observation.py       # Observation space handling
│   ├── reward.py            # Reward computation
│   ├── spaces.py            # Action and observation spaces
│   └── wrapper.py           # Environment wrappers
├── parsers/                  # Task data parsers (ARC dataset loading)
│   ├── __init__.py          # Parser exports
│   ├── base_parser.py       # Base parser interface
│   ├── arc_agi.py           # ARC-AGI dataset parser
│   ├── concept_arc.py       # ConceptARC dataset parser
│   ├── mini_arc.py          # Mini-ARC dataset parser
│   └── utils.py             # Parser utilities
├── utils/                    # Utility functions
│   ├── __init__.py          # Utility exports
│   ├── buffer.py            # Buffer utilities
│   ├── config.py            # Configuration utilities
│   ├── dataset_downloader.py # Dataset downloading utilities
│   ├── dataset_validation.py # Dataset validation utilities
│   ├── grid_utils.py        # Grid manipulation utilities
│   ├── jax_types.py         # JAXTyping type definitions
│   ├── pytree.py            # PyTree manipulation utilities
│   ├── serialization_utils.py # Serialization utilities
│   ├── state_utils.py       # State management utilities
│   ├── task_manager.py      # Task management utilities
│   ├── validation.py        # Validation utilities
│   ├── logging/             # Logging utilities
│   └── visualization/       # Visualization utilities
└── py.typed                  # Type checking marker
```

## Configuration System

```
src/jaxarc/conf/              # Hydra configuration hierarchy
├── config.yaml              # Main configuration with defaults
├── action/                   # Action format configurations
│   ├── standard.yaml        # Default mask-based actions
│   ├── full.yaml           # All operations enabled
│   └── raw.yaml            # Minimal operations
├── dataset/                  # Dataset-specific configurations
│   ├── arc_agi_1.yaml      # ARC-AGI 2024 dataset
│   ├── arc_agi_2.yaml      # ARC-AGI 2025 dataset
│   ├── concept_arc.yaml    # ConceptARC dataset
│   └── mini_arc.yaml       # Mini-ARC dataset for testing
└── reward/                   # Reward function configurations
    ├── training.yaml        # Training-optimized rewards
    └── evaluation.yaml      # Evaluation-focused rewards
```

## Key Directories

- **`data/`**: Raw and processed ARC datasets (arc-prize-2024, arc-prize-2025)
- **`tests/`**: Comprehensive test suite targeting 100% coverage
- **`examples/`**: Usage examples and demos (config API, Hydra integration,
  visualization)
- **`notebooks/`**: Jupyter notebooks for exploration and experimentation
- **`docs/`**: Documentation and guides
- **`planning-docs/`**: Architecture and design documents
  (PROJECT_ARCHITECTURE.md)
- **`scripts/`**: Utility scripts (dataset download, demo parser)

## Architecture Patterns

### Single-Agent RL Focus

- **SARL Environment**: Clean single-agent implementation optimized for learning
- **Extensible Design**: Architecture supports future HRL, Meta-RL, Multi-Task
  RL extensions
- **PureJaxRL Integration**: Designed to work with PureJaxRL for agent training

### Functional Core Design

- **Pure Functions**: Core environment operations (`reset`, `step`) are
  pure functions
- **Immutable State**: All state updates return new state objects using
  `equinox.Module`
- **Explicit Dependencies**: Configuration and PRNG keys passed explicitly
- **JAX Compatibility**: Full support for `jax.jit`, `jax.vmap`, `jax.pmap`

### Configuration-Driven Architecture

- **Typed Configs**: All configuration uses `equinox.Module` with validation
- **Hierarchical Composition**: Hydra manages complex configuration hierarchies
- **Unified Configuration**: `JaxArcConfig` provides single entry point for all
  settings
- **Cross-Validation**: Automatic validation of configuration consistency

### Action System Design

- **Mask-Based Actions**: Primary focus on mask-based action format
- **Validation Pipeline**: Actions validated and transformed through consistent
  pipeline
- **Grid Operations**: Comprehensive set of grid transformation operations
- **Action Space Controller**: Dynamic action space management and filtering

## File Naming Conventions

- **Tests**: `test_*.py` following pytest conventions with comprehensive
  coverage
- **Configs**: `*.yaml` files in hierarchical structure under `conf/`
- **Examples**: Descriptive names like `config_api_demo.py`,
  `visualization_demo.py`
- **Utilities**: Grouped by functionality (`visualization.py`, `grid_utils.py`)

## Import Patterns

```python
# Core configuration and types
from jaxarc import JaxArcConfig
from jaxarc.types import Grid, JaxArcTask, ARCAction, EnvParams, TimeStep

# Environment creation using registration system
from jaxarc.registration import make, available_task_ids

# Environment classes and functional API
from jaxarc.envs import Environment, reset, step

# Action creation utilities
from jaxarc.envs import create_mask_action

# Parsers for data loading
from jaxarc.parsers import ArcAgiParser

# Utilities
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg
from jaxarc.utils.config import get_config
```

## Data Flow Architecture

1. **Task Loading**: `parsers/` → `JaxArcTask` data structures with static
   shapes and JAXTyping annotations
2. **Environment Setup**: `configs/` → `JaxArcConfig` unified configuration with
   Equinox validation
3. **Environment Execution**: Pure functions in `envs/functional.py` with
   immutable `State`
4. **Action Processing**: Action handlers transform inputs to grid operations
5. **State Management**: Centralized `State` with PyTree utilities for
   updates
6. **Visualization**: `utils/visualization/` for debugging with JAX debug
   callbacks
7. **Testing**: Comprehensive test suite with `chex` assertions for JAX
   compatibility

## Development Workflow

- **TDD Approach**: Test-driven development with 100% coverage goal
- **Planning First**: New features require planning documents in
  `planning-docs/`
- **JAX Validation**: All functions tested for JIT compilation and batch
  processing
- **Configuration Management**: Use Hydra for complex parameter hierarchies
- **Visualization**: Rich terminal and SVG rendering for debugging grid
  transformations
