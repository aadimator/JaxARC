---
applyTo: "**"
---

# Project Structure

## Source Code Organization

```
src/jaxarc/                    # Main package
├── __init__.py               # Package exports (JaxArcConfig, State, Action, EnvParams, TimeStep, __version__)
├── types.py                  # Core data structures (Grid, JaxArcTask, Action, EnvParams, TimeStep)
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
│   ├── actions.py           # Action handlers (Action-based)
│   ├── grid_operations.py   # Grid transformation operations
│   ├── grid_initialization.py # Grid initialization utilities
│   ├── spaces.py            # Action and observation spaces
│   ├── wrappers.py          # Action and observation wrappers
│   └── observation_wrappers.py # Composable observation wrappers
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
│   ├── core.py              # Core utilities
│   ├── dataset_manager.py   # Dataset management utilities
│   ├── grid_utils.py        # Grid manipulation utilities
│   ├── serialization_utils.py # Serialization utilities
│   ├── state_utils.py       # State management utilities
│   ├── task_manager.py      # Task management utilities
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
- **`test/`**: Comprehensive test suite targeting 100% coverage
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

- **Pure Functions**: Core environment operations (`reset`, `step`) are pure
  functions
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

- **Action-Based Core**: All actions are ultimately processed as Action objects
- **Action Wrappers**: PointActionWrapper and BboxActionWrapper convert other formats to actions
- **Clean Separation**: Core environment only knows about actions, wrappers handle format conversion
- **Grid Operations**: Comprehensive set of grid transformation operations
- **Extensible Design**: New action formats can be added as wrappers without changing core logic

### Observation System Design

- **Composable Wrappers**: The environment uses a wrapper-based system for building complex, multi-channel observations.
- **Base Observation**: The default environment provides a single-channel (H, W, 1) observation representing the agent's working grid.
- **Stackable Channels**: Wrappers like `InputGridObservationWrapper`, `AnswerObservationWrapper`, `ClipboardObservationWrapper`, and `ContextualObservationWrapper` can be stacked to add new channels to the observation tensor.
- **Flexibility**: This design allows for easy experimentation with different observation formats without changing the core environment logic.

## File Naming Conventions

- **Tests**: `test_*.py` following pytest conventions with comprehensive
  coverage
- **Configs**: `*.yaml` files in hierarchical structure under `conf/`
- **Examples**: Descriptive names like `registry_bootstrap_demo.py`,
  `action_wrappers_example.py`
- **Utilities**: Grouped by functionality (`visualization/`, `grid_utils.py`)

## Import Patterns

```python
# Core configuration and types
from jaxarc import JaxArcConfig
from jaxarc.types import Grid, JaxArcTask, EnvParams, TimeStep

# Environment creation using registration system
from jaxarc.registration import make, available_task_ids

# Environment classes and functional API
from jaxarc.envs import (
    Environment,
    reset,
    step,
    AnswerObservationWrapper,
    ClipboardObservationWrapper,
    ContextualObservationWrapper,
    InputGridObservationWrapper,
)

# Action creation utilities
from jaxarc.envs import Action, create_action

# Parsers for data loading
from jaxarc.parsers import ArcAgiParser

# Utilities
from jaxarc.utils.visualization import log_grid_to_console, draw_grid_svg
```

## Data Flow Architecture

1. **Task Loading**: `parsers/` → `JaxArcTask` data structures with static
   shapes and JAXTyping annotations
2. **Environment Setup**: `configs/` → `JaxArcConfig` unified configuration with
   Equinox validation
3. **Environment Execution**: Pure functions in `envs/functional.py` with
   immutable `State`
4. **Action Processing**: Action objects processed through grid operations, with wrappers for other formats
5. **State Management**: Centralized `State` with PyTree utilities for updates
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
