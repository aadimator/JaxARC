# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

JaxARC is a JAX-based Single-Agent Reinforcement Learning (SARL) environment for solving Abstraction and Reasoning Corpus (ARC) tasks. It provides a high-performance, functionally-pure environment for training AI agents on abstract reasoning puzzles with architecture designed for future extensions to Hierarchical RL, Meta-RL, and Multi-Task RL.

## Core Development Commands

### Environment Setup
```bash
# Clone and setup development environment
git clone https://github.com/aadimator/JaxARC.git
cd JaxARC
pixi shell                                    # Activate project environment  
pixi run -e dev pre-commit install          # Set up pre-commit hooks
```

### Testing
```bash
pixi run -e test test                        # Run full test suite
pixi run python test_my_feature.py          # Run temporary test scripts (delete after use)
```

### Code Quality
```bash
pixi run lint                                # Run linter (ruff + black)
pixi run -e dev pylint                       # Run pylint
```

### Dataset Management
```bash
# Download datasets (stored in data/raw/)
python scripts/download_dataset.py miniarc      # Fast experimentation (5x5 grids)
python scripts/download_dataset.py arc-agi-2   # Full challenge dataset (up to 30x30)
python scripts/download_dataset.py conceptarc  # Systematic evaluation dataset
python scripts/download_dataset.py all         # All datasets
```

### Documentation
```bash
pixi run docs-serve                          # Build and serve documentation locally
```

### Examples
```bash
# Basic environment and configuration demos
python examples/config_api_demo.py
python examples/hydra_integration_example.py

# Dataset-specific examples
python examples/miniarc_usage_example.py --performance-comparison
python examples/conceptarc_usage_example.py --concept Center --visualize
python examples/visualization_demo.py
```

## Architecture Overview

### Core Design Principles

**JAX-Native Implementation**
- All core functionality implemented as pure functions
- Full compatibility with `jax.jit`, `jax.vmap`, and `jax.pmap`
- Immutable state management using `equinox.Module`
- Static array shapes with padding and masks for JIT compilation
- Explicit PRNG key management with `jax.random.split`

**Action System Architecture**
- **Core Format**: All actions are ultimately `MaskAction` objects with operation ID and boolean selection mask
- **Action Wrappers**: `PointActionWrapper` and `BboxActionWrapper` convert other formats to masks
- **Clean Separation**: Core environment only processes masks; wrappers handle format conversion
- **Extensible Design**: New action formats can be added as wrappers without changing core logic

**Configuration System**
- Modular configuration using typed `equinox.Module` dataclasses
- Hydra integration for complex parameter hierarchies
- Main config: `JaxArcConfig` with nested configs for actions, rewards, datasets, etc.
- Configuration validation with comprehensive type safety

### Key Architectural Components

**Functional API (`src/jaxarc/envs/functional.py`)**
- Primary interface: `reset(params: EnvParams, key: PRNGKey) -> TimeStep`
- Step function: `step(params: EnvParams, timestep: TimeStep, action) -> TimeStep`
- Pure functional design decoupled from framework configurations

**Environment State (`src/jaxarc/state.py`)**
- Central `State` class containing all environment state
- Immutable state updates through PyTree operations
- Includes working grids, masks, clipboard, similarity scores, step counts

**Dataset Parsers (`src/jaxarc/parsers/`)**
- `ArcAgiParser`: ARC-AGI-1 and ARC-AGI-2 datasets
- `ConceptArcParser`: ConceptARC systematic evaluation dataset  
- `MiniArcParser`: MiniARC rapid prototyping dataset
- Common base class `ArcDataParserBase` for consistent functionality

**Grid Operations (`src/jaxarc/envs/grid_operations.py`)**
- 35 total operations including fill, flood fill, movement, rotation, clipboard
- Pure functional operations with similarity computation
- Operation categories: Basic, Advanced, Movement, Clipboard, Special

**Visualization System (`src/jaxarc/utils/visualization/`)**
- Terminal rendering with Rich for debugging
- SVG export for documentation and analysis
- Episode visualization for training analysis
- Task visualization for dataset exploration

### Package Structure

```
src/jaxarc/
├── __init__.py              # Main API exports
├── configs/                 # Modular configuration system
├── envs/                    # Core RL environment
│   ├── actions.py           # Action system and validation
│   ├── functional.py        # Pure functional API
│   ├── grid_operations.py   # 35 ARC operations
│   └── spaces.py           # Action/observation spaces
├── parsers/                 # Dataset loading
├── utils/                   # Utilities and visualization
├── state.py                 # Central state management
└── types.py                 # Core type definitions
```

## Development Guidelines

### Code Evolution Philosophy
- **No Backwards Compatibility**: Replace old implementations when introducing new features
- **Single Source of Truth**: Maintain one correct way to perform each task
- **Clean Migration**: Update all usage sites and remove old code when replacing functionality

### JAX Best Practices
- Use pure functions for all core functionality
- Maintain static shapes using padding and masks
- Manage PRNG keys explicitly
- Avoid Python control flow inside JIT-compiled functions
- Use `equinox.Module` for state management

### Testing Strategy
- Write temporary test scripts in root directory for initial validation
- Run with `pixi run python test_my_feature.py`
- Delete temporary test files after validation
- Add permanent tests to `tests/` directory (currently empty)

### Code Style
- Use `black` for formatting and `ruff` for linting
- Add type hints for all public APIs
- Write comprehensive docstrings
- Follow JAX functional programming patterns
- Use `loguru` for logging, `typer` for CLIs, `pyprojroot.here()` for paths

## Key Configuration Patterns

### Standard Configurations
```python
from jaxarc.envs import (
    create_standard_config,  # Balanced for training (recommended)
    create_full_config,      # All 35 operations
    create_point_config,     # Point-based actions
)

config = create_standard_config(max_episode_steps=100)
```

### Dataset Configuration
```python
from jaxarc.configs import DatasetConfig

dataset_config = DatasetConfig(
    dataset_path="data/raw/MiniARC",
    max_grid_height=5,
    max_grid_width=5,
    max_colors=10,
    background_color=0,
    task_split="train",
)
```

## Performance Characteristics

JaxARC is optimized for high-performance training:
- **JIT Compilation**: 100x+ speedup for environment steps
- **Vectorization**: Batch processing with `vmap`  
- **Memory Efficient**: Static shapes and pre-allocated arrays
- **Scalable**: Supports large-scale distributed training

Typical performance (per operation):
- Environment Reset: 0.1ms, 2.5MB
- Single Step: 0.05ms, 1.2MB
- JIT Compiled Step: 0.0005ms, 1.2MB
- Batch (1000 envs): 0.5ms, 120MB

## Special Considerations

### Action System
The core environment only understands `MaskAction` objects. Other action formats (point, bounding box) are converted to masks by wrapper classes. This design keeps the core simple while supporting multiple interaction modalities.

### Grid Initialization
Supports multiple initialization strategies including demo-based, permutation, empty, and random grids for diverse training experiences.

### Reward System  
Submit-aware reward calculation with similarity improvement shaping, success bonuses, efficiency bonuses, and submission penalties. Supports both training and evaluation modes.

### Dataset Management
All datasets are downloaded directly from GitHub with no authentication required. Datasets are stored in `data/raw/` and managed through the download script.
