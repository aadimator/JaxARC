# JaxARC

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

A JAX-based Single-Agent Reinforcement Learning (SARL) environment for solving
ARC (Abstraction and Reasoning Corpus) tasks. JaxARC provides a
high-performance, functionally-pure environment designed for training AI agents
on abstract reasoning puzzles, with architecture designed to support future
extensions to Hierarchical RL, Meta-RL, and Multi-Task RL.

## 🚀 Key Features

- **🔥 JAX-Native**: Pure functional API with full `jax.jit`, `jax.vmap`, and
  `jax.pmap` support
- **⚡ High Performance**: 100x+ speedup with JIT compilation
- **🔧 Extensible Architecture**: Designed to support future HRL, Meta-RL, and
  Multi-Task RL
- **🧩 Type Safety**: Typed configuration dataclasses with comprehensive
  validation
- **🎨 Rich Visualization**: Terminal and SVG grid rendering utilities
- **📊 Multiple Datasets**: ARC-AGI, ConceptARC, and MiniARC with GitHub-based
  download

## 📦 Installation

### Using pip

```bash
pip install jaxarc
```

### Development Installation

```bash
git clone https://github.com/aadimator/JaxARC.git
cd JaxARC
pixi shell  # Activate project environment
pixi run -e dev pre-commit install  # Set up pre-commit hooks
```

## 📊 Supported Datasets

| Dataset        | Tasks                  | Grid Size   | Use Case               |
| -------------- | ---------------------- | ----------- | ---------------------- |
| **ARC-AGI-2**  | 1000 train + 120 eval  | Up to 30×30 | Full challenge dataset |
| **ConceptARC** | 160 (16 concepts × 10) | Up to 30×30 | Systematic evaluation  |
| **MiniARC**    | 400+                   | 5×5         | Rapid prototyping      |
| **ARC-AGI-1**  | 400 train + 400 eval   | Up to 30×30 | Original 2024 dataset  |

### Quick Download

```bash
# Download your first dataset
python scripts/download_dataset.py miniarc      # Fast experimentation
python scripts/download_dataset.py arc-agi-2   # Full challenge dataset
python scripts/download_dataset.py all         # All datasets
```

All datasets are downloaded directly from GitHub with no authentication
required.

## 🚀 Quick Start

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig

# 1. Download dataset
# python scripts/download_dataset.py miniarc

# 2. Load a task
from jaxarc.envs.config import DatasetConfig

# Preferred: Use typed configuration
dataset_config = DatasetConfig(
    dataset_path="data/raw/MiniARC",
    max_grid_height=5,
    max_grid_width=5,
    max_colors=10,
    background_color=0,
    task_split="train",
)
parser = MiniArcParser(dataset_config)

# Alternative: Use Hydra config with from_hydra method
# parser_config = DictConfig({...})
# parser = MiniArcParser.from_hydra(parser_config)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# 3. Create environment
config = create_standard_config(max_episode_steps=50)
state, observation = arc_reset(key, config, task)

# 4. Take action
action = {
    "selection": jnp.ones((2, 2), dtype=jnp.bool_),
    "operation": jnp.array(1, dtype=jnp.int32),  # Fill with color 1
}
state, obs, reward, done, info = arc_step(state, action, config)
print(f"Reward: {reward:.3f}, Similarity: {info['similarity']:.3f}")
```

**Next Steps**: See the [Getting Started Guide](docs/getting-started.md) for a
complete walkthrough.

## 🎛️ Configuration & Actions

JaxARC uses typed configuration dataclasses with preset options:

```python
from jaxarc.envs import (
    create_standard_config,  # Balanced for training (recommended)
    create_full_config,  # All 35 operations
    create_point_config,  # Point-based actions
)

config = create_standard_config(max_episode_steps=100)
```

**Action Formats**: Selection-based (default), point-based, or bounding box
actions **Operations**: 35 total operations including fill, flood fill,
movement, rotation, and clipboard

See the [Configuration Guide](docs/configuration.md) for complete details.

## 🔧 Development

### Running Tests

```bash
pixi run -e test test
```

### Linting

```bash
pixi run lint
```

### Documentation

```bash
pixi run docs-serve
```

## 🤝 Contributing

We welcome contributions! Please see our
[Contributing Guidelines](CONTRIBUTING.md).

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes following our coding standards
4. Add tests for new functionality
5. Run `pixi run lint` and `pixi run -e test test`
6. Submit a pull request

### Code Style

- Use `black` for code formatting
- Use `ruff` for linting
- Follow JAX best practices (pure functions, immutable state)
- Add type hints for all public APIs
- Write comprehensive docstrings

## 🔗 Related Projects

- **[ARC Challenge](https://github.com/fchollet/ARC)**: Original ARC dataset and
  challenge
- **[ARCLE](https://github.com/ConfeitoHS/arcle)**: ARC Learning Environment
- **[Stoix](https://github.com/EdanToledo/Stoix)**: A research-friendly codebase
  for fast experimentation of single-agent reinforcement learning in JAX

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## 🙏 Acknowledgments

- François Chollet for creating the ARC challenge
- The JAX team for the incredible framework
- The Edan Toledo for Stoix and Stoa implementations
- The Hydra team for configuration management

## 📞 Support

- **GitHub Issues**:
  [Report bugs or request features](https://github.com/aadimator/JaxARC/issues)
- **Discussions**:
  [Ask questions and share ideas](https://github.com/aadimator/JaxARC/discussions)

---

<!-- Links -->

[actions-badge]: https://github.com/aadimator/JaxARC/workflows/CI/badge.svg
[actions-link]: https://github.com/aadimator/JaxARC/actions
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/JaxARC
[conda-link]: https://github.com/conda-forge/JaxARC-feedstock
[github-discussions-badge]:
  https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]: https://github.com/aadimator/JaxARC/discussions
[pypi-link]: https://pypi.org/project/JaxARC/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/JaxARC
[pypi-version]: https://img.shields.io/pypi/v/JaxARC
[rtd-badge]: https://readthedocs.org/projects/JaxARC/badge/?version=latest
[rtd-link]: https://JaxARC.readthedocs.io/en/latest/?badge=latest
