# JaxARC

<!-- [![Actions Status][actions-badge]][actions-link] -->

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/jaxarc?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/jaxarc)
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

JaxARC is a JAX-based reinforcement learning environment for the
[Abstraction and Reasoning Corpus](https://github.com/fchollet/ARC) (ARC)
challenge. It's built for researchers who want to use extremely fast vectorized
environments to explore reinforcement learning, and meta-learning techniques for
abstract reasoning.

## Why JaxARC?

**Speed.** Environments compile with `jax.jit` and vectorize with `jax.vmap`.
Run thousands of episodes in parallel on GPU/TPU.

![JaxARC throughput as compared with ARCLE](docs/_static/images/sps_vs_envs_linear_h100.png)

**Flexible.** Multiple action spaces (point-based, selection masks, bounding
boxes). Multiple datasets (ARC-AGI, ConceptARC, MiniARC). Observation wrappers
for different input formats. Configure everything via typed dataclasses or YAML.

**Extensible.** Clean parser interface for custom datasets. Wrapper system for
custom observations and actions. Built with future HRL and Meta-RL experiments
in mind.

## Key Features

- **JAX-Native**: Pure functional API — every function is `jax.jit`-compatible
- **Lightning Fast**: JIT compilation turns Python into XLA-optimized machine
  code
- **Configurable**: Multiple action spaces, reward functions, and observation
  formats
- **Multiple Datasets**: ARC-AGI-1, ARC-AGI-2, ConceptARC, and MiniARC included
- **Type-Safe**: Full type hints with runtime validation
- **Visual Debug**: Terminal and SVG rendering for development

![JaxARC System Architecture](docs/_static/images/jaxarc_system_architecture.svg)

## Installation

```bash
pip install jaxarc
```

### Want to contribute?

```bash
git clone https://github.com/aadimator/JaxARC.git
cd JaxARC
pixi shell  # Sets up the environment
pixi run -e dev pre-commit install  # Hooks for code quality
```

**See the [tutorials](https://jaxarc.readthedocs.io/en/latest/tutorials/)** for
training loops, custom wrappers, and dataset management.

## Stoix Integration

JaxARC uses the [Stoa API](https://github.com/EdanToledo/Stoa), allowing
seamless integration with [Stoix](https://github.com/EdanToledo/Stoix), which is
a JAX-based reinforcement learning codebase supporting various RL algorithms.

This means you can easily plug JaxARC environments into Stoix's training
pipelines to leverage its efficient implementations of RL algorithms.

You can explore
[jaxarc-baselines](https://github.com/aadimator/jaxarc-baselines) repository for
example implementations of training agents on JaxARC environments using Stoix.

## Contributing

Found a bug? Want a feature?
**[Open an issue](https://github.com/aadimator/JaxARC/issues)** or submit a PR.

## Related Work

JaxARC builds on great work from the community:

- **[ARC Challenge](https://github.com/fchollet/ARC)** by François Chollet — The
  original dataset and challenge
- **[ARCLE](https://github.com/ConfeitoHS/arcle)** — Python-based ARC
  environment (inspiration for our design)
- **[Stoix](https://github.com/EdanToledo/Stoix)** by Edan Toledo — Single-agent
  RL in JAX (we use their Stoa API)

## Citation

If you use JaxARC in your research:

```bibtex
@software{jaxarc2025,
  author = {Aadam},
  title = {JaxARC: JAX-based Reinforcement Learning for Abstract Reasoning},
  year = {2025},
  url = {https://github.com/aadimator/JaxARC}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Questions?

- **Bugs/Features**: [GitHub Issues](https://github.com/aadimator/JaxARC/issues)
- **Discussions**:
  [GitHub Discussions](https://github.com/aadimator/JaxARC/discussions)
- **Docs**: [jaxarc.readthedocs.io](https://jaxarc.readthedocs.io)

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
