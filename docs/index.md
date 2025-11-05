# JaxARC Documentation

Welcome to JaxARC - A JAX-native Reinforcement Learning environment for the
Abstraction and Reasoning Corpus (ARC).

## What is JaxARC?

JaxARC provides a high-performance, functionally-pure environment for training
AI agents on abstract reasoning puzzles from the ARC challenge. Built entirely
in JAX, it enables researchers and developers to leverage modern hardware
acceleration (GPUs/TPUs) while maintaining clean, composable code.

The ARC challenge tests an AI system's ability to solve novel reasoning tasks by
observing a few examples and inferring the underlying transformation rule.
JaxARC makes it easy to experiment with different agent architectures and
training strategies on this challenging benchmark.

## Why JaxARC?

- **JAX-Native Performance**: Full support for JIT compilation, vectorization
  (`vmap`), and parallelization (`pmap`) means you can scale from single
  environments to thousands of parallel rollouts with minimal code changes.

- **Fast & Efficient**: Leverage GPU/TPU acceleration for environment
  simulation. Run hundreds of thousands of steps per second on modern hardware.

- **Flexible & Composable**: Clean functional API makes it easy to combine with
  other JAX libraries like Optax, Flax, and Haiku for agent training.

- **Reproducible**: Explicit PRNG key management and immutable state ensure your
  experiments are perfectly reproducible across runs and platforms.

## Key Features

- **Multiple Environments**: Support for ARC-AGI-1, ARC-AGI-2, ConceptARC, and
  MiniARC datasets
- **Parser Registry**: Extensible system for loading and parsing different ARC
  dataset formats
- **Wrappers**: Transform observations, rewards, and states with composable
  wrappers
- **Visualization**: Rich terminal and SVG rendering for debugging and analysis
- **Type Safety**: Precise array shape annotations with `jaxtyping` for better
  error messages
- **Comprehensive Testing**: Extensive test suite ensuring JAX compatibility and
  correctness

## Architecture Overview

JaxARC follows a modular, functional design that keeps the core environment fast
(JIT-compatible) while allowing flexible customization:

```{image} _static/images/jaxarc_system_architecture.svg
:alt: JaxARC System Architecture
:class: dark-light
:width: 100%
:align: center
```

**Key Components:**

- **Datasets**: ARC-AGI-1/2, ConceptARC, MiniARC with JSON task definitions
- **Parsers**: Convert JSON → JAX arrays with validation and normalization
- **Task Buffer**: JIT-compatible stacked arrays with fixed shapes
- **Registry**: Manages dataset variants and task subsets
- **Environment**: Pure functional reset/step API following the Stoa standard
- **State**: Immutable dataclass with working grid, input, target, and clipboard
- **Actions**: Multiple representations (mask, point, bbox) with transformations
- **Wrappers**: Modular observation and action space transforms
- **Visualization**: Terminal and SVG rendering for debugging

## Quick Navigation

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting-started/index
:link-type: doc

Install JaxARC and run your first environment in under 10 minutes.
:::

:::{grid-item-card} 📚 Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step guides for common tasks like downloading datasets, creating agents, and using wrappers.
:::

:::{grid-item-card} 📖 API Reference
:link: api/index
:link-type: doc

Complete API documentation for all modules, classes, and functions.
:::

::::

## Installation

```bash
# Using Pixi (recommended)
pixi add jaxarc

# Or using pip
pip install jaxarc
```

## Quick Example

```python
import jax
import jaxarc

# Create an environment
env, env_params = jaxarc.make("Mini-Most_Common_color_l6ab0lf3xztbyxsu3p")

# Reset to get initial state
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key, env_params)

# Take a step
action_space = env.action_space(env_params)
action = action_space.sample(key)
next_state, next_timestep = env.step(state, action)

print(f"Observation shape: {next_timestep.observation.shape}")
print(f"Reward: {next_timestep.reward}")
```

## Next Steps

- **New to JaxARC?** Start with the [Getting Started](getting-started/index.md)
  guide
- **Want to learn specific tasks?** Check out the
  [Tutorials](tutorials/index.md)
- **Need API details?** Browse the [API Reference](api/index.md)

## Contents

```{toctree}
:maxdepth: 2
:caption: Documentation

getting-started/index
tutorials/index
api/index
```

## Community & Support

- **GitHub**: [github.com/aadimator/JaxARC](https://github.com/aadimator/JaxARC)
- **Issues**: Report bugs or request features on
  [GitHub Issues](https://github.com/aadimator/JaxARC/issues)
- **Discussions**: Ask questions on
  [GitHub Discussions](https://github.com/aadimator/JaxARC/discussions)
