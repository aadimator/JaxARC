# Product Overview

JaxARC is a JAX-based Single-Agent Reinforcement Learning (SARL) environment for
solving ARC (Abstraction and Reasoning Corpus) tasks. It provides a
high-performance, functionally-pure environment designed for training AI agents
on abstract reasoning puzzles, with architecture designed to support future
extensions to Hierarchical RL (HRL), Meta-RL (MTRL), Multi-Task RL, and
potentially Multi-Agent RL (MARL).

## Key Features

- **JAX-Native**: Pure functional API with full `jax.jit`, `jax.vmap`, and
  `jax.pmap` support for 100x+ speedup
- **Single-Agent Focus**: Clean SARL implementation optimized for learning and
  iteration
- **Extensible Architecture**: Designed to support future HRL, Meta-RL, and
  Multi-Task RL extensions
- **Type Safety**: Typed configuration using `equinox.Module` with comprehensive
  validation and JAXTyping annotations
- **Modular Design**: Composable configuration components with Hydra integration
- **Rich Visualization**: Terminal and SVG grid rendering utilities with JAX
  debug callbacks
- **Mask-Based Actions**: Comprehensive mask-based action system for grid operations
- **Functional API**: Pure functional environment operations for maximum JAX
  compatibility

## Core Purpose

The project provides a robust, JAX-optimized environment for training single
agents on ARC tasks, focusing on pattern recognition and symbolic reasoning. The
architecture prioritizes performance and extensibility, allowing researchers to
experiment with various RL paradigms while maintaining JAX compatibility and
speed.

## Target Users

- Researchers working on abstract reasoning and symbolic AI
- Single-agent reinforcement learning practitioners
- JAX/functional programming enthusiasts
- ARC challenge participants and researchers
- Developers interested in HRL, Meta-RL, and Multi-Task RL applications
