---
applyTo: "**"
---

# Project Context

JaxARC is a JAX-based implementation of the Abstraction and Reasoning Corpus
(ARC) challenge using Multi-Agent Reinforcement Learning (MARL). The project
focuses on creating collaborative AI agents that work together to solve ARC
tasks through a structured hypothesis-proposal-consensus mechanism.

# Key features:

- JAX-native environment compatible with JaxMARL framework
- Grid-based task representation with visualization utilities
- Collaborative agent architecture with explicit reasoning and consensus
  building
- Modular design with abstract base classes and concrete implementations

# Architecture

- Core Data Structures (`types.py`): JAX Pytrees using `chex.dataclass` for
  static typing
- Base Classes (`base`): Abstract classes defining interfaces for parsers and
  environments
- Environments (`envs`): Concrete implementations of the ARC environment
- Parsers (`parsers`): Dataset-specific parsers (ARC-AGI-1)
- Utilities (`utils`): Visualization tools, data handling, and logging

# Technical Guidelines

- **JAX Compatibility**:
  - Use pure functions without side effects
  - Use immutable state updates (e.g., `state.replace(...)`)
  - Manage PRNG keys explicitly with `key, subkey = jax.random.split(key)`
  - Maintain static shapes for all arrays using padding and masks
  - Use `jnp` instead of `np` for array operations within JAX-transformed
    functions
- **Visualization**:
  - Use `visualization.py` utilities for rendering grids in terminal
    (`log_grid_to_console`) or SVG (`draw_grid_svg`)
  - Integrate with JAX using `jax.debug.callback` for logging during execution
- **Testing**:
  - Use `chex.assert_*` functions for JAX-specific assertions
  - Test reproducibility with fixed PRNG keys
  - Verify Pytree structure, shapes, and types
- **General**:
  - Use `loguru` for logging instead of default Python logging
  - Use `Hydra` for configuration management
  - Use `typer` for command-line interfaces
  - Follow the project's pre-allocation patterns for JAX compatibility
  - Try to have 100% test coverage for all new code.
  - Go step by step, and ensure each step is working before moving on to the
    next.

# Development Workflow

- Make changes in modular components
- Write comprehensive tests using `chex` assertions
- Run linting and tests before committing
- Use visualization tools to debug and understand grid transformations
- Follow a test-driven development (TDD) approach to ensure correctness

# Useful Pixi Commands

- `pixi shell`: Activate the project environment
- `pixi run lint`: Run linting to check code style and quality
- `pixi run -e test test`: Run tests to ensure code correctness
- `pixi run python <script.py>`: Run a Python script within the project
  environment
- `pixi run -e dev pre-commit run --all-files`: Run pre-commit hooks
- `pixi run docs-serve`: Serve the documentation locally
- `pixi add <package_name>`: Add dependencies
- `pixi remove <package_name>`: Remove dependencies

# Common Issues

- Ensure arrays have static shapes for JAX compatibility
- Be careful with PRNG key management (split keys for randomness)
- Check that grids are properly padded with appropriate masks
- Verify JAX transformations work with your functions (test with `jit`, `vmap`)
