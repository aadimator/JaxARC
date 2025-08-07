# AGENTS.md: Instructions for AI Agents

This document provides guidelines and instructions for AI agents working on the JaxARC repository.

## 1. Project Overview

JaxARC is a JAX-based Single-Agent Reinforcement Learning (SARL) environment for solving Abstraction and Reasoning Corpus (ARC) tasks. The project aims to provide a high-performance, functionally-pure environment for training AI agents on abstract reasoning puzzles.

### Key Features:
- **JAX-Native**: Fully compatible with `jax.jit`, `jax.vmap`, and `jax.pmap`.
- **High Performance**: Optimized for speed with JIT compilation.
- **Extensible Architecture**: Designed to support future extensions like Hierarchical RL, Meta-RL, and Multi-Task RL.
- **Type Safety**: Uses typed configuration dataclasses with validation.
- **Rich Visualization**: Includes utilities for rendering grids in the terminal and as SVG images.
- **Multiple Datasets**: Supports ARC-AGI, ConceptARC, and MiniARC.

## 2. Core Principles

### Simplicity First
- Prioritize clear, readable, and simple code.
- Avoid over-engineering. This is a research project, so simplicity is preferred over complex optimizations unless performance is critical.

### JAX Compliance
- **Pure Functions**: All core functionality must be implemented as pure functions.
- **JIT Compatibility**: Ensure all functions work with `jax.jit`, `jax.vmap`, and `jax.pmap`.
- **Immutable State**: Use `equinox.Module` for state management.
- **Static Shapes**: Maintain static array shapes using padding and masks.
- **PRNG Management**: Manage PRNG keys explicitly using `jax.random.split`.

### Code Evolution
- **No Backwards Compatibility**: When introducing new features, replace old implementations.
- **Single Source of Truth**: Maintain a single, correct way to perform each task.
- **Clean Migration**: When replacing functionality, update all usage sites and remove the old code.

## 3. Development Workflow

### Environment Setup
1. Clone the repository: `git clone https://github.com/aadimator/JaxARC.git`
2. Navigate to the project directory: `cd JaxARC`
3. Activate the project environment: `pixi shell`
4. Install pre-commit hooks: `pixi run -e dev pre-commit install`

### Typical Workflow
1. Create a feature branch.
2. Make changes in modular components.
3. Write tests for new functionality.
4. Run linting and tests to ensure code quality and correctness.
5. Submit a pull request.

### Testing
- Write temporary test scripts in the root directory (e.g., `test_my_feature.py`) for initial validation.
- Run these scripts with `pixi run python test_my_feature.py`.
- Delete temporary test files after validation.
- For permanent tests, add them to the `tests/` directory.
- Run the full test suite with `pixi run -e test test`.

## 4. Technical Guidelines

### Configuration
- Use `equinox.Module` for configuration structures.
- Use Hydra for managing complex parameter hierarchies.

### Logging
- Use `loguru` for logging instead of Python's default logging module.

### Command-Line Interfaces
- Use `typer` for creating command-line interfaces.

### File Paths
- Use `pyprojroot.here()` to get the project root directory.

### Code Style
- **Formatting**: Use `black` for code formatting.
- **Linting**: Use `ruff` for linting.
- **Type Hints**: Add type hints for all public APIs.
- **Docstrings**: Write comprehensive docstrings.

## 5. Project Structure

- `src/jaxarc`: The main source code for the JaxARC library.
    - `src/jaxarc/envs`: Core RL environment components.
    - `src/jaxarc/conf`: Default Hydra configuration files.
    - `src/jaxarc/parsers`: Parsers for different ARC datasets.
    - `src/jaxarc/utils`: Utility functions.
- `tests`: The test suite for the project.
- `docs`: Project documentation.
- `examples`: Example scripts demonstrating how to use JaxARC.
- `scripts`: Helper scripts for tasks like downloading datasets.

## 6. Key Commands

- `pixi shell`: Activate the project environment.
- `pixi run lint`: Run the linter.
- `pixi run -e test test`: Run the full test suite.
- `pixi run docs-serve`: Serve the documentation locally.
- `pixi add <package>`: Add a new dependency.
- `pixi run python <script.py>`: Run a python script in the environment.
