---
applyTo: '**'
---

# Project Context

This project is a Python package for designing and training a JAX-based MARL environment. It uses [Pixi](https://pixi.js.org) for project management and [Jupyter Book](https://jupyterbook.org) for documentation.
It is designed to be modular and extensible, allowing users to easily add new features or modify existing ones.

# Guidelines

- Use [Pixi](https://pixi.js.org) for project management and build processes.
- Use the utility functions already defined in `.utils` package for loading configuration files, and getting data directory paths.
- Try to have 100% test coverage for all new code.
- Go step by step, and ensure each step is working before moving on to the next.

# Useful Pixi Commands

- `pixi run lint`: Run linting to check code style and quality.
- `pixi run docs-serve`: Serve the documentation locally.
- `pixi run test`: Run tests to ensure code correctness.
- `pixi run python <script.py>`: Run a Python script within the project environment.
