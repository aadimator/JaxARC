# API Reference

Complete API documentation for all JaxARC modules, auto-generated from source
code docstrings.

JaxARC provides a JAX-native environment for training agents on ARC (Abstraction
and Reasoning Corpus) tasks. The API is organized into several key modules:

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Core API
:link: core
:link-type: doc

`make()`, `Environment`, `State`, `Action` - Essential functions and classes for creating and running environments
:::

:::{grid-item-card} Registration
:link: registration
:link-type: doc

Environment registration system, task IDs, and dataset management
:::

:::{grid-item-card} Wrappers
:link: wrappers
:link-type: doc

Action and observation wrappers for transforming environment interface
:::

:::{grid-item-card} Configuration
:link: config
:link-type: doc

`JaxArcConfig` and configuration system with Hydra integration
:::

:::{grid-item-card} Types
:link: types
:link-type: doc

Type definitions: `EnvParams`, `TimeStep`, and other core types
:::

:::{grid-item-card} Utilities
:link: utils
:link-type: doc

Helper functions for visualization, data loading, and more
:::

::::

## Module Index

```{toctree}
:maxdepth: 2

core
registration
wrappers
config
types
utils
```
