# Phase 1 Implementation Guide

## Chapter 1: Introduction and Phase 1 Objectives

### 1.1. Purpose of This Guide

This document provides a focused, technical developer guide for implementing
Phase 1 of the JaxARC project, with an initial emphasis on the ARC-AGI-1
dataset. The Abstraction and Reasoning Corpus (ARC) presents a significant
challenge in artificial intelligence, requiring systems to demonstrate fluid
intelligence by solving novel visual puzzles from very few examples The JaxARC
project aims to explore a novel Multi-Agent Reinforcement Learning (MARL)
approach to tackle this challenge

This guide is specifically designed to be detailed enough for an LLM (Large
Language Model) code agent to follow. It covers all essential implementation
aspects for Phase 1, including environment setup, core data structure
definition, data loading and preprocessing for the ARC-AGI-1 dataset, and a
comprehensive unit testing strategy to ensure high code coverage and robustness.
The successful completion of this phase will result in a foundational software
framework upon which more complex MARL capabilities can be built.

### 1.2. Overview of JaxARC and the ARC-AGI-1 Challenge

The ARC-AGI-1 benchmark, introduced by François Chollet, is designed to measure
a system's ability to reason abstractly and solve entirely new problems with
minimal prior experience, a hallmark of fluid intelligence Tasks are presented
as visual puzzles on a grid, typically no larger than 30x30 cells, with each
cell displaying one of ten possible colors A system must infer underlying
abstract transformation rules from a small set of input-output demonstration
pairs and apply these rules to a new test input grid The ARC-AGI-1 dataset
comprises 400 public training tasks and 400 public evaluation tasks, among
others, designed to test these capabilities

The JaxARC project envisions a system where specialized AI agents
collaboratively reason about and solve these ARC puzzles. This approach deviates
from monolithic models by employing "cognitive decomposition," where agents
specialize in distinct functional aspects of the task, such as inferring grid
properties, identifying objects, or proposing transformations, rather than
operating on predefined spatial sub-grids Collaboration is facilitated through a
shared reasoning workspace and a consensus mechanism The environment will be
built using JAX for high performance and JaxMARL as the underlying MARL
framework, with Hydra for configuration management

### 1.3. Phase 1 Goals: Laying the Foundation

Phase 1 of the JaxARC project is dedicated to establishing a robust and
functional software foundation. The primary objectives are:

1. **Environment Setup:** Configure a reproducible development environment using
   Pixi, Git, and Python.
2. **Project Structure:** Implement the defined project directory structure and
   initialize Hydra for configuration management, specifically tailored for the
   ARC-AGI-1 dataset.
3. **Core Data Types:** Define essential JAX Pytrees (using `chex.dataclass`)
   for representing the environment state, agent actions, hypotheses, and parsed
   task data. These structures are critical for JAX compatibility and
   performance.
4. **ARC-AGI-1 Data Pipeline:** Develop a data loading and preprocessing
   pipeline for the ARC-AGI-1 dataset. This includes:
   - A dedicated parser (`ArcAgi1Parser`) to read raw ARC-AGI-1 JSON task files.
   - Logic to convert list-based grids into JAX arrays.
   - Implementation of padding for all grids to a fixed maximum size (30x30 for
     ARC-AGI-1) and creation of corresponding boolean masks to ensure static
     array shapes for JAX JIT compilation.
   - Batching utilities to prepare data for vectorized processing.
5. **Unit Testing Strategy:** Define and prepare for the implementation of
   comprehensive unit tests for all components developed in Phase 1, ensuring
   correctness, JAX compatibility, and high code coverage.

Successfully completing these goals will result in a system capable of loading,
preprocessing, and structuring ARC-AGI-1 task data in a JAX-idiomatic way, ready
for integration with the MARL environment logic in subsequent phases. This
meticulous groundwork is essential for the stability, performance, and
extensibility of the JaxARC project

## Chapter 2: Setting Up the JaxARC Development Environment

A consistent and reproducible development environment is paramount for any
software project, particularly one involving complex AI models and multiple
dependencies like JaxARC This chapter details the tools and steps required to
establish this environment, focusing on Pixi for dependency management and Hydra
for configuration.

### 2.1. Essential Tools: Git, Python, and Pixi

The development of JaxARC will rely on standard, robust tooling:

- **Git:** Essential for version control. All code changes will be tracked using
  Git, facilitating collaboration, history tracking, and branching for new
  features or experiments
- **Python:** The primary programming language. The project requires Python
  version 3.10 or higher, as specified in the project configuration
- **Pixi:** Chosen for dependency and environment management. Pixi ensures
  reproducible environments across different machines and for different
  developers by managing all Python packages and their dependencies. This
  simplifies the setup process and minimizes issues related to incompatible
  package versions, which is particularly beneficial for developers who might be
  earlier in their careers

The selection of Pixi underscores a commitment to minimizing setup friction and
ensuring that the development environment is consistent everywhere, a vital
aspect for collaborative research and development

### 2.2. Project Structure and Initialization

#### 2.2.1. Creating the `jax-arc-marl/` Directory Structure

The JaxARC project will be organized within a single main repository. The
initial step is to create the root project directory and the basic internal
structure. If a starter repository is provided, it should be cloned. Otherwise,
create the directory manually:

Bash

```sh
mkdir jax-arc-marl
cd jax-arc-marl
mkdir conf
mkdir conf/environment
mkdir conf/algorithm
mkdir conf/agent
mkdir src
mkdir src/jaxarc
mkdir src/jaxarc/base
mkdir src/jaxarc/envs
mkdir src/jaxarc/parsers
mkdir src/jaxarc/utils
mkdir src/training
touch src/jaxarc/__init__.py
touch src/jaxarc/base/__init__.py
touch src/jaxarc/envs/__init__.py
touch src/jaxarc/parsers/__init__.py
touch src/jaxarc/utils/__init__.py
```

This structure, detailed in the project blueprint 1, separates concerns
logically (e.g., configuration in `conf/`, core library code in `src/jaxarc/`,
training scripts in `src/training/`). The internal layout of `src/jaxarc/` with
`base/`, `envs/`, and `parsers/` directly reflects the modular design intended
to support multiple ARC datasets

#### 2.2.2. The `pyproject.toml` File for Pixi

Pixi uses a `pyproject.toml` file located at the root of the project to manage
configurations and dependencies Create this file in the `jax-arc-marl/`
directory with the following content. Note the inclusion of `drawsvg` for SVG
logging, replacing the unmaintained `svgwrite`

Ini, TOML

```toml
[project]
name = "jax-arc-marl"
version = "0.1.0"
description = "A JaxMARL environment for the ARC challenge."
authors = # Placeholder, update if needed
requires-python = ">=3.10"

[tool.pixi.environments.default]
# Specifies the default environment
solve-group = "default" # Optional: for grouping solves

[tool.pixi.dependencies]
python = ">=3.10"
jax = ">=0.4.20"
# For Apple Silicon (macOS) - uncomment if applicable
# jax-metal = ">=0.0.5"
# For NVIDIA GPUs (Linux/WSL) - ensure jaxlib matches your CUDA version
# jaxlib = {version = ">=0.4.20", extras = ["cuda12_pip"]} # Example for CUDA 12
jaxmarl = ">=0.2.0"
hydra-core = ">=1.3.2"
loguru = ">=0.7.2"
chex = ">=0.1.8"
rich = ">=13.0.0"      # For enhanced console logging
drawsvg = ">=1.8.1"   # For SVG logging (replaces svgwrite)
# Add other dependencies as they become necessary, e.g., flax, optax

[tool.pixi.tasks]
# Define any common project tasks here, e.g., linting, testing, running training
# train = "python src/training/train.py"
```

A critical aspect of this configuration is the `jaxlib` dependency, which is
hardware-specific. The developer must ensure the correct `jaxlib` version
compatible with their JAX version and hardware (CPU, specific CUDA version for
NVIDIA, or Metal for macOS) is specified and uncommented. Failure to correctly
specify this can lead to JAX not utilizing available GPUs or TPUs, significantly
impacting performance, or even causing installation failures If no specific
hardware accelerator is targeted initially, JAX will default to CPU.

#### 2.2.3. Initializing the Environment with `pixi install`

Once the `pyproject.toml` file is in place, the environment can be initialized
by running the following command in the terminal from the `jax-arc-marl/`
directory 1:

Bash

```sh
pixi install
```

This command reads `pyproject.toml`, resolves all dependencies, and installs
them into a project-specific environment managed by Pixi. It also creates a
`pixi.lock` file, which records the exact versions of all resolved dependencies.
This lock file is the cornerstone of reproducibility, ensuring that anyone
setting up the project using `pixi install` will get an identical environment,
thus guaranteeing consistent code behavior across different setups

To activate the environment managed by Pixi, use 1:

Bash

```sh
pixi shell
```

This command opens a new shell session with the project's environment activated.

### 2.3. Hydra Configuration for ARC-AGI-1

Hydra is used for managing all experiment configurations, allowing a clean
separation of parameters (defined in YAML files) from Python code This is
particularly useful for JaxARC's modular design, enabling easy switching between
different ARC datasets and their specific configurations.

#### 2.3.1. Main Configuration (`conf/config.yaml`)

Create the main Hydra configuration file `conf/config.yaml`. This file serves as
the entry point, defining default settings and specifying which
sub-configurations (e.g., for environment, algorithm) to compose

YAML

```yaml
# conf/config.yaml
defaults:
  - environment: arc_agi_1 # Default dataset/environment configuration
  - algorithm: mappo # Example: Default algorithm configuration
  # For colored logging output in Hydra itself
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog

# Experiment-level settings
seed: 42
num_env_steps: 10_000_000 # Example global parameter
# num_test_steps: 10 # For basic training script testing
# log_level_console: "INFO"
# log_level_file: "DEBUG"
# log_file_rotation: "10 MB"
# log_file_retention: "5 files"

# Default environment parameters that might be part of the environment's own config node
# but can also be defined here for global access if needed.
# These will be overridden by specific environment YAMLs if defined there.
max_steps_per_episode: 200 # Example: default max steps for an episode
```

#### 2.3.2. ARC-AGI-1 Environment Configuration (`conf/environment/arc_agi_1.yaml`)

Create the environment-specific configuration file
`conf/environment/arc_agi_1.yaml`. This file will define parameters specific to
the ARC-AGI-1 environment, including the `_target_` classes for the environment
and its data parser, and dataset-specific parameters like `max_grid_size`

The `_target_` key is a powerful feature of Hydra that enables the direct
instantiation of Python objects (like specific environment or parser classes)
based on their path specified in the configuration file This mechanism is
fundamental to JaxARC's modularity, as it allows the main training script to
remain agnostic to the concrete classes being used, instantiating them purely
based on the selected YAML configuration. For example, when
`hydra.utils.instantiate(cfg.environment)` is called in the main script, Hydra
looks at the `_target_` key within the `cfg.environment` block (which, if
`environment=arc_agi_1` is active, comes from `arc_agi_1.yaml`). It then creates
an instance of the class specified by that `_target_` path (e.g.,
`src.jaxarc.envs.arc_agi_1_env.ArcAgi1MarlEnv`), passing the remaining keys and
values from that YAML block as arguments to the class's `__init__` method. This
elegant decoupling facilitates easy swapping of dataset modules or other
components simply by changing a configuration file or a command-line argument,
without altering the core codebase.

Table 2.1 provides the content for this file.

**Table 2.1: Example `conf/environment/arc_agi_1.yaml`**

YAML

```yaml
# conf/environment/arc_agi_1.yaml
_target_: src.jaxarc.envs.arc_agi_1_env.ArcAgi1MarlEnv # Target class for ARC-AGI-1 env
env_name: "JaxARC-AGI-1"
max_grid_size: 30 # ARC-AGI-1 tasks up to 30x30 [3]
num_agents: 4 # Example, can be configured
agent_specializations: [
    "dimension_agent",
    "color_agent",
    "object_agent",
    "transform_agent",
  ] # Example

parser_config:
  _target_: src.jaxarc.parsers.arc_agi_1_parser.ArcAgi1Parser # Target class for ARC-AGI-1 parser
  dataset_path: "path/to/arc-agi-1/dataset" # Placeholder - user must provide actual path
  # Other ARC-AGI-1 specific parser params if any

reward_config: # Placeholders, to be detailed in later phases
  R_complete_solution: 100.0
  R_correct_grid_size: 10.0
  # R_object_replication: 5.0 # Example
  # R_successful_proposal: 1.0 # Example
  # P_incorrect_persistent_hypothesis: -0.5 # Example

consensus_config: # Placeholders, to be detailed in later phases
  support_threshold: 0.6
  # max_hypotheses: 32 # Max active hypotheses at a time
  # max_proposal_data_dim: 10 # Max dimension for proposal_data array
  # workspace_size: 100 # Size of shared_reasoning_workspace
  # workspace_feature_dim: 10 # Feature dimension for workspace

# Parameters for Pytree pre-allocation in State can also be defined here
# These are used by ArcMarlEnvBase and its children during initialization
max_hypotheses: 32
max_proposal_data_dim: 10 # Max dimension for Hypothesis.proposal_data and State.agent_hypotheses_data's last dim
workspace_size: 100
workspace_feature_dim: 10

# Environment specific parameters that might be used by default_params in ArcMarlEnvBase
max_steps_per_episode: 200
```

The `dataset_path` within `parser_config` is a crucial placeholder; the user
must replace `"path/to/arc-agi-1/dataset"` with the actual local file system
path to their ARC-AGI-1 dataset. The `max_grid_size: 30` is set based on the
ARC-AGI-1 specification that tasks involve grids up to 30x30.3

## Chapter 3: Defining Core JAX Pytrees for JaxARC (`types.py`)

The definition of core data structures using `chex.dataclass` is a foundational
step in Phase 1. These structures, known as JAX Pytrees, are essential for
organizing data within the JaxARC environment and ensuring compatibility with
JAX's function transformations like JIT compilation (`jax.jit`) and automatic
vectorization (`jax.vmap`) These Pytrees will reside in `src/jaxarc/types.py`.

### 3.1. Rationale: Why `chex.dataclass` for JAX Pytrees

JAX functions primarily operate on JAX arrays. However, real-world applications
often require complex, structured data. Pytrees are JAX's mechanism for handling
such nested Python containers (lists, tuples, dictionaries, and custom objects)
by allowing transformations to operate on their "leaf" JAX arrays while
preserving the overall structure

`chex.dataclass` instances are automatically registered as JAX Pytrees. This
means that JAX transformations can seamlessly interact with the JAX array
attributes (leaves) of these dataclass objects without requiring manual Pytree
registration, leading to cleaner, more robust, and more JAX-idiomatic code Using
`chex.dataclass` is thus not merely a preference but a strategic choice for
effective JAX development, particularly when dealing with complex state
representations as required by JaxARC.

### 3.2. Implementing `Hypothesis` Dataclass

The `Hypothesis` Pytree represents a single proposal made by an agent regarding
some aspect of the ARC puzzle solution. This structure is fundamental for the
collaborative reasoning mechanism envisioned for JaxARC, where agents share and
debate ideas Its fields are designed to capture the essential elements of an
agent's proposal, including its origin, content, and the agent's confidence,
which are crucial for the consensus process.

Create the file `src/jaxarc/types.py` and add the following definition:

Python

```python
# src/jaxarc/types.py
import chex
from jax import numpy as jnp
from typing import List, Any, Optional # Optional added for consistency

# It's good practice to define Enums for proposal_type and action_type
# For simplicity in this guide, we might use integers and document their meaning,
# but Enums (e.g., from enum.IntEnum) are more robust.

@chex.dataclass
class Hypothesis:
    """
    A single proposal made by an agent regarding some aspect of the ARC solution.
    """
    agent_id: int  # ID of the agent making the proposal
    proposal_type: int  # Enum: e.g., 0 for grid_size, 1 for color_palette, 2 for object_prop, 3 for transform
    proposal_data: jnp.ndarray  # Padded JAX array for the proposal data itself (flexible shape based on type)
    confidence: float  # Agent's confidence in this hypothesis (e.g., 0.0 to 1.0)
    vote_count: int  # Number of votes received (can be weighted sum in more advanced versions)
    # Optional: unique_id for the hypothesis if needed for voting on specific proposals
    # proposal_id: int
```

- **`agent_id`**: An integer identifying the agent that originated this
  hypothesis.
- **`proposal_type`**: An integer (ideally an Enum in a full implementation)
  indicating the nature of the proposal (e.g., output grid dimensions, color
  palette, a specific transformation rule).
- **`proposal_data`**: A JAX array containing the actual data of the proposal.
  Its shape and content will vary depending on `proposal_type` and will be
  padded to a maximum configured size.
- **`confidence`**: A float (e.g., between 0.0 and 1.0) indicating the proposing
  agent's confidence in this hypothesis.
- **`vote_count`**: An integer tracking the support this hypothesis has received
  from other agents.

### 3.3. Implementing `State` Dataclass

The `State` Pytree is arguably the most critical data structure in JaxARC. It
encapsulates the complete state of the MARL environment at any given timestep
Its design must be JAX-compatible, meaning all JAX array attributes must have
static shapes for efficient JIT compilation. Furthermore, it must be
comprehensive enough to hold all ARC task data (including training examples and
the current test input as per ARC-AGI-1's format 3), facilitate collaborative
reasoning mechanisms (like shared hypotheses and a workspace), and track
environment metadata.

The JAX-idiomatic pattern of using fixed-size arrays alongside an active mask
(e.g., `agent_hypotheses_data` and `agent_hypotheses_active_mask`) is employed
to handle a potentially variable number of active hypotheses while ensuring the
`State` Pytree maintains a static structure suitable for JAX transformations

Add the following definition to `src/jaxarc/types.py`:

Python

```python
# src/jaxarc/types.py (continued)

@chex.dataclass
class State:
    """
    The complete state of the MARL environment at any given time step.
    This structure is necessarily complex to support the collaborative reasoning model.
    """
    # Core ARC task data (typically static for the duration of a task)
    # These are for the 'train' examples of an ARC task
    input_grids_examples: jnp.ndarray     # Shape: (num_train_examples, max_grid_h, max_grid_w)
    input_masks_examples: jnp.ndarray     # Shape: (num_train_examples, max_grid_h, max_grid_w)
    output_grids_examples: jnp.ndarray    # Shape: (num_train_examples, max_grid_h, max_grid_w)
    output_masks_examples: jnp.ndarray    # Shape: (num_train_examples, max_grid_h, max_grid_w)

    # This is the specific 'test' input grid the agents are trying to solve
    current_test_input_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    current_test_input_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)

    # Ground truth for the current test input grid (hidden from agents during solving)
    # For ARC-AGI-1, test outputs are used for evaluation, not available during solving.
    true_test_output_grid: jnp.ndarray    # Shape: (max_grid_h, max_grid_w)
    true_test_output_mask: jnp.ndarray    # Shape: (max_grid_h, max_grid_w)

    # Collaborative workspace & solution construction
    working_output_grid: jnp.ndarray      # Shape: (max_grid_h, max_grid_w) - agents modify this
    working_output_mask: jnp.ndarray      # Shape: (max_grid_h, max_grid_w) - mask for working_output_grid

    # Store of all active hypotheses from agents
    # A concrete way for a fixed number of hypotheses:
    agent_hypotheses_ids: jnp.ndarray           # (max_hypotheses,) storing agent_id
    agent_hypotheses_types: jnp.ndarray         # (max_hypotheses,) storing proposal_type
    agent_hypotheses_data: jnp.ndarray          # (max_hypotheses, max_proposal_data_dim)
    agent_hypotheses_confidence: jnp.ndarray    # (max_hypotheses,)
    agent_hypotheses_votes: jnp.ndarray         # (max_hypotheses,)
    agent_hypotheses_active_mask: jnp.ndarray   # (max_hypotheses,) boolean, True if slot is used

    # Flexible blackboard for intermediate reasoning, could be a structured Pytree
    # For simplicity, a placeholder fixed-size array. Actual structure depends on reasoning needs.
    shared_reasoning_workspace: jnp.ndarray # Shape: (workspace_size, feature_dim)

    # Aggregated agreement from consensus mechanism
    # Structure depends on what aspects can reach consensus (e.g., agreed_dims, agreed_colors)
    # Example:
    consensus_grid_dimensions: jnp.ndarray  # (2,) e.g., [height, width] if agreed, else [-1,-1]
    #… other consensus items

    # Environment metadata
    step_count: int
    terminal: bool
    # PRNGKey for any stochasticity within the state itself (if needed, else pass to methods)
    key: chex.PRNGKey
```

Key fields include:

- ARC Task Data: `input_grids_examples`, `input_masks_examples`,
  `output_grids_examples`, `output_masks_examples` for training pairs;
  `current_test_input_grid`, `current_test_input_mask` for the test input being
  solved; and `true_test_output_grid`, `true_test_output_mask` for the ground
  truth solution (used for reward calculation and evaluation). All grid arrays
  are padded to `max_grid_size` (e.g., 30x30 for ARC-AGI-1).
- Collaborative Workspace: `working_output_grid` and `working_output_mask`
  represent the grid agents collaboratively construct. The `agent_hypotheses_*`
  arrays store active agent proposals, using `agent_hypotheses_active_mask` to
  indicate used slots, maintaining static shapes. `shared_reasoning_workspace`
  serves as a flexible "blackboard."
- Consensus State: Fields like `consensus_grid_dimensions` store outcomes of the
  consensus process.
- Metadata: `step_count`, `terminal` flag, and a JAX `key` for PRNG management.

### 3.4. Implementing `Action` Dataclass

The `Action` Pytree defines the structure of actions taken by individual agents.
It needs to be flexible enough to represent diverse agent intentions, such as
proposing a hypothesis, voting on an existing hypothesis, or directly
manipulating the working grid The `action_type` field acts as a discriminator,
while the generic `action_data` field (padded to a maximum possible size to
ensure static shape) carries the specific payload for that action type. This
design allows for a single, consistent `Action` Pytree structure to be used in
the environment's `step` method.

Add the following definition to `src/jaxarc/types.py`:

Python

```python
# src/jaxarc/types.py (continued)

@chex.dataclass
class Action:
    """
    An action taken by a single agent.
    """
    action_type: int  # Enum: e.g., 0 for PROPOSE, 1 for VOTE, 2 for MANIPULATE_GRID
    # Data associated with the action. Structure/size depends on action_type.
    # Examples:
    # - PROPOSE: could contain a full Hypothesis structure (or parts of it)
    # - VOTE: could contain proposal_id_to_vote_on, vote_value (support/oppose)
    # - MANIPULATE_GRID: could contain coordinates, color, transformation_id
    action_data: jnp.ndarray # Flexible JAX array, padded to max possible size for any action data
    # Optional: agent_id if actions are processed in a way that needs it explicitly here,
    # though JaxMARL usually provides actions in a dict keyed by agent_id.
```

- **`action_type`**: An integer (ideally an Enum) specifying the category of
  action.
- **`action_data`**: A JAX array containing specific parameters for the chosen
  `action_type`. Its content and interpretation vary based on `action_type`. It
  must be padded to a pre-configured maximum size.

### 3.5. Implementing `ParsedTaskData` Dataclass

The `ParsedTaskData` Pytree serves as a crucial data contract between the
dataset-specific parser (e.g., `ArcAgi1Parser`) and the environment's `reset`
method. It encapsulates all the JAX-ified grid and mask arrays for a single,
fully preprocessed ARC task, ensuring that the environment receives data in a
standardized, JAX-compatible format (padded arrays with masks) irrespective of
the raw dataset's original structure This Pytree is the direct output of the
parser and is used to initialize the relevant fields of the main `State` Pytree
during an environment reset.

Add the following definition to `src/jaxarc/types.py`:

Python

```python
# src/jaxarc/types.py (continued)

@chex.dataclass
class ParsedTaskData:
    """
    This dataclass will hold the preprocessed JAX arrays for a single task
    (training examples, test input, true test output, and all associated masks)
    as output by the parser.
    """
    input_grids_examples: jnp.ndarray     # Shape: (N_ex, max_grid_h, max_grid_w)
    input_masks_examples: jnp.ndarray     # Shape: (N_ex, max_grid_h, max_grid_w)
    output_grids_examples: jnp.ndarray    # Shape: (N_ex, max_grid_h, max_grid_w)
    output_masks_examples: jnp.ndarray    # Shape: (N_ex, max_grid_h, max_grid_w)
    current_test_input_grid: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    current_test_input_mask: jnp.ndarray  # Shape: (max_grid_h, max_grid_w)
    true_test_output_grid: jnp.ndarray    # Shape: (max_grid_h, max_grid_w)
    true_test_output_mask: jnp.ndarray    # Shape: (max_grid_h, max_grid_w)
    # Add other fields to ParsedTaskData if needed, e.g., task_id (str, non-JAX, handle carefully in Pytrees)
```

Table 3.1 provides detailed descriptions for each field in the `ParsedTaskData`
Pytree. This level of specification is vital for the LLM agent to correctly
implement both the parser's output and the environment's consumption of this
data, particularly concerning JAX array shapes and data types. For ARC-AGI-1,
`max_grid_h` and `max_grid_w` will be 30.3 `N_ex` represents the number of
training examples for a given task, which can vary.

**Table 3.1: `ParsedTaskData` Pytree Field Descriptions**

|                           |                                                 |                                                                                                                                  |
| ------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Field Name**            | **JAX Data Type (Example Shape for ARC-AGI-1)** | **Description/Purpose**                                                                                                          |
| `input_grids_examples`    | `jnp.ndarray (N_ex, 30, 30)`                    | Padded input grids from all training pairs of the current ARC task. Used for rule inference.                                     |
| `input_masks_examples`    | `jnp.ndarray (N_ex, 30, 30)`                    | Boolean masks for `input_grids_examples`, indicating valid data areas.                                                           |
| `output_grids_examples`   | `jnp.ndarray (N_ex, 30, 30)`                    | Padded output grids from all training pairs. Used for rule inference.                                                            |
| `output_masks_examples`   | `jnp.ndarray (N_ex, 30, 30)`                    | Boolean masks for `output_grids_examples`.                                                                                       |
| `current_test_input_grid` | `jnp.ndarray (30, 30)`                          | The specific padded input grid for the single test case agents are currently solving.                                            |
| `current_test_input_mask` | `jnp.ndarray (30, 30)`                          | Boolean mask for `current_test_input_grid`.                                                                                      |
| `true_test_output_grid`   | `jnp.ndarray (30, 30)`                          | Ground truth solution (padded) for the `current_test_input_grid`. Hidden from agents during solving; used for evaluation/reward. |
| `true_test_output_mask`   | `jnp.ndarray (30, 30)`                          | Boolean mask indicating the valid area of `true_test_output_grid`.                                                               |

## Chapter 4: ARC-AGI-1 Data Handling: Parsing and Preprocessing

This chapter details the implementation of the data pipeline for the ARC-AGI-1
dataset, a core component of Phase 1. This pipeline transforms raw ARC-AGI-1
JSON task files into the JAX-compatible `ParsedTaskData` Pytrees defined in
Chapter 3.

### 4.1. Overview of the ARC-AGI-1 Dataset Format

The ARC-AGI-1 dataset consists of tasks presented in JSON format Each task is
designed to test abstract reasoning and problem-solving skills. Key
characteristics relevant to parsing include:

- **Task Structure:** Each JSON file typically represents a single task. The
  root of the JSON is a dictionary where keys are unique task identifiers
  (strings). Each task ID maps to a dictionary containing "train" and "test"
  keys
- **Demonstration (Train) Pairs:** The "train" key maps to a list of
  demonstration pairs. Each pair is a dictionary with "input" and "output" keys,
  where values are 2D lists of integers representing the grids The number of
  training pairs can vary per task (median of three, but can be two or more)
- **Test Cases:** The "test" key maps to a list of test cases. Each test case is
  a dictionary with an "input" grid (2D list of integers) and, for evaluation
  purposes, an "output" grid During solving, the "output" grid of a test case is
  hidden from the agent. ARC-AGI-1 tasks typically have one test input per file,
  though the format allows for more
- **Grid Properties:** Grids are rectangular, with dimensions up to 30x30 cells.
  Each cell contains an integer from 0 to 9, representing one of ten distinct
  colors
- **Dataset Splits:** The publicly available ARC-AGI-1 dataset includes 400
  "public training tasks" and 400 "public evaluation tasks" The parser should be
  capable of loading tasks from these specified splits, typically organized in
  subdirectories.

The nested JSON structure and the variable number of training examples per task
directly inform the looping and data extraction logic required within the
`ArcAgi1Parser`. The parser must be robust to these variations to correctly
process all tasks.

Table 4.1 provides a simplified visual example of the ARC-AGI-1 JSON task
structure, which is crucial for understanding the raw data format the parser
will consume.

**Table 4.1: Key ARC-AGI-1 JSON Task Structure for Parsing**

JSON

```json
{
  "00576224": {
    // Example Task ID
    "train": [
      {
        "input": [,], // Example 2x2 input grid
        "output": [,] // Example 2x2 output grid
      }
      //… potentially more training pairs
    ],
    "test": [
      {
        "input": [,], // Example 2x2 test input grid
        "output": [,] // Ground truth for evaluation, hidden during solving
      }
      //… potentially more test inputs, though typically one for ARC-AGI-1
    ]
  }
  //… more tasks in the file if the file contains multiple tasks,
  // or this structure is per task file. For ARC-AGI-1, typically one task per JSON file.
}
```

### 4.2. Designing the `ArcAgi1Parser`

The `ArcAgi1Parser` will be responsible for loading and preprocessing ARC-AGI-1
tasks. Create the file `src/jaxarc/parsers/arc_agi_1_parser.py`.

The `ArcAgi1Parser` class will inherit from `ArcDataParserBase` This inheritance
ensures that the parser adheres to a common interface expected by the JaxARC
environment.

Constructor (`__init__`):

The constructor will initialize the parser:

1. Call `super().__init__(cfg, max_grid_size_h, max_grid_size_w)`.
2. Retrieve the `dataset_path` from the Hydra configuration:
   `self.dataset_path = cfg.parser_config.dataset_path`. This path points to the
   root directory of the ARC-AGI-1 dataset.
3. Scan the `self.dataset_path` for all ARC-AGI-1 task JSON files. This
   typically involves iterating through subdirectories like "training" and
   "evaluation" 3 and collecting all `.json` file paths. Store these paths in a
   list, e.g., `self.task_files`. This list will be used by the
   `get_random_task` method to sample tasks. It is important that the
   `dataset_path` is provided via the Hydra configuration
   (`cfg.parser_config.dataset_path`) rather than being hardcoded. This makes
   the parser flexible and allows users to easily specify the location of their
   local ARC-AGI-1 dataset copy.

Python

```python
# src/jaxarc/parsers/arc_agi_1_parser.py
import jax
import jax.numpy as jnp
import json
import os
from typing import Any, List, Tuple # Tuple added for type hint consistency
import chex
from omegaconf import DictConfig

# Assuming ArcDataParserBase is in src.jaxarc.base.base_parser
from src.jaxarc.base.base_parser import ArcDataParserBase
# Assuming ParsedTaskData is in src.jaxarc.types
from src.jaxarc.types import ParsedTaskData

class ArcAgi1Parser(ArcDataParserBase):
    def __init__(self, cfg: DictConfig, max_grid_size_h: int, max_grid_size_w: int):
        super().__init__(cfg, max_grid_size_h, max_grid_size_w)
        self.dataset_path = self.parser_cfg.get("dataset_path", None)
        if not self.dataset_path:
            raise ValueError("dataset_path not specified in parser_cfg for ArcAgi1Parser")
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Specified dataset_path does not exist or is not a directory: {self.dataset_path}")

        self.task_files: List[str] =
        # ARC-AGI-1 typically has 'training' and 'evaluation' subdirectories for public tasks [3]
        # Other potential splits like 'test' (for private eval) might exist.
        # For this guide, focusing on common public splits.
        expected_subdirs = ["training", "evaluation"] # Add more if needed based on dataset structure

        # Check if the dataset_path itself contains JSON files (e.g., if it's a flat directory of tasks)
        for filename in os.listdir(self.dataset_path):
            if filename.endswith(".json"):
                self.task_files.append(os.path.join(self.dataset_path, filename))

        # Check subdirectories
        for subdir_name in expected_subdirs:
            subset_path = os.path.join(self.dataset_path, subdir_name)
            if os.path.isdir(subset_path):
                for filename in os.listdir(subset_path):
                    if filename.endswith(".json"):
                        self.task_files.append(os.path.join(subset_path, filename))

        if not self.task_files:
            # Consider if this should be a warning or an error depending on expected use.
            # For now, making it an error if no tasks are found.
            raise FileNotFoundError(f"No ARC-AGI-1 task files (.json) found in {self.dataset_path} or its expected subdirectories ({', '.join(expected_subdirs)}).")

        # Optional: Log the number of tasks found
        # print(f"Found {len(self.task_files)} ARC-AGI-1 task files.")

    # load_task_file, _preprocess_grid, preprocess_task_data, get_random_task methods to follow
```

### 4.3. Implementing `load_task_file` for ARC-AGI-1 JSON Structure

The `load_task_file` method is responsible for reading a single ARC-AGI-1 task
JSON file from the given path and loading its content into a Python dictionary
This dictionary then serves as the input to the `preprocess_task_data` method.

Python

```python
# src/jaxarc/parsers/arc_agi_1_parser.py (continued)

    def load_task_file(self, task_file_path: str) -> Any: # Returns raw Python dict
        """Loads a single ARC task file from the given path."""
        if not os.path.exists(task_file_path):
            raise FileNotFoundError(f"Task file not found: {task_file_path}")
        try:
            with open(task_file_path, 'r') as f:
                # ARC-AGI-1 JSON files often contain a single top-level key which is the task ID,
                # and its value is the dictionary with "train" and "test" keys.
                # Example: {"task_id_xyz": {"train": […], "test": […]}}
                # Some loaders might expect to directly get the inner {"train":…, "test":…} dict.
                # For simplicity, this implementation assumes the file contains one task,
                # and we might need to extract the actual task data if it's nested under a task ID key.
                # However, many ARC dataset versions have one JSON file per task, where the content
                # *is* directly the {"train":…, "test":…} dict.
                # The Kaggle ARC dataset format (often one task per file) is typically:
                # { "train": [ { "input": G, "output": G },… ],
                #   "test":  [ { "input": G, "output": G },… ] }
                # We will assume this simpler structure for now. If files have a root task_id key,
                # this logic would need adjustment (e.g., data = json.load(f); return list(data.values());)
                raw_task_data = json.load(f)
            return raw_task_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from task file {task_file_path}: {e}")
        except Exception as e:
            # Catch other potential errors during file loading
            raise IOError(f"Error loading task file {task_file_path}: {e}")
```

### 4.4. Implementing `preprocess_task_data`

This method is the core of the parser's transformation logic. It takes the raw
task data (a Python dictionary loaded by `load_task_file`) and converts it into
a JAX-compatible `ParsedTaskData` Pytree This involves converting list-based
grids to JAX arrays, padding them to the configured `max_grid_size` (30x30 for
ARC-AGI-1), and creating corresponding boolean masks. The requirement for static
array shapes in JAX JIT-compiled functions is the primary driver for this
preprocessing pipeline All grids passed to the JAX environment must have a
fixed, predetermined shape. Padding achieves this, and masks are then essential
to differentiate original data from the padding.

A helper method, `_preprocess_grid`, is highly recommended to encapsulate the
logic for processing a single grid (conversion, padding, masking), promoting
code reuse and clarity.

Python

```python
# src/jaxarc/parsers/arc_agi_1_parser.py (continued)

    def _preprocess_grid(self, grid_list: List[List[int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Converts a single grid (list of lists) to a padded JAX array and a mask."""
        if not isinstance(grid_list, list) or not all(isinstance(row, list) for row in grid_list):
            raise ValueError("Invalid grid format: Expected list of lists.")
        if not grid_list or not grid_list: # Handle empty or malformed grid
             # Create empty grid and mask according to max size if input is empty
            grid_array_unpadded = jnp.empty((0,0), dtype=jnp.int32)
        else:
            try:
                grid_array_unpadded = jnp.array(grid_list, dtype=jnp.int32)
            except Exception as e: # Catch potential errors during jnp.array conversion (e.g. ragged lists)
                raise ValueError(f"Error converting grid to JAX array. Ensure grid is not ragged. Details: {e}")

        h, w = grid_array_unpadded.shape

        if h > self.max_grid_size_h or w > self.max_grid_size_w:
            # This case should ideally be handled by either raising an error,
            # truncating, or having a config option. For ARC-AGI-1, grids are <= 30x30.
            # If a grid exceeds this, it might indicate a data issue or misconfiguration.
            raise ValueError(f"Grid dimensions ({h}x{w}) exceed max_grid_size ({self.max_grid_size_h}x{self.max_grid_size_w}).")

        # Padding value, e.g., -1 or a specific color index not used by ARC (0-9)
        # For ARC, colors are 0-9. -1 is a common choice for padding.
        pad_value = -1

        padded_grid = jnp.full((self.max_grid_size_h, self.max_grid_size_w), pad_value, dtype=jnp.int32)
        if h > 0 and w > 0 : # only set if there is actual data
            padded_grid = padded_grid.at[:h, :w].set(grid_array_unpadded)

        mask = jnp.zeros((self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_)
        if h > 0 and w > 0 : # only set if there is actual data
            mask = mask.at[:h, :w].set(True)

        return padded_grid, mask

    def preprocess_task_data(self, raw_task_data: Any, key: chex.PRNGKey) -> ParsedTaskData:
        """
        Preprocesses raw task data into JAX-compatible structures.
        This includes padding grids, creating masks, and organizing
        data into the ParsedTaskData Pytree.
        The PRNGKey is part of the signature but not used in this deterministic preprocessing.
        """
        train_pairs = raw_task_data.get("train",)
        test_pairs = raw_task_data.get("test",) # ARC-AGI-1 usually has one test pair [3]

        num_train_examples = len(train_pairs)

        # Initialize lists to hold JAX arrays before stacking
        train_input_grids_list: List[jnp.ndarray] =
        train_input_masks_list: List[jnp.ndarray] =
        train_output_grids_list: List[jnp.ndarray] =
        train_output_masks_list: List[jnp.ndarray] =

        for pair in train_pairs:
            input_grid, input_mask = self._preprocess_grid(pair["input"])
            output_grid, output_mask = self._preprocess_grid(pair["output"])
            train_input_grids_list.append(input_grid)
            train_input_masks_list.append(input_mask)
            train_output_grids_list.append(output_grid)
            train_output_masks_list.append(output_mask)

        # Stack training examples along a new leading dimension
        if num_train_examples > 0:
            stacked_train_input_grids = jnp.stack(train_input_grids_list)
            stacked_train_input_masks = jnp.stack(train_input_masks_list)
            stacked_train_output_grids = jnp.stack(train_output_grids_list)
            stacked_train_output_masks = jnp.stack(train_output_masks_list)
        else: # Handle case with no training examples
            shape_no_examples = (0, self.max_grid_size_h, self.max_grid_size_w)
            stacked_train_input_grids = jnp.empty(shape_no_examples, dtype=jnp.int32)
            stacked_train_input_masks = jnp.empty(shape_no_examples, dtype=jnp.bool_)
            stacked_train_output_grids = jnp.empty(shape_no_examples, dtype=jnp.int32)
            stacked_train_output_masks = jnp.empty(shape_no_examples, dtype=jnp.bool_)

        # Process test input (assuming one test input for ARC-AGI-1 for simplicity in State)
        if not test_pairs:
            raise ValueError("Task has no test pairs, which is unexpected for ARC-AGI-1.")

        # For this guide, we assume the environment handles one test input at a time from the ParsedTaskData.
        # We'll take the first test input.
        first_test_pair = test_pairs
        current_test_input_grid, current_test_input_mask = self._preprocess_grid(first_test_pair["input"])

        # The "output" for the test case is the ground truth solution
        # It's crucial this exists in the JSON for evaluation, even if hidden from agent during solving.
        if "output" not in first_test_pair:
             raise ValueError("First test pair in task JSON is missing the 'output' grid.")
        true_test_output_grid, true_test_output_mask = self._preprocess_grid(first_test_pair["output"])

        return ParsedTaskData(
            input_grids_examples=stacked_train_input_grids,
            input_masks_examples=stacked_train_input_masks,
            output_grids_examples=stacked_train_output_grids,
            output_masks_examples=stacked_train_output_masks,
            current_test_input_grid=current_test_input_grid,
            current_test_input_mask=current_test_input_mask,
            true_test_output_grid=true_test_output_grid,
            true_test_output_mask=true_test_output_mask
            # task_id could be added here if needed, but ParsedTaskData currently only holds JAX arrays.
            # If task_id (string) is needed, ParsedTaskData definition and handling would need adjustment.
        )
```

### 4.5. Implementing `get_random_task`

The `get_random_task` method orchestrates the selection of a random task from
the dataset, loads its raw data, and then preprocesses it into the
JAX-compatible `ParsedTaskData` Pytree The use of a JAX PRNG key for random
selection ensures reproducibility.

Python

```python
# src/jaxarc/parsers/arc_agi_1_parser.py (continued)

    def get_random_task(self, key: chex.PRNGKey) -> ParsedTaskData:
        """
        Loads and preprocesses a random task from the dataset.
        This is typically used by the environment's reset method.
        The PRNGKey is used for randomly selecting a task.
        """
        if not self.task_files:
            # This should have been caught in __init__, but as a safeguard:
            raise RuntimeError("No task files loaded by ArcAgi1Parser. Cannot select a random task.")

        # Select a random task file path using the PRNG key
        # key_select, key_preprocess = jax.random.split(key) # Split key if preprocess_task_data uses it for stochasticity
        # For deterministic preprocessing, the original key can be passed if not split elsewhere.
        # For this implementation, preprocess_task_data's key is unused, so we only need one for selection.

        task_idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.task_files))

        # Convert JAX array to Python int for file indexing.
        # This happens outside JIT context, so direct conversion is fine.
        selected_task_file_path = self.task_files[int(task_idx)]

        raw_task_data = self.load_task_file(selected_task_file_path)

        # The key passed to preprocess_task_data is part of the ABC signature.
        # If preprocess_task_data were stochastic, we'd use a subkey. Here it's deterministic.
        return self.preprocess_task_data(raw_task_data, key) # Pass original key or a subkey
```

### 4.6. Implementing Batching Logic for `ParsedTaskData` (in `src/jaxarc/utils/data_utils.py`)

To leverage JAX's vectorization capabilities (e.g., `jax.vmap`), a utility is
needed to batch multiple `ParsedTaskData` instances. This involves stacking the
corresponding JAX arrays from a list of `ParsedTaskData` objects along a new
leading "batch" dimension This batching enables efficient parallel processing of
multiple ARC tasks, which is crucial for speeding up data collection during MARL
training.

Create the file `src/jaxarc/utils/data_utils.py` and implement the batching
function:

Python

```python
# src/jaxarc/utils/data_utils.py
import jax
import jax.numpy as jnp
from typing import List
# Assuming ParsedTaskData is in src.jaxarc.types
from src.jaxarc.types import ParsedTaskData
import chex # For type hinting if needed

def collate_tasks_to_batch(task_list: List) -> ParsedTaskData:
    """
    Collates a list of ParsedTaskData Pytrees into a single batched ParsedTaskData Pytree.
    Each JAX array in the output Pytree will have an additional leading batch dimension.
    """
    if not task_list:
        raise ValueError("Input task_list cannot be empty for batching.")

    # Use jax.tree_map to stack corresponding leaves (JAX arrays) from all Pytrees in the list.
    # The first argument to tree_map is the function to apply to the leaves.
    # The subsequent arguments are the Pytrees themselves.
    # lambda *arrays: jnp.stack(arrays) takes corresponding arrays from each Pytree and stacks them.
    try:
        # This assumes all ParsedTaskData objects in task_list have identical Pytree structure.
        # jax.tree_util.tree_map (or jax.tree.map) applies a function to each corresponding leaf.
        batched_task_data = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *task_list)
    except Exception as e:
        # More specific error handling might be needed if Pytree structures mismatch.
        # For example, if one ParsedTaskData object was missing a field that others had.
        # However, they should all be instances of the same chex.dataclass.
        # This could also fail if leaves are not JAX arrays or cannot be stacked (e.g. different ranks beyond first dim).
        print(f"Error during jax.tree_map for batching. Ensure all ParsedTaskData objects have identical structure and compatible leaf shapes. Details: {e}")
        # Potentially re-raise or handle based on how robust this needs to be.
        # For now, let's provide a more manual way if tree_map is tricky for an LLM to get right initially
        # or if there are subtle issues with it.
        # Fallback manual stacking (more verbose, but perhaps clearer for initial generation):
        # This illustrative fallback assumes ParsedTaskData fields are known and are JAX arrays.
        # It's generally better to rely on tree_map if possible.
        #
        # print("Attempting manual fallback for batching due to tree_map error or for illustration.")
        # stacked_input_grids_examples = jnp.stack([task.input_grids_examples for task in task_list])
        # stacked_input_masks_examples = jnp.stack([task.input_masks_examples for task in task_list])
        # stacked_output_grids_examples = jnp.stack([task.output_grids_examples for task in task_list])
        # stacked_output_masks_examples = jnp.stack([task.output_masks_examples for task in task_list])
        # stacked_current_test_input_grid = jnp.stack([task.current_test_input_grid for task in task_list])
        # stacked_current_test_input_mask = jnp.stack([task.current_test_input_mask for task in task_list])
        # stacked_true_test_output_grid = jnp.stack([task.true_test_output_grid for task in task_list])
        # stacked_true_test_output_mask = jnp.stack([task.true_test_output_mask for task in task_list])
        #
        # batched_task_data = ParsedTaskData(
        #     input_grids_examples=stacked_input_grids_examples,
        #     input_masks_examples=stacked_input_masks_examples,
        #     output_grids_examples=stacked_output_grids_examples,
        #     output_masks_examples=stacked_output_masks_examples,
        #     current_test_input_grid=stacked_current_test_input_grid,
        #     current_test_input_mask=stacked_current_test_input_mask,
        #     true_test_output_grid=stacked_true_test_output_grid,
        #     true_test_output_mask=stacked_true_test_output_mask
        # )
        raise # Re-raise the original error if tree_map fails, as it's the preferred method.

    return batched_task_data

# Example Usage (for testing this utility):
# if __name__ == '__main__':
#     # This would require a dummy ArcAgi1Parser and some dummy task files or raw data.
#     # For simplicity, let's create some dummy ParsedTaskData objects directly.
#     max_h, max_w = 30, 30
#     num_ex = 2
#     dummy_task_1 = ParsedTaskData(
#         input_grids_examples=jnp.zeros((num_ex, max_h, max_w), dtype=jnp.int32),
#         input_masks_examples=jnp.zeros((num_ex, max_h, max_w), dtype=jnp.bool_),
#         output_grids_examples=jnp.zeros((num_ex, max_h, max_w), dtype=jnp.int32),
#         output_masks_examples=jnp.zeros((num_ex, max_h, max_w), dtype=jnp.bool_),
#         current_test_input_grid=jnp.zeros((max_h, max_w), dtype=jnp.int32),
#         current_test_input_mask=jnp.zeros((max_h, max_w), dtype=jnp.bool_),
#         true_test_output_grid=jnp.zeros((max_h, max_w), dtype=jnp.int32),
#         true_test_output_mask=jnp.zeros((max_h, max_w), dtype=jnp.bool_),
#     )
#     dummy_task_2 = ParsedTaskData( # Identical structure, different data if needed
#         input_grids_examples=jnp.ones((num_ex, max_h, max_w), dtype=jnp.int32),
#         input_masks_examples=jnp.ones((num_ex, max_h, max_w), dtype=jnp.bool_),
#         output_grids_examples=jnp.ones((num_ex, max_h, max_w), dtype=jnp.int32),
#         output_masks_examples=jnp.ones((num_ex, max_h, max_w), dtype=jnp.bool_),
#         current_test_input_grid=jnp.ones((max_h, max_w), dtype=jnp.int32),
#         current_test_input_mask=jnp.ones((max_h, max_w), dtype=jnp.bool_),
#         true_test_output_grid=jnp.ones((max_h, max_w), dtype=jnp.int32),
#         true_test_output_mask=jnp.ones((max_h, max_w), dtype=jnp.bool_),
#     )
#
#     task_batch = [dummy_task_1, dummy_task_2]
#     batched_data = collate_tasks_to_batch(task_batch)
#
#     print("Batched input_grids_examples shape:", batched_data.input_grids_examples.shape)
#     # Expected: (2, num_ex, max_h, max_w) -> (2, 2, 30, 30)
#     print("Batched current_test_input_grid shape:", batched_data.current_test_input_grid.shape)
#     # Expected: (2, max_h, max_w) -> (2, 30, 30)

```

## Chapter 5: Comprehensive Unit Testing Strategy for Phase 1

A comprehensive unit testing strategy is essential for ensuring the correctness,
robustness, and JAX compatibility of the components developed in Phase 1. This
chapter outlines the principles and specific tests for verifying the environment
setup, configuration loading, core data type definitions, the ARC-AGI-1 data
parser, and the batching utilities.

### 5.1. Principles of Unit Testing in a JAX Environment

Unit testing in a JAX-based project like JaxARC involves more than just
verifying functional correctness; it also encompasses ensuring JAX
compatibility. Key principles include:

- **Testing Pure Functions:** JAX thrives on pure functions (outputs depend only
  on inputs, no side effects) Tests should verify that for given inputs, pure
  functions produce the expected outputs.
- **Pytree Integrity:** JAX operations rely on the precise structure, shapes,
  and data types of Pytrees Tests must validate these aspects rigorously.
- **JAX-Specific Assertions:** Standard Python assertion libraries are often
  insufficient. The `chex` library provides utilities like `chex.assert_shape`,
  `chex.assert_type`, `chex.assert_trees_all_close`, and
  `chex.assert_trees_all_equal_shapes_types_structs`, which are indispensable
  for JAX-specific checks
- **Reproducibility:** For components involving randomness (like
  `get_random_task`), tests should verify that behavior is reproducible given
  the same JAX PRNG key.

Adopting these principles and tools like `pytest` for test execution and `chex`
for assertions will lead to a robust test suite.

### 5.2. Testing Environment Setup (`pyproject.toml`, Pixi installation)

While not traditional unit tests, these are crucial verification steps for the
development environment:

- **Pixi Installation:**
  - **Check:** After running `pixi install`, confirm that the command completes
    without errors.
  - **Verification:** Activate the Pixi shell (`pixi shell`) and attempt to
    import key packages (e.g., `import jax`, `import jaxmarl`, `import hydra`,
    `import chex`, `import drawsvg`). Verify that their versions match those
    specified in `pyproject.toml`. This ensures that the foundational
    environment is correctly established.

### 5.3. Testing Hydra Configuration Loading

The system's behavior is heavily driven by Hydra configurations. Testing their
loading and integrity is vital:

- **Strategy:** Create a minimal Python test script (e.g.,
  `tests/test_config_loading.py`) that uses `@hydra.main` to load
  `conf/config.yaml` and, by default, `conf/environment/arc_agi_1.yaml`.
- **Checks:**
  - Verify that the `DictConfig` object (`cfg`) received by the main function is
    populated correctly.
  - Access specific configuration values (e.g., `cfg.environment.max_grid_size`,
    `cfg.environment.num_agents`, `cfg.environment.parser_config._target_`,
    `cfg.seed`) and assert that they match the values defined in the YAML files.
  - This ensures that components instantiated via Hydra will receive the correct
    parameters.

### 5.4. Testing Core Data Type Definitions (`types.py`)

The Pytrees defined in `src/jaxarc/types.py` are fundamental data contracts.
Tests must ensure their definitions enforce the expected structure and JAX
compatibility:

- **Strategy:** Write tests (e.g., in `tests/test_types.py`) that instantiate
  `Hypothesis`, `State`, `Action`, and `ParsedTaskData` with sample JAX arrays
  conforming to their expected shapes and dtypes.
- **Checks:**
  - For each dataclass, instantiate it with valid JAX arrays for its fields.
  - Use `chex.assert_shape` and `chex.assert_type` to verify that the JAX array
    fields within the instantiated objects have the correct shapes and dtypes.
    For example, for `ParsedTaskData`, check that `current_test_input_grid` has
    shape `(30,30)` and `dtype=jnp.int32` when initialized with appropriate data
    for ARC-AGI-1.
  - Verify that these dataclass instances are recognized as JAX Pytrees (e.g.,
    by passing an instance to `jax.tree_util.tree_leaves` and checking if it
    traverses correctly, or by ensuring
    `chex.assert_trees_all_equal_shapes_types_structs` works between two
    identically structured instances).

### 5.5. Unit Testing the `ArcAgi1Parser`

The `ArcAgi1Parser` is critical as it's the entry point for all external
ARC-AGI-1 data. Its correctness and robustness are paramount. Tests should be
placed in, for example, `tests/parsers/test_arc_agi_1_parser.py`.

- **Test Data:** Create a small set of test ARC-AGI-1 JSON files with known,
  diverse structures (e.g., varying numbers of train examples, different grid
  sizes up to 30x30, empty grids, grids at max size). Place these in a dedicated
  test assets directory.
- **Tests for `__init__`:**
  - Instantiate `ArcAgi1Parser` with a valid path to test assets and mock Hydra
    config. Assert `self.task_files` is populated correctly.
  - Test instantiation with an invalid/empty `dataset_path`, assert appropriate
    errors are raised.
- **Tests for `load_task_file`:**
  - Input: Path to a valid test JSON file. Assert: Output is a Python dictionary
    matching the file's content.
  - Input: Path to a non-existent file. Assert: `FileNotFoundError` (or similar
    `IOError`) is raised.
  - Input: Path to a malformed JSON file. Assert: `json.JSONDecodeError` (or
    `ValueError`) is raised.
- **Tests for `_preprocess_grid` (if factored out as a helper):**
  - Input: Various sample grids (Python lists of lists: empty, small, max-size).
  - Assert: The returned JAX array (padded grid) and JAX boolean array (mask)
    have:
    - Correct shapes: `(max_grid_size_h, max_grid_size_w)` (e.g., (30,30)).
    - Correct dtypes: `jnp.int32` for grid, `jnp.bool_` for mask.
    - Correct values: Padding value in padded regions of the grid, `True` in
      original data regions of the mask and `False` in padded regions. Sum of
      mask should equal original area (`h*w`).
- **Tests for `preprocess_task_data`:**
  - Input: Raw task data (Python dictionary from a test JSON).
  - Assert: The output `ParsedTaskData` Pytree is correctly formed:
    - Use `chex.assert_trees_all_equal_shapes_types_structs` if an expected
      `ParsedTaskData` object with JAX arrays is pre-constructed for comparison.
    - Individually verify the `shape` and `dtype` of all JAX array fields in the
      returned `ParsedTaskData` object (e.g.,
      `parsed_task.input_grids_examples.shape` should be `(N_ex, 30, 30)`).
    - Verify the correctness of mask values (e.g., sum of
      `parsed_task.current_test_input_mask` equals the area of the original test
      input grid).
    - Verify that padding values are present in the padded regions of grids.
    - Test with tasks having zero training examples.
- **Tests for `get_random_task`:**
  - Mock `self.task_files` in an `ArcAgi1Parser` instance to point to a few
    known test JSON files.
  - Call `get_random_task` multiple times with different JAX PRNG keys.
  - Assert: The returned `ParsedTaskData` objects correspond to the content of
    one of the mocked task files and are correctly preprocessed (as per
    `preprocess_task_data` tests).
  - Assert: Calling `get_random_task` with the _same_ JAX PRNG key (and same
    `self.task_files` list) consistently returns the same task, correctly
    processed. This verifies reproducible random selection.

### 5.6. Unit Testing Batching Logic (`data_utils.py`)

Correct batching is essential for enabling vectorized operations with
`jax.vmap`. Tests should be placed in, for example,
`tests/utils/test_data_utils.py`.

- **Strategy:**

1.  Create a list of 2-3 sample `ParsedTaskData` Pytrees. These can be generated
    using the (tested) `ArcAgi1Parser` with different test task files, or
    constructed manually with JAX arrays of conforming shapes/dtypes.
2.  Pass this list to the batching function (e.g., `collate_tasks_to_batch`).

- **Checks:**
  - Assert: The output is a single `ParsedTaskData` Pytree.
  - For every JAX array field in the output batched Pytree (e.g.,
    `batched_data.input_grids_examples`,
    `batched_data.current_test_input_grid`):
    - Use `chex.assert_shape` to verify that its shape has a leading dimension
      equal to `len(list_of_sample_tasks)` (the batch size).
    - Verify that the subsequent dimensions of the array match the original
      dimensions of the corresponding array in an unbatched `ParsedTaskData`
      Pytree. For example, if an unbatched `current_test_input_grid` has shape
      `(30, 30)`, the batched version should have shape `(batch_size, 30, 30)`.
  - `chex.assert_trees_all_equal_shapes_types_structs` can be very effective if
    an expected batched `ParsedTaskData` Pytree is constructed manually for
    comparison, ensuring the entire Pytree structure and all leaf properties are
    correct.
  - Test with an empty input list; expect a `ValueError`.

### 5.7. Tools and Practices: `pytest`, `chex.assert_*` utilities

- **Test Runner:** `pytest` is recommended for test discovery and execution due
  to its ease of use and rich ecosystem.
- **JAX Assertions:** Consistently use `chex.assert_*` utilities
  (`chex.assert_shape`, `chex.assert_type`, `chex.assert_trees_all_close`,
  `chex.assert_trees_all_equal_shapes_types_structs`, etc.) for all JAX-specific
  validations of arrays and Pytrees
- **Test File Organization:** Place unit tests in a top-level `tests/`
  directory, mirroring the structure of the `src/` directory. For example, tests
  for `src/jaxarc/parsers/arc_agi_1_parser.py` would go into
  `tests/parsers/test_arc_agi_1_parser.py`.
- **Test Naming:** Follow standard test naming conventions (e.g., `test_*.py`
  for files, `test_*` for functions).

Adopting these standardized tools and practices contributes to a high-quality,
maintainable, and easily verifiable codebase.

Table 5.1 provides a summary checklist of the unit tests crucial for Phase 1
components, ensuring comprehensive coverage and acting as a quick reference.

**Table 5.1: Checklist of Unit Tests for Phase 1 Components**

|                                                         |                                                                                                                 |                                                                                                                                                                                      |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Component**                                           | **Key Test Areas**                                                                                              | **Example chex Assertions or Verification Steps**                                                                                                                                    |
| **Environment Setup** (`pyproject.toml`, Pixi)          | Pixi installation success, correct package versions.                                                            | Manual: `pixi install` no errors. `python -c "import jax; print(jax.__version__)"` in `pixi shell`.                                                                                  |
| **Hydra Configuration** (`conf/`)                       | Correct loading of YAML, parameter access.                                                                      | Script: Load `cfg` via `@hydra.main`, assert `cfg.environment.max_grid_size == 30`, `cfg.seed == 42`.                                                                                |
| **Core Data Types** (`src/jaxarc/types.py`)             | Pytree instantiation, field shapes/dtypes, JAX Pytree registration.                                             | `chex.assert_shape(state.current_test_input_grid, (30,30))`, `chex.assert_type(action.action_type, int)`. Verify `jax.tree_leaves` works.                                            |
| `ArcAgi1Parser.__init__`                                | Correct initialization of `task_files` list from `dataset_path`.                                                | Assert `len(parser.task_files) > 0` with valid path. Test error on invalid path.                                                                                                     |
| `ArcAgi1Parser.load_task_file`                          | Correct JSON loading, error handling for missing/malformed files.                                               | Assert output is `dict`. Test `FileNotFoundError`, `json.JSONDecodeError`.                                                                                                           |
| `ArcAgi1Parser._preprocess_grid`                        | Correct padding, mask generation, shapes, dtypes, values.                                                       | `chex.assert_shape(grid, (30,30))`, `chex.assert_shape(mask, (30,30))`, `chex.assert_type(grid, jnp.int32)`, `chex.assert_type(mask, jnp.bool_)`. Verify padding value and mask sum. |
| `ArcAgi1Parser.preprocess_task_data`                    | Correct `ParsedTaskData` Pytree output, all fields correctly processed (padded, masked), correct shapes/dtypes. | `chex.assert_trees_all_equal_shapes_types_structs` against expected. Check shapes of all `ParsedTaskData` fields (e.g., `N_ex, 30, 30` for example grids).                           |
| `ArcAgi1Parser.get_random_task`                         | Reproducible random task selection, correct preprocessing of selected task.                                     | Assert same output `ParsedTaskData` for same PRNG key. Assert output is valid `ParsedTaskData`.                                                                                      |
| **Batching Utility** (`src/jaxarc/utils/data_utils.py`) | Correct batch dimension addition, Pytree structure preservation, shapes/dtypes of batched arrays.               | `chex.assert_shape(batched.current_test_input_grid, (batch_size, 30,30))`. `chex.assert_trees_all_equal_shapes_types_structs` against expected batched Pytree.                       |

This comprehensive testing strategy for Phase 1 will ensure that the
foundational components of JaxARC are robust, correct, and fully compatible with
the JAX ecosystem, setting a strong stage for subsequent development phases.
