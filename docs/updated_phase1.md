# Architecting a JaxMARL ARC: Core Structures and Implementation Strategy

## 1. Introduction

This document outlines the architectural design for MARL environment for ARC-AGI
dataset, leveraging the JAX library for high-performance numerical computation
and the JaxMARL framework for multi-agent interactions.

This design has been updated to incorporate a sophisticated **selection and
manipulation mechanism**. Agents can now define a selection mask—highlighting
arbitrary groups of pixels—and then apply powerful manipulation actions to that
selection. This design has been updated to incorporate a sophisticated
**"Commit-and-Resolve"** cycle. Agents work on private scratchpads, explicitly
commit their proposed changes backed by reasoned hypotheses, and the environment
intelligently resolves conflicts based on the collective consensus (votes and
confidence) of the group. This provides a powerful and structured framework for
emergent collaboration.

Here is a step-by-step explanation of the overall flow, showing how an agent's
private work, a shared hypothesis, and a final committed change all connect.

Think of the process as a structured debate among agents, where they have to
justify their actions before they can modify the final answer.

### The Core Idea: Justify Before You Change

The old model was like multiple people trying to draw on the same canvas at
once—chaotic. The new model is like a team of architects working on a blueprint:

1. Each architect works on their own copy of the plan (**Private Scratchpad**).
2. When an architect has an idea (e.g., "move this wall"), they formally propose
   it and explain why (**Make Hypothesis**).
3. The other architects review the proposal and vote on it (**Vote on
   Hypothesis**).
4. Only the changes backed by proposals that have strong team support (high
   votes) are actually merged into the master blueprint
   (**Commit-and-Resolve**).

---

### Step-by-Step Flow of an Agent's Thought Process

Let's imagine two agents, Agent A and Agent B, working on a task.

#### **Phase 1: Private Ideation (Happening on the Scratchpad)**

1. **Observation:** Agent A looks at the `committed_output_grid` (the current
   official solution) and the task's training examples. It notices a pattern
   where blue objects seem to move down by one square.
2. **Private Work:** Agent A decides to test this idea. It performs a sequence
   of actions that **only affect its own private workspace**:
   - **`ACTION_SELECT_OBJECT_BY_COLOR`**: Agent A selects a blue object on its
     `agent_scratchpad_grids[A]`. This updates `agent_selection_masks[A]`.
   - **`ACTION_MOVE_SELECTION`**: Agent A moves the selected pixels down by one
     square on its `agent_scratchpad_grids[A]`.

At this point, Agent A's scratchpad now contains a _draft_ of what it thinks the
solution should be. The shared `committed_output_grid` is still untouched. Agent
B is completely unaware of Agent A's draft.

#### **Phase 2: Proposing and Justifying (Making the Hypothesis)**

3. **Making it Official:** Agent A is confident in its change. To get it into
   the final solution, it needs to convince the others. It takes the
   **`ACTION_MAKE_PROPOSAL`** action.
4. **The Hypothesis:** This action creates a new entry on the shared hypothesis
   blackboard. For example:
   - `Hypothesis #5:`
     - `agent_id`: A
     - `proposal_type`: "MOVE_RULE"
     - `proposal_data`: `[color=blue, dy=1, dx=0]`
     - `confidence`: 0.9
     - `vote_count`: 1 (the proposer's own vote)
     - `active_mask`: True

Now, this explicit, structured idea—"I propose a rule to move blue things down
by one"—is visible to everyone.

#### **Phase 3: Collaboration and Consensus (Voting)**

5. **Peer Review:** In the next step, Agent B sees `Hypothesis #5` in its
   observation. It can now evaluate this rule. It might check if this rule
   applies to the training examples or if it makes sense with its own ideas.
6. **Agreement:** Agent B agrees with the rule. It takes the
   **`ACTION_VOTE_HYPOTHESIS`** action with `params = [5, 1]` (targeting
   hypothesis #5 with a +1 vote).
7. **Consensus Builds:** The `vote_count` for `Hypothesis #5` is now **2**. It
   has strong support.

#### **Phase 4: The "Commit-and-Resolve" Step**

8. **Committing the Change:** Agent A (or Agent B, if it also applied the rule
   to its scratchpad) decides it's time to merge the change into the master
   solution. It takes the **`ACTION_COMMIT_CHANGES`** action. The most important
   parameter is the justification: `params = [5]` (I am committing the changes
   on my scratchpad, and I am justifying it with `Hypothesis #5`).
9. **Intelligent Conflict Resolution:** The environment's `_resolve_commits`
   function now runs.
   - It sees Agent A made a commit backed by `Hypothesis #5`, which has a vote
     count of 2. It calculates a "strength" for this commit (e.g.,
     `strength = vote_count + confidence = 2 + 0.9 = 2.9`).
   - Let's say another agent, Agent C, simultaneously tried to commit a change
     to the same pixels, but its change was backed by a different hypothesis
     with only 1 vote. Its strength would be lower.
   - For every pixel on the grid, the environment asks: "Of all the agents who
     want to change this pixel, which one has the commit backed by the strongest
     hypothesis?"
   - Agent A's commit wins because its backing hypothesis has higher vote count.
     The changes from Agent A's `scratchpad_grid` are copied to the shared
     `committed_output_grid`. Agent C's conflicting changes are ignored.

### Summary of the Flow

This cycle allows for a clear and powerful workflow:

**Private Work (`scratchpad`) -> Public Justification (`hypothesis`) -> Team
Consensus (`votes`) -> Intelligent Merge (`commit/resolve`)**

Hypotheses are the critical link. They are no longer just abstract ideas; they
are the **explicit justification** for any change to the shared solution and the
**primary mechanism for resolving conflicts**. The agent with the most popular,
well-supported idea wins the right to modify the final grid.

## 2. Core Data Structures for JaxARC: JAX Pytrees (`src/jax_arc/types.py`)

The foundation of a JAX-based environment lies in its data structures. For
JaxARC, these are defined as `chex.dataclass` Pytrees, ensuring seamless
integration with JAX's function transformations (e.g., `jax.jit`, `jax.vmap`)
and facilitating type safety. These Pytrees will reside in
`src/jax_arc/types.py`.

### 2.1. Preamble: Pre-allocation Constants and `chex.dataclass`

JAX achieves high performance, particularly through its JIT compiler, by
operating on arrays with static, predetermined shapes. To accommodate the
variable dimensions inherent in ARC tasks (e.g., grid sizes, number of training
examples), a strategy of pre-allocation and padding is employed. This involves
defining maximum dimensions for various data elements. These constants are
crucial for ensuring that all JAX arrays within the Pytrees maintain static
shapes, a prerequisite for efficient XLA compilation

The following constants, configurable via Hydra, establish these maximums:

```python
# src/jax_arc/types.py
import chex
import jax.numpy as jnp

# Maximums for pre-allocation, will be configured by Hydra
MAX_GRID_H, MAX_GRID_W = 30, 30
MAX_TRAIN_PAIRS = 5
MAX_TEST_PAIRS = 5
MAX_HYPOTHESES = 32
MAX_ACTION_PARAMS = 10 # Max number of parameters for a single action
MAX_PROPOSAL_DATA_DIM = 10
```

- `MAX_GRID_H`, `MAX_GRID_W`: Define the maximum height and width (e.g., 30x30)
  to which all ARC grids will be padded
- `MAX_TRAIN_PAIRS`: Specifies the maximum number of training examples an ARC
  task can have; tasks with fewer examples will have their example arrays padded
  up to this limit
- `MAX_TEST_PAIRS`: Specifies the maximum number of test examples an ARC task
  can have. Tasks with fewer test pairs will be padded.
- `MAX_HYPOTHESES`: The maximum number of concurrent hypotheses agents can
  propose and store in the environment state
- `MAX_ACTION_PARAMS`: The maximum dimension for the data payload associated
  with a single agent action, ensuring a static shape for the action parameters
  array.
- `MAX_PROPOSAL_DATA_DIM`: The maximum dimension for the data payload associated
  with a single hypothesis, ensuring static sizing for hypothesis data arrays.

### 2.2. `ParsedTaskData` Pytree

The `ParsedTaskData` Pytree serves as a standardized, JAX-compatible container
for a single, fully preprocessed ARC task. It acts as the data contract between
the ARC data parser and the environment's `reset` method, ensuring that the
environment receives data in a consistent format irrespective of the raw
dataset's original structure.

```python
@chex.dataclass
class ParsedTaskData:
    # Padded JAX arrays for 'train' pairs
    input_grids_examples: jnp.ndarray  # Shape: (MAX_TRAIN_PAIRS, MAX_GRID_H, MAX_GRID_W)
    input_masks_examples: jnp.ndarray  # Shape: (MAX_TRAIN_PAIRS, MAX_GRID_H, MAX_GRID_W)
    output_grids_examples: jnp.ndarray # Shape: (MAX_TRAIN_PAIRS, MAX_GRID_H, MAX_GRID_W)
    output_masks_examples: jnp.ndarray # Shape: (MAX_TRAIN_PAIRS, MAX_GRID_H, MAX_GRID_W)
    num_train_pairs: int # Actual number of training pairs for this task

    # Padded JAX arrays for ALL 'test' pairs in the task
    test_input_grids: jnp.ndarray      # Shape: (MAX_TEST_PAIRS, MAX_GRID_H, MAX_GRID_W)
    test_input_masks: jnp.ndarray      # Shape: (MAX_TEST_PAIRS, MAX_GRID_H, MAX_GRID_W)
    true_test_output_grids: jnp.ndarray  # Shape: (MAX_TEST_PAIRS, MAX_GRID_H, MAX_GRID_W)
    true_test_output_masks: jnp.ndarray  # Shape: (MAX_TEST_PAIRS, MAX_GRID_H, MAX_GRID_W)
    num_test_pairs: int # Actual number of test pairs for this task
```

Key fields include:

- `input_grids_examples`, `output_grids_examples`: These store the JAX arrays
  for the input and output grids of the training pairs. They are padded along
  the first dimension up to `MAX_TRAIN_PAIRS` and along the spatial dimensions
  up to `(MAX_GRID_H, MAX_GRID_W)`.

- `input_masks_examples`, `output_masks_examples`: Boolean masks corresponding
  to the training pair grids, indicating valid data cells versus padded cells.

- `num_train_pairs`: An integer storing the actual number of training pairs for
  the task, crucial for correctly processing the examples despite the padding in
  the arrays.

- `test_input_grids`, `true_test_output_grids`: The JAX arrays for the single
  test input grid and its corresponding ground truth output grid, both padded to
  `(MAX_TEST_PAIRS, MAX_GRID_H, MAX_GRID_W)`.

- `test_input_masks`, `true_test_output_masks`: Boolean masks for the test input
  and output grids.

The structure of `ParsedTaskData` is detailed in Table 1.

Table 1: ParsedTaskData Pytree Field Descriptions

| Field Name                | JAX Data Type (Example Shape for ARC)                         | Description/Purpose                                                                                                          |
| ------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `input_grids_examples`    | `jnp.ndarray` (`MAX_TRAIN_PAIRS`, `MAX_GRID_H`, `MAX_GRID_W`) | Padded input grids from all training pairs of the current ARC task. Used for rule inference.                                 |
| `input_masks_examples`    | `jnp.ndarray` (`MAX_TRAIN_PAIRS`, `MAX_GRID_H`, `MAX_GRID_W`) | Boolean masks for `input_grids_examples`, indicating valid data areas.                                                       |
| `output_grids_examples`   | `jnp.ndarray` (`MAX_TRAIN_PAIRS`, `MAX_GRID_H`, `MAX_GRID_W`) | Padded output grids from all training pairs. Used for rule inference.                                                        |
| `output_masks_examples`   | `jnp.ndarray` (`MAX_TRAIN_PAIRS`, `MAX_GRID_H`, `MAX_GRID_W`) | Boolean masks for `output_grids_examples`.                                                                                   |
| `num_train_pairs`         | `int`                                                         | Actual number of training pairs for this task.                                                                               |
| `current_test_input_grid` | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                    | The specific padded input grid for the single test case agents are currently solving.                                        |
| `current_test_input_mask` | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                    | Boolean mask for `current_test_input_grid`.                                                                                  |
| `true_test_output_grid`   | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                    | Ground truth solution (padded) for `current_test_input_grid`. Hidden from agents during solving; used for evaluation/reward. |
| `true_test_output_mask`   | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                    | Boolean mask indicating the valid area of `true_test_output_grid`.                                                           |

### 2.3. `Hypothesis` Pytree

The `Hypothesis` Pytree represents a single, structured proposal made by an
agent concerning an aspect of the ARC puzzle's solution, such as output grid
dimensions or color palettes. This structure is fundamental for the
collaborative reasoning mechanism envisioned, where agents share and debate
ideas.

Python

```
@chex.dataclass
class Hypothesis:
    agent_id: int
    proposal_type: int  # Enum: e.g., 0=GRID_SIZE, 1=COLOR_PALETTE
    proposal_data: jnp.ndarray # Padded to MAX_PROPOSAL_DATA_DIM
    confidence: float
    vote_count: int
```

Key fields include:

- `agent_id`: An integer identifying the agent that originated this hypothesis.
- `proposal_type`: An integer (intended to be an enumeration) indicating the
  nature of the proposal (e.g., output grid dimensions, color palette, a
  specific transformation rule). This allows for diverse agent specializations.
- `proposal_data`: A JAX array containing the actual data of the proposal. Its
  shape and content vary depending on `proposal_type` and it is padded to
  `MAX_PROPOSAL_DATA_DIM` to ensure a static shape for JAX compatibility. For
  instance, a grid dimension proposal might be a `(2,)` array `[height, width]`.
- `confidence`: A float (e.g., between 0.0 and 1.0) indicating the proposing
  agent's confidence in this hypothesis.
- `vote_count`: An integer tracking the support this hypothesis has received
  from other agents.

The structure of `Hypothesis` is detailed in Table 2.

**Table 2: Hypothesis Pytree Field Descriptions**

| Field Name      | Data Type                                         | Description/Purpose                                                                                                   |
| --------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `agent_id`      | `int`                                             | Identifier of the agent that originated the hypothesis.                                                               |
| `proposal_type` | `int`                                             | Enum-like integer specifying the nature of the proposal (e.g., `0` for grid size, `1` for color palette).             |
| `proposal_data` | `jnp.ndarray` (shape: `(MAX_PROPOSAL_DATA_DIM,)`) | Padded JAX array holding the content of the proposal (e.g., `[height, width]`). Padding ensures static shape for JAX. |
| `confidence`    | `float`                                           | The proposing agent's confidence in the hypothesis (e.g., 0.0 to 1.0).                                                |
| `vote_count`    | `int`                                             | Number of votes (or cumulative support) this hypothesis has received from other agents.                               |

### 2.4. `State` Pytree

The `State` Pytree is arguably the most critical data structure, encapsulating
the complete state of the MARL environment at any given timestep. Its design
must be JAX-compatible, meaning all JAX array attributes must have static shapes
for efficient JIT compilation. It comprehensively holds ARC task data,
facilitates collaborative reasoning mechanisms (like shared hypotheses and a
workspace), and tracks environment metadata.

Python

```python
@chex.dataclass
class State:
    # Static ARC Task Data, nested for clarity and organization.
    task: ParsedTaskData

    # Per-agent private workspaces
    agent_scratchpad_grids: jnp.ndarray  # Shape: (num_agents, MAX_GRID_H, MAX_GRID_W)
    agent_selection_masks: jnp.ndarray   # Shape: (num_agents, MAX_GRID_H, MAX_GRID_W)

    # The final, shared grid after resolving commits from the previous step.
    committed_output_grid: jnp.ndarray   # Shape: (MAX_GRID_H, MAX_GRID_W)
    committed_output_mask: jnp.ndarray   # Shape: (MAX_GRID_H, MAX_GRID_W)

    # Hypothesis storage (fixed-size arrays with active mask)
    agent_hypotheses_ids: jnp.ndarray           # (MAX_HYPOTHESES,)
    agent_hypotheses_types: jnp.ndarray         # (MAX_HYPOTHESES,)
    agent_hypotheses_data: jnp.ndarray          # (MAX_HYPOTHESES, MAX_ACTION_PARAMS)
    agent_hypotheses_confidence: jnp.ndarray    # (MAX_HYPOTHESES,)
    agent_hypotheses_votes: jnp.ndarray         # (MAX_HYPOTHESES,)
    agent_hypotheses_active_mask: jnp.ndarray   # (MAX_HYPOTHESES,) boolean

    # Metadata
    current_test_case_idx: int # New: Index of the active test case
    step_count: int
    terminal: bool
    key: chex.PRNGKey
```

Key fields include:

- **ARC Task Data:** All fields from `ParsedTaskData` are directly included to
  represent the current ARC puzzle.
- `working_output_grid`, `working_output_mask`: These represent the shared grid
  that agents collaboratively construct or modify. They are initialized based on
  the task and consensus.
- `agent_hypotheses_*` arrays: A set of fixed-size JAX arrays for storing active
  agent proposals. The `agent_hypotheses_active_mask` (a boolean array) is
  crucial: it indicates which slots in these arrays are currently occupied by an
  active hypothesis. This pattern allows for a potentially variable number of
  active hypotheses while maintaining static array shapes for all
  `agent_hypotheses_*` fields, a key requirement for JAX's JIT compilation.
- `step_count`, `terminal`: Standard RL environment metadata tracking the
  current step and episode termination status.
- `key`: A JAX PRNGKey for managing any stochasticity within the environment's
  state transitions, ensuring reproducibility.

Table 3: State Pytree Field Descriptions

| Field Name                     | JAX Data Type (Example Shape)                             | Description/Purpose in ARC Context                                                                     |
| ------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `task`                         | `ParsedTaskData`                                          |                                                                                                        |
| `working_output_grid`          | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                | The grid agents collaboratively construct. Initialized based on task/consensus.                        |
| `working_output_mask`          | `jnp.ndarray` (`MAX_GRID_H`, `MAX_GRID_W`)                | Mask for `working_output_grid`, potentially dynamic if grid size changes by consensus.                 |
| `agent_hypotheses_ids`         | `jnp.ndarray` (`MAX_HYPOTHESES,`)                         | Stores `agent_id` for each active hypothesis.                                                          |
| `agent_hypotheses_types`       | `jnp.ndarray` (`MAX_HYPOTHESES,`)                         | Stores `proposal_type` for each active hypothesis.                                                     |
| `agent_hypotheses_data`        | `jnp.ndarray` (`MAX_HYPOTHESES`, `MAX_PROPOSAL_DATA_DIM`) | Stores `proposal_data` for each active hypothesis.                                                     |
| `agent_hypotheses_confidence`  | `jnp.ndarray` (`MAX_HYPOTHESES,`)                         | Stores `confidence` for each active hypothesis.                                                        |
| `agent_hypotheses_votes`       | `jnp.ndarray` (`MAX_HYPOTHESES,`)                         | Stores `vote_count` for each active hypothesis.                                                        |
| `agent_hypotheses_active_mask` | `jnp.ndarray` (`MAX_HYPOTHESES,`) (boolean)               | Boolean mask indicating which hypothesis slots are currently active/used.                              |
| `consensus_grid_dimensions`    | `jnp.ndarray` (`2,`)                                      | Example field: stores agreed-upon output grid dimensions `[height, width]`. Initialized to `[-1, -1]`. |
| `step_count`                   | `int`                                                     | Current step number in the episode.                                                                    |
| `terminal`                     | `bool`                                                    | `True` if the episode has ended (solved or max steps reached).                                         |
| `key`                          | `chex.PRNGKey`                                            | JAX PRNG key for any stochastic operations within the environment step/reset that modify state.        |

### 2.5. `Action` Pytree

To address the need for concrete grid manipulation capabilities inspired by
environments like ARCLE [2], the `Action` Pytree is redesigned. It now supports
a richer set of operations, from low-level pixel editing to high-level
proposals, while maintaining a static structure for JAX compatibility.

**Note**: Later on, we'll have to restructure it a bit, as we also need to
figure out a way

```python
# Conceptual action type constants
# --- Selection Actions ---
ACTION_CLEAR_SELECTION = 0
ACTION_SELECT_PIXEL = 1
# … other selection actions …
# --- Manipulation Actions (on private scratchpad) ---
ACTION_CHANGE_COLOR = 4
ACTION_MOVE_SELECTION = 5
# --- Commit and Reasoning Actions ---
ACTION_COMMIT_CHANGES = 6
ACTION_MAKE_PROPOSAL = 7
ACTION_VOTE_HYPOTHESIS = 8
ACTION_SUBMIT_SOLUTION = 9
NUM_ACTION_TYPES = 10

@chex.dataclass
class Action:
    """
    Represents a single agent's action, combining both concrete grid
    manipulation and abstract reasoning proposals into a unified structure.
    """
    action_type: jnp.ndarray  # Shape (), dtype=int32. The category of action to perform.
    params: jnp.ndarray       # Shape (MAX_ACTION_PARAMS,), dtype=int32. A parameter vector.
```

The `params` vector is a versatile, fixed-size array whose interpretation
depends on the `action_type`. This design allows for diverse, parameterized
actions within a static Pytree structure suitable for JAX.

**Table 1: `Action.params` Interpretation by `action_type`**

| `action_type`                  | Action Name              | `params` Index Interpretation              | Effect On…                      | Description                                                                                                                       |
| ------------------------------ | ------------------------ | ------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Selection Actions**          |                          |                                            |                                 |                                                                                                                                   |
| 0                              | `CLEAR_SELECTION`        | (none)                                     | `selection_mask`                | Resets the selection mask to all `False`.                                                                                         |
| 1                              | `SELECT_PIXEL`           | `[0]: y`, `[1]: x`                         | `selection_mask`                | Selects a single pixel, clearing any previous selection.                                                                          |
| 2                              | `SELECT_RECT`            | `[0]: y1`, `[1]: x1`, `[2]: y2`, `[3]: x2` | `selection_mask`                | Selects all pixels within a bounding box.                                                                                         |
| 3                              | `SELECT_OBJECT_BY_COLOR` | `[0]: y`, `[1]: x`                         | `selection_mask`                | "Magic wand" tool. Selects all contiguous pixels of the same color as the one at `(y, x)`.                                        |
| **Manipulation Actions**       |                          |                                            |                                 |                                                                                                                                   |
| 4                              | `CHANGE_COLOR`           | `[0]: color` (0-9)                         | _Agent's own_ `scratchpad_grid` | Changes color of selected pixels on the agent's **private scratchpad**.                                                           |
| 5                              | `MOVE_SELECTION`         | `[0]: dy`, `[1]: dx`                       | _Agent's own_ `scratchpad_grid` | Moves selected pixels on the agent's **private scratchpad**.                                                                      |
| **Commit & Reasoning Actions** |                          |                                            |                                 |                                                                                                                                   |
| 6                              | `COMMIT_CHANGES`         | `[0]: hyp_idx`                             | (Triggers commit resolution)    | Proposes to merge changes from the agent's scratchpad into the shared `committed_output_grid`, justified by hypothesis `hyp_idx`. |
| 7                              | `MAKE_PROPOSAL`          | `[0]: type`, etc.                          | `agent_hypotheses_*`            | Adds a new hypothesis to the shared blackboard.                                                                                   |
| 8                              | `VOTE_HYPOTHESIS`        | `[0]: hyp_idx`, `[1]: vote`                | `agent_hypotheses_votes`        | Casts a vote on an existing hypothesis.                                                                                           |
| 9                              | `SUBMIT_SOLUTION`        | (none)                                     | (Triggers terminal check)       | Signals that the agent believes the current `committed_output_grid` is the final solution for the current test case.              |

_Note: More complex actions from ARCLE, like `FloodFill` or `CopyPaste`, can be
added to this structure by defining new action types and parameter mappings.
Their implementation within `_process_actions` would require more complex JAX
logic, potentially involving `jax.lax.scan`._

### 2.6. Elaboration on Pytree Design Principles

The design of these core Pytrees (`ParsedTaskData`, `Hypothesis`, `State`,
`Action`) is fundamentally shaped by the operational requirements of JAX and the
collaborative nature of the ARC problem-solving task.

A primary driver is the JAX JIT compiler's preference for static array shapes.
ARC tasks inherently involve variable grid sizes, a varying number of training
examples, and a dynamic number of agent-generated hypotheses. To reconcile this
with JAX's need for static shapes to achieve optimal performance, strategies
like padding and active masks are employed ubiquitously. For instance, all grids
are padded to `(MAX_GRID_H, MAX_GRID_W)`, training example arrays are padded to
`MAX_TRAIN_PAIRS`, and hypothesis data arrays are padded to
`MAX_PROPOSAL_DATA_DIM`. The `agent_hypotheses_active_mask` in the `State`
Pytree allows the system to manage a variable number of active hypotheses within
fixed-size arrays. This prioritization of static shapes is a deliberate
trade-off, favoring computational performance and JAX compatibility over the
dynamic flexibility common in non-JAX environments

The data flow from raw task files to the environment's active state is also
structured for clarity and modularity. The `ArcAgiParser` (or any concrete
parser) is responsible for transforming raw JSON data into the standardized
`ParsedTaskData` Pytree. This Pytree then serves as a clean input to the
environment's `reset` function, which uses it to initialize the task-specific
portions of the main `State` Pytree. This separation of concerns—parsing and
initial preprocessing versus environment state management and dynamics—enhances
maintainability and testability.

Furthermore, the `Hypothesis` Pytree and the `agent_hypotheses_*` fields within
the `State` Pytree are not merely data containers; they are the fundamental
primitives that enable the planned collaborative reasoning among agents. The
structured nature of `Hypothesis` (including `proposal_type`, `proposal_data`,
`confidence`, and `vote_count`) provides a rich medium for agents to formulate,
share, and collectively evaluate diverse ideas about the ARC puzzle's solution.
This design directly supports the envisioned multi-faceted proposal and
consensus mechanisms.

## 3. Abstract Base Classes for Modularity (`src/jax_arc/base/`)

To foster modularity and facilitate the support of diverse ARC-like datasets
(e.g., ARC-AGI-1, ConceptARC) within the JaxARC project, abstract base classes
(ABCs) are employed. These are defined in the `src/jax_arc/base/` directory.
ABCs establish core API contracts for both ARC MARL environments and their
associated data parsers.

### 3.1. `ArcDataParserBase` (`base_parser.py`)

The `ArcDataParserBase` class defines the standard interface for all data
parsers within the JaxARC project. Concrete parsers, such as the user's existing
`ArcAgiParser`, will inherit from this base class. Its primary role is to
abstract the process of loading raw task data (typically from JSON files) and
transforming it into JAX-compatible Pytrees (`ParsedTaskData`) that the
environment can consume

Python

```python
# src/jax_arc/base/base_parser.py
from abc import ABC, abstractmethod
import chex
from omegaconf import DictConfig
from src.jax_arc.types import ParsedTaskData # Assuming types.py is in src.jax_arc

class ArcParserBase(ABC):
    """
    Abstract base class for all ARC dataset parsers.

    This class defines a standard interface for loading and parsing ARC tasks,
    while allowing concrete implementations to handle different file structures
    (e.g., all tasks in one file vs. one task per file).

    It provides concrete helper methods for parsing common structures like grids
    and pairs, which are shared across all dataset formats.
    """

    def __init__(self, dataset_path: str | Path):
        """
        Initializes the parser.

        Args:
            dataset_path: The root path to the dataset directory.
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            msg = f"Dataset path not found or not a directory: {self.dataset_path}"
            raise FileNotFoundError(msg)

        # Each concrete parser must populate this list with its task identifiers
        # (e.g., task IDs from a JSON file or paths to individual task files).
        self.task_identifiers: List[str] = self._discover_tasks()

        if not self.task_identifiers:
            logger.warning(f"No task identifiers found for parser {self.__class__.__name__} at path {self.dataset_path}")

    @abstractmethod
    def _discover_tasks(self) -> List[str]:
        """
        Scans the dataset path to find all available task identifiers.

        This method must be implemented by concrete subclasses. For example, it
        could read task IDs from a central JSON file or find all `.json` files
        in a directory.

        Returns:
            A list of unique strings identifying each task.
        """
        pass

    @abstractmethod
    def load_and_parse_task(self, task_identifier: str) -> ArcTask:
        """
        Loads the raw data for a single task and parses it into an ArcTask.

        This is the core abstract method that defines how a specific dataset
        format is read and processed.

        Args:
            task_identifier: The unique identifier for the task (e.g., '007bbfb7'
                             or 'path/to/task.json').

        Returns:
            A fully populated ArcTask object.
        """
        pass

    def get_random_task(self, key: chex.PRNGKey) -> ArcTask:
        """
        Selects a random task from the dataset and returns it.

        This is the primary public method used by the environment to sample tasks.

        Args:
            key: A JAX PRNGKey for reproducible random selection.

        Returns:
            A randomly selected ArcTask object.
        """
        if not self.task_identifiers:
            msg = "Cannot get a random task because no task identifiers were loaded."
            raise RuntimeError(msg)

        task_id = jax.random.choice(self.task_identifiers)

        logger.info(f"Loading random task: {task_id}")
        return self.load_and_parse_task(task_id)

    def get_all_tasks(self) -> Dict[str, ArcTask]:
        """
        Loads and parses all tasks from the dataset.

        This method iterates through all discovered task identifiers and
        loads each one, returning them in a dictionary. This is useful for
        evaluation or analysis over the entire dataset.

        Returns:
            A dictionary mapping task identifiers to their corresponding ArcTask objects.
        """
        if not self.task_identifiers:
            logger.warning("No task identifiers found, returning an empty dictionary.")
            return {}

        logger.info(f"Loading all {len(self.task_identifiers)} tasks…")
        all_tasks = {
            task_identifier: self.load_and_parse_task(task_identifier)
            for task_identifier in self.task_identifiers
        }
        logger.info("Finished loading all tasks.")
        return all_tasks

    # ----------------------------------------------------------------------------
    # Concrete Helper Methods (Shared across all parsers)
    # ----------------------------------------------------------------------------
    def _parse_grid_json(self, grid_json: list[list[int]]) -> Grid:
        """Converts a JSON representation of a grid to a Grid object."""
        if (
            not grid_json
            or not isinstance(grid_json, list)
            or not all(isinstance(row, list) for row in grid_json)
            or (grid_json and any(len(row) != len(grid_json[0]) for row in grid_json))
        ):
            msg = "Grid JSON must be a list of lists with consistent row lengths."
            raise ValueError(msg)

        if not all(all(isinstance(cell, int) for cell in row) for row in grid_json):
            msg = "Grid cells must be integers."
            raise ValueError(msg)
        return Grid(array=jnp.array(grid_json, dtype=jnp.int32))

    def _parse_pair_json(
        self, pair_json: dict[str, Any], is_train_pair: bool
    ) -> TaskPair:
        """Converts a JSON representation of an input-output pair to a TaskPair object."""
        if "input" not in pair_json:
            msg = "Task pair JSON must contain an 'input' key."
            raise ValueError(msg)

        input_grid = self._parse_grid_json(pair_json["input"])
        output_grid: Grid | None = None

        if is_train_pair:
            if "output" not in pair_json:
                msg = "Training task pair JSON must contain an 'output' key."
                raise ValueError(msg)
            output_grid = self._parse_grid_json(pair_json["output"])
        elif "output" in pair_json and not is_train_pair:
            logger.warning(
                "Test input pair contains an 'output' key. "
                "This will be ignored, as test outputs are loaded separately."
            )

        return TaskPair(input=input_grid, output=output_grid)

class ArcDataParserBase(ABC):
    def __init__(self, cfg: DictConfig, max_grid_size_h: int, max_grid_size_w: int):
        self.parser_cfg = cfg # Specific configuration for the parser instance
        self.max_grid_size_h = max_grid_size_h
        self.max_grid_size_w = max_grid_size_w
        # Example: self.dataset_path = self.parser_cfg.get("dataset_path", None)
        # Concrete parsers will use this path to find task files.

    @abstractmethod
    def load_task_file(self, task_file_path: str) -> any:
        """Loads a single ARC task file from the given path.
        Returns raw data structure (e.g., dict for JSON).
        """
        pass

    @abstractmethod
    def preprocess_task_data(self, raw_task_data: any, key: chex.PRNGKey) -> ParsedTaskData:
        """Preprocesses raw task data into JAX-compatible ParsedTaskData.
        Includes padding grids, creating masks.
        The PRNGKey can be used if any stochastic preprocessing is needed.
        """
        pass

    @abstractmethod
    def get_random_task(self, key: chex.PRNGKey) -> ParsedTaskData:
        """Loads and preprocesses a random task from the dataset.
        The PRNGKey is used for randomly selecting a task.
        """
        pass
```

**Analysis of Abstract Methods:**

- `__init__(self, cfg: DictConfig, max_grid_size_h: int, max_grid_size_w: int)`:
  The constructor takes a Hydra `DictConfig` object (for parser-specific
  settings like `dataset_path`) and the maximum grid dimensions required for
  padding operations.

- `load_task_file(self, task_file_path: str) -> any`: This abstract method
  defines the contract for loading the raw content of a single task file.

- `preprocess_task_data(self, raw_task_data: any, key: chex.PRNGKey) -> ParsedTaskData`:
  This is the core transformation method. It takes the raw data loaded by
  `load_task_file` and converts it into a `ParsedTaskData` Pytree. This involves
  converting list-based grids to JAX arrays, padding them to
  `max_grid_size_h/w`, and creating corresponding boolean masks. The `key` is
  provided for any stochastic preprocessing steps, though typically this stage
  is deterministic.

- `get_random_task(self, key: chex.PRNGKey) -> ParsedTaskData`: This method
  orchestrates the selection of a random task (using the provided `key`), loads
  its raw data via `load_task_file`, and then preprocesses it using
  `preprocess_task_data`, returning the final `ParsedTaskData` Pytree.

Table 5: ArcDataParserBase Abstract Methods Summary

| Abstract Method        | Parameters                                                        | Return Type (Conceptual)                   | Purpose                                                                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `__init__`             | `cfg: DictConfig`, `max_grid_size_h: int`, `max_grid_size_w: int` | `None`                                     | Initialize parser with its specific Hydra configuration and maximum grid dimensions. Load dataset metadata (e.g., list of task files from `cfg.dataset_path`).                                         |
| `load_task_file`       | `task_file_path: str`                                             | Raw dataset-specific object (e.g., `dict`) | Load the content of a single task file (e.g., JSON data) from the specified path.                                                                                                                      |
| `preprocess_task_data` | `raw_task_data: any`, `key: chex.PRNGKey`                         | `ParsedTaskData` (JAX Pytree)              | Convert raw data to JAX arrays, pad grids to `max_grid_size_h/w`, create boolean masks, and structure into a `ParsedTaskData` Pytree suitable for the environment. `key` for stochastic preprocessing. |
| `get_random_task`      | `key: chex.PRNGKey`                                               | `ParsedTaskData` (JAX Pytree)              | Select a random task from the dataset using `key`, load it using `load_task_file`, and preprocess it using `preprocess_task_data`. This is the primary interface for the environment's `reset` method. |

#### Concrete Implementation Examples

```python
# ----------------------------------------------------------------------------
# 2. Your Refactored ArcAgiParser (for Kaggle-style datasets)
# ----------------------------------------------------------------------------
class ArcAgiParser(ArcParserBase):
    """
    Parses ARC-AGI datasets (Kaggle format) where all tasks are in a
    single challenge file and solutions are in a separate file.
    """
    def __init__(self, dataset_path: str | Path, challenges_filename: str, solutions_filename: str | None = None):
        super().__init__(dataset_path)
        self.challenges_file = self.dataset_path / challenges_filename
        self.solutions_file = self.dataset_path / solutions_filename if solutions_filename else None

        # Load all data into memory upfront
        self.challenges_data = self._load_json_file(self.challenges_file)
        self.solutions_data = self._load_json_file(self.solutions_file) if self.solutions_file else {}

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Helper to load a JSON file and handle errors."""
        if not file_path.exists():
            msg = f"Required file not found: {file_path}"
            raise FileNotFoundError(msg)
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {file_path}: {e}"
            raise ValueError(msg) from e

    def _discover_tasks(self) -> List[str]:
        """For this format, task identifiers are the keys in the challenges file."""
        challenges_data = self._load_json_file(self.dataset_path / "challenges.json") # Example filename
        return list(challenges_data.keys())

    def load_and_parse_task(self, task_id: str) -> ArcTask:
        """
        Implementation for the Kaggle format. It uses the pre-loaded data
        to assemble the task.
        """
        if task_id not in self.challenges_data:
            msg = f"Task ID '{task_id}' not found in {self.challenges_file}"
            raise KeyError(msg)

        task_json_content = self.challenges_data[task_id]

        # --- Parse training pairs (they are self-contained) ---
        train_pairs = [
            self._parse_pair_json(pair, is_train_pair=True)
            for pair in task_json_content.get("train", [])
        ]

        # --- Parse test pairs (input from challenges, output from solutions) ---
        test_inputs_json = task_json_content.get("test", [])
        solution_outputs_json = self.solutions_data.get(task_id, [])

        if self.solutions_data and task_id not in self.solutions_data:
             logger.warning(f"Task {task_id}: No solutions found in solutions file.")

        if len(test_inputs_json) != len(solution_outputs_json) and self.solutions_data:
            logger.warning(
                f"Task {task_id}: Mismatch between number of test inputs "
                f"({len(test_inputs_json)}) and solution outputs ({len(solution_outputs_json)})."
            )

        test_pairs: List[TaskPair] = []
        for i, test_input_pair_json in enumerate(test_inputs_json):
            input_grid = self._parse_grid_json(test_input_pair_json["input"])
            output_grid = None
            if i < len(solution_outputs_json):
                try:
                    output_grid = self._parse_grid_json(solution_outputs_json[i])
                except ValueError as e:
                    logger.warning(f"Task {task_id}, test pair {i}: Error parsing solution grid: {e}. Output will be None.")

            test_pairs.append(TaskPair(input=input_grid, output=output_grid))

        return ArcTask(task_id=task_id, train_pairs=train_pairs, test_pairs=test_pairs)

# ----------------------------------------------------------------------------
# 3. A Hypothetical Parser for a Different Dataset Format
# ----------------------------------------------------------------------------
class IndividualFileParser(ArcParserBase):
    """
    Parses an ARC-like dataset where each task is in its own separate JSON file.

    Assumes a directory structure like:
    /dataset_path/
        /tasks/
            task1.json
            task2.json
            …

    Each JSON file is expected to contain 'train' and 'test' keys, with
    test pairs including both 'input' and 'output'.
    """

    def _discover_tasks(self) -> List[str]:
        """For this format, task identifiers are the file paths themselves."""
        tasks_dir = self.dataset_path / "tasks"
        if not tasks_dir.is_dir():
            return []
        # Return the string representation of the full path for each task file.
        return [str(p) for p in tasks_dir.glob("*.json")]

    def load_and_parse_task(self, task_identifier: str) -> ArcTask:
        """
        Implementation for the individual file format. It loads a single file
        and parses it.
        """
        task_file = Path(task_identifier)
        if not task_file.exists():
            msg = f"Task file not found: {task_file}"
            raise FileNotFoundError(msg)

        try:
            task_json_content = json.loads(task_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {task_file}: {e}"
            raise ValueError(msg) from e

        task_id = task_file.stem  # Use the filename without extension as the ID

        train_pairs = [
            self._parse_pair_json(pair, is_train_pair=True)
            for pair in task_json_content.get("train", [])
        ]

        # In this format, test pairs are assumed to be complete.
        test_pairs = [
            self._parse_pair_json(pair, is_train_pair=True) # Re-use is_train_pair=True logic to get output
            for pair in task_json_content.get("test", [])
        ]

        return ArcTask(task_id=task_id, train_pairs=train_pairs, test_pairs=test_pairs)
```

### 3.2. `ArcMarlEnvBase` (`base_env.py`)

The `ArcMarlEnvBase` class defines the fundamental interface for all ARC-like
MARL environments developed within the JaxARC project. Crucially, it inherits
from `jaxmarl.environments.multi_agent_env.MultiAgentEnv`, ensuring seamless
integration with the broader JaxMARL ecosystem, and from `abc.ABC` to define
abstract methods

Python

```python
# src/jax_arc/base/base_env.py
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces # For type hinting spaces
from omegaconf import DictConfig
import chex
from src.jax_arc.types import State, Action # Assuming types.py is in src.jax_arc

class ArcMarlEnvBase(MultiAgentEnv, ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg # Store the full Hydra config for the environment
        self.env_name = cfg.get("env_name", "JaxARCBaseEnv")
        self.num_agents = cfg.num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Grid and Pytree pre-allocation parameters from config
        self.max_grid_size_h = cfg.max_grid_size # Assuming square, or use cfg.max_grid_h
        self.max_grid_size_w = cfg.max_grid_size # Assuming square, or use cfg.max_grid_w
        self.reward_config = cfg.reward_config
        self.consensus_config = cfg.consensus_config
        self.max_hypotheses = cfg.get("max_hypotheses", 32)
        self.max_proposal_data_dim = cfg.get("max_proposal_data_dim", 10)
        self.workspace_size = cfg.get("workspace_size", 100) # As per [1]
        self.workspace_feature_dim = cfg.get("workspace_feature_dim", 10) # As per [1]

        # Note: The data parser is instantiated in the concrete child class (e.g., ArcEnv)

    @property
    def default_params(self):
        """JaxMARL required property. Can provide default env parameters.
        Many are managed by Hydra, but max_steps_per_episode is common here.
        """
        return {"max_steps_per_episode": self.cfg.get("max_steps_per_episode", 200)}

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple: # obs, state
        """Resets the environment to an initial state with a new ARC task."""
        pass

    @abstractmethod
    def step(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]
             ) -> tuple: # obs, state, rewards, dones, info
        """Runs one timestep of the environment's dynamics."""
        pass

    @abstractmethod
    def get_obs(self, state: State) -> dict:
        """Returns the observation for each agent based on the current state."""
        pass

    @abstractmethod
    def observation_space(self, agent: str) -> spaces.Space:
        """Defines the observation space for an agent."""
        pass

    @abstractmethod
    def action_space(self, agent: str) -> spaces.Space:
        """Defines the action space for an agent."""
        pass

    # Optional: Common helper methods with base implementations or as abstract
    # def _process_actions(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> State: pass
    # def _update_consensus(self, key: chex.PRNGKey, state: State) -> State: pass
    # def _apply_consensus_to_grid(self, key: chex.PRNGKey, state: State) -> State: pass
    # def _calculate_rewards(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> dict: pass
    # def is_terminal(self, state: State) -> bool: pass
```

(Derived from 1 Section 10.2)

**Analysis of `ArcMarlEnvBase`:**

- `__init__(self, cfg: DictConfig)`: Initializes common environment parameters
  from the Hydra configuration object. This includes agent count, maximum grid
  dimensions, configurations for rewards and consensus mechanisms, and
  parameters for pre-allocating Pytree field sizes (e.g., `max_hypotheses`,
  `max_proposal_data_dim`, `workspace_size`).

- `default_params` (property): This is required by the JaxMARL `MultiAgentEnv`
  class. It can provide default environment parameters, such as
  `max_steps_per_episode`, which are typically also configurable via Hydra.

- Abstract Methods: `reset`, `step`, `get_obs`, `observation_space`, and
  `action_space` are declared as abstract. This mandates their implementation by
  any concrete ARC environment subclass, ensuring a consistent API that aligns
  with JaxMARL's expectations.

Table 6 summarizes the key elements of `ArcMarlEnvBase`.

**Table 6: ArcMarlEnvBase Abstract Methods and Key Properties Summary**

| Element             | Type                                                                              | Description/Purpose                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `__init__`          | Method                                                                            | Initializes common parameters from Hydra config (agent count, grid sizes, reward/consensus configs, Pytree pre-allocation sizes). |
| `default_params`    | Property                                                                          | JaxMARL required property, typically provides `max_steps_per_episode`.                                                            |
| `reset`             | Abstract Method (`key: chex.PRNGKey`) -> `tuple`                                  | Resets the environment to an initial state with a new ARC task, returning initial observations and state.                         |
| `step`              | Abstract Method (`key: chex.PRNGKey`, `state: State`, `actions: dict`) -> `tuple` | Runs one timestep, returning new observations, state, rewards, dones, and info.                                                   |
| `get_obs`           | Abstract Method (`state: State`) -> `dict`                                        | Returns the observation dictionary for all agents based on the current state.                                                     |
| `observation_space` | Abstract Method (`agent: str`) -> `spaces.Space`                                  | Defines the observation space for a given agent using `jaxmarl.environments.spaces`.                                              |
| `action_space`      | Abstract Method (`agent: str`) -> `spaces.Space`                                  | Defines the action space for a given agent using `jaxmarl.environments.spaces`.                                                   |

### 3.3. Elaboration on Base Class Design Principles

The adoption of these abstract base classes is central to achieving a modular
and extensible JaxARC system. `ArcMarlEnvBase`'s inheritance from
`jaxmarl.environments.multi_agent_env.MultiAgentEnv` is fundamental, as it
ensures that any concrete JaxARC environment will seamlessly integrate with the
broader JaxMARL ecosystem, including its algorithms, wrappers, and utility
functions. The abstract methods defined in `ArcMarlEnvBase` directly mirror the
API expected by JaxMARL components.

A key design pattern evident in both base classes is configuration-driven
initialization. Both `ArcDataParserBase` and `ArcMarlEnvBase` constructors
accept a Hydra `DictConfig` object. This allows critical parameters—such as the
number of agents, maximum grid sizes, paths to datasets, reward structures,
consensus rules, and even the dimensions for Pytree pre-allocation (like
`max_hypotheses`)—to be defined in external YAML configuration files rather than
being hardcoded. This approach significantly enhances the flexibility of the
system, making it well-suited for research and experimentation where parameters
are frequently tuned

Furthermore, the `ArcDataParserBase` establishes a clean separation between the
concerns of data loading and preprocessing, and the environment's core
simulation dynamics defined in `ArcMarlEnvBase`. A concrete environment (like
`ArcEnv`) will utilize a concrete parser (like `ArcAgiParser`), but its primary
logic for methods such as `step` and `reset` will depend on the _interface_
defined by `ArcDataParserBase` (i.e., the `ParsedTaskData` Pytree it returns),
not on the specific implementation details of how a particular dataset was
parsed. This decoupling is vital for easily incorporating new ARC-like datasets
in the future by simply creating new parser implementations.

## 4. Implementing the Concrete `ArcAgiEnv` (as `ArcEnv` in `src/jax_arc/envs/arc_env.py`)

The concrete environment class, referred to as `ArcEnv` in the primary
development guide 1 (and conceptually `ArcAgiEnv` for the ARC-AGI dataset), is
where the ARC task-solving dynamics are implemented. This class inherits from
`ArcMarlEnvBase` and provides concrete implementations for all its abstract
methods.

```python
# src/jax_arc/envs/arc_env.py
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import chex
from hydra.utils import instantiate

from src.jax_arc.base.base_env import ArcMarlEnvBase
# Assuming ArcAgiParser is the concrete parser implemented by the user,
# or ArcParser as per.[1] For generality, we'll use a type hint.
from src.jax_arc.base.base_parser import ArcDataParserBase
from src.jax_arc.types import State, Action, ParsedTaskData
from jaxmarl.environments import spaces

# Define constants for action/proposal types if not centrally defined
# These would ideally be enums or part of the config
ACTION_NO_OP = 0
# … (other constants)
NUM_ACTION_TYPES = 6
#… other action types

GRID_SIZE_PROPOSAL_TYPE = 0
#… other proposal types

class ArcEnv(ArcMarlEnvBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg) # Initializes common attributes from ArcMarlEnvBase

        # Instantiate the specific data parser using Hydra configuration
        # The parser_config within cfg.environment specifies the _target_ class for the parser
        # and its specific parameters (e.g., dataset_path).
        # max_grid_size_h and max_grid_size_w are passed from self (initialized in super().__init__)
        self.parser: ArcDataParserBase = instantiate(
            cfg.parser_config,
            max_grid_size_h=self.max_grid_size_h,
            max_grid_size_w=self.max_grid_size_w
        )

        # Example: Number of action types could be loaded from config or defined
        self.num_action_types = cfg.get("num_action_types", 2) # e.g., PROPOSE, VOTE
        self.num_proposal_types = cfg.get("num_proposal_types", 1) # e.g., GRID_SIZE

    def reset(self, key: chex.PRNGKey) -> tuple:
        key_task_sample, key_state_init = jax.random.split(key)

        # 1. Sample a new ARC task using the instantiated parser
        task_data: ParsedTaskData = self.parser.get_random_task(key_task_sample)

        # Initialize all grids (private and committed) to a default state
        initial_grids = jnp.zeros((self.num_agents, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32)
        initial_committed_grid = jnp.zeros((self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32)
        initial_committed_mask = jnp.zeros((self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_)

        # 2. Initialize the State Pytree
        state = State(
            # Nest the entire ParsedTaskData object
            task=task_data,

            agent_scratchpad_grids=initial_grids,
            agent_selection_masks=jnp.zeros_like(initial_grids, dtype=jnp.bool_),
            committed_output_grid=initial_committed_grid,
            committed_output_mask=initial_committed_mask, # Initialize the new mask

            # Hypothesis Storage Initialization
            agent_hypotheses_ids=jnp.full((self.max_hypotheses,), -1, dtype=jnp.int32),
            agent_hypotheses_types=jnp.full((self.max_hypotheses,), -1, dtype=jnp.int32),
            agent_hypotheses_data=jnp.zeros((self.max_hypotheses, self.max_proposal_data_dim), dtype=jnp.int32), # Or float, per use case
            agent_hypotheses_confidence=jnp.zeros((self.max_hypotheses,), dtype=jnp.float32),
            agent_hypotheses_votes=jnp.zeros((self.max_hypotheses,), dtype=jnp.int32),
            agent_hypotheses_active_mask=jnp.zeros((self.max_hypotheses,), dtype=jnp.bool_),

            # Consensus State Initialization
            consensus_grid_dimensions=jnp.array([-1, -1], dtype=jnp.int32), # Default: no consensus

            # Metadata Initialization
            current_test_case_idx=0, # Start with the first test case
            step_count=0,
            terminal=False,
            key=key_state_init
        )

        obs = self.get_obs(state)
        return obs, state

    def step(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]
             ) -> tuple:
        key_process, key_consensus, key_apply, key_reward, key_next_state = jax.random.split(key, 5)

        # 1. Process local actions (selection/manipulation on private scratchpads)
        state_after_actions = self._process_actions(key_process, state, actions)

        # 2. Resolve commits and update the single shared grid
        state_after_resolve = self._resolve_commits(key_resolve, state_after_actions, actions)

        # 3. Calculate rewards and check for termination based on the resolved grid
        rewards, is_case_solved = self._calculate_rewards(key_reward, state_after_resolve)

        # Transition to the next test case if the current one is solved
        next_idx = state.current_test_case_idx + 1

        def _advance_case(s: State) -> State:
            """Logic to advance to the next test case."""
            return s.replace(
                current_test_case_idx=next_idx,
                # Reset workspace for the new case. Could be initialized from the new input.
                working_output_grid=jnp.zeros_like(s.working_output_grid),
                working_output_mask=jnp.zeros_like(s.working_output_mask),
                # Optionally reset hypotheses
            )

        # Use lax.cond to conditionally advance the state
        state_after_solve = lax.cond(
            is_case_solved,
            _advance_case,      # If true, apply this function
            lambda s: s,        # If false, do nothing
            state_after_actions # Operand
        )

        done = self.is_terminal(state_after_grid_apply)

        # Update final state elements
        final_state = state_after_grid_apply.replace(
            step_count=state_after_grid_apply.step_count + 1,
            terminal=done,
            key=key_next_state # Propagate PRNG key
        )

        obs = self.get_obs(final_state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done  # JaxMARL convention

        info = {} # Can be populated with diagnostic data

        return obs, final_state, rewards, dones, info

    def get_obs(self, state: State) -> dict:
        """ Returns a shared observation for all agents.
            Can be customized for agent specializations if needed.
        """
        # Get the currently active test input grid
        active_test_input_grid = state.task.test_input_grids[state.current_test_case_idx]
        active_test_input_mask = state.task.test_input_masks[state.current_test_case_idx]
        # As per [1], agents get a shared view of relevant state parts
        shared_observation_components = {
            "input_grids_examples": state.task.input_grids_examples,
            "input_masks_examples": state.task.input_masks_examples,
            "output_grids_examples": state.task.output_grids_examples,
            "output_masks_examples": state.task.output_masks_examples,
            "current_test_input_grid": active_test_input_grid,
            "current_test_input_mask": active_test_input_mask,
            "agent_scratchpad_grids": state.agent_scratchpad_grids,
            "agent_selection_masks": state.agent_selection_masks,
            "committed_output_grid": state.committed_output_grid,
            "committed_output_mask": state.committed_output_mask,
            "agent_hypotheses_ids": state.agent_hypotheses_ids,
            "agent_hypotheses_types": state.agent_hypotheses_types,
            "agent_hypotheses_data": state.agent_hypotheses_data,
            "agent_hypotheses_confidence": state.agent_hypotheses_confidence,
            "agent_hypotheses_votes": state.agent_hypotheses_votes,
            "agent_hypotheses_active_mask": state.agent_hypotheses_active_mask,
            "consensus_grid_dimensions": state.consensus_grid_dimensions,
            "step_count": state.step_count
            # Potentially state.shared_reasoning_workspace if included from /[1]
        }
        observations = {agent: shared_observation_components for agent in self.agents}
        return observations

    def _process_actions(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> State:
        """
        Processes all agent actions for a single timestep using lax.switch
        to dispatch to the appropriate helper function based on action_type.
        """
        def body_fn(agent_idx, current_state):
            agent_id_str = self.agents[agent_idx]
            action = actions[agent_id_str]

            # Dispatch to the correct JAX-native helper function
            new_state = lax.switch(
                action.action_type,
                [
                    self._apply_clear_selection,         # 0
                    self._apply_select_pixel,            # 1
                    self._apply_select_rect,             # 2
                    self._apply_select_object_by_color,  # 3
                    self._apply_change_color,            # 4
                    self._apply_move_selection,          # 5
                    lambda s, a, idx: s,                 # 6: SUBMIT (handled in reward/terminal)
                    self._apply_make_proposal,           # 7
                    self._apply_vote_hypothesis,         # 8
                ],
                current_state,
                action,
                agent_idx,
            )
            return new_state

        final_state = lax.fori_loop(0, self.num_agents, body_fn, state)
        return final_state

    def _update_consensus(self, key: chex.PRNGKey, state: State) -> State:
        # Placeholder: Process votes, check support_threshold, update consensus_grid_dimensions
        # [1]: "Process votes and check if any hypothesis's vote_count exceeds a support_threshold
        # from the Hydra config. Use jax.lax.cond to conditionally update
        # state.consensus_grid_dimensions if a consensus is reached."

        # Example: Check if any GRID_SIZE proposal has enough votes
        # This is highly conceptual.
        # active_grid_proposals_mask = (state.agent_hypotheses_active_mask &
        #                               (state.agent_hypotheses_types == GRID_SIZE_PROPOSAL_TYPE))
        # votes_for_grid_proposals = state.agent_hypotheses_votes * active_grid_proposals_mask
        # best_proposal_idx = jnp.argmax(votes_for_grid_proposals)
        # best_proposal_votes = state.agent_hypotheses_votes[best_proposal_idx]
        # support_threshold_val = self.consensus_config.get("support_threshold", 0.6) * self.num_agents

        # new_consensus_dims = jax.lax.cond(
        #     best_proposal_votes >= support_threshold_val,
        #     lambda _: state.agent_hypotheses_data[best_proposal_idx, :2].astype(jnp.int32), # Assuming data[:2] is [h,w]
        #     lambda _: state.consensus_grid_dimensions,
        #     operand=None
        # )
        # state = state.replace(consensus_grid_dimensions=new_consensus_dims)
        return state

    def _apply_consensus_to_grid(self, key: chex.PRNGKey, state: State) -> State:
        # Placeholder: Modify working_output_grid based on consensus_grid_dimensions
        # [1]: "If state.consensus_grid_dimensions is valid (not [-1, -1]), modify
        # state.working_output_grid and state.working_output_mask."

        # def resize_grid_fn(current_grid, current_mask, new_dims):
        #    # Create new grid of new_dims, potentially copy content
        #    new_grid = jnp.full((self.max_grid_size_h, self.max_grid_size_w), -1, dtype=jnp.int32)
        #    new_mask = jnp.zeros((self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_)
        #    #… logic to set new_grid.at[:new_dims, :new_dims]…
        #    #… logic to set new_mask.at[:new_dims, :new_dims] = True…
        #    return new_grid, new_mask

        # new_working_grid, new_working_mask = jax.lax.cond(
        #     jnp.all(state.consensus_grid_dimensions > -1),
        #     lambda s: resize_grid_fn(s.working_output_grid, s.working_output_mask, s.consensus_grid_dimensions),
        #     lambda s: (s.working_output_grid, s.working_output_mask),
        #     operand=state
        # )
        # state = state.replace(working_output_grid=new_working_grid, working_output_mask=new_working_mask)
        return state

        def _resolve_commits(self, key: chex.PRNGKey, state: State, actions: dict) -> State:
    # 1. Calculate a "strength" for each agent's potential commit.
    # Strength is based on the vote count and confidence of the backing hypothesis.

    def get_strength(agent_idx: int) -> float:
        action = actions[f"agent_{agent_idx}"]
        hyp_idx = action.params[0]

        # Fetch vote count and confidence for the referenced hypothesis
        vote_count = state.agent_hypotheses_votes[hyp_idx]
        confidence = state.agent_hypotheses_confidence[hyp_idx]

        # Strength is a combination of votes and confidence (e.g., votes + confidence)
        # Only grant strength if the action is actually COMMIT_CHANGES
        is_commit_action = (action.action_type == ACTION_COMMIT_CHANGES)
        strength = (vote_count + confidence) * is_commit_action
        return strength

    # Vectorize the strength calculation across all agents
    agent_strengths = jax.vmap(get_strength)(jnp.arange(self.num_agents))

    # 2. For each pixel, find the agent with the highest strength who is modifying it.
    # Create a tensor of shape (num_agents, H, W) where each slice is the agent's strength
    # repeated across the grid, but only where they made a change.

    # Get changes from the previous committed grid
    changes = state.agent_scratchpad_grids != state.committed_output_grid

    # Broadcast strengths and combine with changes
    strength_map = jnp.expand_dims(agent_strengths, axis=(1, 2)) # Shape: (num_agents, 1, 1)
    write_strength_map = strength_map * changes # Shape: (num_agents, H, W)

    # 3. Determine the "winning" agent for each pixel.
    # This gives the index of the agent with the max strength for each pixel.
    winning_agent_idx_map = jnp.argmax(write_strength_map, axis=0) # Shape: (H, W)

    # 4. Construct the new committed grid by gathering pixels from the winners' scratchpads.
    # We can use the winning_agent_idx_map to index into agent_scratchpad_grids.
    # This is an advanced use of jax.vmap or indexing. A simpler, equivalent loop:

    def get_pixel_from_winner(y, x):
        winner_idx = winning_agent_idx_map[y, x]
        return state.agent_scratchpad_grids[winner_idx, y, x]

    # This can be vectorized, but a conceptual map shows the logic
    # Simplified approach:
    # Gather all candidate pixels and their strengths
    candidate_pixels = state.agent_scratchpad_grids # (A, H, W)

    # Create a one-hot encoding of the winning agent for each pixel
    winner_one_hot = jax.nn.one_hot(winning_agent_idx_map, self.num_agents, axis=0) # (A, H, W)

    # Mask out pixels from non-winning agents
    winning_pixels = candidate_pixels * winner_one_hot

    # Sum along the agent axis to select the single winning pixel for each location
    new_grid_from_winners = jnp.sum(winning_pixels, axis=0)

    # Only update pixels where there was at least one commit
    any_commit_mask = jnp.any(changes, axis=0)

    final_committed_grid = jnp.where(
        any_commit_mask,
        new_grid_from_winners,
        state.committed_output_grid # Keep the old pixel if no one committed a change
    )

    # Note: This logic only updates the grid. A similar, more complex logic
    # would be needed to update the committed_output_mask, for example if
    # an agent's commit implies a resizing of the grid.
    return state.replace(committed_output_grid=final_committed_grid)

    def _calculate_rewards(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> dict:
        # Placeholder: Implement multi-tiered reward logic
        # [1]: "Tier 1: Check for full solution… Tier 2: Check for correct grid size consensus…
        # Tier 3: Reward for correct proposals…"
        idx = state.current_test_case_idx
        # Compare against the currently active true output grid
        true_grid = state.task.true_test_output_grids[idx]
        true_mask = state.task.true_test_output_masks[idx]

        rewards = {agent: 0.0 for agent in self.agents}

        # Tier 1: Full solution
        is_correct = jnp.all(state.working_output_grid[true_mask] == true_grid[true_mask])

        completion_reward_val = self.reward_config.get("R_complete_solution", 100.0)

        # Reward can be tiered: small reward for solving a case, large for the whole task
        is_last_case = (idx + 1) >= state.task.num_test_pairs
        final_solution_reward = self.reward_config.get("final_solution", 10.0)
        case_solved_reward = self.reward_config.get("case_solved", 1.0)

        # Reward is high if this was the last case, otherwise it's a smaller intermediate reward
        reward_val = lax.cond(is_last_case, lambda _: final_solution_reward, lambda _: case_solved_reward, None)
        # Only give reward if the case is actually solved
        reward_val *= is_correct

        rewards = {agent: reward_val for agent in self.agents}
        return rewards, is_correct

    def is_terminal(self, state: State) -> bool:
        # [1]: "Check for full solution… or max_steps"
        # Check if the state has advanced PAST the last valid index
        all_cases_solved = state.current_test_case_idx >= state.task.num_test_pairs
        max_steps_reached = state.step_count >= self.default_params["max_steps_per_episode"]
        done = jnp.logical_or(all_cases_solved, max_steps_reached)
        return done

    def observation_space(self, agent: str) -> spaces.Space:
        # Based on get_obs structure and types.py constants
        obs_dict = {
            "input_grids_examples": spaces.Box(low=-1, high=9, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
            "input_masks_examples": spaces.Box(low=0, high=1, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
            "output_grids_examples": spaces.Box(low=-1, high=9, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
            "output_masks_examples": spaces.Box(low=0, high=1, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
            "current_test_input_grid": spaces.Box(low=-1, high=9, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
            "current_test_input_mask": spaces.Box(low=0, high=1, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
            "working_output_grid": spaces.Box(low=-1, high=9, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
            "working_output_mask": spaces.Box(low=0, high=1, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
            "agent_hypotheses_ids": spaces.Box(low=-1, high=self.num_agents -1 if self.num_agents > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
            "agent_hypotheses_types": spaces.Box(low=-1, high=self.num_proposal_types -1 if self.num_proposal_types > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
            "agent_hypotheses_data": spaces.Box(low=-1, high=30, shape=(self.max_hypotheses, self.max_proposal_data_dim), dtype=jnp.int32), # Bounds depend on data
            "agent_hypotheses_confidence": spaces.Box(low=0.0, high=1.0, shape=(self.max_hypotheses,), dtype=jnp.float32),
            "agent_hypotheses_votes": spaces.Box(low=0, high=self.num_agents if self.num_agents > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
            "agent_hypotheses_active_mask": spaces.Box(low=0, high=1, shape=(self.max_hypotheses,), dtype=jnp.bool_),
            "consensus_grid_dimensions": spaces.Box(low=-1, high=max(self.max_grid_size_h, self.max_grid_size_w), shape=(2,), dtype=jnp.int32),
            "step_count": spaces.Box(low=0, high=self.default_params["max_steps_per_episode"], shape=(), dtype=jnp.int32)
        }
        return spaces.Dict(obs_dict)

    def action_space(self, agent: str) -> spaces.Space:
        # From [1] Action Space Example
        return spaces.Dict({
            "action_type": spaces.Discrete(self.num_action_types),
            "action_data": spaces.Box(
                low=-1, high=30, shape=(self.max_proposal_data_dim,), dtype=jnp.int32
            ) # Bounds depend on data (e.g. grid values, coordinates)
        })

```

### 4.1. Implementing Selection and Manipulation Actions

The core of the new logic lies in implementing a suite of helper functions, each
corresponding to an action type. These must be pure JAX functions.

- `_apply_clear_selection(state, action, agent_idx) -> State`: Returns
  `state.replace(selection_mask=jnp.zeros_like(state.selection_mask))`.

- `_apply_select_pixel(state, action, agent_idx) -> State`: Creates a new
  all-false mask and sets the single pixel at `(y, x)` to `True`.

- `_apply_change_color(state, action, agent_idx) -> State`: This is a key
  manipulation function. It uses the existing `state.selection_mask` to update
  the `working_output_grid`.

  ```
    def _apply_change_color(self, state: State, action: Action, agent_idx: int) -> State:
        new_color = action.params[0]
        # jnp.where is a perfect JAX-native conditional update
        new_grid = jnp.where(
            state.selection_mask,       # condition: where selection_mask is True
            new_color,                  # if true, use the new color
            state.working_output_grid   # if false, keep the old color
        )
        return state.replace(working_output_grid=new_grid)
  ```

- **`_apply_select_object_by_color`**: This is the most complex new function.
  Implementing a flood-fill/flood-select in a purely functional,
  JAX-JIT-compatible way is non-trivial. It typically requires an iterative
  approach using `jax.lax.scan` or `jax.lax.while_loop` to expand the selection
  from a seed point until no new pixels of the target color are found at the
  boundary. This is an advanced JAX technique but is essential for creating a
  powerful object selection tool.

### 4.1. `__init__(self, cfg: DictConfig)`

The constructor initializes the `ArcEnv` by first calling
`super().__init__(cfg)` to set up common attributes inherited from
`ArcMarlEnvBase` (like `self.num_agents`, `self.max_grid_size_h`,
`self.reward_config`, etc.). A crucial step is the instantiation of its
dataset-specific data parser. This is achieved using
`hydra.utils.instantiate(cfg.parser_config, max_grid_size_h=self.max_grid_size_h, max_grid_size_w=self.max_grid_size_w)`.
The `cfg.parser_config` (part of the environment's configuration, e.g., from
`conf/environment/arc.yaml`) contains the `_target_` path to the concrete parser
class 1 and any parser-specific parameters. The maximum grid dimensions are
passed to the parser's constructor, as these are essential for the parser's
padding operations

### 4.2. `reset(self, key: chex.PRNGKey) -> tuple`

The `reset` method is responsible for starting a new episode. It must be a pure
JAX function.

1. **PRNG Key Management:** The input `key` is split (e.g.,
   `key_task, key_state_init = jax.random.split(key)`) to ensure that distinct
   random operations use independent keys, a cornerstone of JAX's
   reproducibility

2. **Task Loading:** It uses the instantiated parser (`self.parser`) to sample
   and preprocess a new ARC task:
   `task_data: ParsedTaskData = self.parser.get_random_task(key_task)`. The
   parser returns a `ParsedTaskData` Pytree.

3. **State Initialization:** A new `State` Pytree is created and returned.

   - Task-specific fields (e.g., `input_grids_examples`,
     `current_test_input_grid`, and their masks) are populated directly from the
     `task_data` Pytree.

   - Collaborative workspace fields like `working_output_grid` and
     `working_output_mask` are initialized (e.g., `working_output_grid` to an
     empty grid of `(MAX_GRID_H, MAX_GRID_W)` filled with a padding value like
     -1, and `working_output_mask` to all `False`).

   - Hypothesis storage arrays (`agent_hypotheses_ids`,
     `agent_hypotheses_types`, `agent_hypotheses_data`,
     `agent_hypotheses_confidence`, `agent_hypotheses_votes`) are initialized to
     default "empty" values (e.g., -1 for IDs/types, zeros for
     data/confidence/votes). The `agent_hypotheses_active_mask` is initialized
     to all `False`

   - Consensus-related fields like `consensus_grid_dimensions` are initialized
     to indicate no consensus (e.g., `jnp.array([-1, -1], dtype=jnp.int32)`).

   - Metadata fields are set: `step_count = 0`, `terminal = False`, and
     `key = key_state_init`.

4. **Initial Observations:** `obs = self.get_obs(state)` is called to generate
   the initial observations for all agents.

5. The method returns the tuple `(obs, state)`.

### 4.3. `step(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> tuple`

The `step` method advances the environment by one timestep based on agent
actions. It must also be a pure JAX function.

1. **PRNG Key Management:** The input `key` is split for any stochastic
   operations within the step.

2. **Orchestration of Internal Logic:** The `step` method primarily orchestrates
   calls to a sequence of internal helper methods, as prescribed by `1`:

   - `state_after_actions = self._process_actions(key_process, state, actions)`:
     Processes actions from all agents.

   - `state_after_consensus = self._update_consensus(key_consensus, state_after_actions)`:
     Updates any consensus based on current hypotheses and votes.

   - `state_after_grid_apply = self._apply_consensus_to_grid(key_apply, state_after_consensus)`:
     Applies agreed-upon changes to the `working_output_grid`.

   - `rewards = self._calculate_rewards(key_reward, state_after_grid_apply, actions)`:
     Calculates rewards for each agent. Note that `actions` might be needed for
     certain reward calculations (e.g., rewarding specific types of actions).

   - `done = self.is_terminal(state_after_grid_apply)`: Checks if the episode
     has terminated.

3. **State Update:** The `state` is updated with the new `step_count`
   (incremented by 1) and the `terminal` status using `state.replace(…)`. The
   PRNG `key` within the state is also updated.

4. **Next Observations:** `obs = self.get_obs(final_state)` generates the
   observations for the next timestep.

5. **Dones Dictionary:** A `dones` dictionary is created, mapping each agent ID
   to the `done` flag. Crucially, it must also include an `__all__` key mapping
   to the overall episode `done` flag, as per JaxMARL convention.

6. **Info Dictionary:** An `info` dictionary (initially empty) can be populated
   with any auxiliary diagnostic information.

7. The method returns the tuple `(obs, final_state, rewards, dones, info)`.

The design of `step` as an orchestrator of smaller, pure functions
(`_process_actions`, etc.) promotes modularity and makes the complex state
transition logic easier to manage and test. This contrasts with potentially
monolithic step functions or object-oriented approaches with in-place state
modifications (like those possible in ARCLE 2), which are not amenable to JAX's
JIT compilation. JaxARC's functional approach, where each helper returns a new
state, is essential.

### 4.4. `get_obs(self, state: State) -> dict`

This method constructs and returns the observation dictionary for all agents
based on the current `State` Pytree. As specified in `1`, agents receive a
shared view of the environment. The observation for each agent is a dictionary
containing relevant slices of the `State` Pytree. Based on `1` and `1` (Section
6.3), this typically includes:

- The full ARC task specification (`input_grids_examples`,
  `output_grids_examples`, `current_test_input_grid`, and their masks).

- The current state of the collaborative `working_output_grid` and its mask.

- The complete set of active agent hypotheses (`agent_hypotheses_ids`, `types`,
  `data`, `confidence`, `votes`, `active_mask`).

- Current consensus information (e.g., `consensus_grid_dimensions`).

- Environment metadata like step_count.

  The implementation would construct a dictionary holding these components and
  then create a per-agent dictionary where each agent receives this shared
  observation structure.

### 4.5. Internal Helper Methods

These methods encapsulate specific parts of the environment's logic and are
called by `step`. They must all be pure functions, taking the current state (and
other necessary inputs like actions or keys) and returning an updated state or
other results (like rewards).

- `_process_actions(key, state, actions) -> State`: This method is responsible
  for interpreting agent actions. For `PROPOSE` actions, it involves finding an
  available slot in the `state.agent_hypotheses_*` arrays (e.g., using
  `jnp.argmin(state.agent_hypotheses_active_mask)` to find the first `False`
  entry, if one is guaranteed) and then immutably updating the arrays at that
  index using JAX's `.at[index].set(value)` syntax to store the new hypothesis

- `_update_consensus(key, state) -> State`: This method processes votes on
  existing hypotheses and checks if any hypothesis meets the consensus criteria
  (e.g., `vote_count` exceeding a `support_threshold` from
  `self.consensus_config`). It uses `jax.lax.cond` for conditional updates to
  `state.consensus_grid_dimensions` or other consensus fields if agreement is
  reached

- `_apply_consensus_to_grid(key, state) -> State`: If consensus has been reached
  on certain aspects (e.g., `state.consensus_grid_dimensions` is valid and not
  `[-1,-1]`), this method modifies `state.working_output_grid` and
  `state.working_output_mask` accordingly. This might involve creating a new
  grid of the agreed-upon size

- `_calculate_rewards(key, state, actions) -> dict`: This pure JAX function
  implements the multi-tiered reward system. Tier 1 rewards for full solution,
  Tier 2 for sub-goals like correct grid size consensus, and Tier 3 for
  collaboration (e.g., correct proposals)

- `is_terminal(state) -> bool`: Determines if the episode ends, either by
  achieving a full solution
  (`jnp.all((state.working_output_grid == state.true_test_output_grid) & state.true_test_output_mask)`)
  or by reaching the maximum step count

The consistent use of immutable updates (e.g., `state.replace(…)` or returning
new arrays from helpers) and explicit PRNG key management throughout these
methods is paramount for JAX compatibility and reproducible behavior

## 5. Defining Observation and Action Spaces (`jaxmarl.environments.spaces`)

The `observation_space` and `action_space` methods are critical components of
any MARL environment. They define the structure, shape, and data type of the
observations agents receive and the actions they can take. These definitions are
essential for compatibility with MARL algorithms, which use them to construct
agent policies (e.g., neural network architectures) and to validate interactions
with the environment For JaxARC, these spaces are defined using

`jaxmarl.environments.spaces`, which aligns with the user's requirement to use
JaxMARL's native space implementation.

### 5.1. `action_space(self, agent: str) -> spaces.Space`

This method must return a `jaxmarl.environments.spaces.Dict` instance that
precisely matches the structure of the `Action` Pytree.

Python

```python
# In ArcEnv class
# from jaxmarl.environments import spaces # Ensure this is imported
# import jax.numpy as jnp
# MAX_PROPOSAL_DATA_DIM from types.py

def action_space(self, agent: str) -> spaces.Space:
    """Defines the action space for a single agent."""
    return spaces.Dict({
        "action_type": spaces.Discrete(NUM_ACTION_TYPES), # Now has more actions
        "params": spaces.Box(
            low=-self.max_grid_size_h, high=self.max_grid_size_h,
            shape=(MAX_ACTION_PARAMS,), dtype=jnp.int32
        ) # Bounds widened to allow for negative delta in MOVE
    })
```

1

**Analysis:**

- `"action_type"`: Defined as `spaces.Discrete(self.num_action_types)`. This
  allows agents to select one from `self.num_action_types` distinct categories
  of actions (e.g., PROPOSE, VOTE). `self.num_action_types` would typically be
  loaded from the Hydra configuration.

- `"action_data"`: Defined as `spaces.Box(…)`. This is suitable for numerical
  data associated with an action. The `shape` is
  `(self.max_proposal_data_dim,)`, matching the padding of `Action.action_data`.
  The `low`, `high`, and `dtype` parameters must be set according to the
  expected range and type of values in `action_data` (e.g., grid cell values,
  coordinates, indices). The example values `low=-1, high=30, dtype=jnp.int32`
  are illustrative.

### 5.2. `observation_space(self, agent: str) -> spaces.Space`

This method must return a `jaxmarl.environments.spaces.Dict` instance that
mirrors the structure and content of the observation dictionary produced by the
`get_obs` method. Each key in the observation dictionary must have a
corresponding space definition.

Python

```python
# In ArcEnv class
# from jaxmarl.environments import spaces
# import jax.numpy as jnp
# MAX_GRID_H, MAX_GRID_W, MAX_TRAIN_PAIRS, etc. from types.py

def observation_space(self, agent: str) -> spaces.Space:
    # Define num_proposal_types, e.g., from config or as a constant
    # num_proposal_types = self.cfg.get("num_proposal_types", 2) # Example

    obs_space_dict = {
        "input_grids_examples": spaces.Box(low=-1, high=9, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32), # ARC colors 0-9, -1 for padding
        "input_masks_examples": spaces.Box(low=0, high=1, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
        "output_grids_examples": spaces.Box(low=-1, high=9, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
        "output_masks_examples": spaces.Box(low=0, high=1, shape=(MAX_TRAIN_PAIRS, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
        "current_test_input_grid": spaces.Box(low=-1, high=9, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
        "current_test_input_mask": spaces.Box(low=0, high=1, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
        "agent_scratchpad_grids": spaces.Box(low=-1, high=9, shape=(self.num_agents, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
        "agent_selection_masks": spaces.Box(low=0, high=1, shape=(self.num_agents, self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
        "committed_output_grid": spaces.Box(low=-1, high=9, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.int32),
        "committed_output_mask": spaces.Box(low=0, high=1, shape=(self.max_grid_size_h, self.max_grid_size_w), dtype=jnp.bool_),
        "agent_hypotheses_ids": spaces.Box(low=-1, high=self.num_agents - 1 if self.num_agents > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
        "agent_hypotheses_types": spaces.Box(low=-1, high=self.num_proposal_types - 1 if self.num_proposal_types > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
        "agent_hypotheses_data": spaces.Box(low=-1, high=30, shape=(self.max_hypotheses, self.max_proposal_data_dim), dtype=jnp.int32), # Bounds depend on data
        "agent_hypotheses_confidence": spaces.Box(low=0.0, high=1.0, shape=(self.max_hypotheses,), dtype=jnp.float32),
        "agent_hypotheses_votes": spaces.Box(low=0, high=self.num_agents if self.num_agents > 0 else 0, shape=(self.max_hypotheses,), dtype=jnp.int32),
        "agent_hypotheses_active_mask": spaces.Box(low=0, high=1, shape=(self.max_hypotheses,), dtype=jnp.bool_),
        "consensus_grid_dimensions": spaces.Box(low=-1, high=max(self.max_grid_size_h, self.max_grid_size_w), shape=(2,), dtype=jnp.int32),
        "current_test_case_idx": spaces.Box(low=0, high=MAX_TEST_PAIRS - 1, shape=(), dtype=jnp.int32),
        "step_count": spaces.Box(low=0, high=self.default_params["max_steps_per_episode"], shape=(), dtype=jnp.int32)
    }
    return spaces.Dict(obs_space_dict)
```

Analysis:

Each field from the shared*observation_components dictionary (described in
Section 4.4) must have a corresponding entry in the spaces.Dict. spaces.Box is
typically used for array-like data, specifying low and high bounds for the
values, the exact shape of the array, and its dtype. These parameters must
precisely match the data returned by get_obs. For example, grid data uses low=-1
(for padding) and high=9 (for ARC colors), with dtype=jnp.int32. Mask arrays use
low=0, high=1 and dtype=jnp.bool*. The MAX\_\* constants defined in
src/jax_arc/types.py (e.g., MAX_HYPOTHESES, MAX_PROPOSAL_DATA_DIM,
MAX_TRAIN_PAIRS) directly inform the shape parameters for many of these
spaces.Box definitions, demonstrating the tight coupling between Pytree design
for static data representation and the definition of observation/action spaces.

The explicit use of JaxMARL's native `jaxmarl.environments.spaces` is important.
While Gymnasium is a dependency of JaxMARL 1, relying on JaxMARL's own space API
ensures full compatibility and may leverage JAX-specific optimizations or
features within the JaxMARL framework itself. These space definitions are not
merely descriptive; they constitute a strict API contract with MARL learning
algorithms, which use these definitions to configure neural network
architectures and validate environment interactions.

## 6. Alignment with ARCLE and JaxMARL Principles

The design of the JaxARC environment draws conceptual inspiration from existing
ARC environments like ARCLE 2 while strictly adhering to the API and
implementation paradigms of JaxMARL and JAX.

### 6.1. ARCLE (Abstraction and Reasoning Corpus Learning Environment)

ARCLE, being a Gymnasium-based environment for ARC, offers valuable conceptual
precedents for framing ARC tasks as RL problems Its state representations and
action definitions (e.g., coloring, resizing, selection, copy-paste operations
in

`ARCEnv` or `O2ARCv2Env`) provide insights into the types of interactions agents
might have with an ARC grid. The `O2ARCv2Env` in ARCLE, inspired by a human
interface for solving ARC, suggests the potential for a rich and complex action
space

However, there are key differences that necessitate careful adaptation for a
JAX-native, JaxMARL-based environment:

- **Framework:** ARCLE uses Gymnasium, while JaxARC uses JaxMARL. This means
  JaxARC must implement the `MultiAgentEnv` interface from JaxMARL and embrace
  JAX's functional programming principles (purity, immutability), explicit PRNG
  key management, and Pytree-based state representations.

- **State Mutability:** ARCLE operations may modify the state in-place (e.g.,
  "All operations receives state and action, and it changes state in-place" 2).
  In contrast, JaxARC's

  `step` method and all its internal helper functions must be pure, returning
  new state instances rather than modifying them in-place While ARCLE's

  `env.transition(state_copied, action)` 2 hints at a functional approach for
  obtaining next states without side effects, JaxARC mandates this purity for
  all JAX-transformed components.

- **Observation/Action Spaces:** ARCLE uses `gymnasium.spaces`. JaxARC must use
  `jaxmarl.environments.spaces`. While the underlying concepts of defining
  discrete or continuous spaces are similar, the specific API provided by
  JaxMARL must be utilized.

Thus, ARCLE serves as a valuable source for understanding the ARC problem domain
within an RL context, but the JaxARC implementation must be guided by the
specific requirements and idioms of JAX and JaxMARL, as detailed in the
project's development guides

### 6.2. JaxMARL Principles

The JaxARC environment is designed to align with the core principles of JaxMARL:

- **`MultiAgentEnv` Inheritance:** `ArcMarlEnvBase` (and consequently `ArcEnv`)
  inherits from `jaxmarl.environments.multi_agent_env.MultiAgentEnv`. This
  ensures adherence to the JaxMARL API, including the signatures for `reset` and
  `step`, the methods for defining `observation_space` and `action_space`, and
  conventions such as dictionary-based actions, observations, rewards, and dones
  keyed by agent ID, including the `dones["__all__"]` flag for episode
  termination.

- **JAX-Native Design:** The entire environment, from its Pytree data structures
  (`State`, `Action`, etc.) to the logic within `step` and `reset`, is designed
  to be JAX-native. This involves using `chex.dataclass` for Pytrees,
  `jax.numpy` for array operations, explicit PRNG key management, and writing
  pure functions. This design enables efficient JIT compilation, vectorization
  (e.g., `jax.vmap` for running multiple environment instances in parallel), and
  execution on hardware accelerators

- **State as a Pytree:** Consistent with typical JaxMARL environments, the
  JaxARC environment manages its entire state as a JAX Pytree (specifically, the
  `State` dataclass).

- **Configuration via Hydra:** The extensive use of Hydra for managing
  configurations, as detailed in the JaxARC development guides 1, aligns with
  common best practices in modern JAX-based machine learning projects,
  facilitating reproducibility and systematic experimentation.

In essence, JaxMARL defines the _interface_ (the "what," e.g., method names and
signatures), while JAX's functional programming principles dictate the
_implementation style_ (the "how," e.g., pure functions, immutable updates,
static shapes). The provided development guides 1 are specifically tailored for
building such a JaxMARL-compatible environment for ARC, inherently incorporating
these JAX-native patterns.

## 7. Conclusion and Path Forward

This report has detailed the architectural blueprint for a JaxMARL environment
tailored to the Abstraction and Reasoning Corpus. Key components include
precisely defined JAX Pytrees (`ParsedTaskData`, `Hypothesis`, `State`,
`Action`) that prioritize static shapes for JAX compatibility; abstract base
classes (`ArcDataParserBase`, `ArcMarlEnvBase`) that promote modularity and
adherence to the JaxMARL API; the structure of the concrete `ArcEnv` class with
its core `reset` and `step` methods; and the definitions for `observation_space`
and `action_space` using `jaxmarl.environments.spaces`.

The design emphasizes JAX-idiomatic principles: function purity, immutable state
updates, explicit PRNG key management, and static data structures achieved
through padding and masking. These are crucial for leveraging JAX's performance
capabilities, particularly JIT compilation.

The path forward involves implementing these detailed structures and methods.
Subsequent development will focus on the internal logic of the helper functions
orchestrated by `ArcEnv.step`—namely `_process_actions`, `_update_consensus`,
`_apply_consensus_to_grid`, and `_calculate_rewards`—as well as developing
comprehensive testing scripts. The phased implementation plan detailed in the
provided developer guides 1 offers a structured approach to completing a fully
functional and robust ARC environment within the JaxMARL framework.

## 8. Visualization and Logging Utilities (`src/jax_arc/utils/logging_utils.py`)

For effective debugging and monitoring of the environment and agents, it is
crucial to visualize the state of the ARC grids. This is accomplished through a
dedicated module, `src/jax_arc/utils/logging_utils.py`, which provides functions
to render grids in different formats.

### 8.1 Code Implementation

```python
# src/jax_arc/utils/logging_utils.py

import jax.numpy as jnp
import chex
from rich.table import Table
from rich.console import Console
import drawsvg as draw
from typing import Optional

# Define the standard ARC color palette. Using hex codes is compatible with both rich and drawsvg.
ARC_COLOR_PALETTE = {
    -1: "#C0C0C0",  # Padding/Empty color (Silver)
    0: "#000000",   # Black
    1: "#0074D9",   # Blue
    2: "#FF4136",   # Red
    3: "#2ECC40",   # Green
    4: "#FFDC00",   # Yellow
    5: "#AAAAAA",   # Grey
    6: "#F012BE",   # Fuchsia
    7: "#FF851B",   # Orange
    8: "#7FDBFF",   # Cyan
    9: "#870C25"    # Brown (Maroon)
}

# --- Rich-based Terminal Visualization ---

def render_grid_rich(
    grid: chex.Array,
    mask: Optional[chex.Array] = None,
    title: Optional[str] = "ARC Grid"
) -> Table:
    """
    Renders an ARC grid as a rich.table.Table for terminal display.

    Args:
        grid: A 2D JAX numpy array representing the grid.
        mask: An optional boolean mask. If provided, masked-out cells are rendered differently.
        title: An optional title for the table.

    Returns:
        A rich.table.Table object ready to be printed to the console.
    """
    if grid.ndim != 2:
        raise ValueError(f"Input grid must be 2D, but got shape {grid.shape}")

    h, w = grid.shape
    table = Table(title=title, show_header=False, show_edge=True, box=None, padding=0, pad_edge=False)
    for _ in range(w):
        table.add_column(justify="center")

    for i in range(h):
        row_items = []
        for j in range(w):
            if mask is not None and not mask[i, j]:
                # Render masked-out cells with a faint pattern
                row_items.append("[grey23]·[/]")
            else:
                color_val = int(grid[i, j])
                hex_color = ARC_COLOR_PALETTE.get(color_val, "#FFFFFF") # Default to white if color is unknown
                # Use a full block character to represent a pixel, styled with its hex color.
                row_items.append(f"[{hex_color}]██[/]")
        table.add_row(*row_items)
    return table

# --- DrawSVG-based SVG Visualization ---

def render_grid_svg(
    grid: chex.Array,
    mask: Optional[chex.Array] = None,
    cell_size: int = 20,
    padding: int = 2
) -> draw.Drawing:
    """
    Renders an ARC grid as a drawsvg.Drawing object.

    Args:
        grid: A 2D JAX numpy array representing the grid.
        mask: An optional boolean mask. If provided, masked-out cells are rendered differently.
        cell_size: The size of each grid cell in pixels.
        padding: The padding between cells.

    Returns:
        A drawsvg.Drawing object that can be saved as an SVG file.
    """
    if grid.ndim != 2:
        raise ValueError(f"Input grid must be 2D, but got shape {grid.shape}")

    h, w = grid.shape
    total_cell_size = cell_size + padding
    drawing = draw.Drawing(w * total_cell_size, h * total_cell_size)
    drawing.append(draw.Rectangle(0, 0, '100%', '100%', fill='#333333')) # Background

    for i in range(h):
        for j in range(w):
            y, x = i * total_cell_size, j * total_cell_size

            if mask is not None and not mask[i, j]:
                # Render masked-out cells as a slightly different background color
                fill_color = "#444444"
            else:
                color_val = int(grid[i, j])
                fill_color = ARC_COLOR_PALETTE.get(color_val, "#FFFFFF")

            drawing.append(
                draw.Rectangle(
                    x, y,
                    cell_size, cell_size,
                    fill=fill_color,
                    stroke='grey',
                    stroke_width=0.5
                )
            )
    return drawing

# --- JAX Debug Callback Wrapper ---

def log_grid_to_console(grid: chex.Array, title: str = "Grid State"):
    """
    A simple wrapper function to be used with jax.debug.callback.
    It takes a grid and a title, renders it with Rich, and prints to the console.

    This function itself contains a side-effect (print) and is not JAX-compatible,
    which is why it must be called via `jax.debug.callback`.

    Example Usage in a JIT-ted function:
        >>> jax.debug.print(
        …     "{title}",
        …     title=title,
        …     x=grid,
        …     callback=lambda x: log_grid_to_console(x, title="My Grid")
        … )

    Or more directly:
        >>> jax.debug.callback(log_grid_to_console, grid, title="My Grid")

    Args:
        grid: The grid to visualize.
        title: The title to display above the grid.
    """
    console = Console()
    grid_table = render_grid_rich(grid, title=title)
    console.print(grid_table)
```

### 8.2. Visualization Functions

The module provides two main rendering functions:

- `render_grid_rich`: Uses the `rich` library to create a colorized, textual
  representation of the grid suitable for printing directly to a modern
  terminal.

- `render_grid_svg`: Uses the `drawsvg` library to generate a vector-based SVG
  image of the grid, which is ideal for saving to files, logging to platforms
  like Weights & Biases, or embedding in reports.

### 8.3. Integration with JAX using `jax.debug.callback`

A core principle of JAX is that functions intended for JIT compilation must be
_pure_—they cannot have side effects like printing to the console or writing to
a file. Our visualization functions are inherently side-effectful.

To bridge this gap, we use `jax.debug.callback`. This function allows a
JIT-compiled function to safely "call back" to the Python interpreter to execute
a regular, side-effectful Python function on intermediate values from the JAX
computation.

The `logging_utils.py` module includes `log_grid_to_console`, a simple wrapper
designed for this purpose. Here is how you can integrate it into your
environment's `step` method to print the `working_output_grid` at the beginning
of every step:

```python
# Inside ArcEnv.step() method
import jax
from src.jax_arc.utils import logging_utils

def step(self, key: chex.PRNGKey, state: State, actions: dict[str, Action]) -> tuple:
    # This callback will execute at the start of each step, printing the grid.
    # The title can be dynamically formatted with JAX-traced values like state.step_count.
    jax.debug.callback(
        logging_utils.log_grid_to_console,
        state.working_output_grid,
        title=f"Step {state.step_count}: Working Grid"
    )

    # … The rest of the step logic follows …
```

This approach allows you to gain invaluable insight into the environment's state
over time without breaking the purity and performance benefits of JAX's JIT
compilation. For more complex logging, such as saving SVG files, a similar
callback approach would be used in your main training script.

The existing logging utilities can be easily extended to visualize the new
per-agent selection masks. When calling `jax.debug.callback` in a training loop,
one can iterate through the masks to gain insight into what each agent is
focusing on.

```python
# Example of logging multiple masks inside a JIT-ted function
def log_all_selections(masks):
    for i in range(masks.shape[0]):
        # This calls the non-JAX logging function for each mask
        logging_utils.log_grid_to_console(masks[i], title=f"Agent {i} Selection")

# Inside ArcEnv.step()
jax.debug.callback(log_all_selections, state.agent_selection_masks)
```

This provides invaluable debugging insight into individual agent behaviors and
potential coordination.
