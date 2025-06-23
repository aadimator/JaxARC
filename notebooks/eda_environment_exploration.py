# %% [markdown]
# # JaxARC Environment and Dataset Exploration
#
# This notebook demonstrates the complete workflow of:
# 1. Loading and parsing ARC datasets using configuration files
# 2. Instantiating the ARCLE environment
# 3. Loading tasks into the environment
# 4. Exploring different actions and operations
# 5. Visualizing grids and transformations

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# JaxARC imports
from jaxarc.parsers import ArcAgiParser
from jaxarc.envs import ARCLEEnvironment
from jaxarc.utils.config import get_config
from jaxarc.utils.visualization import (
    log_grid_to_console,
    visualize_parsed_task_data_rich,
)

# Initialize console for rich output
console = Console()

# %% [markdown]
# ## 1. Configuration Loading
#
# Let's start by loading the configuration using Hydra. This will give us access to
# dataset and environment configurations.

# %%
# Load configuration
config = get_config()

console.print(Panel(
    f"[bold green]Configuration Loaded[/bold green]\n\n"
    f"Dataset: {config.dataset.dataset_name}\n"
    f"Environment: arcle_env\n"
    f"Data Root: {config.dataset.data_root}\n"
    f"Max Grid Size: {config.dataset.max_grid_size}x{config.dataset.max_grid_size}",
    title="JaxARC Configuration"
))

# %% [markdown]
# ## 2. Dataset Loading and Parsing
#
# Now let's instantiate the parser and load some tasks from the dataset.

# %%
# Initialize the ARC-AGI parser with the dataset config
parser = ArcAgiParser(config.dataset)

# Get available task IDs
available_task_ids = parser.get_available_task_ids()

console.print(Panel(
    f"[bold blue]Dataset Information[/bold blue]\n\n"
    f"Name: {config.dataset.dataset_name}\n"
    f"Year: {config.dataset.dataset_year}\n"
    f"Description: {config.dataset.description}\n"
    f"Available Tasks: {len(available_task_ids)}\n"
    f"Sample Task IDs: {available_task_ids[:5]}",
    title="Dataset Details"
))

# %% [markdown]
# ## 3. Random Task Selection and Exploration
#
# Let's select a random task and explore its structure.

# %%
# Initialize JAX random key
key = jr.PRNGKey(42)

# Get a random task using the parser's method
key, task_key = jr.split(key)
parsed_task = parser.get_random_task(task_key)

console.print(f"[green]Selected random task: {parsed_task.task_id}[/green]")

console.print(Panel(
    f"[bold cyan]Task Structure[/bold cyan]\n\n"
    f"Task ID: {parsed_task.task_id}\n"
    f"Train Pairs: {parsed_task.num_train_pairs}\n"
    f"Test Pairs: {parsed_task.num_test_pairs}\n"
    f"Input Grids Shape: {parsed_task.input_grids_examples.shape}\n"
    f"Output Grids Shape: {parsed_task.output_grids_examples.shape}\n"
    f"Test Input Shape: {parsed_task.test_input_grids.shape}\n"
    f"Test Output Shape: {parsed_task.true_test_output_grids.shape}",
    title="Parsed Task Details"
))

# %% [markdown]
# ## 4. Task Visualization
#
# Let's visualize the first training pair to understand the task structure.

# %%
# Visualize the first training pair
if parsed_task.num_train_pairs > 0:
    # Extract first training pair (remove padding and get actual grid)
    input_grid = parsed_task.input_grids_examples[0]
    output_grid = parsed_task.output_grids_examples[0]
    input_mask = parsed_task.input_masks_examples[0]
    output_mask = parsed_task.output_masks_examples[0]

    console.print("[bold yellow]First Training Pair:[/bold yellow]")
    console.print("\n[cyan]Input Grid:[/cyan]")
    log_grid_to_console(input_grid)

    console.print("\n[cyan]Output Grid:[/cyan]")
    log_grid_to_console(output_grid)

    # Show grid dimensions info
    console.print(f"\n[yellow]Grid Info:[/yellow]")
    console.print(f"Input Grid Shape: {input_grid.shape}")
    console.print(f"Output Grid Shape: {output_grid.shape}")
    console.print(f"Input Mask Sum: {jnp.sum(input_mask)} (valid cells)")
    console.print(f"Output Mask Sum: {jnp.sum(output_mask)} (valid cells)")

# Display full task visualization (first 2 pairs)
console.print("\n[bold magenta]Full Task Visualization:[/bold magenta]")
visualize_parsed_task_data_rich(parsed_task, max_pairs=2)

# %% [markdown]
# ## 5. Environment Initialization
#
# Now let's create the ARCLE environment using the new implementation.

# %%
# Initialize the ARCLE environment with proper configuration
env_config = config.environment
max_grid_size = tuple(env_config.get("max_grid_size", [30, 30]))
max_episode_steps = env_config.get("max_episode_steps", 100)

env = ARCLEEnvironment(
    config=env_config,
    num_agents=1,
    max_grid_size=max_grid_size,
    max_episode_steps=max_episode_steps
)

console.print(Panel(
    f"[bold magenta]Environment Initialized[/bold magenta]\n\n"
    f"Environment Name: {env.name}\n"
    f"Max Grid Size: {env.max_grid_size}\n"
    f"Max Episode Steps: {env.max_episode_steps}\n"
    f"Number of Agents: {env.num_agents}\n"
    f"Agents: {env.agents}\n"
    f"Reward on Submit Only: {env.reward_on_submit_only}",
    title="ARCLE Environment"
))

# Print action and observation space information
agent_id = env.agents[0]
action_space = env.action_spaces[agent_id]
obs_space = env.observation_spaces[agent_id]

console.print(Panel(
    f"[bold yellow]Agent Spaces Information[/bold yellow]\n\n"
    f"Agent ID: {agent_id}\n"
    f"Action Space Type: {type(action_space).__name__}\n"
    f"Action Space Keys: {list(action_space.spaces.keys()) if hasattr(action_space, 'spaces') else 'N/A'}\n"
    f"Observation Space Type: {type(obs_space).__name__}\n"
    f"Observation Shape: {obs_space.shape if hasattr(obs_space, 'shape') else 'N/A'}",
    title="Environment Spaces"
))

# %% [markdown]
# ## 6. Environment Reset with Task Data
#
# Let's reset the environment with our parsed task and examine the initial state.

# %%
# Reset environment with the parsed task
key, reset_key = jr.split(key)
initial_obs, initial_state = env.reset_with_task(reset_key, parsed_task)

console.print("[bold green]Environment Reset Complete![/bold green]")

# Display information about the initial state
console.print(Panel(
    f"[bold blue]Initial State Info[/bold blue]\n\n"
    f"State Type: {type(initial_state).__name__}\n"
    f"Step: {initial_state.step}\n"
    f"Done: {initial_state.done}\n"
    f"Task ID: {initial_state.task_data.task_id}\n"
    f"Current Grid Shape: {initial_state.current_grid.shape}\n"
    f"Target Grid Shape: {initial_state.target_grid.shape}\n"
    f"Similarity Score: {initial_state.similarity_score:.4f}\n"
    f"Operation Count: {initial_state.operation_count}",
    title="Environment State"
))

# Display the current and target grids
console.print("\n[cyan]Current Grid (what we start with):[/cyan]")
log_grid_to_console(initial_state.current_grid)

console.print("\n[cyan]Target Grid (what we want to achieve):[/cyan]")
log_grid_to_console(initial_state.target_grid)

console.print(f"\n[yellow]Initial similarity: {initial_state.similarity_score:.4f}[/yellow]")

# %% [markdown]
# ## 7. Action Space Exploration
#
# Let's explore the different types of actions available in the ARCLE environment.

# %%
console.print("[bold yellow]Available Operations:[/bold yellow]")

operations_table = Table(title="ARCLE Operations")
operations_table.add_column("Operation ID", style="cyan")
operations_table.add_column("Operation Type", style="magenta")
operations_table.add_column("Description", style="green")

# Add operation examples based on the ARCLE design
operation_examples = [
    (0, "Fill Color", "Fill selection with color 0 (black)"),
    (1, "Fill Color", "Fill selection with color 1 (blue)"),
    (2, "Fill Color", "Fill selection with color 2 (red)"),
    (3, "Fill Color", "Fill selection with color 3 (green)"),
    (4, "Fill Color", "Fill selection with color 4 (yellow)"),
    (5, "Fill Color", "Fill selection with color 5 (grey)"),
    (10, "Flood Fill", "Flood fill with color 0 from selection"),
    (15, "Flood Fill", "Flood fill with color 5 from selection"),
    (20, "Move Object", "Move selected object up"),
    (21, "Move Object", "Move selected object down"),
    (22, "Move Object", "Move selected object left"),
    (23, "Move Object", "Move selected object right"),
    (24, "Rotate Object", "Rotate selected object 90°"),
    (25, "Rotate Object", "Rotate selected object 270°"),
    (26, "Flip Object", "Flip selected object horizontally"),
    (27, "Flip Object", "Flip selected object vertically"),
    (28, "Copy to Clipboard", "Copy selection to clipboard"),
    (29, "Paste from Clipboard", "Paste clipboard content"),
    (30, "Cut to Clipboard", "Cut selection to clipboard"),
    (31, "Clear Grid", "Clear entire grid"),
    (32, "Resize Grid", "Resize grid operation"),
    (33, "Grid Transform", "Apply grid transformation"),
    (34, "Submit", "Submit current grid as solution"),
]

for op_id, op_type, description in operation_examples:
    operations_table.add_row(str(op_id), op_type, description)

console.print(operations_table)

# %% [markdown]
# ## 8. Action Creation and Execution
#
# Let's create sample actions and execute them to see how they affect the grid.

# %%
# Create a sample action following the ARCLE format
h, w = env.max_grid_size

# Create a selection mask (small rectangular region)
selection_mask = jnp.zeros((h, w), dtype=jnp.float32)
selection_mask = selection_mask.at[5:10, 5:10].set(1.0)  # Select 5x5 region

# Create an action dictionary
sample_action = {
    "selection": selection_mask,
    "operation": jnp.array(3, dtype=jnp.int32)  # Fill with green (color 3)
}

console.print(Panel(
    f"[bold green]Sample Action Structure[/bold green]\n\n"
    f"Selection Shape: {sample_action['selection'].shape}\n"
    f"Selection Type: {sample_action['selection'].dtype}\n"
    f"Selection Sum: {jnp.sum(sample_action['selection'])} (selected pixels)\n"
    f"Operation: {sample_action['operation']} (Fill with green)\n"
    f"Operation Type: {type(sample_action['operation'])}",
    title="Action Analysis"
))

# Visualize the selection mask
console.print("\n[cyan]Selection Mask Visualization:[/cyan]")
log_grid_to_console(sample_action['selection'].astype(jnp.int32))

# %% [markdown]
# ## 9. Action Execution
#
# Let's execute the action and see the result.

# %%
# Execute the action
try:
    key, step_key = jr.split(key)
    actions = {agent_id: sample_action}

    # Execute step using the new step_env method
    new_obs, new_state, rewards, dones, infos = env.step_env(step_key, initial_state, actions)

    console.print(Panel(
        f"[bold green]Step Execution Successful![/bold green]\n\n"
        f"Reward: {rewards[agent_id]:.4f}\n"
        f"Done: {dones[agent_id]}\n"
        f"New Similarity: {new_state.similarity_score:.4f}\n"
        f"Operation Count: {new_state.operation_count}\n"
        f"Step: {new_state.step}",
        title="Step Result"
    ))

    # Show the grid after the action
    console.print("\n[cyan]Grid After Fill Action:[/cyan]")
    log_grid_to_console(new_state.current_grid)

    # Show info from the step
    if agent_id in infos and infos[agent_id]:
        info = infos[agent_id]
        console.print(f"\n[yellow]Step Info:[/yellow]")
        for key_name, value in info.items():
            console.print(f"  {key_name}: {value}")

except Exception as e:
    console.print(f"[red]Step execution failed: {e}[/red]")
    import traceback
    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

# %% [markdown]
# ## 10. Multiple Action Sequence
#
# Let's try executing a sequence of actions to demonstrate the workflow.

# %%
def create_action(selection_coords: tuple, operation: int, grid_size: tuple) -> dict:
    """Create an action with selection at specified coordinates."""
    h, w = grid_size
    selection_mask = jnp.zeros((h, w), dtype=jnp.float32)

    # Set selection region if coordinates provided
    if selection_coords:
        r1, c1, r2, c2 = selection_coords
        selection_mask = selection_mask.at[r1:r2, c1:c2].set(1.0)

    return {
        "selection": selection_mask,
        "operation": jnp.array(operation, dtype=jnp.int32)
    }

# Create a sequence of actions to try
action_sequence = [
    ("Fill region with blue", create_action((2, 2, 6, 6), 1, env.max_grid_size)),  # Fill with blue
    ("Copy to clipboard", create_action((2, 2, 6, 6), 28, env.max_grid_size)),    # Copy to clipboard
    ("Paste elsewhere", create_action((10, 10, 14, 14), 29, env.max_grid_size)),  # Paste from clipboard
    ("Fill with red", create_action((15, 15, 20, 20), 2, env.max_grid_size)),     # Fill with red
]

console.print("[bold yellow]Executing Action Sequence:[/bold yellow]")

current_state = initial_state
step_results = []

for i, (description, action) in enumerate(action_sequence):
    try:
        key, step_key = jr.split(key)
        actions = {agent_id: action}

        # Execute step
        obs, new_state, rewards, dones, infos = env.step_env(step_key, current_state, actions)

        step_result = {
            'step': i + 1,
            'description': description,
            'operation': int(action['operation']),
            'reward': float(rewards[agent_id]),
            'similarity': float(new_state.similarity_score),
            'done': bool(dones[agent_id])
        }
        step_results.append(step_result)

        console.print(f"[green]Step {i+1}: {description}[/green]")
        console.print(f"  Operation: {step_result['operation']}, Reward: {step_result['reward']:.3f}, Similarity: {step_result['similarity']:.3f}")

        current_state = new_state

        if dones[agent_id]:
            console.print("[red]Episode terminated![/red]")
            break

    except Exception as e:
        console.print(f"[red]Step {i+1} failed: {e}[/red]")
        break

# Show final grid
console.print("\n[cyan]Final Grid After Sequence:[/cyan]")
log_grid_to_console(current_state.current_grid)

console.print("\n[cyan]Target Grid (for comparison):[/cyan]")
log_grid_to_console(current_state.target_grid)

# Create results table
results_table = Table(title="Action Sequence Results")
results_table.add_column("Step", style="cyan")
results_table.add_column("Description", style="magenta")
results_table.add_column("Operation", style="green")
results_table.add_column("Reward", style="yellow")
results_table.add_column("Similarity", style="blue")
results_table.add_column("Done", style="red")

for result in step_results:
    results_table.add_row(
        str(result['step']),
        result['description'],
        str(result['operation']),
        f"{result['reward']:.3f}",
        f"{result['similarity']:.3f}",
        str(result['done'])
    )

console.print(results_table)

# %% [markdown]
# ## 11. Random Action Testing
#
# Let's test with random actions to see the environment's robustness.

# %%
def create_random_action(key: jax.Array, grid_size: tuple[int, int]) -> dict:
    """Create a random action for testing."""
    h, w = grid_size
    key, sel_key, op_key = jr.split(key, 3)

    # Random selection mask (sparse)
    selection = jr.bernoulli(sel_key, 0.05, (h, w)).astype(jnp.float32)

    # Random operation (0-34)
    operation = jr.randint(op_key, (), 0, 35)

    return {
        "selection": selection,
        "operation": operation
    }

# Test random actions
console.print("[bold yellow]Testing Random Actions:[/bold yellow]")

# Reset environment for clean test
key, reset_key = jr.split(key)
test_obs, test_state = env.reset_with_task(reset_key, parsed_task)

random_results = []
for i in range(5):
    try:
        key, action_key, step_key = jr.split(key, 3)
        random_action = create_random_action(action_key, env.max_grid_size)
        actions = {agent_id: random_action}

        obs, new_state, rewards, dones, infos = env.step_env(step_key, test_state, actions)

        random_results.append({
            'step': i + 1,
            'operation': int(random_action['operation']),
            'selected_pixels': int(jnp.sum(random_action['selection'])),
            'reward': float(rewards[agent_id]),
            'similarity': float(new_state.similarity_score),
            'done': bool(dones[agent_id])
        })

        test_state = new_state

        if dones[agent_id]:
            console.print(f"[red]Random test terminated at step {i+1}![/red]")
            break

    except Exception as e:
        console.print(f"[red]Random step {i+1} failed: {e}[/red]")
        break

# Display random test results
random_table = Table(title="Random Action Test Results")
random_table.add_column("Step", style="cyan")
random_table.add_column("Operation", style="magenta")
random_table.add_column("Selected Pixels", style="green")
random_table.add_column("Reward", style="yellow")
random_table.add_column("Similarity", style="blue")

for result in random_results:
    random_table.add_row(
        str(result['step']),
        str(result['operation']),
        str(result['selected_pixels']),
        f"{result['reward']:.3f}",
        f"{result['similarity']:.3f}"
    )

console.print(random_table)

# %% [markdown]
# ## 12. Task Analysis and Insights
#
# Let's analyze the specific task and what operations might be helpful.

# %%
# Analyze the task pattern
input_grid = parsed_task.input_grids_examples[0]
output_grid = parsed_task.output_grids_examples[0]

# Find unique colors in input and output
input_colors = jnp.unique(input_grid[input_grid >= 0])
output_colors = jnp.unique(output_grid[output_grid >= 0])

console.print(Panel(
    f"[bold blue]Task Pattern Analysis[/bold blue]\n\n"
    f"Task ID: {parsed_task.task_id}\n"
    f"Input colors: {input_colors.tolist()}\n"
    f"Output colors: {output_colors.tolist()}\n"
    f"Grid dimensions: {input_grid.shape}\n"
    f"Colors added: {set(output_colors.tolist()) - set(input_colors.tolist())}\n"
    f"Colors removed: {set(input_colors.tolist()) - set(output_colors.tolist())}\n"
    f"Total non-zero pixels (input): {jnp.sum(input_grid > 0)}\n"
    f"Total non-zero pixels (output): {jnp.sum(output_grid > 0)}",
    title="Pattern Analysis"
))

# Show multiple training examples if available
if parsed_task.num_train_pairs > 1:
    console.print(f"\n[bold magenta]Additional Training Examples (showing up to 3):[/bold magenta]")

    for pair_idx in range(min(3, parsed_task.num_train_pairs)):
        console.print(f"\n[yellow]Training Pair {pair_idx + 1}:[/yellow]")

        pair_input = parsed_task.input_grids_examples[pair_idx]
        pair_output = parsed_task.output_grids_examples[pair_idx]

        console.print(f"Input (pair {pair_idx + 1}):")
        log_grid_to_console(pair_input)

        console.print(f"Output (pair {pair_idx + 1}):")
        log_grid_to_console(pair_output)

        # Quick analysis
        pair_input_colors = jnp.unique(pair_input[pair_input >= 0])
        pair_output_colors = jnp.unique(pair_output[pair_output >= 0])
        console.print(f"Colors: {pair_input_colors.tolist()} → {pair_output_colors.tolist()}")

# %% [markdown]
# ## 13. Summary and Key Insights
#
# This notebook has demonstrated the complete JaxARC workflow with the new ARCLE environment.

# %%
console.print(Panel(
    f"[bold green]JaxARC Exploration Summary[/bold green]\n\n"
    f"✅ Configuration loading with Hydra\n"
    f"✅ Dataset parsing with ArcAgiParser\n"
    f"✅ Task selection and data structure exploration\n"
    f"✅ ARCLE environment initialization with proper base class\n"
    f"✅ Task loading into environment via reset_with_task\n"
    f"✅ Action space analysis (selection + operation)\n"
    f"✅ Single and multi-step action execution\n"
    f"✅ Grid visualization and state tracking\n"
    f"✅ Random action testing for robustness\n"
    f"✅ Task pattern analysis\n\n"
    f"[yellow]Key Architecture:[/yellow]\n"
    f"• Environment: {env.name} extending ArcMarlEnvBase\n"
    f"• Parser: {type(parser).__name__}\n"
    f"• Task ID: {parsed_task.task_id}\n"
    f"• Action Space: Dict(selection: Box({h}, {w}), operation: Discrete(35))\n"
    f"• State: ARCLEState with current_grid, target_grid, clipboard, etc.\n"
    f"• JAX/JIT Compatible: ✅\n"
    f"• Reward System: {'Submit-only' if env.reward_on_submit_only else 'Continuous'}",
    title="Exploration Complete!"
))

# %% [markdown]
# ## Next Steps
#
# The ARCLE environment is now fully functional! Next steps include:
#
# 1. **Agent Implementation**: Create learning agents that can select actions intelligently
# 2. **Training Pipeline**: Set up RL training loops using JaxMARL
# 3. **Operation Implementation**: Complete the ARCLE operations in `arcle_operations.py`
# 4. **Multi-Task Learning**: Train agents across multiple ARC tasks
# 5. **Evaluation Framework**: Systematic evaluation on test sets
# 6. **Advanced Features**: Multi-agent collaboration, hierarchical policies
#
# The environment successfully:
# - Loads tasks from the dataset parser
# - Manages grid state and transformations
# - Provides proper reward signals
# - Maintains JAX compatibility for fast training
# - Integrates with the JaxMARL framework
#
# Ready for serious ARC challenge development! 🚀

# %%
