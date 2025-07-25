#!/usr/bin/env python3
"""
Multi-Demonstration Training Example

This example demonstrates the enhanced ARC environment's multi-demonstration training
capabilities, including:
- Training on multiple demonstration pairs within a single episode
- Dynamic pair switching using non-parametric control operations
- Episode management with configurable termination criteria
- Progress tracking across demonstration pairs

Requirements: 1.4, 2.5
"""

import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.factory import ConfigFactory
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.visualization import log_grid_to_console
from omegaconf import DictConfig


def create_multi_demo_config():
    """Create configuration optimized for multi-demonstration training."""
    return ConfigFactory.create_research_config(
        # Episode management for multi-demonstration training
        episode_mode="train",
        demo_selection_strategy="sequential",  # Go through demos in order
        allow_demo_switching=True,
        require_all_demos_solved=False,  # Don't require solving all demos
        terminate_on_first_success=False,  # Continue after solving one
        max_pairs_per_episode=4,  # Process up to 4 demo pairs per episode
        training_reward_frequency="step",  # Get rewards at each step
        
        # Action history for analysis
        history_enabled=True,
        max_history_length=1000,
        store_selection_data=True,
        
        # Enhanced action space
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        
        # Rich observation for training
        include_completion_status=True,
        include_action_space_info=True,
        observation_format="rich",
        
        # Debug settings
        debug_level="standard",
        visualization_enabled=True
    )


def load_multi_demo_task():
    """Load a task with multiple demonstration pairs."""
    parser_config = DictConfig({
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,  # Ensure we have multiple demo pairs
        "max_test_pairs": 2,
    })
    
    parser = ArcAgiParser(parser_config)
    
    # Find a task with multiple demonstration pairs
    task_ids = parser.get_available_task_ids()
    for task_id in task_ids[:10]:  # Check first 10 tasks
        task = parser.get_task_by_id(task_id)
        if task.num_train_pairs >= 3:  # Want at least 3 demo pairs
            print(f"Selected task {task_id} with {task.num_train_pairs} demonstration pairs")
            return task
    
    # Fallback to random task
    print("Using random task (may have fewer demo pairs)")
    return parser.get_random_task(jax.random.PRNGKey(42))


def demonstrate_pair_switching(state, config):
    """Demonstrate different pair switching operations."""
    print("\n=== Demonstrating Pair Switching Operations ===")
    
    # Show current state
    print(f"Current demo pair: {state.current_example_idx}")
    print(f"Available demo pairs: {jnp.where(state.available_demo_pairs)[0].tolist()}")
    print(f"Completion status: {state.demo_completion_status.tolist()}")
    
    # Switch to next demo pair
    print("\n1. Switching to next demo pair...")
    next_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),  # Selection ignored for control ops
        "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
    }
    
    state, observation, reward, done, info = arc_step(state, next_action, config)
    print(f"   Now on demo pair: {state.current_example_idx}")
    print(f"   Reward for switching: {reward}")
    
    # Do some work on this pair
    print("\n2. Working on current pair...")
    work_action = {
        "selection": jnp.zeros((30, 30), dtype=bool).at[5:10, 5:10].set(True),
        "operation": 0  # FILL operation
    }
    
    for i in range(3):
        state, observation, reward, done, info = arc_step(state, work_action, config)
        print(f"   Step {i+1}: reward = {reward:.3f}")
    
    # Switch to first unsolved demo
    print("\n3. Switching to first unsolved demo...")
    unsolved_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),
        "operation": 40  # SWITCH_TO_FIRST_UNSOLVED_DEMO
    }
    
    state, observation, reward, done, info = arc_step(state, unsolved_action, config)
    print(f"   Switched to demo pair: {state.current_example_idx}")
    
    # Reset current pair
    print("\n4. Resetting current pair...")
    reset_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),
        "operation": 39  # RESET_CURRENT_PAIR
    }
    
    state, observation, reward, done, info = arc_step(state, reset_action, config)
    print(f"   Pair reset, similarity score: {state.similarity_score:.3f}")
    
    return state


def run_multi_demo_training_episode(config, task, key):
    """Run a complete multi-demonstration training episode."""
    print("\n=== Multi-Demonstration Training Episode ===")
    
    # Reset in training mode
    state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
    
    print(f"Episode started in mode: {'train' if state.episode_mode == 0 else 'test'}")
    print(f"Initial demo pair: {state.current_example_idx}")
    print(f"Available demo pairs: {state.available_demo_pairs.sum()} total")
    print(f"Target grid available: {observation.target_grid is not None}")
    
    # Show initial grids
    print("\nInitial working grid:")
    log_grid_to_console(state.working_grid, "Working Grid")
    
    if observation.target_grid is not None:
        print("\nTarget grid:")
        log_grid_to_console(observation.target_grid, "Target Grid")
    
    total_reward = 0.0
    pairs_attempted = set()
    
    # Training loop with strategic pair switching
    for step in range(200):
        pairs_attempted.add(int(state.current_example_idx))
        
        # Strategy: Work on each pair for a while, then switch
        if step > 0 and step % 40 == 0 and state.available_demo_pairs.sum() > 1:
            # Switch to next available demo pair
            action = {
                "selection": jnp.zeros((30, 30), dtype=bool),
                "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
            }
            print(f"\nStep {step}: Switching to next demo pair")
        
        elif step > 0 and step % 80 == 0:
            # Occasionally switch to first unsolved demo
            action = {
                "selection": jnp.zeros((30, 30), dtype=bool),
                "operation": 40  # SWITCH_TO_FIRST_UNSOLVED_DEMO
            }
            print(f"\nStep {step}: Switching to first unsolved demo")
        
        else:
            # Regular grid operations
            # Simple strategy: fill different regions with different colors
            region_size = 3
            row_start = (step // 10) % (30 - region_size)
            col_start = (step // 5) % (30 - region_size)
            color = (step % 9) + 1  # Colors 1-9
            
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[
                row_start:row_start + region_size,
                col_start:col_start + region_size
            ].set(True)
            
            action = {
                "selection": selection,
                "operation": 0  # FILL operation
            }
        
        # Execute action
        state, observation, reward, done, info = arc_step(state, action, config)
        total_reward += reward
        
        # Log progress periodically
        if step % 50 == 0:
            print(f"Step {step}: pair={state.current_example_idx}, "
                  f"reward={reward:.3f}, total_reward={total_reward:.3f}")
            print(f"  Completion status: {state.demo_completion_status.tolist()}")
            print(f"  Action history length: {state.action_history_length}")
        
        # Check for episode completion
        if done:
            print(f"\nEpisode completed at step {step}")
            break
    
    print(f"\nTraining Episode Summary:")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Pairs attempted: {sorted(pairs_attempted)}")
    print(f"  Final completion status: {state.demo_completion_status.tolist()}")
    print(f"  Final action history length: {state.action_history_length}")
    
    return state, total_reward


def analyze_multi_demo_performance(state, config):
    """Analyze performance across multiple demonstration pairs."""
    print("\n=== Multi-Demo Performance Analysis ===")
    
    # Completion analysis
    completed_demos = state.demo_completion_status.sum()
    available_demos = state.available_demo_pairs.sum()
    
    print(f"Demonstration Pairs:")
    print(f"  Available: {available_demos}")
    print(f"  Completed: {completed_demos}")
    print(f"  Success rate: {completed_demos / available_demos * 100:.1f}%")
    
    # Action history analysis
    print(f"\nAction History:")
    print(f"  Total actions recorded: {state.action_history_length}")
    print(f"  History storage utilization: {state.action_history_length / config.history.max_history_length * 100:.1f}%")
    
    # Current state analysis
    print(f"\nFinal State:")
    print(f"  Current pair: {state.current_example_idx}")
    print(f"  Similarity score: {state.similarity_score:.3f}")
    print(f"  Episode done: {state.episode_done}")


def main():
    """Main demonstration of multi-demonstration training."""
    print("Multi-Demonstration Training Example")
    print("=" * 50)
    
    # Create configuration
    config = create_multi_demo_config()
    print("Created multi-demonstration training configuration")
    
    # Load task with multiple demo pairs
    task = load_multi_demo_task()
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Demonstrate pair switching operations
    state, _ = arc_reset(key, config, task_data=task, episode_mode="train")
    state = demonstrate_pair_switching(state, config)
    
    # Run complete training episode
    key = jax.random.split(key)[0]
    final_state, total_reward = run_multi_demo_training_episode(config, task, key)
    
    # Analyze performance
    analyze_multi_demo_performance(final_state, config)
    
    print("\n" + "=" * 50)
    print("Multi-demonstration training example completed!")
    print(f"Final total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()