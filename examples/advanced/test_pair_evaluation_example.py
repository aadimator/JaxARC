#!/usr/bin/env python3
"""
Test Pair Evaluation Example

This example demonstrates the enhanced ARC environment's test pair evaluation
capabilities, including:
- Evaluation mode without access to target grids
- Test pair switching and management
- Alternative reward signals during evaluation
- Proper separation of training and evaluation phases

Requirements: 1.4, 2.5
"""

import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.factory import ConfigFactory
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.visualization import log_grid_to_console
from omegaconf import DictConfig


def create_evaluation_config():
    """Create configuration optimized for test pair evaluation."""
    return ConfigFactory.create_research_config(
        # Episode management for evaluation
        episode_mode="test",
        test_selection_strategy="sequential",  # Go through test pairs in order
        allow_test_switching=True,
        require_all_tests_solved=True,  # Try to solve all test pairs
        terminate_on_first_success=False,  # Continue after solving one
        max_pairs_per_episode=4,  # Process all available test pairs
        evaluation_reward_frequency="submit",  # Only reward on submission
        
        # Minimal action history for evaluation
        history_enabled=True,
        max_history_length=500,
        store_selection_data=False,  # Save memory during evaluation
        
        # Standard action space for evaluation
        dynamic_action_filtering=False,  # Let agent try all operations
        context_dependent_operations=True,
        
        # Focused observation for evaluation
        include_completion_status=True,
        include_action_space_info=False,  # Don't give hints about allowed ops
        observation_format="standard",
        
        # Minimal debug for evaluation
        debug_level="minimal",
        visualization_enabled=False
    )


def load_test_task():
    """Load a task with multiple test pairs."""
    parser_config = DictConfig({
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,  # Ensure we have multiple test pairs
    })
    
    parser = ArcAgiParser(parser_config)
    
    # Find a task with multiple test pairs
    task_ids = parser.get_available_task_ids()
    for task_id in task_ids[:10]:  # Check first 10 tasks
        task = parser.get_task_by_id(task_id)
        if task.num_test_pairs >= 2:  # Want at least 2 test pairs
            print(f"Selected task {task_id} with {task.num_test_pairs} test pairs")
            return task
    
    # Fallback to random task
    print("Using random task (may have fewer test pairs)")
    return parser.get_random_task(jax.random.PRNGKey(42))


def demonstrate_training_phase(config, task, key):
    """Demonstrate training phase before evaluation."""
    print("\n=== Training Phase ===")
    
    # Create training configuration
    train_config = ConfigFactory.create_research_config(
        episode_mode="train",
        demo_selection_strategy="random",
        allow_demo_switching=True,
        max_pairs_per_episode=2,  # Quick training
        training_reward_frequency="step"
    )
    
    # Reset in training mode
    state, observation = arc_reset(key, train_config, task_data=task, episode_mode="train")
    
    print(f"Training on demo pair: {state.current_example_idx}")
    print(f"Available demo pairs: {state.available_demo_pairs.sum()}")
    print(f"Target grid available: {observation.target_grid is not None}")
    
    # Show training grids
    print("\nTraining input grid:")
    log_grid_to_console(state.working_grid, "Input Grid")
    
    if observation.target_grid is not None:
        print("\nTraining target grid:")
        log_grid_to_console(observation.target_grid, "Target Grid")
    
    # Quick training simulation
    total_reward = 0.0
    for step in range(50):
        # Simple training strategy
        if step < 25:
            # Fill some regions
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[step:step+2, step:step+2].set(True)
            action = {"selection": selection, "operation": 0}  # FILL
        else:
            # Try different operations
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[step-25:step-23, step-25:step-23].set(True)
            action = {"selection": selection, "operation": (step % 10) + 1}
        
        state, observation, reward, done, info = arc_step(state, action, train_config)
        total_reward += reward
        
        if done:
            break
    
    print(f"Training completed: {step + 1} steps, total reward: {total_reward:.3f}")
    print(f"Final similarity score: {state.similarity_score:.3f}")
    
    return state


def demonstrate_test_switching(state, config):
    """Demonstrate test pair switching operations."""
    print("\n=== Test Pair Switching Operations ===")
    
    # Show current state
    print(f"Current test pair: {state.current_example_idx}")
    print(f"Available test pairs: {jnp.where(state.available_test_pairs)[0].tolist()}")
    print(f"Test completion status: {state.test_completion_status.tolist()}")
    
    # Switch to next test pair
    print("\n1. Switching to next test pair...")
    next_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),
        "operation": 37  # SWITCH_TO_NEXT_TEST_PAIR
    }
    
    state, observation, reward, done, info = arc_step(state, next_action, config)
    print(f"   Now on test pair: {state.current_example_idx}")
    print(f"   Reward for switching: {reward}")
    print(f"   Target grid available: {observation.target_grid is not None}")  # Should be None
    
    # Work on this test pair
    print("\n2. Working on current test pair...")
    work_action = {
        "selection": jnp.zeros((30, 30), dtype=bool).at[10:15, 10:15].set(True),
        "operation": 2  # Different operation
    }
    
    for i in range(3):
        state, observation, reward, done, info = arc_step(state, work_action, config)
        print(f"   Step {i+1}: reward = {reward:.3f} (should be 0 until submission)")
    
    # Switch to first unsolved test
    print("\n3. Switching to first unsolved test...")
    unsolved_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),
        "operation": 41  # SWITCH_TO_FIRST_UNSOLVED_TEST
    }
    
    state, observation, reward, done, info = arc_step(state, unsolved_action, config)
    print(f"   Switched to test pair: {state.current_example_idx}")
    
    return state


def run_evaluation_episode(config, task, key):
    """Run a complete test pair evaluation episode."""
    print("\n=== Test Pair Evaluation Episode ===")
    
    # Reset in evaluation mode
    state, observation = arc_reset(key, config, task_data=task, episode_mode="test")
    
    print(f"Episode started in mode: {'train' if state.episode_mode == 0 else 'test'}")
    print(f"Initial test pair: {state.current_example_idx}")
    print(f"Available test pairs: {state.available_test_pairs.sum()} total")
    print(f"Target grid available: {observation.target_grid is not None}")  # Should be None
    
    # Show initial test input
    print("\nTest input grid:")
    log_grid_to_console(state.working_grid, "Test Input")
    
    total_reward = 0.0
    pairs_attempted = set()
    submission_rewards = []
    
    # Evaluation loop with test pair switching
    for step in range(150):
        pairs_attempted.add(int(state.current_example_idx))
        
        # Strategy: Work on each test pair, then submit and switch
        if step > 0 and step % 30 == 0 and state.available_test_pairs.sum() > 1:
            # Submit current solution
            submit_action = {
                "selection": jnp.ones((30, 30), dtype=bool),  # Select all
                "operation": 34  # SUBMIT operation
            }
            print(f"\nStep {step}: Submitting solution for test pair {state.current_example_idx}")
            
            state, observation, reward, done, info = arc_step(state, submit_action, config)
            submission_rewards.append(reward)
            total_reward += reward
            print(f"   Submission reward: {reward:.3f}")
            
            # Switch to next test pair
            if not done and state.available_test_pairs.sum() > 1:
                switch_action = {
                    "selection": jnp.zeros((30, 30), dtype=bool),
                    "operation": 37  # SWITCH_TO_NEXT_TEST_PAIR
                }
                state, observation, reward, done, info = arc_step(state, switch_action, config)
                print(f"   Switched to test pair: {state.current_example_idx}")
        
        else:
            # Regular grid operations for test solving
            # Strategy: systematic pattern filling
            region_size = 2
            row_start = (step // 8) % (30 - region_size)
            col_start = (step // 4) % (30 - region_size)
            color = ((step // 2) % 9) + 1  # Colors 1-9
            
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[
                row_start:row_start + region_size,
                col_start:col_start + region_size
            ].set(True)
            
            # Vary operations for different test strategies
            operation = [0, 1, 2, 5, 8][step % 5]  # Different operations
            
            action = {
                "selection": selection,
                "operation": operation
            }
        
        # Execute action
        state, observation, reward, done, info = arc_step(state, action, config)
        
        # Note: reward should be 0 for non-submission actions in evaluation mode
        if action.get("operation") != 34:  # Not a submission
            assert reward == 0.0, f"Expected 0 reward for non-submission, got {reward}"
        
        # Log progress periodically
        if step % 40 == 0:
            print(f"Step {step}: test_pair={state.current_example_idx}, "
                  f"total_reward={total_reward:.3f}")
            print(f"  Test completion status: {state.test_completion_status.tolist()}")
        
        # Check for episode completion
        if done:
            print(f"\nEvaluation episode completed at step {step}")
            break
    
    # Final submission if not done
    if not done:
        final_submit = {
            "selection": jnp.ones((30, 30), dtype=bool),
            "operation": 34  # SUBMIT
        }
        state, observation, reward, done, info = arc_step(state, final_submit, config)
        submission_rewards.append(reward)
        total_reward += reward
        print(f"Final submission reward: {reward:.3f}")
    
    print(f"\nEvaluation Episode Summary:")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Test pairs attempted: {sorted(pairs_attempted)}")
    print(f"  Submission rewards: {[f'{r:.3f}' for r in submission_rewards]}")
    print(f"  Final completion status: {state.test_completion_status.tolist()}")
    
    return state, total_reward, submission_rewards


def analyze_evaluation_performance(state, config, submission_rewards):
    """Analyze evaluation performance across test pairs."""
    print("\n=== Evaluation Performance Analysis ===")
    
    # Completion analysis
    completed_tests = state.test_completion_status.sum()
    available_tests = state.available_test_pairs.sum()
    
    print(f"Test Pairs:")
    print(f"  Available: {available_tests}")
    print(f"  Completed: {completed_tests}")
    print(f"  Success rate: {completed_tests / available_tests * 100:.1f}%")
    
    # Submission analysis
    print(f"\nSubmission Analysis:")
    print(f"  Number of submissions: {len(submission_rewards)}")
    print(f"  Average submission reward: {jnp.mean(jnp.array(submission_rewards)):.3f}")
    print(f"  Best submission reward: {max(submission_rewards) if submission_rewards else 0:.3f}")
    
    # Evaluation-specific metrics
    print(f"\nEvaluation Metrics:")
    print(f"  Final test pair: {state.current_example_idx}")
    print(f"  Final similarity score: {state.similarity_score:.3f}")
    print(f"  Action history length: {state.action_history_length}")
    
    # Verify no target access during evaluation
    print(f"\nEvaluation Integrity:")
    print(f"  Episode mode: {'test' if state.episode_mode == 1 else 'train'}")
    print(f"  Target access prevented: {state.episode_mode == 1}")


def compare_train_vs_eval_modes(task, key):
    """Compare training and evaluation modes side by side."""
    print("\n=== Training vs Evaluation Mode Comparison ===")
    
    # Training mode
    train_config = ConfigFactory.create_research_config(episode_mode="train")
    train_state, train_obs = arc_reset(key, train_config, task_data=task, episode_mode="train")
    
    # Evaluation mode
    eval_config = ConfigFactory.create_research_config(episode_mode="test")
    eval_state, eval_obs = arc_reset(key, eval_config, task_data=task, episode_mode="test")
    
    print("Training Mode:")
    print(f"  Episode mode: {train_state.episode_mode} (0=train)")
    print(f"  Current pair: {train_state.current_example_idx}")
    print(f"  Target available: {train_obs.target_grid is not None}")
    print(f"  Available demo pairs: {train_state.available_demo_pairs.sum()}")
    
    print("\nEvaluation Mode:")
    print(f"  Episode mode: {eval_state.episode_mode} (1=test)")
    print(f"  Current pair: {eval_state.current_example_idx}")
    print(f"  Target available: {eval_obs.target_grid is not None}")
    print(f"  Available test pairs: {eval_state.available_test_pairs.sum()}")
    
    # Test reward differences
    test_action = {
        "selection": jnp.zeros((30, 30), dtype=bool).at[0:5, 0:5].set(True),
        "operation": 0
    }
    
    # Training step
    train_state, train_obs, train_reward, _, _ = arc_step(train_state, test_action, train_config)
    
    # Evaluation step
    eval_state, eval_obs, eval_reward, _, _ = arc_step(eval_state, test_action, eval_config)
    
    print(f"\nReward Comparison (same action):")
    print(f"  Training reward: {train_reward:.3f}")
    print(f"  Evaluation reward: {eval_reward:.3f} (should be 0 for non-submission)")


def main():
    """Main demonstration of test pair evaluation."""
    print("Test Pair Evaluation Example")
    print("=" * 50)
    
    # Create evaluation configuration
    config = create_evaluation_config()
    print("Created test pair evaluation configuration")
    
    # Load task with test pairs
    task = load_test_task()
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Demonstrate training phase first
    demonstrate_training_phase(config, task, key)
    
    # Compare training vs evaluation modes
    key = jax.random.split(key)[0]
    compare_train_vs_eval_modes(task, key)
    
    # Reset for evaluation
    key = jax.random.split(key)[0]
    state, _ = arc_reset(key, config, task_data=task, episode_mode="test")
    
    # Demonstrate test pair switching
    state = demonstrate_test_switching(state, config)
    
    # Run complete evaluation episode
    key = jax.random.split(key)[0]
    final_state, total_reward, submission_rewards = run_evaluation_episode(config, task, key)
    
    # Analyze evaluation performance
    analyze_evaluation_performance(final_state, config, submission_rewards)
    
    print("\n" + "=" * 50)
    print("Test pair evaluation example completed!")
    print(f"Final evaluation reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()