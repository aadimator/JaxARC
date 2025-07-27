#!/usr/bin/env python3
"""
Restricted Action Space Example

This example demonstrates the enhanced ARC environment's action space control
capabilities, including:
- Dynamic action filtering based on context
- Custom operation restrictions for focused training
- Action validation and error handling policies
- Context-dependent operation availability

Requirements: 4.4
"""

import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.factory import ConfigFactory
from jaxarc.envs.action_space import ActionSpaceController
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig
from typing import Dict


def create_restricted_action_configs():
    """Create different restricted action space configurations."""
    
    configs = {}
    
    # Basic operations only (filling and simple transforms)
    configs["basic"] = ConfigFactory.create_research_config(
        # Restrict to basic operations
        allowed_operations=[0, 1, 2, 3, 4],  # FILL, CLEAR, etc.
        max_operations=5,
        dynamic_action_filtering=True,
        context_dependent_operations=False,
        validate_actions=True,
        allow_invalid_actions=False,  # Strict validation
        
        # Episode settings
        episode_mode="train",
        demo_selection_strategy="sequential",
        
        # Observation
        include_action_space_info=True,
        observation_format="standard"
    )
    
    # Movement operations only
    configs["movement"] = ConfigFactory.create_research_config(
        allowed_operations=[10, 11, 12, 13],  # MOVE operations
        max_operations=4,
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        validate_actions=True,
        allow_invalid_actions=False,
        
        episode_mode="train",
        include_action_space_info=True
    )
    
    # Transformation operations (rotation, reflection)
    configs["transform"] = ConfigFactory.create_research_config(
        allowed_operations=[20, 21, 22, 23, 24, 25, 26, 27],  # Transform ops
        max_operations=8,
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        validate_actions=True,
        allow_invalid_actions=True,  # Allow with penalty
        
        episode_mode="train",
        include_action_space_info=True
    )
    
    # Clipboard operations
    configs["clipboard"] = ConfigFactory.create_research_config(
        allowed_operations=[30, 31, 32],  # COPY, PASTE, CLEAR_CLIPBOARD
        max_operations=3,
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        validate_actions=True,
        allow_invalid_actions=False,
        
        episode_mode="train",
        include_action_space_info=True
    )
    
    # Control operations only (for multi-demo training)
    configs["control"] = ConfigFactory.create_research_config(
        allowed_operations=[34, 35, 36, 39, 40],  # SUBMIT, pair switching, reset
        max_operations=5,
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        validate_actions=True,
        allow_invalid_actions=False,
        
        episode_mode="train",
        allow_demo_switching=True,
        include_action_space_info=True
    )
    
    # Progressive learning (start with basic, expand over time)
    configs["progressive"] = ConfigFactory.create_research_config(
        allowed_operations=None,  # Will be set dynamically
        max_operations=42,
        dynamic_action_filtering=True,
        context_dependent_operations=True,
        validate_actions=True,
        allow_invalid_actions=True,  # Allow for learning
        
        episode_mode="train",
        include_action_space_info=True
    )
    
    return configs


def load_restriction_task():
    """Load a task suitable for action space restriction experiments."""
    parser_config = DictConfig({
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 3,
        "max_test_pairs": 2,
    })
    
    parser = ArcAgiParser(parser_config)
    return parser.get_random_task(jax.random.PRNGKey(42))


def demonstrate_basic_restrictions(config, task, key):
    """Demonstrate basic action space restrictions."""
    print(f"\n=== Basic Action Space Restrictions ===")
    
    # Initialize environment
    state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
    
    # Show allowed operations
    allowed_ops = jnp.where(observation.allowed_operations_mask)[0]
    print(f"Allowed operations: {allowed_ops.tolist()}")
    print(f"Total allowed: {len(allowed_ops)} out of 42")
    
    # Initialize action space controller
    action_controller = ActionSpaceController()
    
    # Test valid operations
    print(f"\nTesting valid operations:")
    for op_id in allowed_ops[:3]:  # Test first 3 allowed operations
        is_valid, error_msg = action_controller.validate_operation(
            int(op_id), state, config.action
        )
        print(f"  Operation {op_id}: {'Valid' if is_valid else f'Invalid - {error_msg}'}")
    
    # Test invalid operations
    print(f"\nTesting invalid operations:")
    all_ops = set(range(42))
    allowed_set = set(allowed_ops.tolist())
    invalid_ops = list(all_ops - allowed_set)[:3]  # Test first 3 invalid
    
    for op_id in invalid_ops:
        is_valid, error_msg = action_controller.validate_operation(
            op_id, state, config.action
        )
        print(f"  Operation {op_id}: {'Valid' if is_valid else f'Invalid - {error_msg}'}")
    
    # Execute some valid actions
    print(f"\nExecuting valid actions:")
    total_reward = 0.0
    
    for i, op_id in enumerate(allowed_ops[:5]):
        # Create selection
        selection = jnp.zeros((30, 30), dtype=bool)
        selection = selection.at[i*3:(i+1)*3, i*3:(i+1)*3].set(True)
        
        action = {"selection": selection, "operation": int(op_id)}
        
        state, observation, reward, done, info = arc_step(state, action, config)
        total_reward += reward
        
        print(f"  Step {i+1}: operation {op_id}, reward {reward:.3f}")
        
        if done:
            break
    
    print(f"Total reward with restricted actions: {total_reward:.3f}")
    return state


def demonstrate_context_dependent_restrictions(config, task, key):
    """Demonstrate context-dependent action restrictions."""
    print(f"\n=== Context-Dependent Action Restrictions ===")
    
    # Initialize environment
    state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
    
    action_controller = ActionSpaceController()
    
    # Show initial context
    print(f"Initial context:")
    print(f"  Episode mode: {'train' if state.episode_mode == 0 else 'test'}")
    print(f"  Current pair: {state.current_example_idx}")
    print(f"  Available demo pairs: {state.available_demo_pairs.sum()}")
    print(f"  Demo completion: {state.demo_completion_status.tolist()}")
    
    # Test pair switching operations in different contexts
    pair_switch_ops = [35, 36, 40]  # NEXT_DEMO, PREV_DEMO, FIRST_UNSOLVED_DEMO
    
    print(f"\nTesting pair switching operations:")
    for op_id in pair_switch_ops:
        is_valid, error_msg = action_controller.validate_operation(
            op_id, state, config.action
        )
        print(f"  Operation {op_id} (pair switch): {'Valid' if is_valid else f'Invalid - {error_msg}'}")
    
    # Modify state to have multiple demo pairs available
    if state.available_demo_pairs.sum() > 1:
        print(f"\nWith multiple demo pairs available:")
        
        # Test switching to next demo
        switch_action = {
            "selection": jnp.zeros((30, 30), dtype=bool),
            "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
        }
        
        old_pair = state.current_example_idx
        state, observation, reward, done, info = arc_step(state, switch_action, config)
        new_pair = state.current_example_idx
        
        print(f"  Switched from pair {old_pair} to pair {new_pair}")
        print(f"  Switch reward: {reward:.3f}")
        
        # Now test operations in new context
        print(f"\nOperations available in new context:")
        allowed_mask = action_controller.get_allowed_operations(state, config.action)
        allowed_ops = jnp.where(allowed_mask)[0]
        print(f"  Allowed operations: {len(allowed_ops)} total")
        
        # Test context-specific availability
        if state.available_demo_pairs.sum() > 1:
            print(f"  Pair switching still available: {35 in allowed_ops}")
        
        # Test reset operation (should be available if pair was modified)
        reset_available = 39 in allowed_ops  # RESET_CURRENT_PAIR
        print(f"  Pair reset available: {reset_available}")
    
    else:
        print(f"\nOnly one demo pair available - pair switching should be restricted")
    
    return state


def demonstrate_validation_policies(configs, task, key):
    """Demonstrate different action validation policies."""
    print(f"\n=== Action Validation Policies ===")
    
    policies = {
        "strict": configs["basic"],      # allow_invalid_actions=False
        "permissive": configs["transform"]  # allow_invalid_actions=True
    }
    
    for policy_name, config in policies.items():
        print(f"\n--- {policy_name.title()} Policy ---")
        
        # Initialize environment
        state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
        action_controller = ActionSpaceController()
        
        # Get allowed operations
        allowed_ops = jnp.where(observation.allowed_operations_mask)[0]
        print(f"Allowed operations: {allowed_ops.tolist()}")
        
        # Try an invalid operation
        all_ops = set(range(42))
        allowed_set = set(allowed_ops.tolist())
        invalid_ops = list(all_ops - allowed_set)
        
        if invalid_ops:
            invalid_op = invalid_ops[0]
            print(f"Testing invalid operation {invalid_op}:")
            
            # Create action with invalid operation
            action = {
                "selection": jnp.zeros((30, 30), dtype=bool).at[0:5, 0:5].set(True),
                "operation": invalid_op
            }
            
            try:
                # Filter the invalid operation
                filtered_op = action_controller.filter_invalid_operation(
                    invalid_op, state, config.action
                )
                
                print(f"  Original operation: {invalid_op}")
                print(f"  Filtered operation: {filtered_op}")
                
                # Execute with filtered operation
                filtered_action = {
                    "selection": action["selection"],
                    "operation": filtered_op
                }
                
                state, observation, reward, done, info = arc_step(state, filtered_action, config)
                print(f"  Execution result: reward={reward:.3f}, done={done}")
                
            except Exception as e:
                print(f"  Exception raised: {type(e).__name__}: {e}")
        
        key = jax.random.split(key)[0]  # New key for next policy


def demonstrate_progressive_learning(config, task, key):
    """Demonstrate progressive action space expansion."""
    print(f"\n=== Progressive Action Space Learning ===")
    
    # Define learning phases
    phases = [
        {"name": "Basic", "operations": [0, 1, 2, 3], "steps": 20},
        {"name": "Movement", "operations": [0, 1, 2, 3, 10, 11, 12, 13], "steps": 20},
        {"name": "Transform", "operations": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23], "steps": 20},
        {"name": "Full", "operations": None, "steps": 20}  # None = all operations
    ]
    
    # Initialize environment
    state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
    action_controller = ActionSpaceController()
    
    total_reward = 0.0
    
    for phase_idx, phase in enumerate(phases):
        print(f"\n--- Phase {phase_idx + 1}: {phase['name']} ---")
        
        # Update allowed operations for this phase
        if phase["operations"] is not None:
            # Create new config with restricted operations
            phase_config = ConfigFactory.create_research_config(
                allowed_operations=phase["operations"],
                max_operations=len(phase["operations"]),
                dynamic_action_filtering=True,
                context_dependent_operations=True,
                validate_actions=True,
                allow_invalid_actions=True,
                episode_mode="train",
                include_action_space_info=True
            )
        else:
            # Use full action space
            phase_config = ConfigFactory.create_research_config(
                allowed_operations=None,
                max_operations=42,
                dynamic_action_filtering=True,
                context_dependent_operations=True,
                validate_actions=True,
                allow_invalid_actions=True,
                episode_mode="train",
                include_action_space_info=True
            )
        
        # Get allowed operations for this phase
        allowed_mask = action_controller.get_allowed_operations(state, phase_config.action)
        allowed_ops = jnp.where(allowed_mask)[0]
        
        print(f"  Allowed operations: {allowed_ops.tolist()}")
        print(f"  Operation count: {len(allowed_ops)}")
        
        # Execute actions in this phase
        phase_reward = 0.0
        for step in range(phase["steps"]):
            # Select random allowed operation
            if len(allowed_ops) > 0:
                op_idx = step % len(allowed_ops)
                operation = int(allowed_ops[op_idx])
                
                # Create selection
                selection = jnp.zeros((30, 30), dtype=bool)
                row_start = (step * 2) % 25
                col_start = (step * 3) % 25
                selection = selection.at[row_start:row_start+3, col_start:col_start+3].set(True)
                
                action = {"selection": selection, "operation": operation}
                
                # Execute action
                state, observation, reward, done, info = arc_step(state, action, phase_config)
                phase_reward += reward
                total_reward += reward
                
                if done:
                    print(f"  Episode completed at step {step + 1}")
                    break
        
        print(f"  Phase reward: {phase_reward:.3f}")
        print(f"  Cumulative reward: {total_reward:.3f}")
    
    print(f"\nProgressive learning completed:")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Final similarity: {state.similarity_score:.3f}")
    
    return state, total_reward


def analyze_restriction_effectiveness(results: Dict[str, float]):
    """Analyze the effectiveness of different action space restrictions."""
    print(f"\n=== Restriction Effectiveness Analysis ===")
    
    print(f"Performance Comparison:")
    print(f"{'Restriction':<15} {'Reward':<10} {'Relative':<10}")
    print("-" * 35)
    
    # Find baseline (unrestricted) performance
    baseline = results.get("unrestricted", 0.0)
    
    for restriction, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        relative = (reward / baseline * 100) if baseline > 0 else 0
        print(f"{restriction:<15} {reward:<10.3f} {relative:<10.1f}%")
    
    print(f"\nInsights:")
    print(f"  - Restrictions can focus learning on specific skills")
    print(f"  - Progressive expansion balances exploration and exploitation")
    print(f"  - Context-dependent restrictions prevent invalid actions")
    print(f"  - Validation policies affect learning dynamics")


def main():
    """Main demonstration of restricted action space functionality."""
    print("Restricted Action Space Example")
    print("=" * 50)
    
    # Create different restriction configurations
    configs = create_restricted_action_configs()
    print(f"Created {len(configs)} different action space configurations")
    
    # Load task
    task = load_restriction_task()
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Results storage
    results = {}
    
    # Demonstrate basic restrictions
    print(f"\n{'='*20} BASIC RESTRICTIONS {'='*20}")
    key = jax.random.split(key)[0]
    state = demonstrate_basic_restrictions(configs["basic"], task, key)
    results["basic"] = float(state.similarity_score)
    
    # Demonstrate context-dependent restrictions
    print(f"\n{'='*15} CONTEXT-DEPENDENT RESTRICTIONS {'='*15}")
    key = jax.random.split(key)[0]
    state = demonstrate_context_dependent_restrictions(configs["control"], task, key)
    results["context"] = float(state.similarity_score)
    
    # Demonstrate validation policies
    print(f"\n{'='*20} VALIDATION POLICIES {'='*20}")
    key = jax.random.split(key)[0]
    demonstrate_validation_policies(configs, task, key)
    
    # Demonstrate progressive learning
    print(f"\n{'='*20} PROGRESSIVE LEARNING {'='*20}")
    key = jax.random.split(key)[0]
    final_state, total_reward = demonstrate_progressive_learning(configs["progressive"], task, key)
    results["progressive"] = total_reward
    
    # Test different restriction types
    restriction_types = ["movement", "transform", "clipboard"]
    for restriction in restriction_types:
        print(f"\n--- Testing {restriction} restrictions ---")
        key = jax.random.split(key)[0]
        state, _ = arc_reset(key, configs[restriction], task_data=task, episode_mode="train")
        
        # Quick test with restricted operations
        for step in range(10):
            allowed_ops = jnp.where(state.allowed_operations_mask)[0]
            if len(allowed_ops) > 0:
                op = int(allowed_ops[step % len(allowed_ops)])
                selection = jnp.zeros((30, 30), dtype=bool).at[step:step+2, step:step+2].set(True)
                action = {"selection": selection, "operation": op}
                state, _, reward, done, _ = arc_step(state, action, configs[restriction])
                if done:
                    break
        
        results[restriction] = float(state.similarity_score)
    
    # Add unrestricted baseline
    print(f"\n--- Testing unrestricted baseline ---")
    unrestricted_config = ConfigFactory.create_research_config(
        allowed_operations=None,  # All operations allowed
        dynamic_action_filtering=False,
        validate_actions=False
    )
    key = jax.random.split(key)[0]
    state, _ = arc_reset(key, unrestricted_config, task_data=task, episode_mode="train")
    
    for step in range(20):
        op = step % 35  # Use original 35 operations
        selection = jnp.zeros((30, 30), dtype=bool).at[step:step+2, step:step+2].set(True)
        action = {"selection": selection, "operation": op}
        state, _, reward, done, _ = arc_step(state, action, unrestricted_config)
        if done:
            break
    
    results["unrestricted"] = float(state.similarity_score)
    
    # Analyze effectiveness
    analyze_restriction_effectiveness(results)
    
    print("\n" + "=" * 50)
    print("Restricted action space example completed!")
    print(f"Best performing restriction: {max(results.items(), key=lambda x: x[1])}")


if __name__ == "__main__":
    main()