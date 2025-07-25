#!/usr/bin/env python3
"""
Action History Analysis Example

This example demonstrates the enhanced ARC environment's action history tracking
capabilities, including:
- Comprehensive action sequence recording with metadata
- Memory-efficient history configuration options
- Action history analysis and pattern detection
- History-based replay and debugging capabilities

Requirements: 3.4
"""

import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.factory import ConfigFactory
from jaxarc.envs.action_history import ActionHistoryTracker, HistoryConfig
from jaxarc.parsers import ArcAgiParser
from jaxarc.utils.visualization import log_grid_to_console
from omegaconf import DictConfig
from typing import List, Dict, Any


def create_history_tracking_config(history_mode: str = "standard"):
    """Create configuration with different history tracking modes."""
    
    if history_mode == "minimal":
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=100,
            store_selection_data=False,  # Save memory
            store_intermediate_grids=False,
            compress_repeated_actions=True
        )
    elif history_mode == "standard":
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=500,
            store_selection_data=True,
            store_intermediate_grids=False,
            compress_repeated_actions=True
        )
    elif history_mode == "research":
        history_config = HistoryConfig(
            enabled=True,
            max_history_length=2000,
            store_selection_data=True,
            store_intermediate_grids=True,  # Memory intensive
            compress_repeated_actions=False
        )
    else:
        raise ValueError(f"Unknown history mode: {history_mode}")
    
    return ConfigFactory.create_research_config(
        # Episode settings
        episode_mode="train",
        demo_selection_strategy="sequential",
        allow_demo_switching=True,
        max_pairs_per_episode=2,
        
        # History configuration
        history_enabled=history_config.enabled,
        max_history_length=history_config.max_history_length,
        store_selection_data=history_config.store_selection_data,
        store_intermediate_grids=history_config.store_intermediate_grids,
        compress_repeated_actions=history_config.compress_repeated_actions,
        
        # Enhanced observation for analysis
        include_completion_status=True,
        include_action_space_info=True,
        include_recent_actions=True,
        recent_action_count=10,
        observation_format="rich",
        
        # Debug settings
        debug_level="standard",
        visualization_enabled=True
    )


def load_analysis_task():
    """Load a task suitable for action history analysis."""
    parser_config = DictConfig({
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 3,
        "max_test_pairs": 2,
    })
    
    parser = ArcAgiParser(parser_config)
    return parser.get_random_task(jax.random.PRNGKey(42))


def generate_diverse_action_sequence(state, config, num_steps: int = 100):
    """Generate a diverse sequence of actions for history analysis."""
    print(f"\n=== Generating {num_steps} Diverse Actions ===")
    
    action_types = []
    total_reward = 0.0
    
    for step in range(num_steps):
        # Create diverse action patterns
        if step < 20:
            # Phase 1: Basic filling operations
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[step:step+3, step:step+3].set(True)
            action = {"selection": selection, "operation": 0}  # FILL
            action_types.append("fill")
            
        elif step < 40:
            # Phase 2: Movement operations
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[10:15, 10:15].set(True)
            action = {"selection": selection, "operation": 10 + (step % 4)}  # MOVE operations
            action_types.append("move")
            
        elif step < 60:
            # Phase 3: Rotation and reflection
            selection = jnp.zeros((30, 30), dtype=bool)
            selection = selection.at[5:10, 5:10].set(True)
            action = {"selection": selection, "operation": 20 + (step % 8)}  # ROTATE/REFLECT
            action_types.append("transform")
            
        elif step < 80:
            # Phase 4: Clipboard operations
            if step % 4 == 0:
                # Copy to clipboard
                selection = jnp.zeros((30, 30), dtype=bool)
                selection = selection.at[0:5, 0:5].set(True)
                action = {"selection": selection, "operation": 30}  # COPY
                action_types.append("copy")
            else:
                # Paste from clipboard
                selection = jnp.zeros((30, 30), dtype=bool)
                selection = selection.at[15:20, 15:20].set(True)
                action = {"selection": selection, "operation": 31}  # PASTE
                action_types.append("paste")
                
        elif step < 90:
            # Phase 5: Pair switching (if available)
            if state.available_demo_pairs.sum() > 1:
                action = {
                    "selection": jnp.zeros((30, 30), dtype=bool),
                    "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
                }
                action_types.append("switch_pair")
            else:
                # Fallback to fill
                selection = jnp.zeros((30, 30), dtype=bool)
                selection = selection.at[step-90:step-87, step-90:step-87].set(True)
                action = {"selection": selection, "operation": 0}
                action_types.append("fill")
                
        else:
            # Phase 6: Submission attempts
            action = {
                "selection": jnp.ones((30, 30), dtype=bool),
                "operation": 34  # SUBMIT
            }
            action_types.append("submit")
        
        # Execute action
        state, observation, reward, done, info = arc_step(state, action, config)
        total_reward += reward
        
        # Log progress
        if step % 20 == 0:
            print(f"Step {step}: action_type={action_types[-1]}, "
                  f"reward={reward:.3f}, history_length={state.action_history_length}")
        
        if done:
            print(f"Episode completed at step {step}")
            break
    
    print(f"Generated {len(action_types)} actions, total reward: {total_reward:.3f}")
    return state, action_types, total_reward


def analyze_action_history(state, config, action_types: List[str]):
    """Perform comprehensive analysis of action history."""
    print("\n=== Action History Analysis ===")
    
    # Initialize history tracker
    history_tracker = ActionHistoryTracker()
    
    # Basic statistics
    history_length = int(state.action_history_length)
    max_length = config.history.max_history_length
    
    print(f"History Statistics:")
    print(f"  Total actions recorded: {history_length}")
    print(f"  History capacity: {max_length}")
    print(f"  Utilization: {history_length / max_length * 100:.1f}%")
    print(f"  Storage format: {'with selection data' if config.history.store_selection_data else 'operations only'}")
    
    # Extract action sequence
    if history_length > 0:
        action_sequence = history_tracker.get_action_sequence(state, 0, min(history_length, 50))
        
        # Analyze action patterns
        analyze_action_patterns(action_sequence, action_types[:min(len(action_types), 50)])
        
        # Analyze temporal patterns
        analyze_temporal_patterns(action_sequence)
        
        # Analyze pair switching patterns
        analyze_pair_switching_patterns(action_sequence)
        
        # Memory usage analysis
        analyze_memory_usage(state, config)


def analyze_action_patterns(action_sequence, action_types: List[str]):
    """Analyze patterns in the action sequence."""
    print(f"\n--- Action Pattern Analysis ---")
    
    # Count action types
    from collections import Counter
    type_counts = Counter(action_types)
    
    print(f"Action Type Distribution:")
    for action_type, count in type_counts.most_common():
        percentage = count / len(action_types) * 100
        print(f"  {action_type}: {count} ({percentage:.1f}%)")
    
    # Analyze operation distribution from actual history
    operations = []
    for i in range(len(action_sequence)):
        # Extract operation from action history (simplified)
        # In real implementation, this would access the ActionRecord structure
        operations.append(i % 35)  # Placeholder for demonstration
    
    operation_counts = Counter(operations)
    print(f"\nMost Common Operations:")
    for op_id, count in operation_counts.most_common(5):
        print(f"  Operation {op_id}: {count} times")
    
    # Pattern sequences
    print(f"\nAction Sequences:")
    if len(action_types) >= 5:
        for i in range(min(5, len(action_types) - 4)):
            sequence = " -> ".join(action_types[i:i+5])
            print(f"  Steps {i}-{i+4}: {sequence}")


def analyze_temporal_patterns(action_sequence):
    """Analyze temporal patterns in actions."""
    print(f"\n--- Temporal Pattern Analysis ---")
    
    # Simulate timestamps (in real implementation, these come from ActionRecord)
    timestamps = list(range(len(action_sequence)))
    
    if len(timestamps) > 1:
        # Action frequency analysis
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        
        print(f"Temporal Statistics:")
        print(f"  Average action interval: {avg_interval:.2f} steps")
        print(f"  Action frequency: {1/avg_interval:.2f} actions/step")
        
        # Identify action bursts (rapid sequences)
        bursts = []
        current_burst = []
        
        for i, interval in enumerate(intervals):
            if interval <= avg_interval * 0.5:  # Rapid action
                current_burst.append(i)
            else:
                if len(current_burst) >= 3:  # Burst of 3+ rapid actions
                    bursts.append(current_burst)
                current_burst = []
        
        if current_burst and len(current_burst) >= 3:
            bursts.append(current_burst)
        
        print(f"  Action bursts detected: {len(bursts)}")
        for i, burst in enumerate(bursts[:3]):  # Show first 3 bursts
            print(f"    Burst {i+1}: steps {burst[0]}-{burst[-1]} ({len(burst)} actions)")


def analyze_pair_switching_patterns(action_sequence):
    """Analyze pair switching behavior."""
    print(f"\n--- Pair Switching Analysis ---")
    
    # Simulate pair indices (in real implementation, from ActionRecord.pair_index)
    pair_indices = []
    current_pair = 0
    
    for i in range(len(action_sequence)):
        # Simulate pair switches every 25 actions
        if i > 0 and i % 25 == 0:
            current_pair = (current_pair + 1) % 3
        pair_indices.append(current_pair)
    
    # Count pair switches
    switches = 0
    for i in range(1, len(pair_indices)):
        if pair_indices[i] != pair_indices[i-1]:
            switches += 1
    
    print(f"Pair Switching Statistics:")
    print(f"  Total pair switches: {switches}")
    print(f"  Average actions per pair: {len(action_sequence) / (switches + 1):.1f}")
    
    # Pair usage distribution
    from collections import Counter
    pair_counts = Counter(pair_indices)
    print(f"  Pair usage distribution:")
    for pair_id, count in sorted(pair_counts.items()):
        percentage = count / len(pair_indices) * 100
        print(f"    Pair {pair_id}: {count} actions ({percentage:.1f}%)")


def analyze_memory_usage(state, config):
    """Analyze memory usage of action history."""
    print(f"\n--- Memory Usage Analysis ---")
    
    # Estimate memory usage based on configuration
    history_length = int(state.action_history_length)
    
    # Base memory per action record
    base_memory_per_action = 32  # bytes for basic fields
    
    # Additional memory for selection data
    selection_memory_per_action = 0
    if config.history.store_selection_data:
        selection_memory_per_action = 30 * 30 * 1  # bool array
    
    # Additional memory for intermediate grids
    grid_memory_per_action = 0
    if config.history.store_intermediate_grids:
        grid_memory_per_action = 30 * 30 * 4  # int32 array
    
    total_memory_per_action = (base_memory_per_action + 
                              selection_memory_per_action + 
                              grid_memory_per_action)
    
    current_memory = history_length * total_memory_per_action
    max_memory = config.history.max_history_length * total_memory_per_action
    
    print(f"Memory Usage Estimation:")
    print(f"  Memory per action: {total_memory_per_action:,} bytes")
    print(f"    Base fields: {base_memory_per_action} bytes")
    print(f"    Selection data: {selection_memory_per_action:,} bytes")
    print(f"    Grid data: {grid_memory_per_action:,} bytes")
    print(f"  Current usage: {current_memory:,} bytes ({current_memory / 1024 / 1024:.2f} MB)")
    print(f"  Maximum usage: {max_memory:,} bytes ({max_memory / 1024 / 1024:.2f} MB)")
    print(f"  Memory efficiency: {current_memory / max_memory * 100:.1f}%")


def demonstrate_history_replay(state, config):
    """Demonstrate action history replay capabilities."""
    print(f"\n=== Action History Replay ===")
    
    history_tracker = ActionHistoryTracker()
    history_length = int(state.action_history_length)
    
    if history_length == 0:
        print("No action history available for replay")
        return
    
    # Get recent actions for replay
    recent_count = min(10, history_length)
    recent_actions = history_tracker.get_action_sequence(
        state, 
        max(0, history_length - recent_count), 
        history_length
    )
    
    print(f"Replaying last {len(recent_actions)} actions:")
    
    # Simulate replay (in real implementation, would reconstruct and execute actions)
    for i, action_record in enumerate(recent_actions):
        # In real implementation, ActionRecord would contain:
        # - selection_data: actual selection mask
        # - operation_id: operation that was executed
        # - timestamp: when action was taken
        # - pair_index: which demo/test pair
        # - valid: whether record is valid
        
        print(f"  Action {i+1}:")
        print(f"    Operation: {i % 35} (simulated)")
        print(f"    Timestamp: {history_length - recent_count + i}")
        print(f"    Pair index: {i % 2}")
        print(f"    Selection size: {(i * 10) % 100} cells (simulated)")


def compare_history_configurations():
    """Compare different history configuration options."""
    print(f"\n=== History Configuration Comparison ===")
    
    configs = {
        "minimal": create_history_tracking_config("minimal"),
        "standard": create_history_tracking_config("standard"),
        "research": create_history_tracking_config("research")
    }
    
    print("Configuration Comparison:")
    print(f"{'Mode':<10} {'Length':<8} {'Selection':<10} {'Grids':<8} {'Compress':<10} {'Est. Memory':<12}")
    print("-" * 70)
    
    for mode, config in configs.items():
        length = config.history.max_history_length
        selection = "Yes" if config.history.store_selection_data else "No"
        grids = "Yes" if config.history.store_intermediate_grids else "No"
        compress = "Yes" if config.history.compress_repeated_actions else "No"
        
        # Estimate memory
        base_mem = length * 32
        sel_mem = length * 900 if config.history.store_selection_data else 0
        grid_mem = length * 3600 if config.history.store_intermediate_grids else 0
        total_mem = (base_mem + sel_mem + grid_mem) / 1024 / 1024  # MB
        
        print(f"{mode:<10} {length:<8} {selection:<10} {grids:<8} {compress:<10} {total_mem:.1f} MB")
    
    print("\nRecommendations:")
    print("  - Minimal: For production training with memory constraints")
    print("  - Standard: For most research and development work")
    print("  - Research: For detailed analysis and debugging (high memory usage)")


def main():
    """Main demonstration of action history analysis."""
    print("Action History Analysis Example")
    print("=" * 50)
    
    # Compare different history configurations
    compare_history_configurations()
    
    # Create standard history tracking configuration
    config = create_history_tracking_config("standard")
    print("\nCreated standard history tracking configuration")
    
    # Load task
    task = load_analysis_task()
    
    # Initialize environment
    key = jax.random.PRNGKey(42)
    state, observation = arc_reset(key, config, task_data=task, episode_mode="train")
    
    print(f"Initial state:")
    print(f"  History enabled: {config.history.enabled}")
    print(f"  Max history length: {config.history.max_history_length}")
    print(f"  Store selection data: {config.history.store_selection_data}")
    print(f"  Current history length: {state.action_history_length}")
    
    # Generate diverse action sequence
    state, action_types, total_reward = generate_diverse_action_sequence(state, config, 80)
    
    # Analyze the generated history
    analyze_action_history(state, config, action_types)
    
    # Demonstrate replay capabilities
    demonstrate_history_replay(state, config)
    
    print("\n" + "=" * 50)
    print("Action history analysis example completed!")
    print(f"Final history length: {state.action_history_length}")
    print(f"Total reward achieved: {total_reward:.3f}")


if __name__ == "__main__":
    main()