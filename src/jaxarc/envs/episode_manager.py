"""
Episode management system for enhanced ARC step logic.

This module provides the ArcEpisodeManager class for managing demonstration and test pair
selection, episode lifecycle, and non-parametric pair control operations. It is separate
from the existing visualization EpisodeManager and focuses on core environment logic.

Key Features:
- Configurable pair selection strategies (sequential, random)
- Context-aware pair switching (next/prev/first_unsolved)
- Flexible episode termination criteria
- Non-parametric control operations for pair management
- Full JAX compatibility with pure functions

The episode manager handles the logic for:
1. Initial pair selection based on configuration
2. Episode continuation decisions
3. Non-parametric pair control operations (switching between pairs)
4. Mode-specific behavior (training vs test)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from ..state import ArcEnvState
from ..types import JaxArcTask
from ..utils.jax_types import PairIndex

# JAX-compatible episode mode constants
EPISODE_MODE_TRAIN = 0
EPISODE_MODE_TEST = 1


class ArcEpisodeConfig(eqx.Module):
    """Configuration for episode management behavior.
    
    This configuration is separate from the existing visualization EpisodeConfig
    and focuses on core environment episode management logic.
    
    Attributes:
        # Mode settings
        episode_mode: Episode mode (0=train, 1=test)
        
        # Multi-demonstration settings
        demo_selection_strategy: Strategy for selecting demonstration pairs
        allow_demo_switching: Whether to allow switching between demo pairs
        require_all_demos_solved: Whether all demos must be solved to complete episode
        
        # Test evaluation settings  
        test_selection_strategy: Strategy for selecting test pairs
        allow_test_switching: Whether to allow switching between test pairs
        require_all_tests_solved: Whether all tests must be solved to complete episode
        
        # Termination criteria
        terminate_on_first_success: Whether to terminate episode on first successful pair
        max_pairs_per_episode: Maximum number of pairs to work on per episode
        success_threshold: Similarity threshold for considering a pair "solved"
        
        # Reward settings
        training_reward_frequency: When to calculate rewards in training mode
        evaluation_reward_frequency: When to calculate rewards in evaluation mode
    """
    
    # Mode settings
    episode_mode: int = EPISODE_MODE_TRAIN  # 0=train, 1=test
    
    # Multi-demonstration settings
    demo_selection_strategy: Literal["sequential", "random"] = "random"
    allow_demo_switching: bool = True
    require_all_demos_solved: bool = False
    
    # Test evaluation settings  
    test_selection_strategy: Literal["sequential", "random"] = "sequential"
    allow_test_switching: bool = False
    require_all_tests_solved: bool = True
    
    # Termination criteria
    terminate_on_first_success: bool = False
    max_pairs_per_episode: int = 4
    success_threshold: float = 1.0
    
    # Reward settings
    training_reward_frequency: Literal["step", "submit"] = "step"
    evaluation_reward_frequency: Literal["submit"] = "submit"

    def validate(self) -> tuple[str, ...]:
        """Validate episode configuration and return list of errors."""
        errors = []
        
        # Validate mode
        if self.episode_mode not in [EPISODE_MODE_TRAIN, EPISODE_MODE_TEST]:
            errors.append(f"episode_mode must be {EPISODE_MODE_TRAIN} (train) or {EPISODE_MODE_TEST} (test), got '{self.episode_mode}'")
        
        # Validate selection strategies
        valid_demo_strategies = ["sequential", "random"]
        if self.demo_selection_strategy not in valid_demo_strategies:
            errors.append(f"demo_selection_strategy must be one of {valid_demo_strategies}, got '{self.demo_selection_strategy}'")
        
        valid_test_strategies = ["sequential", "random"]
        if self.test_selection_strategy not in valid_test_strategies:
            errors.append(f"test_selection_strategy must be one of {valid_test_strategies}, got '{self.test_selection_strategy}'")
        
        # Validate numeric fields
        if not isinstance(self.max_pairs_per_episode, int) or self.max_pairs_per_episode <= 0:
            errors.append(f"max_pairs_per_episode must be a positive integer, got {self.max_pairs_per_episode}")
        
        if not isinstance(self.success_threshold, (int, float)) or not 0.0 <= self.success_threshold <= 1.0:
            errors.append(f"success_threshold must be a float in [0.0, 1.0], got {self.success_threshold}")
        
        # Validate reward frequency settings
        valid_train_freq = ["step", "submit"]
        if self.training_reward_frequency not in valid_train_freq:
            errors.append(f"training_reward_frequency must be one of {valid_train_freq}, got '{self.training_reward_frequency}'")
        
        valid_eval_freq = ["submit"]
        if self.evaluation_reward_frequency not in valid_eval_freq:
            errors.append(f"evaluation_reward_frequency must be one of {valid_eval_freq}, got '{self.evaluation_reward_frequency}'")
        
        return errors

    @classmethod
    def from_hydra(cls, cfg: dict) -> "ArcEpisodeConfig":
        """Create episode config from Hydra configuration dictionary.
        
        Args:
            cfg: Hydra configuration dictionary containing episode settings
            
        Returns:
            ArcEpisodeConfig instance with settings from Hydra config
            
        Examples:
            ```python
            from omegaconf import DictConfig
            hydra_cfg = DictConfig({
                "episode_mode": "train",
                "demo_selection_strategy": "random",
                "max_pairs_per_episode": 3
            })
            config = ArcEpisodeConfig.from_hydra(hydra_cfg)
            ```
        """
        return cls(
            episode_mode=cfg.get("episode_mode", "train"),
            demo_selection_strategy=cfg.get("demo_selection_strategy", "random"),
            allow_demo_switching=cfg.get("allow_demo_switching", True),
            require_all_demos_solved=cfg.get("require_all_demos_solved", False),
            test_selection_strategy=cfg.get("test_selection_strategy", "sequential"),
            allow_test_switching=cfg.get("allow_test_switching", False),
            require_all_tests_solved=cfg.get("require_all_tests_solved", True),
            terminate_on_first_success=cfg.get("terminate_on_first_success", False),
            max_pairs_per_episode=cfg.get("max_pairs_per_episode", 4),
            success_threshold=cfg.get("success_threshold", 1.0),
            training_reward_frequency=cfg.get("training_reward_frequency", "step"),
            evaluation_reward_frequency=cfg.get("evaluation_reward_frequency", "submit")
        )


class ArcEpisodeManager:
    """Episode manager for pair selection and lifecycle management.
    
    This class handles the logic for managing training and test episodes with
    non-parametric control operations. It is separate from the existing
    visualization EpisodeManager and focuses on core environment functionality.
    
    Key responsibilities:
    - Initial pair selection based on configuration strategies
    - Episode continuation decisions with flexible termination criteria
    - Non-parametric pair control operations (next/prev/first_unsolved)
    - Context-aware operation validation
    
    All methods are designed to be JAX-compatible pure functions.
    """
    
    @staticmethod
    def select_initial_pair(
        key: PRNGKeyArray, 
        task_data: JaxArcTask, 
        config: ArcEpisodeConfig
    ) -> Tuple[PairIndex, Bool[Array, ""]]:
        """Select initial demonstration or test pair based on configuration.
        
        Args:
            key: JAX PRNG key for random selection
            task_data: ARC task data containing available pairs
            config: Episode configuration specifying selection strategy
            
        Returns:
            Tuple of (selected_pair_index, selection_successful)
            
        Examples:
            ```python
            key = jax.random.PRNGKey(42)
            config = ArcEpisodeConfig(demo_selection_strategy="random")
            pair_idx, success = ArcEpisodeManager.select_initial_pair(key, task_data, config)
            ```
        """
        if config.episode_mode == EPISODE_MODE_TRAIN:
            return ArcEpisodeManager._select_demo_pair(key, task_data, config.demo_selection_strategy)
        else:
            return ArcEpisodeManager._select_test_pair(key, task_data, config.test_selection_strategy)
    
    @staticmethod
    def _select_demo_pair(
        key: PRNGKeyArray,
        task_data: JaxArcTask,
        strategy: Literal["sequential", "random"]
    ) -> Tuple[PairIndex, Bool[Array, ""]]:
        """Select a demonstration pair using the specified strategy."""
        available_pairs = task_data.get_available_demo_pairs()
        
        # Check if any pairs are available
        has_available = jnp.sum(available_pairs) > 0
        
        if strategy == "sequential":
            # Find first available pair using JAX-compatible operations
            indices = jnp.arange(len(available_pairs))
            # Use argmax to find first True value (available pair)
            first_available_idx = jnp.argmax(available_pairs)
            # Verify it's actually available (argmax returns 0 even if no True values)
            is_valid = available_pairs[first_available_idx]
            selected_idx = jnp.where(is_valid, first_available_idx, -1)
        else:  # random
            # Randomly select from available pairs using JAX-compatible operations
            num_available = jnp.sum(available_pairs)
            
            # Create cumulative sum for weighted selection
            cumsum = jnp.cumsum(available_pairs.astype(jnp.int32))
            
            # Generate random number in range [1, num_available]
            random_choice = jax.random.randint(key, (), 1, jnp.maximum(num_available, 1) + 1)
            
            # Find the index where cumsum >= random_choice
            selected_idx = jnp.argmax(cumsum >= random_choice)
            
            # Ensure we have a valid selection
            selected_idx = jnp.where(has_available, selected_idx, -1)
        
        success = has_available & (selected_idx >= 0)
        return jnp.array(selected_idx, dtype=jnp.int32), success
    
    @staticmethod
    def _select_test_pair(
        key: PRNGKeyArray,
        task_data: JaxArcTask,
        strategy: Literal["sequential", "random"]
    ) -> Tuple[PairIndex, Bool[Array, ""]]:
        """Select a test pair using the specified strategy."""
        available_pairs = task_data.get_available_test_pairs()
        
        # Check if any pairs are available
        has_available = jnp.sum(available_pairs) > 0
        
        if strategy == "sequential":
            # Find first available pair using JAX-compatible operations
            indices = jnp.arange(len(available_pairs))
            # Use argmax to find first True value (available pair)
            first_available_idx = jnp.argmax(available_pairs)
            # Verify it's actually available (argmax returns 0 even if no True values)
            is_valid = available_pairs[first_available_idx]
            selected_idx = jnp.where(is_valid, first_available_idx, -1)
        else:  # random
            # Randomly select from available pairs using JAX-compatible operations
            num_available = jnp.sum(available_pairs)
            
            # Create cumulative sum for weighted selection
            cumsum = jnp.cumsum(available_pairs.astype(jnp.int32))
            
            # Generate random number in range [1, num_available]
            random_choice = jax.random.randint(key, (), 1, jnp.maximum(num_available, 1) + 1)
            
            # Find the index where cumsum >= random_choice
            selected_idx = jnp.argmax(cumsum >= random_choice)
            
            # Ensure we have a valid selection
            selected_idx = jnp.where(has_available, selected_idx, -1)
        
        success = has_available & (selected_idx >= 0)
        return jnp.array(selected_idx, dtype=jnp.int32), success
    
    @staticmethod
    def should_continue_episode(
        state: ArcEnvState, 
        config: ArcEpisodeConfig
    ) -> Bool[Array, ""]:
        """Determine if episode should continue or terminate.
        
        Args:
            state: Current environment state
            config: Episode configuration with termination criteria
            
        Returns:
            JAX boolean scalar indicating whether episode should continue
            
        Examples:
            ```python
            config = ArcEpisodeConfig(terminate_on_first_success=True)
            should_continue = ArcEpisodeManager.should_continue_episode(state, config)
            ```
        """
        # Check if episode is already marked as done
        episode_done_check = state.episode_done
        
        # Check success-based termination
        success_termination = jnp.where(
            config.terminate_on_first_success,
            state.similarity_score >= config.success_threshold,
            jnp.array(False)
        )
        
        # Check completion requirements
        is_train_mode = config.episode_mode == EPISODE_MODE_TRAIN
        
        # For training mode: check if all demos solved (if required)
        train_completion_check = jnp.where(
            is_train_mode & config.require_all_demos_solved,
            jnp.all(state.demo_completion_status | jnp.logical_not(state.available_demo_pairs)),
            jnp.array(False)
        )
        
        # For test mode: check if all tests solved (if required)
        test_completion_check = jnp.where(
            jnp.logical_not(is_train_mode) & config.require_all_tests_solved,
            jnp.all(state.test_completion_status | jnp.logical_not(state.available_test_pairs)),
            jnp.array(False)
        )
        
        # Check pair limit
        completed_pairs = jnp.where(
            is_train_mode,
            jnp.sum(state.demo_completion_status),
            jnp.sum(state.test_completion_status)
        )
        
        pair_limit_reached = completed_pairs >= config.max_pairs_per_episode
        
        # Combine all termination conditions
        should_terminate = (
            episode_done_check |
            success_termination |
            train_completion_check |
            test_completion_check |
            pair_limit_reached
        )
        
        # Return negation (should continue = not should terminate)
        return jnp.logical_not(should_terminate)
    
    @staticmethod
    def execute_pair_control_operation(
        state: ArcEnvState,
        operation_id: int,
        config: ArcEpisodeConfig
    ) -> ArcEnvState:
        """Execute non-parametric pair control operations.
        
        Handles operations like:
        - SWITCH_TO_NEXT_DEMO_PAIR: Move to next available demo pair
        - SWITCH_TO_PREV_DEMO_PAIR: Move to previous demo pair
        - SWITCH_TO_FIRST_UNSOLVED_DEMO: Jump to first unsolved demo
        - Similar operations for test pairs
        - RESET_CURRENT_PAIR: Reset current pair to initial state
        
        Args:
            state: Current environment state
            operation_id: Operation ID (35-41 for control operations)
            config: Episode configuration
            
        Returns:
            Updated environment state with new pair selection
            
        Examples:
            ```python
            # Switch to next demo pair
            new_state = ArcEpisodeManager.execute_pair_control_operation(
                state, 35, config  # SWITCH_TO_NEXT_DEMO_PAIR
            )
            ```
        """
        # Import operation constants
        from ..types import ARCLEOperationType
        
        # Use JAX-compatible conditional logic with individual field updates
        
        # Calculate new index for each operation type
        next_demo_idx = ArcEpisodeManager._find_next_available_index(state.current_example_idx, state.available_demo_pairs)
        prev_demo_idx = ArcEpisodeManager._find_prev_available_index(state.current_example_idx, state.available_demo_pairs)
        
        # First unsolved demo
        unsolved_demo_mask = state.available_demo_pairs & ~state.demo_completion_status
        first_unsolved_demo_idx = jnp.where(
            jnp.any(unsolved_demo_mask),
            jnp.argmax(unsolved_demo_mask),
            state.current_example_idx
        )
        
        # Test pair operations
        next_test_idx = ArcEpisodeManager._find_next_available_index(state.current_example_idx, state.available_test_pairs)
        prev_test_idx = ArcEpisodeManager._find_prev_available_index(state.current_example_idx, state.available_test_pairs)
        
        # First unsolved test
        unsolved_test_mask = state.available_test_pairs & ~state.test_completion_status
        first_unsolved_test_idx = jnp.where(
            jnp.any(unsolved_test_mask),
            jnp.argmax(unsolved_test_mask),
            state.current_example_idx
        )
        
        # Determine which operation to apply and calculate new index
        new_idx = jnp.where(
            operation_id == ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR,
            jnp.where(
                config.allow_demo_switching & (state.episode_mode == EPISODE_MODE_TRAIN),  # Training mode
                next_demo_idx,
                state.current_example_idx
            ),
            jnp.where(
                operation_id == ARCLEOperationType.SWITCH_TO_PREV_DEMO_PAIR,
                jnp.where(
                    config.allow_demo_switching & (state.episode_mode == EPISODE_MODE_TRAIN),  # Training mode
                    prev_demo_idx,
                    state.current_example_idx
                ),
                jnp.where(
                    operation_id == ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_DEMO,
                    jnp.where(
                        config.allow_demo_switching & (state.episode_mode == EPISODE_MODE_TRAIN),  # Training mode
                        first_unsolved_demo_idx,
                        state.current_example_idx
                    ),
                    jnp.where(
                        operation_id == ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR,
                        jnp.where(
                            config.allow_test_switching & (state.episode_mode == EPISODE_MODE_TEST),  # Test mode
                            next_test_idx,
                            state.current_example_idx
                        ),
                        jnp.where(
                            operation_id == ARCLEOperationType.SWITCH_TO_PREV_TEST_PAIR,
                            jnp.where(
                                config.allow_test_switching & (state.episode_mode == EPISODE_MODE_TEST),  # Test mode
                                prev_test_idx,
                                state.current_example_idx
                            ),
                            jnp.where(
                                operation_id == ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_TEST,
                                jnp.where(
                                    config.allow_test_switching & (state.episode_mode == EPISODE_MODE_TEST),  # Test mode
                                    first_unsolved_test_idx,
                                    state.current_example_idx
                                ),
                                state.current_example_idx  # Default or RESET_CURRENT_PAIR
                            )
                        )
                    )
                )
            )
        )
        
        # Handle reset operation separately (resets more than just index)
        should_reset = operation_id == ARCLEOperationType.RESET_CURRENT_PAIR
        
        # Reset similarity score if reset operation
        reset_similarity = jnp.where(should_reset, jnp.array(0.0), state.similarity_score)
        
        # Update state with new index and potentially reset fields using JAX-compatible approach
        return state.replace(
            current_example_idx=new_idx,
            similarity_score=reset_similarity
        )
    

    
    @staticmethod
    def _find_next_available_index(current_idx: Int[Array, ""], available_mask: Bool[Array, "max_pairs"]) -> Int[Array, ""]:
        """Find the next available index in a circular manner using JAX-compatible operations."""
        num_pairs = len(available_mask)
        
        # Create offset array for circular search
        offsets = jnp.arange(1, num_pairs + 1)
        candidate_indices = (current_idx + offsets) % num_pairs
        
        # Check which candidates are available
        candidates_available = available_mask[candidate_indices]
        
        # Find first available candidate
        has_available = jnp.any(candidates_available)
        first_available_offset = jnp.argmax(candidates_available)
        next_idx = candidate_indices[first_available_offset]
        
        # If no available candidate found, return current index
        return jnp.where(has_available, next_idx, current_idx)
    
    @staticmethod
    def _find_prev_available_index(current_idx: Int[Array, ""], available_mask: Bool[Array, "max_pairs"]) -> Int[Array, ""]:
        """Find the previous available index in a circular manner using JAX-compatible operations."""
        num_pairs = len(available_mask)
        
        # Create offset array for circular search (backward)
        offsets = jnp.arange(1, num_pairs + 1)
        candidate_indices = (current_idx - offsets) % num_pairs
        
        # Check which candidates are available
        candidates_available = available_mask[candidate_indices]
        
        # Find first available candidate
        has_available = jnp.any(candidates_available)
        first_available_offset = jnp.argmax(candidates_available)
        prev_idx = candidate_indices[first_available_offset]
        
        # If no available candidate found, return current index
        return jnp.where(has_available, prev_idx, current_idx)
    
    @staticmethod
    def validate_pair_control_operation(
        state: ArcEnvState,
        operation_id: int,
        config: ArcEpisodeConfig
    ) -> Tuple[Bool[Array, ""], Optional[str]]:
        """Validate if a pair control operation is allowed in the current context.
        
        Args:
            state: Current environment state
            operation_id: Operation ID to validate
            config: Episode configuration
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Examples:
            ```python
            is_valid, error = ArcEpisodeManager.validate_pair_control_operation(
                state, 35, config  # SWITCH_TO_NEXT_DEMO_PAIR
            )
            ```
        """
        from ..types import ARCLEOperationType
        
        # Check demo pair operations
        is_demo_op = (
            (operation_id == ARCLEOperationType.SWITCH_TO_NEXT_DEMO_PAIR) |
            (operation_id == ARCLEOperationType.SWITCH_TO_PREV_DEMO_PAIR) |
            (operation_id == ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_DEMO)
        )
        
        demo_valid = (
            (state.episode_mode == EPISODE_MODE_TRAIN) &  # Training mode
            config.allow_demo_switching &
            (jnp.sum(state.available_demo_pairs) > 1)
        )
        
        # Check test pair operations
        is_test_op = (
            (operation_id == ARCLEOperationType.SWITCH_TO_NEXT_TEST_PAIR) |
            (operation_id == ARCLEOperationType.SWITCH_TO_PREV_TEST_PAIR) |
            (operation_id == ARCLEOperationType.SWITCH_TO_FIRST_UNSOLVED_TEST)
        )
        
        test_valid = (
            (state.episode_mode == EPISODE_MODE_TEST) &  # Test mode
            config.allow_test_switching &
            (jnp.sum(state.available_test_pairs) > 1)
        )
        
        # Check reset operation
        is_reset_op = operation_id == ARCLEOperationType.RESET_CURRENT_PAIR
        reset_valid = jnp.array(True)  # Reset is generally allowed
        
        # Check if operation is known
        is_known_op = is_demo_op | is_test_op | is_reset_op
        
        # Combine validation results
        is_valid = jnp.where(
            is_demo_op,
            demo_valid,
            jnp.where(
                is_test_op,
                test_valid,
                jnp.where(
                    is_reset_op,
                    reset_valid,
                    jnp.array(False)  # Unknown operation
                )
            )
        )
        
        # For JAX compatibility, we can't return different error messages
        # In a real implementation, error handling would be done outside JAX
        return is_valid, None