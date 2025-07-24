"""
ARC environment observation space for agent-focused views.

This module provides the ArcObservation class and related functionality for creating
focused, agent-friendly observations from the full environment state. The observation
space is designed to provide agents with the essential information they need while
hiding internal implementation details and supporting configurable observation formats.

Key Features:
- Separate observation space from internal environment state
- Configurable observation components for research flexibility
- Information hiding to prevent agents from accessing internal details
- Support for partial observability scenarios
- JAX-compatible structures for efficient processing
- Mode-aware observations (training vs evaluation)
- Demonstration pairs for few-shot learning and pattern recognition

Design Principles:
- Clean separation between environment internals and agent interface
- Focused information delivery (only what agents need)
- Configurable observation formats for different research scenarios
- JAX compatibility for efficient batch processing
- Type safety with comprehensive validation

Examples:
    ```python
    from jaxarc.envs.observations import ArcObservation, ObservationConfig, create_observation
    from jaxarc.state import ArcEnvState
    
    # Create observation configuration
    obs_config = ObservationConfig(
        include_target_grid=True,
        include_recent_actions=True,
        recent_action_count=5,
        observation_format="standard"
    )
    
    # Create observation from state
    observation = create_observation(state, obs_config)
    
    # Access agent-relevant information
    working_grid = observation.working_grid
    episode_mode = observation.episode_mode
    allowed_ops = observation.allowed_operations_mask
    ```
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import chex
import equinox as eqx
import jax.numpy as jnp

from ..utils.jax_types import (
    ActionHistory,
    EpisodeIndex,
    EpisodeMode,
    GridArray,
    MaskArray,
    OperationMask,
    StepCount,
    TaskInputGrids,
    TaskInputMasks,
    TaskOutputGrids,
    TaskOutputMasks,
    TestCompletionStatus,
    TrainCompletionStatus,
)


@chex.dataclass
class ObservationConfig:
    """Configuration for observation space construction.
    
    This configuration allows researchers to customize what information is included
    in agent observations, enabling experiments with different levels of observability
    and information access patterns.
    
    Attributes:
        include_target_grid: Whether to include target grid in training mode
        include_completion_status: Whether to include progress tracking information
        include_action_space_info: Whether to include allowed operations mask
        include_recent_actions: Whether to include recent action history
        recent_action_count: Number of recent actions to include (if enabled)
        include_step_count: Whether to include step counter
        observation_format: Level of detail in observations
        mask_internal_state: Whether to hide internal implementation details
        
    Examples:
        ```python
        # Minimal observation for basic agents
        minimal_config = ObservationConfig(
            observation_format="minimal",
            include_recent_actions=False,
            include_completion_status=False
        )
        
        # Rich observation for advanced agents
        rich_config = ObservationConfig(
            observation_format="rich",
            include_recent_actions=True,
            recent_action_count=10,
            include_completion_status=True
        )
        
        # Evaluation-specific configuration
        eval_config = ObservationConfig(
            include_target_grid=False,  # Hidden in evaluation
            include_completion_status=True,
            observation_format="standard"
        )
        ```
    """
    
    # Core observation components
    include_target_grid: bool = True
    """Include target grid in training mode (automatically masked in test mode)."""
    
    include_completion_status: bool = True
    """Include progress tracking for demonstration/test pairs."""
    
    include_action_space_info: bool = True
    """Include information about currently allowed operations."""
    
    # Optional components
    include_recent_actions: bool = False
    """Include recent action history in observations."""
    
    recent_action_count: int = 10
    """Number of recent actions to include (if include_recent_actions=True)."""
    
    include_step_count: bool = True
    """Include step counter in observations."""
    
    include_demonstration_pairs: bool = True
    """Include demonstration pairs for pattern recognition and few-shot learning."""
    
    max_demonstration_pairs: int = 10
    """Maximum number of demonstration pairs to include (dataset-dependent)."""
    
    # Research flexibility options
    observation_format: Literal["minimal", "standard", "rich"] = "standard"
    """Level of detail in observations:
    - minimal: Only core grids and basic context
    - standard: Core grids plus episode context and progress
    - rich: All available information for advanced agents
    """
    
    mask_internal_state: bool = True
    """Hide internal implementation details from agents."""
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.recent_action_count < 0:
            raise ValueError("recent_action_count must be non-negative")
        
        if self.observation_format not in ("minimal", "standard", "rich"):
            raise ValueError("observation_format must be 'minimal', 'standard', or 'rich'")
        
        # Adjust settings based on format
        if self.observation_format == "minimal":
            # Override some settings for minimal format
            object.__setattr__(self, "include_recent_actions", False)
            object.__setattr__(self, "include_completion_status", False)
        elif self.observation_format == "rich":
            # Enable more features for rich format
            object.__setattr__(self, "include_completion_status", True)
            object.__setattr__(self, "include_action_space_info", True)
            object.__setattr__(self, "include_demonstration_pairs", True)


class ArcObservation(eqx.Module):
    """Agent observation space - focused view of environment state.
    
    This class provides a clean, focused interface for agents to observe the ARC
    environment state. It separates the agent's view from internal implementation
    details and supports configurable observation formats for research flexibility.
    
    Key Design Principles:
    - Information hiding: Agents don't see internal implementation details
    - Focused delivery: Only essential information for decision making
    - Mode awareness: Different information available in train vs test modes
    - Configurability: Researchers can customize observation components
    - JAX compatibility: Efficient processing and batch operations
    
    Attributes:
        working_grid: Current grid being modified by the agent
        working_grid_mask: Valid cells mask for the working grid
        episode_mode: Current episode mode (0=train, 1=test)
        current_pair_idx: Index of currently active demonstration/test pair
        step_count: Number of steps taken in current episode
        demo_completion_status: Which demonstration pairs are completed
        test_completion_status: Which test pairs are completed
        allowed_operations_mask: Currently allowed operations for action space
        target_grid: Target grid (only in training mode, None in test mode)
        recent_actions: Recent action history (optional, configurable)
        
    Examples:
        ```python
        # Access core grid information
        current_grid = observation.working_grid
        valid_cells = observation.working_grid_mask
        
        # Check episode context
        is_training = observation.episode_mode == 0
        current_pair = observation.current_pair_idx
        
        # Check progress
        completed_demos = jnp.sum(observation.demo_completion_status)
        
        # Check available actions
        allowed_ops = observation.allowed_operations_mask
        num_allowed = jnp.sum(allowed_ops)
        
        # Access target (if in training mode)
        if observation.target_grid is not None:
            target = observation.target_grid
        
        # Access demonstration pairs (if included)
        if observation.has_demonstration_pairs():
            demo_inputs = observation.demo_input_grids
            demo_outputs = observation.demo_output_grids
            num_demos = observation.get_num_demo_pairs()
        ```
    """
    
    # Core grid information (what agent directly works with)
    working_grid: GridArray
    """Current grid being modified by the agent."""
    
    working_grid_mask: MaskArray
    """Valid cells mask indicating which cells are part of the actual grid."""
    
    # Episode context (what agent needs to know about current situation)
    episode_mode: EpisodeMode
    """Current episode mode: 0=training (with target access), 1=test (evaluation)."""
    
    current_pair_idx: EpisodeIndex
    """Index of currently active demonstration/test pair."""
    
    step_count: StepCount
    """Number of steps taken in current episode."""
    
    # Progress tracking (helps agent understand accomplishments)
    demo_completion_status: TrainCompletionStatus
    """Boolean mask indicating which demonstration pairs have been completed."""
    
    test_completion_status: TestCompletionStatus
    """Boolean mask indicating which test pairs have been completed."""
    
    # Action space information (what agent can do)
    allowed_operations_mask: OperationMask
    """Boolean mask indicating which operations are currently allowed."""
    
    # Target information (training only, masked in test mode)
    target_grid: GridArray
    """Target grid for current pair (masked with zeros in test mode for JAX compatibility)."""
    
    # Recent action history (configurable, masked when disabled)
    recent_actions: ActionHistory
    """Recent action history (masked with zeros when disabled for JAX compatibility)."""
    
    # Demonstration pairs for pattern recognition (configurable, masked when disabled)
    demo_input_grids: TaskInputGrids
    """Demonstration input grids (masked with zeros when disabled for JAX compatibility)."""
    
    demo_output_grids: TaskOutputGrids
    """Demonstration output grids (masked with zeros when disabled for JAX compatibility)."""
    
    demo_input_masks: TaskInputMasks
    """Masks for demonstration input grids (masked with zeros when disabled for JAX compatibility)."""
    
    demo_output_masks: TaskOutputMasks
    """Masks for demonstration output grids (masked with zeros when disabled for JAX compatibility)."""
    
    num_demo_pairs: int
    """Number of valid demonstration pairs included in observation (0 when disabled)."""
    
    def __check_init__(self) -> None:
        """Validate observation structure.
        
        This validation ensures that all observation components have the correct
        shapes and types, and that the observation is internally consistent.
        """
        # Skip validation during JAX transformations
        if not hasattr(self.working_grid, "shape"):
            return
            
        try:
            # Validate core grid information
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)
            
            # Validate episode context
            chex.assert_type(self.episode_mode, jnp.integer)
            chex.assert_type(self.current_pair_idx, jnp.integer)
            chex.assert_type(self.step_count, jnp.integer)
            chex.assert_shape(self.episode_mode, ())
            chex.assert_shape(self.current_pair_idx, ())
            chex.assert_shape(self.step_count, ())
            
            # Validate progress tracking
            chex.assert_type(self.demo_completion_status, jnp.bool_)
            chex.assert_type(self.test_completion_status, jnp.bool_)
            chex.assert_rank(self.demo_completion_status, 1)
            chex.assert_rank(self.test_completion_status, 1)
            
            # Validate action space information
            chex.assert_type(self.allowed_operations_mask, jnp.bool_)
            chex.assert_rank(self.allowed_operations_mask, 1)
            
            # Validate target grid (always present, masked when not applicable)
            chex.assert_rank(self.target_grid, 2)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_shape(self.target_grid, self.working_grid.shape)
            
            # Validate recent actions (always present, masked when disabled)
            chex.assert_type(self.recent_actions, jnp.floating)
            chex.assert_rank(self.recent_actions, 2)
            
            # Validate demonstration pairs (always present, masked when disabled)
            chex.assert_type(self.demo_input_grids, jnp.integer)
            chex.assert_rank(self.demo_input_grids, 3)
            chex.assert_type(self.demo_output_grids, jnp.integer)
            chex.assert_rank(self.demo_output_grids, 3)
            chex.assert_shape(self.demo_output_grids, self.demo_input_grids.shape)
            chex.assert_type(self.demo_input_masks, jnp.bool_)
            chex.assert_rank(self.demo_input_masks, 3)
            chex.assert_shape(self.demo_input_masks, self.demo_input_grids.shape)
            chex.assert_type(self.demo_output_masks, jnp.bool_)
            chex.assert_rank(self.demo_output_masks, 3)
            chex.assert_shape(self.demo_output_masks, self.demo_input_grids.shape)
                
            # Validate episode mode value
            if hasattr(self.episode_mode, "item"):
                mode_val = int(self.episode_mode.item())
                if mode_val not in (0, 1):
                    raise ValueError(f"Episode mode must be 0 (train) or 1 (test), got {mode_val}")
                    
        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass
    
    def is_training_mode(self) -> bool:
        """Check if observation is from training mode.
        
        Returns:
            True if in training mode (episode_mode == 0)
        """
        return self.episode_mode == 0
    
    def is_test_mode(self) -> bool:
        """Check if observation is from test/evaluation mode.
        
        Returns:
            True if in test mode (episode_mode == 1)
        """
        return self.episode_mode == 1
    
    def has_target_access(self) -> bool:
        """Check if target grid is available in this observation.
        
        Returns:
            True if target grid contains meaningful data (training mode)
        """
        # Check if target grid has non-zero values (meaningful data)
        return jnp.any(self.target_grid != 0)
    
    def get_completed_demo_count(self) -> int:
        """Get number of completed demonstration pairs.
        
        Returns:
            Number of demonstration pairs marked as completed
        """
        return int(jnp.sum(self.demo_completion_status))
    
    def get_completed_test_count(self) -> int:
        """Get number of completed test pairs.
        
        Returns:
            Number of test pairs marked as completed
        """
        return int(jnp.sum(self.test_completion_status))
    
    def get_allowed_operations_count(self) -> int:
        """Get number of currently allowed operations.
        
        Returns:
            Number of operations that are currently allowed
        """
        return int(jnp.sum(self.allowed_operations_mask))
    
    def is_operation_allowed(self, operation_id: int) -> bool:
        """Check if a specific operation is currently allowed.
        
        Args:
            operation_id: Operation ID to check (0-41)
            
        Returns:
            True if the operation is allowed, False otherwise
        """
        if 0 <= operation_id < len(self.allowed_operations_mask):
            return bool(self.allowed_operations_mask[operation_id])
        return False
    
    def get_grid_shape(self) -> tuple[int, int]:
        """Get the actual shape of the working grid.
        
        Returns:
            Tuple of (height, width) for the working grid
        """
        return self.working_grid.shape
    
    def has_demonstration_pairs(self) -> bool:
        """Check if demonstration pairs are available in this observation.
        
        Returns:
            True if demonstration pairs contain meaningful data
        """
        return self.num_demo_pairs > 0
    
    def get_num_demo_pairs(self) -> int:
        """Get number of demonstration pairs included in observation.
        
        Returns:
            Number of demonstration pairs
        """
        return self.num_demo_pairs
    
    def get_observation_summary(self) -> Dict[str, Any]:
        """Get a summary of key observation information.
        
        Returns:
            Dictionary containing key observation metrics and status
        """
        return {
            "grid_shape": self.get_grid_shape(),
            "episode_mode": "train" if self.is_training_mode() else "test",
            "current_pair_index": int(self.current_pair_idx),
            "step_count": int(self.step_count),
            "has_target_access": self.has_target_access(),
            "completed_demos": self.get_completed_demo_count(),
            "completed_tests": self.get_completed_test_count(),
            "allowed_operations": self.get_allowed_operations_count(),
            "has_recent_actions": jnp.any(self.recent_actions != 0),
            "has_demonstration_pairs": self.has_demonstration_pairs(),
            "num_demo_pairs": self.get_num_demo_pairs(),
        }


def create_observation(
    state: "ArcEnvState",  # Forward reference to avoid circular import
    config: ObservationConfig
) -> ArcObservation:
    """Create agent observation from environment state.
    
    This function extracts relevant information from the full environment state
    and constructs a focused observation for the agent. It applies the observation
    configuration to determine what information to include and how to format it.
    
    Key Features:
    - Information filtering based on configuration
    - Mode-aware target grid masking (hidden in test mode)
    - Configurable action history inclusion
    - Format-specific information levels
    - JAX-compatible processing
    
    Args:
        state: Full environment state containing all information
        config: Configuration specifying what to include in observation
        
    Returns:
        ArcObservation containing agent-focused view of the state
        
    Examples:
        ```python
        # Standard observation
        config = ObservationConfig()
        obs = create_observation(state, config)
        
        # Minimal observation for simple agents
        minimal_config = ObservationConfig(observation_format="minimal")
        minimal_obs = create_observation(state, minimal_config)
        
        # Rich observation with action history
        rich_config = ObservationConfig(
            observation_format="rich",
            include_recent_actions=True,
            recent_action_count=5
        )
        rich_obs = create_observation(state, rich_config)
        ```
    """
    # Core grid information (always included)
    working_grid = state.working_grid
    working_grid_mask = state.working_grid_mask
    
    # Episode context (always included)
    episode_mode = state.episode_mode
    current_pair_idx = state.current_example_idx
    
    # Step count (configurable)
    step_count = state.step_count if config.include_step_count else jnp.array(0)
    
    # Progress tracking (configurable)
    if config.include_completion_status:
        demo_completion_status = state.demo_completion_status
        test_completion_status = state.test_completion_status
    else:
        # Provide empty arrays with correct shapes
        demo_completion_status = jnp.zeros_like(state.demo_completion_status)
        test_completion_status = jnp.zeros_like(state.test_completion_status)
    
    # Action space information (configurable)
    if config.include_action_space_info:
        allowed_operations_mask = state.allowed_operations_mask
    else:
        # Provide all-allowed mask as default
        allowed_operations_mask = jnp.ones_like(state.allowed_operations_mask)
    
    # Target grid (mode-aware and configurable)
    # For JAX compatibility, always provide a target grid but mask it appropriately
    if config.include_target_grid:
        # Use JAX where to conditionally include target based on training mode
        target_grid = jnp.where(
            state.episode_mode == 0,  # 0 = training mode
            state.target_grid,
            jnp.zeros_like(state.target_grid)  # Masked in test mode
        )
    else:
        target_grid = jnp.zeros_like(state.target_grid)  # Disabled by config
    
    # Recent action history (optional and configurable)
    # For JAX compatibility, always provide action history but mask it appropriately
    if config.include_recent_actions:
        # Use the full action history for JAX compatibility
        # The actual recent actions can be extracted using action_history_length
        recent_actions = state.action_history
    else:
        recent_actions = jnp.zeros_like(state.action_history)  # Disabled by config
    
    # Demonstration pairs (optional and configurable)
    # For JAX compatibility, always provide demonstration pairs but mask them appropriately
    if config.include_demonstration_pairs:
        # Always include demonstration pairs from task data
        demo_input_grids = state.task_data.input_grids_examples
        demo_output_grids = state.task_data.output_grids_examples
        demo_input_masks = state.task_data.input_masks_examples
        demo_output_masks = state.task_data.output_masks_examples
        num_demo_pairs = state.task_data.num_train_pairs
    else:
        # Provide zero arrays with correct shapes when disabled
        demo_input_grids = jnp.zeros_like(state.task_data.input_grids_examples)
        demo_output_grids = jnp.zeros_like(state.task_data.output_grids_examples)
        demo_input_masks = jnp.zeros_like(state.task_data.input_masks_examples)
        demo_output_masks = jnp.zeros_like(state.task_data.output_masks_examples)
        num_demo_pairs = 0
    
    return ArcObservation(
        working_grid=working_grid,
        working_grid_mask=working_grid_mask,
        episode_mode=episode_mode,
        current_pair_idx=current_pair_idx,
        step_count=step_count,
        demo_completion_status=demo_completion_status,
        test_completion_status=test_completion_status,
        allowed_operations_mask=allowed_operations_mask,
        target_grid=target_grid,
        recent_actions=recent_actions,
        demo_input_grids=demo_input_grids,
        demo_output_grids=demo_output_grids,
        demo_input_masks=demo_input_masks,
        demo_output_masks=demo_output_masks,
        num_demo_pairs=num_demo_pairs,
    )


def create_minimal_observation(state: "ArcEnvState") -> ArcObservation:
    """Create minimal observation with only essential information.
    
    Convenience function for creating observations with minimal information,
    suitable for simple agents or memory-constrained scenarios.
    
    Args:
        state: Environment state to extract observation from
        
    Returns:
        ArcObservation with minimal information set
    """
    config = ObservationConfig(observation_format="minimal")
    return create_observation(state, config)


def create_standard_observation(state: "ArcEnvState") -> ArcObservation:
    """Create standard observation with balanced information.
    
    Convenience function for creating observations with standard information,
    suitable for most RL agents and research scenarios.
    
    Args:
        state: Environment state to extract observation from
        
    Returns:
        ArcObservation with standard information set
    """
    config = ObservationConfig(observation_format="standard")
    return create_observation(state, config)


def create_rich_observation(
    state: "ArcEnvState", 
    include_action_history: bool = True,
    action_history_length: int = 10
) -> ArcObservation:
    """Create rich observation with comprehensive information.
    
    Convenience function for creating observations with rich information,
    suitable for advanced agents that can leverage additional context.
    
    Args:
        state: Environment state to extract observation from
        include_action_history: Whether to include recent action history
        action_history_length: Number of recent actions to include
        
    Returns:
        ArcObservation with rich information set
    """
    config = ObservationConfig(
        observation_format="rich",
        include_recent_actions=include_action_history,
        recent_action_count=action_history_length
    )
    return create_observation(state, config)


# Convenience factory functions for common observation configurations
def create_training_observation(state: "ArcEnvState") -> ArcObservation:
    """Create observation optimized for training scenarios.
    
    Includes target grid access and progress tracking for training agents.
    
    Args:
        state: Environment state to extract observation from
        
    Returns:
        ArcObservation optimized for training
    """
    config = ObservationConfig(
        include_target_grid=True,
        include_completion_status=True,
        include_action_space_info=True,
        observation_format="standard"
    )
    return create_observation(state, config)


def create_evaluation_observation(state: "ArcEnvState") -> ArcObservation:
    """Create observation optimized for evaluation scenarios.
    
    Hides target grid access to simulate real evaluation conditions.
    
    Args:
        state: Environment state to extract observation from
        
    Returns:
        ArcObservation optimized for evaluation
    """
    config = ObservationConfig(
        include_target_grid=False,  # Hidden during evaluation
        include_completion_status=True,
        include_action_space_info=True,
        observation_format="standard"
    )
    return create_observation(state, config)


def create_debug_observation(state: "ArcEnvState") -> ArcObservation:
    """Create observation with all available information for debugging.
    
    Includes all possible information for debugging and analysis purposes.
    
    Args:
        state: Environment state to extract observation from
        
    Returns:
        ArcObservation with all available information
    """
    config = ObservationConfig(
        include_target_grid=True,
        include_completion_status=True,
        include_action_space_info=True,
        include_recent_actions=True,
        recent_action_count=20,
        include_step_count=True,
        observation_format="rich",
        mask_internal_state=False  # Show internal details for debugging
    )
    return create_observation(state, config)