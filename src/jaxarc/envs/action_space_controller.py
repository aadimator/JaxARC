"""
Action space controller for context-aware operation management.

This module provides the ActionSpaceController class for managing which operations
are available at runtime based on context (episode mode, available pairs, etc.).
It supports basic dynamic action space control with context-dependent operation
availability and configurable validation policies.

Key Features:
- Context-aware operation mask generation
- Operation validation with mode and pair context
- Configurable invalid operation handling policies
- JAX-compatible operation filtering
- Support for demo/test pair switching context

Examples:
    ```python
    from jaxarc.envs.action_space_controller import ActionSpaceController
    from jaxarc.envs.config import ActionConfig
    from jaxarc.state import ArcEnvState
    
    # Create controller
    controller = ActionSpaceController()
    
    # Get allowed operations for current context
    allowed_mask = controller.get_allowed_operations(state, action_config)
    
    # Validate specific operation
    is_valid, error_msg = controller.validate_operation(35, state, action_config)
    
    # Filter invalid operation according to policy
    filtered_op = controller.filter_invalid_operation(42, state, action_config)
    ```
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp

from ..state import ArcEnvState
from ..utils.jax_types import (
    NUM_OPERATIONS,
    OperationMask,
)


class ActionSpaceController:
    """Controls which operations are available at runtime with context awareness.
    
    This class provides basic control over the action space by generating
    context-aware operation masks and validating operations based on the
    current environment state and configuration.
    
    The controller supports:
    - Context-dependent operation availability (demo/test switching)
    - Mode-aware operation filtering (train vs test)
    - Configurable validation policies
    - JAX-compatible operation filtering
    
    Examples:
        ```python
        controller = ActionSpaceController()
        
        # Get operations allowed in current context
        mask = controller.get_allowed_operations(state, config)
        
        # Check if specific operation is valid
        is_valid, msg = controller.validate_operation(35, state, config)
        
        # Apply filtering policy to invalid operation
        filtered_op = controller.filter_invalid_operation(42, state, config)
        ```
    """
    
    def get_allowed_operations(
        self, 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> OperationMask:
        """Get current allowed operations mask based on configuration and context.
        
        This method generates a boolean mask indicating which operations are
        currently allowed based on the environment state and action configuration.
        
        Context-aware filtering includes:
        - Demo pair switching only available in train mode with multiple demos
        - Test pair switching only available in test mode with multiple tests
        - Pair reset only available if current pair has been modified
        - Basic operation filtering based on configuration
        
        Args:
            state: Current environment state
            config: Action configuration with filtering settings
            
        Returns:
            JAX boolean array indicating which operations are currently allowed
            
        Examples:
            ```python
            # Get allowed operations in training mode
            state_train = state.replace(episode_mode=jnp.array(0))
            mask = controller.get_allowed_operations(state_train, config)
            
            # Check if demo switching is allowed
            can_switch_demo = mask[35]  # SWITCH_TO_NEXT_DEMO_PAIR
            
            # Get allowed operations in test mode
            state_test = state.replace(episode_mode=jnp.array(1))
            mask = controller.get_allowed_operations(state_test, config)
            
            # Check if test switching is allowed
            can_switch_test = mask[37]  # SWITCH_TO_NEXT_TEST_PAIR
            ```
        """
        # Start with base allowed operations from configuration
        if hasattr(config, 'allowed_operations') and config.allowed_operations is not None:
            # Create mask from explicitly allowed operations
            base_mask = jnp.zeros(NUM_OPERATIONS, dtype=bool)
            for op_id in config.allowed_operations:
                if 0 <= op_id < NUM_OPERATIONS:
                    base_mask = base_mask.at[op_id].set(True)
        else:
            # All operations allowed by default
            base_mask = jnp.ones(NUM_OPERATIONS, dtype=bool)
        
        # Apply dynamic filtering if enabled
        if hasattr(config, 'dynamic_action_filtering') and config.dynamic_action_filtering:
            context_mask = self._get_context_dependent_mask(state, config)
            base_mask = base_mask & context_mask
        
        # Apply any existing allowed_operations_mask from state
        if hasattr(state, 'allowed_operations_mask'):
            base_mask = base_mask & state.allowed_operations_mask
        
        return base_mask
    
    def validate_operation(
        self, 
        operation_id: int, 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> tuple[bool, Optional[str]]:
        """Validate if operation is currently allowed in current context.
        
        This method performs comprehensive validation of an operation ID,
        checking both basic validity (ID range) and context-specific
        availability (mode, available pairs, etc.). The validation logic
        is designed to be JAX-compatible for JIT compilation.
        
        Args:
            operation_id: Operation ID to validate (0-41)
            state: Current environment state
            config: Action configuration
            
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
            
        Examples:
            ```python
            # Validate basic grid operation
            is_valid, msg = controller.validate_operation(15, state, config)
            
            # Validate demo switching in training mode
            train_state = state.replace(episode_mode=jnp.array(0))
            is_valid, msg = controller.validate_operation(35, train_state, config)
            
            # Validate test switching in test mode
            test_state = state.replace(episode_mode=jnp.array(1))
            is_valid, msg = controller.validate_operation(37, test_state, config)
            ```
        """
        # Use JAX-compatible validation for core logic
        op_id_array = jnp.array(operation_id, dtype=jnp.int32)
        allowed_mask = self.get_allowed_operations(state, config)
        is_valid_jax = self._is_operation_valid_jax(op_id_array, allowed_mask)
        
        # Convert JAX result to Python bool for return type compatibility
        is_basic_valid = bool(is_valid_jax)
        
        if not is_basic_valid:
            # Basic range or mask validation failed
            if not (0 <= operation_id < NUM_OPERATIONS):
                return False, f"Operation ID {operation_id} is out of range [0, {NUM_OPERATIONS-1}]"
            else:
                return False, f"Operation {operation_id} is not currently allowed"
        
        # Context-specific validation (can include non-JAX logic for error messages)
        context_valid, context_msg = self._validate_operation_context_jax(operation_id, state, config)
        if not context_valid:
            return False, context_msg
        
        return True, None
    
    def validate_operation_jax(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> jnp.ndarray:
        """JAX-compatible operation validation for JIT compilation.
        
        This method provides a pure JAX implementation of operation validation
        that can be used within JIT-compiled functions. It returns only a boolean
        result without error messages for maximum JIT compatibility.
        
        Args:
            operation_id: Operation ID as JAX array
            state: Current environment state
            config: Action configuration
            
        Returns:
            JAX boolean array indicating if operation is valid
            
        Examples:
            ```python
            # Use in JIT-compiled function
            @jax.jit
            def step_with_validation(state, action, config):
                op_id = jnp.array(action["operation"])
                is_valid = controller.validate_operation_jax(op_id, state, config)
                # Handle validation result...
                return new_state
            ```
        """
        # Get allowed operations mask
        allowed_mask = self.get_allowed_operations(state, config)
        
        # Basic validation using JAX operations
        is_basic_valid = self._is_operation_valid_jax(operation_id, allowed_mask)
        
        # Context validation using JAX operations
        is_context_valid = self._validate_operation_context_jax_only(operation_id, state)
        
        return is_basic_valid & is_context_valid
    
    def filter_invalid_operation(
        self, 
        operation_id: Union[int, jnp.ndarray], 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> Union[int, jnp.ndarray]:
        """Filter invalid operations according to policy (clip, reject, etc).
        
        This method applies the configured policy for handling invalid operations,
        such as clipping to valid range, replacing with no-op, or passing through
        for explicit error handling. All operations are JAX-compatible and JIT-compilable.
        
        Args:
            operation_id: Original operation ID (int or JAX array)
            state: Current environment state
            config: Action configuration with validation policy
            
        Returns:
            Filtered operation ID according to policy (same type as input)
            
        Examples:
            ```python
            # Clip invalid operation to valid range
            config_clip = config.replace(invalid_operation_policy="clip")
            filtered = controller.filter_invalid_operation(50, state, config_clip)
            
            # Pass through for explicit error handling
            config_pass = config.replace(invalid_operation_policy="passthrough")
            filtered = controller.filter_invalid_operation(50, state, config_pass)
            
            # JAX array input
            filtered_array = controller.filter_invalid_operation(
                jnp.array([50, 15, 100]), state, config
            )
            ```
        """
        return self._apply_validation_policy_jax(operation_id, state, config)
    
    def filter_invalid_operation_jax(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> jnp.ndarray:
        """JAX-only version of filter_invalid_operation for JIT compilation.
        
        This method is specifically designed for use within JIT-compiled functions
        and only accepts JAX arrays as input.
        
        Args:
            operation_id: Original operation ID as JAX array
            state: Current environment state
            config: Action configuration with validation policy
            
        Returns:
            Filtered operation ID as JAX array
            
        Examples:
            ```python
            @jax.jit
            def step_with_filtering(state, action, config):
                op_id = jnp.array(action["operation"])
                filtered_op = controller.filter_invalid_operation_jax(op_id, state, config)
                return filtered_op
            ```
        """
        return self._apply_validation_policy_jax(operation_id, state, config)
    
    def _apply_validation_policy_jax(
        self,
        operation_id: Union[int, jnp.ndarray],
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> Union[int, jnp.ndarray]:
        """JAX-compatible validation policy application.
        
        This method implements all validation policies using JAX operations
        to ensure JIT compatibility and efficient execution.
        
        Args:
            operation_id: Original operation ID (int or JAX array)
            state: Current environment state
            config: Action configuration with validation policy
            
        Returns:
            Filtered operation ID according to policy (same type as input)
        """
        # Convert operation_id to JAX array for consistent operations
        if isinstance(operation_id, int):
            op_id_array = jnp.array(operation_id, dtype=jnp.int32)
            return_int = True
        else:
            op_id_array = operation_id.astype(jnp.int32)
            return_int = False
        
        # Get allowed operations mask
        allowed_mask = self.get_allowed_operations(state, config)
        
        # Check if operation is valid using JAX operations
        is_valid = self._is_operation_valid_jax(op_id_array, allowed_mask)
        
        # Get validation policy from config (with fallback)
        policy = getattr(config, 'invalid_operation_policy', 'clip')
        
        # Apply policy using JAX conditional operations for JIT compatibility
        if policy == "clip":
            # Clip to valid range and find nearest valid operation
            clipped_op = jnp.clip(op_id_array, 0, NUM_OPERATIONS - 1)
            # If clipped operation is still invalid, find nearest valid operation
            filtered_op = jnp.where(
                is_valid,
                op_id_array,
                self._find_nearest_valid_operation_jax(clipped_op, allowed_mask)
            )
        elif policy == "reject":
            # Return a special invalid operation ID (-1) for explicit error handling
            reject_op = jnp.array(-1, dtype=jnp.int32)
            filtered_op = jnp.where(is_valid, op_id_array, reject_op)
        elif policy == "penalize":
            # For penalize policy, clip to valid operation but penalty is applied elsewhere
            clipped_op = jnp.clip(op_id_array, 0, NUM_OPERATIONS - 1)
            filtered_op = jnp.where(
                is_valid,
                op_id_array,
                self._find_nearest_valid_operation_jax(clipped_op, allowed_mask)
            )
        elif policy == "passthrough":
            # Pass through unchanged for explicit error handling
            filtered_op = op_id_array
        else:
            # Default to clipping for unknown policies
            clipped_op = jnp.clip(op_id_array, 0, NUM_OPERATIONS - 1)
            filtered_op = jnp.where(
                is_valid,
                op_id_array,
                self._find_nearest_valid_operation_jax(clipped_op, allowed_mask)
            )
        
        # Return appropriate type based on input
        if return_int:
            return int(filtered_op)
        else:
            return filtered_op
    
    def _is_operation_valid_jax(
        self,
        operation_id: jnp.ndarray,
        allowed_mask: OperationMask
    ) -> jnp.ndarray:
        """JAX-compatible operation validity check.
        
        Args:
            operation_id: Operation ID as JAX array
            allowed_mask: Boolean mask of allowed operations
            
        Returns:
            JAX boolean array indicating if operation is valid
        """
        # Check if operation is in valid range
        in_range = (operation_id >= 0) & (operation_id < NUM_OPERATIONS)
        
        # Check if operation is allowed (safe indexing with bounds check)
        is_allowed = jnp.where(
            in_range,
            allowed_mask[jnp.clip(operation_id, 0, NUM_OPERATIONS - 1)],
            False
        )
        
        return in_range & is_allowed
    
    def _find_nearest_valid_operation_jax(
        self,
        operation_id: jnp.ndarray,
        allowed_mask: OperationMask
    ) -> jnp.ndarray:
        """Find the nearest valid operation using JAX operations.
        
        This method finds the closest valid operation to the given operation ID
        using JAX-compatible operations for JIT compilation.
        
        Args:
            operation_id: Target operation ID as JAX array
            allowed_mask: Boolean mask of allowed operations
            
        Returns:
            JAX array containing the nearest valid operation ID
        """
        # Create array of all operation indices
        all_ops = jnp.arange(NUM_OPERATIONS, dtype=jnp.int32)
        
        # Calculate distances from target operation
        distances = jnp.abs(all_ops - operation_id)
        
        # Mask out invalid operations by setting their distance to infinity
        masked_distances = jnp.where(allowed_mask, distances, jnp.inf)
        
        # Find the operation with minimum distance
        nearest_idx = jnp.argmin(masked_distances)
        
        # Return the nearest valid operation, or fallback to operation 0 if none valid
        return jnp.where(
            jnp.any(allowed_mask),
            nearest_idx,
            jnp.array(0, dtype=jnp.int32)  # Fallback to first operation
        )
    
    def _get_context_dependent_mask(
        self, 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> OperationMask:
        """Generate context-dependent operation mask.
        
        This method creates a mask based on the current context, such as
        episode mode, available pairs, and other state-dependent factors.
        
        Args:
            state: Current environment state
            config: Action configuration
            
        Returns:
            JAX boolean array indicating context-dependent operation availability
        """
        # Start with all operations allowed
        mask = jnp.ones(NUM_OPERATIONS, dtype=bool)
        
        # Check if context-dependent operations are enabled
        if not getattr(config, 'context_dependent_operations', False):
            return mask
        
        # Get current mode and pair information
        is_training = state.is_training_mode()
        is_test = state.is_test_mode()
        
        # Demo pair switching operations (35, 36, 40) - only in training mode
        demo_switching_ops = jnp.array([35, 36, 40])  # NEXT_DEMO, PREV_DEMO, FIRST_UNSOLVED_DEMO
        
        # Check if multiple demo pairs are available
        available_demo_count = state.get_available_demo_count()
        has_multiple_demos = available_demo_count > 1
        
        # Demo switching only allowed in training mode with multiple demos
        demo_switching_allowed = is_training & has_multiple_demos
        
        # Update mask for demo switching operations
        for op_id in demo_switching_ops:
            mask = mask.at[op_id].set(demo_switching_allowed)
        
        # Test pair switching operations (37, 38, 41) - only in test mode
        test_switching_ops = jnp.array([37, 38, 41])  # NEXT_TEST, PREV_TEST, FIRST_UNSOLVED_TEST
        
        # Check if multiple test pairs are available
        available_test_count = state.get_available_test_count()
        has_multiple_tests = available_test_count > 1
        
        # Test switching only allowed in test mode with multiple tests
        test_switching_allowed = is_test & has_multiple_tests
        
        # Update mask for test switching operations
        for op_id in test_switching_ops:
            mask = mask.at[op_id].set(test_switching_allowed)
        
        # Pair reset operation (39) - only if current pair has been modified
        # For now, we'll allow it if step_count > 0 (simple heuristic)
        pair_reset_allowed = state.step_count > 0
        mask = mask.at[39].set(pair_reset_allowed)  # RESET_CURRENT_PAIR
        
        return mask
    
    def _validate_operation_context_jax(
        self, 
        operation_id: int, 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> tuple[bool, Optional[str]]:
        """Validate operation in current context with error messages.
        
        This method performs context-specific validation for operations,
        checking if they make sense in the current environment state.
        This version provides error messages for debugging.
        
        Args:
            operation_id: Operation ID to validate
            state: Current environment state
            config: Action configuration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use JAX-compatible validation for core logic
        op_id_array = jnp.array(operation_id, dtype=jnp.int32)
        is_valid_jax = self._validate_operation_context_jax_only(op_id_array, state)
        
        if bool(is_valid_jax):
            return True, None
        
        # Generate specific error messages for debugging (non-JAX logic)
        # Demo pair switching operations (35, 36, 40)
        if operation_id in [35, 36, 40]:  # Demo switching operations
            if not bool(state.is_training_mode()):
                return False, f"Demo switching operation {operation_id} not allowed in test mode"
            
            available_demo_count = int(state.get_available_demo_count())
            if available_demo_count <= 1:
                return False, f"Demo switching operation {operation_id} requires multiple demo pairs"
        
        # Test pair switching operations (37, 38, 41)
        elif operation_id in [37, 38, 41]:  # Test switching operations
            if not bool(state.is_test_mode()):
                return False, f"Test switching operation {operation_id} not allowed in training mode"
            
            available_test_count = int(state.get_available_test_count())
            if available_test_count <= 1:
                return False, f"Test switching operation {operation_id} requires multiple test pairs"
        
        # Pair reset operation (39)
        elif operation_id == 39:  # RESET_CURRENT_PAIR
            if int(state.step_count) == 0:
                return False, "Pair reset operation not meaningful at step 0"
        
        # Generic error for other cases
        return False, f"Operation {operation_id} is not valid in current context"
    
    def _validate_operation_context_jax_only(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState
    ) -> jnp.ndarray:
        """Pure JAX context validation for JIT compilation.
        
        This method performs context-specific validation using only JAX operations
        for maximum JIT compatibility. It returns only boolean results.
        
        Args:
            operation_id: Operation ID as JAX array
            state: Current environment state
            
        Returns:
            JAX boolean array indicating if operation is valid in context
        """
        # Demo pair switching operations (35, 36, 40)
        is_demo_switching = jnp.isin(operation_id, jnp.array([35, 36, 40]))
        demo_switching_valid = (
            state.is_training_mode() & 
            (state.get_available_demo_count() > 1)
        )
        
        # Test pair switching operations (37, 38, 41)
        is_test_switching = jnp.isin(operation_id, jnp.array([37, 38, 41]))
        test_switching_valid = (
            state.is_test_mode() & 
            (state.get_available_test_count() > 1)
        )
        
        # Pair reset operation (39)
        is_pair_reset = (operation_id == 39)
        pair_reset_valid = (state.step_count > 0)
        
        # Grid operations (0-34) are generally context-independent
        is_grid_operation = (operation_id >= 0) & (operation_id <= 34)
        
        # Combine all validation conditions
        valid = jnp.where(
            is_demo_switching,
            demo_switching_valid,
            jnp.where(
                is_test_switching,
                test_switching_valid,
                jnp.where(
                    is_pair_reset,
                    pair_reset_valid,
                    jnp.where(
                        is_grid_operation,
                        True,  # Grid operations are generally valid
                        False  # Unknown operations are invalid
                    )
                )
            )
        )
        
        return valid
    
    def get_operation_availability_summary(
        self, 
        state: ArcEnvState, 
        config: "ActionConfig"
    ) -> dict:
        """Get a summary of operation availability in current context.
        
        This method provides a detailed breakdown of which operations are
        available and why, useful for debugging and analysis.
        
        Args:
            state: Current environment state
            config: Action configuration
            
        Returns:
            Dictionary containing operation availability summary
            
        Examples:
            ```python
            summary = controller.get_operation_availability_summary(state, config)
            
            print(f"Total allowed operations: {summary['total_allowed']}")
            print(f"Context restrictions: {summary['context_restrictions']}")
            
            for category, ops in summary['by_category'].items():
                print(f"{category}: {len(ops['allowed'])} allowed, {len(ops['blocked'])} blocked")
            ```
        """
        allowed_mask = self.get_allowed_operations(state, config)
        
        # Categorize operations
        categories = {
            "fill": list(range(10)),
            "flood_fill": list(range(10, 20)),
            "movement": list(range(20, 24)),
            "transformation": list(range(24, 28)),
            "editing": list(range(28, 32)),
            "special": list(range(32, 35)),
            "control": list(range(35, 42)),
        }
        
        by_category = {}
        for category, op_ids in categories.items():
            allowed = [op_id for op_id in op_ids if bool(allowed_mask[op_id])]
            blocked = [op_id for op_id in op_ids if not bool(allowed_mask[op_id])]
            
            by_category[category] = {
                "allowed": allowed,
                "blocked": blocked,
                "total": len(op_ids),
                "allowed_count": len(allowed),
                "blocked_count": len(blocked),
            }
        
        # Context information
        context_info = {
            "episode_mode": "train" if bool(state.is_training_mode()) else "test",
            "step_count": int(state.step_count),
            "available_demo_pairs": int(state.get_available_demo_count()),
            "available_test_pairs": int(state.get_available_test_count()),
            "completed_demo_pairs": int(state.get_completed_demo_count()),
            "completed_test_pairs": int(state.get_completed_test_count()),
            "current_pair_index": int(state.current_example_idx),
        }
        
        # Configuration information
        config_info = {
            "dynamic_filtering_enabled": getattr(config, 'dynamic_action_filtering', False),
            "context_dependent_enabled": getattr(config, 'context_dependent_operations', False),
            "explicit_allowed_operations": getattr(config, 'allowed_operations', None),
            "invalid_operation_policy": getattr(config, 'invalid_operation_policy', 'clip'),
        }
        
        # Context restrictions analysis
        context_restrictions = []
        
        if not state.is_training_mode():
            context_restrictions.append("Demo switching disabled (test mode)")
        elif int(state.get_available_demo_count()) <= 1:
            context_restrictions.append("Demo switching disabled (single demo)")
        
        if not state.is_test_mode():
            context_restrictions.append("Test switching disabled (training mode)")
        elif int(state.get_available_test_count()) <= 1:
            context_restrictions.append("Test switching disabled (single test)")
        
        if int(state.step_count) == 0:
            context_restrictions.append("Pair reset disabled (no steps taken)")
        
        return {
            "total_operations": NUM_OPERATIONS,
            "total_allowed": int(jnp.sum(allowed_mask)),
            "total_blocked": NUM_OPERATIONS - int(jnp.sum(allowed_mask)),
            "allowed_operations": [i for i in range(NUM_OPERATIONS) if bool(allowed_mask[i])],
            "blocked_operations": [i for i in range(NUM_OPERATIONS) if not bool(allowed_mask[i])],
            "by_category": by_category,
            "context_info": context_info,
            "config_info": config_info,
            "context_restrictions": context_restrictions,
        }
    
    def apply_operation_mask_jax(
        self,
        action_logits: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig",
        mask_value: float = -jnp.inf
    ) -> jnp.ndarray:
        """Apply operation mask to action logits in a JAX-compatible way.
        
        This method applies the current operation mask to action logits,
        setting invalid operations to a specified mask value (typically -inf
        for softmax compatibility). This is useful for RL agents that need
        to respect action space constraints.
        
        Args:
            action_logits: Action logits array with shape (..., num_operations)
            state: Current environment state
            config: Action configuration
            mask_value: Value to set for invalid operations (default: -inf)
            
        Returns:
            Masked action logits with invalid operations set to mask_value
            
        Examples:
            ```python
            # Apply mask to policy logits
            policy_logits = model(observation)
            masked_logits = controller.apply_operation_mask_jax(
                policy_logits, state, config
            )
            action_probs = jax.nn.softmax(masked_logits)
            
            # Use with different mask value
            masked_logits = controller.apply_operation_mask_jax(
                policy_logits, state, config, mask_value=-1e9
            )
            ```
        """
        # Get allowed operations mask
        allowed_mask = self.get_allowed_operations(state, config)
        
        # Ensure mask has correct shape for broadcasting
        # allowed_mask shape: (num_operations,)
        # action_logits shape: (..., num_operations)
        mask_expanded = jnp.broadcast_to(
            allowed_mask, 
            action_logits.shape
        )
        
        # Apply mask: keep original values for allowed ops, mask_value for invalid ops
        masked_logits = jnp.where(
            mask_expanded,
            action_logits,
            mask_value
        )
        
        return masked_logits
    
    def get_valid_operations_indices_jax(
        self,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> jnp.ndarray:
        """Get indices of valid operations as JAX array with static shape.
        
        This method returns the indices of all currently valid operations
        as a JAX array with static shape, useful for sampling or filtering operations.
        Invalid indices are filled with -1.
        
        Args:
            state: Current environment state
            config: Action configuration
            
        Returns:
            JAX array containing indices of valid operations (padded with -1)
            
        Examples:
            ```python
            # Get valid operation indices
            valid_indices = controller.get_valid_operations_indices_jax(state, config)
            
            # Filter out padding values
            num_valid = jnp.sum(controller.get_allowed_operations(state, config))
            actual_valid = valid_indices[:num_valid]
            
            # Sample from valid operations
            key = jax.random.PRNGKey(42)
            sampled_idx = jax.random.choice(key, actual_valid)
            ```
        """
        allowed_mask = self.get_allowed_operations(state, config)
        return jnp.where(allowed_mask, size=NUM_OPERATIONS, fill_value=-1)[0]
    
    def sample_valid_operation_jax(
        self,
        key: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> jnp.ndarray:
        """Sample a valid operation using JAX random operations.
        
        This method samples uniformly from the set of currently valid operations
        using JAX-compatible random number generation with static shapes.
        
        Args:
            key: JAX PRNG key for random sampling
            state: Current environment state
            config: Action configuration
            
        Returns:
            JAX array containing a randomly sampled valid operation ID
            
        Examples:
            ```python
            # Sample valid operation
            key = jax.random.PRNGKey(42)
            operation = controller.sample_valid_operation_jax(key, state, config)
            
            # Use in JIT-compiled function
            @jax.jit
            def random_valid_action(key, state, config):
                operation = controller.sample_valid_operation_jax(key, state, config)
                return {"operation": operation, "selection": default_selection}
            ```
        """
        allowed_mask = self.get_allowed_operations(state, config)
        
        # Create probability distribution based on allowed mask
        probs = jnp.where(allowed_mask, 1.0, 0.0)
        
        # Handle case where no operations are valid (shouldn't happen in practice)
        num_valid = jnp.sum(allowed_mask)
        
        # Normalize probabilities, with fallback for all-zero case
        normalized_probs = jnp.where(
            num_valid > 0,
            probs / num_valid,
            jnp.ones(NUM_OPERATIONS) / NUM_OPERATIONS  # Uniform fallback
        )
        
        # Sample using categorical distribution
        sampled_idx = jax.random.categorical(key, jnp.log(normalized_probs + 1e-8))
        
        return sampled_idx
    
    def get_next_valid_operation(
        self, 
        operation_id: int, 
        state: ArcEnvState, 
        config: "ActionConfig",
        direction: str = "forward"
    ) -> int:
        """Get the next valid operation in the specified direction.
        
        This method finds the next valid operation starting from the given
        operation ID, useful for implementing operation cycling or finding
        alternatives to invalid operations.
        
        Args:
            operation_id: Starting operation ID
            state: Current environment state
            config: Action configuration
            direction: Search direction ("forward" or "backward")
            
        Returns:
            Next valid operation ID, or -1 if none found
            
        Examples:
            ```python
            # Find next valid operation after current
            next_op = controller.get_next_valid_operation(35, state, config, "forward")
            
            # Find previous valid operation
            prev_op = controller.get_next_valid_operation(35, state, config, "backward")
            ```
        """
        # Use JAX-compatible implementation
        op_id_array = jnp.array(operation_id, dtype=jnp.int32)
        step = 1 if direction == "forward" else -1
        
        allowed_mask = self.get_allowed_operations(state, config)
        next_op = self._find_next_valid_operation_jax(op_id_array, allowed_mask, step)
        
        return int(next_op)
    
    def _find_next_valid_operation_jax(
        self,
        operation_id: jnp.ndarray,
        allowed_mask: OperationMask,
        step: int
    ) -> jnp.ndarray:
        """Find next valid operation using JAX operations.
        
        Args:
            operation_id: Starting operation ID as JAX array
            allowed_mask: Boolean mask of allowed operations
            step: Search step (1 for forward, -1 for backward)
            
        Returns:
            Next valid operation ID, or -1 if none found
        """
        # Create array of candidate operations
        offsets = jnp.arange(1, NUM_OPERATIONS) * step
        candidates = (operation_id + offsets) % NUM_OPERATIONS
        
        # Check which candidates are valid
        candidate_valid = allowed_mask[candidates]
        
        # Find first valid candidate
        first_valid_idx = jnp.argmax(candidate_valid)
        has_valid = jnp.any(candidate_valid)
        
        # Return first valid candidate or -1 if none found
        return jnp.where(
            has_valid,
            candidates[first_valid_idx],
            jnp.array(-1, dtype=jnp.int32)
        )
    
    def is_control_operation(self, operation_id: int) -> bool:
        """Check if operation is a control operation (pair switching, etc.).
        
        Args:
            operation_id: Operation ID to check
            
        Returns:
            True if operation is a control operation (35-41)
        """
        return 35 <= operation_id <= 41
    
    def is_grid_operation(self, operation_id: int) -> bool:
        """Check if operation is a grid manipulation operation.
        
        Args:
            operation_id: Operation ID to check
            
        Returns:
            True if operation is a grid operation (0-34)
        """
        return 0 <= operation_id <= 34
    
    def handle_invalid_operation_jax(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Handle invalid operations with JAX-compatible error handling.
        
        This method provides a JAX-compatible way to handle invalid operations
        by returning both the filtered operation and an error flag. This allows
        JIT-compiled functions to handle validation errors without exceptions.
        
        Args:
            operation_id: Operation ID as JAX array
            state: Current environment state
            config: Action configuration
            
        Returns:
            Tuple of (filtered_operation_id, error_flag) where error_flag
            is True if the original operation was invalid
            
        Examples:
            ```python
            @jax.jit
            def step_with_error_handling(state, action, config):
                op_id = jnp.array(action["operation"])
                filtered_op, had_error = controller.handle_invalid_operation_jax(
                    op_id, state, config
                )
                
                # Apply penalty for invalid operations
                penalty = jnp.where(had_error, -1.0, 0.0)
                
                # Use filtered operation for actual step
                filtered_action = action.copy()
                filtered_action["operation"] = filtered_op
                
                return step_function(state, filtered_action), penalty
            ```
        """
        # Check if operation is valid
        is_valid = self.validate_operation_jax(operation_id, state, config)
        
        # Apply filtering policy
        filtered_op = self._apply_validation_policy_jax(operation_id, state, config)
        
        # Return filtered operation and error flag
        error_flag = ~is_valid
        
        return filtered_op, error_flag
    
    def get_validation_penalty_jax(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig",
        penalty_value: float = -1.0
    ) -> jnp.ndarray:
        """Get penalty value for invalid operations in JAX-compatible way.
        
        This method returns a penalty value for invalid operations, which can
        be used in reward calculation or loss functions. The penalty is applied
        only when the "penalize" policy is configured.
        
        Args:
            operation_id: Operation ID as JAX array
            state: Current environment state
            config: Action configuration
            penalty_value: Penalty value for invalid operations
            
        Returns:
            JAX array containing penalty value (0.0 for valid ops, penalty_value for invalid)
            
        Examples:
            ```python
            # Apply penalty in reward calculation
            penalty = controller.get_validation_penalty_jax(
                op_id, state, config, penalty_value=-0.1
            )
            total_reward = base_reward + penalty
            ```
        """
        # Check if operation is valid
        is_valid = self.validate_operation_jax(operation_id, state, config)
        
        # Get policy from config
        policy = getattr(config, 'invalid_operation_policy', 'clip')
        
        # Apply penalty only for "penalize" policy
        should_penalize = (policy == "penalize")
        
        # Return penalty for invalid operations when policy is "penalize"
        # Logic: if operation is valid OR we shouldn't penalize, return 0.0
        # Otherwise, return penalty_value
        penalty = jnp.where(
            is_valid,
            0.0,  # Valid operations get no penalty
            jnp.where(
                should_penalize,
                penalty_value,  # Invalid operations get penalty if policy is "penalize"
                0.0  # Invalid operations get no penalty for other policies
            )
        )
        
        return penalty
    
    def create_action_mask_for_agent(
        self,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> dict:
        """Create action mask dictionary for RL agents.
        
        This method creates a comprehensive action mask that can be used by
        RL agents to understand which actions are currently valid. It includes
        both the operation mask and additional context information.
        
        Args:
            state: Current environment state
            config: Action configuration
            
        Returns:
            Dictionary containing action mask information for agents
            
        Examples:
            ```python
            # Get action mask for agent
            action_mask = controller.create_action_mask_for_agent(state, config)
            
            # Use in agent policy
            if action_mask["has_restrictions"]:
                masked_logits = apply_mask(logits, action_mask["operation_mask"])
            
            # Check specific operation availability
            if action_mask["operation_mask"][35]:  # Can switch demo pairs
                # Enable demo switching in UI
                pass
            ```
        """
        operation_mask = self.get_allowed_operations(state, config)
        
        # Count valid operations by category
        categories = {
            "fill": list(range(10)),
            "flood_fill": list(range(10, 20)),
            "movement": list(range(20, 24)),
            "transformation": list(range(24, 28)),
            "editing": list(range(28, 32)),
            "special": list(range(32, 35)),
            "control": list(range(35, 42)),
        }
        
        category_counts = {}
        for category, op_ids in categories.items():
            category_mask = operation_mask[jnp.array(op_ids)]
            category_counts[category] = {
                "allowed": int(jnp.sum(category_mask)),
                "total": len(op_ids),
                "mask": category_mask.tolist()
            }
        
        return {
            "operation_mask": operation_mask.tolist(),
            "total_allowed": int(jnp.sum(operation_mask)),
            "total_operations": NUM_OPERATIONS,
            "has_restrictions": int(jnp.sum(operation_mask)) < NUM_OPERATIONS,
            "category_breakdown": category_counts,
            "context_info": {
                "episode_mode": "train" if bool(state.is_training_mode()) else "test",
                "step_count": int(state.step_count),
                "can_switch_demos": bool(operation_mask[35]),  # SWITCH_TO_NEXT_DEMO_PAIR
                "can_switch_tests": bool(operation_mask[37]),  # SWITCH_TO_NEXT_TEST_PAIR
                "can_reset_pair": bool(operation_mask[39]),    # RESET_CURRENT_PAIR
            },
            "validation_policy": getattr(config, 'invalid_operation_policy', 'clip'),
            "dynamic_filtering_enabled": getattr(config, 'dynamic_action_filtering', False),
        }
    
    def validate_operation_range_jax(
        self,
        operation_id: jnp.ndarray
    ) -> jnp.ndarray:
        """Validate operation ID is within valid range using JAX operations.
        
        This method provides a pure JAX implementation for range validation
        that can be used in JIT-compiled functions.
        
        Args:
            operation_id: Operation ID as JAX array
            
        Returns:
            JAX boolean array indicating if operation is in valid range
            
        Examples:
            ```python
            @jax.jit
            def validate_batch_operations(op_ids):
                return controller.validate_operation_range_jax(op_ids)
            
            # Test with batch of operations
            ops = jnp.array([15, 50, -1, 35])
            valid = validate_batch_operations(ops)  # [True, False, False, True]
            ```
        """
        return (operation_id >= 0) & (operation_id < NUM_OPERATIONS)
    
    def apply_operation_constraints_jax(
        self,
        operation_id: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig",
        apply_penalty: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply all operation constraints and return comprehensive results.
        
        This method combines range validation, context validation, and policy
        application into a single JAX-compatible function that returns all
        relevant information for downstream processing.
        
        Args:
            operation_id: Operation ID as JAX array
            state: Current environment state
            config: Action configuration
            apply_penalty: Whether to calculate penalty values
            
        Returns:
            Tuple of (filtered_operation, is_valid, penalty_value)
            
        Examples:
            ```python
            @jax.jit
            def process_action_with_constraints(action, state, config):
                op_id = jnp.array(action["operation"])
                filtered_op, is_valid, penalty = controller.apply_operation_constraints_jax(
                    op_id, state, config, apply_penalty=True
                )
                
                # Use filtered operation for execution
                filtered_action = action.copy()
                filtered_action["operation"] = filtered_op
                
                # Apply penalty to reward
                reward_penalty = jnp.where(is_valid, 0.0, penalty)
                
                return filtered_action, reward_penalty
            ```
        """
        # Validate operation
        is_valid = self.validate_operation_jax(operation_id, state, config)
        
        # Apply filtering policy
        filtered_op = self.filter_invalid_operation_jax(operation_id, state, config)
        
        # Calculate penalty if requested
        if apply_penalty:
            penalty = self.get_validation_penalty_jax(
                operation_id, state, config, penalty_value=-1.0
            )
        else:
            penalty = jnp.array(0.0)
        
        return filtered_op, is_valid, penalty
    
    def batch_validate_operations_jax(
        self,
        operation_ids: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig"
    ) -> jnp.ndarray:
        """Validate a batch of operations using JAX vectorization.
        
        This method efficiently validates multiple operations simultaneously
        using JAX's vectorization capabilities.
        
        Args:
            operation_ids: Batch of operation IDs as JAX array with shape (batch_size,)
            state: Current environment state
            config: Action configuration
            
        Returns:
            JAX boolean array with shape (batch_size,) indicating validity
            
        Examples:
            ```python
            @jax.jit
            def validate_policy_outputs(logits, state, config):
                # Get top-k operations from policy
                top_ops = jnp.argsort(logits)[-5:]
                
                # Validate all top operations
                valid_mask = controller.batch_validate_operations_jax(
                    top_ops, state, config
                )
                
                # Filter to only valid operations
                valid_ops = top_ops[valid_mask]
                return valid_ops
            ```
        """
        # Use vmap to vectorize validation across the batch
        validate_fn = lambda op_id: self.validate_operation_jax(op_id, state, config)
        return jax.vmap(validate_fn)(operation_ids)
    
    def create_operation_filter_mask_jax(
        self,
        operation_logits: jnp.ndarray,
        state: ArcEnvState,
        config: "ActionConfig",
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """Create a filtered probability distribution over operations.
        
        This method creates a valid probability distribution by masking invalid
        operations and renormalizing, useful for sampling valid actions.
        
        Args:
            operation_logits: Raw operation logits with shape (..., num_operations)
            state: Current environment state
            config: Action configuration
            temperature: Temperature for softmax (default: 1.0)
            
        Returns:
            JAX array with valid probability distribution over operations
            
        Examples:
            ```python
            @jax.jit
            def sample_valid_action(key, logits, state, config):
                # Create valid probability distribution
                probs = controller.create_operation_filter_mask_jax(
                    logits, state, config, temperature=0.8
                )
                
                # Sample from valid distribution
                op_id = jax.random.categorical(key, jnp.log(probs + 1e-8))
                return op_id
            ```
        """
        # Apply operation mask to logits
        masked_logits = self.apply_operation_mask_jax(
            operation_logits, state, config, mask_value=-jnp.inf
        )
        
        # Apply temperature scaling
        scaled_logits = masked_logits / temperature
        
        # Convert to probabilities
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        
        return probs