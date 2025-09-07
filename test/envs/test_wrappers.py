"""
Tests for wrapper functionality in jaxarc.envs.

This module tests action wrapper functionality, compatibility,
and the base wrapper system.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey

from jaxarc import JaxArcConfig
from jaxarc.envs.action_wrappers import BboxActionWrapper, PointActionWrapper
from jaxarc.envs.actions import Action, create_action
from jaxarc.envs.environment import Environment
from jaxarc.envs.spaces import ARCActionSpace, DictSpace, DiscreteSpace
from jaxarc.envs.wrapper import GymAutoResetWrapper, Wrapper
from jaxarc.registration import make
from jaxarc.state import State
from jaxarc.types import EnvParams, TimeStep


class MockEnvironment:
    """Mock environment for testing wrappers."""
    
    def __init__(self):
        self.reset_called = False
        self.step_called = False
        self.last_action = None
    
    def reset(self, params: EnvParams, key: PRNGKey) -> TimeStep:
        self.reset_called = True
        # Create a minimal mock timestep
        mock_state = State(
            working_grid=jnp.ones((5, 5), dtype=jnp.int32),
            working_grid_mask=jnp.ones((5, 5), dtype=jnp.bool_),
            input_grid=jnp.ones((5, 5), dtype=jnp.int32),
            input_grid_mask=jnp.ones((5, 5), dtype=jnp.bool_),
            target_grid=jnp.zeros((5, 5), dtype=jnp.int32),
            target_grid_mask=jnp.ones((5, 5), dtype=jnp.bool_),
            selected=jnp.zeros((5, 5), dtype=jnp.bool_),
            clipboard=jnp.zeros((5, 5), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(35, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=key,
        )
        
        return TimeStep(
            state=mock_state,
            step_type=jnp.array(0, dtype=jnp.int32),  # FIRST
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=mock_state.working_grid,
            extras={}
        )
    
    def step(self, params: EnvParams, timestep: TimeStep, action: Action) -> TimeStep:
        self.step_called = True
        self.last_action = action
        
        # Create a mock next timestep using eqx.tree_at for immutable updates
        import equinox as eqx
        new_state = eqx.tree_at(
            lambda state: state.step_count,
            timestep.state,
            timestep.state.step_count + 1
        )
        
        return TimeStep(
            state=new_state,
            step_type=jnp.array(1, dtype=jnp.int32),  # MID
            reward=jnp.array(1.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=new_state.working_grid,
            extras={}
        )
    
    def action_space(self, params: EnvParams):
        """Mock action space."""
        return ARCActionSpace(max_height=5, max_width=5)
    
    def observation_space(self, params: EnvParams):
        """Mock observation space."""
        from jaxarc.envs.spaces import GridSpace
        return GridSpace(max_height=5, max_width=5)


def create_test_env_params():
    """Helper function to create test EnvParams."""
    config = JaxArcConfig()
    return EnvParams(
        dataset=config.dataset,
        action=config.action,
        reward=config.reward,
        grid_initialization=config.grid_initialization,
        max_episode_steps=100,
        buffer=jnp.array([0]),
        subset_indices=jnp.array([0])
    )


class TestBaseWrapper:
    """Test base Wrapper class functionality."""
    
    def test_wrapper_creation(self):
        """Test basic wrapper creation."""
        mock_env = MockEnvironment()
        wrapper = Wrapper(mock_env)
        
        assert wrapper._env is mock_env
    
    def test_wrapper_delegation(self):
        """Test that wrapper properly delegates to wrapped environment."""
        mock_env = MockEnvironment()
        wrapper = Wrapper(mock_env)
        
        # Test attribute delegation
        assert hasattr(wrapper, 'action_space')
        assert hasattr(wrapper, 'observation_space')
        
        # Test method delegation
        env_params = create_test_env_params()
        
        action_space = wrapper.action_space(env_params)
        assert isinstance(action_space, ARCActionSpace)
    
    def test_wrapper_reset_delegation(self, prng_key: PRNGKey):
        """Test that wrapper delegates reset calls."""
        mock_env = MockEnvironment()
        wrapper = Wrapper(mock_env)
        
        env_params = create_test_env_params()
        
        timestep = wrapper.reset(env_params, prng_key)
        
        assert mock_env.reset_called
        assert isinstance(timestep, TimeStep)
    
    def test_wrapper_step_delegation(self, prng_key: PRNGKey):
        """Test that wrapper delegates step calls."""
        mock_env = MockEnvironment()
        wrapper = Wrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset first
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create action
        action = create_action(
            operation=jnp.array(5, dtype=jnp.int32),
            selection=jnp.ones((5, 5), dtype=jnp.bool_)
        )
        
        # Step
        next_timestep = wrapper.step(env_params, timestep, action)
        
        assert mock_env.step_called
        assert isinstance(next_timestep, TimeStep)
        assert mock_env.last_action is action


class TestPointActionWrapper:
    """Test PointActionWrapper functionality."""
    
    def test_point_action_wrapper_creation(self):
        """Test PointActionWrapper creation."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        assert isinstance(wrapper, PointActionWrapper)
        assert wrapper._env is mock_env
    
    def test_point_action_space(self):
        """Test PointActionWrapper action space."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        action_space = wrapper.action_space(env_params)
        
        assert isinstance(action_space, DictSpace)
        assert "operation" in action_space._spaces
        assert "row" in action_space._spaces
        assert "col" in action_space._spaces
        
        assert isinstance(action_space._spaces["operation"], DiscreteSpace)
        assert isinstance(action_space._spaces["row"], DiscreteSpace)
        assert isinstance(action_space._spaces["col"], DiscreteSpace)
    
    def test_point_action_conversion(self, prng_key: PRNGKey):
        """Test point action to mask conversion."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create point action
        point_action = {
            "operation": jnp.array(10, dtype=jnp.int32),
            "row": jnp.array(2, dtype=jnp.int32),
            "col": jnp.array(3, dtype=jnp.int32)
        }
        
        # Step with point action
        next_timestep = wrapper.step(env_params, timestep, point_action)
        
        # Verify that the action was converted to mask action
        assert mock_env.step_called
        assert isinstance(mock_env.last_action, Action)
        assert mock_env.last_action.operation == 10
        
        # Verify that only the specified point is selected
        selection = mock_env.last_action.selection
        assert selection.shape == (5, 5)
        assert selection[2, 3] == True  # The specified point
        assert jnp.sum(selection) == 1  # Only one point selected
    
    def test_point_action_bounds_clipping(self, prng_key: PRNGKey):
        """Test that point coordinates are clipped to valid bounds."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create point action with out-of-bounds coordinates
        point_action = {
            "operation": jnp.array(5, dtype=jnp.int32),
            "row": jnp.array(10, dtype=jnp.int32),  # Out of bounds (grid is 5x5)
            "col": jnp.array(-1, dtype=jnp.int32)   # Out of bounds
        }
        
        # Step with point action
        next_timestep = wrapper.step(env_params, timestep, point_action)
        
        # Verify that coordinates were clipped
        selection = mock_env.last_action.selection
        assert selection.shape == (5, 5)
        assert jnp.sum(selection) == 1  # Only one point selected
        
        # The point should be clipped to valid bounds (4, 0)
        assert selection[4, 0] == True
    
    def test_point_action_jax_compatibility(self, prng_key: PRNGKey):
        """Test PointActionWrapper JAX compatibility."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Test JIT compilation of step function
        def step_fn(env_params, timestep, action):
            return wrapper.step(env_params, timestep, action)
        
        jitted_step = jax.jit(step_fn)
        
        # Reset and create action
        timestep = wrapper.reset(env_params, prng_key)
        point_action = {
            "operation": jnp.array(7, dtype=jnp.int32),
            "row": jnp.array(1, dtype=jnp.int32),
            "col": jnp.array(2, dtype=jnp.int32)
        }
        
        # Test JIT compilation works
        next_timestep = jitted_step(env_params, timestep, point_action)
        assert isinstance(next_timestep, TimeStep)


class TestBboxActionWrapper:
    """Test BboxActionWrapper functionality."""
    
    def test_bbox_action_wrapper_creation(self):
        """Test BboxActionWrapper creation."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        assert isinstance(wrapper, BboxActionWrapper)
        assert wrapper._env is mock_env
    
    def test_bbox_action_space(self):
        """Test BboxActionWrapper action space."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        action_space = wrapper.action_space(env_params)
        
        assert isinstance(action_space, DictSpace)
        assert "operation" in action_space._spaces
        assert "r1" in action_space._spaces
        assert "c1" in action_space._spaces
        assert "r2" in action_space._spaces
        assert "c2" in action_space._spaces
        
        for key in ["operation", "r1", "c1", "r2", "c2"]:
            assert isinstance(action_space._spaces[key], DiscreteSpace)
    
    def test_bbox_action_conversion(self, prng_key: PRNGKey):
        """Test bbox action to mask conversion."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create bbox action (2x2 rectangle from (1,1) to (2,2))
        bbox_action = {
            "operation": jnp.array(15, dtype=jnp.int32),
            "r1": jnp.array(1, dtype=jnp.int32),
            "c1": jnp.array(1, dtype=jnp.int32),
            "r2": jnp.array(2, dtype=jnp.int32),
            "c2": jnp.array(2, dtype=jnp.int32)
        }
        
        # Step with bbox action
        next_timestep = wrapper.step(env_params, timestep, bbox_action)
        
        # Verify that the action was converted to mask action
        assert mock_env.step_called
        assert isinstance(mock_env.last_action, Action)
        assert mock_env.last_action.operation == 15
        
        # Verify that the rectangular region is selected
        selection = mock_env.last_action.selection
        assert selection.shape == (5, 5)
        
        # Check that the 2x2 rectangle is selected
        expected_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        expected_selection = expected_selection.at[1:3, 1:3].set(True)
        
        assert jnp.array_equal(selection, expected_selection)
        assert jnp.sum(selection) == 4  # 2x2 = 4 cells
    
    def test_bbox_action_coordinate_ordering(self, prng_key: PRNGKey):
        """Test that bbox coordinates are properly ordered (min, max)."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create bbox action with reversed coordinates
        bbox_action = {
            "operation": jnp.array(8, dtype=jnp.int32),
            "r1": jnp.array(3, dtype=jnp.int32),  # Larger row
            "c1": jnp.array(3, dtype=jnp.int32),  # Larger col
            "r2": jnp.array(1, dtype=jnp.int32),  # Smaller row
            "c2": jnp.array(1, dtype=jnp.int32)   # Smaller col
        }
        
        # Step with bbox action
        next_timestep = wrapper.step(env_params, timestep, bbox_action)
        
        # Verify that coordinates were properly ordered
        selection = mock_env.last_action.selection
        
        # Should select the same 3x3 rectangle regardless of coordinate order
        expected_selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        expected_selection = expected_selection.at[1:4, 1:4].set(True)
        
        assert jnp.array_equal(selection, expected_selection)
        assert jnp.sum(selection) == 9  # 3x3 = 9 cells
    
    def test_bbox_action_bounds_clipping(self, prng_key: PRNGKey):
        """Test that bbox coordinates are clipped to valid bounds."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create bbox action with out-of-bounds coordinates
        bbox_action = {
            "operation": jnp.array(12, dtype=jnp.int32),
            "r1": jnp.array(-1, dtype=jnp.int32),   # Out of bounds
            "c1": jnp.array(-1, dtype=jnp.int32),   # Out of bounds
            "r2": jnp.array(10, dtype=jnp.int32),   # Out of bounds (grid is 5x5)
            "c2": jnp.array(10, dtype=jnp.int32)    # Out of bounds
        }
        
        # Step with bbox action
        next_timestep = wrapper.step(env_params, timestep, bbox_action)
        
        # Verify that coordinates were clipped to valid bounds
        selection = mock_env.last_action.selection
        
        # Should select the entire grid (clipped to 0,0 -> 4,4)
        expected_selection = jnp.ones((5, 5), dtype=jnp.bool_)
        
        assert jnp.array_equal(selection, expected_selection)
        assert jnp.sum(selection) == 25  # Entire 5x5 grid
    
    def test_bbox_action_single_point(self, prng_key: PRNGKey):
        """Test bbox action that selects a single point."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create bbox action that selects a single point
        bbox_action = {
            "operation": jnp.array(20, dtype=jnp.int32),
            "r1": jnp.array(2, dtype=jnp.int32),
            "c1": jnp.array(3, dtype=jnp.int32),
            "r2": jnp.array(2, dtype=jnp.int32),  # Same as r1
            "c2": jnp.array(3, dtype=jnp.int32)   # Same as c1
        }
        
        # Step with bbox action
        next_timestep = wrapper.step(env_params, timestep, bbox_action)
        
        # Verify that only a single point is selected
        selection = mock_env.last_action.selection
        assert selection.shape == (5, 5)
        assert selection[2, 3] == True
        assert jnp.sum(selection) == 1  # Only one point selected


class TestGymAutoResetWrapper:
    """Test GymAutoResetWrapper functionality."""
    
    def test_gym_auto_reset_wrapper_creation(self):
        """Test GymAutoResetWrapper creation."""
        mock_env = MockEnvironment()
        wrapper = GymAutoResetWrapper(mock_env)
        
        assert isinstance(wrapper, GymAutoResetWrapper)
        assert wrapper._env is mock_env
    
    def test_gym_auto_reset_normal_step(self, prng_key: PRNGKey):
        """Test that normal steps are passed through unchanged."""
        mock_env = MockEnvironment()
        wrapper = GymAutoResetWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create action
        action = create_action(
            operation=jnp.array(5, dtype=jnp.int32),
            selection=jnp.ones((5, 5), dtype=jnp.bool_)
        )
        
        # Step (should be normal, not terminal)
        next_timestep = wrapper.step(env_params, timestep, action)
        
        # Should pass through normally
        assert isinstance(next_timestep, TimeStep)
        assert next_timestep.step_type == 1  # MID step
    
    def test_gym_auto_reset_terminal_step(self, prng_key: PRNGKey):
        """Test auto-reset behavior on terminal steps."""
        # Create a mock environment that returns terminal timesteps
        class TerminalMockEnvironment(MockEnvironment):
            def step(self, params: EnvParams, timestep: TimeStep, action: Action) -> TimeStep:
                self.step_called = True
                self.last_action = action
                
                # Return a terminal timestep
                import equinox as eqx
                terminal_state = eqx.tree_at(
                    lambda state: state.step_count,
                    timestep.state,
                    timestep.state.step_count + 1
                )
                
                return TimeStep(
                    state=terminal_state,
                    step_type=jnp.array(2, dtype=jnp.int32),  # LAST (terminal)
                    reward=jnp.array(10.0, dtype=jnp.float32),
                    discount=jnp.array(0.0, dtype=jnp.float32),  # Terminal discount
                    observation=terminal_state.working_grid,
                    extras={}
                )
        
        mock_env = TerminalMockEnvironment()
        wrapper = GymAutoResetWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Reset environment
        timestep = wrapper.reset(env_params, prng_key)
        
        # Create action
        action = create_action(
            operation=jnp.array(5, dtype=jnp.int32),
            selection=jnp.ones((5, 5), dtype=jnp.bool_)
        )
        
        # Step (should trigger auto-reset)
        next_timestep = wrapper.step(env_params, timestep, action)
        
        # Should have auto-reset while preserving terminal info
        assert isinstance(next_timestep, TimeStep)
        assert next_timestep.step_type == 2  # Still LAST (terminal info preserved)
        assert next_timestep.reward == 10.0  # Terminal reward preserved
        assert next_timestep.discount == 0.0  # Terminal discount preserved
        
        # But state should be reset (step_count should be 0)
        assert next_timestep.state.step_count == 0


class TestWrapperIntegration:
    """Test wrapper integration and compatibility."""
    
    def test_wrapper_chaining(self, prng_key: PRNGKey):
        """Test that wrappers can be chained together."""
        mock_env = MockEnvironment()
        
        # Chain multiple wrappers
        wrapped_env = PointActionWrapper(mock_env)
        double_wrapped = GymAutoResetWrapper(wrapped_env)
        
        env_params = create_test_env_params()
        
        # Test that chained wrappers work
        timestep = double_wrapped.reset(env_params, prng_key)
        assert isinstance(timestep, TimeStep)
        
        # Test point action through chained wrappers
        point_action = {
            "operation": jnp.array(7, dtype=jnp.int32),
            "row": jnp.array(1, dtype=jnp.int32),
            "col": jnp.array(2, dtype=jnp.int32)
        }
        
        next_timestep = double_wrapped.step(env_params, timestep, point_action)
        assert isinstance(next_timestep, TimeStep)
        
        # Verify that the action was properly converted
        assert mock_env.step_called
        assert isinstance(mock_env.last_action, Action)
    
    def test_wrapper_jax_compatibility(self, prng_key: PRNGKey):
        """Test that wrappers maintain JAX compatibility."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        # Test JIT compilation of wrapped functions
        def reset_step_fn(env_params, key):
            timestep = wrapper.reset(env_params, key)
            action = {
                "operation": jnp.array(5, dtype=jnp.int32),
                "row": jnp.array(2, dtype=jnp.int32),
                "col": jnp.array(2, dtype=jnp.int32)
            }
            return wrapper.step(env_params, timestep, action)
        
        jitted_fn = jax.jit(reset_step_fn)
        result = jitted_fn(env_params, prng_key)
        
        assert isinstance(result, TimeStep)
    
    def test_wrapper_with_real_environment(self):
        """Test wrappers with a real environment (if available)."""
        # This test would use a real environment if available
        # For now, we'll skip it since we're using mock environments
        pytest.skip("Real environment integration test - requires full environment setup")
    
    def test_action_wrapper_space_consistency(self):
        """Test that action wrappers provide consistent action spaces."""
        mock_env = MockEnvironment()
        
        env_params = create_test_env_params()
        
        # Test PointActionWrapper
        point_wrapper = PointActionWrapper(mock_env)
        point_space = point_wrapper.action_space(env_params)
        
        assert isinstance(point_space, DictSpace)
        assert len(point_space._spaces) == 3  # operation, row, col
        
        # Test BboxActionWrapper
        bbox_wrapper = BboxActionWrapper(mock_env)
        bbox_space = bbox_wrapper.action_space(env_params)
        
        assert isinstance(bbox_space, DictSpace)
        assert len(bbox_space._spaces) == 5  # operation, r1, c1, r2, c2
        
        # Both should have the same operation space
        assert point_space._spaces["operation"].num_values == bbox_space._spaces["operation"].num_values


class TestWrapperEdgeCases:
    """Test edge cases and error conditions for wrappers."""
    
    def test_wrapper_with_none_environment(self):
        """Test wrapper behavior with None environment."""
        # The wrapper may accept None and only fail when methods are called
        wrapper = Wrapper(None)
        assert wrapper._env is None
    
    def test_point_action_with_missing_keys(self, prng_key: PRNGKey):
        """Test PointActionWrapper with missing action keys."""
        mock_env = MockEnvironment()
        wrapper = PointActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        timestep = wrapper.reset(env_params, prng_key)
        
        # Action missing required keys
        incomplete_action = {
            "operation": jnp.array(5, dtype=jnp.int32),
            "row": jnp.array(2, dtype=jnp.int32)
            # Missing "col"
        }
        
        # Should raise KeyError or similar
        with pytest.raises(KeyError):
            wrapper.step(env_params, timestep, incomplete_action)
    
    def test_bbox_action_with_missing_keys(self, prng_key: PRNGKey):
        """Test BboxActionWrapper with missing action keys."""
        mock_env = MockEnvironment()
        wrapper = BboxActionWrapper(mock_env)
        
        env_params = create_test_env_params()
        
        timestep = wrapper.reset(env_params, prng_key)
        
        # Action missing required keys
        incomplete_action = {
            "operation": jnp.array(5, dtype=jnp.int32),
            "r1": jnp.array(1, dtype=jnp.int32),
            "c1": jnp.array(1, dtype=jnp.int32)
            # Missing "r2" and "c2"
        }
        
        # Should raise KeyError or similar
        with pytest.raises(KeyError):
            wrapper.step(env_params, timestep, incomplete_action)
    
    def test_wrapper_attribute_error_handling(self):
        """Test wrapper behavior when accessing non-existent attributes."""
        mock_env = MockEnvironment()
        wrapper = Wrapper(mock_env)
        
        # Should raise AttributeError for non-existent attributes
        with pytest.raises(AttributeError):
            _ = wrapper.non_existent_method()