from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxarc.base.types import AgentID, ArcTask, Grid, Hypothesis, TaskPair


# Test Grid class
def test_grid_creation():
    """Tests the creation of a Grid object with valid data."""
    array = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    grid = Grid(array=array)

    chex.assert_shape(grid.array, (2, 3))
    chex.assert_type(grid.array, jnp.integer)
    chex.assert_trees_all_equal(grid.array, array)


def test_grid_invalid_rank():
    """Tests that Grid raises assertion error for non-2D arrays."""
    with pytest.raises(AssertionError):
        Grid(array=jnp.array([1, 2, 3], dtype=jnp.int32))  # 1D array

    with pytest.raises(AssertionError):
        Grid(array=jnp.array([[[1]]], dtype=jnp.int32))  # 3D array


def test_grid_non_integer_type():
    """Tests that Grid raises assertion error for non-integer arrays."""
    with pytest.raises(AssertionError):
        Grid(array=jnp.array([[1.0, 2.0]], dtype=jnp.float32))


# Test TaskPair class
def test_task_pair_creation():
    """Tests the creation of a TaskPair object."""
    input_grid = Grid(array=jnp.array([[0, 1]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1, 0]], dtype=jnp.int32))

    task_pair = TaskPair(input=input_grid, output=output_grid)

    assert isinstance(task_pair.input, Grid)
    assert isinstance(task_pair.output, Grid)
    chex.assert_trees_all_equal(
        task_pair.input.array, jnp.array([[0, 1]], dtype=jnp.int32)
    )
    chex.assert_trees_all_equal(
        task_pair.output.array, jnp.array([[1, 0]], dtype=jnp.int32)
    )


# Test ArcTask class
def test_arc_task_creation():
    """Tests the creation of an ArcTask object."""
    input_grid = Grid(array=jnp.array([[0, 1]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1, 0]], dtype=jnp.int32))
    task_pair = TaskPair(input=input_grid, output=output_grid)

    arc_task = ArcTask(
        train_pairs=[task_pair], test_pairs=[task_pair], task_id="test_task_001"
    )

    assert len(arc_task.train_pairs) == 1
    assert len(arc_task.test_pairs) == 1
    assert arc_task.task_id == "test_task_001"
    assert isinstance(arc_task.train_pairs[0], TaskPair)
    assert isinstance(arc_task.test_pairs[0], TaskPair)


def test_arc_task_optional_id():
    """Tests ArcTask creation with optional task_id."""
    input_grid = Grid(array=jnp.array([[0]], dtype=jnp.int32))
    output_grid = Grid(array=jnp.array([[1]], dtype=jnp.int32))
    task_pair = TaskPair(input=input_grid, output=output_grid)

    arc_task = ArcTask(train_pairs=[task_pair], test_pairs=[task_pair])

    assert arc_task.task_id is None


# Test Hypothesis class
def test_hypothesis_creation():
    """Tests the creation of a Hypothesis object with valid data."""
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        data=jnp.array([1, 2, 3], dtype=jnp.int32),
        description="Test hypothesis",
    )

    chex.assert_shape(hypothesis.step_number, ())
    chex.assert_shape(hypothesis.confidence, ())
    chex.assert_shape(hypothesis.data, (3,))
    assert hypothesis.agent_id == AgentID(1)
    assert hypothesis.description == "Test hypothesis"


def test_hypothesis_pytree_compatibility():
    """Tests if Hypothesis JAX arrays can be used in JAX transformations."""

    def process_array_fields(data: jax.Array) -> jax.Array:
        return data * 2

    # Test that the JAX arrays within Hypothesis can be processed
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        data=jnp.array([1, 2, 3], dtype=jnp.int32),
        description="Test hypothesis",
    )

    # Compile the function that works on individual JAX arrays
    jitted_process = jax.jit(process_array_fields)
    processed_data = jitted_process(hypothesis.data) # pylint: disable=not-callable
    chex.assert_trees_all_equal(processed_data, jnp.array([2, 4, 6], dtype=jnp.int32))

    # Verify that the Hypothesis can be created and its array fields are valid JAX arrays
    assert isinstance(hypothesis.data, jnp.ndarray)
    assert isinstance(hypothesis.step_number, jnp.ndarray)
    assert isinstance(hypothesis.confidence, jnp.ndarray)


def test_hypothesis_optional_fields():
    """Tests Hypothesis creation with optional fields set to None."""
    hypothesis = Hypothesis(
        agent_id=AgentID(1),
        step_number=jnp.array(5, dtype=jnp.int32),
        confidence=jnp.array(0.8, dtype=jnp.float32),
        # data and description are optional
    )

    assert hypothesis.data is None
    assert hypothesis.description is None

    # Test with data provided but description as None
    hypothesis_with_data = Hypothesis(
        agent_id=AgentID(2),
        step_number=jnp.array(1, dtype=jnp.int32),
        confidence=jnp.array(0.9, dtype=jnp.float32),
        data=jnp.array([4, 5], dtype=jnp.int32),
    )

    assert hypothesis_with_data.description is None
    chex.assert_trees_all_equal(
        hypothesis_with_data.data, jnp.array([4, 5], dtype=jnp.int32)
    )


def test_hypothesis_field_types():
    """Tests that Hypothesis fields enforce correct types where possible."""
    # Test that invalid JAX array types/shapes cause errors during __post_init__

    # Test with wrong dtype for step_number (should be int32)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            step_number=jnp.array(5, dtype=jnp.float32),  # Wrong dtype
            confidence=jnp.array(0.8, dtype=jnp.float32),
        )

    # Test with wrong dtype for confidence (should be float32)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            step_number=jnp.array(5, dtype=jnp.int32),
            confidence=jnp.array(0.8, dtype=jnp.int32),  # Wrong dtype
        )

    # Test with wrong shape for step_number (should be scalar)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            step_number=jnp.array([5, 6], dtype=jnp.int32),  # Wrong shape
            confidence=jnp.array(0.8, dtype=jnp.float32),
        )

    # Test with wrong shape for confidence (should be scalar)
    with pytest.raises(AssertionError):
        Hypothesis(
            agent_id=AgentID(1),
            step_number=jnp.array(5, dtype=jnp.int32),
            confidence=jnp.array([0.8, 0.9], dtype=jnp.float32),  # Wrong shape
        )


def test_hypothesis_confidence_bounds():
    """Tests that confidence values are properly validated."""
    # Valid confidence values should work
    for conf in [0.0, 0.5, 1.0]:
        hypothesis = Hypothesis(
            agent_id=AgentID(1),
            step_number=jnp.array(1, dtype=jnp.int32),
            confidence=jnp.array(conf, dtype=jnp.float32),
        )
        assert hypothesis.confidence == conf
