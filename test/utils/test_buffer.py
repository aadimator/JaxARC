"""Tests for buffer utility functions."""

import jax
import jax.numpy as jnp
import pytest
import chex

from jaxarc.types import JaxArcTask
from jaxarc.utils.buffer import (
    stack_task_list,
    gather_task,
    buffer_size,
    _to_jax_scalar_if_int,
    _preprocess_task_for_stacking,
)


@pytest.fixture
def sample_task():
    """Create a sample JaxArcTask for testing."""
    return JaxArcTask(
        task_id="sample_task",
        input_grids_examples=jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        output_grids_examples=jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
        input_grids_test=jnp.array([[[17, 18], [19, 20]]]),
        output_grids_test=jnp.array([[[21, 22], [23, 24]]]),
        num_examples=2,
        num_test=1,
        max_grid_height=2,
        max_grid_width=2,
    )


@pytest.fixture
def sample_task_list(sample_task):
    """Create a list of sample tasks for testing."""
    tasks = []
    for i in range(3):
        task = JaxArcTask(
            task_id=f"task_{i}",
            input_grids_examples=jnp.array([[[i, i+1], [i+2, i+3]], [[i+4, i+5], [i+6, i+7]]]),
            output_grids_examples=jnp.array([[[i+8, i+9], [i+10, i+11]], [[i+12, i+13], [i+14, i+15]]]),
            input_grids_test=jnp.array([[[i+16, i+17], [i+18, i+19]]]),
            output_grids_test=jnp.array([[[i+20, i+21], [i+22, i+23]]]),
            num_examples=2,
            num_test=1,
            max_grid_height=2,
            max_grid_width=2,
        )
        tasks.append(task)
    return tasks


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_to_jax_scalar_if_int_with_int(self):
        """Test converting Python int to JAX scalar."""
        result = _to_jax_scalar_if_int(42)
        
        chex.assert_shape(result, ())
        assert result.dtype == jnp.int32
        assert int(result) == 42

    def test_to_jax_scalar_if_int_with_bool(self):
        """Test that Python bool is left unchanged."""
        result = _to_jax_scalar_if_int(True)
        assert result is True
        assert isinstance(result, bool)
        
        result = _to_jax_scalar_if_int(False)
        assert result is False
        assert isinstance(result, bool)

    def test_to_jax_scalar_if_int_with_jax_array(self):
        """Test that JAX arrays are left unchanged."""
        array = jnp.array([1, 2, 3])
        result = _to_jax_scalar_if_int(array)
        
        chex.assert_trees_all_equal(result, array)

    def test_to_jax_scalar_if_int_with_other_types(self):
        """Test that other types are left unchanged."""
        # Test string
        result = _to_jax_scalar_if_int("test")
        assert result == "test"
        
        # Test float
        result = _to_jax_scalar_if_int(3.14)
        assert result == 3.14
        
        # Test None
        result = _to_jax_scalar_if_int(None)
        assert result is None

    def test_preprocess_task_for_stacking(self, sample_task):
        """Test preprocessing task for stacking."""
        # Create a task with Python int fields
        task_with_ints = JaxArcTask(
            task_id="test_task",
            input_grids_examples=jnp.array([[[1, 2]]]),
            output_grids_examples=jnp.array([[[3, 4]]]),
            input_grids_test=jnp.array([[[5, 6]]]),
            output_grids_test=jnp.array([[[7, 8]]]),
            num_examples=2,  # Python int
            num_test=1,      # Python int
            max_grid_height=3,  # Python int
            max_grid_width=4,   # Python int
        )
        
        processed = _preprocess_task_for_stacking(task_with_ints)
        
        # Check that Python ints were converted to JAX scalars
        assert isinstance(processed.num_examples, jnp.ndarray)
        assert processed.num_examples.dtype == jnp.int32
        assert int(processed.num_examples) == 2
        
        assert isinstance(processed.num_test, jnp.ndarray)
        assert processed.num_test.dtype == jnp.int32
        assert int(processed.num_test) == 1
        
        # Check that arrays were left unchanged
        chex.assert_trees_all_equal(processed.input_grids_examples, task_with_ints.input_grids_examples)


class TestStackTaskList:
    """Test the stack_task_list function."""

    def test_stack_task_list_basic(self, sample_task_list):
        """Test basic task list stacking."""
        buffer = stack_task_list(sample_task_list)
        
        # Check that buffer has same structure as individual task
        assert hasattr(buffer, 'task_id')
        assert hasattr(buffer, 'input_grids_examples')
        assert hasattr(buffer, 'output_grids_examples')
        assert hasattr(buffer, 'num_examples')
        
        # Check that arrays have leading batch dimension
        chex.assert_shape(buffer.input_grids_examples, (3, 2, 2, 2))  # (batch, examples, height, width)
        chex.assert_shape(buffer.output_grids_examples, (3, 2, 2, 2))
        chex.assert_shape(buffer.input_grids_test, (3, 1, 2, 2))
        chex.assert_shape(buffer.output_grids_test, (3, 1, 2, 2))
        
        # Check that scalar fields are stacked
        chex.assert_shape(buffer.num_examples, (3,))
        chex.assert_shape(buffer.num_test, (3,))
        chex.assert_shape(buffer.max_grid_height, (3,))
        chex.assert_shape(buffer.max_grid_width, (3,))

    def test_stack_task_list_single_task(self, sample_task):
        """Test stacking single task."""
        buffer = stack_task_list([sample_task])
        
        # Should have batch dimension of 1
        chex.assert_shape(buffer.input_grids_examples, (1, 2, 2, 2))
        chex.assert_shape(buffer.num_examples, (1,))
        
        # Values should match original task
        chex.assert_trees_all_equal(buffer.input_grids_examples[0], sample_task.input_grids_examples)
        assert int(buffer.num_examples[0]) == sample_task.num_examples

    def test_stack_task_list_empty_raises_error(self):
        """Test that empty task list raises error."""
        with pytest.raises(ValueError, match="must be a non-empty sequence"):
            stack_task_list([])

    def test_stack_task_list_preserves_dtypes(self, sample_task_list):
        """Test that stacking preserves data types."""
        buffer = stack_task_list(sample_task_list)
        
        # Check that array dtypes are preserved
        assert buffer.input_grids_examples.dtype == sample_task_list[0].input_grids_examples.dtype
        assert buffer.output_grids_examples.dtype == sample_task_list[0].output_grids_examples.dtype
        
        # Check that converted scalars have int32 dtype
        assert buffer.num_examples.dtype == jnp.int32
        assert buffer.num_test.dtype == jnp.int32

    def test_stack_task_list_with_different_task_ids(self):
        """Test stacking tasks with different task IDs."""
        tasks = []
        task_ids = ["task_a", "task_b", "task_c"]
        
        for task_id in task_ids:
            task = JaxArcTask(
                task_id=task_id,
                input_grids_examples=jnp.array([[[1, 2]]]),
                output_grids_examples=jnp.array([[[3, 4]]]),
                input_grids_test=jnp.array([[[5, 6]]]),
                output_grids_test=jnp.array([[[7, 8]]]),
                num_examples=1,
                num_test=1,
                max_grid_height=1,
                max_grid_width=2,
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        
        # Task IDs should be stacked as strings
        assert len(buffer.task_id) == 3
        for i, expected_id in enumerate(task_ids):
            assert buffer.task_id[i] == expected_id

    def test_stack_task_list_jit_compatible(self, sample_task_list):
        """Test that stacked buffer is JIT compatible."""
        buffer = stack_task_list(sample_task_list)
        
        # Test that we can JIT a function that uses the buffer
        @jax.jit
        def process_buffer(buf):
            return jnp.sum(buf.input_grids_examples)
        
        result = process_buffer(buffer)
        expected = jnp.sum(buffer.input_grids_examples)
        
        chex.assert_trees_all_equal(result, expected)


class TestGatherTask:
    """Test the gather_task function."""

    def test_gather_task_basic(self, sample_task_list):
        """Test basic task gathering from buffer."""
        buffer = stack_task_list(sample_task_list)
        
        # Gather first task
        gathered_task = gather_task(buffer, 0)
        
        # Should have same structure as original task
        assert hasattr(gathered_task, 'task_id')
        assert hasattr(gathered_task, 'input_grids_examples')
        assert hasattr(gathered_task, 'num_examples')
        
        # Arrays should have batch dimension removed
        chex.assert_shape(gathered_task.input_grids_examples, (2, 2, 2))
        chex.assert_shape(gathered_task.output_grids_examples, (2, 2, 2))
        chex.assert_shape(gathered_task.input_grids_test, (1, 2, 2))
        
        # Scalars should remain scalars
        chex.assert_shape(gathered_task.num_examples, ())
        chex.assert_shape(gathered_task.num_test, ())
        
        # Values should match original first task
        chex.assert_trees_all_equal(gathered_task.input_grids_examples, sample_task_list[0].input_grids_examples)
        assert int(gathered_task.num_examples) == sample_task_list[0].num_examples
        assert gathered_task.task_id == sample_task_list[0].task_id

    def test_gather_task_different_indices(self, sample_task_list):
        """Test gathering different tasks by index."""
        buffer = stack_task_list(sample_task_list)
        
        for i in range(len(sample_task_list)):
            gathered_task = gather_task(buffer, i)
            
            # Should match original task at index i
            chex.assert_trees_all_equal(gathered_task.input_grids_examples, sample_task_list[i].input_grids_examples)
            assert gathered_task.task_id == sample_task_list[i].task_id
            assert int(gathered_task.num_examples) == sample_task_list[i].num_examples

    def test_gather_task_jax_index(self, sample_task_list):
        """Test gathering with JAX array index."""
        buffer = stack_task_list(sample_task_list)
        
        # Use JAX array as index
        jax_index = jnp.array(1, dtype=jnp.int32)
        gathered_task = gather_task(buffer, jax_index)
        
        # Should match second task
        chex.assert_trees_all_equal(gathered_task.input_grids_examples, sample_task_list[1].input_grids_examples)
        assert gathered_task.task_id == sample_task_list[1].task_id

    def test_gather_task_jit_compatible(self, sample_task_list):
        """Test that gather_task is JIT compatible."""
        buffer = stack_task_list(sample_task_list)
        
        @jax.jit
        def jitted_gather(buf, idx):
            return gather_task(buf, idx)
        
        # Test with different indices
        for i in range(len(sample_task_list)):
            jit_result = jitted_gather(buffer, i)
            expected_result = gather_task(buffer, i)
            
            chex.assert_trees_all_equal(jit_result, expected_result)

    def test_gather_task_preserves_scalars(self):
        """Test that gather_task preserves scalar (0-dim) arrays."""
        # Create buffer with scalar fields
        tasks = []
        for i in range(2):
            task = JaxArcTask(
                task_id=f"task_{i}",
                input_grids_examples=jnp.array([[[i]]]),
                output_grids_examples=jnp.array([[[i+1]]]),
                input_grids_test=jnp.array([[[i+2]]]),
                output_grids_test=jnp.array([[[i+3]]]),
                num_examples=jnp.array(1),  # Already JAX scalar
                num_test=jnp.array(1),
                max_grid_height=jnp.array(1),
                max_grid_width=jnp.array(1),
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        gathered = gather_task(buffer, 0)
        
        # Scalars should remain scalars after gathering
        chex.assert_shape(gathered.num_examples, ())
        chex.assert_shape(gathered.num_test, ())


class TestBufferSize:
    """Test the buffer_size function."""

    def test_buffer_size_basic(self, sample_task_list):
        """Test basic buffer size calculation."""
        buffer = stack_task_list(sample_task_list)
        size = buffer_size(buffer)
        
        assert size == len(sample_task_list)
        assert size == 3

    def test_buffer_size_single_task(self, sample_task):
        """Test buffer size for single task."""
        buffer = stack_task_list([sample_task])
        size = buffer_size(buffer)
        
        assert size == 1

    def test_buffer_size_canonical_field(self, sample_task_list):
        """Test buffer size using canonical field."""
        buffer = stack_task_list(sample_task_list)
        
        # Should use input_grids_examples.shape[0] as canonical field
        expected_size = buffer.input_grids_examples.shape[0]
        actual_size = buffer_size(buffer)
        
        assert actual_size == expected_size

    def test_buffer_size_fallback_to_first_array(self):
        """Test buffer size fallback when canonical field is not available."""
        # Create a mock buffer without input_grids_examples
        class MockBuffer:
            def __init__(self):
                self.some_array = jnp.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
                self.scalar_field = jnp.array(42)  # Scalar
        
        mock_buffer = MockBuffer()
        size = buffer_size(mock_buffer)
        
        assert size == 3  # Should use first array's leading dimension

    def test_buffer_size_no_arrays_raises_error(self):
        """Test that buffer with no arrays raises error."""
        class EmptyBuffer:
            def __init__(self):
                self.scalar_only = 42
        
        empty_buffer = EmptyBuffer()
        
        with pytest.raises(ValueError, match="could not infer leading batch size"):
            buffer_size(empty_buffer)

    def test_buffer_size_large_buffer(self):
        """Test buffer size with large number of tasks."""
        # Create many simple tasks
        tasks = []
        num_tasks = 100
        
        for i in range(num_tasks):
            task = JaxArcTask(
                task_id=f"task_{i:03d}",
                input_grids_examples=jnp.array([[[i]]]),
                output_grids_examples=jnp.array([[[i+1]]]),
                input_grids_test=jnp.array([[[i+2]]]),
                output_grids_test=jnp.array([[[i+3]]]),
                num_examples=1,
                num_test=1,
                max_grid_height=1,
                max_grid_width=1,
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        size = buffer_size(buffer)
        
        assert size == num_tasks


class TestIntegrationAndWorkflows:
    """Test integration scenarios and complete workflows."""

    def test_stack_gather_roundtrip(self, sample_task_list):
        """Test that stack -> gather preserves original tasks."""
        buffer = stack_task_list(sample_task_list)
        
        for i, original_task in enumerate(sample_task_list):
            gathered_task = gather_task(buffer, i)
            
            # Compare all fields
            chex.assert_trees_all_equal(gathered_task.input_grids_examples, original_task.input_grids_examples)
            chex.assert_trees_all_equal(gathered_task.output_grids_examples, original_task.output_grids_examples)
            chex.assert_trees_all_equal(gathered_task.input_grids_test, original_task.input_grids_test)
            chex.assert_trees_all_equal(gathered_task.output_grids_test, original_task.output_grids_test)
            
            assert gathered_task.task_id == original_task.task_id
            assert int(gathered_task.num_examples) == original_task.num_examples
            assert int(gathered_task.num_test) == original_task.num_test
            assert int(gathered_task.max_grid_height) == original_task.max_grid_height
            assert int(gathered_task.max_grid_width) == original_task.max_grid_width

    def test_random_task_selection_workflow(self, sample_task_list):
        """Test workflow for random task selection."""
        buffer = stack_task_list(sample_task_list)
        key = jax.random.PRNGKey(42)
        
        # Simulate random task selection
        @jax.jit
        def select_random_task(buf, rng_key):
            size = buffer_size(buf)
            idx = jax.random.randint(rng_key, (), 0, size)
            return gather_task(buf, idx), idx
        
        selected_task, selected_idx = select_random_task(buffer, key)
        
        # Verify selected task matches the one at selected index
        expected_task = gather_task(buffer, selected_idx)
        chex.assert_trees_all_equal(selected_task, expected_task)
        
        # Verify index is in valid range
        assert 0 <= int(selected_idx) < len(sample_task_list)

    def test_vmap_compatibility(self, sample_task_list):
        """Test that buffer operations work with vmap."""
        buffer = stack_task_list(sample_task_list)
        
        # Create multiple indices for vmapping
        indices = jnp.array([0, 1, 2])
        
        # Vmap gather_task over indices
        vmapped_gather = jax.vmap(lambda buf, idx: gather_task(buf, idx), in_axes=(None, 0))
        gathered_tasks = vmapped_gather(buffer, indices)
        
        # Check that we got all tasks back
        chex.assert_shape(gathered_tasks.input_grids_examples, (3, 2, 2, 2))
        chex.assert_shape(gathered_tasks.num_examples, (3,))
        
        # Verify each gathered task matches original
        for i in range(len(sample_task_list)):
            original_task = sample_task_list[i]
            gathered_slice = jax.tree_util.tree_map(lambda x: x[i], gathered_tasks)
            
            chex.assert_trees_all_equal(gathered_slice.input_grids_examples, original_task.input_grids_examples)
            assert gathered_slice.task_id == original_task.task_id

    def test_buffer_with_different_sized_tasks(self):
        """Test buffer operations with tasks of different content sizes."""
        # Create tasks with different numbers of examples (but same padded shape)
        tasks = []
        for i in range(3):
            # All tasks have same padded shape but different actual content
            task = JaxArcTask(
                task_id=f"task_{i}",
                input_grids_examples=jnp.array([[[i, i+1], [i+2, i+3]], [[0, 0], [0, 0]]]),  # Second example is padding
                output_grids_examples=jnp.array([[[i+4, i+5], [i+6, i+7]], [[0, 0], [0, 0]]]),
                input_grids_test=jnp.array([[[i+8, i+9], [i+10, i+11]]]),
                output_grids_test=jnp.array([[[i+12, i+13], [i+14, i+15]]]),
                num_examples=1 if i == 0 else 2,  # Different number of actual examples
                num_test=1,
                max_grid_height=2,
                max_grid_width=2,
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        
        # Verify stacking worked
        assert buffer_size(buffer) == 3
        
        # Verify gathering preserves different num_examples
        for i, original_task in enumerate(tasks):
            gathered = gather_task(buffer, i)
            assert int(gathered.num_examples) == original_task.num_examples


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_buffer_with_zero_dimensional_arrays(self):
        """Test buffer operations with 0-dimensional arrays."""
        tasks = []
        for i in range(2):
            task = JaxArcTask(
                task_id=f"task_{i}",
                input_grids_examples=jnp.array([[[i]]]),
                output_grids_examples=jnp.array([[[i+1]]]),
                input_grids_test=jnp.array([[[i+2]]]),
                output_grids_test=jnp.array([[[i+3]]]),
                num_examples=jnp.array(1),  # 0-dim array
                num_test=jnp.array(1),
                max_grid_height=jnp.array(1),
                max_grid_width=jnp.array(1),
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        
        # 0-dim arrays should become 1-dim after stacking
        chex.assert_shape(buffer.num_examples, (2,))
        
        # Gathering should restore 0-dim
        gathered = gather_task(buffer, 0)
        chex.assert_shape(gathered.num_examples, ())

    def test_buffer_with_mixed_dtypes(self):
        """Test buffer with mixed data types."""
        tasks = []
        for i in range(2):
            task = JaxArcTask(
                task_id=f"task_{i}",
                input_grids_examples=jnp.array([[[i]]], dtype=jnp.int32),
                output_grids_examples=jnp.array([[[i+1]]], dtype=jnp.int16),  # Different dtype
                input_grids_test=jnp.array([[[i+2]]], dtype=jnp.int32),
                output_grids_test=jnp.array([[[i+3]]], dtype=jnp.int32),
                num_examples=1,
                num_test=1,
                max_grid_height=1,
                max_grid_width=1,
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        
        # Dtypes should be preserved
        assert buffer.input_grids_examples.dtype == jnp.int32
        assert buffer.output_grids_examples.dtype == jnp.int16
        
        # Gathering should preserve dtypes
        gathered = gather_task(buffer, 0)
        assert gathered.input_grids_examples.dtype == jnp.int32
        assert gathered.output_grids_examples.dtype == jnp.int16

    def test_buffer_operations_with_large_indices(self, sample_task_list):
        """Test buffer operations with edge case indices."""
        buffer = stack_task_list(sample_task_list)
        size = buffer_size(buffer)
        
        # Test with maximum valid index
        max_idx = size - 1
        gathered = gather_task(buffer, max_idx)
        expected = sample_task_list[max_idx]
        chex.assert_trees_all_equal(gathered.input_grids_examples, expected.input_grids_examples)
        
        # Test with JAX array indices
        jax_max_idx = jnp.array(max_idx, dtype=jnp.int32)
        gathered_jax = gather_task(buffer, jax_max_idx)
        chex.assert_trees_all_equal(gathered_jax.input_grids_examples, expected.input_grids_examples)

    def test_buffer_memory_efficiency(self):
        """Test that buffer operations are memory efficient."""
        # Create tasks with larger arrays to test memory patterns
        tasks = []
        for i in range(5):
            # Larger grids to test memory usage
            large_grid = jnp.full((10, 10), i, dtype=jnp.int32)
            task = JaxArcTask(
                task_id=f"large_task_{i}",
                input_grids_examples=jnp.array([large_grid]),
                output_grids_examples=jnp.array([large_grid + 1]),
                input_grids_test=jnp.array([large_grid + 2]),
                output_grids_test=jnp.array([large_grid + 3]),
                num_examples=1,
                num_test=1,
                max_grid_height=10,
                max_grid_width=10,
            )
            tasks.append(task)
        
        buffer = stack_task_list(tasks)
        
        # Verify buffer has expected shape
        chex.assert_shape(buffer.input_grids_examples, (5, 1, 10, 10))
        
        # Test that gathering doesn't create unnecessary copies
        gathered = gather_task(buffer, 2)
        chex.assert_shape(gathered.input_grids_examples, (1, 10, 10))
        
        # Verify values are correct
        expected_value = 2  # Task 2 should have grids filled with 2
        assert jnp.all(gathered.input_grids_examples == expected_value)