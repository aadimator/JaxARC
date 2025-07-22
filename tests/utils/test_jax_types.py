"""Tests for JAXTyping definitions and validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.utils.jax_types import (
    MAX_GRID_SIZE,
    MAX_TEST_PAIRS,
    MAX_TRAIN_PAIRS,
    NUM_COLORS,
    NUM_OPERATIONS,
    BboxActionData,
    BboxCoords,
    BoundingBox,
    ColIndex,
    ColorHex,
    ColorValue,
    ContinuousSelectionArray,
    DebugInfo,
    EpisodeDone,
    EpisodeIndex,
    GridArray,
    GridHeight,
    GridWidth,
    MaskActionData,
    MaskArray,
    MaxGridSize,
    MaxTestPairs,
    MaxTrainPairs,
    NumColors,
    NumOperations,
    ObservationArray,
    OperationId,
    PaddingValue,
    PointActionData,
    PointCoords,
    PRNGKey,
    RewardValue,
    RGBColor,
    RowIndex,
    SelectionArray,
    SimilarityScore,
    StepCount,
    TaskIndex,
    TaskInputGrids,
    TaskInputMasks,
    TaskOutputGrids,
    TaskOutputMasks,
    ValidationMask,
)


class TestCoreGridTypes:
    """Test core grid type definitions."""

    def test_grid_array_single(self):
        """Test GridArray with single grid."""
        grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        assert grid.shape == (2, 2)
        assert grid.dtype == jnp.int32
        assert jnp.all(grid >= 0) and jnp.all(grid <= 9)  # Valid color range

    def test_grid_array_batched(self):
        """Test GridArray with batched grids."""
        batch: GridArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )
        assert batch.shape == (2, 2, 2)
        assert batch.dtype == jnp.int32

    def test_mask_array_single(self):
        """Test MaskArray with single mask."""
        mask: MaskArray = jnp.array([[True, False], [False, True]], dtype=bool)
        assert mask.shape == (2, 2)
        assert mask.dtype == bool

    def test_mask_array_batched(self):
        """Test MaskArray with batched masks."""
        batch_mask: MaskArray = jnp.array(
            [[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool
        )
        assert batch_mask.shape == (2, 2, 2)
        assert batch_mask.dtype == bool

    def test_selection_array_single(self):
        """Test SelectionArray with single selection."""
        selection: SelectionArray = jnp.array(
            [[True, False], [False, True]], dtype=bool
        )
        assert selection.shape == (2, 2)
        assert selection.dtype == bool

    def test_continuous_selection_array(self):
        """Test ContinuousSelectionArray with float values."""
        selection: ContinuousSelectionArray = jnp.array(
            [[0.8, 0.2], [0.1, 0.9]], dtype=jnp.float32
        )
        assert selection.shape == (2, 2)
        assert selection.dtype == jnp.float32
        assert jnp.all(selection >= 0.0) and jnp.all(selection <= 1.0)


class TestTaskDataTypes:
    """Test task data structure types."""

    def test_task_input_grids(self):
        """Test TaskInputGrids with maximum dimensions."""
        max_pairs, max_height, max_width = 5, 10, 10
        inputs: TaskInputGrids = jnp.zeros(
            (max_pairs, max_height, max_width), dtype=jnp.int32
        )
        assert inputs.shape == (max_pairs, max_height, max_width)
        assert inputs.dtype == jnp.int32

    def test_task_output_grids(self):
        """Test TaskOutputGrids with maximum dimensions."""
        max_pairs, max_height, max_width = 5, 10, 10
        outputs: TaskOutputGrids = jnp.ones(
            (max_pairs, max_height, max_width), dtype=jnp.int32
        )
        assert outputs.shape == (max_pairs, max_height, max_width)
        assert outputs.dtype == jnp.int32

    def test_task_input_masks(self):
        """Test TaskInputMasks with maximum dimensions."""
        max_pairs, max_height, max_width = 5, 10, 10
        masks: TaskInputMasks = jnp.ones((max_pairs, max_height, max_width), dtype=bool)
        assert masks.shape == (max_pairs, max_height, max_width)
        assert masks.dtype == bool

    def test_task_output_masks(self):
        """Test TaskOutputMasks with maximum dimensions."""
        max_pairs, max_height, max_width = 5, 10, 10
        masks: TaskOutputMasks = jnp.zeros(
            (max_pairs, max_height, max_width), dtype=bool
        )
        assert masks.shape == (max_pairs, max_height, max_width)
        assert masks.dtype == bool


class TestActionTypes:
    """Test action-related type definitions."""

    def test_point_coords(self):
        """Test PointCoords type."""
        coords: PointCoords = jnp.array([2, 3], dtype=jnp.int32)
        assert coords.shape == (2,)
        assert coords.dtype == jnp.int32

    def test_bbox_coords(self):
        """Test BboxCoords type."""
        bbox: BboxCoords = jnp.array([1, 2, 5, 6], dtype=jnp.int32)
        assert bbox.shape == (4,)
        assert bbox.dtype == jnp.int32

    def test_operation_id(self):
        """Test OperationId type."""
        op_id: OperationId = jnp.array(15, dtype=jnp.int32)
        assert op_id.shape == ()
        assert op_id.dtype == jnp.int32
        assert 0 <= op_id < NUM_OPERATIONS

    def test_point_action_data(self):
        """Test PointActionData type."""
        action: PointActionData = jnp.array([4, 7], dtype=jnp.int32)
        assert action.shape == (2,)
        assert action.dtype == jnp.int32

    def test_bbox_action_data(self):
        """Test BboxActionData type."""
        action: BboxActionData = jnp.array([1, 1, 3, 3], dtype=jnp.int32)
        assert action.shape == (4,)
        assert action.dtype == jnp.int32

    def test_mask_action_data(self):
        """Test MaskActionData type."""
        height_width = 25  # 5x5 flattened
        action: MaskActionData = jnp.ones(height_width, dtype=jnp.float32)
        assert action.shape == (height_width,)
        assert action.dtype == jnp.float32


class TestSimilarityAndScoringTypes:
    """Test similarity and scoring type definitions."""

    def test_similarity_score_single(self):
        """Test SimilarityScore with single score."""
        score: SimilarityScore = jnp.array(0.85, dtype=jnp.float32)
        assert score.shape == ()
        assert score.dtype == jnp.float32
        assert 0.0 <= score <= 1.0

    def test_similarity_score_batched(self):
        """Test SimilarityScore with batched scores."""
        scores: SimilarityScore = jnp.array([0.8, 0.9, 0.7], dtype=jnp.float32)
        assert scores.shape == (3,)
        assert scores.dtype == jnp.float32

    def test_reward_value_single(self):
        """Test RewardValue with single reward."""
        reward: RewardValue = jnp.array(1.0, dtype=jnp.float32)
        assert reward.shape == ()
        assert reward.dtype == jnp.float32

    def test_reward_value_batched(self):
        """Test RewardValue with batched rewards."""
        rewards: RewardValue = jnp.array([1.0, 0.5, -0.1], dtype=jnp.float32)
        assert rewards.shape == (3,)
        assert rewards.dtype == jnp.float32

    def test_observation_array_single(self):
        """Test ObservationArray with single observation."""
        obs: ObservationArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        assert obs.shape == (2, 2)
        assert obs.dtype == jnp.int32

    def test_observation_array_batched(self):
        """Test ObservationArray with batched observations."""
        batch_obs: ObservationArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )
        assert batch_obs.shape == (2, 2, 2)
        assert batch_obs.dtype == jnp.int32


class TestEnvironmentStateTypes:
    """Test environment state type definitions."""

    def test_step_count(self):
        """Test StepCount type."""
        steps: StepCount = jnp.array(42, dtype=jnp.int32)
        assert steps.shape == ()
        assert steps.dtype == jnp.int32
        assert steps >= 0

    def test_episode_index(self):
        """Test EpisodeIndex type."""
        episode: EpisodeIndex = jnp.array(5, dtype=jnp.int32)
        assert episode.shape == ()
        assert episode.dtype == jnp.int32
        assert episode >= 0

    def test_task_index(self):
        """Test TaskIndex type."""
        task: TaskIndex = jnp.array(123, dtype=jnp.int32)
        assert task.shape == ()
        assert task.dtype == jnp.int32

    def test_episode_done(self):
        """Test EpisodeDone type."""
        done: EpisodeDone = jnp.array(True, dtype=bool)
        assert done.shape == ()
        assert done.dtype == bool


class TestUtilityTypes:
    """Test utility type definitions."""

    def test_color_value(self):
        """Test ColorValue type."""
        color: ColorValue = jnp.array(7, dtype=jnp.int32)
        assert color.shape == ()
        assert color.dtype == jnp.int32
        assert 0 <= color < NUM_COLORS

    def test_padding_value_array(self):
        """Test PaddingValue as JAX array."""
        padding: PaddingValue = jnp.array(0, dtype=jnp.int32)
        assert padding.shape == ()
        assert padding.dtype == jnp.int32

    def test_padding_value_int(self):
        """Test PaddingValue as Python int."""
        padding: PaddingValue = 0
        assert isinstance(padding, int)

    def test_prng_key(self):
        """Test PRNGKey type."""
        key: PRNGKey = jax.random.PRNGKey(42)
        assert key.shape == (2,)
        assert key.dtype == jnp.uint32

    def test_grid_dimensions(self):
        """Test GridHeight and GridWidth types."""
        height: GridHeight = jnp.array(10, dtype=jnp.int32)
        width: GridWidth = jnp.array(15, dtype=jnp.int32)

        assert height.shape == ()
        assert width.shape == ()
        assert height.dtype == jnp.int32
        assert width.dtype == jnp.int32
        assert height > 0 and width > 0

    def test_coordinate_types(self):
        """Test RowIndex and ColIndex types."""
        row: RowIndex = jnp.array(3, dtype=jnp.int32)
        col: ColIndex = jnp.array(7, dtype=jnp.int32)

        assert row.shape == ()
        assert col.shape == ()
        assert row.dtype == jnp.int32
        assert col.dtype == jnp.int32
        assert row >= 0 and col >= 0

    def test_bounding_box(self):
        """Test BoundingBox type."""
        min_row: RowIndex = jnp.array(1, dtype=jnp.int32)
        max_row: RowIndex = jnp.array(5, dtype=jnp.int32)
        min_col: ColIndex = jnp.array(2, dtype=jnp.int32)
        max_col: ColIndex = jnp.array(8, dtype=jnp.int32)

        bbox: BoundingBox = (min_row, max_row, min_col, max_col)
        assert len(bbox) == 4
        assert min_row <= max_row
        assert min_col <= max_col


class TestValidationTypes:
    """Test validation and debug type definitions."""

    def test_validation_mask(self):
        """Test ValidationMask type."""
        mask: ValidationMask = jnp.array([[True, False], [False, True]], dtype=bool)
        assert mask.shape == (2, 2)
        assert mask.dtype == bool

    def test_debug_info(self):
        """Test DebugInfo flexible type."""
        # Test with different shapes
        debug1: DebugInfo = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        debug2: DebugInfo = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        debug3: DebugInfo = jnp.array(42.0, dtype=jnp.float32)

        assert debug1.dtype == jnp.float32
        assert debug2.dtype == jnp.float32
        assert debug3.dtype == jnp.float32


class TestVisualizationTypes:
    """Test visualization type definitions."""

    def test_color_hex(self):
        """Test ColorHex type."""
        color: ColorHex = "#FF0000"
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7

    def test_rgb_color(self):
        """Test RGBColor type."""
        color: RGBColor = (255, 0, 0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(isinstance(c, int) for c in color)
        assert all(0 <= c <= 255 for c in color)


class TestConstants:
    """Test constant definitions."""

    def test_max_grid_size(self):
        """Test MAX_GRID_SIZE constant."""
        assert MAX_GRID_SIZE == 30
        assert isinstance(MAX_GRID_SIZE, int)

    def test_max_pairs_constants(self):
        """Test MAX_TRAIN_PAIRS and MAX_TEST_PAIRS constants."""
        assert MAX_TRAIN_PAIRS == 10
        assert MAX_TEST_PAIRS == 3
        assert isinstance(MAX_TRAIN_PAIRS, int)
        assert isinstance(MAX_TEST_PAIRS, int)

    def test_num_colors(self):
        """Test NUM_COLORS constant."""
        assert NUM_COLORS == 10
        assert isinstance(NUM_COLORS, int)

    def test_num_operations(self):
        """Test NUM_OPERATIONS constant."""
        assert NUM_OPERATIONS == 35
        assert isinstance(NUM_OPERATIONS, int)

    def test_type_aliases_for_constants(self):
        """Test type aliases for constants."""
        max_size: MaxGridSize = 30
        max_train: MaxTrainPairs = 10
        max_test: MaxTestPairs = 3
        num_colors: NumColors = 10
        num_ops: NumOperations = 35

        assert isinstance(max_size, int)
        assert isinstance(max_train, int)
        assert isinstance(max_test, int)
        assert isinstance(num_colors, int)
        assert isinstance(num_ops, int)


class TestJAXCompatibility:
    """Test JAX compatibility of type definitions."""

    def test_jit_compilation_with_types(self):
        """Test that types work with JAX JIT compilation."""

        @jax.jit
        def process_grid(grid: GridArray) -> GridArray:
            return grid + 1

        grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        result = process_grid(grid)
        expected = jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)

    def test_vmap_with_types(self):
        """Test that types work with JAX vmap."""

        def process_single_grid(grid: GridArray) -> SimilarityScore:
            return jnp.mean(grid.astype(jnp.float32))

        # Create batched grids
        batch_grids: GridArray = jnp.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.int32
        )

        # Apply vmap
        vmapped_fn = jax.vmap(process_single_grid)
        scores: SimilarityScore = vmapped_fn(batch_grids)

        assert scores.shape == (2,)
        assert scores.dtype == jnp.float32

    def test_grad_with_types(self):
        """Test that types work with JAX grad."""

        def loss_fn(weights: ContinuousSelectionArray) -> RewardValue:
            return jnp.sum(weights**2)

        grad_fn = jax.grad(loss_fn)
        weights: ContinuousSelectionArray = jnp.array(
            [[0.5, 0.3], [0.8, 0.2]], dtype=jnp.float32
        )
        gradients = grad_fn(weights)

        assert gradients.shape == weights.shape
        assert gradients.dtype == jnp.float32

    def test_tree_operations_with_types(self):
        """Test that types work with JAX tree operations."""
        # Create a simple tree structure
        tree = {
            "grid": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
            "mask": jnp.array([[True, False], [False, True]], dtype=bool),
            "reward": jnp.array(1.0, dtype=jnp.float32),
        }

        # Test tree_map
        def add_one_to_arrays(x):
            if x.dtype in [jnp.int32, jnp.float32]:
                return x + 1
            return x

        result = jax.tree.map(add_one_to_arrays, tree)

        assert jnp.array_equal(
            result["grid"], jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
        )
        assert jnp.array_equal(result["mask"], tree["mask"])  # Boolean unchanged
        assert result["reward"] == 2.0


class TestTypeValidation:
    """Test runtime type validation capabilities."""

    def test_shape_validation(self):
        """Test that shapes match expected patterns."""
        # Valid shapes
        grid_2d: GridArray = jnp.zeros((5, 5), dtype=jnp.int32)
        grid_3d: GridArray = jnp.zeros((2, 5, 5), dtype=jnp.int32)  # Batched

        assert grid_2d.ndim == 2
        assert grid_3d.ndim == 3

    def test_dtype_validation(self):
        """Test that dtypes match expected types."""
        grid: GridArray = jnp.zeros((5, 5), dtype=jnp.int32)
        mask: MaskArray = jnp.zeros((5, 5), dtype=bool)
        selection: ContinuousSelectionArray = jnp.zeros((5, 5), dtype=jnp.float32)

        assert grid.dtype == jnp.int32
        assert mask.dtype == bool
        assert selection.dtype == jnp.float32

    def test_value_range_validation(self):
        """Test that values are in expected ranges."""
        # Color values should be 0-9
        colors: GridArray = jnp.array([[0, 5, 9], [2, 7, 1]], dtype=jnp.int32)
        assert jnp.all(colors >= 0) and jnp.all(colors < NUM_COLORS)

        # Continuous selections should be 0-1
        selection: ContinuousSelectionArray = jnp.array(
            [[0.0, 0.5, 1.0], [0.2, 0.8, 0.3]], dtype=jnp.float32
        )
        assert jnp.all(selection >= 0.0) and jnp.all(selection <= 1.0)

        # Operation IDs should be 0-34
        op_id: OperationId = jnp.array(20, dtype=jnp.int32)
        assert 0 <= op_id < NUM_OPERATIONS
