"""Comprehensive tests for parser refactoring and inheritance behavior.

This test suite specifically focuses on:
- Base class method implementations that were moved from specific parsers
- Inheritance behavior in specific parsers (calling super() methods)
- Ensuring no regression in data loading capabilities
- Testing the DRY (Don't Repeat Yourself) principle implementation

Requirements covered: 3.1, 3.2
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.parsers.base_parser import ArcDataParserBase
from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.parsers.mini_arc import MiniArcParser
from jaxarc.types import JaxArcTask


class TestBaseParserMethodImplementations:
    """Test suite for base class method implementations."""

    @pytest.fixture
    def base_parser_config(self):
        """Configuration for base parser testing."""
        return DictConfig(
            {
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

    @pytest.fixture
    def sample_task_content(self):
        """Sample task content for testing base methods."""
        return {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]},
                {"input": [[4, 5]], "output": [[5, 4]]},
            ],
            "test": [
                {"input": [[6, 7]], "output": [[7, 6]]},
                {"input": [[8, 9]]},  # No output (typical for test)
            ],
        }

    @pytest.fixture
    def concrete_parser(self, base_parser_config):
        """Create a concrete parser for testing base methods."""

        class TestableParser(ArcDataParserBase):
            """Concrete implementation for testing base methods."""

            def load_task_file(self, task_file_path: str):
                return {"mock": "data"}

            def preprocess_task_data(self, raw_task_data, key):
                return MagicMock(spec=JaxArcTask)

            def get_random_task(self, key):
                return MagicMock(spec=JaxArcTask)

        return TestableParser(base_parser_config)

    def test_process_training_pairs_base_implementation(
        self, concrete_parser, sample_task_content
    ):
        """Test _process_training_pairs method in base class."""
        # Mock the utility functions
        with patch("jaxarc.parsers.base_parser.convert_grid_to_jax") as mock_convert:
            # Set up mock to return JAX arrays
            mock_convert.side_effect = lambda x: jnp.array(x, dtype=jnp.int32)

            train_inputs, train_outputs = concrete_parser._process_training_pairs(
                sample_task_content
            )

            # Should have processed both training pairs
            assert len(train_inputs) == 2
            assert len(train_outputs) == 2

            # Should have called convert_grid_to_jax for each input/output
            assert mock_convert.call_count == 4

    def test_process_training_pairs_validation_errors(
        self, concrete_parser, sample_task_content
    ):
        """Test _process_training_pairs validation error handling."""
        # Test empty training pairs
        empty_task = {"train": [], "test": [{"input": [[1]]}]}
        with pytest.raises(ValueError, match="at least one training pair"):
            concrete_parser._process_training_pairs(empty_task)

        # Test missing input
        invalid_task = {
            "train": [{"output": [[1, 2]]}],  # Missing input
            "test": [{"input": [[1]]}],
        }
        with pytest.raises(ValueError, match="missing input or output"):
            concrete_parser._process_training_pairs(invalid_task)

    def test_process_test_pairs_base_implementation(
        self, concrete_parser, sample_task_content
    ):
        """Test _process_test_pairs method in base class."""
        with patch("jaxarc.parsers.base_parser.convert_grid_to_jax") as mock_convert:
            # Set up mock to return JAX arrays
            mock_convert.side_effect = lambda x: jnp.array(x, dtype=jnp.int32)

            test_inputs, test_outputs = concrete_parser._process_test_pairs(
                sample_task_content
            )

            # Should have processed both test pairs
            assert len(test_inputs) == 2
            assert len(test_outputs) == 2

            # Should have called convert_grid_to_jax for inputs and available outputs
            assert mock_convert.call_count >= 2  # At least inputs

    def test_validate_grid_colors_base_implementation(self, concrete_parser):
        """Test _validate_grid_colors method in base class."""
        # Test valid grid
        valid_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        concrete_parser._validate_grid_colors(valid_grid)  # Should not raise

        # Test invalid grid (color out of range)
        invalid_grid = jnp.array([[0, 1, 15]], dtype=jnp.int32)  # 15 > max_colors
        with pytest.raises(ValueError, match="Invalid color in grid"):
            concrete_parser._validate_grid_colors(invalid_grid)


class TestParserInheritanceBehavior:
    """Test suite for inheritance behavior in specific parsers."""

    def test_concept_arc_parser_calls_super_methods(self):
        """Test that ConceptArcParser calls super() methods from base class."""
        config = DictConfig(
            {
                "corpus": {"path": "/mock/path", "concept_groups": []},
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 4,
                "max_test_pairs": 3,
            }
        )

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                parser = ConceptArcParser(config)

            # Test that preprocessing calls base class methods
            sample_task = {
                "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                "test": [{"input": [[3, 4]]}],
            }

            with patch.object(
                ArcDataParserBase, "_process_training_pairs"
            ) as mock_train:
                with patch.object(ArcDataParserBase, "_process_test_pairs") as mock_test:
                    with patch.object(
                        ArcDataParserBase, "_pad_and_create_masks"
                    ) as mock_pad:
                        with patch.object(
                            ArcDataParserBase, "_log_parsing_stats"
                        ) as mock_log:
                            # Set up return values
                            mock_train.return_value = (
                                [jnp.array([[1, 2]])],
                                [jnp.array([[2, 1]])],
                            )
                            mock_test.return_value = (
                                [jnp.array([[3, 4]])],
                                [jnp.array([[0, 0]])],
                            )
                            mock_pad.return_value = {
                                "train_inputs": jnp.zeros((4, 30, 30)),
                                "train_input_masks": jnp.zeros((4, 30, 30), dtype=bool),
                                "train_outputs": jnp.zeros((4, 30, 30)),
                                "train_output_masks": jnp.zeros((4, 30, 30), dtype=bool),
                                "test_inputs": jnp.zeros((3, 30, 30)),
                                "test_input_masks": jnp.zeros((3, 30, 30), dtype=bool),
                                "test_outputs": jnp.zeros((3, 30, 30)),
                                "test_output_masks": jnp.zeros((3, 30, 30), dtype=bool),
                            }

                            key = jax.random.PRNGKey(0)
                            result = parser.preprocess_task_data(
                                ("test_task", sample_task), key
                            )

                            # Verify that base class methods were called
                            mock_train.assert_called_once()
                            mock_test.assert_called_once()
                            mock_pad.assert_called_once()
                            mock_log.assert_called_once()

                            # Result should be a JaxArcTask
                            assert isinstance(result, JaxArcTask)

    def test_parser_specific_error_message_customization(self):
        """Test that parsers customize error messages while calling super()."""
        # Test ConceptArcParser error message customization
        concept_config = DictConfig(
            {
                "corpus": {"path": "/mock/path", "concept_groups": []},
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 4,
                "max_test_pairs": 3,
            }
        )

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                concept_parser = ConceptArcParser(concept_config)

            # Test ConceptARC-specific error message
            empty_task = {"train": [], "test": [{"input": [[1]]}]}
            with pytest.raises(ValueError, match="ConceptARC task must have"):
                concept_parser._process_training_pairs(empty_task)

        # Test MiniArcParser error message customization
        mini_config = DictConfig(
            {
                "tasks": {"path": "/mock/path"},
                "grid": {
                    "max_grid_height": 5,
                    "max_grid_width": 5,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 3,
                "max_test_pairs": 1,
            }
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                mini_parser = MiniArcParser(mini_config)

            # Test MiniARC-specific error message
            empty_task = {"train": [], "test": [{"input": [[1]]}]}
            with pytest.raises(ValueError, match="MiniARC task must have"):
                mini_parser._process_training_pairs(empty_task)


class TestParserFunctionalityRegression:
    """Test suite to ensure no regression in data loading capabilities."""

    @pytest.fixture
    def comprehensive_task_data(self):
        """Comprehensive task data for regression testing."""
        return {
            "train": [
                {
                    "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                    "output": [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
                },
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]],
                },
            ],
            "test": [
                {
                    "input": [[2, 3, 4], [5, 6, 7]],
                    "output": [[7, 6, 5], [4, 3, 2]],
                },
            ],
        }

    @pytest.fixture
    def temp_comprehensive_directory(self, comprehensive_task_data):
        """Create temporary directory with comprehensive test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir)

            # Create multiple task files with different characteristics
            task_files = {
                "simple_task.json": {
                    "train": [{"input": [[1]], "output": [[0]]}],
                    "test": [{"input": [[2]]}],
                },
                "complex_task.json": comprehensive_task_data,
            }

            for filename, task_data in task_files.items():
                task_file = tasks_dir / filename
                with task_file.open("w") as f:
                    json.dump(task_data, f)

            yield temp_dir

    def test_arc_agi_parser_data_loading_regression(
        self, temp_comprehensive_directory
    ):
        """Test that ArcAgiParser data loading hasn't regressed."""
        config = DictConfig(
            {
                "default_split": "training",
                "training": {"path": str(temp_comprehensive_directory)},
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path(config.training.path)

            parser = ArcAgiParser(config)

            # Test basic functionality
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 2
            assert "simple_task" in task_ids
            assert "complex_task" in task_ids

            # Test getting specific tasks
            for task_id in task_ids:
                task = parser.get_task_by_id(task_id)
                assert isinstance(task, JaxArcTask)
                assert task.num_train_pairs > 0
                assert task.num_test_pairs > 0

                # Test array shapes and types
                assert task.input_grids_examples.shape == (5, 30, 30)
                assert task.output_grids_examples.shape == (5, 30, 30)
                assert task.test_input_grids.shape == (3, 30, 30)
                assert task.input_grids_examples.dtype == jnp.int32
                assert task.input_masks_examples.dtype == jnp.bool_

            # Test random task selection
            key = jax.random.PRNGKey(42)
            random_task = parser.get_random_task(key)
            assert isinstance(random_task, JaxArcTask)

    def test_concept_arc_parser_data_loading_regression(
        self, temp_comprehensive_directory
    ):
        """Test that ConceptArcParser data loading hasn't regressed."""
        # Create concept group structure
        corpus_dir = Path(temp_comprehensive_directory) / "corpus"
        concept_dir = corpus_dir / "TestConcept"
        concept_dir.mkdir(parents=True)

        # Move files to concept directory
        for file in Path(temp_comprehensive_directory).glob("*.json"):
            new_path = concept_dir / file.name
            file.rename(new_path)

        config = DictConfig(
            {
                "corpus": {
                    "path": str(corpus_dir),
                    "concept_groups": ["TestConcept"],
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 4,
                "max_test_pairs": 3,
            }
        )

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = corpus_dir

            parser = ConceptArcParser(config)

            # Test concept-specific functionality
            concept_groups = parser.get_concept_groups()
            assert "TestConcept" in concept_groups

            tasks_in_concept = parser.get_tasks_in_concept("TestConcept")
            assert len(tasks_in_concept) == 2

            # Test getting tasks from concept
            key = jax.random.PRNGKey(42)
            concept_task = parser.get_random_task_from_concept("TestConcept", key)
            assert isinstance(concept_task, JaxArcTask)

    def test_mini_arc_parser_data_loading_regression(self, temp_comprehensive_directory):
        """Test that MiniArcParser data loading hasn't regressed."""
        # Create only 5x5 compatible tasks for MiniARC
        mini_tasks_dir = Path(temp_comprehensive_directory) / "mini"
        mini_tasks_dir.mkdir()

        mini_task_data = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[2]], "output": [[3]]},
            ],
            "test": [{"input": [[4, 5]], "output": [[5, 4]]}],
        }

        for i in range(2):
            task_file = mini_tasks_dir / f"mini_task_{i}.json"
            with task_file.open("w") as f:
                json.dump(mini_task_data, f)

        config = DictConfig(
            {
                "tasks": {"path": str(mini_tasks_dir)},
                "grid": {
                    "max_grid_height": 5,
                    "max_grid_width": 5,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 3,
                "max_test_pairs": 1,
            }
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = mini_tasks_dir

            parser = MiniArcParser(config)

            # Test MiniARC-specific functionality
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 2

            # Test optimized dimensions
            for task_id in task_ids:
                task = parser.get_task_by_id(task_id)
                assert isinstance(task, JaxArcTask)

                # Should use 5x5 dimensions, not 30x30
                assert task.input_grids_examples.shape == (3, 5, 5)
                assert task.output_grids_examples.shape == (3, 5, 5)
                assert task.test_input_grids.shape == (1, 5, 5)


class TestParserMethodDuplicationElimination:
    """Test suite to verify that method duplication has been eliminated."""

    def test_no_duplicate_process_training_pairs_implementations(self):
        """Test that _process_training_pairs is only implemented in base class."""
        # ConceptArcParser should call super() but may customize error messages
        concept_method = getattr(ConceptArcParser, "_process_training_pairs", None)
        if concept_method:
            # If it exists, it should be a wrapper that calls super()
            import inspect

            source = inspect.getsource(concept_method)
            assert "super()" in source, "ConceptArcParser should call super()"

        # MiniArcParser should call super() but may customize error messages
        mini_method = getattr(MiniArcParser, "_process_training_pairs", None)
        if mini_method:
            # If it exists, it should be a wrapper that calls super()
            import inspect

            source = inspect.getsource(mini_method)
            assert "super()" in source, "MiniArcParser should call super()"

    def test_no_duplicate_process_test_pairs_implementations(self):
        """Test that _process_test_pairs is only implemented in base class."""
        # ConceptArcParser and MiniArcParser may customize but should call super()
        for parser_class in [ConceptArcParser, MiniArcParser]:
            method = getattr(parser_class, "_process_test_pairs", None)
            if method:
                import inspect

                source = inspect.getsource(method)
                assert (
                    "super()" in source
                ), f"{parser_class.__name__} should call super()"

    def test_no_duplicate_pad_and_create_masks_implementations(self):
        """Test that _pad_and_create_masks is only implemented in base class."""
        # This method should only exist in the base class
        for parser_class in [ArcAgiParser, ConceptArcParser, MiniArcParser]:
            assert not hasattr(parser_class, "_pad_and_create_masks") or (
                getattr(parser_class, "_pad_and_create_masks")
                is ArcDataParserBase._pad_and_create_masks
            ), f"{parser_class.__name__} should not override _pad_and_create_masks"

    def test_no_duplicate_validate_grid_colors_implementations(self):
        """Test that _validate_grid_colors is only implemented in base class."""
        # This method should only exist in the base class
        for parser_class in [ArcAgiParser, ConceptArcParser, MiniArcParser]:
            assert not hasattr(parser_class, "_validate_grid_colors") or (
                getattr(parser_class, "_validate_grid_colors")
                is ArcDataParserBase._validate_grid_colors
            ), f"{parser_class.__name__} should not override _validate_grid_colors"

    def test_base_class_methods_exist(self):
        """Test that all expected methods exist in the base class."""
        base_methods = [
            "_process_training_pairs",
            "_process_test_pairs",
            "_pad_and_create_masks",
            "_validate_grid_colors",
            "_log_parsing_stats",
        ]

        for method_name in base_methods:
            assert hasattr(
                ArcDataParserBase, method_name
            ), f"Base class should have {method_name}"
            method = getattr(ArcDataParserBase, method_name)
            assert callable(method), f"{method_name} should be callable"

    def test_inheritance_chain_integrity(self):
        """Test that inheritance chain is properly maintained."""
        # All parsers should inherit from ArcDataParserBase
        assert issubclass(ArcAgiParser, ArcDataParserBase)
        assert issubclass(ConceptArcParser, ArcDataParserBase)
        assert issubclass(MiniArcParser, ArcDataParserBase)

        # Test that parsers can access base class methods
        config = DictConfig(
            {
                "grid": {
                    "max_grid_height": 10,
                    "max_grid_width": 10,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 2,
                "max_test_pairs": 1,
            }
        )

        # Create mock parsers to test method access
        with patch("jaxarc.parsers.concept_arc.here"):
            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                concept_parser = ConceptArcParser(
                    DictConfig({**config, "corpus": {"path": "/mock"}})
                )

        with patch("jaxarc.parsers.mini_arc.here"):
            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                mini_parser = MiniArcParser(
                    DictConfig({**config, "tasks": {"path": "/mock"}})
                )

        # Test that parsers have access to base class methods
        for parser in [concept_parser, mini_parser]:
            assert hasattr(parser, "_process_training_pairs")
            assert hasattr(parser, "_process_test_pairs")
            assert hasattr(parser, "_pad_and_create_masks")
            assert hasattr(parser, "_validate_grid_colors")
            assert hasattr(parser, "_log_parsing_stats")

    def test_method_resolution_order(self):
        """Test that method resolution order is correct for inheritance."""
        # Test ConceptArcParser MRO
        concept_mro = ConceptArcParser.__mro__
        assert ArcDataParserBase in concept_mro
        assert concept_mro.index(ConceptArcParser) < concept_mro.index(ArcDataParserBase)

        # Test MiniArcParser MRO
        mini_mro = MiniArcParser.__mro__
        assert ArcDataParserBase in mini_mro
        assert mini_mro.index(MiniArcParser) < mini_mro.index(ArcDataParserBase)

        # Test ArcAgiParser MRO
        arc_agi_mro = ArcAgiParser.__mro__
        assert ArcDataParserBase in arc_agi_mro
        assert arc_agi_mro.index(ArcAgiParser) < arc_agi_mro.index(ArcDataParserBase)