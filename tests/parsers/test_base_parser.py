"""Tests for the ArcDataParserBase abstract base class."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import chex
import jax
import pytest
from omegaconf import DictConfig

from jaxarc.parsers import ArcDataParserBase
from jaxarc.types import ParsedTaskData


class ConcreteParser(ArcDataParserBase):
    """Concrete implementation of ArcDataParserBase for testing."""

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)
        self.load_task_file_mock = MagicMock()
        self.preprocess_task_data_mock = MagicMock()
        self.get_random_task_mock = MagicMock()

    def load_task_file(self, task_file_path: str) -> Any:
        return self.load_task_file_mock(task_file_path)

    def preprocess_task_data(
        self, raw_task_data: Any, key: chex.PRNGKey
    ) -> ParsedTaskData:
        return self.preprocess_task_data_mock(raw_task_data, key)

    def get_random_task(self, key: chex.PRNGKey) -> ParsedTaskData:
        return self.get_random_task_mock(key)


class TestArcDataParserBase:
    """Test suite for ArcDataParserBase abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that the abstract base class cannot be instantiated directly."""
        cfg = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ArcDataParserBase(cfg)  # type: ignore[abstract]

    def test_concrete_implementation_initialization(self):
        """Test that concrete implementations can be instantiated properly."""
        cfg = DictConfig(
            {
                "dataset_path": "/some/path",
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        parser = ConcreteParser(cfg)

        assert parser.cfg == cfg
        assert parser.max_grid_height == 30
        assert parser.max_grid_width == 30
        assert parser.max_train_pairs == 5
        assert parser.max_test_pairs == 5

    def test_initialization_validation_grid_dimensions(self):
        """Test that initialization validates grid dimensions."""
        # Test negative height
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": -1,
                    "max_grid_width": 30,
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

        # Test zero height
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 0,
                    "max_grid_width": 30,
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

        # Test negative width
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": -1,
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

        # Test zero width
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 0,
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

    def test_initialization_validation_pair_counts(self):
        """Test that initialization validates pair count limits."""
        # Test negative train pairs
        with pytest.raises(ValueError, match="Max pairs must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "max_train_pairs": -1,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

        # Test zero train pairs
        with pytest.raises(ValueError, match="Max pairs must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "max_train_pairs": 0,
                    "max_test_pairs": 5,
                }
            )
            ConcreteParser(cfg)

        # Test negative test pairs
        with pytest.raises(ValueError, match="Max pairs must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "max_train_pairs": 5,
                    "max_test_pairs": -1,
                }
            )
            ConcreteParser(cfg)

        # Test zero test pairs
        with pytest.raises(ValueError, match="Max pairs must be positive"):
            cfg = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "max_train_pairs": 5,
                    "max_test_pairs": 0,
                }
            )
            ConcreteParser(cfg)

    def test_get_max_dimensions(self):
        """Test the get_max_dimensions method."""
        cfg = DictConfig(
            {
                "max_grid_height": 25,
                "max_grid_width": 20,
                "max_train_pairs": 3,
                "max_test_pairs": 2,
            }
        )
        parser = ConcreteParser(cfg)
        dims = parser.get_max_dimensions()
        assert dims == (25, 20, 3, 2)

    def test_validate_grid_dimensions(self):
        """Test the validate_grid_dimensions method."""
        cfg = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        parser = ConcreteParser(cfg)

        # Should not raise for valid dimensions
        parser.validate_grid_dimensions(30, 30)
        parser.validate_grid_dimensions(1, 1)
        parser.validate_grid_dimensions(15, 20)

        # Should raise for dimensions exceeding max
        with pytest.raises(ValueError, match="Grid dimensions .* exceed maximum"):
            parser.validate_grid_dimensions(31, 30)

        with pytest.raises(ValueError, match="Grid dimensions .* exceed maximum"):
            parser.validate_grid_dimensions(30, 31)

        with pytest.raises(ValueError, match="Grid dimensions .* exceed maximum"):
            parser.validate_grid_dimensions(31, 31)

    def test_missing_required_configuration(self):
        """Test that KeyError is raised when required config fields are missing."""
        # Missing max_grid_height
        with pytest.raises(KeyError, match="Missing required configuration field"):
            cfg = DictConfig(
                {"max_grid_width": 30, "max_train_pairs": 5, "max_test_pairs": 5}
            )
            ConcreteParser(cfg)

        # Missing max_grid_width
        with pytest.raises(KeyError, match="Missing required configuration field"):
            cfg = DictConfig(
                {"max_grid_height": 30, "max_train_pairs": 5, "max_test_pairs": 5}
            )
            ConcreteParser(cfg)

        # Missing max_train_pairs
        with pytest.raises(KeyError, match="Missing required configuration field"):
            cfg = DictConfig(
                {"max_grid_height": 30, "max_grid_width": 30, "max_test_pairs": 5}
            )
            ConcreteParser(cfg)

        # Missing max_test_pairs
        with pytest.raises(KeyError, match="Missing required configuration field"):
            cfg = DictConfig(
                {"max_grid_height": 30, "max_grid_width": 30, "max_train_pairs": 5}
            )
            ConcreteParser(cfg)

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteParser(ArcDataParserBase):
            """Incomplete implementation of ArcDataParserBase."""

            # No implementation of abstract methods

        cfg = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteParser(cfg)  # type: ignore[abstract]

    def test_load_task_file_is_called(self):
        """Test that load_task_file is called with the correct path."""
        cfg = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        parser = ConcreteParser(cfg)
        parser.load_task_file_mock.return_value = {"some": "data"}

        result = parser.load_task_file("/path/to/task.json")
        parser.load_task_file_mock.assert_called_once_with("/path/to/task.json")
        assert result == {"some": "data"}

    def test_preprocess_task_data_is_called(self):
        """Test that preprocess_task_data is called with the correct arguments."""
        cfg = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        parser = ConcreteParser(cfg)
        raw_data = {"some": "raw_data"}
        key = jax.random.PRNGKey(0)

        # Create a mock return value that matches the ParsedTaskData interface
        mock_return = MagicMock(spec=ParsedTaskData)
        parser.preprocess_task_data_mock.return_value = mock_return

        result = parser.preprocess_task_data(raw_data, key)
        parser.preprocess_task_data_mock.assert_called_once()
        assert parser.preprocess_task_data_mock.call_args[0][0] == raw_data
        assert (parser.preprocess_task_data_mock.call_args[0][1] == key).all()
        assert result == mock_return

    def test_get_random_task_is_called(self):
        """Test that get_random_task is called with the correct key."""
        cfg = DictConfig(
            {
                "max_grid_height": 25,
                "max_grid_width": 20,
                "max_train_pairs": 3,
                "max_test_pairs": 2,
            }
        )
        parser = ConcreteParser(cfg)
        key = jax.random.PRNGKey(42)

        # Create a mock return value that matches the ParsedTaskData interface
        mock_return = MagicMock(spec=ParsedTaskData)
        parser.get_random_task_mock.return_value = mock_return

        result = parser.get_random_task(key)
        parser.get_random_task_mock.assert_called_once()
        assert (parser.get_random_task_mock.call_args[0][0] == key).all()
        assert result == mock_return
