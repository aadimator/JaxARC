"""Simple verification tests for parser refactoring.

This test suite verifies that the refactoring maintains functionality
and that base class methods are properly implemented and used.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.parsers.base_parser import ArcDataParserBase
from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.parsers.mini_arc import MiniArcParser


def test_base_parser_has_refactored_methods():
    """Test that base parser has the methods that were moved from specific parsers."""
    expected_methods = [
        "_process_training_pairs",
        "_process_test_pairs", 
        "_pad_and_create_masks",
        "_validate_grid_colors",
        "_log_parsing_stats",
    ]
    
    for method_name in expected_methods:
        assert hasattr(ArcDataParserBase, method_name), f"Base class should have {method_name}"
        method = getattr(ArcDataParserBase, method_name)
        assert callable(method), f"{method_name} should be callable"


def test_specific_parsers_inherit_from_base():
    """Test that specific parsers properly inherit from base class."""
    parsers = [ArcAgiParser, ConceptArcParser, MiniArcParser]
    
    for parser_class in parsers:
        assert issubclass(parser_class, ArcDataParserBase), f"{parser_class.__name__} should inherit from ArcDataParserBase"


def test_base_parser_process_training_pairs():
    """Test that base parser _process_training_pairs method works correctly."""
    
    class TestParser(ArcDataParserBase):
        def load_task_file(self, task_file_path: str):
            return {}
        
        def preprocess_task_data(self, raw_task_data, key):
            return None
            
        def get_random_task(self, key):
            return None
    
    config = DictConfig({
        "grid": {
            "max_grid_height": 10,
            "max_grid_width": 10,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 3,
        "max_test_pairs": 2,
    })
    
    parser = TestParser(config)
    
    # Test with valid task content
    task_content = {
        "train": [
            {"input": [[1, 2]], "output": [[2, 1]]},
            {"input": [[3]], "output": [[4]]},
        ]
    }
    
    with patch("jaxarc.parsers.utils.convert_grid_to_jax") as mock_convert:
        mock_convert.side_effect = lambda x: jnp.array(x, dtype=jnp.int32)
        
        train_inputs, train_outputs = parser._process_training_pairs(task_content)
        
        assert len(train_inputs) == 2
        assert len(train_outputs) == 2
        assert mock_convert.call_count == 4  # 2 inputs + 2 outputs


def test_base_parser_process_test_pairs():
    """Test that base parser _process_test_pairs method works correctly."""
    
    class TestParser(ArcDataParserBase):
        def load_task_file(self, task_file_path: str):
            return {}
        
        def preprocess_task_data(self, raw_task_data, key):
            return None
            
        def get_random_task(self, key):
            return None
    
    config = DictConfig({
        "grid": {
            "max_grid_height": 10,
            "max_grid_width": 10,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 3,
        "max_test_pairs": 2,
    })
    
    parser = TestParser(config)
    
    # Test with valid task content
    task_content = {
        "test": [
            {"input": [[1, 2]], "output": [[2, 1]]},
            {"input": [[3]]},  # No output
        ]
    }
    
    with patch("jaxarc.parsers.utils.convert_grid_to_jax") as mock_convert:
        mock_convert.side_effect = lambda x: jnp.array(x, dtype=jnp.int32)
        
        test_inputs, test_outputs = parser._process_test_pairs(task_content)
        
        assert len(test_inputs) == 2
        assert len(test_outputs) == 2  # Should create dummy output for missing one


def test_base_parser_validate_grid_colors():
    """Test that base parser _validate_grid_colors method works correctly."""
    
    class TestParser(ArcDataParserBase):
        def load_task_file(self, task_file_path: str):
            return {}
        
        def preprocess_task_data(self, raw_task_data, key):
            return None
            
        def get_random_task(self, key):
            return None
    
    config = DictConfig({
        "grid": {
            "max_grid_height": 10,
            "max_grid_width": 10,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 3,
        "max_test_pairs": 2,
    })
    
    parser = TestParser(config)
    
    # Test valid grid
    valid_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    parser._validate_grid_colors(valid_grid)  # Should not raise
    
    # Test invalid grid
    invalid_grid = jnp.array([[0, 1, 15]], dtype=jnp.int32)  # 15 > max_colors
    with pytest.raises(ValueError, match="Invalid color in grid"):
        parser._validate_grid_colors(invalid_grid)


def test_concept_arc_parser_calls_super():
    """Test that ConceptArcParser calls super() methods."""
    config = DictConfig({
        "corpus": {"path": "/mock/path", "concept_groups": []},
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
    })
    
    with patch("jaxarc.parsers.concept_arc.here"):
        with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
            parser = ConceptArcParser(config)
    
    # Test that ConceptARC-specific error message is used
    empty_task = {"train": [], "test": [{"input": [[1]]}]}
    with pytest.raises(ValueError, match="ConceptARC task must have"):
        parser._process_training_pairs(empty_task)


def test_mini_arc_parser_calls_super():
    """Test that MiniArcParser calls super() methods."""
    config = DictConfig({
        "tasks": {"path": "/mock/path"},
        "grid": {
            "max_grid_height": 5,
            "max_grid_width": 5,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        },
        "max_train_pairs": 2,
        "max_test_pairs": 1,
    })
    
    with patch("jaxarc.parsers.mini_arc.here"):
        with patch.object(MiniArcParser, "_load_and_cache_tasks"):
            parser = MiniArcParser(config)
    
    # Test that MiniARC-specific error message is used
    empty_task = {"train": [], "test": [{"input": [[1]]}]}
    with pytest.raises(ValueError, match="MiniARC task must have"):
        parser._process_training_pairs(empty_task)


def test_no_duplicate_implementations():
    """Test that methods are not duplicated across parsers."""
    # Check that _pad_and_create_masks is only in base class
    for parser_class in [ArcAgiParser, ConceptArcParser, MiniArcParser]:
        if hasattr(parser_class, "_pad_and_create_masks"):
            # If it exists, it should be the same as base class (not overridden)
            assert (getattr(parser_class, "_pad_and_create_masks") 
                   is ArcDataParserBase._pad_and_create_masks), \
                   f"{parser_class.__name__} should not override _pad_and_create_masks"
    
    # Check that _validate_grid_colors is only in base class
    for parser_class in [ArcAgiParser, ConceptArcParser, MiniArcParser]:
        if hasattr(parser_class, "_validate_grid_colors"):
            # If it exists, it should be the same as base class (not overridden)
            assert (getattr(parser_class, "_validate_grid_colors") 
                   is ArcDataParserBase._validate_grid_colors), \
                   f"{parser_class.__name__} should not override _validate_grid_colors"


def test_inheritance_chain_integrity():
    """Test that inheritance chain is properly maintained."""
    # All parsers should inherit from ArcDataParserBase
    assert issubclass(ArcAgiParser, ArcDataParserBase)
    assert issubclass(ConceptArcParser, ArcDataParserBase)
    assert issubclass(MiniArcParser, ArcDataParserBase)
    
    # Test method resolution order
    for parser_class in [ConceptArcParser, MiniArcParser]:
        mro = parser_class.__mro__
        assert ArcDataParserBase in mro
        assert mro.index(parser_class) < mro.index(ArcDataParserBase)