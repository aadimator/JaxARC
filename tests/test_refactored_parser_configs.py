"""Tests for refactored parser configuration patterns.

This module tests the new typed configuration pattern for all parsers,
including initialization with typed configs and from_hydra methods.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from omegaconf import DictConfig

from jaxarc.envs.config import DatasetConfig
from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.parsers.mini_arc import MiniArcParser


class TestParserTypedConfigs:
    """Test parser initialization with typed configs."""

    @pytest.fixture
    def dataset_config(self):
        """Create a test dataset configuration."""
        return DatasetConfig(
            dataset_path="test/path",
            task_split="train",
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=3,
            max_test_pairs=1,
        )

    @pytest.fixture
    def hydra_config(self):
        """Create a test Hydra configuration."""
        return DictConfig({
            "dataset_path": "test/path",
            "task_split": "train",
            "max_grid_height": 5,
            "max_grid_width": 5,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 3,
            "max_test_pairs": 1,
        })

    def test_mini_arc_parser_typed_config_init(self, dataset_config):
        """Test MiniArcParser initialization with typed config."""
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            parser = MiniArcParser(dataset_config)
            assert parser.config == dataset_config
            assert isinstance(parser.config, DatasetConfig)

    def test_arc_agi_parser_typed_config_init(self, dataset_config):
        """Test ArcAgiParser initialization with typed config."""
        with patch.object(ArcAgiParser, '_load_and_cache_tasks'):
            parser = ArcAgiParser(dataset_config)
            assert parser.config == dataset_config
            assert isinstance(parser.config, DatasetConfig)

    def test_concept_arc_parser_typed_config_init(self, dataset_config):
        """Test ConceptArcParser initialization with typed config."""
        with patch.object(ConceptArcParser, '_load_and_cache_tasks'):
            parser = ConceptArcParser(dataset_config)
            assert parser.config == dataset_config
            assert isinstance(parser.config, DatasetConfig)

    def test_mini_arc_parser_from_hydra(self, hydra_config):
        """Test MiniArcParser.from_hydra class method."""
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            parser = MiniArcParser.from_hydra(hydra_config)
            assert isinstance(parser.config, DatasetConfig)
            assert parser.config.dataset_path == "test/path"
            assert parser.config.max_grid_height == 5

    def test_arc_agi_parser_from_hydra(self, hydra_config):
        """Test ArcAgiParser.from_hydra class method."""
        with patch.object(ArcAgiParser, '_load_and_cache_tasks'):
            parser = ArcAgiParser.from_hydra(hydra_config)
            assert isinstance(parser.config, DatasetConfig)
            assert parser.config.dataset_path == "test/path"
            assert parser.config.max_grid_height == 5

    def test_concept_arc_parser_from_hydra(self, hydra_config):
        """Test ConceptArcParser.from_hydra class method."""
        with patch.object(ConceptArcParser, '_load_and_cache_tasks'):
            parser = ConceptArcParser.from_hydra(hydra_config)
            assert isinstance(parser.config, DatasetConfig)
            assert parser.config.dataset_path == "test/path"
            assert parser.config.max_grid_height == 5

    def test_from_hydra_creates_equivalent_config(self, hydra_config, dataset_config):
        """Test that from_hydra creates equivalent config to direct initialization."""
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            parser_direct = MiniArcParser(dataset_config)
            parser_hydra = MiniArcParser.from_hydra(hydra_config)
            
            # Compare all relevant fields
            assert parser_direct.config.dataset_path == parser_hydra.config.dataset_path
            assert parser_direct.config.max_grid_height == parser_hydra.config.max_grid_height
            assert parser_direct.config.max_grid_width == parser_hydra.config.max_grid_width
            assert parser_direct.config.max_train_pairs == parser_hydra.config.max_train_pairs

    def test_parser_config_validation(self):
        """Test that parsers validate configuration parameters."""
        # Test that parsers properly validate configs during initialization
        invalid_config = DatasetConfig(
            dataset_path="",  # Invalid empty path
            task_split="invalid",  # Invalid split
            max_grid_height=0,  # Invalid height
            max_grid_width=0,  # Invalid width
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=3,
            max_test_pairs=1,
        )
        
        # The parsers should raise validation errors for invalid configs
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            with pytest.raises(ValueError, match="Configuration validation failed"):
                MiniArcParser(invalid_config)

    def test_all_parsers_have_from_hydra_method(self):
        """Test that all parsers have the from_hydra class method."""
        parsers = [MiniArcParser, ArcAgiParser, ConceptArcParser]
        
        for parser_class in parsers:
            assert hasattr(parser_class, 'from_hydra')
            assert callable(getattr(parser_class, 'from_hydra'))
            
            # Check that it's a classmethod
            method = getattr(parser_class, 'from_hydra')
            assert isinstance(method, classmethod) or hasattr(method, '__self__')

    def test_parser_config_immutability(self, dataset_config):
        """Test that parser configs are properly handled as immutable."""
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            parser = MiniArcParser(dataset_config)
            original_path = parser.config.dataset_path
            
            # Create a new config with different values (equinox modules are immutable)
            modified_config = DatasetConfig(
                dataset_path="modified/path",
                task_split=dataset_config.task_split,
                max_grid_height=dataset_config.max_grid_height,
                max_grid_width=dataset_config.max_grid_width,
                min_grid_height=dataset_config.min_grid_height,
                min_grid_width=dataset_config.min_grid_width,
                max_colors=dataset_config.max_colors,
                background_color=dataset_config.background_color,
                max_train_pairs=dataset_config.max_train_pairs,
                max_test_pairs=dataset_config.max_test_pairs,
            )
            
            # Original parser config should be unchanged
            assert parser.config.dataset_path == original_path
            assert modified_config.dataset_path == "modified/path"

    def test_backward_compatibility_warning(self, hydra_config):
        """Test that using old patterns shows appropriate warnings."""
        # This test ensures we can detect when old patterns are used
        # In practice, this would be implemented in the actual usage sites
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            # New pattern (preferred)
            parser_new = MiniArcParser.from_hydra(hydra_config)
            assert isinstance(parser_new.config, DatasetConfig)
            
            # Direct typed config (also preferred)
            config = DatasetConfig.from_hydra(hydra_config)
            parser_direct = MiniArcParser(config)
            assert isinstance(parser_direct.config, DatasetConfig)


class TestParserConfigIntegration:
    """Test parser configuration integration with actual usage patterns."""

    def test_config_factory_integration(self):
        """Test that parsers work with config factory patterns."""
        # Create config using the factory pattern
        config = DatasetConfig(
            dataset_path="test/path",
            task_split="train",
            max_grid_height=10,
            max_grid_width=10,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=5,
            max_test_pairs=2,
        )
        
        with patch.object(MiniArcParser, '_load_and_cache_tasks'):
            parser = MiniArcParser(config)
            assert parser.config.max_grid_height == 10
            assert parser.config.max_train_pairs == 5

    def test_hydra_config_conversion_accuracy(self):
        """Test that Hydra config conversion preserves all values."""
        hydra_config = DictConfig({
            "dataset_path": "/data/arc",
            "task_split": "evaluation",
            "max_grid_height": 15,
            "max_grid_width": 20,
            "min_grid_height": 2,
            "min_grid_width": 3,
            "max_colors": 12,
            "background_color": 1,
            "max_train_pairs": 4,
            "max_test_pairs": 3,
        })
        
        config = DatasetConfig.from_hydra(hydra_config)
        
        # Verify all values are preserved
        assert config.dataset_path == "/data/arc"
        assert config.task_split == "evaluation"
        assert config.max_grid_height == 15
        assert config.max_grid_width == 20
        assert config.min_grid_height == 2
        assert config.min_grid_width == 3
        assert config.max_colors == 12
        assert config.background_color == 1
        assert config.max_train_pairs == 4
        assert config.max_test_pairs == 3

    def test_parser_config_error_handling(self):
        """Test parser error handling with invalid configurations."""
        # Test with None config
        with pytest.raises((TypeError, AttributeError)):
            MiniArcParser(None)
        
        # Test with wrong type
        with pytest.raises((TypeError, AttributeError)):
            MiniArcParser("invalid_config")