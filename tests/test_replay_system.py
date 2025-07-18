"""Tests for episode replay and analysis system."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from jaxarc.utils.logging.structured_logger import (
    EpisodeLogEntry,
    LoggingConfig,
    StepLogEntry,
    StructuredLogger,
)
from jaxarc.utils.visualization.analysis_tools import (
    AnalysisConfig,
    EpisodeAnalysisTools,
    FailureModeAnalysis,
    PerformanceMetrics,
)
from jaxarc.utils.visualization.replay_system import (
    EpisodeReplaySystem,
    ReplayConfig,
    ReplayValidationResult,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_structured_logger(temp_dir):
    """Create mock structured logger with test data."""
    config = LoggingConfig(
        output_dir=str(temp_dir / "logs"),
        include_full_states=False
    )
    logger = StructuredLogger(config)
    return logger


@pytest.fixture
def sample_episode():
    """Create sample episode data for testing."""
    steps = [
        StepLogEntry(
            step_num=0,
            timestamp=time.time(),
            before_state={"type": "ArcEnvState", "step_count": 0},
            action={"operation": 10, "selection": [[1, 0], [0, 1]]},
            after_state={"type": "ArcEnvState", "step_count": 1},
            reward=0.1,
            info={"similarity": 0.2}
        ),
        StepLogEntry(
            step_num=1,
            timestamp=time.time() + 1,
            before_state={"type": "ArcEnvState", "step_count": 1},
            action={"operation": 15, "selection": [[0, 1], [1, 0]]},
            after_state={"type": "ArcEnvState", "step_count": 2},
            reward=0.3,
            info={"similarity": 0.5}
        )
    ]
    
    return EpisodeLogEntry(
        episode_num=1,
        start_timestamp=time.time(),
        end_timestamp=time.time() + 10,
        total_steps=2,
        total_reward=0.4,
        final_similarity=0.5,
        task_id="test_task_001",
        config_hash="abc123",
        steps=steps,
        metadata={"test": True}
    )


@pytest.fixture
def replay_system(mock_structured_logger, temp_dir):
    """Create replay system for testing."""
    config = ReplayConfig(
        output_dir=str(temp_dir / "replay"),
        validate_integrity=True,
        regenerate_visualizations=False
    )
    return EpisodeReplaySystem(mock_structured_logger, config)


class TestReplayConfig:
    """Test replay configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReplayConfig()
        assert config.validate_integrity is True
        assert config.regenerate_visualizations is False
        assert config.max_episodes_to_load == 100
        assert config.include_step_details is True
        assert config.comparison_metrics == ['reward', 'similarity', 'steps']
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReplayConfig(
            validate_integrity=False,
            regenerate_visualizations=True,
            max_episodes_to_load=50,
            comparison_metrics=['reward']
        )
        assert config.validate_integrity is False
        assert config.regenerate_visualizations is True
        assert config.max_episodes_to_load == 50
        assert config.comparison_metrics == ['reward']


class TestEpisodeReplaySystem:
    """Test episode replay system."""
    
    def test_initialization(self, replay_system, temp_dir):
        """Test replay system initialization."""
        assert replay_system.config.output_dir == str(temp_dir / "replay")
        assert replay_system.output_dir.exists()
        assert replay_system._episode_cache == {}
    
    def test_load_episode_not_found(self, replay_system):
        """Test loading non-existent episode."""
        with patch.object(replay_system.structured_logger, 'load_episode', return_value=None):
            result = replay_system.load_episode(999)
            assert result is None
    
    def test_load_episode_with_cache(self, replay_system, sample_episode):
        """Test loading episode with caching."""
        # Mock the structured logger to return our sample episode
        with patch.object(replay_system.structured_logger, 'load_episode', return_value=sample_episode):
            # First load
            result1 = replay_system.load_episode(1, use_cache=True)
            assert result1 == sample_episode
            assert 1 in replay_system._episode_cache
            
            # Second load should use cache
            result2 = replay_system.load_episode(1, use_cache=True)
            assert result2 == sample_episode
            
            # Verify structured logger was only called once
            replay_system.structured_logger.load_episode.assert_called_once_with(1)
    
    def test_validate_episode_integrity_valid(self, replay_system, sample_episode):
        """Test validation of valid episode."""
        result = replay_system.validate_episode_integrity(sample_episode)
        
        assert isinstance(result, ReplayValidationResult)
        assert result.is_valid is True
        assert result.episode_num == 1
        assert result.total_steps == 2
        assert len(result.errors) == 0
    
    def test_validate_episode_integrity_invalid_steps(self, replay_system, sample_episode):
        """Test validation with mismatched step count."""
        # Modify episode to have incorrect step count
        invalid_episode = EpisodeLogEntry(
            episode_num=sample_episode.episode_num,
            start_timestamp=sample_episode.start_timestamp,
            end_timestamp=sample_episode.end_timestamp,
            total_steps=5,  # Wrong count
            total_reward=sample_episode.total_reward,
            final_similarity=sample_episode.final_similarity,
            task_id=sample_episode.task_id,
            config_hash=sample_episode.config_hash,
            steps=sample_episode.steps,  # Only 2 steps
            metadata=sample_episode.metadata
        )
        
        result = replay_system.validate_episode_integrity(invalid_episode)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Step count mismatch" in error for error in result.errors)
    
    def test_validate_episode_integrity_invalid_timestamps(self, replay_system, sample_episode):
        """Test validation with invalid timestamps."""
        # Modify episode to have invalid timestamps
        invalid_episode = EpisodeLogEntry(
            episode_num=sample_episode.episode_num,
            start_timestamp=sample_episode.end_timestamp,  # Start after end
            end_timestamp=sample_episode.start_timestamp,
            total_steps=sample_episode.total_steps,
            total_reward=sample_episode.total_reward,
            final_similarity=sample_episode.final_similarity,
            task_id=sample_episode.task_id,
            config_hash=sample_episode.config_hash,
            steps=sample_episode.steps,
            metadata=sample_episode.metadata
        )
        
        result = replay_system.validate_episode_integrity(invalid_episode)
        
        assert result.is_valid is False
        assert any("Invalid timestamps" in error for error in result.errors)
    
    def test_reconstruct_state_minimal_logging(self, replay_system, sample_episode):
        """Test state reconstruction with minimal logging."""
        step = sample_episode.steps[0]
        result = replay_system.reconstruct_state_from_log(step)
        
        # Should return None for minimal logging
        assert result is None
    
    def test_replay_episode_not_found(self, replay_system):
        """Test replaying non-existent episode."""
        with patch.object(replay_system, 'load_episode', return_value=None):
            result = replay_system.replay_episode(999)
            assert result is None
    
    def test_replay_episode_success(self, replay_system, sample_episode, temp_dir):
        """Test successful episode replay."""
        with patch.object(replay_system, 'load_episode', return_value=sample_episode):
            result = replay_system.replay_episode(1, validate=False, regenerate_visualizations=False)
            
            assert result is not None
            assert result['episode_num'] == 1
            assert result['total_steps'] == 2
            assert result['total_reward'] == 0.4
            assert result['final_similarity'] == 0.5
            assert result['task_id'] == "test_task_001"
            
            # Check that summary file was created
            summary_path = temp_dir / "replay" / "episode_0001_replay" / "replay_summary.json"
            assert summary_path.exists()
    
    def test_list_available_episodes(self, replay_system):
        """Test listing available episodes."""
        with patch.object(replay_system.structured_logger, 'list_episodes', return_value=[1, 2, 3]):
            episodes = replay_system.list_available_episodes()
            assert episodes == [1, 2, 3]
    
    def test_get_episode_summaries(self, replay_system):
        """Test getting episode summaries."""
        mock_summaries = [
            {'episode_num': 1, 'total_reward': 0.5},
            {'episode_num': 2, 'total_reward': 0.8}
        ]
        
        with patch.object(replay_system.structured_logger, 'get_episode_summary') as mock_get:
            mock_get.side_effect = lambda x: mock_summaries[x-1] if x <= 2 else None
            
            summaries = replay_system.get_episode_summaries([1, 2])
            assert len(summaries) == 2
            assert summaries[0]['episode_num'] == 1
            assert summaries[1]['episode_num'] == 2
    
    def test_find_episodes_by_criteria(self, replay_system):
        """Test finding episodes by criteria."""
        mock_summaries = [
            {'episode_num': 1, 'total_reward': 0.3, 'final_similarity': 0.2, 'total_steps': 10, 'task_id': 'task1'},
            {'episode_num': 2, 'total_reward': 0.8, 'final_similarity': 0.9, 'total_steps': 15, 'task_id': 'task2'},
            {'episode_num': 3, 'total_reward': 0.1, 'final_similarity': 0.1, 'total_steps': 5, 'task_id': 'task1'}
        ]
        
        with patch.object(replay_system, 'get_episode_summaries', return_value=mock_summaries):
            # Test reward criteria
            episodes = replay_system.find_episodes_by_criteria(min_reward=0.5)
            assert episodes == [2]
            
            # Test similarity criteria
            episodes = replay_system.find_episodes_by_criteria(min_similarity=0.8)
            assert episodes == [2]
            
            # Test task ID criteria
            episodes = replay_system.find_episodes_by_criteria(task_id='task1')
            assert episodes == [1, 3]
            
            # Test multiple criteria
            episodes = replay_system.find_episodes_by_criteria(min_reward=0.2, task_id='task1')
            assert episodes == [1]
    
    def test_clear_cache(self, replay_system, sample_episode):
        """Test clearing episode cache."""
        # Add something to cache
        replay_system._episode_cache[1] = sample_episode
        assert len(replay_system._episode_cache) == 1
        
        # Clear cache
        replay_system.clear_cache()
        assert len(replay_system._episode_cache) == 0


class TestAnalysisConfig:
    """Test analysis configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AnalysisConfig()
        assert config.generate_plots is True
        assert config.plot_format == "png"
        assert config.include_step_analysis is True
        assert config.failure_threshold == 0.1
        assert config.success_threshold == 0.9
        assert config.max_episodes_per_analysis == 1000


class TestEpisodeAnalysisTools:
    """Test episode analysis tools."""
    
    @pytest.fixture
    def analysis_tools(self, replay_system, temp_dir):
        """Create analysis tools for testing."""
        config = AnalysisConfig(
            output_dir=str(temp_dir / "analysis"),
            generate_plots=False  # Disable plots for testing
        )
        return EpisodeAnalysisTools(replay_system, config)
    
    def test_initialization(self, analysis_tools, temp_dir):
        """Test analysis tools initialization."""
        assert analysis_tools.config.output_dir == str(temp_dir / "analysis")
        assert analysis_tools.output_dir.exists()
    
    def test_analyze_performance_metrics_empty(self, analysis_tools):
        """Test performance analysis with no data."""
        with patch.object(analysis_tools.replay_system, 'get_episode_summaries', return_value=[]):
            metrics = analysis_tools.analyze_performance_metrics([])
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.total_episodes == 0
            assert metrics.success_rate == 0.0
            assert metrics.best_episode == -1
            assert metrics.worst_episode == -1
    
    def test_analyze_performance_metrics_with_data(self, analysis_tools):
        """Test performance analysis with sample data."""
        mock_summaries = [
            {'episode_num': 1, 'total_reward': 0.3, 'final_similarity': 0.2, 'total_steps': 10},
            {'episode_num': 2, 'total_reward': 0.8, 'final_similarity': 0.95, 'total_steps': 15},
            {'episode_num': 3, 'total_reward': 0.1, 'final_similarity': 0.1, 'total_steps': 5}
        ]
        
        with patch.object(analysis_tools.replay_system, 'get_episode_summaries', return_value=mock_summaries):
            metrics = analysis_tools.analyze_performance_metrics([1, 2, 3])
            
            assert metrics.total_episodes == 3
            assert metrics.success_rate == 1/3  # Only episode 2 has similarity >= 0.9
            assert metrics.average_reward == 0.4  # (0.3 + 0.8 + 0.1) / 3
            assert metrics.average_similarity == pytest.approx(0.417, rel=1e-2)  # (0.2 + 0.95 + 0.1) / 3
            assert metrics.average_steps == 10.0  # (10 + 15 + 5) / 3
            assert metrics.best_episode == 2  # Highest similarity
            assert metrics.worst_episode == 3  # Lowest similarity
    
    def test_analyze_failure_modes_no_failures(self, analysis_tools):
        """Test failure mode analysis with no failures."""
        mock_summaries = [
            {'episode_num': 1, 'final_similarity': 0.8},
            {'episode_num': 2, 'final_similarity': 0.9}
        ]
        
        with patch.object(analysis_tools.replay_system, 'get_episode_summaries', return_value=mock_summaries):
            analysis = analysis_tools.analyze_failure_modes([1, 2])
            
            assert isinstance(analysis, FailureModeAnalysis)
            assert len(analysis.failure_episodes) == 0
            assert len(analysis.common_failure_patterns) == 0
    
    def test_analyze_failure_modes_with_failures(self, analysis_tools, sample_episode):
        """Test failure mode analysis with failures."""
        # Create a failure episode
        failure_episode = EpisodeLogEntry(
            episode_num=2,
            start_timestamp=time.time(),
            end_timestamp=time.time() + 5,
            total_steps=3,
            total_reward=-0.5,
            final_similarity=0.05,  # Below failure threshold
            task_id="test_task_002",
            config_hash="def456",
            steps=[
                StepLogEntry(
                    step_num=0,
                    timestamp=time.time(),
                    before_state={"type": "ArcEnvState"},
                    action={"operation": 10},
                    after_state={"type": "ArcEnvState"},
                    reward=-0.2,
                    info={"similarity": 0.05}
                ),
                StepLogEntry(
                    step_num=1,
                    timestamp=time.time() + 1,
                    before_state={"type": "ArcEnvState"},
                    action={"operation": 10},  # Same operation (repetitive)
                    after_state={"type": "ArcEnvState"},
                    reward=-0.3,
                    info={"similarity": 0.05}
                )
            ]
        )
        
        mock_summaries = [
            {'episode_num': 2, 'final_similarity': 0.05}  # Failure
        ]
        
        with patch.object(analysis_tools.replay_system, 'get_episode_summaries', return_value=mock_summaries):
            with patch.object(analysis_tools.replay_system, 'load_episode', return_value=failure_episode):
                analysis = analysis_tools.analyze_failure_modes([2])
                
                assert len(analysis.failure_episodes) == 1
                assert 2 in analysis.failure_episodes
                assert analysis.average_failure_similarity == 0.05
                assert 'early_termination' in analysis.common_failure_patterns
                assert 'negative_reward_spiral' in analysis.common_failure_patterns
                assert 'very_low_similarity' in analysis.common_failure_patterns
    
    def test_compare_episodes(self, analysis_tools):
        """Test episode comparison."""
        mock_summaries = [
            {'episode_num': 1, 'total_reward': 0.3, 'final_similarity': 0.2, 'total_steps': 10},
            {'episode_num': 2, 'total_reward': 0.8, 'final_similarity': 0.9, 'total_steps': 15}
        ]
        
        with patch.object(analysis_tools.replay_system.structured_logger, 'get_episode_summary') as mock_get:
            mock_get.side_effect = lambda x: mock_summaries[x-1] if x <= 2 else None
            
            comparison = analysis_tools.compare_episodes([1, 2], metrics=['reward', 'similarity'])
            
            assert comparison['episodes'] == [1, 2]
            assert 'reward' in comparison['metrics']
            assert 'similarity' in comparison['metrics']
            
            # Check reward comparison
            reward_data = comparison['metrics']['reward']
            assert reward_data['values'] == [0.3, 0.8]
            assert reward_data['best_episode'] == 2
            assert reward_data['worst_episode'] == 1
            assert reward_data['range'] == 0.5
            
            # Check similarity comparison
            similarity_data = comparison['metrics']['similarity']
            assert similarity_data['values'] == [0.2, 0.9]
            assert similarity_data['best_episode'] == 2
            assert similarity_data['worst_episode'] == 1
            assert similarity_data['range'] == 0.7
    
    def test_generate_step_by_step_analysis(self, analysis_tools, sample_episode):
        """Test step-by-step analysis generation."""
        with patch.object(analysis_tools.replay_system, 'load_episode', return_value=sample_episode):
            analysis = analysis_tools.generate_step_by_step_analysis(1)
            
            assert analysis is not None
            assert analysis['episode_num'] == 1
            assert analysis['total_steps'] == 2
            assert analysis['final_similarity'] == 0.5
            assert analysis['total_reward'] == 0.4
            assert len(analysis['step_analysis']) == 2
            
            # Check step analysis details
            step0 = analysis['step_analysis'][0]
            assert step0['step_num'] == 0
            assert step0['operation'] == 10
            assert step0['reward'] == 0.1
            assert step0['similarity'] == 0.2
            
            step1 = analysis['step_analysis'][1]
            assert step1['step_num'] == 1
            assert step1['operation'] == 15
            assert step1['reward'] == 0.3
            assert step1['similarity'] == 0.5
            assert step1['similarity_change'] == 0.3  # 0.5 - 0.2
    
    def test_generate_step_by_step_analysis_not_found(self, analysis_tools):
        """Test step-by-step analysis for non-existent episode."""
        with patch.object(analysis_tools.replay_system, 'load_episode', return_value=None):
            analysis = analysis_tools.generate_step_by_step_analysis(999)
            assert analysis is None
    
    def test_export_analysis_report(self, analysis_tools, temp_dir):
        """Test comprehensive analysis report export."""
        mock_summaries = [
            {'episode_num': 1, 'total_reward': 0.5, 'final_similarity': 0.8, 'total_steps': 10}
        ]
        
        with patch.object(analysis_tools.replay_system, 'list_available_episodes', return_value=[1]):
            with patch.object(analysis_tools.replay_system, 'get_episode_summaries', return_value=mock_summaries):
                with patch.object(analysis_tools.replay_system, 'load_episode', return_value=None):  # No detailed episodes
                    report_path = analysis_tools.export_analysis_report([1], include_plots=False)
                    
                    assert Path(report_path).exists()
                    
                    # Load and verify report content
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    assert 'analysis_timestamp' in report
                    assert report['total_episodes_analyzed'] == 1
                    assert 'performance_metrics' in report
                    assert 'failure_analysis' in report
                    assert report['performance_metrics']['total_episodes'] == 1