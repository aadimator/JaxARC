"""
Serialization utilities for efficient state and configuration management.

This module provides utilities for efficient serialization and deserialization
of JaxARC states and configurations, with special handling for large static
data like task_data.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from loguru import logger

from .task_manager import get_task_id_globally


def extract_task_id_from_index(task_index: jnp.ndarray) -> Optional[str]:
    """Extract the original task ID from a JAX task index.
    
    This function maps a task_index back to its original string task_id
    using the global task manager. This is essential for reconstructing
    task_data during deserialization.
    
    Args:
        task_index: JAX array containing the task index
        
    Returns:
        String task ID if found, None for unknown tasks (-1 index)
        
    Raises:
        ValueError: If task_index is invalid or cannot be processed
        
    Examples:
        ```python
        # Extract task ID from state
        task_id = extract_task_id_from_index(state.task_data.task_index)
        if task_id:
            print(f"Task ID: {task_id}")
        else:
            print("Unknown task")
        ```
    """
    try:
        # Validate input
        if not hasattr(task_index, 'item'):
            raise ValueError(f"task_index must be a JAX array, got {type(task_index)}")
            
        index = int(task_index.item())
        
        # Handle special case for unknown/dummy tasks
        if index == -1:
            logger.debug("Task index is -1 (unknown/dummy task)")
            return None
            
        # Validate index is non-negative
        if index < 0:
            raise ValueError(f"Invalid task index: {index} (must be >= -1)")
            
        # Look up task ID in global manager
        task_id = get_task_id_globally(index)
        if task_id is None:
            logger.warning(f"Task index {index} not found in global task manager")
            
        return task_id
        
    except Exception as e:
        logger.error(f"Error extracting task ID from index: {e}")
        raise ValueError(f"Cannot extract task ID from index: {e}") from e


def validate_task_index_consistency(task_index: jnp.ndarray, parser) -> bool:
    """Validate that a task_index is consistent with the parser's available tasks.
    
    This function checks that a task_index can be resolved to a task_id that
    exists in the provided parser's dataset.
    
    Args:
        task_index: JAX array containing the task index
        parser: ArcDataParserBase instance to validate against
        
    Returns:
        True if task_index is consistent, False otherwise
        
    Examples:
        ```python
        # Validate task index before reconstruction
        is_valid = validate_task_index_consistency(state.task_data.task_index, parser)
        if not is_valid:
            raise ValueError("Task index is inconsistent with parser dataset")
        ```
    """
    try:
        # Extract task ID from index
        task_id = extract_task_id_from_index(task_index)
        if task_id is None:
            # Unknown task (-1 index) is considered valid
            return True
            
        # Check if parser has this task
        available_ids = parser.get_available_task_ids()
        is_available = task_id in available_ids
        
        if not is_available:
            logger.warning(f"Task ID '{task_id}' not available in parser dataset")
            
        return is_available
        
    except Exception as e:
        logger.error(f"Error validating task index consistency: {e}")
        return False


def validate_task_data_reconstruction(original_task_data, reconstructed_task_data) -> bool:
    """Validate that reconstructed task_data matches the original.
    
    This function performs comprehensive validation to ensure that
    task_data reconstructed from task_index is functionally identical
    to the original task_data.
    
    Args:
        original_task_data: Original JaxArcTask before serialization
        reconstructed_task_data: JaxArcTask reconstructed from task_index
        
    Returns:
        True if reconstruction is valid, False otherwise
        
    Examples:
        ```python
        # Validate reconstruction
        is_valid = validate_task_data_reconstruction(original_task, reconstructed_task)
        if not is_valid:
            raise ValueError("Task data reconstruction failed validation")
        ```
    """
    try:
        import equinox as eqx
        
        # Check if both are the same type
        if type(original_task_data) != type(reconstructed_task_data):
            logger.error("Task data types don't match")
            return False
            
        # Compare all fields using equinox tree equality
        if not eqx.tree_equal(original_task_data, reconstructed_task_data):
            logger.error("Task data content doesn't match")
            return False
            
        logger.debug("Task data reconstruction validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating task data reconstruction: {e}")
        return False


def calculate_serialization_savings(original_size: int, compressed_size: int) -> dict:
    """Calculate serialization file size savings.
    
    This function calculates the space savings achieved by excluding
    task_data during serialization.
    
    Args:
        original_size: Size of full serialization (bytes)
        compressed_size: Size of efficient serialization (bytes)
        
    Returns:
        Dictionary with savings statistics
        
    Examples:
        ```python
        # Calculate savings
        savings = calculate_serialization_savings(1000000, 50000)
        print(f"Saved {savings['percentage']:.1f}% space")
        ```
    """
    if original_size == 0:
        return {
            'original_size_bytes': 0,
            'compressed_size_bytes': 0,
            'savings_bytes': 0,
            'percentage': 0.0,
            'compression_ratio': 1.0
        }
        
    savings_bytes = original_size - compressed_size
    percentage = (savings_bytes / original_size) * 100
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'savings_bytes': savings_bytes,
        'percentage': percentage,
        'compression_ratio': compression_ratio
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"