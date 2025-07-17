"""Tests for the operation names module."""

from __future__ import annotations

import pytest

from jaxarc.envs.operations import (
    OPERATION_NAMES,
    get_all_operation_ids,
    get_operation_display_text,
    get_operation_name,
    get_operations_by_category,
    is_valid_operation_id,
)


class TestOperationNames:
    """Test operation names utility functions."""

    def test_get_operation_name_valid_fill_operations(self):
        """Test get_operation_name with valid fill operation IDs."""
        assert get_operation_name(0) == "Fill 0"
        assert get_operation_name(1) == "Fill 1"
        assert get_operation_name(5) == "Fill 5"
        assert get_operation_name(9) == "Fill 9"

    def test_get_operation_name_valid_flood_fill_operations(self):
        """Test get_operation_name with valid flood fill operation IDs."""
        assert get_operation_name(10) == "Flood Fill 0"
        assert get_operation_name(11) == "Flood Fill 1"
        assert get_operation_name(15) == "Flood Fill 5"
        assert get_operation_name(19) == "Flood Fill 9"

    def test_get_operation_name_valid_movement_operations(self):
        """Test get_operation_name with valid movement operation IDs."""
        assert get_operation_name(20) == "Move Up"
        assert get_operation_name(21) == "Move Down"
        assert get_operation_name(22) == "Move Left"
        assert get_operation_name(23) == "Move Right"

    def test_get_operation_name_valid_transformation_operations(self):
        """Test get_operation_name with valid transformation operation IDs."""
        assert get_operation_name(24) == "Rotate CW"
        assert get_operation_name(25) == "Rotate CCW"
        assert get_operation_name(26) == "Flip H"
        assert get_operation_name(27) == "Flip V"

    def test_get_operation_name_valid_editing_operations(self):
        """Test get_operation_name with valid editing operation IDs."""
        assert get_operation_name(28) == "Copy"
        assert get_operation_name(29) == "Paste"
        assert get_operation_name(30) == "Cut"
        assert get_operation_name(31) == "Clear"

    def test_get_operation_name_valid_special_operations(self):
        """Test get_operation_name with valid special operation IDs."""
        assert get_operation_name(32) == "Copy Input"
        assert get_operation_name(33) == "Resize"
        assert get_operation_name(34) == "Submit"

    def test_get_operation_name_invalid_negative(self):
        """Test get_operation_name with negative operation ID."""
        with pytest.raises(ValueError, match="Unknown operation ID: -1"):
            get_operation_name(-1)

    def test_get_operation_name_invalid_too_high(self):
        """Test get_operation_name with too high operation ID."""
        with pytest.raises(ValueError, match="Unknown operation ID: 999"):
            get_operation_name(999)

    def test_get_operation_name_invalid_gap(self):
        """Test get_operation_name with operation ID in gap (not implemented)."""
        # There might be gaps in the operation IDs, test some likely gaps
        with pytest.raises(ValueError, match="Unknown operation ID: 35"):
            get_operation_name(35)

    def test_get_operation_display_text_valid(self):
        """Test get_operation_display_text with valid operation IDs."""
        assert get_operation_display_text(0) == "Op 0: Fill 0"
        assert get_operation_display_text(10) == "Op 10: Flood Fill 0"
        assert get_operation_display_text(20) == "Op 20: Move Up"
        assert get_operation_display_text(34) == "Op 34: Submit"

    def test_get_operation_display_text_invalid(self):
        """Test get_operation_display_text with invalid operation ID."""
        with pytest.raises(ValueError, match="Unknown operation ID: 999"):
            get_operation_display_text(999)

    def test_is_valid_operation_id_valid(self):
        """Test is_valid_operation_id with valid operation IDs."""
        assert is_valid_operation_id(0) is True
        assert is_valid_operation_id(9) is True
        assert is_valid_operation_id(10) is True
        assert is_valid_operation_id(19) is True
        assert is_valid_operation_id(20) is True
        assert is_valid_operation_id(23) is True
        assert is_valid_operation_id(24) is True
        assert is_valid_operation_id(27) is True
        assert is_valid_operation_id(28) is True
        assert is_valid_operation_id(31) is True
        assert is_valid_operation_id(32) is True
        assert is_valid_operation_id(34) is True

    def test_is_valid_operation_id_invalid(self):
        """Test is_valid_operation_id with invalid operation IDs."""
        assert is_valid_operation_id(-1) is False
        assert is_valid_operation_id(35) is False
        assert is_valid_operation_id(999) is False

    def test_get_all_operation_ids(self):
        """Test get_all_operation_ids function."""
        all_ids = get_all_operation_ids()

        # Should be a list
        assert isinstance(all_ids, list)

        # Should contain all expected operation IDs
        expected_ids = (
            list(range(10))
            + list(range(10, 20))
            + list(range(20, 24))
            + list(range(24, 28))
            + list(range(28, 32))
            + list(range(32, 35))
        )

        assert sorted(all_ids) == sorted(expected_ids)

        # Should contain all keys from OPERATION_NAMES
        assert sorted(all_ids) == sorted(OPERATION_NAMES.keys())

    def test_get_operations_by_category(self):
        """Test get_operations_by_category function."""
        categories = get_operations_by_category()

        # Should be a dictionary
        assert isinstance(categories, dict)

        # Should have all expected categories
        expected_categories = [
            "fill",
            "flood_fill",
            "movement",
            "transformation",
            "editing",
            "special",
        ]
        assert sorted(categories.keys()) == sorted(expected_categories)

        # Test each category
        assert categories["fill"] == list(range(10))
        assert categories["flood_fill"] == list(range(10, 20))
        assert categories["movement"] == list(range(20, 24))
        assert categories["transformation"] == list(range(24, 28))
        assert categories["editing"] == list(range(28, 32))
        assert categories["special"] == list(range(32, 35))

    def test_operation_names_completeness(self):
        """Test that OPERATION_NAMES contains all expected operations."""
        # Test that all fill operations are present
        for i in range(10):
            assert i in OPERATION_NAMES
            assert f"Fill {i}" in OPERATION_NAMES[i]

        # Test that all flood fill operations are present
        for i in range(10, 20):
            assert i in OPERATION_NAMES
            assert f"Flood Fill {i - 10}" in OPERATION_NAMES[i]

        # Test movement operations
        movement_ops = {
            20: "Move Up",
            21: "Move Down",
            22: "Move Left",
            23: "Move Right",
        }
        for op_id, name in movement_ops.items():
            assert op_id in OPERATION_NAMES
            assert OPERATION_NAMES[op_id] == name

        # Test transformation operations
        transform_ops = {24: "Rotate CW", 25: "Rotate CCW", 26: "Flip H", 27: "Flip V"}
        for op_id, name in transform_ops.items():
            assert op_id in OPERATION_NAMES
            assert OPERATION_NAMES[op_id] == name

        # Test editing operations
        edit_ops = {28: "Copy", 29: "Paste", 30: "Cut", 31: "Clear"}
        for op_id, name in edit_ops.items():
            assert op_id in OPERATION_NAMES
            assert OPERATION_NAMES[op_id] == name

        # Test special operations
        special_ops = {32: "Copy Input", 33: "Resize", 34: "Submit"}
        for op_id, name in special_ops.items():
            assert op_id in OPERATION_NAMES
            assert OPERATION_NAMES[op_id] == name

    def test_operation_names_consistency(self):
        """Test that all functions are consistent with OPERATION_NAMES."""
        # Test that get_operation_name works for all IDs in OPERATION_NAMES
        for op_id in OPERATION_NAMES:
            assert get_operation_name(op_id) == OPERATION_NAMES[op_id]
            assert is_valid_operation_id(op_id) is True

        # Test that get_all_operation_ids returns exactly the keys of OPERATION_NAMES
        assert set(get_all_operation_ids()) == set(OPERATION_NAMES.keys())

        # Test that get_operations_by_category covers all operations
        categories = get_operations_by_category()
        all_categorized_ops = []
        for category_ops in categories.values():
            all_categorized_ops.extend(category_ops)

        assert set(all_categorized_ops) == set(OPERATION_NAMES.keys())

    def test_operation_names_no_duplicates(self):
        """Test that operation categories don't have overlapping IDs."""
        categories = get_operations_by_category()

        all_ops = []
        for category_ops in categories.values():
            all_ops.extend(category_ops)

        # Check no duplicates
        assert len(all_ops) == len(set(all_ops))

        # Check all categories are disjoint
        category_sets = [set(ops) for ops in categories.values()]
        for i, set1 in enumerate(category_sets):
            for j, set2 in enumerate(category_sets):
                if i != j:
                    assert set1.isdisjoint(set2), (
                        f"Categories {i} and {j} have overlapping operations"
                    )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with the first operation ID
        assert get_operation_name(0) == "Fill 0"
        assert get_operation_display_text(0) == "Op 0: Fill 0"

        # Test with the last operation ID
        assert get_operation_name(34) == "Submit"
        assert get_operation_display_text(34) == "Op 34: Submit"

        # Test boundary between categories
        assert get_operation_name(9) == "Fill 9"
        assert get_operation_name(10) == "Flood Fill 0"
        assert get_operation_name(19) == "Flood Fill 9"
        assert get_operation_name(20) == "Move Up"
