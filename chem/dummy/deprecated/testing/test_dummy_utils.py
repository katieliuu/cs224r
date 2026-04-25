"""
chem/dummy/testing/test_dummy_utils.py

Unit tests for dummy utility functions.
"""

import sys
import numpy as np

from utils.logger import get_logger
from core.structs import MolGraph
from chem.build.create_molgraph import smiles_to_molgraph
from chem.dummy.dummy_utils import (
    relabel_dummy,
    relabel_dummy_safe,
    get_dummy_labels,
    get_labeled_dummies,
    find_available_dummy_label,
    get_dummy_target,
    validate_dummies,
    ensure_unique_dummy_maps,
)

logger = get_logger(__name__)


def test_relabel_dummy():
    """Test relabel_dummy function."""
    print("\n" + "="*60)
    print("TEST: relabel_dummy")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CCC")
    
    print(f"Original labels: {get_dummy_labels(mg)}")
    
    mg = relabel_dummy(mg, "1", "attachment_A")
    
    labels = get_dummy_labels(mg)
    print(f"After relabel: {labels}")
    
    assert "attachment_A" in labels, "New label should be present"
    assert "1" not in labels, "Old label should be removed"
    print("✓ relabel_dummy works")
    
    print("\n✓ test_relabel_dummy PASSED")


def test_relabel_dummy_safe():
    """Test relabel_dummy_safe function (auto-finds highest numbered site)."""
    print("\n" + "="*60)
    print("TEST: relabel_dummy_safe")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:5]CC[*:3]")
    
    labels_before = get_dummy_labels(mg)
    print(f"Original labels: {labels_before}")
    
    # Should find and relabel the highest numbered site (5)
    mg = relabel_dummy_safe(mg, "highest_site")
    
    labels_after = get_dummy_labels(mg)
    print(f"After relabel_safe: {labels_after}")
    
    assert "highest_site" in labels_after, "New label should be present"
    assert "5" not in labels_after, "Highest label (5) should be replaced"
    assert "1" in labels_after or 1 in labels_after, "Label 1 should remain"
    assert "3" in labels_after or 3 in labels_after, "Label 3 should remain"
    print("✓ relabel_dummy_safe correctly identified highest numbered site")
    
    print("\n✓ test_relabel_dummy_safe PASSED")


def test_get_dummy_labels():
    """Test get_dummy_labels function."""
    print("\n" + "="*60)
    print("TEST: get_dummy_labels")
    print("="*60)
    
    # Multiple dummies
    mg = smiles_to_molgraph("[*:1]CC[*:2]CC[*:3]")
    labels = get_dummy_labels(mg)
    
    print(f"SMILES: [*:1]CC[*:2]CC[*:3]")
    print(f"Labels: {labels}")
    
    assert len(labels) == 3, f"Expected 3 labels, got {len(labels)}"
    print("✓ All labels found")
    
    # No dummies
    mg2 = smiles_to_molgraph("CCCC")
    labels2 = get_dummy_labels(mg2)
    
    print(f"\nSMILES: CCCC")
    print(f"Labels: {labels2}")
    
    assert len(labels2) == 0, f"Expected 0 labels, got {len(labels2)}"
    print("✓ No labels for non-dummy molecule")
    
    print("\n✓ test_get_dummy_labels PASSED")


def test_get_labeled_dummies():
    """Test get_labeled_dummies function."""
    print("\n" + "="*60)
    print("TEST: get_labeled_dummies")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    dummies = get_labeled_dummies(mg)
    
    print(f"SMILES: [*:1]CC[*:2]")
    print(f"Labeled dummies: {dummies}")
    
    assert len(dummies) == 2, f"Expected 2 labeled dummies, got {len(dummies)}"
    
    for d in dummies:
        assert "idx" in d, "Should have 'idx' key"
        assert "label" in d, "Should have 'label' key"
        assert d["label"] is not None, "Label should not be None"
        print(f"  Dummy idx={d['idx']}, label={d['label']}")
    
    print("✓ Labeled dummies retrieved correctly")
    
    print("\n✓ test_get_labeled_dummies PASSED")


def test_find_available_dummy_label():
    """Test find_available_dummy_label function."""
    print("\n" + "="*60)
    print("TEST: find_available_dummy_label")
    print("="*60)
    
    # Graph with site_1 and site_2 taken
    mg = smiles_to_molgraph("[*:1]CC")
    
    # Relabel to use "site_1"
    mg = relabel_dummy(mg, "1", "site_1")
    
    labels = get_dummy_labels(mg)
    print(f"Current labels: {labels}")
    
    available = find_available_dummy_label(mg, base="site")
    print(f"Available label: {available}")
    
    assert available == "site_2", f"Expected 'site_2', got '{available}'"
    print("✓ Found correct available label")
    
    # Test with multiple sites taken
    mg2 = smiles_to_molgraph("[*:1]CC[*:2]CC[*:3]")
    mg2 = relabel_dummy(mg2, "1", "site_1")
    mg2 = relabel_dummy(mg2, "2", "site_2")
    mg2 = relabel_dummy(mg2, "3", "site_3")
    
    available2 = find_available_dummy_label(mg2, base="site")
    print(f"\nWith site_1,2,3 taken, available: {available2}")
    
    assert available2 == "site_4", f"Expected 'site_4', got '{available2}'"
    print("✓ Correctly skipped taken numbers")
    
    print("\n✓ test_find_available_dummy_label PASSED")


def test_get_dummy_target():
    """Test get_dummy_target function."""
    print("\n" + "="*60)
    print("TEST: get_dummy_target")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC")
    
    target = get_dummy_target(mg, "1")
    print(f"SMILES: [*:1]CC")
    print(f"Target of dummy '1': {target}")
    
    assert target is not None, "Should find a target"
    # Target should be a carbon (atomic_num=6)
    assert mg.arrays.atomic_num[target] == 6, f"Target should be carbon, got Z={mg.arrays.atomic_num[target]}"
    print("✓ Target found correctly (carbon atom)")
    
    # Non-existent label
    target_none = get_dummy_target(mg, "999")
    print(f"Target of non-existent '999': {target_none}")
    assert target_none is None, "Should return None for non-existent label"
    print("✓ Returns None for non-existent label")
    
    print("\n✓ test_get_dummy_target PASSED")


def test_validate_dummies():
    """Test validate_dummies function."""
    print("\n" + "="*60)
    print("TEST: validate_dummies")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CCC[*:2]")
    
    warnings = validate_dummies(mg)
    print(f"SMILES: [*:1]CCC[*:2]")
    print(f"Validation warnings: {warnings}")
    print(f"Number of warnings: {len(warnings)}")
    
    # Function should complete without error
    print("✓ Validation completed successfully")
    
    print("\n✓ test_validate_dummies PASSED")


def test_ensure_unique_dummy_maps():
    """Test ensure_unique_dummy_maps function."""
    print("\n" + "="*60)
    print("TEST: ensure_unique_dummy_maps")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    print(f"Original atom_map: {mg.arrays.atom_map}")
    
    mg = ensure_unique_dummy_maps(mg)
    
    print(f"After ensure_unique: {mg.arrays.atom_map}")
    
    # Check dummies have unique non-zero maps
    dummy_indices = [i for i, z in enumerate(mg.arrays.atomic_num) if z == 0]
    map_nums = [mg.arrays.atom_map[i] for i in dummy_indices]
    
    print(f"Dummy indices: {dummy_indices}")
    print(f"Their map numbers: {map_nums}")
    
    assert all(m > 0 for m in map_nums), "All should have non-zero map"
    assert len(set(map_nums)) == len(map_nums), "Maps should be unique"
    print("✓ Maps are unique and non-zero")
    
    print("\n✓ test_ensure_unique_dummy_maps PASSED")


def run_all_tests():
    """Run all dummy_utils tests."""
    print("\n" + "#"*60)
    print("# RUNNING ALL DUMMY_UTILS TESTS")
    print("#"*60)
    
    tests = [
        test_relabel_dummy,
        test_relabel_dummy_safe,
        test_get_dummy_labels,
        test_get_labeled_dummies,
        test_find_available_dummy_label,
        test_get_dummy_target,
        test_validate_dummies,
        test_ensure_unique_dummy_maps,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*60)
    print(f"# RESULTS: {passed} passed, {failed} failed")
    print("#"*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)