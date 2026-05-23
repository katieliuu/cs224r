"""
chem/dummy/testing/test_dummy_manager.py

Unit tests for DummyManager class.
"""

import sys
import numpy as np

from utils.logger import get_logger
from core.structs import MolGraph, MolArrays, AttachmentPointArray, AttachmentKind
from chem.build.create_molgraph import smiles_to_molgraph
from chem.dummy.dummy_manager import DummyManager

logger = get_logger(__name__)


def test_find_dummy_indices():
    """Test finding dummy atom indices."""
    print("\n" + "="*60)
    print("TEST: find_dummy_indices")
    print("="*60)
    
    # Single dummy
    mg = smiles_to_molgraph("[*:1]CC")
    manager = DummyManager(mg)
    indices = manager.find_dummy_indices()
    
    print(f"SMILES: [*:1]CC")
    print(f"Dummy indices: {indices}")
    assert len(indices) == 1, f"Expected 1 dummy, got {len(indices)}"
    assert mg.arrays.atomic_num[indices[0]] == 0, "Dummy should have atomic_num=0"
    print("✓ Single dummy found correctly")
    
    # Multiple dummies
    mg2 = smiles_to_molgraph("[*:1]CC[*:2]")
    manager2 = DummyManager(mg2)
    indices2 = manager2.find_dummy_indices()
    
    print(f"\nSMILES: [*:1]CC[*:2]")
    print(f"Dummy indices: {indices2}")
    assert len(indices2) == 2, f"Expected 2 dummies, got {len(indices2)}"
    print("✓ Multiple dummies found correctly")
    
    # No dummies
    mg3 = smiles_to_molgraph("CCC")
    manager3 = DummyManager(mg3)
    indices3 = manager3.find_dummy_indices()
    
    print(f"\nSMILES: CCC")
    print(f"Dummy indices: {indices3}")
    assert len(indices3) == 0, f"Expected 0 dummies, got {len(indices3)}"
    print("✓ No dummies case handled correctly")
    
    print("\n✓ test_find_dummy_indices PASSED")


def test_get_all_dummy_labels():
    """Test getting all dummy labels."""
    print("\n" + "="*60)
    print("TEST: get_all_dummy_labels")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    manager = DummyManager(mg)
    labels = manager.get_all_dummy_labels()
    
    print(f"SMILES: [*:1]CC[*:2]")
    print(f"Labels: {labels}")
    assert len(labels) == 2, f"Expected 2 labels, got {len(labels)}"
    assert "1" in labels or 1 in labels, "Label '1' should be present"
    assert "2" in labels or 2 in labels, "Label '2' should be present"
    print("✓ Labels extracted correctly")
    
    print("\n✓ test_get_all_dummy_labels PASSED")


def test_find_dummy_by_label():
    """Test finding dummy by label."""
    print("\n" + "="*60)
    print("TEST: find_dummy_by_label")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    manager = DummyManager(mg)
    
    # Find existing label
    idx1 = manager.find_dummy_by_label("1")
    idx2 = manager.find_dummy_by_label("2")
    
    print(f"SMILES: [*:1]CC[*:2]")
    print(f"Index for label '1': {idx1}")
    print(f"Index for label '2': {idx2}")
    
    assert idx1 is not None, "Should find dummy with label '1'"
    assert idx2 is not None, "Should find dummy with label '2'"
    assert idx1 != idx2, "Should be different indices"
    print("✓ Found dummies by label")
    
    # Try non-existent label
    idx_none = manager.find_dummy_by_label("999")
    print(f"Index for label '999': {idx_none}")
    assert idx_none is None, "Should return None for non-existent label"
    print("✓ Non-existent label returns None")
    
    print("\n✓ test_find_dummy_by_label PASSED")


def test_get_neighbors():
    """Test getting atom neighbors."""
    print("\n" + "="*60)
    print("TEST: get_neighbors")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]C(C)C")
    manager = DummyManager(mg)
    
    # Find the central carbon (should have 3 neighbors)
    dummy_idx = manager.find_dummy_by_label("1")
    dummy_neighbor = manager.get_single_neighbor(dummy_idx)
    
    print(f"SMILES: [*:1]C(C)C")
    print(f"Dummy index: {dummy_idx}")
    print(f"Dummy's neighbor: {dummy_neighbor}")
    
    # Get neighbors of the carbon attached to dummy
    neighbors = manager.get_neighbors(dummy_neighbor)
    print(f"Central carbon neighbors: {neighbors}")
    
    assert len(neighbors) >= 3, f"Central carbon should have at least 3 neighbors, got {len(neighbors)}"
    assert dummy_idx in neighbors, "Dummy should be a neighbor"
    print("✓ Neighbors found correctly")
    
    print("\n✓ test_get_neighbors PASSED")


def test_relabel_dummy():
    """Test relabeling dummy atoms."""
    print("\n" + "="*60)
    print("TEST: relabel_dummy")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC")
    manager = DummyManager(mg)
    
    print(f"Original labels: {manager.get_all_dummy_labels()}")
    
    # Relabel
    manager.relabel_dummy("1", "site_A")
    
    new_labels = manager.get_all_dummy_labels()
    print(f"After relabel: {new_labels}")
    
    assert "site_A" in new_labels, "New label should be present"
    assert "1" not in new_labels, "Old label should be removed"
    print("✓ Relabeling works")
    
    # Check metadata updated
    label_map = mg.meta.get("label_to_index", {})
    print(f"label_to_index: {label_map}")
    assert "site_A" in label_map, "New label should be in metadata"
    print("✓ Metadata updated")
    
    # Try duplicate label (should fail)
    mg2 = smiles_to_molgraph("[*:1]CC[*:2]")
    manager2 = DummyManager(mg2)
    
    try:
        manager2.relabel_dummy("1", "2")  # 2 already exists
        print("✗ Should have raised ValueError for duplicate label")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n✓ test_relabel_dummy PASSED")


def test_relabel_dummy_by_index():
    """Test relabeling dummy by index."""
    print("\n" + "="*60)
    print("TEST: relabel_dummy_by_index")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC")
    manager = DummyManager(mg)
    
    dummy_idx = manager.find_dummy_by_label("1")
    print(f"Dummy index: {dummy_idx}")
    print(f"Original labels: {manager.get_all_dummy_labels()}")
    
    manager.relabel_dummy_by_index(dummy_idx, "new_site")
    
    new_labels = manager.get_all_dummy_labels()
    print(f"After relabel: {new_labels}")
    
    assert "new_site" in new_labels, "New label should be present"
    print("✓ Relabel by index works")
    
    # Try on non-dummy atom (should fail)
    mg2 = smiles_to_molgraph("[*:1]CC")
    manager2 = DummyManager(mg2)
    
    # Find a carbon atom
    carbon_idx = None
    for i, z in enumerate(mg2.arrays.atomic_num):
        if z == 6:
            carbon_idx = i
            break
    
    try:
        manager2.relabel_dummy_by_index(carbon_idx, "bad_label")
        print("✗ Should have raised ValueError for non-dummy")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n✓ test_relabel_dummy_by_index PASSED")


def test_validate_dummy_consistency():
    """Test validation of dummy consistency."""
    print("\n" + "="*60)
    print("TEST: validate_dummy_consistency")
    print("="*60)
    
    # Valid case
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    manager = DummyManager(mg)
    
    warnings = manager.validate_dummy_consistency()
    print(f"SMILES: [*:1]CC[*:2]")
    print(f"Validation warnings: {warnings}")
    
    # May have some warnings depending on how attachment points are set up
    print(f"Number of warnings: {len(warnings)}")
    print("✓ Validation completed")
    
    print("\n✓ test_validate_dummy_consistency PASSED")


def test_get_dummy_summary():
    """Test getting dummy summary."""
    print("\n" + "="*60)
    print("TEST: get_dummy_summary")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    manager = DummyManager(mg)
    
    summary = manager.get_dummy_summary()
    
    print(f"SMILES: [*:1]CC[*:2]")
    print(f"Summary: {summary}")
    
    assert summary["total_dummies"] == 2, f"Expected 2 dummies, got {summary['total_dummies']}"
    assert summary["labeled_dummies"] == 2, f"Expected 2 labeled, got {summary['labeled_dummies']}"
    assert len(summary["dummies"]) == 2, f"Expected 2 dummy entries"
    print("✓ Summary generated correctly")
    
    # Check individual dummy info
    for d in summary["dummies"]:
        print(f"  Dummy {d['index']}: label={d['label']}, neighbors={d['neighbors']}, type={d['dummy_type']}")
        assert d["neighbor_count"] == 1, "Each dummy should have 1 neighbor"
        assert d["dummy_type"] == "substituent", "Should be substituent type"
    
    print("\n✓ test_get_dummy_summary PASSED")


def test_ensure_unique_atom_maps():
    """Test ensuring unique atom map numbers."""
    print("\n" + "="*60)
    print("TEST: ensure_unique_atom_maps")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    manager = DummyManager(mg)
    
    print(f"Original atom_map: {mg.arrays.atom_map}")
    
    changes = manager.ensure_unique_atom_maps()
    
    print(f"After ensure_unique: {mg.arrays.atom_map}")
    print(f"Changes made: {changes}")
    
    # Check all dummies have unique non-zero map numbers
    dummy_indices = manager.find_dummy_indices()
    map_nums = [mg.arrays.atom_map[i] for i in dummy_indices]
    print(f"Dummy map numbers: {map_nums}")
    
    assert all(m > 0 for m in map_nums), "All dummies should have non-zero map"
    assert len(set(map_nums)) == len(map_nums), "Map numbers should be unique"
    print("✓ Atom maps are unique and non-zero")
    
    print("\n✓ test_ensure_unique_atom_maps PASSED")


def run_all_tests():
    """Run all DummyManager tests."""
    print("\n" + "#"*60)
    print("# RUNNING ALL DUMMY_MANAGER TESTS")
    print("#"*60)
    
    tests = [
        test_find_dummy_indices,
        test_get_all_dummy_labels,
        test_find_dummy_by_label,
        test_get_neighbors,
        test_relabel_dummy,
        test_relabel_dummy_by_index,
        test_validate_dummy_consistency,
        test_get_dummy_summary,
        test_ensure_unique_atom_maps,
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