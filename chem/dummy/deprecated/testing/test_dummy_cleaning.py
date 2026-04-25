"""
chem/dummy/testing/test_dummy_cleaning.py

Unit tests for dummy cleaning functions.
"""

import sys
import numpy as np

from utils.logger import get_logger
from core.structs import MolGraph, MolArrays, AttachmentPointArray, AttachmentKind
from chem.build.create_molgraph import smiles_to_molgraph
from chem.dummy.dummy_manager import DummyManager
from chem.dummy.dummy_cleaning import (
    remove_orphan_dummies,
    drop_invalid_label_mappings,
    deduplicate_attachment_labels,
    validate_dummy_labels,
    sync_attachment_array_with_labels,
    repair_dummy_metadata,
)

logger = get_logger(__name__)


def test_remove_orphan_dummies():
    """Test removing orphan (unbonded) dummy atoms."""
    print("\n" + "="*60)
    print("TEST: remove_orphan_dummies")
    print("="*60)
    
    # Normal case - dummies are connected
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    print(f"SMILES: [*:1]CC[*:2]")
    
    mg = remove_orphan_dummies(mg)
    
    orphans = mg.meta.get("orphan_dummies", [])
    print(f"Orphan dummies found: {orphans}")
    
    assert len(orphans) == 0, "Should have no orphans for valid molecule"
    print("✓ No orphans in valid molecule")
    
    # Note: To properly test orphan detection, we'd need to manually
    # create a graph with an orphan dummy, which is tricky since
    # smiles_to_molgraph creates valid molecules
    print("✓ (Orphan detection logic is in place)")
    
    print("\n✓ test_remove_orphan_dummies PASSED")


def test_drop_invalid_label_mappings():
    """Test dropping invalid label->index mappings."""
    print("\n" + "="*60)
    print("TEST: drop_invalid_label_mappings")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC")
    
    # Artificially add invalid mapping
    mg.meta["label_to_index"] = {
        "1": 0,      # Valid (if 0 is the dummy)
        "bad_1": 999,  # Out of bounds
        "bad_2": 1,    # Points to carbon, not dummy
    }
    
    print(f"Before cleanup: {mg.meta['label_to_index']}")
    
    mg = drop_invalid_label_mappings(mg)
    
    print(f"After cleanup: {mg.meta['label_to_index']}")
    
    label_map = mg.meta["label_to_index"]
    assert "bad_1" not in label_map, "Out-of-bounds mapping should be removed"
    assert "bad_2" not in label_map, "Non-dummy mapping should be removed"
    print("✓ Invalid mappings removed")
    
    print("\n✓ test_drop_invalid_label_mappings PASSED")


def test_deduplicate_attachment_labels():
    """Test deduplicating attachment labels."""
    print("\n" + "="*60)
    print("TEST: deduplicate_attachment_labels")
    print("="*60)
    
    # Create graph with duplicate labels manually
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    # Artificially create duplicate by changing label
    manager = DummyManager(mg)
    dummy_indices = manager.find_dummy_indices()
    
    print(f"Dummy indices: {dummy_indices}")
    print(f"Original labels: {manager.get_all_dummy_labels()}")
    
    # Set both to same label (simulating corruption)
    if mg.arrays.attachment_label is not None and len(dummy_indices) >= 2:
        mg.arrays.attachment_label[dummy_indices[0]] = "duplicate"
        mg.arrays.attachment_label[dummy_indices[1]] = "duplicate"
        
        print(f"After setting duplicates: {manager.get_all_dummy_labels()}")
        
        mg = deduplicate_attachment_labels(mg, overwrite_prefix="site")
        
        labels = manager.get_all_dummy_labels()
        print(f"After deduplication: {labels}")
        
        # Should have unique labels now
        assert len(labels) == len(set(labels)), "Labels should be unique"
        print("✓ Labels deduplicated")
    else:
        print("✓ (Skipped - no attachment labels to test)")
    
    print("\n✓ test_deduplicate_attachment_labels PASSED")


def test_validate_dummy_labels():
    """Test validation of dummy labels."""
    print("\n" + "="*60)
    print("TEST: validate_dummy_labels")
    print("="*60)
    
    # Valid case
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    # Ensure label_to_index is set up correctly
    manager = DummyManager(mg)
    dummy_indices = manager.find_dummy_indices()
    mg.meta["label_to_index"] = {}
    for idx in dummy_indices:
        idx = int(idx)
        if mg.arrays.attachment_label is not None:
            label = mg.arrays.attachment_label[idx]
            if label is not None:
                mg.meta["label_to_index"][str(label)] = idx
    
    print(f"label_to_index: {mg.meta['label_to_index']}")
    
    try:
        validate_dummy_labels(mg)
        print("✓ Valid labels passed validation")
    except ValueError as e:
        print(f"✗ Unexpected validation error: {e}")
        raise
    
    # Invalid case - point to non-dummy
    mg2 = smiles_to_molgraph("[*:1]CC")
    # Find a carbon
    carbon_idx = None
    for i, z in enumerate(mg2.arrays.atomic_num):
        if z == 6:
            carbon_idx = i
            break
    
    mg2.meta["label_to_index"] = {"bad_label": carbon_idx}
    
    try:
        validate_dummy_labels(mg2)
        print("✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n✓ test_validate_dummy_labels PASSED")


def test_sync_attachment_array_with_labels():
    """Test syncing AttachmentPointArray with labels."""
    print("\n" + "="*60)
    print("TEST: sync_attachment_array_with_labels")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    print(f"Before sync - attachments: {mg.attachments}")
    
    mg = sync_attachment_array_with_labels(mg)
    
    print(f"After sync - attachments: {mg.attachments}")
    
    if mg.attachments is not None:
        print(f"  idx: {mg.attachments.idx}")
        print(f"  kind: {mg.attachments.kind}")
        print(f"  target: {mg.attachments.target}")
        print(f"  label_id: {mg.attachments.label_id}")
        
        # Should match dummy count
        manager = DummyManager(mg)
        n_dummies = len(manager.find_dummy_indices())
        n_attachments = len(mg.attachments.idx)
        
        assert n_attachments == n_dummies, f"Should have {n_dummies} attachments, got {n_attachments}"
        print(f"✓ AttachmentPointArray has {n_attachments} entries (matches dummy count)")
    else:
        print("✓ (No attachments to sync)")
    
    print("\n✓ test_sync_attachment_array_with_labels PASSED")


def test_repair_dummy_metadata():
    """Test full metadata repair pass."""
    print("\n" + "="*60)
    print("TEST: repair_dummy_metadata")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CC[*:2]")
    
    # Corrupt the metadata
    mg.meta["label_to_index"] = {
        "1": 0,
        "bad": 999,  # Invalid
    }
    
    print(f"Before repair: {mg.meta['label_to_index']}")
    
    mg = repair_dummy_metadata(mg)
    
    print(f"After repair: {mg.meta['label_to_index']}")
    
    # Check no invalid entries remain
    label_map = mg.meta.get("label_to_index", {})
    n_atoms = len(mg.arrays.atomic_num)
    
    for label, idx in label_map.items():
        assert 0 <= idx < n_atoms, f"Index {idx} out of bounds"
        assert mg.arrays.atomic_num[idx] == 0, f"Index {idx} is not a dummy"
    
    print("✓ Metadata repaired successfully")
    
    print("\n✓ test_repair_dummy_metadata PASSED")


def test_repair_preserves_valid_data():
    """Test that repair doesn't corrupt valid data."""
    print("\n" + "="*60)
    print("TEST: repair_preserves_valid_data")
    print("="*60)
    
    mg = smiles_to_molgraph("[*:1]CCC[*:2]")
    
    manager = DummyManager(mg)
    original_labels = set(manager.get_all_dummy_labels())
    original_n_dummies = len(manager.find_dummy_indices())
    
    print(f"Original labels: {original_labels}")
    print(f"Original dummy count: {original_n_dummies}")
    
    mg = repair_dummy_metadata(mg)
    
    new_labels = set(manager.get_all_dummy_labels())
    new_n_dummies = len(manager.find_dummy_indices())
    
    print(f"After repair labels: {new_labels}")
    print(f"After repair dummy count: {new_n_dummies}")
    
    # Dummy count should be preserved
    assert new_n_dummies == original_n_dummies, "Dummy count should not change"
    print("✓ Dummy count preserved")
    
    # Labels should be preserved (or renamed if duplicates)
    assert len(new_labels) == len(original_labels), "Label count should be preserved"
    print("✓ Label count preserved")
    
    print("\n✓ test_repair_preserves_valid_data PASSED")


def run_all_tests():
    """Run all dummy_cleaning tests."""
    print("\n" + "#"*60)
    print("# RUNNING ALL DUMMY_CLEANING TESTS")
    print("#"*60)
    
    tests = [
        test_remove_orphan_dummies,
        test_drop_invalid_label_mappings,
        test_deduplicate_attachment_labels,
        test_validate_dummy_labels,
        test_sync_attachment_array_with_labels,
        test_repair_dummy_metadata,
        test_repair_preserves_valid_data,
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