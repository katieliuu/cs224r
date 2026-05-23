"""
chem/dummy/dummy_cleaning.py

Cleaning and repair utilities for dummy atom metadata in MolGraph objects.

Functions for:
- Removing orphan dummies (not bonded to anything)
- Cleaning invalid label mappings
- Deduplicating attachment labels
- Validating dummy atom consistency
"""

from __future__ import annotations
from typing import Set
import numpy as np

from core.structs import MolGraph, AttachmentPointArray, AttachmentKind
from chem.dummy.dummy_manager import DummyManager
from utils.logger import get_logger

logger = get_logger(__name__)


def remove_orphan_dummies(graph: MolGraph) -> MolGraph:
    """
    Removes dummy atoms that are not bonded to any other atom.
    
    Note: This modifies indices, so other arrays need to be updated accordingly.
    For now, this just identifies orphans and logs them.
    
    Args:
        graph: MolGraph object
        
    Returns:
        MolGraph with orphaned dummies identified (actual removal requires GraphEditor)
    """
    logger.debug("remove_orphan_dummies() called")
    
    manager = DummyManager(graph)
    dummy_indices = set(manager.find_dummy_indices())
    
    # Find which dummies are connected
    connected = set()
    bonds = graph.arrays.bonds
    for u, v in bonds:
        if u in dummy_indices:
            connected.add(int(u))
        if v in dummy_indices:
            connected.add(int(v))
    
    orphans = dummy_indices - connected
    
    if orphans:
        logger.warning(f"Found {len(orphans)} orphan dummy atoms: {orphans}")
        # Store in meta for later cleanup
        graph.meta["orphan_dummies"] = list(orphans)
    
    return graph


def drop_invalid_label_mappings(graph: MolGraph) -> MolGraph:
    """
    Removes any label->index mappings that point to invalid or deleted atoms.
    
    Args:
        graph: MolGraph object
        
    Returns:
        MolGraph with cleaned label_to_index map
    """
    logger.debug("drop_invalid_label_mappings() called")
    
    n_atoms = len(graph.arrays.atomic_num)
    label_map = graph.meta.get("label_to_index", {})
    
    valid = {}
    dropped = []
    
    for label, idx in label_map.items():
        if 0 <= idx < n_atoms:
            # Also check it's actually a dummy
            if graph.arrays.atomic_num[idx] == 0:
                valid[label] = idx
            else:
                dropped.append(f"{label} (atom {idx} is not a dummy)")
        else:
            dropped.append(f"{label} (index {idx} out of bounds)")
    
    if dropped:
        logger.warning(f"Dropped invalid label mappings: {dropped}")
    
    graph.meta["label_to_index"] = valid
    return graph


def deduplicate_attachment_labels(
    graph: MolGraph,
    overwrite_prefix: str = "site"
) -> MolGraph:
    """
    Ensures all attachment point labels are unique. If duplicates are found,
    renames later ones using an incrementing counter.
    
    Args:
        graph: MolGraph object
        overwrite_prefix: Prefix to use for auto-generated new labels.
        
    Returns:
        MolGraph with deduplicated dummy labels
    """
    logger.debug("deduplicate_attachment_labels() called")
    
    if graph.arrays.attachment_label is None:
        return graph
    
    seen: dict = {}  # label -> first index that has it
    n_atoms = len(graph.arrays.atomic_num)
    label_map = graph.meta.setdefault("label_to_index", {})
    
    # Collect all existing labels (including non-dummies for conflict detection)
    all_labels = set()
    for label in graph.arrays.attachment_label:
        if label is not None:
            all_labels.add(str(label))
    
    renamed = []
    
    for i in range(n_atoms):
        if graph.arrays.atomic_num[i] != 0:
            continue  # Skip non-dummies
        
        label = graph.arrays.attachment_label[i]
        if label is None:
            continue
        
        label_str = str(label)
        
        if label_str not in seen:
            seen[label_str] = i
        else:
            # Duplicate found - generate new label
            counter = 1
            new_label = f"{overwrite_prefix}_{counter}"
            while new_label in all_labels or new_label in seen:
                counter += 1
                new_label = f"{overwrite_prefix}_{counter}"
            
            # Update arrays
            old_label = label_str
            graph.arrays.attachment_label[i] = new_label
            seen[new_label] = i
            all_labels.add(new_label)
            
            # Update label_to_index
            label_map[new_label] = i
            
            renamed.append(f"atom {i}: {old_label} -> {new_label}")
    
    if renamed:
        logger.info(f"Deduplicated labels: {renamed}")
    
    return graph


def validate_dummy_labels(graph: MolGraph) -> None:
    """
    Checks that all labels in label_to_index point to valid dummy atoms.
    
    Args:
        graph: MolGraph object
        
    Raises:
        ValueError: If any label maps to a non-dummy atom.
    """
    logger.debug("validate_dummy_labels() called")
    
    atomic_nums = graph.arrays.atomic_num
    label_map = graph.meta.get("label_to_index", {})
    
    errors = []
    for label, idx in label_map.items():
        if idx >= len(atomic_nums):
            errors.append(f"Label '{label}' points to out-of-bounds index {idx}")
        elif atomic_nums[idx] != 0:
            errors.append(f"Label '{label}' points to non-dummy atom {idx} (Z={atomic_nums[idx]})")
    
    if errors:
        raise ValueError("Invalid dummy label mappings:\n" + "\n".join(errors))
    
    logger.debug("validate_dummy_labels() passed")


def sync_attachment_array_with_labels(graph: MolGraph) -> MolGraph:
    """
    Synchronize the AttachmentPointArray with the attachment_label array.
    
    Rebuilds the AttachmentPointArray to match what's in the MolArrays.
    
    Args:
        graph: MolGraph object
        
    Returns:
        MolGraph with synchronized AttachmentPointArray
    """
    logger.debug("sync_attachment_array_with_labels() called")
    
    manager = DummyManager(graph)
    dummy_indices = manager.find_dummy_indices()
    
    if len(dummy_indices) == 0:
        graph.attachments = None
        return graph
    
    # Build new AttachmentPointArray
    idx_list = []
    kind_list = []
    target_list = []
    label_list = []
    
    for dummy_idx in dummy_indices:
        dummy_idx = int(dummy_idx)
        idx_list.append(dummy_idx)
        kind_list.append(AttachmentKind.DUMMY)
        
        # Get target (neighbor)
        neighbors = manager.get_neighbors(dummy_idx)
        # Filter to non-dummy neighbors
        real_neighbors = [n for n in neighbors if graph.arrays.atomic_num[n] != 0]
        target = real_neighbors[0] if len(real_neighbors) == 1 else -1
        target_list.append(target)
        
        # Get label
        label = None
        if graph.arrays.attachment_label is not None:
            label = graph.arrays.attachment_label[dummy_idx]
        label_list.append(label)
    
    graph.attachments = AttachmentPointArray(
        idx=np.array(idx_list, dtype=np.int32),
        kind=np.array(kind_list, dtype=np.int8),
        target=np.array(target_list, dtype=np.int32),
        label_id=np.array(label_list, dtype=object) if any(l is not None for l in label_list) else None,
    )
    
    logger.debug(f"Synced AttachmentPointArray with {len(idx_list)} entries")
    return graph


def repair_dummy_metadata(graph: MolGraph) -> MolGraph:
    """
    Runs a full repair pass:
        - Removes orphan dummies (identification)
        - Cleans invalid label mappings
        - Deduplicates labels
        - Syncs AttachmentPointArray
    
    Args:
        graph: MolGraph object
        
    Returns:
        Cleaned MolGraph
    """
    logger.debug("repair_dummy_metadata() called")
    
    graph = remove_orphan_dummies(graph)
    graph = drop_invalid_label_mappings(graph)
    graph = deduplicate_attachment_labels(graph)
    graph = sync_attachment_array_with_labels(graph)
    
    logger.debug("repair_dummy_metadata() complete")
    return graph


logger.debug("dummy_cleaning.py loaded")