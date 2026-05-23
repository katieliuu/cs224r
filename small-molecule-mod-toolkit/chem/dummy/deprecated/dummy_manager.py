"""
chem/dummy/dummy_manager.py

High-level dummy atom management utilities for MolGraph molecular graphs.

This module defines the 'DummyManager' class, which provides comprehensive
operations for manipulating dummy atoms (attachment points) in molecular graphs:

- Relabeling existing dummy atoms
- Converting regular atoms to dummies
- Attaching new dummies to molecules
- Replacing dummies with hydrogens
- Batch operations and validation

Adapted to work with the NumPy-first MolGraph structure from core.structs.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from core.structs import MolGraph, MolArrays, AttachmentPointArray, AttachmentKind
from utils.logger import get_logger

logger = get_logger(__name__)


class DummyManager:
    """
    High-level interface for managing dummy atoms in molecular graphs.
    
    This class provides comprehensive operations for dummy atom manipulation,
    including creation, modification, removal, and validation. It ensures
    graph consistency and proper handling of attachment point metadata.
    
    Attributes:
        graph (MolGraph): The molecular graph being managed.
    """
    
    def __init__(self, graph: MolGraph):
        """
        Initialize the dummy manager with a graph.
        
        Args:
            graph: The MolGraph to manage.
        """
        self.graph = graph
        logger.debug(f"DummyManager initialized with {len(graph.arrays.atomic_num)} atoms")
    
    # ==========================================================================
    # QUERY HELPERS
    # ==========================================================================
    
    def find_dummy_indices(self) -> np.ndarray:
        """Find indices of all dummy atoms (atomic_num == 0)."""
        return np.where(self.graph.arrays.atomic_num == 0)[0]
    
    def get_all_dummy_labels(self) -> List[str]:
        """Get all labels assigned to dummy atoms."""
        labels = []
        if self.graph.arrays.attachment_label is not None:
            for i, label in enumerate(self.graph.arrays.attachment_label):
                if label is not None and self.graph.arrays.atomic_num[i] == 0:
                    labels.append(str(label))
        return labels
    
    def find_dummy_by_label(self, label: str) -> Optional[int]:
        """
        Find the index of a dummy atom by its label.
        
        Args:
            label: The label to search for.
            
        Returns:
            Atom index or None if not found.
        """
        if self.graph.arrays.attachment_label is None:
            return None
        
        for i, atom_label in enumerate(self.graph.arrays.attachment_label):
            if atom_label is not None and str(atom_label) == str(label):
                if self.graph.arrays.atomic_num[i] == 0:
                    return i
        return None
    
    def get_neighbors(self, atom_idx: int) -> List[int]:
        """Get indices of all atoms bonded to the given atom."""
        bonds = self.graph.arrays.bonds
        neighbors = []
        for u, v in bonds:
            if u == atom_idx:
                neighbors.append(int(v))
            elif v == atom_idx:
                neighbors.append(int(u))
        return neighbors
    
    def get_single_neighbor(self, atom_idx: int) -> Optional[int]:
        """Get the single neighbor of an atom, or None if multiple/none."""
        neighbors = self.get_neighbors(atom_idx)
        return neighbors[0] if len(neighbors) == 1 else None
    
    def get_bond_index(self, atom1: int, atom2: int) -> Optional[int]:
        """Get the index of the bond between two atoms."""
        bonds = self.graph.arrays.bonds
        for idx, (u, v) in enumerate(bonds):
            if (u == atom1 and v == atom2) or (u == atom2 and v == atom1):
                return idx
        return None
    
    # ==========================================================================
    # RELABELING OPERATIONS
    # ==========================================================================
    
    def relabel_dummy(self, old_label: str, new_label: str) -> None:
        """
        Relabel an existing dummy atom.
        
        Updates the attachment label in the arrays and metadata mappings.
        
        Args:
            old_label: Current label (e.g., "site_1").
            new_label: New label (e.g., "site_2").
            
        Raises:
            ValueError: If the old label doesn't exist or new label already exists.
        """
        logger.debug(f"relabel_dummy({old_label} -> {new_label})")
        
        # Find the dummy with the old label
        dummy_idx = self.find_dummy_by_label(old_label)
        if dummy_idx is None:
            raise ValueError(f"No dummy found with label '{old_label}'")
        
        # Check if new label already exists
        existing_labels = self.get_all_dummy_labels()
        if new_label in existing_labels:
            raise ValueError(f"Label '{new_label}' already exists")
        
        # Update the arrays
        self.graph.arrays.attachment_label[dummy_idx] = new_label
        
        # Update AttachmentPointArray if present
        if self.graph.attachments is not None and self.graph.attachments.label_id is not None:
            att = self.graph.attachments
            for i, idx in enumerate(att.idx):
                if idx == dummy_idx:
                    att.label_id[i] = new_label
                    break
        
        # Update metadata mapping
        label_map = self.graph.meta.setdefault("label_to_index", {})
        if old_label in label_map:
            del label_map[old_label]
        label_map[new_label] = dummy_idx
        
        logger.debug(f"Relabeled dummy {dummy_idx}: {old_label} -> {new_label}")
    
    def relabel_dummy_by_index(self, dummy_idx: int, new_label: str) -> None:
        """
        Relabel a dummy atom by its index.
        
        Args:
            dummy_idx: Index of the dummy atom.
            new_label: New label to assign.
            
        Raises:
            ValueError: If the atom is not a dummy or label already exists.
        """
        if self.graph.arrays.atomic_num[dummy_idx] != 0:
            raise ValueError(f"Atom {dummy_idx} is not a dummy (atomic_num != 0)")
        
        old_label = None
        if self.graph.arrays.attachment_label is not None:
            old_label = self.graph.arrays.attachment_label[dummy_idx]
        
        if old_label is not None:
            self.relabel_dummy(str(old_label), new_label)
        else:
            # Assign label to unlabeled dummy
            existing_labels = self.get_all_dummy_labels()
            if new_label in existing_labels:
                raise ValueError(f"Label '{new_label}' already exists")
            
            # Initialize attachment_label array if needed
            if self.graph.arrays.attachment_label is None:
                n = len(self.graph.arrays.atomic_num)
                self.graph.arrays.attachment_label = np.empty(n, dtype=object)
                self.graph.arrays.attachment_label[:] = None
            
            self.graph.arrays.attachment_label[dummy_idx] = new_label
            
            # Update AttachmentPointArray
            if self.graph.attachments is not None:
                att = self.graph.attachments
                if att.label_id is not None:
                    for i, idx in enumerate(att.idx):
                        if idx == dummy_idx:
                            att.label_id[i] = new_label
                            break
            
            # Update metadata
            self.graph.meta.setdefault("label_to_index", {})[new_label] = dummy_idx
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    
    def validate_dummy_consistency(self) -> List[str]:
        """
        Validate consistency between dummy atoms and attachment points.
        
        Returns:
            List of validation warnings/errors.
        """
        warnings = []
        n_atoms = len(self.graph.arrays.atomic_num)
        
        # Find all dummy atom indices
        dummy_indices = set(self.find_dummy_indices())
        
        # Check AttachmentPointArray consistency
        if self.graph.attachments is not None:
            att = self.graph.attachments
            ap_indices = set(att.idx)
            
            # Check dummy atoms have corresponding attachment points
            for idx in dummy_indices:
                if idx not in ap_indices:
                    warnings.append(f"Dummy atom {idx} missing in AttachmentPointArray")
            
            # Check attachment points reference valid dummy atoms
            for i, idx in enumerate(att.idx):
                if idx >= n_atoms or idx < 0:
                    label = att.label_id[i] if att.label_id is not None else "unlabeled"
                    warnings.append(f"AttachmentPoint {label} references out-of-bounds atom {idx}")
                elif idx not in dummy_indices:
                    label = att.label_id[i] if att.label_id is not None else "unlabeled"
                    warnings.append(f"AttachmentPoint {label} references non-dummy atom {idx}")
        
        # Check label consistency
        if self.graph.arrays.attachment_label is not None:
            for idx in dummy_indices:
                if idx < n_atoms:
                    label = self.graph.arrays.attachment_label[idx]
                    if label is None:
                        warnings.append(f"Dummy atom {idx} has no attachment label")
        
        # Check metadata consistency
        label_map = self.graph.meta.get("label_to_index", {})
        for label, idx in label_map.items():
            if idx >= n_atoms or idx < 0:
                warnings.append(f"Metadata label '{label}' points to out-of-bounds atom {idx}")
            elif idx not in dummy_indices:
                warnings.append(f"Metadata label '{label}' points to non-dummy atom {idx}")
        
        # Check atom map numbers
        if self.graph.arrays.atom_map is not None:
            map_numbers = {}
            for idx in dummy_indices:
                if idx < n_atoms:
                    map_num = self.graph.arrays.atom_map[idx]
                    if map_num == 0:
                        warnings.append(f"Dummy atom {idx} has atom_map=0 (renders as '*' not '[*:N]')")
                    elif map_num in map_numbers:
                        warnings.append(f"Duplicate atom_map {map_num} on dummies {map_numbers[map_num]} and {idx}")
                    else:
                        map_numbers[map_num] = idx
        
        return warnings
    
    def get_dummy_summary(self) -> Dict:
        """
        Get a summary of all dummy atoms and their properties.
        
        Returns:
            Summary dict with dummy atom information.
        """
        dummy_indices = self.find_dummy_indices()
        
        summary = {
            "total_dummies": len(dummy_indices),
            "labeled_dummies": 0,
            "dummies": []
        }
        
        for idx in dummy_indices:
            idx = int(idx)
            neighbors = self.get_neighbors(idx)
            single_neighbor = self.get_single_neighbor(idx)
            
            # Get label
            label = None
            if self.graph.arrays.attachment_label is not None:
                label = self.graph.arrays.attachment_label[idx]
                if label is not None:
                    summary["labeled_dummies"] += 1
            
            # Get atom map number
            atom_map = 0
            if self.graph.arrays.atom_map is not None:
                atom_map = int(self.graph.arrays.atom_map[idx])
            
            # Determine dummy type
            if len(neighbors) >= 2:
                dummy_type = "insertion"
            else:
                dummy_type = "substituent"
            
            summary["dummies"].append({
                "index": idx,
                "label": label,
                "neighbors": neighbors,
                "neighbor_count": len(neighbors),
                "single_neighbor": single_neighbor,
                "atom_map_num": atom_map,
                "dummy_type": dummy_type,
            })
        
        return summary
    
    # ==========================================================================
    # CLEANUP OPERATIONS
    # ==========================================================================
    
    def ensure_unique_atom_maps(self) -> Dict[int, Tuple[int, int]]:
        """
        Ensure all dummy atoms have unique atom map numbers.
        
        Returns:
            Dict mapping atom index to (old_map, new_map) for changed atoms.
        """
        logger.debug("ensure_unique_atom_maps() called")
        
        dummy_indices = self.find_dummy_indices()
        if len(dummy_indices) == 0:
            return {}
        
        changes = {}
        
        # Initialize atom_map if needed
        if self.graph.arrays.atom_map is None:
            n = len(self.graph.arrays.atomic_num)
            self.graph.arrays.atom_map = np.zeros(n, dtype=np.int32)
        
        # Collect all used map numbers
        all_map_nums = self.graph.arrays.atom_map
        used_numbers = set(all_map_nums[all_map_nums > 0])
        
        next_number = 1
        for idx in dummy_indices:
            idx = int(idx)
            current_map = int(self.graph.arrays.atom_map[idx])
            
            if current_map == 0 or current_map in used_numbers:
                # Find next available number
                while next_number in used_numbers:
                    next_number += 1
                
                old_map = self.graph.arrays.atom_map[idx]
                self.graph.arrays.atom_map[idx] = next_number
                changes[idx] = (int(old_map), next_number)
                used_numbers.add(next_number)
                next_number += 1
            else:
                used_numbers.add(current_map)
        
        logger.debug(f"Updated atom maps for {len(changes)} dummies")
        return changes
    
    def cleanup_orphaned_attachments(self) -> int:
        """
        Remove attachment point entries that reference deleted atoms.
        
        Returns:
            Number of orphaned attachments removed.
        """
        if self.graph.attachments is None:
            return 0
        
        n_atoms = len(self.graph.arrays.atomic_num)
        att = self.graph.attachments
        
        # Find valid entries
        valid_mask = (att.idx >= 0) & (att.idx < n_atoms)
        if att.target is not None:
            valid_mask &= (att.target >= -1) & (att.target < n_atoms)
        
        n_removed = int(np.sum(~valid_mask))
        
        if n_removed > 0:
            # Filter arrays
            att.idx = att.idx[valid_mask]
            att.kind = att.kind[valid_mask]
            att.target = att.target[valid_mask] if att.target is not None else None
            if att.label_id is not None:
                att.label_id = att.label_id[valid_mask]
            if att.is_insertion is not None:
                att.is_insertion = att.is_insertion[valid_mask]
        
        logger.debug(f"Removed {n_removed} orphaned attachments")
        return n_removed


logger.debug("dummy_manager.py loaded")