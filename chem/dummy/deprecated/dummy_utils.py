"""
chem/dummy/dummy_utils.py

Utility functions for dummy atom management using DummyManager.
Provides convenience functions for common operations.
"""

from __future__ import annotations
from typing import List, Optional

from core.structs import MolGraph, AttachmentPointArray, AttachmentKind
from chem.dummy.dummy_manager import DummyManager
from utils.logger import get_logger

logger = get_logger(__name__)


def relabel_dummy_safe(graph, new_label: str, old_label: Optional[str] = None):
    manager = DummyManager(graph)

    if old_label is None:
        labels = manager.get_all_dummy_labels()
        if not labels:
            raise ValueError("No dummy atoms found to relabel")

        def as_int(lbl):
            # lbl may already be str because DummyManager casts to str,
            # but handle robustly anyway.
            try:
                return int(lbl)
            except Exception:
                return None

        numeric = [(as_int(l), l) for l in labels]
        numeric = [(n, l) for (n, l) in numeric if n is not None]

        if numeric:
            # pick highest numeric label (e.g., 5)
            old_label = max(numeric, key=lambda x: x[0])[1]
        else:
            # fallback: first label (non-numeric)
            old_label = labels[0]

    manager.relabel_dummy(old_label, new_label)
    return graph


def relabel_dummy(graph: MolGraph, old_label: str, new_label: str) -> MolGraph:
    """
    Relabel a dummy atom from old_label to new_label.
    
    Args:
        graph: MolGraph object
        old_label: Current label
        new_label: New label
    
    Returns:
        Modified MolGraph object
    """
    manager = DummyManager(graph)
    manager.relabel_dummy(old_label, new_label)
    return graph


def get_labeled_dummies(graph: MolGraph) -> List[dict]:
    """
    Get all labeled dummy attachment points.

    Returns:
        List of dicts: {"idx": int, "label": str, "target": int (optional)}
    """
    # Prefer the authoritative attachment-point container if present
    if getattr(graph, "attachments", None) is not None:
        ap = graph.attachments
        out = []

        # labels can come from ap.label_id (per attachment point)
        # or from arrays.attachment_label (per atom) as a fallback.
        for i, idx in enumerate(ap.idx):
            idx = int(idx)

            label = None
            if ap.label_id is not None:
                label = ap.label_id[i]

            if label is None and graph.arrays.attachment_label is not None:
                label = graph.arrays.attachment_label[idx]

            if label is None:
                continue

            d = {"idx": idx, "label": str(label)}
            if getattr(ap, "target", None) is not None:
                d["target"] = int(ap.target[i])
            out.append(d)

        return out

    # Fallback: derive from DummyManager summary (still normalize keys)
    manager = DummyManager(graph)
    summary = manager.get_dummy_summary()

    out = []
    for d in summary["dummies"]:
        if d.get("label") is None:
            continue
        out.append({
            "idx": int(d["index"]),
            "label": str(d["label"]),
            "target": d.get("single_neighbor"),
        })
    return out




def get_dummy_labels(graph: MolGraph) -> List[str]:
    """
    Get all dummy labels in the graph.
    
    Args:
        graph: MolGraph object
    
    Returns:
        List of dummy labels
    """
    manager = DummyManager(graph)
    return manager.get_all_dummy_labels()


def find_available_dummy_label(graph: MolGraph, base: str = "site") -> str:
    """
    Find an available dummy label.
    
    Args:
        graph: MolGraph object
        base: Base name for label
    
    Returns:
        Available label like "site_1", "site_2", etc.
    """
    manager = DummyManager(graph)
    existing_labels = set(manager.get_all_dummy_labels())
    
    counter = 1
    while f"{base}_{counter}" in existing_labels:
        counter += 1
    
    return f"{base}_{counter}"


def get_dummy_target(graph: MolGraph, label: str) -> Optional[int]:
    """
    Get the target atom (neighbor) of a labeled dummy.
    
    Args:
        graph: MolGraph object
        label: Dummy label
    
    Returns:
        Index of the target atom, or None if not found
    """
    manager = DummyManager(graph)
    dummy_idx = manager.find_dummy_by_label(label)
    
    if dummy_idx is None:
        return None
    
    return manager.get_single_neighbor(dummy_idx)


def validate_dummies(graph: MolGraph) -> List[str]:
    """
    Validate dummy atom consistency and return any warnings.
    
    Args:
        graph: MolGraph object
    
    Returns:
        List of warning/error messages
    """
    manager = DummyManager(graph)
    return manager.validate_dummy_consistency()


def ensure_unique_dummy_maps(graph: MolGraph) -> MolGraph:
    """
    Ensure all dummy atoms have unique atom map numbers.
    
    Args:
        graph: MolGraph object
    
    Returns:
        Modified MolGraph object
    """
    manager = DummyManager(graph)
    manager.ensure_unique_atom_maps()
    return graph


logger.debug("dummy_utils.py loaded")