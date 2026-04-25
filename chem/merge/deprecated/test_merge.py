
"""
test_merge.py

Intense tests for chem/merge.py (new MolGraph/MolArrays merger).

Focus:
- substituent merge (1 heavy neighbor on each side)
- insertion merge (2 heavy neighbors on one side, 1 on the other)
- explicit H node removal on anchors (atomic_num == 1)
- dummy deletion correctness (atomic_num==0 removed, bonds filtered)
- stereochemistry preservation via union + default fill (bond_stereo present if either side has it)
- fragments union with index offset and deletion remapping
- meta merge logging (merge_log) + smiles placeholder behavior

This test suite intentionally avoids relying on legacy GraphMerger tests except for spirit.

Run:
  python -m chem.dummy.testing.test_merge   (adjust module path)
or:
  python test_merge.py

Note: These tests assume the project can import:
  - core.structs (MolGraph, MolArrays, Fragment, etc.)
  - chem.merge (merge_by_labels, resolve_site)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional

import numpy as np

from core.structs import MolArrays, MolGraph, Fragment, BondType, BondStereo

from chem.merge.merge import merge_by_labels, resolve_site


# -----------------------------
# Minimal graph constructors
# -----------------------------

def _base_graph(
    atomic_num: np.ndarray,
    bonds: np.ndarray,
    bond_type: np.ndarray,
    *,
    attachment_label: Optional[np.ndarray] = None,
    bond_stereo: Optional[np.ndarray] = None,
    meta_smiles: Optional[str] = None,
    fragments: Optional[List[Fragment]] = None,
) -> MolGraph:
    n = int(atomic_num.shape[0])
    if attachment_label is None:
        attachment_label = np.empty(n, dtype=object)
        attachment_label[:] = None

    arrays = MolArrays(
        atomic_num=atomic_num.astype(np.int16, copy=False),
        formal_charge=np.zeros(n, dtype=np.int8),
        bonds=bonds.astype(np.int32, copy=False),
        bond_type=bond_type.astype(np.int8, copy=False),
        isotope=np.zeros(n, dtype=np.int16),
        is_aromatic=np.zeros(n, dtype=bool),
        hybridization=np.zeros(n, dtype=np.int8),
        chiral_tag=np.zeros(n, dtype=np.int8),
        cip_code=np.zeros(n, dtype=np.int8),
        atom_map=np.zeros(n, dtype=np.int32),
        attachment_label=attachment_label,
        explicit_h=np.zeros(n, dtype=np.int8),
        implicit_h=np.zeros(n, dtype=np.int8),
        partial_charge=np.zeros(n, dtype=np.float32),
        pos=None,
        is_conjugated=np.zeros(int(bonds.shape[0]), dtype=bool),
        is_in_ring=np.zeros(int(bonds.shape[0]), dtype=bool),
        bond_dir=np.zeros(int(bonds.shape[0]), dtype=np.int8),
        bond_stereo=bond_stereo,
        bond_resonance_type=np.zeros(int(bonds.shape[0]), dtype=np.int8),
        total_charge=0,
        multiplicity=None,
    )

    meta = {}
    if meta_smiles is not None:
        meta["smiles"] = meta_smiles

    g = MolGraph(
        arrays=arrays,
        attachments=None,
        editability=None,
        fragments=fragments or [],
        resonance_systems=[],  # not used in new merge
        meta=meta,
    )
    return g


def make_substituent_graph(label: str, *, meta_smiles: str = "", with_bond_stereo: bool = False) -> MolGraph:
    # atoms: [dummy,label] - [C]
    atomic_num = np.array([0, 6], dtype=np.int16)
    bonds = np.array([[0, 1]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE], dtype=np.int8)
    attach = np.array([label, None], dtype=object)
    bond_stereo = np.array([BondStereo.NONE], dtype=np.int8) if with_bond_stereo else None

    # fragment referencing both dummy + carbon (dummy should be dropped after deletion)
    frag = Fragment(
        atom_indices=np.array([0, 1], dtype=np.int64),
        attachment_indices=np.array([0], dtype=np.int64),
        smiles=None,
    )
    return _base_graph(
        atomic_num, bonds, bond_type,
        attachment_label=attach,
        bond_stereo=bond_stereo,
        meta_smiles=meta_smiles,
        fragments=[frag],
    )


def make_substituent_graph_with_explicit_h(label: str, *, meta_smiles: str = "") -> MolGraph:
    # atoms: [dummy,label] - [C] - [H]
    atomic_num = np.array([0, 6, 1], dtype=np.int16)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE], dtype=np.int8)
    attach = np.array([label, None, None], dtype=object)
    return _base_graph(atomic_num, bonds, bond_type, attachment_label=attach, meta_smiles=meta_smiles)


def make_insertion_graph(label: str, *, meta_smiles: str = "") -> MolGraph:
    # atoms: C0 - dummy(label) - C2
    atomic_num = np.array([6, 0, 6], dtype=np.int16)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE], dtype=np.int8)
    attach = np.array([None, label, None], dtype=object)

    frag = Fragment(
        atom_indices=np.array([0, 1, 2], dtype=np.int64),
        attachment_indices=np.array([1], dtype=np.int64),
        smiles=None,
    )
    return _base_graph(atomic_num, bonds, bond_type, attachment_label=attach, meta_smiles=meta_smiles, fragments=[frag])


# -----------------------------
# Assertions helpers
# -----------------------------

def assert_no_dummies(g: MolGraph) -> None:
    assert np.all(g.arrays.atomic_num != 0), f"Found dummy atoms: {np.where(g.arrays.atomic_num == 0)[0].tolist()}"


def bond_exists(g: MolGraph, u: int, v: int) -> bool:
    a, b = (u, v) if u < v else (v, u)
    if g.arrays.bonds.size == 0:
        return False
    bonds = g.arrays.bonds
    return bool(np.any((bonds[:, 0] == a) & (bonds[:, 1] == b)))


def degree(g: MolGraph, idx: int) -> int:
    if g.arrays.bonds.size == 0:
        return 0
    b = g.arrays.bonds
    return int(np.sum(b[:, 0] == idx) + np.sum(b[:, 1] == idx))


# -----------------------------
# Tests: resolve_site
# -----------------------------

def test_resolve_site_substituent() -> None:
    g = make_substituent_graph("site_A", meta_smiles="[*:1]CC")
    s = resolve_site(g, "site_A")
    assert s.mode == "substituent"
    assert s.dummy_idx == 0
    assert s.heavy_neighbors.tolist() == [1]
    assert s.h_neighbors.size == 0


def test_resolve_site_insertion() -> None:
    g = make_insertion_graph("ins", meta_smiles="C[*:1]C")
    s = resolve_site(g, "ins")
    assert s.mode == "insertion"
    assert s.heavy_neighbors.tolist() == [0, 2]


def test_resolve_site_errors_on_missing_label() -> None:
    g = make_substituent_graph("x", meta_smiles="[*:1]CC")
    try:
        _ = resolve_site(g, "missing")
        assert False, "Expected ValueError on missing label"
    except ValueError:
        pass


# -----------------------------
# Tests: substituent merge
# -----------------------------

def test_merge_substituent_basic() -> None:
    a = make_substituent_graph("A", meta_smiles="[*:1]CC", with_bond_stereo=True)
    b = make_substituent_graph("B", meta_smiles="[*:2]O", with_bond_stereo=False)

    merged = merge_by_labels(a, b, label_a="A", label_b="B", bond_type_code=BondType.SINGLE, validate_with_rdkit=False)

    # Two heavy atoms total: carbon from A + carbon(?) from B is O? Actually B heavy is carbon at idx1 in our constructor
    # Our constructor uses [dummy, C], so both heavies are carbon.
    assert merged.arrays.atomic_num.shape[0] == 2
    assert_no_dummies(merged)

    # There should be exactly one bond connecting them
    assert merged.arrays.bonds.shape[0] == 1
    assert bond_exists(merged, 0, 1)

    # Stereo array should be preserved because A had bond_stereo (even though its only bond was deleted)
    assert merged.arrays.bond_stereo is not None, "bond_stereo should exist via union+default fill"
    assert merged.arrays.bond_stereo.shape[0] == 1
    assert int(merged.arrays.bond_stereo[0]) == int(BondStereo.NONE)

    # Fragments from both graphs should be present (2 fragments), with dummy indices dropped by delete mapping
    assert len(merged.fragments) == 2
    for fr in merged.fragments:
        assert np.all(fr.atom_indices >= 0)
        assert np.all(fr.atom_indices < merged.arrays.atomic_num.shape[0])
        # No dummy indices should remain in fragment atom_indices
        assert np.all(merged.arrays.atomic_num[fr.atom_indices] != 0)

    # Smiles placeholder because validate_with_rdkit=False
    assert isinstance(merged.meta.get("smiles"), str)
    assert merged.meta["smiles"].startswith("MERGED(")

    # Merge log appended and includes original smiles strings
    assert "merge_log" in merged.meta
    assert isinstance(merged.meta["merge_log"], list)
    assert len(merged.meta["merge_log"]) >= 1
    last = merged.meta["merge_log"][-1]
    assert last.get("type") == "merge"
    assert last.get("mode") == "substituent"
    assert "[*:1]CC" in last.get("message", "")
    assert "[*:2]O" in last.get("message", "")


def test_merge_substituent_removes_explicit_h_node() -> None:
    a = make_substituent_graph_with_explicit_h("A", meta_smiles="[*:1]C([H])")
    b = make_substituent_graph("B", meta_smiles="[*:2]CC")

    merged = merge_by_labels(a, b, label_a="A", label_b="B", validate_with_rdkit=False)

    # Should not contain hydrogen atom nodes
    assert not np.any(merged.arrays.atomic_num == 1), "Explicit H node should be removed if present on anchor"
    assert_no_dummies(merged)

    # Should have 2 heavy atoms and 1 bond
    assert merged.arrays.atomic_num.shape[0] == 2
    assert merged.arrays.bonds.shape[0] == 1
    assert bond_exists(merged, 0, 1)


# -----------------------------
# Tests: insertion merge
# -----------------------------

def test_merge_insertion_basic_star_topology() -> None:
    a = make_insertion_graph("insA", meta_smiles="C[*:1]C")
    b = make_substituent_graph("subB", meta_smiles="[*:2]C")

    merged = merge_by_labels(a, b, label_a="insA", label_b="subB", validate_with_rdkit=False)

    assert_no_dummies(merged)

    # Expected atoms: A has 2 carbons after dummy deletion, B has 1 carbon after dummy deletion => 3 atoms
    assert merged.arrays.atomic_num.shape[0] == 3
    assert merged.arrays.bonds.shape[0] == 2

    # In insertion merge implementation: center is single-anchor fragment side (B), connected to both A carbons
    # After concat: A atoms are [0,1] (two C), B atom is [2]
    assert bond_exists(merged, 0, 2)
    assert bond_exists(merged, 1, 2)
    assert degree(merged, 2) == 2


def test_merge_insertion_unsupported_both_insertion() -> None:
    a = make_insertion_graph("a", meta_smiles="C[*:1]C")
    b = make_insertion_graph("b", meta_smiles="C[*:2]C")
    try:
        _ = merge_by_labels(a, b, label_a="a", label_b="b")
        assert False, "Expected ValueError for both-insertion merge"
    except ValueError:
        pass


# -----------------------------
# Tests: error cases
# -----------------------------

def test_merge_raises_on_missing_label() -> None:
    a = make_substituent_graph("A", meta_smiles="[*:1]CC")
    b = make_substituent_graph("B", meta_smiles="[*:2]CC")
    try:
        _ = merge_by_labels(a, b, label_a="MISSING", label_b="B")
        assert False, "Expected ValueError on missing label"
    except ValueError:
        pass


def test_merge_raises_on_orphan_dummy() -> None:
    # dummy with no neighbors
    atomic_num = np.array([0, 6], dtype=np.int16)
    bonds = np.zeros((0, 2), dtype=np.int32)
    bond_type = np.zeros((0,), dtype=np.int8)
    attach = np.array(["A", None], dtype=object)
    a = _base_graph(atomic_num, bonds, bond_type, attachment_label=attach, meta_smiles="orphan")
    b = make_substituent_graph("B", meta_smiles="[*:2]CC")

    try:
        _ = merge_by_labels(a, b, label_a="A", label_b="B")
        assert False, "Expected ValueError for orphan dummy with no neighbors"
    except ValueError:
        pass


# -----------------------------
# Runner
# -----------------------------

def run_all_tests() -> None:
    tests = [
        test_resolve_site_substituent,
        test_resolve_site_insertion,
        test_resolve_site_errors_on_missing_label,
        test_merge_substituent_basic,
        test_merge_substituent_removes_explicit_h_node,
        test_merge_insertion_basic_star_topology,
        test_merge_insertion_unsupported_both_insertion,
        test_merge_raises_on_missing_label,
        test_merge_raises_on_orphan_dummy,
    ]
    passed = 0
    failed = 0

    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"✓ {name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
