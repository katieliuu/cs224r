from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import inspect
from pathlib import Path

import numpy as np

from chem.edit.atoms import replace_atom
from chem.ops.atoms import AddAtom, DeleteAtom, ReplaceAtom
from chem.ops.base import apply_transform
from core.structs import MolArrays, MolGraph
from core.structs_helper import validate_molarrays


ROOT = Path(__file__).resolve().parents[1]


def make_graph(*, n_atoms: int = 2, with_ensemble_pos: bool = False) -> MolGraph:
    atomic_num = np.array([6] * n_atoms, dtype=np.int16)
    formal_charge = np.zeros(n_atoms, dtype=np.int8)
    bonds = np.array([[0, 1]], dtype=np.int32) if n_atoms >= 2 else np.zeros((0, 2), dtype=np.int32)
    bond_type = np.array([0], dtype=np.int8) if n_atoms >= 2 else np.zeros((0,), dtype=np.int8)
    is_aromatic = np.zeros(n_atoms, dtype=bool)

    pos = None
    coord_valid = None
    if with_ensemble_pos:
        pos = np.zeros((2, n_atoms, 3), dtype=np.float32)
        coord_valid = np.ones((2, n_atoms), dtype=bool)

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        is_aromatic=is_aromatic,
        pos=pos,
        coord_frame="etkdg" if with_ensemble_pos else None,
        coord_valid=coord_valid,
    )
    return MolGraph(arrays=arrays, meta={})


def check_ensemble_validation_mismatch() -> None:
    graph = make_graph(with_ensemble_pos=True)
    try:
        validate_molarrays(graph.arrays)
    except ValueError as exc:
        message = str(exc)
        assert "pos must be (N,2) or (N,3)" in message
        print(f"PASS 1: validator rejects ensemble coordinates: {message}")
        return
    raise AssertionError("Expected validate_molarrays() to reject ensemble coordinates")


def check_replace_atom_breaks_with_ensemble_pos() -> None:
    graph = make_graph(n_atoms=3, with_ensemble_pos=True)
    replacement = np.ones((2, 3), dtype=np.float32)
    try:
        replace_atom(graph, 0, pos=replacement)
    except ValueError as exc:
        message = str(exc)
        assert "pos must have shape" in message
        print(f"PASS 2: replace_atom rejects valid per-atom ensemble coordinates: {message}")
        return
    raise AssertionError("Expected replace_atom() to fail with ensemble coordinates")


def check_ops_drop_aromatic_flag() -> None:
    graph_add = make_graph(n_atoms=1)
    graph_add, _ = apply_transform(graph_add, AddAtom(atomic_num=6, aromatic=True))
    add_result = bool(graph_add.arrays.is_aromatic[-1])
    assert add_result is False, "Expected AddAtom(aromatic=True) to leave is_aromatic unchanged"

    graph_replace = make_graph(n_atoms=2)
    graph_replace, _ = apply_transform(
        graph_replace,
        ReplaceAtom(idx=0, atomic_num=6, formal_charge=0, aromatic=True),
    )
    replace_result = bool(graph_replace.arrays.is_aromatic[0])
    assert replace_result is False, "Expected ReplaceAtom(aromatic=True) to leave is_aromatic unchanged"

    print(
        "PASS 3: atom op wrappers drop aromatic flag "
        f"(AddAtom -> {add_result}, ReplaceAtom -> {replace_result})"
    )


def check_duplicate_deleteatom_definition() -> None:
    source_path = ROOT / "chem" / "ops" / "atoms.py"
    source = source_path.read_text(encoding="utf-8")
    class_count = source.count("class DeleteAtom(Transform):")
    apply_line = inspect.getsourcelines(DeleteAtom.apply)[1]

    assert class_count == 2, f"Expected two DeleteAtom class definitions, found {class_count}"
    assert apply_line > 60, f"Expected exported DeleteAtom to come from the later definition, got line {apply_line}"

    print(
        "PASS 4: duplicate DeleteAtom definitions confirmed "
        f"(count={class_count}, exported apply line={apply_line})"
    )


def main() -> None:
    check_ensemble_validation_mismatch()
    check_replace_atom_breaks_with_ensemble_pos()
    check_ops_drop_aromatic_flag()
    check_duplicate_deleteatom_definition()
    print("All reviewed issues were reproduced.")


if __name__ == "__main__":
    main()
