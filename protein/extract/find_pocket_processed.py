"""
protein/extract/find_pocket_processed.py

Extract ligand-centered binding pockets from a preprocessed holo mmCIF.

This variant is intended for the pipeline:
    fetch.py -> preprocess.py -> find_pocket_processed.py

Unlike the unprocessed extractor, this module treats actual per-instance
connection records in ``_struct_conn`` as the primary ligand bond source.
That is the expected path for preprocessed mmCIF files that preserve explicit
connection records.

If a processed file still lacks usable intra-ligand connection records, the
script can optionally fall back to OpenBabel bond perception with
``--allow-openbabel-fallback``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Geometry import Point3D

if __package__ in (None, ""):
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    import find_pocket_unprocessed as base
else:
    from . import find_pocket_unprocessed as base


DEFAULT_OUTPUT_SUFFIX = "_processed_pockets"
PROCESSED_GROUPING_CONN_TYPES = frozenset({"covale", "disulf", "metalc"})
PROCESSED_BOND_CONN_TYPES = frozenset({"covale", "disulf", "metalc"})
MISSING_CONN_VALUES = frozenset({"", ".", "?"})


PocketExtractionError = base.PocketExtractionError
ResidueKey = base.ResidueKey
AtomRecord = base.AtomRecord
LigandGroup = base.LigandGroup


@dataclass(frozen=True, slots=True)
class ProcessedStructConnLink:
    """One processed mmCIF connection record."""

    residue_key_1: ResidueKey
    atom_name_1: str
    residue_key_2: ResidueKey
    atom_name_2: str
    conn_type: str
    value_order: str
    details: str
    role: str


def extract_pockets(args: argparse.Namespace) -> dict[str, object]:
    """Extract ligand pockets from a processed holo mmCIF file."""
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mmCIF file was not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    structure, cif_dict = base.load_structure_and_cif(input_path)
    model = structure[args.model_index]

    component_metadata = base.build_component_metadata(cif_dict)
    struct_conn_links = build_processed_struct_conn_links(cif_dict)
    residue_lookup = base.index_model_residues(model)

    ligand_groups = detect_ligand_groups(
        residue_lookup=residue_lookup,
        component_metadata=component_metadata,
        struct_conn_links=struct_conn_links,
        include_resnames={name.upper() for name in args.include_resname},
        exclude_resnames={name.upper() for name in args.exclude_resname},
        min_heavy_atoms=args.min_heavy_atoms,
    )

    manifest_entries: list[dict[str, object]] = []
    for ligand_group in ligand_groups:
        ligand_residues = [residue_lookup[key] for key in ligand_group.residue_keys]
        ligand_atoms = base.collect_residue_atoms(ligand_residues)
        ligand_mol, atom_records, bond_source = build_ligand_molecule(
            ligand_id=ligand_group.ligand_id,
            ligand_residues=ligand_residues,
            struct_conn_links=struct_conn_links,
            allow_openbabel_fallback=args.allow_openbabel_fallback,
        )

        pocket_residues = base.find_pocket_residues(
            residue_lookup=residue_lookup,
            ligand_group=ligand_group,
            ligand_atoms=ligand_atoms,
            radius=args.radius,
            include_hetero=args.include_hetero_pocket,
            include_water=args.include_water,
        )

        ligand_sdf_path = output_dir / f"{ligand_group.ligand_id}.sdf"
        pocket_json_path = output_dir / f"{ligand_group.ligand_id}.json"
        base.write_ligand_sdf(ligand_mol, ligand_sdf_path)

        pocket_payload = base.build_pocket_payload(
            ligand_group=ligand_group,
            ligand_mol=ligand_mol,
            ligand_atom_records=atom_records,
            pocket_residues=pocket_residues,
            component_metadata=component_metadata,
            ligand_sdf_path=ligand_sdf_path,
            pocket_radius=args.radius,
        )
        pocket_payload["bond_source"] = bond_source
        pocket_json_path.write_text(json.dumps(pocket_payload, indent=2), encoding="utf-8")

        manifest_entries.append(
            {
                "ligand_id": ligand_group.ligand_id,
                "residues": [key.to_dict() for key in ligand_group.residue_keys],
                "ligand_sdf": str(ligand_sdf_path),
                "pocket_json": str(pocket_json_path),
                "pocket_residue_count": len(pocket_residues),
                "ligand_atom_count": len(atom_records),
                "ligand_bond_count": ligand_mol.GetNumBonds(),
                "bond_source": bond_source,
            }
        )

    manifest = {
        "input_mmcif": str(input_path),
        "output_dir": str(output_dir),
        "model_index": args.model_index,
        "pocket_radius": args.radius,
        "ligand_count": len(manifest_entries),
        "ligands": manifest_entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_processed_struct_conn_links(cif_dict: dict[str, object]) -> list[ProcessedStructConnLink]:
    """Parse processed mmCIF ``_struct_conn`` records using auth IDs."""
    conn_types = base.as_list(cif_dict.get("_struct_conn.conn_type_id"))
    atom_1 = base.as_list(cif_dict.get("_struct_conn.ptnr1_label_atom_id"))
    atom_2 = base.as_list(cif_dict.get("_struct_conn.ptnr2_label_atom_id"))
    chain_1 = base.as_list(cif_dict.get("_struct_conn.ptnr1_auth_asym_id"))
    chain_2 = base.as_list(cif_dict.get("_struct_conn.ptnr2_auth_asym_id"))
    resname_1 = base.as_list(cif_dict.get("_struct_conn.ptnr1_auth_comp_id"))
    resname_2 = base.as_list(cif_dict.get("_struct_conn.ptnr2_auth_comp_id"))
    resseq_1 = base.as_list(cif_dict.get("_struct_conn.ptnr1_auth_seq_id"))
    resseq_2 = base.as_list(cif_dict.get("_struct_conn.ptnr2_auth_seq_id"))
    ins_1 = base.as_list(cif_dict.get("_struct_conn.pdbx_ptnr1_PDB_ins_code"))
    ins_2 = base.as_list(cif_dict.get("_struct_conn.pdbx_ptnr2_PDB_ins_code"))
    row_count = len(conn_types)
    value_orders = mmcif_column_or_default(cif_dict, "_struct_conn.pdbx_value_order", row_count, "?")
    details = mmcif_column_or_default(cif_dict, "_struct_conn.details", row_count, "?")
    roles = mmcif_column_or_default(cif_dict, "_struct_conn.pdbx_role", row_count, "?")

    links: list[ProcessedStructConnLink] = []
    for values in zip(
        conn_types,
        chain_1,
        resname_1,
        resseq_1,
        ins_1,
        atom_1,
        chain_2,
        resname_2,
        resseq_2,
        ins_2,
        atom_2,
        value_orders,
        details,
        roles,
    ):
        (
            conn_type,
            auth_chain_1,
            auth_resname_1,
            auth_resseq_1,
            auth_ins_1,
            auth_atom_1,
            auth_chain_2,
            auth_resname_2,
            auth_resseq_2,
            auth_ins_2,
            auth_atom_2,
            value_order,
            detail_text,
            role_text,
        ) = values
        try:
            key_1 = ResidueKey(
                chain_id=str(auth_chain_1).strip(),
                resseq=base.parse_int(auth_resseq_1),
                insertion_code=base.normalize_insertion_code(auth_ins_1),
                resname=str(auth_resname_1).strip().upper(),
            )
            key_2 = ResidueKey(
                chain_id=str(auth_chain_2).strip(),
                resseq=base.parse_int(auth_resseq_2),
                insertion_code=base.normalize_insertion_code(auth_ins_2),
                resname=str(auth_resname_2).strip().upper(),
            )
        except ValueError:
            continue

        links.append(
            ProcessedStructConnLink(
                residue_key_1=key_1,
                atom_name_1=str(auth_atom_1).strip(),
                residue_key_2=key_2,
                atom_name_2=str(auth_atom_2).strip(),
                conn_type=str(conn_type).strip().lower(),
                value_order=str(value_order).strip().lower(),
                details=str(detail_text).strip(),
                role=str(role_text).strip(),
            )
        )
    return links


def detect_ligand_groups(
    *,
    residue_lookup: dict[ResidueKey, object],
    component_metadata: dict[str, dict[str, str]],
    struct_conn_links: Sequence[ProcessedStructConnLink],
    include_resnames: set[str],
    exclude_resnames: set[str],
    min_heavy_atoms: int,
) -> list[LigandGroup]:
    """Detect ligand residue groups, using processed connection records."""
    candidate_keys: list[ResidueKey] = []
    for key, residue in residue_lookup.items():
        if base.is_candidate_ligand_residue(
            residue_key=key,
            residue=residue,
            component_metadata=component_metadata,
            include_resnames=include_resnames,
            exclude_resnames=exclude_resnames,
            min_heavy_atoms=min_heavy_atoms,
        ):
            candidate_keys.append(key)

    candidate_set = set(candidate_keys)
    adjacency: dict[ResidueKey, set[ResidueKey]] = {key: set() for key in candidate_keys}
    for link in struct_conn_links:
        if not is_bond_like_connection(link):
            continue
        if link.residue_key_1 in candidate_set and link.residue_key_2 in candidate_set:
            adjacency[link.residue_key_1].add(link.residue_key_2)
            adjacency[link.residue_key_2].add(link.residue_key_1)

    groups: list[LigandGroup] = []
    seen: set[ResidueKey] = set()
    for index, start_key in enumerate(sorted(candidate_keys, key=base.ligand_sort_key), start=1):
        if start_key in seen:
            continue
        component = base.collect_connected_component(start_key, adjacency)
        seen.update(component)
        ordered_keys = tuple(sorted(component, key=base.ligand_sort_key))
        ligand_id = base.make_ligand_id(index, ordered_keys)
        groups.append(LigandGroup(ligand_id=ligand_id, residue_keys=ordered_keys))
    return groups


def build_ligand_molecule(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    struct_conn_links: Sequence[ProcessedStructConnLink],
    allow_openbabel_fallback: bool,
) -> tuple[Chem.Mol, list[AtomRecord], str]:
    """Build a ligand molecule from processed connection records."""
    try:
        mol, atom_records = build_ligand_molecule_from_connection_records(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
            struct_conn_links=struct_conn_links,
        )
        return mol, atom_records, "struct_conn"
    except PocketExtractionError:
        if not allow_openbabel_fallback:
            raise
        mol, atom_records = base.build_ligand_molecule_with_openbabel(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
        )
        return mol, atom_records, "openbabel_fallback"


def build_ligand_molecule_from_connection_records(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    struct_conn_links: Sequence[ProcessedStructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit molecule using processed mmCIF connection records."""
    rw_mol = Chem.RWMol()
    atom_records: list[AtomRecord] = []
    atom_index_lookup: dict[tuple[ResidueKey, str], int] = {}
    residue_keys = {base.residue_key_from_biopython(residue.parent.id, residue) for residue in ligand_residues}
    conformer = Chem.Conformer(sum(1 for residue in ligand_residues for _ in residue.get_atoms()))

    for residue in ligand_residues:
        residue_key = base.residue_key_from_biopython(residue.parent.id, residue)
        for atom in residue.get_atoms():
            atom_name = atom.get_name().strip()
            rd_atom = Chem.Atom(atom.element)
            rd_atom.SetProp("atom_name", atom_name)
            rd_atom.SetProp("resname", residue_key.resname)
            rd_atom.SetProp("chain_id", residue_key.chain_id)
            rd_atom.SetIntProp("resseq", residue_key.resseq)
            if residue_key.insertion_code:
                rd_atom.SetProp("insertion_code", residue_key.insertion_code)
            rd_index = rw_mol.AddAtom(rd_atom)
            atom_index_lookup[(residue_key, atom_name)] = rd_index

            x, y, z = atom.coord
            conformer.SetAtomPosition(rd_index, Point3D(float(x), float(y), float(z)))
            atom_records.append(
                AtomRecord(
                    atom_index=rd_index,
                    atom_name=atom_name,
                    element=atom.element,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    residue_key=residue_key,
                )
            )

    usable_links = [
        link
        for link in struct_conn_links
        if is_bond_like_connection(link)
        and link.residue_key_1 in residue_keys
        and link.residue_key_2 in residue_keys
    ]
    if not usable_links:
        raise PocketExtractionError(
            f"No usable intra-ligand _struct_conn records were found for {ligand_id}."
        )

    added_bonds = 0
    for link in usable_links:
        key_1 = (link.residue_key_1, link.atom_name_1)
        key_2 = (link.residue_key_2, link.atom_name_2)
        if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
            continue
        base.add_rdkit_bond(
            rw_mol,
            atom_index_lookup[key_1],
            atom_index_lookup[key_2],
            order=resolve_connection_order(link),
            aromatic=is_aromatic_connection(link),
        )
        added_bonds += 1

    if added_bonds == 0:
        raise PocketExtractionError(
            f"_struct_conn records were present for {ligand_id}, but none mapped to ligand atoms."
        )

    supplement_hydrogen_bonds_from_openbabel(rw_mol, atom_records)

    mol = rw_mol.GetMol()
    conformer.SetId(0)
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    base.sanitize_molecule_best_effort(mol)

    if has_disconnected_heavy_atom_fragments(mol):
        raise PocketExtractionError(
            f"_struct_conn bond records appear incomplete for {ligand_id}; "
            "heavy-atom connectivity is fragmented."
        )

    return mol, atom_records


def is_bond_like_connection(link: ProcessedStructConnLink) -> bool:
    """Return True when a processed connection record looks like a real bond."""
    if normalize_conn_value(link.value_order) not in MISSING_CONN_VALUES:
        return True
    return link.conn_type in PROCESSED_BOND_CONN_TYPES or link.conn_type in PROCESSED_GROUPING_CONN_TYPES


def resolve_connection_order(link: ProcessedStructConnLink) -> str:
    """Resolve the bond order token used to add an RDKit bond."""
    normalized = normalize_conn_value(link.value_order)
    if normalized not in MISSING_CONN_VALUES:
        return normalized
    if link.conn_type == "disulf":
        return "sing"
    if link.conn_type == "metalc":
        return "sing"
    return "sing"


def is_aromatic_connection(link: ProcessedStructConnLink) -> bool:
    """Infer aromatic bonds from connection record order/details."""
    normalized = normalize_conn_value(link.value_order)
    return normalized in {"arom", "delo"} or "arom" in link.details.lower()


def normalize_conn_value(value: str) -> str:
    """Normalize processed connection fields."""
    return str(value).strip().lower()


def mmcif_column_or_default(
    cif_dict: dict[str, object],
    key: str,
    row_count: int,
    default: str,
) -> list[str]:
    """Return an mmCIF loop column or a same-length default-filled list."""
    values = base.as_list(cif_dict.get(key))
    if not values:
        return [default] * row_count
    return [str(value) for value in values]


def supplement_hydrogen_bonds_from_openbabel(
    rw_mol: Chem.RWMol,
    atom_records: Sequence[AtomRecord],
) -> None:
    """Optionally fill in missing hydrogen bonds without replacing heavy-atom truth."""
    if base.pybel is None or base.ob is None:
        return

    isolated_hydrogen_indices = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0
    ]
    if not isolated_hydrogen_indices:
        return

    xyz_lines = [str(len(atom_records)), "ligand"]
    for atom_record in atom_records:
        xyz_lines.append(
            f"{atom_record.element} {atom_record.x:.6f} {atom_record.y:.6f} {atom_record.z:.6f}"
        )
    pybel_mol = base.pybel.readstring("xyz", "\n".join(xyz_lines))
    ob_mol = pybel_mol.OBMol

    for ob_bond in base.ob.OBMolBondIter(ob_mol):
        begin_idx = ob_bond.GetBeginAtomIdx() - 1
        end_idx = ob_bond.GetEndAtomIdx() - 1
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue

        begin_atom = rw_mol.GetAtomWithIdx(begin_idx)
        end_atom = rw_mol.GetAtomWithIdx(end_idx)
        if begin_atom.GetAtomicNum() != 1 and end_atom.GetAtomicNum() != 1:
            continue

        bond_type = base.map_openbabel_bond_order(
            ob_bond.GetBondOrder(),
            aromatic=bool(ob_bond.IsAromatic()),
        )
        rw_mol.AddBond(begin_idx, end_idx, bond_type)
        bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is not None and bool(ob_bond.IsAromatic()):
            bond.SetIsAromatic(True)
            begin_atom.SetIsAromatic(True)
            end_atom.SetIsAromatic(True)


def has_disconnected_heavy_atom_fragments(mol: Chem.Mol) -> bool:
    """Detect fragmented heavy-atom connectivity."""
    heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(heavy_atom_indices) <= 1:
        return False

    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    heavy_fragments = []
    heavy_set = set(heavy_atom_indices)
    for fragment in fragments:
        fragment_heavy = [idx for idx in fragment if idx in heavy_set]
        if fragment_heavy:
            heavy_fragments.append(fragment_heavy)
    return len(heavy_fragments) > 1


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract ligand-centered pockets from a processed holo mmCIF, "
            "using explicit _struct_conn records as the primary ligand bond source."
        )
    )
    parser.add_argument("input_path", help="Input processed holo mmCIF/CIF file.")
    parser.add_argument(
        "--output-dir",
        help="Directory for extracted ligand SDFs and pocket JSON files.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=base.DEFAULT_POCKET_RADIUS,
        help=f"Pocket radius in angstrom (default: {base.DEFAULT_POCKET_RADIUS}).",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Zero-based model index to process (default: 0).",
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=base.DEFAULT_MIN_HEAVY_ATOMS,
        help=(
            "Minimum heavy atoms required for a ligand candidate "
            f"(default: {base.DEFAULT_MIN_HEAVY_ATOMS})."
        ),
    )
    parser.add_argument(
        "--include-resname",
        action="append",
        default=[],
        help="Force-include one residue name as a ligand candidate. May be repeated.",
    )
    parser.add_argument(
        "--exclude-resname",
        action="append",
        default=[],
        help="Force-exclude one residue name from ligand detection. May be repeated.",
    )
    parser.add_argument(
        "--include-hetero-pocket",
        action="store_true",
        help="Include nearby non-water hetero residues in the pocket output.",
    )
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Include nearby waters in the pocket output.",
    )
    parser.add_argument(
        "--allow-openbabel-fallback",
        action="store_true",
        help=(
            "Fallback to OpenBabel bond perception if the processed mmCIF "
            "_struct_conn records are missing or incomplete for a ligand."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    manifest = extract_pockets(args)
    print(f"Extracted {manifest['ligand_count']} ligand pocket(s) into {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
