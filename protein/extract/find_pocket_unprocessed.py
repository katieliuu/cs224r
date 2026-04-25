"""
protein/extract/find_pocket_unprocessed.py

Extract ligand-centered binding pockets from a holo mmCIF structure using
open-source tooling only.

This module is designed for use from the ``mlchem`` conda environment, where
the project already has RDKit, Biopython, NumPy, and OpenBabel available.

For each detected ligand, the script writes:
    - an SDF file for the ligand with 3D coordinates
    - a JSON file containing ligand atoms, ligand bonds, and nearby pocket
      residues with atom coordinates and residue identities
    - a manifest JSON summarizing all extracted ligands

Primary bond source:
    1. PDBx/mmCIF ``_chem_comp_bond`` templates when available
    2. OpenBabel bond perception from the ligand's 3D coordinates as fallback

Example:
    python protein/extract/find_pocket.py input_holo.cif --output-dir pockets
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from rdkit.Geometry import Point3D

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:  # pragma: no cover - environment-dependent fallback
    ob = None
    pybel = None


DEFAULT_POCKET_RADIUS = 10.0
DEFAULT_MIN_HEAVY_ATOMS = 6
DEFAULT_OUTPUT_SUFFIX = "_pockets"
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O"})
COMMON_EXCLUDED_LIGANDS = frozenset(
    {
        "ACE",
        "ACT",
        "ACY",
        "BME",
        "BR",
        "BU1",
        "BU2",
        "BU3",
        "CA",
        "CD",
        "CL",
        "CO",
        "CU",
        "DMS",
        "DOD",
        "EDO",
        "EOH",
        "FE",
        "FMT",
        "GOL",
        "HG",
        "HOH",
        "IOD",
        "IPA",
        "K",
        "MES",
        "MG",
        "MN",
        "MPD",
        "NA",
        "NH4",
        "NI",
        "NO3",
        "PEG",
        "PGE",
        "PG4",
        "PO4",
        "SO4",
        "TRS",
        "UNX",
        "ZN",
    }
)
METAL_ELEMENTS = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "ZN",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "CD",
        "HG",
    }
)
STRUCT_CONN_GROUPING_TYPES = frozenset({"covale"})


class PocketExtractionError(RuntimeError):
    """Raised when the mmCIF pocket extraction workflow fails."""


@dataclass(frozen=True, slots=True)
class ResidueKey:
    """Stable identifier for one residue instance."""

    chain_id: str
    resseq: int
    insertion_code: str
    resname: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "resseq": self.resseq,
            "insertion_code": self.insertion_code,
            "resname": self.resname,
        }

    def compact_label(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        return f"{self.chain_id}_{self.resname}_{self.resseq}{suffix}"


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Serializable atom record for ligands and pocket residues."""

    atom_index: int
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    residue_key: ResidueKey

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_index": self.atom_index,
            "atom_name": self.atom_name,
            "element": self.element,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "residue": self.residue_key.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TemplateBond:
    """One mmCIF template bond from ``_chem_comp_bond``."""

    atom_name_1: str
    atom_name_2: str
    order: str
    aromatic: bool


@dataclass(frozen=True, slots=True)
class StructConnLink:
    """One residue-residue link from ``_struct_conn``."""

    residue_key_1: ResidueKey
    atom_name_1: str
    residue_key_2: ResidueKey
    atom_name_2: str
    conn_type: str


@dataclass(frozen=True, slots=True)
class LigandGroup:
    """Connected ligand residue group to be extracted as one ligand."""

    ligand_id: str
    residue_keys: tuple[ResidueKey, ...]


@dataclass(slots=True)
class SimpleAtom:
    """Minimal atom adapter for processed mmCIF fallback parsing."""

    name: str
    element: str
    coord: np.ndarray

    def get_name(self) -> str:
        return self.name


@dataclass(slots=True)
class SimpleChain:
    """Minimal chain adapter for processed mmCIF fallback parsing."""

    id: str
    residues: list["SimpleResidue"] = field(default_factory=list)

    def __iter__(self):
        return iter(self.residues)


@dataclass(slots=True)
class SimpleResidue:
    """Minimal residue adapter for processed mmCIF fallback parsing."""

    resname: str
    id: tuple[str, int, str]
    parent: SimpleChain
    atoms: list[SimpleAtom] = field(default_factory=list)

    def __iter__(self):
        return iter(self.atoms)

    def get_atoms(self):
        return iter(self.atoms)


@dataclass(slots=True)
class SimpleModel:
    """Minimal model adapter for processed mmCIF fallback parsing."""

    index: int
    chains: list[SimpleChain]

    def __iter__(self):
        return iter(self.chains)


@dataclass(slots=True)
class SimpleStructure:
    """Minimal structure adapter for processed mmCIF fallback parsing."""

    models: list[SimpleModel]

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index: int) -> SimpleModel:
        return self.models[index]


def extract_pockets(args: argparse.Namespace) -> dict[str, object]:
    """Extract ligands and their surrounding pockets from an mmCIF file."""
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mmCIF file was not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    structure, cif_dict = load_structure_and_cif(input_path)
    model = structure[args.model_index]

    component_metadata = build_component_metadata(cif_dict)
    template_bonds = build_template_bonds(cif_dict)
    struct_conn_links = build_struct_conn_links(cif_dict)

    residue_lookup = index_model_residues(model)
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
        ligand_atoms = collect_residue_atoms(ligand_residues)
        ligand_mol, atom_records = build_ligand_molecule(
            ligand_id=ligand_group.ligand_id,
            ligand_residues=ligand_residues,
            residue_lookup=residue_lookup,
            template_bonds=template_bonds,
            struct_conn_links=struct_conn_links,
        )
        pocket_residues = find_pocket_residues(
            residue_lookup=residue_lookup,
            ligand_group=ligand_group,
            ligand_atoms=ligand_atoms,
            radius=args.radius,
            include_hetero=args.include_hetero_pocket,
            include_water=args.include_water,
        )

        ligand_sdf_path = output_dir / f"{ligand_group.ligand_id}.sdf"
        pocket_json_path = output_dir / f"{ligand_group.ligand_id}.json"
        write_ligand_sdf(ligand_mol, ligand_sdf_path)

        pocket_payload = build_pocket_payload(
            ligand_group=ligand_group,
            ligand_mol=ligand_mol,
            ligand_atom_records=atom_records,
            pocket_residues=pocket_residues,
            component_metadata=component_metadata,
            ligand_sdf_path=ligand_sdf_path,
            pocket_radius=args.radius,
        )
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


def load_structure_and_cif(input_path: Path):
    """Load the structure tree and raw mmCIF dictionary."""
    cif_dict = MMCIF2Dict(str(input_path))
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    try:
        structure = parser.get_structure(input_path.stem, str(input_path))
    except KeyError:
        structure = build_simple_structure_from_atom_site(cif_dict)
    return structure, cif_dict


def index_model_residues(model) -> dict[ResidueKey, object]:
    """Create a residue lookup keyed by chain/resseq/insertion code/resname."""
    residue_lookup: dict[ResidueKey, object] = {}
    for chain in model:
        for residue in chain:
            key = residue_key_from_biopython(chain.id, residue)
            residue_lookup[key] = residue
    return residue_lookup


def residue_key_from_biopython(chain_id: str, residue) -> ResidueKey:
    """Convert a Bio.PDB residue to a ResidueKey."""
    _, resseq, insertion_code = residue.id
    return ResidueKey(
        chain_id=str(chain_id).strip(),
        resseq=int(resseq),
        insertion_code=normalize_insertion_code(insertion_code),
        resname=str(residue.resname).strip().upper(),
    )


def build_simple_structure_from_atom_site(cif_dict: dict[str, object]) -> SimpleStructure:
    """Fallback structure builder for reduced mmCIF files lacking Biopython-required fields."""
    required_keys = [
        "_atom_site.label_atom_id",
        "_atom_site.type_symbol",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    missing_keys = [key for key in required_keys if key not in cif_dict]
    if missing_keys:
        raise PocketExtractionError(
            "The mmCIF file is missing required atom_site columns for fallback parsing: "
            + ", ".join(missing_keys)
        )

    atom_names = as_list(cif_dict.get("_atom_site.label_atom_id"))
    elements = as_list(cif_dict.get("_atom_site.type_symbol"))
    resnames = as_list(cif_dict.get("_atom_site.label_comp_id"))
    chain_ids = as_list(cif_dict.get("_atom_site.auth_asym_id"))
    insertion_codes = as_list(cif_dict.get("_atom_site.pdbx_PDB_ins_code"))
    xs = as_list(cif_dict.get("_atom_site.Cartn_x"))
    ys = as_list(cif_dict.get("_atom_site.Cartn_y"))
    zs = as_list(cif_dict.get("_atom_site.Cartn_z"))

    if "_atom_site.auth_seq_id" in cif_dict:
        seq_ids = as_list(cif_dict.get("_atom_site.auth_seq_id"))
    elif "_atom_site.label_seq_id" in cif_dict:
        seq_ids = as_list(cif_dict.get("_atom_site.label_seq_id"))
    else:
        raise PocketExtractionError(
            "The mmCIF file does not contain _atom_site.auth_seq_id or _atom_site.label_seq_id."
        )

    component_metadata = build_component_metadata(cif_dict)
    chain_map: dict[str, SimpleChain] = {}
    residue_map: dict[tuple[str, int, str, str], SimpleResidue] = {}

    for values in zip(atom_names, elements, resnames, chain_ids, seq_ids, insertion_codes, xs, ys, zs):
        atom_name, element, resname, chain_id, seq_id, ins_code, x, y, z = values
        try:
            resseq = parse_int(seq_id)
        except ValueError:
            continue
        normalized_chain = str(chain_id).strip()
        normalized_resname = str(resname).strip().upper()
        normalized_ins_code = normalize_insertion_code(ins_code)
        hetflag = infer_hetflag_from_component_type(normalized_resname, component_metadata)

        chain = chain_map.get(normalized_chain)
        if chain is None:
            chain = SimpleChain(id=normalized_chain)
            chain_map[normalized_chain] = chain

        residue_key = (normalized_chain, resseq, normalized_ins_code, normalized_resname)
        residue = residue_map.get(residue_key)
        if residue is None:
            residue = SimpleResidue(
                resname=normalized_resname,
                id=(hetflag, resseq, normalized_ins_code if normalized_ins_code else " "),
                parent=chain,
            )
            residue_map[residue_key] = residue
            chain.residues.append(residue)

        residue.atoms.append(
            SimpleAtom(
                name=str(atom_name).strip(),
                element=str(element).strip(),
                coord=np.asarray([float(x), float(y), float(z)], dtype=np.float32),
            )
        )

    for chain in chain_map.values():
        for residue in chain.residues:
            atom_name_set = {atom.get_name().strip().upper() for atom in residue.atoms}
            hetflag, resseq, ins_code = residue.id
            if looks_like_polymer_backbone(atom_name_set):
                residue.id = (" ", resseq, ins_code)

    model = SimpleModel(index=0, chains=list(chain_map.values()))
    return SimpleStructure(models=[model])


def infer_hetflag_from_component_type(
    resname: str,
    component_metadata: dict[str, dict[str, str]],
) -> str:
    """Infer a Bio.PDB-style residue hetflag from chem-comp metadata."""
    metadata = component_metadata.get(resname)
    if metadata is None:
        return f"H_{resname}"
    component_type = metadata.get("type", "").lower()
    if "linking" in component_type and "peptide-like" not in component_type:
        return " "
    return f"H_{resname}"


def looks_like_polymer_backbone(atom_names: set[str]) -> bool:
    """Heuristic to suppress polymer residues when chem_comp metadata is sparse."""
    protein_backbone = {"N", "CA", "C", "O"}
    dna_backbone = {"P", "O5'", "C5'", "C4'", "C3'"}
    rna_backbone = {"P", "O5'", "C5'", "C4'", "C3'", "O2'"}
    return (
        protein_backbone.issubset(atom_names)
        or dna_backbone.issubset(atom_names)
        or rna_backbone.issubset(atom_names)
    )


def build_component_metadata(cif_dict: dict[str, object]) -> dict[str, dict[str, str]]:
    """Build a lookup of ``_chem_comp`` metadata keyed by residue name."""
    component_metadata: dict[str, dict[str, str]] = {}
    ids = as_list(cif_dict.get("_chem_comp.id"))
    types = as_list(cif_dict.get("_chem_comp.type"))
    names = as_list(cif_dict.get("_chem_comp.name"))
    for comp_id, comp_type, comp_name in zip(ids, types, names):
        component_metadata[str(comp_id).upper()] = {
            "type": str(comp_type),
            "name": str(comp_name),
        }
    return component_metadata


def build_template_bonds(cif_dict: dict[str, object]) -> dict[str, list[TemplateBond]]:
    """Build intra-component bond templates from ``_chem_comp_bond``."""
    bonds_by_component: dict[str, list[TemplateBond]] = defaultdict(list)
    comp_ids = as_list(cif_dict.get("_chem_comp_bond.comp_id"))
    atom_1 = as_list(cif_dict.get("_chem_comp_bond.atom_id_1"))
    atom_2 = as_list(cif_dict.get("_chem_comp_bond.atom_id_2"))
    orders = as_list(cif_dict.get("_chem_comp_bond.value_order"))
    aromatics = as_list(cif_dict.get("_chem_comp_bond.pdbx_aromatic_flag"))

    for comp_id, name_1, name_2, order, aromatic in zip(comp_ids, atom_1, atom_2, orders, aromatics):
        bonds_by_component[str(comp_id).upper()].append(
            TemplateBond(
                atom_name_1=str(name_1).strip(),
                atom_name_2=str(name_2).strip(),
                order=str(order).strip(),
                aromatic=str(aromatic).strip().upper() == "Y",
            )
        )
    return dict(bonds_by_component)


def build_struct_conn_links(cif_dict: dict[str, object]) -> list[StructConnLink]:
    """Parse residue-residue links from ``_struct_conn`` using auth IDs."""
    conn_types = as_list(cif_dict.get("_struct_conn.conn_type_id"))
    atom_1 = as_list(cif_dict.get("_struct_conn.ptnr1_label_atom_id"))
    atom_2 = as_list(cif_dict.get("_struct_conn.ptnr2_label_atom_id"))
    chain_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_asym_id"))
    chain_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_asym_id"))
    resname_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_comp_id"))
    resname_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_comp_id"))
    resseq_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_seq_id"))
    resseq_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_seq_id"))
    ins_1 = as_list(cif_dict.get("_struct_conn.pdbx_ptnr1_PDB_ins_code"))
    ins_2 = as_list(cif_dict.get("_struct_conn.pdbx_ptnr2_PDB_ins_code"))

    links: list[StructConnLink] = []
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
        ) = values
        try:
            key_1 = ResidueKey(
                chain_id=str(auth_chain_1).strip(),
                resseq=parse_int(auth_resseq_1),
                insertion_code=normalize_insertion_code(auth_ins_1),
                resname=str(auth_resname_1).strip().upper(),
            )
            key_2 = ResidueKey(
                chain_id=str(auth_chain_2).strip(),
                resseq=parse_int(auth_resseq_2),
                insertion_code=normalize_insertion_code(auth_ins_2),
                resname=str(auth_resname_2).strip().upper(),
            )
        except ValueError:
            continue

        links.append(
            StructConnLink(
                residue_key_1=key_1,
                atom_name_1=str(auth_atom_1).strip(),
                residue_key_2=key_2,
                atom_name_2=str(auth_atom_2).strip(),
                conn_type=str(conn_type).strip().lower(),
            )
        )
    return links


def detect_ligand_groups(
    *,
    residue_lookup: dict[ResidueKey, object],
    component_metadata: dict[str, dict[str, str]],
    struct_conn_links: Sequence[StructConnLink],
    include_resnames: set[str],
    exclude_resnames: set[str],
    min_heavy_atoms: int,
) -> list[LigandGroup]:
    """Detect ligand residue groups, excluding common solvent/ion artifacts."""
    candidate_keys: list[ResidueKey] = []
    for key, residue in residue_lookup.items():
        if is_candidate_ligand_residue(
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
        if link.conn_type not in STRUCT_CONN_GROUPING_TYPES:
            continue
        if link.residue_key_1 in candidate_set and link.residue_key_2 in candidate_set:
            adjacency[link.residue_key_1].add(link.residue_key_2)
            adjacency[link.residue_key_2].add(link.residue_key_1)

    groups: list[LigandGroup] = []
    seen: set[ResidueKey] = set()
    for index, start_key in enumerate(sorted(candidate_keys, key=ligand_sort_key), start=1):
        if start_key in seen:
            continue
        component = collect_connected_component(start_key, adjacency)
        seen.update(component)
        ordered_keys = tuple(sorted(component, key=ligand_sort_key))
        ligand_id = make_ligand_id(index, ordered_keys)
        groups.append(LigandGroup(ligand_id=ligand_id, residue_keys=ordered_keys))
    return groups


def is_candidate_ligand_residue(
    *,
    residue_key: ResidueKey,
    residue,
    component_metadata: dict[str, dict[str, str]],
    include_resnames: set[str],
    exclude_resnames: set[str],
    min_heavy_atoms: int,
) -> bool:
    """Heuristic ligand classifier tuned to avoid waters, ions, and additives."""
    resname = residue_key.resname
    if resname in include_resnames:
        return True
    if resname in WATER_RESNAMES:
        return False
    if resname in COMMON_EXCLUDED_LIGANDS or resname in exclude_resnames:
        return False

    metadata = component_metadata.get(resname)
    if metadata is not None:
        component_type = metadata.get("type", "").lower()
        if "linking" in component_type and "peptide-like" not in component_type:
            return False

    atoms = list(residue.get_atoms())
    if not atoms:
        return False

    heavy_atom_count = sum(1 for atom in atoms if atom.element.upper() != "H")
    if heavy_atom_count < min_heavy_atoms:
        return False

    heavy_elements = {atom.element.upper() for atom in atoms if atom.element.upper() != "H"}
    if heavy_elements and heavy_elements.issubset(METAL_ELEMENTS):
        return False

    atom_names = {atom.get_name().strip().upper() for atom in atoms}
    if looks_like_polymer_backbone(atom_names):
        return False

    hetflag = str(residue.id[0]).strip()
    if metadata is None:
        return bool(hetflag and not hetflag.startswith("W"))

    return True


def collect_connected_component(
    start_key: ResidueKey,
    adjacency: dict[ResidueKey, set[ResidueKey]],
) -> set[ResidueKey]:
    """Collect one graph connected component of ligand residues."""
    stack = [start_key]
    component: set[ResidueKey] = set()
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(adjacency.get(current, ()))
    return component


def build_ligand_molecule(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    residue_lookup: dict[ResidueKey, object],
    template_bonds: dict[str, list[TemplateBond]],
    struct_conn_links: Sequence[StructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit ligand molecule with coordinates."""
    try:
        return build_ligand_molecule_from_templates(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
            template_bonds=template_bonds,
            struct_conn_links=struct_conn_links,
        )
    except PocketExtractionError:
        return build_ligand_molecule_with_openbabel(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
        )


def build_ligand_molecule_from_templates(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    template_bonds: dict[str, list[TemplateBond]],
    struct_conn_links: Sequence[StructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit molecule from mmCIF chem-comp templates plus struct_conn."""
    rw_mol = Chem.RWMol()
    atom_records: list[AtomRecord] = []
    atom_index_lookup: dict[tuple[ResidueKey, str], int] = {}
    residue_keys = {residue_key_from_biopython(residue.parent.id, residue) for residue in ligand_residues}
    conformer = Chem.Conformer(sum(1 for residue in ligand_residues for _ in residue.get_atoms()))

    atom_counter = 0
    for residue in ligand_residues:
        residue_key = residue_key_from_biopython(residue.parent.id, residue)
        atom_names_in_residue = {atom.get_name() for atom in residue.get_atoms()}
        if residue_key.resname not in template_bonds:
            raise PocketExtractionError(
                f"No _chem_comp_bond template was found for ligand component {residue_key.resname}."
            )

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
            atom_counter += 1
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

        for bond in template_bonds[residue_key.resname]:
            key_1 = (residue_key, bond.atom_name_1)
            key_2 = (residue_key, bond.atom_name_2)
            if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
                if {bond.atom_name_1, bond.atom_name_2}.issubset(atom_names_in_residue):
                    raise PocketExtractionError(
                        f"Failed to map bond template atoms for {residue_key.compact_label()}."
                    )
                continue
            add_rdkit_bond(
                rw_mol,
                atom_index_lookup[key_1],
                atom_index_lookup[key_2],
                order=bond.order,
                aromatic=bond.aromatic,
            )

    for link in struct_conn_links:
        if link.conn_type not in STRUCT_CONN_GROUPING_TYPES:
            continue
        if link.residue_key_1 not in residue_keys or link.residue_key_2 not in residue_keys:
            continue
        key_1 = (link.residue_key_1, link.atom_name_1)
        key_2 = (link.residue_key_2, link.atom_name_2)
        if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
            continue
        add_rdkit_bond(
            rw_mol,
            atom_index_lookup[key_1],
            atom_index_lookup[key_2],
            order="sing",
            aromatic=False,
        )

    mol = rw_mol.GetMol()
    conformer.SetId(0)
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    sanitize_molecule_best_effort(mol)
    return mol, atom_records


def build_ligand_molecule_with_openbabel(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Fallback ligand builder using OpenBabel bond perception from coordinates."""
    if pybel is None or ob is None:
        raise PocketExtractionError(
            "OpenBabel is required for fallback bond perception, but it is not installed."
        )

    ligand_atoms = collect_residue_atoms(ligand_residues)
    xyz_lines = [str(len(ligand_atoms)), ligand_id]
    for atom_record in ligand_atoms:
        xyz_lines.append(
            f"{atom_record.element} {atom_record.x:.6f} {atom_record.y:.6f} {atom_record.z:.6f}"
        )
    xyz_block = "\n".join(xyz_lines)

    pybel_mol = pybel.readstring("xyz", xyz_block)
    ob_mol = pybel_mol.OBMol

    rw_mol = Chem.RWMol()
    conformer = Chem.Conformer(len(ligand_atoms))
    for atom_record in ligand_atoms:
        rd_atom = Chem.Atom(atom_record.element)
        rd_atom.SetProp("atom_name", atom_record.atom_name)
        rd_atom.SetProp("resname", atom_record.residue_key.resname)
        rd_atom.SetProp("chain_id", atom_record.residue_key.chain_id)
        rd_atom.SetIntProp("resseq", atom_record.residue_key.resseq)
        if atom_record.residue_key.insertion_code:
            rd_atom.SetProp("insertion_code", atom_record.residue_key.insertion_code)
        rd_index = rw_mol.AddAtom(rd_atom)
        conformer.SetAtomPosition(
            rd_index,
            Point3D(atom_record.x, atom_record.y, atom_record.z),
        )

    for ob_bond in ob.OBMolBondIter(ob_mol):
        begin_idx = ob_bond.GetBeginAtomIdx() - 1
        end_idx = ob_bond.GetEndAtomIdx() - 1
        order = ob_bond.GetBondOrder()
        aromatic = bool(ob_bond.IsAromatic())
        bond_type = map_openbabel_bond_order(order, aromatic=aromatic)
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        rw_mol.AddBond(begin_idx, end_idx, bond_type)
        bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is not None and aromatic:
            bond.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(begin_idx).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(end_idx).SetIsAromatic(True)

    mol = rw_mol.GetMol()
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    sanitize_molecule_best_effort(mol)
    return mol, ligand_atoms


def collect_residue_atoms(residues: Sequence[object]) -> list[AtomRecord]:
    """Collect atoms from one or more residues in a stable order."""
    atom_records: list[AtomRecord] = []
    atom_index = 0
    for residue in residues:
        residue_key = residue_key_from_biopython(residue.parent.id, residue)
        for atom in residue.get_atoms():
            x, y, z = atom.coord
            atom_records.append(
                AtomRecord(
                    atom_index=atom_index,
                    atom_name=atom.get_name().strip(),
                    element=atom.element,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    residue_key=residue_key,
                )
            )
            atom_index += 1
    return atom_records


def find_pocket_residues(
    *,
    residue_lookup: dict[ResidueKey, object],
    ligand_group: LigandGroup,
    ligand_atoms: Sequence[AtomRecord],
    radius: float,
    include_hetero: bool,
    include_water: bool,
) -> list[dict[str, object]]:
    """Find residues with any atom within the requested ligand distance cutoff."""
    ligand_coords = np.asarray([[atom.x, atom.y, atom.z] for atom in ligand_atoms], dtype=np.float32)
    ligand_key_set = set(ligand_group.residue_keys)
    pocket_residues: list[dict[str, object]] = []

    for residue_key, residue in sorted(residue_lookup.items(), key=lambda item: ligand_sort_key(item[0])):
        if residue_key in ligand_key_set:
            continue
        hetflag = str(residue.id[0]).strip()
        if not include_water and residue_key.resname in WATER_RESNAMES:
            continue
        if not include_hetero and bool(hetflag) and not hetflag.startswith("W"):
            continue

        atom_records = collect_residue_atoms([residue])
        if not atom_records:
            continue
        residue_coords = np.asarray([[atom.x, atom.y, atom.z] for atom in atom_records], dtype=np.float32)
        min_distance = minimum_distance_between_sets(ligand_coords, residue_coords)
        if min_distance > radius:
            continue

        pocket_residues.append(
            {
                "residue": residue_key.to_dict(),
                "min_distance": min_distance,
                "atom_count": len(atom_records),
                "atoms": [record.to_dict() for record in atom_records],
            }
        )

    return pocket_residues


def minimum_distance_between_sets(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute the minimum inter-atomic distance between two coordinate sets."""
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    return float(math.sqrt(float(np.min(squared))))


def build_pocket_payload(
    *,
    ligand_group: LigandGroup,
    ligand_mol: Chem.Mol,
    ligand_atom_records: Sequence[AtomRecord],
    pocket_residues: Sequence[dict[str, object]],
    component_metadata: dict[str, dict[str, str]],
    ligand_sdf_path: Path,
    pocket_radius: float,
) -> dict[str, object]:
    """Build a JSON-serializable payload for one extracted ligand pocket."""
    component_names = sorted({record.residue_key.resname for record in ligand_atom_records})
    component_descriptions = {
        name: component_metadata.get(name, {})
        for name in component_names
    }

    return {
        "ligand_id": ligand_group.ligand_id,
        "ligand_residues": [key.to_dict() for key in ligand_group.residue_keys],
        "ligand_components": component_descriptions,
        "ligand_sdf": str(ligand_sdf_path),
        "ligand_atom_count": len(ligand_atom_records),
        "ligand_bond_count": ligand_mol.GetNumBonds(),
        "ligand_atoms": [record.to_dict() for record in ligand_atom_records],
        "ligand_bonds": ligand_bonds_to_dicts(ligand_mol, ligand_atom_records),
        "pocket_radius": pocket_radius,
        "pocket_residue_count": len(pocket_residues),
        "pocket_residues": list(pocket_residues),
    }


def ligand_bonds_to_dicts(
    ligand_mol: Chem.Mol,
    ligand_atom_records: Sequence[AtomRecord],
) -> list[dict[str, object]]:
    """Serialize ligand bonds with atom names and bond order."""
    atom_records = list(ligand_atom_records)
    bond_entries: list[dict[str, object]] = []
    for bond in ligand_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_atom = atom_records[begin_idx]
        end_atom = atom_records[end_idx]
        bond_entries.append(
            {
                "begin_atom_index": begin_idx,
                "end_atom_index": end_idx,
                "begin_atom_name": begin_atom.atom_name,
                "end_atom_name": end_atom.atom_name,
                "begin_residue": begin_atom.residue_key.to_dict(),
                "end_residue": end_atom.residue_key.to_dict(),
                "bond_type": str(bond.GetBondType()),
                "aromatic": bool(bond.GetIsAromatic()),
            }
        )
    return bond_entries


def write_ligand_sdf(mol: Chem.Mol, output_path: Path) -> None:
    """Write one ligand molecule to SDF."""
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()


def add_rdkit_bond(
    rw_mol: Chem.RWMol,
    atom_index_1: int,
    atom_index_2: int,
    *,
    order: str,
    aromatic: bool,
) -> None:
    """Add a bond if it does not already exist."""
    if atom_index_1 == atom_index_2:
        return
    if rw_mol.GetBondBetweenAtoms(atom_index_1, atom_index_2) is not None:
        return

    bond_type = map_template_bond_order(order, aromatic=aromatic)
    rw_mol.AddBond(atom_index_1, atom_index_2, bond_type)
    bond = rw_mol.GetBondBetweenAtoms(atom_index_1, atom_index_2)
    if bond is not None and (aromatic or bond_type == Chem.BondType.AROMATIC):
        bond.SetIsAromatic(True)
        rw_mol.GetAtomWithIdx(atom_index_1).SetIsAromatic(True)
        rw_mol.GetAtomWithIdx(atom_index_2).SetIsAromatic(True)


def map_template_bond_order(order: str, *, aromatic: bool) -> Chem.BondType:
    """Map mmCIF bond order tokens to RDKit bond types."""
    normalized = str(order).strip().lower()
    if aromatic or normalized in {"arom", "delo"}:
        return Chem.BondType.AROMATIC
    return {
        "sing": Chem.BondType.SINGLE,
        "doub": Chem.BondType.DOUBLE,
        "trip": Chem.BondType.TRIPLE,
        "quad": Chem.BondType.QUADRUPLE,
    }.get(normalized, Chem.BondType.SINGLE)


def map_openbabel_bond_order(order: int, *, aromatic: bool) -> Chem.BondType:
    """Map OpenBabel bond orders to RDKit bond types."""
    if aromatic:
        return Chem.BondType.AROMATIC
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }.get(int(order), Chem.BondType.SINGLE)


def sanitize_molecule_best_effort(mol: Chem.Mol) -> None:
    """Sanitize when possible but keep partially valid molecules writable."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)


def ligand_sort_key(residue_key: ResidueKey) -> tuple[str, int, str, str]:
    """Stable sorting key for residues and ligands."""
    return (
        residue_key.chain_id,
        residue_key.resseq,
        residue_key.insertion_code,
        residue_key.resname,
    )


def make_ligand_id(index: int, residue_keys: Sequence[ResidueKey]) -> str:
    """Create a filesystem-safe ligand identifier."""
    residue_fragment = "__".join(key.compact_label() for key in residue_keys)
    safe_fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", residue_fragment)
    return f"ligand_{index:03d}_{safe_fragment}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract ligand-centered pockets from a holo mmCIF using "
            "Biopython, RDKit, and OpenBabel fallback bond perception."
        )
    )
    parser.add_argument("input_path", help="Input holo preprocessed mmCIF/CIF file.")
    parser.add_argument(
        "--output-dir",
        help="Directory for extracted ligand SDFs and pocket JSON files.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_POCKET_RADIUS,
        help=f"Pocket radius in angstrom (default: {DEFAULT_POCKET_RADIUS}).",
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
        default=DEFAULT_MIN_HEAVY_ATOMS,
        help=(
            "Minimum heavy atoms required for a ligand candidate "
            f"(default: {DEFAULT_MIN_HEAVY_ATOMS})."
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
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    manifest = extract_pockets(args)
    print(f"Extracted {manifest['ligand_count']} ligand pocket(s) into {manifest['output_dir']}")
    return 0


def as_list(value: object | None) -> list[object]:
    """Normalize mmCIF dictionary values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_int(value: object) -> int:
    """Parse integer-like mmCIF fields."""
    text = str(value).strip()
    if text in {"", ".", "?"}:
        raise ValueError("Missing integer value.")
    return int(float(text))


def normalize_insertion_code(value: object) -> str:
    """Normalize mmCIF insertion codes so missing values become empty strings."""
    text = str(value).strip()
    if text in {"", ".", "?", " "}:
        return ""
    return text


if __name__ == "__main__":
    raise SystemExit(main())
