# -*- coding: utf-8 -*-

from chemgraph.cleaning.clean_molecule import MoleculeCleaner
from rdkit import Chem
from rdkit.Chem import rdmolops, SanitizeFlags
import numpy as np
from ..core.mol_schema import MolGraph, Editability

from chemgraph.building.helpers.valence_helpers import _default_valence

__all__ = ['clean_molecule', 'clean_graph_data']

def clean_molecule(mol: Chem.Mol,
                   preserve_aromaticity: bool = False,
                   verbose: bool = False) -> Chem.Mol:
    """
    Perform a single MoleculeCleaner pass. No direct RDKit sanitization.
    Raises ValueError if it can't repair the molecule.
    """
    cleaner = MoleculeCleaner(preserveAromaticity=preserve_aromaticity,
                              return_log=False,
                              verbose=verbose)
    cleaned = cleaner.clean(mol)
    if cleaned is None:
        raise ValueError("MoleculeCleaner failed: {}".format(Chem.MolToSmiles(mol)))
    return cleaned

"""
DO NOT USE: OLD METHOD
"""
def clean_graph_data(graph: MolGraph) -> MolGraph:
    """
    DO NOT USE: OLD METHOD [Note 3]
    Convert MolGraph -> Mol -> clean_molecule -> fully sanitize/kekulize ->
    MolGraph, preserving and remapping original metadata & attachment points.
    """
    from ..building.graph_to_mol import graph_to_mol
    from ..building.mol_to_graph import mol_to_graph_data_obj
    
    coords = graph.meta["coords"]
    
    # 1) Build & clean
    mol = graph_to_mol(graph)
    mol_clean = clean_molecule(mol)

    # 2) Fully sanitize except kekulize
    flags = SanitizeFlags.SANITIZE_ALL & ~SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(mol_clean, flags)

    # 3) Explicitly perceive aromaticity & conjugation
    rdmolops.SetAromaticity(mol_clean)
    rdmolops.SetConjugation(mol_clean)

    # 4) Kekulize (with aromatic flags preserved)
    Chem.Kekulize(mol_clean, clearAromaticFlags=False)

    # 5) SMILES from the now fully valid Mol
    smiles = Chem.MolToSmiles(mol_clean, canonical=True)

    # 6) Rebuild MolGraph
    cleaned_graph = mol_to_graph_data_obj(smiles, coords=coords)

    # 7) Merge & override metadata
    merged_meta = cleaned_graph.meta.copy()
    for k, v in graph.meta.items():
        if v is not None:
            merged_meta[k] = v

    # 8) Re-map attachment points by label_to_index
    label_to_index = merged_meta.get("label_to_index", {})
    remapped_aps = []
    for ap in graph.mutation_attachment_points:
        new_idx = label_to_index.get(ap.label)
        if new_idx is None:
            continue
        new_target = label_to_index.get(ap.target) if ap.target is not None else None
        ap.idx = new_idx
        ap.target = new_target
        remapped_aps.append(ap)

    # 9) Build fresh editability
    nodes = cleaned_graph.nodes
    edges = cleaned_graph.edges
    editability = Editability(
        atom_scores=np.zeros(len(next(iter(nodes.values()))), dtype=np.float32),
        edge_scores=np.zeros(edges.shape[0], dtype=np.float32),
    )

    # 10) Return a brand-new MolGraph
    return MolGraph(
        nodes=nodes,
        electronic_nodes=cleaned_graph.electronic_nodes,
        edges=edges,
        edge_features=cleaned_graph.edge_features,
        edge_electronic_features=cleaned_graph.edge_electronic_features,
        mutation_attachment_points=remapped_aps,
        mutation_fragments=cleaned_graph.mutation_fragments,
        mutation_history=cleaned_graph.mutation_history,
        mutation_editability=editability,
        meta=merged_meta,
    )

"""
DO NOT USE: OLD METHOD [Note 4]
"""
def _find_overloaded_atoms(graph: MolGraph) -> list[int]:
    """
    DO NOT USE: OLD METHOD
    Identify atoms whose current total bond order exceeds their standard valence.
    """
    atomic_nums = graph.nodes["atomic_num"]
    N = atomic_nums.shape[0]

    u_indices = graph.edges[:, 0]
    bond_types = graph.edge_features["bond_type"]

    order_map = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
    bond_orders = np.fromiter(
        (order_map.get(bt, 1) for bt in bond_types),
        dtype=np.int32,
        count=bond_types.shape[0]
    )

    valence_counts = np.bincount(u_indices, weights=bond_orders, minlength=N)

    offenders = []
    for idx, Z in enumerate(atomic_nums):
        if Z == 0:
            continue  # skip dummies
        if valence_counts[idx] > _default_valence(int(Z)):
            offenders.append(idx)

    return offenders