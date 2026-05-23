"""
chem/build/create_molgraph.py

Minimal RDKit → MolGraph (structs.py) conversion.

Provides two entry points:
- mol_to_molgraph(mol) — convert RDKit Mol to MolGraph
- smiles_to_molgraph(smiles) — convenience wrapper for SMILES strings
"""

from __future__ import annotations

from typing import Optional
from rdkit import Chem

from core.structs import MolGraph, MolArrays
from chem.build.build_utils import (
    extract_atom_arrays,
    extract_bond_arrays,
    extract_attachment_points,
    prepare_molecule,
)
from utils.logger import get_logger

logger = get_logger(__name__)
logger.debug("Loading create_molgraph.py...")


def mol_to_molgraph(
    mol: Chem.Mol,
    compute_charges: bool = True,
    keep_rdkit_mol: bool = False,
) -> MolGraph:
    """Convert an RDKit Mol object to a MolGraph."""
    logger.debug("mol_to_molgraph() called")
    
    logger.debug("Extracting atom arrays...")
    atom_data = extract_atom_arrays(mol, compute_charges=compute_charges)
    logger.debug("Atom arrays extracted")

    logger.debug("Extracting bond arrays...")
    bond_data = extract_bond_arrays(mol)
    logger.debug("Bond arrays extracted")

    logger.debug("Building MolArrays...")
    arrays = MolArrays(
        # Required fields first (positional order matters)
        atomic_num=atom_data["atomic_num"],
        formal_charge=atom_data["formal_charge"],
        bonds=bond_data["bonds"],
        bond_type=bond_data["bond_type"],
        # Optional atom fields
        isotope=atom_data["isotope"],
        is_aromatic=atom_data["is_aromatic"],
        hybridization=atom_data["hybridization"],
        chiral_tag=atom_data["chiral_tag"],
        atom_map=atom_data["atom_map"],
        attachment_label=atom_data["attachment_label"],
        explicit_h=atom_data["explicit_h"],
        implicit_h=atom_data["implicit_h"],
        partial_charge=atom_data["partial_charge"],
        pos=atom_data["pos"],
        # Optional bond fields
        is_conjugated=bond_data["is_conjugated"],
        is_in_ring=bond_data["is_in_ring"],
        bond_dir=bond_data["bond_dir"],
        bond_stereo=bond_data["bond_stereo"],
        bond_resonance_type=bond_data["bond_resonance_type"],
        # Globals
        total_charge=Chem.GetFormalCharge(mol),
    )
    logger.debug("MolArrays built")

    logger.debug("Extracting attachment points...")
    attachments = extract_attachment_points(mol)
    logger.debug("Attachment points extracted")

    logger.debug("Building MolGraph...")
    smiles = Chem.MolToSmiles(mol)
    logger.debug(f"SMILES: {smiles}")
    
    result = MolGraph(
        arrays=arrays,
        attachments=attachments,
        rdkit_mol=mol if keep_rdkit_mol else None,
        meta={"smiles": smiles},
    )
    logger.debug("mol_to_molgraph() complete")
    return result


def smiles_to_molgraph(
    smiles: str,
    add_hs: bool = False,
    compute_charges: bool = True,
    keep_rdkit_mol: bool = False,
) -> Optional[MolGraph]:
    """
    Convenience function: SMILES string → MolGraph.
    
    Uses MoleculeCleaner to parse and sanitize the SMILES string, then converts
    to a MolGraph with atom/bond arrays.
    
    Args:
        smiles: SMILES string
        add_hs: Whether to add explicit hydrogens
        compute_charges: Whether to compute Gasteiger partial charges
        keep_rdkit_mol: Whether to store the RDKit mol in the output
        
    Returns:
        MolGraph or None if conversion fails
    """
    logger.debug(f"smiles_to_molgraph() called with: {smiles}")
    
    try:
        mol, repair_log = prepare_molecule(
            smiles,
            add_hs=add_hs,
            compute_charges=compute_charges,
        )
    except ValueError as e:
        logger.error(f"Failed to prepare molecule: {e}")
        return None
    
    logger.debug("Molecule prepared successfully")
    
    # Convert to MolGraph - charges are already on mol as properties,
    # extract_atom_arrays will find them
    result = mol_to_molgraph(
        mol,
        compute_charges=compute_charges,  # Will extract existing charges or compute if needed
        keep_rdkit_mol=keep_rdkit_mol,
    )
    
    # Store repair log and original SMILES in meta
    if result is not None:
        result.meta["input_smiles"] = smiles
        result.meta["repair_log"] = repair_log
    
    logger.debug("smiles_to_molgraph() complete")
    return result


logger.debug("create_molgraph.py loaded successfully")