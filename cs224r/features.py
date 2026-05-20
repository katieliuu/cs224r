"""
features.py
State featurisation, property computation, and action feature vectors.

State  = concat(mol_fp: FP_BITS, goal_norm: GOAL_DIM)
Action = concat(frag_fp: FP_BITS, one_hot(brics_frag): BRICS_DIM, one_hot(brics_mol): BRICS_DIM)
"""
import _path_bootstrap  # noqa: F401

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.QED import qed as _rdkit_qed

from core.structs import MolGraph
from chem.build.molgraph_to_mol import molgraph_to_mol
from data import normalize_props, GOAL_DIM

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
FP_BITS = 512
BRICS_DIM = 17          # BRICS types 0-16 (0 = unknown/fallback)
ACTION_FEAT_DIM = FP_BITS + 2 * BRICS_DIM   # 512 + 34 = 546
STATE_DIM = FP_BITS + GOAL_DIM              # 512 + 3 = 515


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _mol_to_fp(mol: Chem.Mol) -> np.ndarray:
    try:
        # Partial sanitize so ring-info / aromaticity are available for Morgan FP.
        Chem.SanitizeMol(mol, catchErrors=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(FP_BITS, dtype=np.float32)


def molgraph_to_fp(mg: MolGraph) -> np.ndarray:
    """Morgan FP of a (possibly partial) MolGraph. Dummies (atomic_num=0) are kept."""
    if mg is None or mg.arrays is None or mg.arrays.atomic_num.shape[0] == 0:
        return np.zeros(FP_BITS, dtype=np.float32)
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return _fallback_count_fp(mg)
    return _mol_to_fp(mol)


def _fallback_count_fp(mg: MolGraph) -> np.ndarray:
    """Atom-count feature vector padded to FP_BITS when RDKit conversion fails."""
    feat = np.zeros(FP_BITS, dtype=np.float32)
    z = mg.arrays.atomic_num
    for i, elem in enumerate([6, 7, 8, 9, 15, 16, 17, 35, 53]):
        feat[i] = float(np.sum(z == elem))
    feat[9] = float(z.shape[0])
    return feat


def smiles_to_fp(smiles: str) -> np.ndarray:
    """Morgan FP from a SMILES string (handles dummy atoms)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    return _mol_to_fp(mol)


# ---------------------------------------------------------------------------
# BRICS type one-hot encoding
# ---------------------------------------------------------------------------

def brics_onehot(label: str) -> np.ndarray:
    v = np.zeros(BRICS_DIM, dtype=np.float32)
    try:
        idx = int(label)
        if 0 <= idx < BRICS_DIM:
            v[idx] = 1.0
    except (ValueError, TypeError):
        pass
    return v


# ---------------------------------------------------------------------------
# State and action features
# ---------------------------------------------------------------------------

def state_features(mg: MolGraph, goal_norm: np.ndarray) -> np.ndarray:
    """Concatenate mol fingerprint and normalised goal vector → (STATE_DIM,)."""
    fp = molgraph_to_fp(mg)
    return np.concatenate([fp, goal_norm.astype(np.float32)])


def action_features(frag_fp: np.ndarray, frag_label: str, mol_label: str) -> np.ndarray:
    """Feature vector for one candidate attach action → (ACTION_FEAT_DIM,)."""
    return np.concatenate([
        frag_fp,
        brics_onehot(frag_label),
        brics_onehot(mol_label),
    ])


# ---------------------------------------------------------------------------
# Property computation (on complete RDKit molecules)
# ---------------------------------------------------------------------------

def compute_raw_properties(mol: Chem.Mol) -> Optional[np.ndarray]:
    """Return (sLogP, QED, TPSA) as float32 array, or None on failure."""
    try:
        logp = Descriptors.MolLogP(mol)
        q    = _rdkit_qed(mol)
        tpsa = Descriptors.TPSA(mol)
        return np.array([logp, q, tpsa], dtype=np.float32)
    except Exception:
        return None


def compute_norm_properties(mol: Chem.Mol) -> Optional[np.ndarray]:
    """Normalised [0,1] property vector, or None on failure."""
    raw = compute_raw_properties(mol)
    return None if raw is None else normalize_props(raw)
