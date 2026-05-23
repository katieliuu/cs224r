"""
env.py
MolEnv — gym-style environment for goal-conditioned fragment assembly.

Episode lifecycle
-----------------
reset() → initial seed fragment as MolGraph, sample goal g from target dist.
step(action) → attach a fragment at one open dummy site, or terminate.
                Terminal reward: -||φ_norm(m_capped) − g||₂ - 0.05·n_open_dummies

Action space (enumerated fresh each step)
-----------------------------------------
  actions[0]          → terminate
  actions[1:]         → (frag_idx, mol_label, frag_label) triples
                         filtered to BRICS-compatible (mol_type, frag_type) pairs.

BRICS compatibility
-------------------
Only type pairs that actually appear in the M3-20M attach_demos data are
allowed.  This reduces the action space from O(N_frags × 16²) to
O(N_frags × ~5) and eliminates chemically nonsensical bonds.

Terminal reward
---------------
  All remaining dummy atoms are capped with H before property computation
  so _finalise always produces a valid closed-shell molecule and the reward
  is always meaningful.  A small penalty (0.05 per remaining dummy) biases
  the agent toward filling all attachment sites.
"""
import _path_bootstrap  # noqa: F401

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem

from core.structs import MolGraph, BondType
from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_mol
from chem.merge.merge import merge_by_labels
from chem.dummy.query import dummy_indices

from .data import FragInfo, sample_target
from .features import (
    state_features, action_features, compute_norm_properties,
    smiles_to_fp, STATE_DIM, ACTION_FEAT_DIM,
)


# ---------------------------------------------------------------------------
# BRICS SMILES normalisation:  [5*] → [*:5]
# ---------------------------------------------------------------------------

_BRICS_RE = re.compile(r"\[(\d+)\*\]")


def _normalise_brics_smiles(smiles: str) -> str:
    """Convert [N*] → [*:N] so atom-map labels propagate correctly."""
    return _BRICS_RE.sub(r"[*:\1]", smiles)


# ---------------------------------------------------------------------------
# BRICS compatibility table
# ---------------------------------------------------------------------------
# Derived from the M3-20M attach_demos.parquet (attach_edge[:, 2:4]).
# Maps mol_dummy_type → {valid frag_dummy_types}.
# Only pairs that appear in real drug-like BRICS bonds are included.

BRICS_COMPAT: Dict[int, Set[int]] = {
    1:  {3, 5, 10},
    3:  {1, 4, 13, 14, 15, 16},
    4:  {3, 5, 11},
    5:  {1, 4, 12, 13, 14, 15, 16},
    6:  {13, 14, 15, 16},
    7:  {7},
    8:  {9, 10, 13, 14, 15, 16},
    9:  {8, 13, 14, 15, 16},
    10: {1, 8, 13, 14, 15, 16},
    11: {4, 13, 14, 15, 16},
    12: {5},
    13: {3, 5, 6, 8, 9, 10, 11, 14, 15, 16},
    14: {3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16},
    15: {3, 5, 6, 8, 9, 10, 11, 13, 14, 16},
    16: {3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16},
}

OPEN_DUMMY_PENALTY = 0.05   # reward penalty per remaining dummy at termination


def _brics_compatible(mol_label: str, frag_label: str) -> bool:
    """Return True iff (mol_type, frag_type) is a valid BRICS bond pair."""
    try:
        mt = int(mol_label)
        ft = int(frag_label)
    except (ValueError, TypeError):
        return False
    return ft in BRICS_COMPAT.get(mt, set())


# ---------------------------------------------------------------------------
# Dummy capping helper (used in _finalise)
# ---------------------------------------------------------------------------

def _cap_dummies_with_h(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Replace every remaining dummy atom (atomic_num=0) with an explicit H,
    then sanitise.  Ensures _finalise always produces a closed-shell mol.
    """
    try:
        rw = Chem.RWMol(Chem.Mol(mol))
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(rw, catchErrors=True)
        return rw.GetMol()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Action descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    is_terminate: bool
    frag_idx: int = -1
    mol_label: str = ""
    frag_label: str = ""


TERMINATE = Action(is_terminate=True)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MolEnv:
    def __init__(
        self,
        fragment_library: List[FragInfo],
        target_distribution: np.ndarray,   # (N, GOAL_DIM) normalised
        max_steps: int = 8,
    ):
        self.library = fragment_library
        self.targets = target_distribution
        self.max_steps = max_steps

        self.frag_fps: List[np.ndarray] = [
            smiles_to_fp(_normalise_brics_smiles(f.smiles)) for f in fragment_library
        ]
        self._frag_graphs: Dict[int, Optional[MolGraph]] = {}

        self._mol:  Optional[MolGraph]    = None
        self._goal: Optional[np.ndarray] = None
        self._step: int                   = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray, List[Action]]:
        for _ in range(len(self.library)):
            idx = np.random.randint(len(self.library))
            mg  = self._frag_graph(idx)
            if mg is not None and dummy_indices(mg).size > 0:
                break
        else:
            raise RuntimeError("No valid seed fragment found in library.")

        self._mol  = mg
        self._goal = sample_target(self.targets)
        self._step = 0

        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        return state, self._goal.copy(), valid

    def step(self, action: Action) -> StepResult:
        self._step += 1

        if action.is_terminate or self._step >= self.max_steps:
            return self._finalise()

        frag_mg = self._frag_graph(action.frag_idx)
        if frag_mg is None:
            return self._soft_fail("fragment_load_failed")

        try:
            merged = merge_by_labels(
                self._mol,
                frag_mg,
                label_a=action.mol_label,
                label_b=action.frag_label,
                bond_type_code=BondType.SINGLE,
                validate_with_rdkit=False,
            )
        except Exception as exc:
            return self._soft_fail(str(exc))

        self._mol = merged

        if not self._open_labels():
            return self._finalise()

        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        return StepResult(state=state, reward=0.0, done=False,
                          info={"valid_actions": valid})

    # ------------------------------------------------------------------
    # Action enumeration
    # ------------------------------------------------------------------

    def _open_labels(self) -> List[str]:
        g = self._mol
        if g is None:
            return []
        d_idx = dummy_indices(g)
        if d_idx.size == 0:
            return []
        lab_arr = g.arrays.attachment_label
        labels = []
        for i in d_idx:
            i   = int(i)
            lbl = lab_arr[i] if lab_arr is not None else None
            labels.append(str(lbl) if lbl is not None else "1")
        return labels

    def _valid_actions(self) -> List[Action]:
        """
        Enumerate BRICS-compatible attach actions + terminate.
        Only (mol_label, frag_label) pairs that appear in the real
        BRICS compatibility table are proposed.
        """
        actions: List[Action] = [TERMINATE]
        mol_labels = self._open_labels()
        if not mol_labels:
            return actions

        seen: set = set()
        for fi, frag in enumerate(self.library):
            for flab in frag.labels:
                for mlab in mol_labels:
                    if not _brics_compatible(mlab, flab):
                        continue
                    key = (fi, flab, mlab)
                    if key not in seen:
                        seen.add(key)
                        actions.append(Action(
                            is_terminate=False,
                            frag_idx=fi,
                            mol_label=mlab,
                            frag_label=flab,
                        ))
        return actions

    def get_action_features(self, actions: List[Action]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for a in actions:
            if a.is_terminate:
                rows.append(np.zeros(ACTION_FEAT_DIM, dtype=np.float32))
            else:
                rows.append(action_features(
                    self.frag_fps[a.frag_idx], a.frag_label, a.mol_label
                ))
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Terminal helper
    # ------------------------------------------------------------------

    def _finalise(self) -> StepResult:
        """
        Compute terminal reward.
        1. Cap any remaining dummy atoms with H (always yields a closed mol).
        2. Compute normalised properties on the capped molecule.
        3. reward = -||props - goal||₂  -  0.05 * n_open_dummies
        """
        n_open = len(self._open_labels())

        # Get base mol (unsanitised, with dummies).
        mol = molgraph_to_mol(self._mol, sanitize=False, remove_hs=False)

        # Cap dummies → closed shell molecule.
        closed = _cap_dummies_with_h(mol) if mol is not None else None

        # If capping failed, try sanitising the original as a last resort.
        if closed is None and mol is not None:
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
                closed = mol
            except Exception:
                pass

        props_norm = compute_norm_properties(closed) if closed is not None else None
        open_pen   = OPEN_DUMMY_PENALTY * n_open
        state      = state_features(self._mol, self._goal)

        if props_norm is None:
            return StepResult(
                state=state, reward=-2.0 - open_pen, done=True,
                info={"valid": False, "achieved_goal": None,
                      "n_open_dummies": n_open, "valid_actions": [TERMINATE]},
            )

        dist = float(np.linalg.norm(props_norm - self._goal))
        return StepResult(
            state=state,
            reward=-(dist + open_pen),
            done=True,
            info={"valid": True, "achieved_goal": props_norm.copy(),
                  "n_open_dummies": n_open, "valid_actions": [TERMINATE]},
        )

    def _soft_fail(self, reason: str) -> StepResult:
        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        return StepResult(state=state, reward=-0.1, done=False,
                          info={"error": reason, "valid_actions": valid})

    # ------------------------------------------------------------------
    # Fragment graph cache
    # ------------------------------------------------------------------

    def _frag_graph(self, idx: int) -> Optional[MolGraph]:
        if idx not in self._frag_graphs:
            norm_smi = _normalise_brics_smiles(self.library[idx].smiles)
            self._frag_graphs[idx] = smiles_to_molgraph(norm_smi)
        return self._frag_graphs[idx]

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_feat_dim(self) -> int:
        return ACTION_FEAT_DIM
