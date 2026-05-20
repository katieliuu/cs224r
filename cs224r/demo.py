"""
demo.py
Run one short episode and display the seed fragment, goal, and output molecule.

Usage:
  python demo.py [--n_frags 50] [--steps 4] [--seed 42]

Saves seed.png and output.png in the current directory.
"""
import _path_bootstrap  # noqa: F401

import argparse

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.QED import qed as _qed
from rdkit.Chem.Draw import rdMolDraw2D

from data import (
    load_fragment_library, load_target_distribution,
    denormalize_props, PROP_NAMES,
)
from env import MolEnv, TERMINATE, _normalise_brics_smiles
from chem.build.molgraph_to_mol import molgraph_to_mol


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_mol(mol: Chem.Mol, path: str, label: str = "", size: tuple = (400, 300)) -> None:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = False
    if label:
        drawer.drawOptions().legendFontSize = 14
    try:
        Chem.SanitizeMol(mol)
        drawer.DrawMolecule(mol, legend=label)
    except Exception:
        drawer.DrawMolecule(mol, legend=label)
    drawer.FinishDrawing()
    with open(path, "wb") as fh:
        fh.write(drawer.GetDrawingText())
    print(f"  Saved: {path}")


def _prop_table(raw: np.ndarray) -> str:
    names = ["LogP", "QED ", "TPSA"]
    units = ["    ", "    ", " Å² "]
    lines = []
    for nm, u, v in zip(names, units, raw):
        lines.append(f"    {nm} {u} {v:+.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(n_frags: int = 50, max_steps: int = 4, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data …")
    frags   = load_fragment_library(
        "/mnt/data/m3_20m/outputs/fragments.parquet",
        n=n_frags, min_count=5_000,
    )
    targets = load_target_distribution(
        "/mnt/data/m3_20m/outputs/parents.parquet",
        n=200,
    )

    env = MolEnv(frags, targets, max_steps=max_steps)

    print("\n─── Episode start ───────────────────────────────────────")
    state, goal_norm, valid_actions = env.reset()

    # --- Seed fragment ---
    seed_mg  = env._mol
    seed_mol = molgraph_to_mol(seed_mg, sanitize=False, remove_hs=False)
    seed_smi = Chem.MolToSmiles(seed_mol) if seed_mol else "(unknown)"

    print(f"\nSeed fragment SMILES : {seed_smi}")
    print(f"Open attachment sites: {env._open_labels()}")

    # --- Goal ---
    goal_raw = denormalize_props(goal_norm)
    print(f"\nTarget goal (normalised): [{', '.join(f'{v:.3f}' for v in goal_norm)}]")
    print(f"Target goal (raw):\n{_prop_table(goal_raw)}")

    if seed_mol:
        _draw_mol(seed_mol, "seed.png", label="Seed fragment")

    # --- Random-policy rollout ---
    print("\n─── Rollout ─────────────────────────────────────────────")
    done = False
    step = 0
    while not done:
        idx    = np.random.randint(len(valid_actions))
        action = valid_actions[idx]
        result = env.step(action)

        if action.is_terminate:
            tag = "terminate"
        else:
            tag = (f"attach frag#{action.frag_idx} "
                   f"mol_lbl={action.mol_label!r} frag_lbl={action.frag_label!r}")

        print(f"  step {step}: {tag}  reward={result.reward:.4f}  done={result.done}")
        step  += 1
        done   = result.done
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    # --- Output molecule ---
    print("\n─── Result ──────────────────────────────────────────────")
    achieved = result.info.get("achieved_goal")
    is_valid = result.info.get("valid", False)

    out_mg  = env._mol
    out_mol = molgraph_to_mol(out_mg, sanitize=True, remove_hs=True)
    if out_mol is None:
        out_mol = molgraph_to_mol(out_mg, sanitize=False, remove_hs=False)
    out_smi = Chem.MolToSmiles(out_mol) if out_mol else "(could not convert)"

    print(f"Output SMILES  : {out_smi}")
    print(f"Valid molecule : {is_valid}")

    if achieved is not None:
        raw = denormalize_props(achieved)
        print(f"Achieved (normalised): [{', '.join(f'{v:.3f}' for v in achieved)}]")
        print(f"Achieved (raw):\n{_prop_table(raw)}")
        dist = float(np.linalg.norm(achieved - goal_norm))
        print(f"\nL2 distance to goal  : {dist:.4f}")
        print(f"Per-property delta   :")
        for nm, g, a in zip(["LogP", "QED ", "TPSA"], goal_raw, raw):
            print(f"    {nm}  goal={g:+.3f}  achieved={a:+.3f}  Δ={a-g:+.3f}")
    else:
        print("(molecule did not produce valid properties)")

    if out_mol:
        label = f"Output | reward={result.reward:.3f}"
        _draw_mol(out_mol, "output.png", label=label)

    print("\nDone.  Images written to seed.png and output.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",  type=int, default=50)
    p.add_argument("--steps",    type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    run_demo(n_frags=args.n_frags, max_steps=args.steps, seed=args.seed)
