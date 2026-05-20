"""
trajectory.py
Visualise the full assembly trajectory for a handful of seeds.

For each seed, produces one figure (trajectory_seed{N}.png) showing:
  - Every intermediate molecule in the episode (column per step)
  - At each state: partial property estimates (dummies capped with H)
  - The target property vector (constant across the episode)
  - The normalised L2 distance to the target at each step

Partial property estimation
---------------------------
Completed molecules have all dummy atoms removed by RDKit, so standard
RDKit descriptors apply.  For *partial* states (dummies still present),
we substitute every dummy atom (atomic_num=0) with an explicit H before
computing properties.  This gives a lower-bound / fragment-contribution
approximation rather than the true final value.

Usage
-----
  python trajectory.py [--n_frags 50] [--max_steps 6] [--seeds 7 42 99]
"""
import _path_bootstrap  # noqa: F401

import argparse
import io
from copy import deepcopy
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.QED import qed as _rdkit_qed
from rdkit.Chem.Draw import rdMolDraw2D

from data import (
    load_fragment_library, load_target_distribution,
    denormalize_props, PROP_NAMES,
)
from env import MolEnv, TERMINATE, _normalise_brics_smiles
from features import compute_norm_properties, normalize_props
from chem.build.molgraph_to_mol import molgraph_to_mol
from core.structs import MolGraph


# ---------------------------------------------------------------------------
# Partial property computation (dummies capped with H)
# ---------------------------------------------------------------------------

def _cap_dummies(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Replace every dummy atom (atomic_num=0) with H, then sanitise."""
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


def partial_properties(mg: MolGraph) -> Optional[np.ndarray]:
    """
    Estimate (sLogP, QED, TPSA) on a partial molecule by capping dummies
    with H.  Returns normalised [0,1] array, or None on failure.
    """
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return None
    capped = _cap_dummies(mol)
    if capped is None:
        return None
    return compute_norm_properties(capped)


# ---------------------------------------------------------------------------
# Molecule → PIL image
# ---------------------------------------------------------------------------

def mol_to_pil(mol: Optional[Chem.Mol], size: Tuple[int, int] = (300, 240)) -> Image.Image:
    if mol is None:
        img = Image.new("RGB", size, (240, 240, 240))
        return img
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        Chem.SanitizeMol(mol, catchErrors=True)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        img = Image.new("RGB", size, (240, 240, 240))
        return img


def molgraph_to_pil(mg: MolGraph, size: Tuple[int, int] = (300, 240)) -> Image.Image:
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return Image.new("RGB", size, (240, 240, 240))
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    return mol_to_pil(mol, size)


# ---------------------------------------------------------------------------
# Episode data collection
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class StepSnapshot:
    step:        int
    action_desc: str
    mol_graph:   MolGraph
    props_norm:  Optional[np.ndarray]   # normalised, None if not computable
    reward:      float
    done:        bool
    n_open_sites: int


def run_episode(env: MolEnv, seed: int) -> Tuple[np.ndarray, List[StepSnapshot]]:
    """
    Roll out one episode using a random policy and record every step.
    Returns (goal_norm, snapshots).
    """
    np.random.seed(seed)
    state, goal_norm, valid_actions = env.reset()

    # Snapshot t=0: seed fragment
    snapshots: List[StepSnapshot] = [StepSnapshot(
        step=0,
        action_desc="seed",
        mol_graph=deepcopy(env._mol),
        props_norm=partial_properties(env._mol),
        reward=0.0,
        done=False,
        n_open_sites=len(env._open_labels()),
    )]

    done = False
    step = 0
    while not done:
        idx    = np.random.randint(len(valid_actions))
        action = valid_actions[idx]
        result = env.step(action)
        step  += 1

        if action.is_terminate:
            desc = "terminate"
        else:
            desc = (f"attach frag#{action.frag_idx}\n"
                    f"mol←{action.mol_label!r} frag←{action.frag_label!r}")

        props = partial_properties(env._mol)
        if result.done and result.info.get("achieved_goal") is not None:
            props = result.info["achieved_goal"]   # exact terminal props

        # For the terminal frame: cap dummies so the displayed molecule matches
        # what was actually scored (no dangling *:N attachment points).
        display_mg = deepcopy(env._mol)
        if result.done:
            raw_mol = molgraph_to_mol(display_mg, sanitize=False, remove_hs=False)
            capped  = _cap_dummies(raw_mol) if raw_mol is not None else None
            if capped is not None:
                from chem.build.create_molgraph import smiles_to_molgraph as _s2mg
                from rdkit.Chem import MolToSmiles as _smi
                try:
                    capped_smi  = _smi(capped)
                    display_mg  = _s2mg(capped_smi) or display_mg
                except Exception:
                    pass   # fall back to raw if conversion fails

        snapshots.append(StepSnapshot(
            step=step,
            action_desc=desc,
            mol_graph=display_mg,
            props_norm=props,
            reward=result.reward,
            done=result.done,
            n_open_sites=len(env._open_labels()),
        ))

        done = result.done
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    return goal_norm, snapshots


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

PROP_LABELS = ["LogP", "QED", "TPSA(norm)"]
PROP_UNITS  = ["", "", ""]

# Colour thresholds: green if |delta| < 0.1, amber if < 0.25, red otherwise
def _delta_colour(delta: float) -> str:
    a = abs(delta)
    if a < 0.10: return "#2ecc71"
    if a < 0.25: return "#f39c12"
    return "#e74c3c"


def _prop_block(
    ax: plt.Axes,
    props_norm: Optional[np.ndarray],
    goal_norm:  np.ndarray,
    label:      str,
    is_target:  bool = False,
) -> None:
    """Render a property table into a matplotlib Axes (text only, no ticks)."""
    ax.axis("off")
    lines = [label, ""]

    if is_target:
        raw = denormalize_props(goal_norm)
        raw_names = ["LogP", "QED", "TPSA"]
        raw_units = ["", "", " Å²"]
        for nm, u, vn, vr in zip(raw_names, raw_units, goal_norm, raw):
            lines.append(f"{nm}: {vr:+.3f}{u}  [{vn:.3f}]")
        ax.text(0.5, 0.5, "\n".join(lines),
                transform=ax.transAxes, va="center", ha="center",
                fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#dce8f5", ec="#3498db", lw=1.5))
        return

    if props_norm is None:
        lines.append("(no props)")
        ax.text(0.5, 0.5, "\n".join(lines),
                transform=ax.transAxes, va="center", ha="center",
                fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#aaa"))
        return

    raw_props = denormalize_props(props_norm)
    raw_goal  = denormalize_props(goal_norm)
    raw_names = ["LogP", "QED", "TPSA"]
    raw_units = ["", "", " Å²"]

    dist = float(np.linalg.norm(props_norm - goal_norm))
    lines.append(f"L2 dist: {dist:.3f}")
    lines.append("")
    for nm, u, pn, pr, gn, gr in zip(
        raw_names, raw_units,
        props_norm, raw_props,
        goal_norm,  raw_goal,
    ):
        delta = pn - gn
        sign  = "+" if delta >= 0 else ""
        lines.append(f"{nm}: {pr:+.3f}{u} [Δ={sign}{delta:.3f}]")

    dist_colour = _delta_colour(dist)
    ax.text(0.5, 0.5, "\n".join(lines),
            transform=ax.transAxes, va="center", ha="center",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9",
                      ec=dist_colour, lw=2.0))


def render_trajectory(
    goal_norm:  np.ndarray,
    snapshots:  List[StepSnapshot],
    seed:       int,
    out_path:   str,
    mol_size:   Tuple[int, int] = (300, 240),
):
    """
    Layout per step row (3 columns):
      [step info + action text] | [molecule image] | [property block]

    Each piece has its own dedicated cell so nothing ever overlaps.
    """
    n_rows = 1 + len(snapshots)   # header + one row per snapshot
    n_cols = 3

    fig_w = 12.0
    row_h  = 3.2
    fig_h  = row_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=[0.9, 1.6, 1.2],
        hspace=0.12,
        wspace=0.08,
        left=0.02, right=0.98,
        top=0.98, bottom=0.01,
    )

    # ── Header row ─────────────────────────────────────────────────────────
    ax_head = fig.add_subplot(gs[0, :])   # span all 3 cols
    ax_head.axis("off")
    goal_raw = denormalize_props(goal_norm)
    head_txt = (
        f"Trajectory  (seed={seed})\n"
        f"TARGET:  LogP={goal_raw[0]:+.3f}    QED={goal_raw[1]:.3f}    "
        f"TPSA={goal_raw[2]:.1f} Å²    "
        f"(norm [{', '.join(f'{v:.3f}' for v in goal_norm)}])"
    )
    ax_head.text(
        0.5, 0.5, head_txt,
        transform=ax_head.transAxes,
        va="center", ha="center",
        fontsize=10, fontweight="bold", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#dce8f5", ec="#2980b9", lw=2),
    )

    # ── One row per snapshot ────────────────────────────────────────────────
    for row_idx, snap in enumerate(snapshots):
        r = row_idx + 1

        # ── Col 0: step label + action text ──
        ax_info = fig.add_subplot(gs[r, 0])
        ax_info.axis("off")

        if snap.step == 0:
            top_line   = "t = 0  (seed)"
            action_txt = ""
            bg_colour  = "#eaf7ea"
            border     = "#27ae60"
        elif snap.done:
            top_line   = f"t = {snap.step}   DONE"
            action_txt = f"reward = {snap.reward:.4f}\nopen sites = {snap.n_open_sites}"
            bg_colour  = "#fdecea"
            border     = "#c0392b"
        else:
            top_line   = f"t = {snap.step}   open = {snap.n_open_sites}"
            action_txt = snap.action_desc
            bg_colour  = "#fef9e7"
            border     = "#f39c12"

        info_txt = top_line
        if action_txt:
            info_txt += "\n\n" + action_txt

        ax_info.text(
            0.5, 0.5, info_txt,
            transform=ax_info.transAxes,
            va="center", ha="center",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=bg_colour, ec=border, lw=1.5),
        )

        # ── Col 1: molecule image ──
        ax_mol = fig.add_subplot(gs[r, 1])
        ax_mol.axis("off")
        pil_img = molgraph_to_pil(snap.mol_graph, size=mol_size)
        ax_mol.imshow(np.array(pil_img))

        # ── Col 2: property block ──
        ax_prop = fig.add_subplot(gs[r, 2])
        block_label = "partial  (dummies → H)" if not snap.done else "FINAL  (exact)"
        _prop_block(ax_prop, snap.props_norm, goal_norm, block_label)

    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_frags: int = 50, max_steps: int = 6, seeds: List[int] = None):
    if seeds is None:
        seeds = [7, 42, 99]

    print("Loading data …")
    frags   = load_fragment_library(
        "/mnt/data/m3_20m/outputs/fragments.parquet",
        n=n_frags, min_count=5_000,
    )
    targets = load_target_distribution(
        "/mnt/data/m3_20m/outputs/parents.parquet",
        n=300,
    )

    env = MolEnv(frags, targets, max_steps=max_steps)

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────")
        goal_norm, snapshots = run_episode(env, seed)

        goal_raw = denormalize_props(goal_norm)
        print(f"  Goal: LogP={goal_raw[0]:+.3f}  QED={goal_raw[1]:.3f}  "
              f"TPSA={goal_raw[2]:.1f} Å²")
        print(f"  Steps: {len(snapshots)-1}  "
              f"(+1 seed state = {len(snapshots)} total snapshots)")

        for snap in snapshots:
            if snap.props_norm is not None:
                dist = float(np.linalg.norm(snap.props_norm - goal_norm))
                dist_s = f"L2={dist:.3f}"
            else:
                dist_s = "L2=N/A"
            tag = "DONE" if snap.done else "    "
            print(f"    t={snap.step} {tag}  {dist_s}  "
                  f"open_sites={snap.n_open_sites}  r={snap.reward:.3f}")

        out = f"trajectory_seed{seed}.png"
        render_trajectory(goal_norm, snapshots, seed=seed, out_path=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",  type=int,   default=50)
    p.add_argument("--max_steps",type=int,   default=6)
    p.add_argument("--seeds",    type=int,   nargs="+", default=[7, 42, 99])
    args = p.parse_args()
    main(n_frags=args.n_frags, max_steps=args.max_steps, seeds=args.seeds)
