"""
plot_trajectory.py
Publication figure showing one trained-agent trajectory.

Layout
------
  Top row : molecule images, one per step, with step label and L2 below.
  Bottom  : L2 distance vs step as a line plot (science+ieee style).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import io
import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from env import load_fragment_library, load_target_distribution, denormalize_props
from env import MolEnv, TERMINATE
from env import compute_norm_properties
from models import Actor
from chem.build.molgraph_to_mol import molgraph_to_mol
from core.structs import MolGraph

# ── scienceplots style ─────────────────────────────────────────────────────

_SD = Path("/home/ayamin/projects/cs273b/asmani-branch/plot/scienceplots/styles")
plt.style.use([str(_SD / "science.mplstyle"), str(_SD / "journals" / "ieee.mplstyle")])


# ── molecule rendering ─────────────────────────────────────────────────────

def _cap_dummies(mol: Chem.Mol) -> Optional[Chem.Mol]:
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


def _clean_for_display(mol: Chem.Mol) -> Chem.Mol:
    """Strip atom-map numbers and isotopes so labels like H:16 don't appear."""
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    return rw.GetMol()


def mol_to_pil(mol: Optional[Chem.Mol], size=(300, 260)) -> Image.Image:
    blank = Image.new("RGB", size, (255, 255, 255))
    if mol is None:
        return blank
    try:
        mol = _clean_for_display(mol)
        Chem.SanitizeMol(mol, catchErrors=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().padding = 0.15
        drawer.drawOptions().bondLineWidth = 1.5
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        return blank


def mg_to_pil(mg: MolGraph, size=(300, 260)) -> Image.Image:
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return Image.new("RGB", size, (255, 255, 255))
    capped = _cap_dummies(mol)
    mol_to_draw = capped if capped is not None else mol
    try:
        Chem.SanitizeMol(mol_to_draw, catchErrors=True)
    except Exception:
        pass
    return mol_to_pil(mol_to_draw, size)


# ── episode rollout ────────────────────────────────────────────────────────

def run_episode(env, seed, actor, device, greedy=True):
    np.random.seed(seed)
    state, goal_norm, valid_actions = env.reset()

    frames = [{"mol": deepcopy(env._mol), "step": 0}]

    done = False
    while not done:
        af_np   = env.get_action_features(valid_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t    = torch.tensor(af_np,  dtype=torch.float32, device=device)
        with torch.no_grad():
            d = actor.action_dist(state_t, af_t)
            idx = int(d.probs.argmax().item()) if greedy else int(d.sample().item())

        result = env.step(valid_actions[idx])
        state  = result.state
        done   = result.done
        frames.append({"mol": deepcopy(env._mol), "step": len(frames),
                       "achieved": result.info.get("achieved_goal"),
                       "done": result.done})
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    return goal_norm, frames


# ── figure ─────────────────────────────────────────────────────────────────

def make_figure(goal_norm, frames, seed, out):
    from data import denormalize_props as _denorm
    n = len(frames)
    mol_size = (300, 240)

    # Per-step normalised properties and L2
    props_list = []
    l2s = []
    for f in frames:
        mol = molgraph_to_mol(f["mol"], sanitize=False, remove_hs=False)
        capped = _cap_dummies(mol) if mol is not None else None
        props  = compute_norm_properties(capped) if capped is not None else None
        if f.get("achieved") is not None:
            props = f["achieved"]
        props_list.append(props)
        l2s.append(float(np.linalg.norm(props - goal_norm)) if props is not None else float("nan"))

    # PIL images — always cap dummies for display
    pils = []
    for f in frames:
        mol    = molgraph_to_mol(f["mol"], sanitize=False, remove_hs=False)
        capped = _cap_dummies(mol) if mol is not None else None
        pils.append(mol_to_pil(capped, mol_size))

    goal_raw = _denorm(goal_norm)

    # ── layout: molecule strip (top) + L2 line plot (bottom) ──
    fig = plt.figure(figsize=(6.6, 4.4))
    gs  = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.6, 1.0],
        hspace=0.05,
        left=0.07, right=0.98, top=0.91, bottom=0.09,
    )

    # target label at top of figure
    fig.text(
        0.5, 0.97,
        rf"Target:  LogP $= {goal_raw[0]:+.2f}$,  QED $= {goal_raw[1]:.2f}$,  TPSA $= {goal_raw[2]:.1f}$ \AA$^2$",
        ha="center", va="top", fontsize=7,
    )

    # ── molecule strip ──
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=gs[0], wspace=0.02)

    for i, (pil, l2, props) in enumerate(zip(pils, l2s, props_list)):
        ax = fig.add_subplot(gs_top[i])
        ax.imshow(np.array(pil))
        ax.axis("off")

        # label: "Seed" for t=0, "t = N" otherwise
        title = "Seed" if i == 0 else rf"$t = {i}$"
        ax.set_title(title, fontsize=7, pad=2)

        # property text below molecule
        if props is not None:
            raw = _denorm(props)
            lines = [
                rf"$\ell_2 = {l2:.3f}$",
                rf"LogP $= {raw[0]:+.2f}$",
                rf"QED $= {raw[1]:.2f}$",
                rf"TPSA $= {raw[2]:.1f}$",
            ]
        else:
            lines = [""]
        ax.text(0.5, -0.02, "\n".join(lines),
                transform=ax.transAxes,
                ha="center", va="top", fontsize=5.5, linespacing=1.5)

        # arrow to next frame
        if i < n - 1:
            ax.annotate("", xy=(1.04, 0.52), xytext=(0.88, 0.52),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7))

    # ── L2 line plot ──
    ax2 = fig.add_subplot(gs[1])
    steps = list(range(n))
    ax2.plot(steps, l2s, color="k", lw=1.0, marker="o", markersize=2.5, clip_on=False)
    ax2.set_xlabel("Step")
    ax2.set_ylabel(r"$\ell_2$ to goal")
    ax2.set_xticks(steps)
    step_labels = ["Seed"] + [str(i) for i in steps[1:]]
    ax2.set_xticklabels(step_labels, fontsize=6)
    ax2.set_ylim(bottom=0)

    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_ep6000.pt")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n_frags",    type=int, default=200)
    p.add_argument("--out",        type=str, default=None)
    p.add_argument("--greedy",     action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cpu")

    frags   = load_fragment_library("/mnt/data/m3_20m/outputs/fragments.parquet",
                                    n=args.n_frags, min_count=5_000)
    targets = load_target_distribution("/mnt/data/m3_20m/outputs/parents.parquet", n=300)
    env     = MolEnv(frags, targets, max_steps=6)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    actor = Actor(hidden_dim=ckpt["config"].get("hidden_dim", 256)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    goal_norm, frames = run_episode(env, args.seed, actor, device, greedy=args.greedy)

    out = args.out or f"fig_trajectory_seed{args.seed}.png"
    make_figure(goal_norm, frames, seed=args.seed, out=out)


if __name__ == "__main__":
    main()
