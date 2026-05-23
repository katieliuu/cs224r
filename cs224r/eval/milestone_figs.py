"""
make_milestone_figs.py
Generate all milestone 1 figures into milestone1_figs/{baseline,exp1,exp2,combined}/.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import re
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_SD = Path("/home/ayamin/projects/cs273b/asmani-branch/plot/scienceplots/styles")
plt.style.use([str(_SD / "science.mplstyle"), str(_SD / "journals" / "ieee.mplstyle")])

_ROOT = _Path(__file__).resolve().parent.parent
OUT  = _ROOT / "results" / "milestone1_figs"
BASE = OUT / "baseline"
E1   = OUT / "exp1"
E2   = OUT / "exp2"
COMB = OUT / "combined"
for d in (BASE, E1, E2, COMB):
    d.mkdir(parents=True, exist_ok=True)

C0 = "k"   # A2C+HER
C1 = "r"   # PPO
C2 = "b"   # GNN
C3 = "g"   # random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG_RE = re.compile(
    r"\[\s*(\d+)\].*dist=(\d+\.\d+).*actor=([+-]?\d+\.\d+)"
    r".*critic=(\d+\.\d+).*H=(\d+\.\d+)"
)

def parse_log(path):
    ep, dist, entropy = [], [], []
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                ep.append(int(m.group(1)))
                dist.append(float(m.group(2)))
                entropy.append(float(m.group(5)))
    return np.array(ep), np.array(dist), np.array(entropy)


def load_val(path, every=1000):
    try:
        raw  = json.loads(Path(path).read_text())
        pairs = sorted([(int(k), v) for k, v in raw.items() if int(k) % every == 0])
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    except Exception:
        return None, None


def smooth(y, w=5):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(y)]


def fmt():
    return mticker.FuncFormatter(
        lambda x, _: rf"${int(x/1000)}\mathrm{{k}}$" if x >= 1000 else rf"${int(x)}$"
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

ep_a, dist_a, ent_a = parse_log(str(_ROOT / "logs" / "train_gpu.log"))
ep_p, dist_p, ent_p = parse_log(str(_ROOT / "logs" / "train_ppo.log"))
ep_g, dist_g, ent_g = parse_log(str(_ROOT / "logs" / "train_gnn.log"))

val_ep_a, val_a = load_val(str(_ROOT / "results" / "val_results.json"))
val_ep_p, val_p = load_val(str(_ROOT / "results" / "val_results_ppo.json"))
val_ep_g, val_g = load_val(str(_ROOT / "results" / "val_results_gnn.json"))

rand_a = float(np.mean(dist_a[:5]))
rand_p = float(np.mean(dist_p[:5]))
rand_g = float(np.mean(dist_g[:5]))
random_baseline = float(np.mean([rand_a, rand_p, rand_g]))


# ---------------------------------------------------------------------------
# baseline/
# ---------------------------------------------------------------------------

print("\nbaseline/")

# Training curve: train + val + random
fig, ax = plt.subplots()
ax.plot(ep_a, dist_a, color=C0, lw=0.6, alpha=0.35)
ax.plot(ep_a, smooth(dist_a), color=C0, lw=1.2, ls="-", label="Train")
if val_ep_a is not None:
    ax.plot(val_ep_a, val_a, color=C1, lw=1.2, ls="--",
            marker="o", markersize=2.5, label="Validation")
ax.axhline(rand_a, color=C3, lw=0.8, ls=":", label="Random policy")
ax.set_title("A2C+HER baseline")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, BASE / "fig_training.png")

# Entropy
fig, ax = plt.subplots()
ax.plot(ep_a, ent_a, color=C0, lw=0.6, alpha=0.35)
ax.plot(ep_a, smooth(ent_a), color=C0, lw=1.2, ls="-", label=r"$H[\pi]$")
ax.axhline(np.log(35), color=C1, lw=1.0, ls="--", label="Uniform")
ax.set_title("A2C+HER policy entropy")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Policy entropy $H[\pi]$ (nats)")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, BASE / "fig_entropy.png")

# Combined 2-panel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax1.plot(ep_a, dist_a, color=C0, lw=0.6, alpha=0.35)
ax1.plot(ep_a, smooth(dist_a), color=C0, lw=1.2, ls="-", label="Train")
if val_ep_a is not None:
    ax1.plot(val_ep_a, val_a, color=C1, lw=1.2, ls="--",
             marker="o", markersize=2.5, label="Validation")
ax1.axhline(rand_a, color=C3, lw=0.8, ls=":", label="Random")
ax1.set_xlabel("Episode"); ax1.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax1.set_ylim(bottom=0); ax1.legend(fontsize=6)
ax1.set_title("(a) Training curve", loc="left")
ax1.xaxis.set_major_formatter(fmt())

ax2.plot(ep_a, ent_a, color=C0, lw=0.6, alpha=0.35)
ax2.plot(ep_a, smooth(ent_a), color=C0, lw=1.2, ls="-", label=r"$H[\pi]$")
ax2.axhline(np.log(35), color=C1, lw=1.0, ls="--", label="Uniform")
ax2.set_xlabel("Episode"); ax2.set_ylabel(r"Policy entropy $H[\pi]$ (nats)")
ax2.set_ylim(bottom=0); ax2.legend(fontsize=6)
ax2.set_title("(b) Policy entropy", loc="left")
ax2.xaxis.set_major_formatter(fmt())
save(fig, BASE / "fig_combined.png")


# ---------------------------------------------------------------------------
# exp1/  PPO
# ---------------------------------------------------------------------------

print("\nexp1/ (PPO)")

# Training curve: PPO vs A2C+HER vs random
fig, ax = plt.subplots()
ax.plot(ep_p, dist_p, color=C1, lw=0.6, alpha=0.25)
ax.plot(ep_p, smooth(dist_p), color=C1, lw=1.2, ls="-", label="PPO")
ax.plot(ep_a, smooth(dist_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax.axhline(random_baseline, color=C3, lw=0.8, ls=":", label="Random policy")
ax.set_title("PPO vs A2C+HER")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, E1 / "fig_ppo_training.png")

# Train + val + entropy 3-panel
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.6, 2.2))

ax1.plot(ep_p, dist_p, color=C1, lw=0.5, alpha=0.25)
ax1.plot(ep_p, smooth(dist_p), color=C1, lw=1.2, ls="-", label="PPO")
ax1.plot(ep_a, smooth(dist_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax1.axhline(random_baseline, color=C3, lw=0.8, ls=":", label="Random")
ax1.set_xlabel("Episode"); ax1.set_ylabel(r"$\ell_2$ distance")
ax1.set_ylim(bottom=0); ax1.legend(fontsize=5)
ax1.set_title("(a) Train", loc="left")
ax1.xaxis.set_major_formatter(fmt())

if val_ep_p is not None:
    ax2.plot(val_ep_p, val_p, color=C1, lw=1.2, ls="-",
             marker="s", markersize=2.5, label="PPO")
if val_ep_a is not None:
    ax2.plot(val_ep_a, val_a, color=C0, lw=1.0, ls="--",
             marker="o", markersize=2.5, label="A2C+HER")
ax2.set_xlabel("Episode"); ax2.set_ylabel(r"$\ell_2$ distance (val)")
ax2.set_ylim(bottom=0); ax2.legend(fontsize=5)
ax2.set_title("(b) Validation", loc="left")
ax2.xaxis.set_major_formatter(fmt())

ax3.plot(ep_p, ent_p, color=C1, lw=0.5, alpha=0.25)
ax3.plot(ep_p, smooth(ent_p), color=C1, lw=1.2, ls="-", label="PPO")
ax3.plot(ep_a, smooth(ent_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax3.set_xlabel("Episode"); ax3.set_ylabel(r"$H[\pi]$ (nats)")
ax3.set_ylim(bottom=0); ax3.legend(fontsize=5)
ax3.set_title("(c) Entropy", loc="left")
ax3.xaxis.set_major_formatter(fmt())

save(fig, E1 / "fig_ppo_combined.png")


# ---------------------------------------------------------------------------
# exp2/  GNN
# ---------------------------------------------------------------------------

print("\nexp2/ (GNN)")

# Training curve: GNN vs A2C+HER vs random
fig, ax = plt.subplots()
ax.plot(ep_g, dist_g, color=C2, lw=0.6, alpha=0.25)
ax.plot(ep_g, smooth(dist_g), color=C2, lw=1.2, ls="-", label="A2C+HER+GNN")
ax.plot(ep_a, smooth(dist_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax.axhline(random_baseline, color=C3, lw=0.8, ls=":", label="Random policy")
ax.set_title("GNN encoder vs Morgan fingerprint")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, E2 / "fig_gnn_training.png")

# Train + val + entropy 3-panel
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.6, 2.2))

ax1.plot(ep_g, dist_g, color=C2, lw=0.5, alpha=0.25)
ax1.plot(ep_g, smooth(dist_g), color=C2, lw=1.2, ls="-", label="GNN")
ax1.plot(ep_a, smooth(dist_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax1.axhline(random_baseline, color=C3, lw=0.8, ls=":", label="Random")
ax1.set_xlabel("Episode"); ax1.set_ylabel(r"$\ell_2$ distance")
ax1.set_ylim(bottom=0); ax1.legend(fontsize=5)
ax1.set_title("(a) Train", loc="left")
ax1.xaxis.set_major_formatter(fmt())

if val_ep_g is not None:
    ax2.plot(val_ep_g, val_g, color=C2, lw=1.2, ls="-",
             marker="^", markersize=2.5, label="GNN")
if val_ep_a is not None:
    ax2.plot(val_ep_a, val_a, color=C0, lw=1.0, ls="--",
             marker="o", markersize=2.5, label="A2C+HER")
ax2.set_xlabel("Episode"); ax2.set_ylabel(r"$\ell_2$ distance (val)")
ax2.set_ylim(bottom=0); ax2.legend(fontsize=5)
ax2.set_title("(b) Validation", loc="left")
ax2.xaxis.set_major_formatter(fmt())

ax3.plot(ep_g, ent_g, color=C2, lw=0.5, alpha=0.25)
ax3.plot(ep_g, smooth(ent_g), color=C2, lw=1.2, ls="-", label="GNN")
ax3.plot(ep_a, smooth(ent_a), color=C0, lw=1.0, ls="--", label="A2C+HER")
ax3.set_xlabel("Episode"); ax3.set_ylabel(r"$H[\pi]$ (nats)")
ax3.set_ylim(bottom=0); ax3.legend(fontsize=5)
ax3.set_title("(c) Entropy", loc="left")
ax3.xaxis.set_major_formatter(fmt())

save(fig, E2 / "fig_gnn_combined.png")


# ---------------------------------------------------------------------------
# combined/
# ---------------------------------------------------------------------------

print("\ncombined/")

# All training curves
fig, ax = plt.subplots()
ax.plot(ep_a, dist_a, color=C0, lw=0.5, alpha=0.2)
ax.plot(ep_a, smooth(dist_a), color=C0, lw=1.2, ls="-",  label="A2C+HER")
ax.plot(ep_p, dist_p, color=C1, lw=0.5, alpha=0.2)
ax.plot(ep_p, smooth(dist_p), color=C1, lw=1.2, ls="--", label="PPO")
ax.plot(ep_g, dist_g, color=C2, lw=0.5, alpha=0.2)
ax.plot(ep_g, smooth(dist_g), color=C2, lw=1.2, ls=":",  label="A2C+HER+GNN")
ax.axhline(random_baseline, color=C3, lw=0.8, ls="-.", label="Random policy")
ax.set_title("Method comparison: training")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, COMB / "fig_comparison_train.png")

# All validation curves
fig, ax = plt.subplots()
if val_ep_a is not None:
    ax.plot(val_ep_a, val_a, color=C0, lw=1.2, ls="-",
            marker="o", markersize=2.5, label="A2C+HER")
if val_ep_p is not None:
    ax.plot(val_ep_p, val_p, color=C1, lw=1.2, ls="--",
            marker="s", markersize=2.5, label="PPO")
if val_ep_g is not None:
    ax.plot(val_ep_g, val_g, color=C2, lw=1.2, ls=":",
            marker="^", markersize=2.5, label="A2C+HER+GNN")
ax.set_title("Method comparison: validation")
ax.set_xlabel("Episode"); ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax.set_ylim(bottom=0); ax.legend(fontsize=6)
ax.xaxis.set_major_formatter(fmt())
save(fig, COMB / "fig_comparison_val.png")

# 2-panel combined: train + val side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

ax1.plot(ep_a, smooth(dist_a), color=C0, lw=1.2, ls="-",  label="A2C+HER")
ax1.plot(ep_p, smooth(dist_p), color=C1, lw=1.2, ls="--", label="PPO")
ax1.plot(ep_g, smooth(dist_g), color=C2, lw=1.2, ls=":",  label="A2C+HER+GNN")
ax1.axhline(random_baseline, color=C3, lw=0.8, ls="-.", label="Random")
ax1.set_xlabel("Episode"); ax1.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax1.set_ylim(bottom=0); ax1.legend(fontsize=5)
ax1.set_title("(a) Train", loc="left")
ax1.xaxis.set_major_formatter(fmt())

if val_ep_a is not None:
    ax2.plot(val_ep_a, val_a, color=C0, lw=1.2, ls="-",
             marker="o", markersize=2.5, label="A2C+HER")
if val_ep_p is not None:
    ax2.plot(val_ep_p, val_p, color=C1, lw=1.2, ls="--",
             marker="s", markersize=2.5, label="PPO")
if val_ep_g is not None:
    ax2.plot(val_ep_g, val_g, color=C2, lw=1.2, ls=":",
             marker="^", markersize=2.5, label="A2C+HER+GNN")
ax2.set_xlabel("Episode"); ax2.set_ylabel(r"Mean $\ell_2$ distance to goal")
ax2.set_ylim(bottom=0); ax2.legend(fontsize=5)
ax2.set_title("(b) Validation", loc="left")
ax2.xaxis.set_major_formatter(fmt())

save(fig, COMB / "fig_comparison_combined.png")

print("\nDone.")
