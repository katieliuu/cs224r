"""
plot_training.py
Parse training logs and produce publication-quality figures using
local scienceplots (science + ieee style).

Outputs
-------
  fig_training_curve.png   baseline A2C+HER only (train + val + random)
  fig_entropy.png          baseline entropy
  fig_combined.png         baseline 2-panel
  fig_comparison.png       all three methods: A2C+HER, PPO, GNN (train curves)
  fig_comparison_val.png   all three methods: validation curves only
"""
import _path_bootstrap  # noqa: F401

import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_STYLE_DIR = Path("/home/ayamin/projects/cs273b/asmani-branch/plot/scienceplots/styles")
_SCIENCE = str(_STYLE_DIR / "science.mplstyle")
_IEEE    = str(_STYLE_DIR / "journals" / "ieee.mplstyle")
plt.style.use([_SCIENCE, _IEEE])

C0 = "k"    # black  — A2C+HER
C1 = "r"    # red    — PPO
C2 = "b"    # blue   — GNN
C3 = "g"    # green  — random baseline


# ── parse log ──────────────────────────────────────────────────────────────

LOG_RE = re.compile(
    r"\[\s*(\d+)\]\s+reward=([+-]?\d+\.\d+)\s+dist=(\d+\.\d+)"
    r"\s+valid=(\d+\.\d+)%\s+actor=([+-]?\d+\.\d+)"
    r"\s+critic=(\d+\.\d+)\s+H=(\d+\.\d+)"
)

def parse_log(path: str):
    ep, reward, dist, valid, actor, critic, entropy = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                ep.append(int(m.group(1)))
                reward.append(float(m.group(2)))
                dist.append(float(m.group(3)))
                valid.append(float(m.group(4)))
                actor.append(float(m.group(5)))
                critic.append(float(m.group(6)))
                entropy.append(float(m.group(7)))
    return (np.array(ep), np.array(reward), np.array(dist),
            np.array(valid), np.array(actor), np.array(critic),
            np.array(entropy))


def smooth(y, w=5):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    padded = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


# ── validation loading ──────────────────────────────────────────────────────

def load_val(path="val_results.json", every=1000):
    """Load validation results; filter to multiples of `every`."""
    try:
        raw = json.loads(Path(path).read_text())
        pairs = [(int(k), v) for k, v in raw.items() if int(k) % every == 0]
        pairs.sort()
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    except Exception:
        return None, None


def _ep_formatter():
    return mticker.FuncFormatter(
        lambda x, _: rf"${int(x/1000)}\mathrm{{k}}$" if x >= 1000 else rf"${int(x)}$"
    )


# ── baseline figures (unchanged) ────────────────────────────────────────────

def fig_training_curve(ep, dist, out):
    baseline = float(np.mean(dist[:5]))
    val_ep, val_dist = load_val()

    fig, ax = plt.subplots()
    ax.plot(ep, dist,         color=C0, lw=0.6, alpha=0.35)
    ax.plot(ep, smooth(dist), color=C0, lw=1.2, ls="-",  label="Train (A2C+HER)")
    if val_ep is not None:
        ax.plot(val_ep, val_dist, color=C1, lw=1.2, ls="--", marker="o",
                markersize=2.5, label="Validation")
    ax.axhline(baseline, color=C2, lw=0.8, ls=":", label="Random policy")
    ax.set_title("Training curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_formatter(_ep_formatter())
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


def fig_entropy(ep, entropy, out):
    fig, ax = plt.subplots()
    ax.plot(ep, entropy,         color=C0, lw=0.6, alpha=0.35)
    ax.plot(ep, smooth(entropy), color=C0, lw=1.2, ls="-",  label=r"$H[\pi]$")
    ax.axhline(np.log(35),       color=C1, lw=1.0, ls="--", label="Uniform")
    ax.set_title("Policy entropy")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Policy entropy $H[\pi]$ (nats)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_formatter(_ep_formatter())
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


def fig_combined(ep, dist, entropy, out):
    baseline = float(np.mean(dist[:5]))
    val_ep, val_dist = load_val()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    ax1.plot(ep, dist,         color=C0, lw=0.6, alpha=0.35)
    ax1.plot(ep, smooth(dist), color=C0, lw=1.2, ls="-",  label="Train")
    if val_ep is not None:
        ax1.plot(val_ep, val_dist, color=C1, lw=1.2, ls="--", marker="o",
                 markersize=2.5, label="Validation")
    ax1.axhline(baseline, color=C2, lw=0.8, ls=":", label="Random")
    ax1.set_xlabel("Episode"); ax1.set_ylabel(r"Mean $\ell_2$ distance to goal")
    ax1.set_ylim(bottom=0); ax1.legend(fontsize=6)
    ax1.set_title("(a) Training curve", loc="left")
    ax1.xaxis.set_major_formatter(_ep_formatter())

    ax2.plot(ep, entropy,         color=C0, lw=0.6, alpha=0.35)
    ax2.plot(ep, smooth(entropy), color=C0, lw=1.2, ls="-",  label=r"$H[\pi]$")
    ax2.axhline(np.log(35),       color=C1, lw=1.0, ls="--", label="Uniform")
    ax2.set_xlabel("Episode"); ax2.set_ylabel(r"Policy entropy $H[\pi]$ (nats)")
    ax2.set_ylim(bottom=0); ax2.legend(fontsize=6)
    ax2.set_title("(b) Policy entropy", loc="left")
    ax2.xaxis.set_major_formatter(_ep_formatter())

    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ── experiment comparison figures ───────────────────────────────────────────

def fig_comparison(ep_a2c, dist_a2c, ep_ppo, dist_ppo, ep_gnn, dist_gnn, out):
    """Smoothed training curves for all three methods."""
    baseline = float(np.mean(dist_a2c[:5]))

    fig, ax = plt.subplots()

    ax.plot(ep_a2c, dist_a2c,         color=C0, lw=0.5, alpha=0.25)
    ax.plot(ep_a2c, smooth(dist_a2c), color=C0, lw=1.2, ls="-",  label="A2C+HER")

    ax.plot(ep_ppo, dist_ppo,         color=C1, lw=0.5, alpha=0.25)
    ax.plot(ep_ppo, smooth(dist_ppo), color=C1, lw=1.2, ls="--", label="PPO")

    ax.plot(ep_gnn, dist_gnn,         color=C2, lw=0.5, alpha=0.25)
    ax.plot(ep_gnn, smooth(dist_gnn), color=C2, lw=1.2, ls=":",  label="A2C+HER+GNN")

    ax.axhline(baseline, color=C3, lw=0.8, ls="-.", label="Random policy")

    ax.set_title("Method comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Mean $\ell_2$ distance to goal")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_formatter(_ep_formatter())

    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


def fig_comparison_val(out):
    """Validation curves for all three methods."""
    val_a2c_ep, val_a2c  = load_val("val_results.json",     every=1000)
    val_ppo_ep, val_ppo  = load_val("val_results_ppo.json", every=1000)
    val_gnn_ep, val_gnn  = load_val("val_results_gnn.json", every=1000)

    if val_a2c_ep is None and val_ppo_ep is None and val_gnn_ep is None:
        print("  No validation results found, skipping fig_comparison_val")
        return

    fig, ax = plt.subplots()

    if val_a2c_ep is not None:
        ax.plot(val_a2c_ep, val_a2c, color=C0, lw=1.2, ls="-",
                marker="o", markersize=2.5, label="A2C+HER")
    if val_ppo_ep is not None:
        ax.plot(val_ppo_ep, val_ppo, color=C1, lw=1.2, ls="--",
                marker="s", markersize=2.5, label="PPO")
    if val_gnn_ep is not None:
        ax.plot(val_gnn_ep, val_gnn, color=C2, lw=1.2, ls=":",
                marker="^", markersize=2.5, label="A2C+HER+GNN")

    ax.set_title("Validation comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Mean $\ell_2$ distance to goal (validation)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_formatter(_ep_formatter())

    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main(log_path: str = "train_gpu.log"):
    print(f"Parsing {log_path} ...")
    ep, _, dist, _, _, _, entropy = parse_log(log_path)
    print(f"  {len(ep)} log points  |  ep {ep[0]}-{ep[-1]}")
    print(f"  dist:    init={dist[0]:.3f}  best={dist.min():.3f}  final={dist[-1]:.3f}")
    print(f"  entropy: init={entropy[0]:.3f}  final={entropy[-1]:.3f}")
    print()

    fig_training_curve(ep, dist, "fig_training_curve.png")
    fig_entropy(ep, entropy, "fig_entropy.png")
    fig_combined(ep, dist, entropy, "fig_combined.png")

    # Experiment comparison (train curves)
    ppo_exists = Path("train_ppo.log").exists()
    gnn_exists = Path("train_gnn.log").exists()
    if ppo_exists and gnn_exists:
        print("Generating comparison figures ...")
        ep_ppo, _, dist_ppo, *_ = parse_log("train_ppo.log")
        ep_gnn, _, dist_gnn, *_ = parse_log("train_gnn.log")
        fig_comparison(ep, dist, ep_ppo, dist_ppo, ep_gnn, dist_gnn, "fig_comparison.png")

    # Validation comparison
    fig_comparison_val("fig_comparison_val.png")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="train_gpu.log")
    args = p.parse_args()
    main(args.log)
