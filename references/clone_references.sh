#!/usr/bin/env bash
# clone_references.sh
# Clones all reference repositories into:
#   /home/ayamin/projects/cs224r/cs224r/references

set -euo pipefail

REFS_DIR="/home/ayamin/projects/cs224r/cs224r/references"
mkdir -p "$REFS_DIR"
cd "$REFS_DIR"

echo "==> Cloning into $REFS_DIR"
echo ""

# ── Closest full-pipeline match ────────────────────────────────────────────
# Pocket-conditional GFlowNet + synthesis fragments + UniDock + CrossDocked2020
git clone https://github.com/SeonghwanSeo/RxnFlow.git

# ── Pocket-conditional GFlowNet (original, GVP-GNN encoder) ───────────────
# TMLR 2024 — the amortised SBDD paper your proposal cites directly
git clone https://github.com/tsa87/tacogfn.git

# ── 3D pose + synthesis pathway co-design (builds on RxnFlow + TacoGFN) ───
# ICML 2025 — most recent evolution of the same stack
git clone https://github.com/tsa87/cgflow.git

# ── EGNN implementation (Satorras et al. ICML 2021) ───────────────────────
# Clean PyTorch EGNN — drop-in for your pocket encoder
git clone https://github.com/lucidrains/egnn-pytorch.git

# ── EGNN + virtual nodes for binding-site identification ──────────────────
# State-of-art on COACH420 / HOLO4K / PDBbind2020
git clone https://github.com/ml-jku/vnegnn.git

# ── Core GFlowNet library (SubTB, fragment env, MOO, replay, conditioning) ─
# The training backbone used by RxnFlow and TacoGFN internally
git clone https://github.com/recursionpharma/gflownet.git

# ── Synthesisable fragment GFlowNet via reaction space (NeurIPS 2024) ──────
# BRICS-adjacent: reaction-template action space, GPU docking oracle
git clone https://github.com/koziarskilab/RGFN.git

# ── Scalable template-based generation (NeurIPS 2025, same lab as RGFN) ───
git clone https://github.com/koziarskilab/SCENT.git

# ── Modular GFlowNet research library (SubTB, DB, TB, FM, ZVar) ───────────
# GFNOrg reference implementation — useful for SubTB details
git clone https://github.com/GFNOrg/torchgfn.git

# ── Pocket proxy / pharmacophore encoder used inside RxnFlow & TacoGFN ────
git clone https://github.com/SeonghwanSeo/PharmacoNet.git

echo ""
echo "==> Done. Cloned repos:"
ls -1 "$REFS_DIR"