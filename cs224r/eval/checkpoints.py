"""
eval_checkpoints.py
Evaluate every checkpoint on a fixed held-out validation set and save results.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from env import load_fragment_library, load_target_distribution
from env import MolEnv, TERMINATE
from models import Actor
from training.a2c import DEFAULT_CFG

_ROOT    = _Path(__file__).resolve().parent.parent
CKPT_DIR = _ROOT / "checkpoints"
OUT_FILE  = _ROOT / "results" / "val_results.json"
VAL_SEED  = 99991          # fixed seed for reproducible validation episodes
VAL_N     = 200


def eval_actor(actor, env, n=VAL_N, device=torch.device("cpu")):
    np.random.seed(VAL_SEED)
    actor.eval()
    dists = []
    for _ in range(n):
        state, goal, valid_actions = env.reset()
        done = False
        while not done:
            af   = env.get_action_features(valid_actions)
            st   = torch.tensor(state, dtype=torch.float32, device=device)
            af_t = torch.tensor(af,    dtype=torch.float32, device=device)
            with torch.no_grad():
                idx = int(actor.action_dist(st, af_t).probs.argmax().item())
            result = env.step(valid_actions[idx])
            state  = result.state
            done   = result.done
            if not done:
                valid_actions = result.info.get("valid_actions", [TERMINATE])
        props = result.info.get("achieved_goal")
        if props is not None:
            dists.append(float(np.linalg.norm(props - goal)))
    return float(np.mean(dists)) if dists else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",   type=int, default=DEFAULT_CFG["n_frags"])
    p.add_argument("--n_targets", type=int, default=DEFAULT_CFG["n_targets"])
    p.add_argument("--val_n",     type=int, default=VAL_N)
    args = p.parse_args()

    device = torch.device("cpu")

    print("Loading data ...")
    frags   = load_fragment_library(DEFAULT_CFG["fragments_parquet"],
                                    n=args.n_frags, min_count=DEFAULT_CFG["min_frag_count"])
    targets = load_target_distribution(DEFAULT_CFG["parents_parquet"], n=args.n_targets)
    env     = MolEnv(frags, targets, max_steps=DEFAULT_CFG["max_steps"])

    ckpts = sorted(CKPT_DIR.glob("ckpt_ep*.pt"),
                   key=lambda p: int(p.stem.split("ep")[1]))

    results = {}
    for ckpt_path in ckpts:
        ep = int(ckpt_path.stem.split("ep")[1])
        ckpt = torch.load(ckpt_path, map_location=device)
        hidden = ckpt.get("config", {}).get("hidden_dim", 256)
        actor  = Actor(hidden_dim=hidden).to(device)
        actor.load_state_dict(ckpt["actor"])

        mean_dist = eval_actor(actor, env, n=args.val_n, device=device)
        results[ep] = mean_dist
        print(f"  ep {ep:6d}  val_dist = {mean_dist:.4f}")

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
