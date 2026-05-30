"""
eval_experiments.py
Evaluate PPO and GNN checkpoints on the fixed validation set.
Saves val_results_ppo.json and val_results_gnn.json.
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

from env import load_fragment_library, load_target_distribution, parse_property_names
from env import MolEnv, TERMINATE
from models import Actor, GNNActor
from training.a2c import DEFAULT_CFG

VAL_SEED = 99991
VAL_N    = 200


def eval_actor_fp(actor, env, n=VAL_N, device=torch.device("cpu")):
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


def eval_actor_gnn(actor, env, n=VAL_N, device=torch.device("cpu")):
    from copy import deepcopy
    np.random.seed(VAL_SEED)
    actor.eval()
    dists = []
    for _ in range(n):
        state, goal, valid_actions = env.reset()
        done = False
        while not done:
            mg   = deepcopy(env._mol)
            af   = env.get_action_features(valid_actions)
            goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
            af_t   = torch.tensor(af,   dtype=torch.float32, device=device)
            with torch.no_grad():
                idx = int(actor.action_dist(mg, goal_t, af_t, device).probs.argmax().item())
            result = env.step(valid_actions[idx])
            state  = result.state
            done   = result.done
            if not done:
                valid_actions = result.info.get("valid_actions", [TERMINATE])
        props = result.info.get("achieved_goal")
        if props is not None:
            dists.append(float(np.linalg.norm(props - goal)))
    return float(np.mean(dists)) if dists else float("nan")


def eval_dir(ckpt_dir, actor_type, eval_fn, frags, device, out_file):
    ckpts = sorted(Path(ckpt_dir).glob("ckpt_ep*.pt"),
                   key=lambda p: int(p.stem.split("ep")[1]))
    results = {}
    for ckpt_path in ckpts:
        ep   = int(ckpt_path.stem.split("ep")[1])
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt.get("config", {})
        property_names = parse_property_names(cfg.get("goal_properties", DEFAULT_CFG["goal_properties"]))
        targets = load_target_distribution(
            cfg.get("parents_parquet", DEFAULT_CFG["parents_parquet"]),
            n=cfg.get("n_targets", DEFAULT_CFG["n_targets"]),
            property_names=property_names,
        )
        env = MolEnv(
            frags,
            targets,
            max_steps=cfg.get("max_steps", DEFAULT_CFG["max_steps"]),
            property_names=property_names,
        )

        if actor_type == "fp":
            actor = Actor(state_dim=env.state_dim, hidden_dim=cfg.get("hidden_dim", 256)).to(device)
            actor.load_state_dict(ckpt["actor"])
        else:
            actor = GNNActor(
                gnn_dim=cfg.get("gnn_dim", 128),
                goal_dim=env.goal_dim,
                hidden_dim=cfg.get("hidden_dim", 256),
            ).to(device)
            actor.load_state_dict(ckpt["actor"])

        mean_dist = eval_fn(actor, env, device=device)
        results[ep] = mean_dist
        print(f"  ep {ep:6d}  val_dist = {mean_dist:.4f}")

    Path(out_file).write_text(json.dumps(results, indent=2))
    print(f"Saved {out_file}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["ppo", "gnn", "both"], default="both")
    p.add_argument("--val_n", type=int, default=VAL_N)
    args = p.parse_args()

    device = torch.device("cpu")
    frags   = load_fragment_library(DEFAULT_CFG["fragments_parquet"],
                                    n=DEFAULT_CFG["n_frags"],
                                    min_count=DEFAULT_CFG["min_frag_count"])
    _root = _Path(__file__).resolve().parent.parent
    _res  = _root / "results"
    _res.mkdir(exist_ok=True)

    if args.which in ("ppo", "both"):
        print("Evaluating PPO checkpoints ...")
        eval_dir(str(_root / "checkpoints_ppo"), "fp", eval_actor_fp, frags, device, str(_res / "val_results_ppo.json"))

    if args.which in ("gnn", "both"):
        print("Evaluating GNN checkpoints ...")
        eval_dir(str(_root / "checkpoints_gnn"), "gnn", eval_actor_gnn, frags, device, str(_res / "val_results_gnn.json"))


if __name__ == "__main__":
    main()
