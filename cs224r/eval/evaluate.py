"""
evaluate.py
Evaluation metrics for the trained agent.

Metrics
-------
  mean_distance   Mean L2 distance (in normalised property space) between
                  the generated molecule's properties and the target goal,
                  computed over valid (sanitisable) molecules only.

  success_rate    Fraction of episodes where every property dimension is
                  within ε of the target (default ε = 0.1 in norm. space,
                  ≈ 1.5 LogP units / 0.1 QED / 20 Å² TPSA).

  validity_rate   Fraction of generated molecules that pass RDKit sanitation.

Baselines (available as standalone functions)
---------------------------------------------
  random_baseline   — random valid action at each step.
  terminal_baseline — terminate immediately (single seed fragment as output).

Usage
-----
  python evaluate.py [--checkpoint path/to/ckpt.pt] [--n_episodes 200]
"""
import _path_bootstrap  # noqa: F401

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

from data import load_fragment_library, load_target_distribution
from env import MolEnv, Action, TERMINATE
from model import Actor
from train import DEFAULT_CFG


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    actor: Optional[Actor],
    env: MolEnv,
    n_episodes: int = 200,
    epsilon: float = 0.1,
    device: torch.device = torch.device("cpu"),
    greedy: bool = True,
) -> Dict[str, float]:
    """
    Run n_episodes episodes and return evaluation metrics.
    If actor is None, use a uniformly random policy (random baseline).
    """
    distances: List[float] = []
    successes: List[float] = []
    valids:    List[float] = []

    if actor is not None:
        actor.eval()

    for _ in range(n_episodes):
        state, goal, valid_actions = env.reset()
        done = False

        while not done:
            if actor is None:
                # Random policy.
                action_idx = np.random.randint(len(valid_actions))
            else:
                af_np = env.get_action_features(valid_actions)
                state_t = torch.tensor(state, dtype=torch.float32, device=device)
                af_t    = torch.tensor(af_np,  dtype=torch.float32, device=device)
                with torch.no_grad():
                    dist = actor.action_dist(state_t, af_t)
                    if greedy:
                        action_idx = int(dist.probs.argmax().item())
                    else:
                        action_idx = int(dist.sample().item())

            result = env.step(valid_actions[action_idx])
            state  = result.state
            done   = result.done
            if not done:
                valid_actions = result.info.get("valid_actions", [TERMINATE])

        props = result.info.get("achieved_goal")
        is_valid = result.info.get("valid", False)
        valids.append(float(is_valid))

        if props is not None:
            d = float(np.linalg.norm(props - goal))
            distances.append(d)
            successes.append(float(np.all(np.abs(props - goal) < epsilon)))
        else:
            distances.append(float("nan"))
            successes.append(0.0)

    valid_dists = [d for d in distances if not np.isnan(d)]
    return {
        "mean_distance": float(np.mean(valid_dists)) if valid_dists else float("nan"),
        "success_rate":  float(np.mean(successes)),
        "validity_rate": float(np.mean(valids)),
        "n_episodes":    n_episodes,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers for baselines
# ---------------------------------------------------------------------------

def random_baseline(env: MolEnv, n_episodes: int = 200, **kw) -> Dict[str, float]:
    """Random valid action at every step."""
    return evaluate(None, env, n_episodes=n_episodes, **kw)


def terminate_baseline(env: MolEnv, n_episodes: int = 200, **kw) -> Dict[str, float]:
    """Immediately terminate — output is the seed fragment alone."""
    from env import MolEnv as _Env
    from features import compute_norm_properties
    from chem.build.molgraph_to_mol import molgraph_to_mol

    distances, successes, valids = [], [], []
    epsilon = kw.get("epsilon", 0.1)

    for _ in range(n_episodes):
        state, goal, _ = env.reset()
        # Force terminate immediately by calling _finalise internals.
        # Easier: just step with terminate.
        result = env.step(TERMINATE)
        props = result.info.get("achieved_goal")
        is_valid = result.info.get("valid", False)
        valids.append(float(is_valid))
        if props is not None:
            d = float(np.linalg.norm(props - goal))
            distances.append(d)
            successes.append(float(np.all(np.abs(props - goal) < epsilon)))
        else:
            distances.append(float("nan"))
            successes.append(0.0)

    valid_dists = [d for d in distances if not np.isnan(d)]
    return {
        "mean_distance": float(np.mean(valid_dists)) if valid_dists else float("nan"),
        "success_rate":  float(np.mean(successes)),
        "validity_rate": float(np.mean(valids)),
        "n_episodes":    n_episodes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_metrics(label: str, m: Dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  mean distance  : {m['mean_distance']:.4f}")
    print(f"  success rate   : {m['success_rate']*100:.1f}%")
    print(f"  validity rate  : {m['validity_rate']*100:.1f}%")
    print(f"  n episodes     : {m['n_episodes']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str,   default=None)
    p.add_argument("--n_episodes",    type=int,   default=200)
    p.add_argument("--epsilon",       type=float, default=0.1)
    p.add_argument("--n_frags",       type=int,   default=DEFAULT_CFG["n_frags"])
    p.add_argument("--min_frag_count",type=int,   default=DEFAULT_CFG["min_frag_count"])
    p.add_argument("--n_targets",     type=int,   default=DEFAULT_CFG["n_targets"])
    p.add_argument("--hidden_dim",    type=int,   default=DEFAULT_CFG["hidden_dim"])
    p.add_argument("--device",        type=str,   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    frags   = load_fragment_library(DEFAULT_CFG["fragments_parquet"],
                                    n=args.n_frags,
                                    min_count=args.min_frag_count)
    targets = load_target_distribution(DEFAULT_CFG["parents_parquet"],
                                       n=args.n_targets)
    env = MolEnv(frags, targets, max_steps=DEFAULT_CFG["max_steps"])

    # --- baselines ---
    _print_metrics("Baseline: random policy",
                   random_baseline(env, n_episodes=args.n_episodes, epsilon=args.epsilon))
    _print_metrics("Baseline: terminate immediately",
                   terminate_baseline(env, n_episodes=args.n_episodes, epsilon=args.epsilon))

    # --- trained agent ---
    if args.checkpoint:
        actor = Actor(hidden_dim=args.hidden_dim).to(device)
        ckpt  = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        _print_metrics(f"Trained agent ({args.checkpoint})",
                       evaluate(actor, env, n_episodes=args.n_episodes,
                                epsilon=args.epsilon, device=device))
    print()


if __name__ == "__main__":
    main()
