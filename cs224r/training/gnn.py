"""
train_gnn.py
A2C + HER training with a GNN state encoder (replaces Morgan fingerprint).

The MolGraph is stored in each transition's info dict so HER relabelling
copies it for free.  During updates the GNN processes the raw graph while
the goal comes from the last GOAL_DIM elements of the (possibly relabelled)
state vector.

Usage
-----
  python train_gnn.py
  python train_gnn.py --n_episodes 10000
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import argparse
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from env import load_fragment_library, load_target_distribution, GOAL_DIM
from env import MolEnv, Action, TERMINATE, StepResult
from models import GNNActor, GNNCritic
from env import ReplayBuffer, Episode, Transition


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict = dict(
    fragments_parquet="/mnt/data/m3_20m/outputs/fragments.parquet",
    parents_parquet="/mnt/data/m3_20m/outputs/parents.parquet",
    n_frags=200,
    min_frag_count=5_000,
    n_targets=300,
    max_steps=6,
    n_episodes=10_000,
    batch_size=64,
    hidden_dim=256,
    gnn_dim=128,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    her_k=4,
    entropy_coef=0.005,
    critic_coef=0.5,
    grad_clip=1.0,
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir="checkpoints_gnn",
    exp_name="gnn_run",
    seed=1,
    use_wandb=False,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
)


# ---------------------------------------------------------------------------
# Rollout (stores mol_graph in info)
# ---------------------------------------------------------------------------

def rollout_gnn(env: MolEnv, actor: GNNActor, device: torch.device) -> Episode:
    state, goal, valid_actions = env.reset()
    ep = Episode()

    for _ in range(env.max_steps):
        mol_graph = deepcopy(env._mol)   # snapshot graph before step
        af_np     = env.get_action_features(valid_actions)
        goal_t    = torch.tensor(goal, dtype=torch.float32, device=device)
        af_t      = torch.tensor(af_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist       = actor.action_dist(mol_graph, goal_t, af_t, device)
            action_idx = int(dist.sample().item())

        result: StepResult = env.step(valid_actions[action_idx])
        next_valid  = result.info.get("valid_actions", [TERMINATE])
        next_af_np  = env.get_action_features(next_valid)

        ep.add(Transition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            reward=result.reward,
            next_state=result.state,
            next_action_feats=next_af_np,
            done=result.done,
            goal=goal,
            info={**result.info, "mol_graph": mol_graph},
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break
        state         = result.state
        valid_actions = next_valid

    return ep


# ---------------------------------------------------------------------------
# Parameter update
# ---------------------------------------------------------------------------

def update_gnn(
    actor:     GNNActor,
    critic:    GNNCritic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch:     list,
    cfg:       Dict,
    device:    torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    # First pass: collect values and returns for advantage normalisation
    Vs: List[torch.Tensor] = []
    Gs: List[torch.Tensor] = []
    for (t, G) in batch:
        mg     = t.info.get("mol_graph")
        goal_t = torch.tensor(t.state[-GOAL_DIM:], dtype=torch.float32, device=device)
        G_t    = torch.tensor(G, dtype=torch.float32, device=device)
        Vs.append(critic(mg, goal_t, device))
        Gs.append(G_t)

    Vs_t     = torch.stack(Vs)
    Gs_t     = torch.stack(Gs)
    adv_raw  = Gs_t - Vs_t.detach()
    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    entropies:    List[float]         = []

    for i, (t, G) in enumerate(batch):
        mg     = t.info.get("mol_graph")
        goal_t = torch.tensor(t.state[-GOAL_DIM:], dtype=torch.float32, device=device)
        af_t   = torch.tensor(t.action_feats, dtype=torch.float32, device=device)

        dist     = actor.action_dist(mg, goal_t, af_t, device)
        log_prob = dist.log_prob(torch.tensor(t.action_idx, device=device))
        entropy  = dist.entropy()

        actor_losses.append(-(log_prob * adv_norm[i]) - cfg["entropy_coef"] * entropy)
        critic_losses.append(cfg["critic_coef"] * (Gs_t[i] - Vs_t[i]).pow(2))
        entropies.append(float(entropy.item()))

    total_actor  = torch.stack(actor_losses).mean()
    total_critic = torch.stack(critic_losses).mean()

    opt_actor.zero_grad()
    total_actor.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg["grad_clip"])
    opt_actor.step()

    opt_critic.zero_grad()
    total_critic.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
    opt_critic.step()

    return {
        "actor_loss":  float(total_actor.item()),
        "critic_loss": float(total_critic.item()),
        "entropy":     float(np.mean(entropies)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Optional[Dict] = None) -> Tuple[GNNActor, GNNCritic]:
    if cfg is None:
        cfg = DEFAULT_CFG

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.init(
            project="cs224r-fragment-assembly",
            name=cfg["exp_name"],
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            tags=["gnn"],
        )

    device = torch.device(cfg["device"])

    print("Loading fragment library ...")
    frags = load_fragment_library(
        cfg["fragments_parquet"], n=cfg["n_frags"], min_count=cfg["min_frag_count"]
    )
    print(f"  {len(frags)} fragments")

    print("Loading target distribution ...")
    targets = load_target_distribution(cfg["parents_parquet"], n=cfg["n_targets"])
    print(f"  {len(targets)} targets")

    env    = MolEnv(frags, targets, max_steps=cfg["max_steps"])
    actor  = GNNActor(gnn_dim=cfg["gnn_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    critic = GNNCritic(gnn_dim=cfg["gnn_dim"], hidden_dim=cfg["hidden_dim"]).to(device)

    n_actor  = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"  GNNActor params:  {n_actor:,}")
    print(f"  GNNCritic params: {n_critic:,}")

    opt_actor  = optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    buffer = ReplayBuffer(max_episodes=2_000, gamma=cfg["gamma"], her_k=cfg["her_k"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ep_rewards: List[float] = []
    ep_dists:   List[float] = []
    ep_valid:   List[float] = []

    print(f"\nTraining for {cfg['n_episodes']} episodes ...\n")

    for ep_idx in range(cfg["n_episodes"]):
        actor.train(); critic.train()
        ep = rollout_gnn(env, actor, device)
        buffer.push(ep)

        total_r = sum(t.reward for t in ep.transitions)
        ep_rewards.append(total_r)
        achieved = ep.achieved_goal
        ep_valid.append(1.0 if achieved is not None else 0.0)
        if achieved is not None:
            ep_dists.append(-ep.transitions[-1].reward)

        if len(buffer) >= 5:
            batch   = buffer.sample_transitions(cfg["batch_size"])
            metrics = update_gnn(actor, critic, opt_actor, opt_critic, batch, cfg, device)
        else:
            metrics = {}

        if (ep_idx + 1) % cfg["log_every"] == 0:
            w      = cfg["log_every"]
            r_mean = float(np.mean(ep_rewards[-w:]))
            v_pct  = float(np.mean(ep_valid[-w:])) * 100.0
            d_mean = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
            loss_s = ""
            if metrics:
                loss_s = (f"  actor={metrics['actor_loss']:.4f}"
                          f"  critic={metrics['critic_loss']:.4f}"
                          f"  H={metrics['entropy']:.3f}")
            print(f"[{ep_idx+1:5d}] reward={r_mean:+.3f}  "
                  f"dist={d_mean:.3f}  valid={v_pct:.1f}%{loss_s}")
            if cfg.get("use_wandb") and _wandb is not None:
                log = {
                    "train/mean_reward": r_mean,
                    "train/mean_dist":   d_mean,
                    "train/valid_pct":   v_pct,
                }
                if metrics:
                    log["train/actor_loss"]  = metrics["actor_loss"]
                    log["train/critic_loss"] = metrics["critic_loss"]
                    log["train/entropy"]     = metrics["entropy"]
                _wandb.log(log, step=ep_idx + 1)

        if (ep_idx + 1) % cfg["checkpoint_every"] == 0:
            path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx+1}.pt")
            torch.save({
                "episode": ep_idx + 1,
                "actor":   actor.state_dict(),
                "critic":  critic.state_dict(),
                "config":  cfg,
            }, path)
            print(f"  -> checkpoint: {path}")

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.finish()

    return actor, critic


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Dict:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes",       type=int,   default=DEFAULT_CFG["n_episodes"])
    p.add_argument("--n_frags",          type=int,   default=DEFAULT_CFG["n_frags"])
    p.add_argument("--n_targets",        type=int,   default=DEFAULT_CFG["n_targets"])
    p.add_argument("--hidden_dim",       type=int,   default=DEFAULT_CFG["hidden_dim"])
    p.add_argument("--gnn_dim",          type=int,   default=DEFAULT_CFG["gnn_dim"])
    p.add_argument("--her_k",            type=int,   default=DEFAULT_CFG["her_k"])
    p.add_argument("--entropy_coef",     type=float, default=DEFAULT_CFG["entropy_coef"])
    p.add_argument("--device",           type=str,   default=DEFAULT_CFG["device"])
    p.add_argument("--checkpoint_dir",   type=str,   default=DEFAULT_CFG["checkpoint_dir"])
    p.add_argument("--log_every",        type=int,   default=DEFAULT_CFG["log_every"])
    p.add_argument("--checkpoint_every", type=int,   default=DEFAULT_CFG["checkpoint_every"])
    p.add_argument("--exp_name",   type=str,            default=DEFAULT_CFG["exp_name"])
    p.add_argument("--seed",       type=int,            default=DEFAULT_CFG["seed"])
    p.add_argument("--use_wandb",  action="store_true", default=False)
    args = p.parse_args()
    cfg  = dict(DEFAULT_CFG)
    cfg.update(vars(args))
    return cfg


if __name__ == "__main__":
    train(_parse_args())
