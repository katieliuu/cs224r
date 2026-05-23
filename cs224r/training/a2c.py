"""
train.py
A2C-style (actor-critic with Monte-Carlo returns) training loop.

Algorithm per update step
--------------------------
  Advantage  A = G - V(s)
  Actor loss = -E[log π(a|s) · A] - α · H[π]
  Critic loss= λ · E[(G - V(s))²]

Hindsight Experience Replay is applied transparently inside ReplayBuffer.

Usage
-----
  python train.py                              # default config
  python train.py --n_episodes 500            # quick smoke-test
"""
import _path_bootstrap  # noqa: F401

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_fragment_library, load_target_distribution
from env import MolEnv, Action, TERMINATE, StepResult
from model import Actor, Critic
from replay import ReplayBuffer, Episode, Transition


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict = dict(
    fragments_parquet="/mnt/data/m3_20m/outputs/fragments.parquet",
    parents_parquet="/mnt/data/m3_20m/outputs/parents.parquet",
    # --- data subset (tiny for now) ---
    n_frags=200,
    min_frag_count=5_000,
    n_targets=300,
    # --- environment ---
    max_steps=6,
    # --- training ---
    n_episodes=10_000,
    batch_size=64,
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    her_k=4,
    entropy_coef=0.005,
    critic_coef=0.5,
    grad_clip=1.0,
    # --- logging ---
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir="checkpoints",
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(env: MolEnv, actor: Actor, device: torch.device) -> Episode:
    """Execute one episode under the current policy; return the Episode."""
    state, goal, valid_actions = env.reset()
    ep = Episode()

    for _ in range(env.max_steps):
        af_np = env.get_action_features(valid_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t    = torch.tensor(af_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist = actor.action_dist(state_t, af_t)
            action_idx = int(dist.sample().item())

        result: StepResult = env.step(valid_actions[action_idx])

        next_valid = result.info.get("valid_actions", [TERMINATE])
        next_af_np = env.get_action_features(next_valid)

        ep.add(Transition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            reward=result.reward,
            next_state=result.state,
            next_action_feats=next_af_np,
            done=result.done,
            goal=goal,
            info=result.info,
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break

        state = result.state
        valid_actions = next_valid

    return ep


# ---------------------------------------------------------------------------
# Parameter update
# ---------------------------------------------------------------------------

def update(
    actor: Actor,
    critic: Critic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch: list,
    cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    entropies: List[float] = []
    advantages: List[torch.Tensor] = []

    # First pass: collect raw advantages for normalisation
    Vs: List[torch.Tensor] = []
    Gs: List[torch.Tensor] = []
    for (t, G) in batch:
        state_t = torch.tensor(t.state, dtype=torch.float32, device=device)
        G_t     = torch.tensor(G,       dtype=torch.float32, device=device)
        Vs.append(critic(state_t))
        Gs.append(G_t)

    Vs_t    = torch.stack(Vs)
    Gs_t    = torch.stack(Gs)
    adv_raw = Gs_t - Vs_t.detach()
    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    for i, (t, G) in enumerate(batch):
        state_t = torch.tensor(t.state,        dtype=torch.float32, device=device)
        af_t    = torch.tensor(t.action_feats, dtype=torch.float32, device=device)

        dist     = actor.action_dist(state_t, af_t)
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
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Optional[Dict] = None) -> Tuple[Actor, Critic]:
    if cfg is None:
        cfg = DEFAULT_CFG

    device = torch.device(cfg["device"])

    print("Loading fragment library …")
    frags = load_fragment_library(
        cfg["fragments_parquet"], n=cfg["n_frags"], min_count=cfg["min_frag_count"]
    )
    print(f"  {len(frags)} fragments loaded")

    print("Loading target distribution …")
    targets = load_target_distribution(cfg["parents_parquet"], n=cfg["n_targets"])
    print(f"  {len(targets)} target molecules "
          f"| LogP∈[{targets[:,0].min():.2f},{targets[:,0].max():.2f}] "
          f"| QED∈[{targets[:,1].min():.2f},{targets[:,1].max():.2f}] "
          f"| TPSA∈[{targets[:,2].min():.2f},{targets[:,2].max():.2f}]")

    env    = MolEnv(frags, targets, max_steps=cfg["max_steps"])
    actor  = Actor(hidden_dim=cfg["hidden_dim"]).to(device)
    critic = Critic(hidden_dim=cfg["hidden_dim"]).to(device)

    opt_actor  = optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    buffer = ReplayBuffer(
        max_episodes=2_000, gamma=cfg["gamma"], her_k=cfg["her_k"]
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ep_rewards: List[float] = []
    ep_dists:   List[float] = []
    ep_valid:   List[float] = []

    print(f"\nTraining for {cfg['n_episodes']} episodes …\n")

    for ep_idx in range(cfg["n_episodes"]):
        actor.train(); critic.train()
        ep = rollout(env, actor, device)
        buffer.push(ep)

        total_r = sum(t.reward for t in ep.transitions)
        ep_rewards.append(total_r)

        achieved = ep.achieved_goal
        ep_valid.append(1.0 if achieved is not None else 0.0)
        if achieved is not None:
            # Distance is the negated terminal reward (last transition).
            ep_dists.append(-ep.transitions[-1].reward)

        if len(buffer) >= 5:
            batch = buffer.sample_transitions(cfg["batch_size"])
            metrics = update(actor, critic, opt_actor, opt_critic, batch, cfg, device)
        else:
            metrics = {}

        if (ep_idx + 1) % cfg["log_every"] == 0:
            w = cfg["log_every"]
            r_mean  = float(np.mean(ep_rewards[-w:]))
            v_pct   = float(np.mean(ep_valid[-w:])) * 100.0
            d_mean  = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
            loss_s  = ""
            if metrics:
                loss_s = (f"  actor={metrics['actor_loss']:.4f}"
                          f"  critic={metrics['critic_loss']:.4f}"
                          f"  H={metrics['entropy']:.3f}")
            print(f"[{ep_idx+1:5d}] reward={r_mean:+.3f}  "
                  f"dist={d_mean:.3f}  valid={v_pct:.1f}%{loss_s}")

        if (ep_idx + 1) % cfg["checkpoint_every"] == 0:
            path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx+1}.pt")
            torch.save({
                "episode": ep_idx + 1,
                "actor":   actor.state_dict(),
                "critic":  critic.state_dict(),
                "opt_actor":  opt_actor.state_dict(),
                "opt_critic": opt_critic.state_dict(),
                "config":  cfg,
            }, path)
            print(f"  → checkpoint saved: {path}")

    return actor, critic


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> Dict:
    p = argparse.ArgumentParser(description="Train goal-conditioned fragment assembly agent.")
    p.add_argument("--n_episodes",      type=int,   default=DEFAULT_CFG["n_episodes"])
    p.add_argument("--n_frags",         type=int,   default=DEFAULT_CFG["n_frags"])
    p.add_argument("--min_frag_count",  type=int,   default=DEFAULT_CFG["min_frag_count"])
    p.add_argument("--n_targets",       type=int,   default=DEFAULT_CFG["n_targets"])
    p.add_argument("--max_steps",       type=int,   default=DEFAULT_CFG["max_steps"])
    p.add_argument("--hidden_dim",      type=int,   default=DEFAULT_CFG["hidden_dim"])
    p.add_argument("--lr_actor",        type=float, default=DEFAULT_CFG["lr_actor"])
    p.add_argument("--lr_critic",       type=float, default=DEFAULT_CFG["lr_critic"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--her_k",           type=int,   default=DEFAULT_CFG["her_k"])
    p.add_argument("--entropy_coef",    type=float, default=DEFAULT_CFG["entropy_coef"])
    p.add_argument("--device",          type=str,   default=DEFAULT_CFG["device"])
    p.add_argument("--checkpoint_dir",  type=str,   default=DEFAULT_CFG["checkpoint_dir"])
    p.add_argument("--log_every",         type=int,   default=DEFAULT_CFG["log_every"])
    p.add_argument("--checkpoint_every",  type=int,   default=DEFAULT_CFG["checkpoint_every"])
    args = p.parse_args()
    cfg = dict(DEFAULT_CFG)
    cfg.update(vars(args))
    return cfg


if __name__ == "__main__":
    train(_parse_args())
