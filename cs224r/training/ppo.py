"""
train_ppo.py
PPO (Proximal Policy Optimization) for goal-conditioned fragment assembly.

On-policy: collect `rollout_eps` full episodes, compute advantages, then
run `ppo_epochs` gradient epochs over the entire batch before discarding it.
No replay buffer.  Online HER: each episode is immediately relabelled with
its achieved goal (her_k copies), then added to the batch.

Usage
-----
  python train_ppo.py
  python train_ppo.py --n_episodes 10000 --rollout_eps 16 --ppo_epochs 4
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import argparse
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from env import (
    load_fragment_library, load_target_distribution,
    MolEnv, Action, TERMINATE, StepResult,
    DEFAULT_PROPERTY_NAMES, DEFAULT_PROPERTY_SURROGATE_CFG,
    build_env_reward_config, parse_reward_vector, parse_property_names,
    reward_from_context,
)
from models import Actor, Critic
from env import Episode, Transition


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
    goal_properties=",".join(DEFAULT_PROPERTY_NAMES),
    reward_properties="",
    n_episodes=10_000,
    rollout_eps=16,       # episodes collected per PPO update
    ppo_epochs=4,         # gradient epochs per rollout batch
    clip_eps=0.2,         # PPO clipping epsilon
    batch_size=64,        # mini-batch size within each PPO epoch
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    her_k=4,              # online HER relabellings per episode
    entropy_coef=0.005,
    critic_coef=0.5,
    grad_clip=1.0,
    use_property_surrogate_reward=False,
    property_surrogate_scale=DEFAULT_PROPERTY_SURROGATE_CFG["scale"],
    property_surrogate_dummy_bonus=DEFAULT_PROPERTY_SURROGATE_CFG["dummy_bonus"],
    property_surrogate_step_penalty=DEFAULT_PROPERTY_SURROGATE_CFG["step_penalty"],
    property_surrogate_weights="",
    property_surrogate_temperatures="",
    property_surrogate_invalid_score=DEFAULT_PROPERTY_SURROGATE_CFG["invalid_score"],
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir="checkpoints_ppo",
    exp_name="ppo_run",
    seed=1,
    use_wandb=False,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
)


# ---------------------------------------------------------------------------
# PPO transition (adds old_log_prob to the standard Transition)
# ---------------------------------------------------------------------------

@dataclass
class PPOTransition:
    state:        np.ndarray
    action_feats: np.ndarray
    action_idx:   int
    old_log_prob: float
    reward:       float
    done:         bool
    goal:         np.ndarray
    ret:          float = 0.0   # Monte-Carlo return, filled in after episode ends


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_ppo(env: MolEnv, actor: Actor, device: torch.device) -> Tuple[Episode, List[PPOTransition]]:
    """Run one episode; return the Episode (for HER) and list of PPOTransitions."""
    state, goal, valid_actions = env.reset()
    ep = Episode()
    ppo_ts: List[PPOTransition] = []

    for _ in range(env.max_steps):
        af_np   = env.get_action_features(valid_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t    = torch.tensor(af_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist       = actor.action_dist(state_t, af_t)
            action_idx = int(dist.sample().item())
            old_lp     = float(dist.log_prob(torch.tensor(action_idx, device=device)).item())

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
        ppo_ts.append(PPOTransition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            old_log_prob=old_lp,
            reward=result.reward,
            done=result.done,
            goal=goal,
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break
        state         = result.state
        valid_actions = next_valid

    # Fill in MC returns
    G = 0.0
    for t in reversed(ppo_ts):
        G = t.reward + DEFAULT_CFG["gamma"] * G * (0.0 if t.done else 1.0)
        t.ret = G

    return ep, ppo_ts


def her_transitions(
    ep: Episode,
    her_k: int,
    gamma: float,
    reward_cfg: Optional[Dict] = None,
) -> List[PPOTransition]:
    """Online HER: relabel episode with achieved goal → her_k copies."""
    achieved = ep.achieved_goal
    if achieved is None:
        return []
    reward_cfg = reward_cfg or {"mode": "sparse"}
    out: List[PPOTransition] = []
    for _ in range(her_k):
        relabelled: List[Tuple[np.ndarray, np.ndarray, int, float, float, bool]] = []
        goal_dim = len(achieved)
        for t in ep.transitions:
            new_s = t.state.copy()
            new_s[-goal_dim:] = achieved
            reward_ctx = t.info.get("reward_ctx")
            new_r = (
                reward_from_context(reward_ctx, achieved, reward_cfg)
                if reward_ctx is not None else
                (0.0 if t.done else t.reward)
            )
            relabelled.append((new_s, t.action_feats, t.action_idx, new_r, t.done, t.goal))

        G = 0.0
        rets: List[float] = []
        for _, _, _, r, done, _ in reversed(relabelled):
            G = r + gamma * G * (0.0 if done else 1.0)
            rets.insert(0, G)

        for (new_s, af, aidx, r, done, _), ret in zip(relabelled, rets):
            out.append(PPOTransition(
                state=new_s,
                action_feats=af,
                action_idx=aidx,
                old_log_prob=0.0,   # HER transitions skip ratio clipping (old_lp=0 → ratio=1)
                reward=r,
                done=done,
                goal=achieved,
                ret=ret,
            ))
    return out


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def update_ppo(
    actor:     Actor,
    critic:    Critic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch:     List[PPOTransition],
    cfg:       Dict,
    device:    torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    # Pre-compute advantages (normalised) from full batch before gradient epochs
    with torch.no_grad():
        states_all = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=device
        )
        Vs_all  = critic(states_all)
        Gs_all  = torch.tensor([t.ret for t in batch], dtype=torch.float32, device=device)
        adv_raw = Gs_all - Vs_all
        adv_all = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    actor_losses_log:  List[float] = []
    critic_losses_log: List[float] = []
    entropies_log:     List[float] = []

    for _ in range(cfg["ppo_epochs"]):
        idx_perm = torch.randperm(len(batch))
        for start in range(0, len(batch), cfg["batch_size"]):
            mb_idx = idx_perm[start: start + cfg["batch_size"]]

            actor_losses:  List[torch.Tensor] = []
            critic_losses: List[torch.Tensor] = []

            for i in mb_idx.tolist():
                t   = batch[i]
                adv = adv_all[i]

                state_t = torch.tensor(t.state,        dtype=torch.float32, device=device)
                af_t    = torch.tensor(t.action_feats, dtype=torch.float32, device=device)
                G_t     = torch.tensor(t.ret,          dtype=torch.float32, device=device)

                dist     = actor.action_dist(state_t, af_t)
                new_lp   = dist.log_prob(torch.tensor(t.action_idx, device=device))
                entropy  = dist.entropy()

                # Clipped surrogate (HER transitions have old_lp=0 → ratio≈1, no clipping)
                ratio    = torch.exp(new_lp - t.old_log_prob)
                surr1    = ratio * adv
                surr2    = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * adv
                a_loss   = -torch.min(surr1, surr2) - cfg["entropy_coef"] * entropy

                V_s      = critic(state_t)
                c_loss   = cfg["critic_coef"] * (G_t - V_s).pow(2)

                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropies_log.append(float(entropy.item()))

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

            actor_losses_log.append(float(total_actor.item()))
            critic_losses_log.append(float(total_critic.item()))

    return {
        "actor_loss":  float(np.mean(actor_losses_log)),
        "critic_loss": float(np.mean(critic_losses_log)),
        "entropy":     float(np.mean(entropies_log)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Optional[Dict] = None) -> Tuple[Actor, Critic]:
    if cfg is None:
        cfg = DEFAULT_CFG

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.init(
            project="cs224r-fragment-assembly",
            name=cfg["exp_name"],
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            tags=["ppo"],
        )

    device = torch.device(cfg["device"])

    print("Loading fragment library ...")
    frags = load_fragment_library(
        cfg["fragments_parquet"], n=cfg["n_frags"], min_count=cfg["min_frag_count"]
    )
    print(f"  {len(frags)} fragments")

    print("Loading target distribution ...")
    property_names = parse_property_names(cfg["goal_properties"])
    targets = load_target_distribution(
        cfg["parents_parquet"],
        n=cfg["n_targets"],
        property_names=property_names,
    )
    print(f"  {len(targets)} targets")

    reward_cfg = build_env_reward_config(cfg)
    env    = MolEnv(
        frags,
        targets,
        max_steps=cfg["max_steps"],
        property_names=property_names,
        reward_config=reward_cfg,
    )
    actor  = Actor(state_dim=env.state_dim, hidden_dim=cfg["hidden_dim"]).to(device)
    critic = Critic(state_dim=env.state_dim, hidden_dim=cfg["hidden_dim"]).to(device)

    opt_actor  = optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ep_rewards: List[float] = []
    ep_dists:   List[float] = []
    ep_valid:   List[float] = []
    metrics:    Dict = {}

    print(f"\nTraining for {cfg['n_episodes']} episodes ...\n")

    ep_idx = 0
    while ep_idx < cfg["n_episodes"]:
        actor.train(); critic.train()

        # Collect rollout_eps episodes
        batch: List[PPOTransition] = []
        for _ in range(cfg["rollout_eps"]):
            if ep_idx >= cfg["n_episodes"]:
                break

            ep, ppo_ts = rollout_ppo(env, actor, device)
            batch.extend(ppo_ts)
            batch.extend(her_transitions(ep, cfg["her_k"], cfg["gamma"], reward_cfg))

            total_r = sum(t.reward for t in ep.transitions)
            ep_rewards.append(total_r)
            achieved = ep.achieved_goal
            ep_valid.append(1.0 if achieved is not None else 0.0)
            terminal_dist = ep.transitions[-1].info.get("terminal_distance") if ep.transitions else None
            if terminal_dist is not None:
                ep_dists.append(float(terminal_dist))

            ep_idx += 1

            if ep_idx % cfg["log_every"] == 0:
                w = cfg["log_every"]
                r_mean = float(np.mean(ep_rewards[-w:]))
                v_pct  = float(np.mean(ep_valid[-w:])) * 100.0
                d_mean = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
                loss_s = ""
                if metrics:
                    loss_s = (f"  actor={metrics['actor_loss']:.4f}"
                              f"  critic={metrics['critic_loss']:.4f}"
                              f"  H={metrics['entropy']:.3f}")
                print(f"[{ep_idx:5d}] reward={r_mean:+.3f}  "
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
                    _wandb.log(log, step=ep_idx)

            if ep_idx % cfg["checkpoint_every"] == 0:
                path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx}.pt")
                torch.save({
                    "episode": ep_idx,
                    "actor":   actor.state_dict(),
                    "critic":  critic.state_dict(),
                    "config":  cfg,
                }, path)
                print(f"  -> checkpoint: {path}")

        metrics = update_ppo(actor, critic, opt_actor, opt_critic, batch, cfg, device)

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.finish()

    return actor, critic


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Dict:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes",      type=int,   default=DEFAULT_CFG["n_episodes"])
    p.add_argument("--rollout_eps",     type=int,   default=DEFAULT_CFG["rollout_eps"])
    p.add_argument("--ppo_epochs",      type=int,   default=DEFAULT_CFG["ppo_epochs"])
    p.add_argument("--clip_eps",        type=float, default=DEFAULT_CFG["clip_eps"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--n_frags",         type=int,   default=DEFAULT_CFG["n_frags"])
    p.add_argument("--n_targets",       type=int,   default=DEFAULT_CFG["n_targets"])
    p.add_argument("--goal_properties", type=str,   default=DEFAULT_CFG["goal_properties"])
    p.add_argument("--reward_properties", type=str, default=DEFAULT_CFG["reward_properties"])
    p.add_argument("--hidden_dim",      type=int,   default=DEFAULT_CFG["hidden_dim"])
    p.add_argument("--her_k",           type=int,   default=DEFAULT_CFG["her_k"])
    p.add_argument("--entropy_coef",    type=float, default=DEFAULT_CFG["entropy_coef"])
    p.add_argument("--use_property_surrogate_reward", action="store_true", default=False)
    p.add_argument("--property_surrogate_scale", type=float, default=DEFAULT_CFG["property_surrogate_scale"])
    p.add_argument("--property_surrogate_dummy_bonus", type=float, default=DEFAULT_CFG["property_surrogate_dummy_bonus"])
    p.add_argument("--property_surrogate_step_penalty", type=float, default=DEFAULT_CFG["property_surrogate_step_penalty"])
    p.add_argument(
        "--property_surrogate_weights",
        type=str,
        default=DEFAULT_CFG["property_surrogate_weights"],
    )
    p.add_argument(
        "--property_surrogate_temperatures",
        type=str,
        default=DEFAULT_CFG["property_surrogate_temperatures"],
    )
    p.add_argument("--property_surrogate_invalid_score", type=float, default=DEFAULT_CFG["property_surrogate_invalid_score"])
    p.add_argument("--device",          type=str,   default=DEFAULT_CFG["device"])
    p.add_argument("--checkpoint_dir",  type=str,   default=DEFAULT_CFG["checkpoint_dir"])
    p.add_argument("--log_every",       type=int,   default=DEFAULT_CFG["log_every"])
    p.add_argument("--checkpoint_every",type=int,   default=DEFAULT_CFG["checkpoint_every"])
    p.add_argument("--exp_name",   type=str,            default=DEFAULT_CFG["exp_name"])
    p.add_argument("--seed",       type=int,            default=DEFAULT_CFG["seed"])
    p.add_argument("--use_wandb",  action="store_true", default=False)
    args = p.parse_args()
    cfg  = dict(DEFAULT_CFG)
    cfg.update(vars(args))
    if cfg["property_surrogate_weights"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_weights"] = parse_reward_vector(
            cfg["property_surrogate_weights"], expected_dim=len(reward_names)
        )
    if cfg["property_surrogate_temperatures"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_temperatures"] = parse_reward_vector(
            cfg["property_surrogate_temperatures"], expected_dim=len(reward_names)
        )
    return cfg


if __name__ == "__main__":
    train(_parse_args())
