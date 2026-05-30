"""
train.py
A2C-style (actor-critic with Monte-Carlo returns) training loop.

Supports both fingerprint and GNN encoders behind a single `--encoder`
switch so we keep one training implementation while preserving separate
model definitions in `models/`.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import _path_bootstrap  # noqa: F401

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from env import load_fragment_library, load_target_distribution
from env import (
    MolEnv, TERMINATE, StepResult,
    DEFAULT_PROPERTY_NAMES, DEFAULT_PROPERTY_SURROGATE_CFG,
    build_env_reward_config, parse_reward_vector, parse_property_names,
)
from models import Actor, Critic, GNNActor, GNNCritic
from env import ReplayBuffer, Episode, Transition


ModelActor = Union[Actor, GNNActor]
ModelCritic = Union[Critic, GNNCritic]


DEFAULT_CFG: Dict[str, Any] = dict(
    fragments_parquet="/mnt/data/m3_20m/outputs/fragments.parquet",
    parents_parquet="/mnt/data/m3_20m/outputs/parents.parquet",
    n_frags=200,
    min_frag_count=5_000,
    n_targets=300,
    max_steps=6,
    goal_properties=",".join(DEFAULT_PROPERTY_NAMES),
    reward_properties="",
    encoder="fp",
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
    use_property_surrogate_reward=False,
    property_surrogate_scale=DEFAULT_PROPERTY_SURROGATE_CFG["scale"],
    property_surrogate_dummy_bonus=DEFAULT_PROPERTY_SURROGATE_CFG["dummy_bonus"],
    property_surrogate_step_penalty=DEFAULT_PROPERTY_SURROGATE_CFG["step_penalty"],
    property_surrogate_weights="",
    property_surrogate_temperatures="",
    property_surrogate_invalid_score=DEFAULT_PROPERTY_SURROGATE_CFG["invalid_score"],
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir="checkpoints",
    exp_name="a2c_run",
    seed=1,
    use_wandb=False,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
)


def rollout_fp(env: MolEnv, actor: Actor, device: torch.device) -> Episode:
    state, goal, valid_actions = env.reset()
    ep = Episode()

    for _ in range(env.max_steps):
        af_np = env.get_action_features(valid_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t = torch.tensor(af_np, dtype=torch.float32, device=device)

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


def rollout_gnn(env: MolEnv, actor: GNNActor, device: torch.device) -> Episode:
    state, goal, valid_actions = env.reset()
    ep = Episode()

    for _ in range(env.max_steps):
        mol_graph = deepcopy(env._mol)
        af_np = env.get_action_features(valid_actions)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
        af_t = torch.tensor(af_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist = actor.action_dist(mol_graph, goal_t, af_t, device)
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
            info={**result.info, "mol_graph": mol_graph},
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break

        state = result.state
        valid_actions = next_valid

    return ep


def update_fp(
    actor: Actor,
    critic: Critic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch: list,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    entropies: List[float] = []

    values: List[torch.Tensor] = []
    returns: List[torch.Tensor] = []
    for t, G in batch:
        state_t = torch.tensor(t.state, dtype=torch.float32, device=device)
        G_t = torch.tensor(G, dtype=torch.float32, device=device)
        values.append(critic(state_t))
        returns.append(G_t)

    values_t = torch.stack(values)
    returns_t = torch.stack(returns)
    adv_raw = returns_t - values_t.detach()
    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    for i, (t, _) in enumerate(batch):
        state_t = torch.tensor(t.state, dtype=torch.float32, device=device)
        af_t = torch.tensor(t.action_feats, dtype=torch.float32, device=device)

        dist = actor.action_dist(state_t, af_t)
        log_prob = dist.log_prob(torch.tensor(t.action_idx, device=device))
        entropy = dist.entropy()

        actor_losses.append(-(log_prob * adv_norm[i]) - cfg["entropy_coef"] * entropy)
        critic_losses.append(cfg["critic_coef"] * (returns_t[i] - values_t[i]).pow(2))
        entropies.append(float(entropy.item()))

    total_actor = torch.stack(actor_losses).mean()
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
        "actor_loss": float(total_actor.item()),
        "critic_loss": float(total_critic.item()),
        "entropy": float(np.mean(entropies)),
    }


def update_gnn(
    actor: GNNActor,
    critic: GNNCritic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch: list,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    values: List[torch.Tensor] = []
    returns: List[torch.Tensor] = []
    for t, G in batch:
        mg = t.info.get("mol_graph")
        goal_t = torch.tensor(t.goal, dtype=torch.float32, device=device)
        G_t = torch.tensor(G, dtype=torch.float32, device=device)
        values.append(critic(mg, goal_t, device))
        returns.append(G_t)

    values_t = torch.stack(values)
    returns_t = torch.stack(returns)
    adv_raw = returns_t - values_t.detach()
    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    actor_losses: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []
    entropies: List[float] = []

    for i, (t, _) in enumerate(batch):
        mg = t.info.get("mol_graph")
        goal_t = torch.tensor(t.goal, dtype=torch.float32, device=device)
        af_t = torch.tensor(t.action_feats, dtype=torch.float32, device=device)

        dist = actor.action_dist(mg, goal_t, af_t, device)
        log_prob = dist.log_prob(torch.tensor(t.action_idx, device=device))
        entropy = dist.entropy()

        actor_losses.append(-(log_prob * adv_norm[i]) - cfg["entropy_coef"] * entropy)
        critic_losses.append(cfg["critic_coef"] * (returns_t[i] - values_t[i]).pow(2))
        entropies.append(float(entropy.item()))

    total_actor = torch.stack(actor_losses).mean()
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
        "actor_loss": float(total_actor.item()),
        "critic_loss": float(total_critic.item()),
        "entropy": float(np.mean(entropies)),
    }


def _rollout(
    env: MolEnv,
    actor: ModelActor,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Episode:
    if cfg["encoder"] == "gnn":
        return rollout_gnn(env, actor, device)  # type: ignore[arg-type]
    return rollout_fp(env, actor, device)  # type: ignore[arg-type]


def _update(
    actor: ModelActor,
    critic: ModelCritic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch: list,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    if cfg["encoder"] == "gnn":
        return update_gnn(actor, critic, opt_actor, opt_critic, batch, cfg, device)  # type: ignore[arg-type]
    return update_fp(actor, critic, opt_actor, opt_critic, batch, cfg, device)  # type: ignore[arg-type]


def _build_models(env: MolEnv, cfg: Dict[str, Any], device: torch.device) -> tuple[ModelActor, ModelCritic]:
    if cfg["encoder"] == "gnn":
        actor = GNNActor(
            gnn_dim=cfg["gnn_dim"],
            goal_dim=env.goal_dim,
            hidden_dim=cfg["hidden_dim"],
        ).to(device)
        critic = GNNCritic(
            gnn_dim=cfg["gnn_dim"],
            goal_dim=env.goal_dim,
            hidden_dim=cfg["hidden_dim"],
        ).to(device)
        return actor, critic

    actor = Actor(state_dim=env.state_dim, hidden_dim=cfg["hidden_dim"]).to(device)
    critic = Critic(state_dim=env.state_dim, hidden_dim=cfg["hidden_dim"]).to(device)
    return actor, critic


def train(cfg: Optional[Dict[str, Any]] = None) -> tuple[ModelActor, ModelCritic]:
    if cfg is None:
        cfg = dict(DEFAULT_CFG)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.init(
            project="cs224r-fragment-assembly",
            name=cfg["exp_name"],
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            tags=["a2c", cfg["encoder"]],
        )

    device = torch.device(cfg["device"])

    print("Loading fragment library ...")
    frags = load_fragment_library(
        cfg["fragments_parquet"],
        n=cfg["n_frags"],
        min_count=cfg["min_frag_count"],
    )
    print(f"  {len(frags)} fragments loaded")

    print("Loading target distribution ...")
    property_names = parse_property_names(cfg["goal_properties"])
    targets = load_target_distribution(
        cfg["parents_parquet"],
        n=cfg["n_targets"],
        property_names=property_names,
    )
    print(
        f"  {len(targets)} target molecules "
        + " ".join(
            f"| {name}∈[{targets[:, i].min():.2f},{targets[:, i].max():.2f}]"
            for i, name in enumerate(property_names)
        )
    )

    reward_cfg = build_env_reward_config(cfg)
    env = MolEnv(
        frags,
        targets,
        max_steps=cfg["max_steps"],
        property_names=property_names,
        reward_config=reward_cfg,
    )
    actor, critic = _build_models(env, cfg, device)

    if cfg["encoder"] == "gnn":
        n_actor = sum(p.numel() for p in actor.parameters())
        n_critic = sum(p.numel() for p in critic.parameters())
        print(f"  GNNActor params:  {n_actor:,}")
        print(f"  GNNCritic params: {n_critic:,}")

    opt_actor = optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    buffer = ReplayBuffer(
        max_episodes=2_000,
        gamma=cfg["gamma"],
        her_k=cfg["her_k"],
        goal_dim=env.goal_dim,
        reward_config=reward_cfg,
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    ep_rewards: List[float] = []
    ep_dists: List[float] = []
    ep_valid: List[float] = []

    print(f"\nTraining for {cfg['n_episodes']} episodes ...\n")

    for ep_idx in range(cfg["n_episodes"]):
        actor.train()
        critic.train()
        ep = _rollout(env, actor, cfg, device)
        buffer.push(ep)

        total_r = sum(t.reward for t in ep.transitions)
        ep_rewards.append(total_r)

        achieved = ep.achieved_goal
        ep_valid.append(1.0 if achieved is not None else 0.0)
        terminal_dist = ep.transitions[-1].info.get("terminal_distance") if ep.transitions else None
        if terminal_dist is not None:
            ep_dists.append(float(terminal_dist))

        if len(buffer) >= 5:
            batch = buffer.sample_transitions(cfg["batch_size"])
            metrics = _update(actor, critic, opt_actor, opt_critic, batch, cfg, device)
        else:
            metrics = {}

        if (ep_idx + 1) % cfg["log_every"] == 0:
            w = cfg["log_every"]
            r_mean = float(np.mean(ep_rewards[-w:]))
            v_pct = float(np.mean(ep_valid[-w:])) * 100.0
            d_mean = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
            loss_s = ""
            if metrics:
                loss_s = (
                    f"  actor={metrics['actor_loss']:.4f}"
                    f"  critic={metrics['critic_loss']:.4f}"
                    f"  H={metrics['entropy']:.3f}"
                )
            print(
                f"[{ep_idx+1:5d}] reward={r_mean:+.3f}  "
                f"dist={d_mean:.3f}  valid={v_pct:.1f}%{loss_s}"
            )
            if cfg.get("use_wandb") and _wandb is not None:
                log = {
                    "train/mean_reward": r_mean,
                    "train/mean_dist": d_mean,
                    "train/valid_pct": v_pct,
                }
                if metrics:
                    log["train/actor_loss"] = metrics["actor_loss"]
                    log["train/critic_loss"] = metrics["critic_loss"]
                    log["train/entropy"] = metrics["entropy"]
                _wandb.log(log, step=ep_idx + 1)

        if (ep_idx + 1) % cfg["checkpoint_every"] == 0:
            path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx+1}.pt")
            torch.save({
                "episode": ep_idx + 1,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "opt_actor": opt_actor.state_dict(),
                "opt_critic": opt_critic.state_dict(),
                "config": cfg,
            }, path)
            print(f"  -> checkpoint saved: {path}")

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.finish()

    return actor, critic


def _parse_args(default_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_cfg = dict(DEFAULT_CFG if default_cfg is None else default_cfg)

    p = argparse.ArgumentParser(description="Train goal-conditioned fragment assembly agent.")
    p.add_argument("--n_episodes", type=int, default=base_cfg["n_episodes"])
    p.add_argument("--n_frags", type=int, default=base_cfg["n_frags"])
    p.add_argument("--min_frag_count", type=int, default=base_cfg["min_frag_count"])
    p.add_argument("--n_targets", type=int, default=base_cfg["n_targets"])
    p.add_argument("--max_steps", type=int, default=base_cfg["max_steps"])
    p.add_argument("--goal_properties", type=str, default=base_cfg["goal_properties"])
    p.add_argument("--reward_properties", type=str, default=base_cfg["reward_properties"])
    p.add_argument("--encoder", choices=["fp", "gnn"], default=base_cfg["encoder"])
    p.add_argument("--hidden_dim", type=int, default=base_cfg["hidden_dim"])
    p.add_argument("--gnn_dim", type=int, default=base_cfg["gnn_dim"])
    p.add_argument("--lr_actor", type=float, default=base_cfg["lr_actor"])
    p.add_argument("--lr_critic", type=float, default=base_cfg["lr_critic"])
    p.add_argument("--batch_size", type=int, default=base_cfg["batch_size"])
    p.add_argument("--her_k", type=int, default=base_cfg["her_k"])
    p.add_argument("--entropy_coef", type=float, default=base_cfg["entropy_coef"])
    p.add_argument("--use_property_surrogate_reward", action="store_true", default=base_cfg["use_property_surrogate_reward"])
    p.add_argument("--property_surrogate_scale", type=float, default=base_cfg["property_surrogate_scale"])
    p.add_argument("--property_surrogate_dummy_bonus", type=float, default=base_cfg["property_surrogate_dummy_bonus"])
    p.add_argument("--property_surrogate_step_penalty", type=float, default=base_cfg["property_surrogate_step_penalty"])
    p.add_argument("--property_surrogate_weights", type=str, default=base_cfg["property_surrogate_weights"])
    p.add_argument("--property_surrogate_temperatures", type=str, default=base_cfg["property_surrogate_temperatures"])
    p.add_argument("--property_surrogate_invalid_score", type=float, default=base_cfg["property_surrogate_invalid_score"])
    p.add_argument("--device", type=str, default=base_cfg["device"])
    p.add_argument("--checkpoint_dir", type=str, default=base_cfg["checkpoint_dir"])
    p.add_argument("--log_every", type=int, default=base_cfg["log_every"])
    p.add_argument("--checkpoint_every", type=int, default=base_cfg["checkpoint_every"])
    p.add_argument("--exp_name", type=str, default=base_cfg["exp_name"])
    p.add_argument("--seed", type=int, default=base_cfg["seed"])
    p.add_argument("--use_wandb", action="store_true", default=base_cfg["use_wandb"])
    args = p.parse_args()

    cfg = dict(base_cfg)
    cfg.update(vars(args))

    if cfg["property_surrogate_weights"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_weights"] = parse_reward_vector(
            cfg["property_surrogate_weights"],
            expected_dim=len(reward_names),
        )
    if cfg["property_surrogate_temperatures"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_temperatures"] = parse_reward_vector(
            cfg["property_surrogate_temperatures"],
            expected_dim=len(reward_names),
        )
    return cfg


if __name__ == "__main__":
    train(_parse_args())
