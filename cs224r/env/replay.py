"""
replay.py
Episode replay buffer with Hindsight Experience Replay (HER).

Storage
-------
Push complete episodes (list of Transition) after each rollout.
The buffer keeps up to `max_episodes` episodes, dropping the oldest.

Sampling
--------
sample_transitions(n) draws n (Transition, Monte-Carlo-return) pairs.
For each stored episode it also generates `her_k` HER relabellings:
  - The achieved goal is swapped in for the original goal.
  - The terminal transition gets reward 0 (distance to itself is 0).
  - All other transitions keep reward 0 (or –0.1 for soft-fails).
"""
import _path_bootstrap  # noqa: F401

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .data import GOAL_DIM


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray            # (STATE_DIM,)
    action_feats: np.ndarray     # (n_actions, ACTION_FEAT_DIM)
    action_idx: int              # index into action_feats
    reward: float
    next_state: np.ndarray       # (STATE_DIM,)
    next_action_feats: np.ndarray
    done: bool
    goal: np.ndarray             # (GOAL_DIM,) normalised
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    transitions: List[Transition] = field(default_factory=list)
    achieved_goal: Optional[np.ndarray] = None  # normalised props of completed mol

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(
        self,
        max_episodes: int = 2_000,
        gamma: float = 0.99,
        her_k: int = 4,
    ):
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.her_k = her_k
        self._episodes: List[Episode] = []

    def push(self, ep: Episode) -> None:
        self._episodes.append(ep)
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)

    def __len__(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_transitions(
        self, n: int
    ) -> List[Tuple[Transition, float]]:
        """Return up to n (Transition, MC-return) pairs from real + HER episodes."""
        pool: List[Tuple[Transition, float]] = []
        for ep in self._episodes:
            pool.extend(self._episode_to_pairs(ep, ep.transitions[0].goal))
            for _ in range(self.her_k):
                pairs = self._her_pairs(ep)
                if pairs:
                    pool.extend(pairs)

        if not pool:
            return []
        k = min(n, len(pool))
        indices = np.random.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in indices]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mc_returns(self, rewards: List[float]) -> List[float]:
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _episode_to_pairs(
        self, ep: Episode, goal: np.ndarray
    ) -> List[Tuple[Transition, float]]:
        rewards = [t.reward for t in ep.transitions]
        returns = self._mc_returns(rewards)
        return list(zip(ep.transitions, returns))

    def _her_pairs(self, ep: Episode) -> List[Tuple[Transition, float]]:
        """Relabel episode with achieved goal as new goal."""
        achieved = ep.achieved_goal
        if achieved is None:
            return []

        relabelled: List[Transition] = []
        for t in ep.transitions:
            # Swap the goal embedded in the last GOAL_DIM elements of state.
            new_s = t.state.copy()
            new_s[-GOAL_DIM:] = achieved
            new_ns = t.next_state.copy()
            new_ns[-GOAL_DIM:] = achieved

            # Terminal step has zero distance to itself → reward 0.
            new_r = 0.0 if t.done else t.reward

            relabelled.append(Transition(
                state=new_s,
                action_feats=t.action_feats,
                action_idx=t.action_idx,
                reward=new_r,
                next_state=new_ns,
                next_action_feats=t.next_action_feats,
                done=t.done,
                goal=achieved,
                info=t.info,
            ))

        returns = self._mc_returns([t.reward for t in relabelled])
        return list(zip(relabelled, returns))
