"""
model.py
Actor and Critic networks for goal-conditioned fragment assembly.

Actor  — per-action scorer.
  Input : state s = concat(mol_fp, goal_norm)          shape (STATE_DIM,)
          candidate action features a_i                shape (n_actions, ACTION_FEAT_DIM)
  Output: logits over n_actions                        shape (n_actions,)

  At each step, the actor encodes the state once, then scores every
  candidate action by concatenating the state encoding with the action
  feature vector and passing through a small MLP.  This naturally
  handles a variable-size action set via masking.

Critic — state-value function V(s).
  Input : s = concat(mol_fp, goal_norm)                shape (STATE_DIM,)
  Output: scalar value estimate                        shape ()
"""
import _path_bootstrap  # noqa: F401

import torch
import torch.nn as nn
from features import STATE_DIM, ACTION_FEAT_DIM


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Per-action scoring actor with a shared state encoder."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_enc = _mlp(state_dim, hidden_dim, hidden_dim)
        # Scorer: [state_enc ‖ action_feat] → scalar logit
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,          # (STATE_DIM,) or (B, STATE_DIM)
        action_feats: torch.Tensor,   # (n_actions, ACTION_FEAT_DIM)
    ) -> torch.Tensor:                # logits (n_actions,)
        if state.dim() == 1:
            state = state.unsqueeze(0)              # (1, STATE_DIM)
        s_enc = self.state_enc(state)               # (1, hidden_dim)
        n = action_feats.shape[0]
        s_exp = s_enc.expand(n, -1)                 # (n_actions, hidden_dim)
        inp = torch.cat([s_exp, action_feats], dim=-1)
        return self.scorer(inp).squeeze(-1)         # (n_actions,)

    def action_dist(
        self,
        state: torch.Tensor,
        action_feats: torch.Tensor,
    ) -> torch.distributions.Categorical:
        logits = self.forward(state, action_feats)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    """State-value function V(s) where s already encodes the goal."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (..., STATE_DIM) → (...) scalar values."""
        return self.net(state).squeeze(-1)
