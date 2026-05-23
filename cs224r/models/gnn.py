"""
model_gnn.py
GNN-based Actor and Critic for goal-conditioned fragment assembly.

Replaces the Morgan fingerprint state encoding with a 3-layer MPNN
that operates directly on the MolGraph numpy arrays.

Node features (30-dim):
  10  element one-hot  (dummy=0, C, N, O, F, P, S, Cl, Br, I)
   1  formal charge    (clamped to [-2,2] and divided by 2)
   1  is_aromatic
   1  is_dummy
  17  attachment_label one-hot (BRICS types 0-16, non-zero only for dummy atoms)

Edge features (4-dim):
   4  bond_type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)

Architecture:
  GNN:      Linear(30, gnn_dim) → 3 × MPNNLayer(gnn_dim, 4, gnn_dim) → mean-pool
  state_enc: Linear(gnn_dim+3, H) → ReLU → Linear(H, H) → ReLU → Linear(H, H)
  scorer:   Linear(H + ACTION_FEAT_DIM, H) → ReLU → Linear(H, 1)   [Actor]
  value:    Linear(gnn_dim+3, H) → ReLU → Linear(H, H) → ReLU → Linear(H, 1)  [Critic]
"""
import _path_bootstrap  # noqa: F401

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.structs import MolGraph
from env.features import ACTION_FEAT_DIM
from env.data import GOAL_DIM

# ---------------------------------------------------------------------------
# Graph featurisation
# ---------------------------------------------------------------------------

ELEMENTS = [0, 6, 7, 8, 9, 15, 16, 17, 35, 53]   # dummy + C N O F P S Cl Br I
_ELEM2IDX = {e: i for i, e in enumerate(ELEMENTS)}
N_ELEM    = len(ELEMENTS)   # 10
EDGE_DIM  = 4               # bond type one-hot
NODE_DIM  = N_ELEM + 1 + 1 + 1 + 17   # = 30


def mol_graph_to_tensors(
    mg: MolGraph,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    node_feat  (N, NODE_DIM)   float32
    edge_index (2, 2M)         long   — directed, both directions
    edge_attr  (2M, EDGE_DIM)  float32
    """
    arr = mg.arrays
    N   = int(arr.atomic_num.shape[0])

    node_rows = np.zeros((N, NODE_DIM), dtype=np.float32)
    for i in range(N):
        z = int(arr.atomic_num[i])

        # element one-hot
        node_rows[i, _ELEM2IDX.get(z, 0)] = 1.0

        # scalar features
        node_rows[i, N_ELEM]     = float(arr.formal_charge[i]) / 2.0
        node_rows[i, N_ELEM + 1] = float(arr.is_aromatic[i]) if arr.is_aromatic is not None else 0.0
        node_rows[i, N_ELEM + 2] = 1.0 if z == 0 else 0.0

        # attachment label one-hot (BRICS type, only for dummy atoms)
        if z == 0 and arr.attachment_label is not None:
            lbl = arr.attachment_label[i]
            if lbl is not None:
                try:
                    li = int(lbl)
                    if 0 <= li < 17:
                        node_rows[i, N_ELEM + 3 + li] = 1.0
                except (ValueError, TypeError):
                    pass

    node_feat = torch.tensor(node_rows, dtype=torch.float32, device=device)

    M = int(arr.bonds.shape[0])
    if M > 0:
        b   = arr.bonds.astype(np.int64)
        src = np.concatenate([b[:, 0], b[:, 1]])
        dst = np.concatenate([b[:, 1], b[:, 0]])
        btype = np.concatenate([arr.bond_type, arr.bond_type]).astype(np.int64)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        bt_oh = np.zeros((2 * M, EDGE_DIM), dtype=np.float32)
        for j, bt in enumerate(btype):
            if 0 <= bt < EDGE_DIM:
                bt_oh[j, bt] = 1.0
        edge_attr = torch.tensor(bt_oh, dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float32, device=device)

    return node_feat, edge_index, edge_attr


# ---------------------------------------------------------------------------
# MPNN layer
# ---------------------------------------------------------------------------

class MPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.msg_fn    = nn.Linear(node_dim + edge_dim, out_dim)
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h:          torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (N, out_dim)
        N = h.shape[0]
        if edge_index.shape[1] == 0:
            agg = torch.zeros(N, self.msg_fn.out_features, device=h.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            msgs = F.relu(self.msg_fn(torch.cat([h[src], edge_attr], dim=-1)))   # (E, out_dim)
            agg  = torch.zeros(N, msgs.shape[-1], device=h.device)
            deg  = torch.zeros(N, 1, device=h.device)
            agg.index_add_(0, dst, msgs)
            deg.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device))
            agg = agg / (deg + 1e-6)

        return self.update_fn(torch.cat([h, agg], dim=-1))


# ---------------------------------------------------------------------------
# MolGNN — graph-level encoder
# ---------------------------------------------------------------------------

class MolGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM,
                 hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_feat:  torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (hidden_dim,)
        h = F.relu(self.node_embed(node_feat))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return F.relu(self.out_proj(h.mean(dim=0)))   # mean pooling → (hidden_dim,)


# ---------------------------------------------------------------------------
# GNNActor / GNNCritic
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GNNActor(nn.Module):
    def __init__(
        self,
        gnn_dim:         int = 128,
        goal_dim:        int = GOAL_DIM,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden_dim:      int = 256,
    ):
        super().__init__()
        self.gnn       = MolGNN(hidden_dim=gnn_dim)
        self.state_enc = _mlp(gnn_dim + goal_dim, hidden_dim, hidden_dim)
        self.scorer    = nn.Sequential(
            nn.Linear(hidden_dim + action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        mg:        MolGraph,
        goal_norm: torch.Tensor,   # (GOAL_DIM,)
        device:    torch.device,
    ) -> torch.Tensor:             # (hidden_dim,)
        nf, ei, ea = mol_graph_to_tensors(mg, device)
        g_emb = self.gnn(nf, ei, ea)
        return self.state_enc(torch.cat([g_emb, goal_norm], dim=-1))

    def forward(
        self,
        mg:           MolGraph,
        goal_norm:    torch.Tensor,   # (GOAL_DIM,)
        action_feats: torch.Tensor,   # (n_actions, ACTION_FEAT_DIM)
        device:       torch.device,
    ) -> torch.Tensor:                # logits (n_actions,)
        s_enc = self.encode(mg, goal_norm, device)   # (hidden_dim,)
        n     = action_feats.shape[0]
        inp   = torch.cat([s_enc.unsqueeze(0).expand(n, -1), action_feats], dim=-1)
        return self.scorer(inp).squeeze(-1)

    def action_dist(
        self,
        mg:           MolGraph,
        goal_norm:    torch.Tensor,
        action_feats: torch.Tensor,
        device:       torch.device,
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(
            logits=self.forward(mg, goal_norm, action_feats, device)
        )


class GNNCritic(nn.Module):
    def __init__(
        self,
        gnn_dim:    int = 128,
        goal_dim:   int = GOAL_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.gnn = MolGNN(hidden_dim=gnn_dim)
        self.net = _mlp(gnn_dim + goal_dim, hidden_dim, 1)

    def forward(
        self,
        mg:        MolGraph,
        goal_norm: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        nf, ei, ea = mol_graph_to_tensors(mg, device)
        g_emb = self.gnn(nf, ei, ea)
        return self.net(torch.cat([g_emb, goal_norm], dim=-1)).squeeze(-1)
