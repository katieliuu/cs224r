# Architecture

Goal-conditioned fragment assembly via A2C with Hindsight Experience Replay (HER). The agent builds drug-like molecules by iteratively attaching molecular fragments at BRICS-typed attachment sites, conditioned on a target property vector.

---

## Problem Formulation

Each episode is a Markov decision process over partial molecules:

- **State** $s_t = [\phi(m_t),\, g] \in \mathbb{R}^{515}$, where $\phi(m_t)$ is a Morgan fingerprint of the current partial molecule and $g \in [0,1]^3$ is the normalised target property vector.
- **Goal** $g = [\text{LogP}_\text{norm},\, \text{QED}_\text{norm},\, \text{TPSA}_\text{norm}]$, sampled from a held-out distribution of drug-like molecules.
- **Action**: attach a fragment from the library at a typed dummy site, or terminate. The action space is enumerated fresh at each step.
- **Terminal reward**: $r = -\|\phi_\text{norm}(m_\text{capped}) - g\|_2 - 0.05 \cdot n_\text{open}$, where $m_\text{capped}$ replaces all remaining dummy atoms with H before property computation, and $n_\text{open}$ is the number of unclosed attachment sites.
- **Intermediate reward**: 0 (sparse).

---

## Data

| Source | Description |
|---|---|
| `fragments.parquet` | BRICS fragments from M3-20M; top-200 by occurrence count (min 5,000) |
| `parents.parquet` | Drug-like parent molecules; 300 sampled for target distribution |

**Property normalisation** (fixed bounds):

| Property | Min | Max |
|---|---|---|
| LogP | -5.0 | 10.0 |
| QED | 0.0 | 1.0 |
| TPSA (Å²) | 0.0 | 200.0 |

---

## State and Action Representation

### State vector (515-dim)

```
s = [ mol_fp (512) | goal_norm (3) ]
```

- `mol_fp`: Morgan fingerprint, radius 2, 512 bits, computed on the partial molecule with dummy atoms retained.
- `goal_norm`: normalised (LogP, QED, TPSA) target vector.

### Action feature vector (546-dim)

```
a = [ frag_fp (512) | one_hot(frag_brics_type, 17) | one_hot(mol_brics_type, 17) ]
```

- `frag_fp`: Morgan fingerprint of the candidate fragment.
- The two BRICS one-hots encode the attachment type on the fragment and on the current molecule respectively.
- The terminate action is represented by a zero vector.

### BRICS compatibility filtering

Only (mol\_type, frag\_type) pairs that appear in the M3-20M attach\_demos data are proposed as valid actions, reducing the action space from $O(N_\text{frags} \times 16^2)$ to approximately 150 candidates per step.

---

## Neural Network Architecture

### Actor

Scores each candidate action independently given the current state encoding.

```
state_enc:  Linear(515, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 256)
scorer:     Linear(256 + 546, 256) -> ReLU -> Linear(256, 1)

logits[i] = scorer( [state_enc(s), a_i] )   for each action a_i
pi = Categorical(logits)
```

At each step, the state is encoded once; the scorer is applied independently to each (state\_enc, action\_feat) pair.

### Critic

Estimates the state-value function $V(s)$ where $s$ already encodes the goal.

```
net: Linear(515, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 1)
V(s) = net(s)
```

---

## Training Algorithm: A2C + HER

### Rollout

Each episode runs up to `max_steps = 6` steps. At each step, the actor samples an action from $\pi(\cdot \mid s_t, \{a_i\})$. The full episode is stored as a sequence of `Transition` objects and pushed to the replay buffer.

### Hindsight Experience Replay

For each stored episode with achieved goal $g'$, `her_k = 4` relabelled copies are generated:

1. Replace $g$ with $g'$ in the last 3 elements of every state and next-state vector.
2. Set terminal reward to 0 (distance from $g'$ to itself is 0).
3. Keep all intermediate rewards unchanged (0 or -0.1 for soft fails).

This provides a dense training signal even when the agent fails to reach the original goal.

### Update

Monte-Carlo returns are computed from each episode:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k, \quad \gamma = 0.99$$

Per update batch (size 64, sampled from real + HER transitions):

**Advantage normalisation**:
$$\hat{A}_i = \frac{(G_i - V(s_i)) - \mu_A}{\sigma_A + \epsilon}$$

**Actor loss**:
$$\mathcal{L}_\text{actor} = -\mathbb{E}\left[\log \pi(a \mid s) \cdot \hat{A}\right] - \alpha H[\pi], \quad \alpha = 0.005$$

**Critic loss**:
$$\mathcal{L}_\text{critic} = \lambda \, \mathbb{E}\left[(G - V(s))^2\right], \quad \lambda = 0.5$$

Gradients are clipped to norm 1.0. Separate Adam optimisers are used for actor ($\text{lr} = 3 \times 10^{-4}$) and critic ($\text{lr} = 1 \times 10^{-3}$).

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Fragment library size | 200 |
| Target distribution size | 300 |
| Max steps per episode | 6 |
| Training episodes | 10,000 |
| Batch size | 64 |
| Replay buffer (episodes) | 2,000 |
| HER relabellings per episode | 4 |
| Hidden dimension | 256 |
| Actor learning rate | 3e-4 |
| Critic learning rate | 1e-3 |
| Discount factor $\gamma$ | 0.99 |
| Entropy coefficient $\alpha$ | 0.005 |
| Critic coefficient $\lambda$ | 0.5 |
| Gradient clip norm | 1.0 |
| Open dummy penalty | 0.05 per site |

---

## Environment

**Reset**: sample a seed fragment uniformly from the library (must have at least one open attachment site); sample goal $g$ uniformly from the target distribution.

**Step**: merge the chosen fragment into the current molecule at the specified attachment site using `merge_by_labels`. If no open sites remain after merging, terminate immediately. If `max_steps` is reached, terminate.

**Termination**: cap all remaining dummy atoms with explicit H, compute (LogP, QED, TPSA) via RDKit, compute reward.

**Soft fail**: if merging raises an exception, return reward $-0.1$ and continue the episode.

---

## File Structure

```
cs224r/
    data.py        data loading and property normalisation
    features.py    state/action featurisation, property computation
    env.py         MolEnv (reset, step, action enumeration, BRICS filtering)
    model.py       Actor and Critic network definitions
    replay.py      ReplayBuffer with HER relabelling
    train.py       A2C training loop
    evaluate.py    evaluation metrics (distance, success rate, validity)
```
