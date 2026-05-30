"""
rewards.py
Reward utilities for sparse and dense goal-conditioned molecule rewards.

The default environment reward remains sparse. A dense reward experiment can
be enabled by scoring capped partial molecules against the goal property
vector and rewarding per-step improvements in that surrogate score.
"""
import _path_bootstrap  # noqa: F401

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .properties import DEFAULT_PROPERTY_NAMES, parse_property_names, property_indices, vector_for_indices


OPEN_DUMMY_PENALTY = 0.05


DEFAULT_PROPERTY_SURROGATE_CFG: Dict[str, Any] = {
    "mode": "property_surrogate",
    "scale": 0.25,
    "dummy_bonus": 0.05,
    "step_penalty": 0.005,
    "weights": None,
    "temperatures": None,
    "invalid_score": 0.0,
    "property_names": DEFAULT_PROPERTY_NAMES,
    "property_indices": tuple(range(len(DEFAULT_PROPERTY_NAMES))),
}


def parse_reward_vector(value: str, expected_dim: Optional[int] = None) -> Tuple[float, ...]:
    parts = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if expected_dim is not None and len(parts) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} comma-separated values, got {len(parts)}: {value!r}"
        )
    return parts


def build_env_reward_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cfg.get("use_property_surrogate_reward", False):
        return {"mode": "sparse"}

    goal_property_names = parse_property_names(cfg.get("goal_properties"))
    reward_property_names = parse_property_names(
        cfg.get("reward_properties"),
        default=goal_property_names,
    )
    reward_ix = property_indices(goal_property_names, reward_property_names)

    weights_cfg = cfg.get("property_surrogate_weights")
    if weights_cfg in (None, "", ()):
        weights = tuple(1.0 / len(reward_property_names) for _ in reward_property_names)
    else:
        weights = tuple(float(x) for x in weights_cfg)
        if len(weights) != len(reward_property_names):
            raise ValueError(
                "property_surrogate_weights must match reward_properties length "
                f"({len(reward_property_names)})."
            )

    temps_cfg = cfg.get("property_surrogate_temperatures")
    if temps_cfg in (None, "", ()):
        temperatures = tuple(0.10 for _ in reward_property_names)
    else:
        temperatures = tuple(float(x) for x in temps_cfg)
        if len(temperatures) != len(reward_property_names):
            raise ValueError(
                "property_surrogate_temperatures must match reward_properties length "
                f"({len(reward_property_names)})."
            )

    reward_cfg = dict(DEFAULT_PROPERTY_SURROGATE_CFG)
    reward_cfg.update({
        "scale": float(cfg["property_surrogate_scale"]),
        "dummy_bonus": float(cfg["property_surrogate_dummy_bonus"]),
        "step_penalty": float(cfg["property_surrogate_step_penalty"]),
        "weights": weights,
        "temperatures": temperatures,
        "invalid_score": float(cfg["property_surrogate_invalid_score"]),
        "property_names": reward_property_names,
        "property_indices": reward_ix,
    })
    return reward_cfg


def _to_array(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def terminal_reward(
    final_props: Optional[Sequence[float]],
    goal: Sequence[float],
    n_open: int,
    open_dummy_penalty: float = OPEN_DUMMY_PENALTY,
) -> tuple[float, Optional[float]]:
    final_props_arr = _to_array(final_props)
    goal_arr = np.asarray(goal, dtype=np.float32)
    open_pen = open_dummy_penalty * n_open
    if final_props_arr is None:
        return -2.0 - open_pen, None
    dist = float(np.linalg.norm(final_props_arr - goal_arr))
    return -(dist + open_pen), dist


def _property_match_score(
    props: Optional[Sequence[float]],
    goal: Sequence[float],
    reward_cfg: Dict[str, Any],
) -> float:
    indices = reward_cfg.get("property_indices")
    props_arr = vector_for_indices(props, indices) if indices is not None else _to_array(props)
    if props_arr is None:
        return float(reward_cfg.get("invalid_score", 0.0))

    goal_arr = vector_for_indices(goal, indices) if indices is not None else np.asarray(goal, dtype=np.float32)
    weights = np.asarray(reward_cfg["weights"], dtype=np.float32)
    temperatures = np.asarray(reward_cfg["temperatures"], dtype=np.float32)

    closeness = np.exp(-np.abs(props_arr - goal_arr) / np.maximum(temperatures, 1e-6))
    return float(np.dot(weights, closeness))


def surrogate_step_reward(
    prev_props: Optional[Sequence[float]],
    next_props: Optional[Sequence[float]],
    goal: Sequence[float],
    prev_open: int,
    next_open: int,
    reward_cfg: Dict[str, Any],
) -> float:
    prev_score = _property_match_score(prev_props, goal, reward_cfg)
    next_score = _property_match_score(next_props, goal, reward_cfg)
    return float(
        reward_cfg["scale"] * (next_score - prev_score)
        + reward_cfg["dummy_bonus"] * (prev_open - next_open)
        - reward_cfg["step_penalty"]
    )


def reward_from_context(
    reward_ctx: Optional[Dict[str, Any]],
    goal: Sequence[float],
    reward_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    if reward_ctx is None:
        raise ValueError("reward_from_context requires a stored reward_ctx.")

    if reward_cfg is None:
        reward_cfg = {"mode": "sparse"}

    kind = reward_ctx.get("kind")
    if kind == "constant":
        return float(reward_ctx["value"])

    if kind == "intermediate":
        if reward_cfg.get("mode") != "property_surrogate":
            return float(reward_ctx.get("base_reward", 0.0))
        return surrogate_step_reward(
            prev_props=reward_ctx.get("prev_props"),
            next_props=reward_ctx.get("next_props"),
            goal=goal,
            prev_open=int(reward_ctx.get("prev_open", 0)),
            next_open=int(reward_ctx.get("next_open", 0)),
            reward_cfg=reward_cfg,
        )

    if kind == "terminal":
        base_reward, _ = terminal_reward(
            final_props=reward_ctx.get("final_props"),
            goal=goal,
            n_open=int(reward_ctx.get("final_open", 0)),
            open_dummy_penalty=float(
                reward_ctx.get("open_dummy_penalty", OPEN_DUMMY_PENALTY)
            ),
        )
        if reward_cfg.get("mode") != "property_surrogate":
            return base_reward
        return base_reward + surrogate_step_reward(
            prev_props=reward_ctx.get("prev_props"),
            next_props=reward_ctx.get("final_props"),
            goal=goal,
            prev_open=int(reward_ctx.get("prev_open", reward_ctx.get("final_open", 0))),
            next_open=int(reward_ctx.get("final_open", 0)),
            reward_cfg=reward_cfg,
        )

    raise ValueError(f"Unknown reward context kind: {kind!r}")
