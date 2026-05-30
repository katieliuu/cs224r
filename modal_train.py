"""
modal_train.py
Launch a single training run on Modal.

Usage
-----
  modal run modal_train.py                                          # a2c defaults
  modal run modal_train.py --algo gnn --exp-name gnn-run --use-wandb
  modal run --detach modal_train.py --algo ppo --exp-name ppo-v1 --seed 2 --use-wandb
"""
from modal_config import app, train


@app.local_entrypoint()
def main(
    algo: str = "a2c",
    encoder: str = "fp",
    exp_name: str = "modal_run",
    seed: int = 1,
    n_episodes: int = 10_000,
    n_frags: int = 200,
    n_targets: int = 300,
    goal_properties: str = "sLogP,QED,TPSA",
    reward_properties: str = "",
    hidden_dim: int = 256,
    her_k: int = 4,
    entropy_coef: float = 0.005,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    log_every: int = 50,
    checkpoint_every: int = 500,
    use_property_surrogate_reward: bool = False,
    property_surrogate_scale: float = 0.25,
    property_surrogate_dummy_bonus: float = 0.05,
    property_surrogate_step_penalty: float = 0.005,
    property_surrogate_weights: str = "",
    property_surrogate_temperatures: str = "",
    property_surrogate_invalid_score: float = 0.0,
    use_wandb: bool = False,
):
    run_name = f"{exp_name}_seed{seed}"
    ckpt_dir = f"/mnt/results/checkpoints_{algo}_seed{seed}"

    extra_args = [
        "--seed",             str(seed),
        "--n_episodes",       str(n_episodes),
        "--n_frags",          str(n_frags),
        "--n_targets",        str(n_targets),
        "--goal_properties",  goal_properties,
        "--encoder",          encoder,
        "--hidden_dim",       str(hidden_dim),
        "--her_k",            str(her_k),
        "--entropy_coef",     str(entropy_coef),
        "--lr_actor",         str(lr_actor),
        "--lr_critic",        str(lr_critic),
        "--log_every",        str(log_every),
        "--checkpoint_every", str(checkpoint_every),
        "--checkpoint_dir",   ckpt_dir,
        "--device",           "cuda",
    ]
    if use_property_surrogate_reward:
        extra_args.extend([
            "--use_property_surrogate_reward",
            "--property_surrogate_scale", str(property_surrogate_scale),
            "--property_surrogate_dummy_bonus", str(property_surrogate_dummy_bonus),
            "--property_surrogate_step_penalty", str(property_surrogate_step_penalty),
            "--reward_properties", reward_properties,
            "--property_surrogate_weights", property_surrogate_weights,
            "--property_surrogate_temperatures", property_surrogate_temperatures,
            "--property_surrogate_invalid_score", str(property_surrogate_invalid_score),
        ])
    if use_wandb:
        extra_args.append("--use_wandb")

    print(f"Launching {algo.upper()} training: {run_name}")
    train.remote(algo, run_name, extra_args)
