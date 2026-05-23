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
    exp_name: str = "modal_run",
    seed: int = 1,
    n_episodes: int = 10_000,
    n_frags: int = 200,
    n_targets: int = 300,
    hidden_dim: int = 256,
    her_k: int = 4,
    entropy_coef: float = 0.005,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    log_every: int = 50,
    checkpoint_every: int = 500,
    use_wandb: bool = False,
):
    run_name = f"{exp_name}_seed{seed}"
    ckpt_dir = f"/mnt/results/checkpoints_{algo}_seed{seed}"

    extra_args = [
        "--seed",             str(seed),
        "--n_episodes",       str(n_episodes),
        "--n_frags",          str(n_frags),
        "--n_targets",        str(n_targets),
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
    if use_wandb:
        extra_args.append("--use_wandb")

    print(f"Launching {algo.upper()} training: {run_name}")
    train.remote(algo, run_name, extra_args)
