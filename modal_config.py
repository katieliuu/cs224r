"""
modal_config.py
Shared Modal infrastructure: app, image, volumes, and train function.

Setup (once):
  modal secret create wandb WANDB_API_KEY=<your-key>
  modal volume create cs224r-data      # then upload parquet files
  modal volume create cs224r-results

Run training:
  modal run modal_train.py --algo a2c --exp-name my-run --use-wandb
  modal run --detach modal_train_para.py --algo gnn --exp-name sweep --use-wandb
"""
import modal

app = modal.App("cs224r-fragment-assembly")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libxrender1", "libxext6", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "rdkit",
        "pyarrow",
        "pandas",
        "psutil",
        "Pillow",
        "wandb",
        "matplotlib",
    )
    .add_local_dir(".", remote_path="/root/cs224r",
                   ignore=["cs224r/results/", "cs224r/logs/",
                           "**/__pycache__", "**/*.pyc", "**/*.pth"])
)

# /mnt/data  — upload your parquet files here once with `modal volume put`
# /mnt/results — checkpoints and val_results JSON land here
data_volume    = modal.Volume.from_name("cs224r-data",    create_if_missing=True)
results_volume = modal.Volume.from_name("cs224r-results", create_if_missing=True)

_secrets = [modal.Secret.from_name("wandb")]


@app.function(
    image=image,
    volumes={
        "/mnt/data":    data_volume,
        "/mnt/results": results_volume,
    },
    secrets=_secrets,
    gpu="T4",
    timeout=18_000,  # 5 hours
)
def train(algo: str, exp_name: str, extra_args: list[str]) -> None:
    import os
    import subprocess

    os.chdir("/root/cs224r")

    script = {
        "a2c": "cs224r/training/a2c.py",
        "ppo": "cs224r/training/ppo.py",
        "gnn": "cs224r/training/gnn.py",
    }[algo]

    cmd = ["python", script, "--exp_name", exp_name] + extra_args
    subprocess.run(cmd, check=True)
    results_volume.commit()
