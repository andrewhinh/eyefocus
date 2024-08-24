from pathlib import Path, PurePosixPath

import modal

# Constants
PREFIX_PATH = Path("src/modeldemo")
ARTIFACT_PATH = PREFIX_PATH / "training" / "artifacts"
TRAIN_CONFIG_PATH = PREFIX_PATH / "config" / "train.yml"
TRAIN_SCRIPT_PATH = PREFIX_PATH / "training" / "train.py"
SWEEP_CONFIG_PATH = PREFIX_PATH / "config" / "sweep.yml"
EVAL_CONFIG_PATH = PREFIX_PATH / "config" / "eval.yml"
SERVE_CONFIG_PATH = PREFIX_PATH / "config" / "serve.yml"

# Modal
CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"
PYTHON_VERSION = "3.11"

PRETRAINED_VOLUME = "pretrained"
DATA_VOLUME = "data"
RUNS_VOLUME = "runs"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(PRETRAINED_VOLUME, create_if_missing=True),
    f"/{DATA_VOLUME}": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    f"/{RUNS_VOLUME}": modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
}

CPU = 20  # cores (Modal max)

MINUTES = 60  # seconds
TIMEOUT = 24 * 60 * MINUTES

SERVE_TIMEOUT = 2 * MINUTES
SERVE_CONTAINER_IDLE_TIMEOUT = 5 * MINUTES
SERVE_ALLOW_CONCURRENT_INPUTS = 100

IMAGE = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        TAG, add_python=PYTHON_VERSION
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add Python dependencies
        "bitsandbytes==0.43.3",
        "datasets==2.21.0",
        "hf_transfer==0.1.8",
        "numpy==2.1.0",
        "peft==0.12.0",
        "scipy==1.14.1",
        "torch==2.4.0",
        "transformers==4.44.2",
        "wandb==0.17.7",
        "accelerate==0.33.0",
        "ninja==1.11.1.1",
        "packaging==24.1",
        "wheel==0.44.0",
        "python-dotenv==1.0.1",
        "pydantic==2.8.2",
    )
    .run_commands(  # add FlashAttention for faster inference using a shell command
        "pip install flash-attn==2.6.3 --no-build-isolation"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .copy_local_file(
        TRAIN_SCRIPT_PATH,
        f"/root/{TRAIN_SCRIPT_PATH}",
    )
)
