import subprocess
import warnings
from pathlib import PurePosixPath

import modal

NAME = "eyefocus"

# Modal
CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"
PYTHON_VERSION = "3.12"

PRETRAINED_VOLUME = f"{NAME}-pretrained"
DATA_VOLUME = f"{NAME}-data"
RUNS_VOLUME = f"{NAME}-runs"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(PRETRAINED_VOLUME, create_if_missing=True),
    f"/{DATA_VOLUME}": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    f"/{RUNS_VOLUME}": modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
}

CPU = 20  # cores (Modal max)

SECONDS = 60
MINUTES = 60  # seconds

GPU_IMAGE = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        TAG, add_python=PYTHON_VERSION
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add Python dependencies
        "pillow==10.4.0",
        "torch==2.4.1",
        "accelerate==0.34.2",
        "datasets==3.0.0",
        "python-dotenv==1.0.1",
        "timm==1.0.9",
        "torchvision==0.19.1",
        "hf_transfer==0.1.8",
        "wandb==0.17.7",
        "ninja==1.11.1.1",
        "packaging==24.1",
        "wheel==0.44.0",
        "pydantic==2.8.2",
        "term-image==0.7.2",
        "transformers==4.37.2",
    )
    .run_commands(  # add FlashAttention for faster inference using a shell command
        "pip install flash-attn==2.6.3 --no-build-isolation"
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HUGGINGFACE_HUB_CACHE": f"/{PRETRAINED_VOLUME}",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)


## subprocess for Modal
def _exec_subprocess(cmd: list[str]):
    """Executes subprocess and prints log to terminal while subprocess is running."""
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            line_str = line.decode()
            print(f"{line_str}", end="")

    if exitcode := process.wait() != 0:
        raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))


# Terminal image
warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
