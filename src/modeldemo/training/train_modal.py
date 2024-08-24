import subprocess

import modal
import torch

from src.modeldemo.training.utils import (
    CPU,
    IMAGE,
    PREFIX_PATH,
    TIMEOUT,
    TRAIN_SCRIPT_PATH,
    VOLUME_CONFIG,
)

# Modal
GPU_TYPE = "H100"
GPU_COUNT = 8
GPU_SIZE = None  # options = None, "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "train_model"
app = modal.App(name=APP_NAME)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes=VOLUME_CONFIG,
    cpu=CPU,
)
def run():
    command = (
        f"torchrun --standalone --nproc_per_node={torch.cuda.device_count()} /root/{TRAIN_SCRIPT_PATH} --is_local False"
    )
    subprocess.run(command.split(), check=True)  # noqa: S603


@app.local_entrypoint()
def main():
    run.remote()
