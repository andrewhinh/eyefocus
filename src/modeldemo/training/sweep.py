import argparse
import asyncio
import os
import subprocess
from pathlib import Path

import modal
import torch
import wandb
import yaml
from dotenv import load_dotenv

from src.modeldemo.training.utils import (
    ARTIFACT_PATH,
    IMAGE,
    PREFIX_PATH,
    RUNS_VOLUME,
    SWEEP_CONFIG_PATH,
    TIMEOUT,
    TRAIN_SCRIPT_PATH,
    VOLUME_CONFIG,
)

DEFAULT_N_RUNS = 10
n_avail_gpus = torch.cuda.device_count()


def program(is_local: bool):
    command = f"torchrun --standalone --nproc_per_node={n_avail_gpus} {'' if is_local else '/root/'}{TRAIN_SCRIPT_PATH} --is_local {is_local} --is_sweep True"
    subprocess.run(  # noqa: S603
        command.split(),
        check=True,
    )


async def run_agent(gpu_id: int, sweep_id: str, project: str, n_runs: int, is_local: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Using 1 {torch.cuda.get_device_name()} GPU.")

    wandb.agent(sweep_id, function=lambda: program(is_local), project=project, count=n_runs)


def start_sweep(config: dict, is_local: bool):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    runs_path = ARTIFACT_PATH / RUNS_VOLUME if is_local else Path("/") / RUNS_VOLUME
    os.makedirs(runs_path, exist_ok=True)
    os.environ["WANDB_DIR"] = str(runs_path)
    sweep_id = wandb.sweep(sweep=config, project=config["project"])
    return sweep_id


async def launch_agents(sweep_id: str, project: str, n_runs: int, is_local: bool):
    for gpu_id in range(n_avail_gpus):
        await run_agent(gpu_id, sweep_id, project, n_runs, is_local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=DEFAULT_N_RUNS)
    args = parser.parse_args()

    load_dotenv(PREFIX_PATH)

    config = yaml.safe_load(Path(SWEEP_CONFIG_PATH).read_text())
    sweep_id = start_sweep(config, True)
    asyncio.run(launch_agents(sweep_id, config["project"], args.n_runs, True))


# Modal
GPU_TYPE = "H100"
GPU_COUNT = 8
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "sweep_model"
app = modal.App(name=APP_NAME)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    volumes=VOLUME_CONFIG,
)
def m_run(gpu_id: int, sweep_id: str, project: str, n_runs: int):
    run_agent(gpu_id, sweep_id, project, n_runs, False)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    timeout=TIMEOUT,
)
def m_launch(config: dict, n_runs: int):
    sweep_id = start_sweep(config, False)
    handles = []
    for gpu_id in range(n_avail_gpus):
        handle = m_run.spawn(gpu_id, sweep_id, config["project"], n_runs)
        handles.append(handle)
    return handles


@app.local_entrypoint()
async def main(n_runs: int = DEFAULT_N_RUNS):
    config = yaml.safe_load(Path(SWEEP_CONFIG_PATH).read_text())

    handles = m_launch.remote(config, n_runs)
    await asyncio.gather(*(handle.get() for handle in handles))
