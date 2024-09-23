"""Data pre-processing using parent model for label generation."""

import math
import os
import random
from pathlib import Path

import modal
import torch
from datasets import load_dataset
from tqdm import tqdm

from ft.utils import (
    ARTIFACT_PATH,
    CLASSES,
    CPU,
    DATA_VOLUME,
    IMAGE,
    MAX_NEW_TOKENS,
    PREFIX_PATH,
    PROMPT,
    SAMPLE_BS,
    TEMPERATURE,
    TIMEOUT,
    VOLUME_CONFIG,
    download_model,
    transform_img,
)

# extract
DATASET_NAME = "rootsautomation/ScreenSpot"  # ~1300 samples
DATA_SPLIT = "test"

# transform
VAL_SPLIT = 0.1


def gen_labels(is_local: bool = False) -> None:
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    # TODO: not supported by InternVL2
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     DEVICE = "mps"
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    ds = load_dataset(DATASET_NAME, trust_remote_code=True, num_proc=max(1, os.cpu_count() // 2))[DATA_SPLIT]
    tokenizer, model = download_model(WORLD_SIZE, DEVICE, TORCH_DTYPE, is_local)

    if is_local:
        data_dir = ARTIFACT_PATH / "data"
    else:
        data_dir = Path("/") / DATA_VOLUME
    train_idxs = random.sample(range(len(ds)), k=math.ceil(len(ds) * (1 - VAL_SPLIT)))

    for i in tqdm(range(0, len(ds), SAMPLE_BS), desc="Processing batches"):
        batch = ds[i : i + SAMPLE_BS]
        pixel_vals = [transform_img(image).to(TORCH_DTYPE).to(DEVICE) for image in batch["image"]]
        num_patches_list = [pixel_vals[i].size(0) for i in range(len(pixel_vals))]
        pixel_values = torch.cat(pixel_vals, dim=0)

        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}
        questions = [f"<image>\n{PROMPT}"] * len(num_patches_list)
        with torch.no_grad():
            batch_out = model.batch_chat(
                tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config,
            )

        for j, out in enumerate(batch_out):
            try:
                out = int(out)
            except ValueError:
                out = -1
            pred = CLASSES[out] if out >= 0 else "error"
            label_dir = data_dir / "train" / pred if i + j in train_idxs else data_dir / "validation" / pred
            os.makedirs(label_dir, exist_ok=True)
            img_path = label_dir / f"{i + j}.jpg"
            image = batch["image"][j]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path)


if __name__ == "__main__":
    gen_labels(is_local=True)


# Modal
GPU_TYPE = "H100"
GPU_COUNT = 3  # min for InternVL2-Llama3-76B
GPU_SIZE = None  # options = None (40GB), "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "label_data"
app = modal.App(name=APP_NAME)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=TIMEOUT,
    cpu=CPU,
)
def run():
    gen_labels()


@app.local_entrypoint()
def main():
    run.remote()
