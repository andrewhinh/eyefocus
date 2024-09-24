"""Data pre-processing using parent model for label generation."""

import math
import os
import random
from pathlib import Path

import modal
import torch
from datasets import load_dataset
from PIL import Image
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
dataset_name = "rootsautomation/ScreenSpot"  # ~1300 samples
data_split = "test"
src_dir = "self"
data_dir = "data"

# transform
val_split = 0.1


def gen_labels(is_local: bool = False) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # TODO: not supported by InternVL2
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = "mps"
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer, model = download_model(world_size, device, torch_dtype, is_local)

    # check if images are present in src dir
    ## if not load_dataset
    ## else load from src dir
    if is_local:
        data_dir = ARTIFACT_PATH / src_dir
    else:
        data_dir = Path("/") / DATA_VOLUME / src_dir
    if not os.listdir(data_dir):
        ds = load_dataset(dataset_name, trust_remote_code=True, num_proc=max(1, os.cpu_count() // 2))[data_split]
        images = [s["image"] for s in ds]
    else:
        images = [Image.open(data_dir / f) for f in os.listdir(data_dir)]

    if is_local:
        data_dir = ARTIFACT_PATH / data_dir
    else:
        data_dir = Path("/") / DATA_VOLUME / data_dir
    train_idxs = random.sample(range(len(images)), k=math.ceil(len(images) * (1 - val_split)))

    for i in tqdm(range(0, len(images), SAMPLE_BS), desc="Processing batches"):
        batch = images[i : i + SAMPLE_BS]
        pixel_vals = [transform_img(image).to(torch_dtype).to(device) for image in batch]
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
