"""Data pre-processing using parent model for label generation."""

import math
import os
from pathlib import Path

import modal
import torch
import torchvision.transforms as T
import yaml
from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ft.utils import (
    ARTIFACT_PATH,
    CPU,
    DATA_VOLUME,
    IMAGE,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    PROMPT,
    TIMEOUT,
    TRAIN_CONFIG_PATH,
    VOLUME_CONFIG,
)

# -----------------------------------------------------------------------------
# InternVL2-specific consts/helper functions

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_img(image: Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name: str) -> dict:
    """Creates a device map for the given model name."""
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]

    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


# -----------------------------------------------------------------------------


def gen_labels(config: dict, is_local: bool = False) -> None:
    ds = load_dataset(config["dataset_name"], trust_remote_code=True, num_proc=max(1, os.cpu_count() // 2))[
        config["data_split"]
    ]

    model_name = config["parent_model_path"].split("/")[-1]
    device_map = split_model(model_name)

    # Load tokenizer
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

    if is_local:
        local_model_path = ARTIFACT_PATH / "models" / config["parent_model_path"]
    else:
        local_model_path = Path("/") / PRETRAINED_VOLUME / config["parent_model_path"]
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
        snapshot_download(
            config["parent_model_path"],
            local_dir=local_model_path,
            ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
        )

    model = AutoModel.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, use_fast=False, trust_remote_code=True
    )

    # Generate labels
    responses = [""] * len(ds)
    for i in tqdm(range(0, len(ds), config["sample_bs"]), desc="Processing batches"):
        batch = ds[i : i + config["sample_bs"]]
        pixel_vals = [
            process_img(image, max_num=config["max_num"]).to(torch.bfloat16).cuda() for image in batch["image"]
        ]
        num_patches_list = [pixel_vals[i].size(0) for i in range(len(pixel_vals))]
        pixel_values = torch.cat(pixel_vals, dim=0)

        generation_config = {"max_new_tokens": config["max_new_tokens"], "do_sample": config["do_sample"]}
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
            out = out.replace("```json", "").replace("```", "").strip()
            responses[i + j] = out

    ds = ds.add_column("response", responses)
    if is_local:
        data_root = ARTIFACT_PATH / "data" / config["data_dir"]
    else:
        data_root = Path("/") / DATA_VOLUME / config["data_dir"]

    ds = ds.train_test_split(test_size=config["val_split"], seed=config["seed"])
    ds.save_to_disk(data_root)


if __name__ == "__main__":
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    gen_labels(config, is_local=True)


# Modal
GPU_TYPE = "H100"
GPU_COUNT = 3  # min for InternVL2-Llama3-76B
GPU_SIZE = None  # options = None, "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "label_data"
app = modal.App(name=APP_NAME)

## Modify image to install specific version of transformers
IMAGE = IMAGE.pip_install("transformers==4.37.2")


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=TIMEOUT,
    cpu=CPU,
)
def run():
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    gen_labels(config)


@app.local_entrypoint()
def main():
    run.remote()
