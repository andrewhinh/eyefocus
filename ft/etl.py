"""Data pre-processing using parent model for label generation."""

import math
import os
import random
from pathlib import Path

import modal
import torch
from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration

from ft.utils import (
    ARTIFACT_PATH,
    CLASSES,
    CPU,
    DATA_VOLUME,
    IMAGE,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    TIMEOUT,
    VOLUME_CONFIG,
)

# extract
dataset_name = "rootsautomation/ScreenSpot"  # ~1300 samples
data_split = "test"
src_dir = "self"  # 'ARTIFACT_PATH /' or '/DATA_VOLUME /' + src_dir
save_dir = "data"

# transform
model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # "meta-llama/Llama-3.2-90B-Vision-Instruct"
load_in_8bit = False
load_in_4bit = True
quant_type = "nf4"
use_double_quant = True

val_split = 0.1
max_new_tokens = 1
prompt = """
Task: Analyze the given computer screenshot to determine if it shows evidence of focused, productive activity or potentially distracting activity.

Instructions:
1. Examine the screenshot carefully.
2. Look for indicators of focused, productive activities including but not limited to:
   - Code editors or IDEs in use
   - Document editing software with substantial text visible
   - Spreadsheet applications with data or formulas
   - Research papers or educational materials being read
   - Professional design or modeling software in use
   - Terminal/command prompt windows with active commands
3. Identify potentially distracting activities including but not limited to:
   - Social media websites
   - Video streaming platforms
   - Unrelated news websites or apps
   - Online shopping sites
   - Music or video players
   - Messaging apps
   - Games or gaming platforms
4. Consider the context: e.g. a coding-related YouTube video might be considered focused activity for a programmer.

Response Format:
Return an integer value indicating whether the screenshot primarily shows evidence of distraction or focused activity.
0 for focused, 1 for distracted.
"""

# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, type(None), torch.dtype))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------


def download_model(is_local) -> tuple[AutoProcessor, AutoModel]:
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

    local_model_path = snapshot_download(
        config["model_path"],
        local_dir=Path("/") / PRETRAINED_VOLUME / config["model_path"] if not is_local else None,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
    )

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = MllamaForConditionalGeneration.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant,
        ),
        trust_remote_code=True,
    )
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(
        local_model_path, local_files_only=True, trust_remote_code=True, use_fast=False
    )

    return processor, model


def transform_img(image: Image, processor: AutoProcessor, device) -> torch.Tensor:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": config["prompt"]},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(device)
    return inputs


def gen_labels(is_local: bool = False) -> None:
    assert torch.cuda.is_available(), "GPU required"
    processor, model = download_model(is_local)

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
        out_dir = ARTIFACT_PATH / save_dir
    else:
        out_dir = Path("/") / DATA_VOLUME / save_dir
    train_idxs = random.sample(range(len(images)), k=math.ceil(len(images) * (1 - config["val_split"])))

    for i in tqdm(range(0, len(images), desc="Processing images")):
        inputs = transform_img(images[i], processor, model.device)
        output = model.generate(**inputs, max_new_tokens=config["max_new_tokens"])
        out = processor.decode(output[0])

        try:
            out = int(out)
        except ValueError:
            out = -1
        pred = CLASSES[out] if out >= 0 else "error"
        label_dir = out_dir / "train" / pred if i in train_idxs else out_dir / "validation" / pred
        os.makedirs(label_dir, exist_ok=True)
        img_path = label_dir / f"{i}.jpg"
        image = images[i]
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
