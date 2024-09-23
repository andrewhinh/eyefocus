import math
import os
import warnings
from pathlib import Path, PurePosixPath

import modal
import torch
import torchvision.transforms as T
from fastchat.conversation import get_conv_template
from huggingface_hub import login, snapshot_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ETL
CLASSES = ["focused", "distracted"]

PARENT_MODEL_PATH = "OpenGVLab/InternVL2-8B"  # "OpenGVLab/InternVL2-Llama3-76B"
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
USE_DOUBLE_QUANT = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_IMG_SIZE = 448
MAX_TILES = 12

MAX_NEW_TOKENS = 1
TEMPERATURE = 0.0
SAMPLE_BS = 2  # 8
PROMPT = """
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


### TODO: remove once original model.chat() .cuda() calls are removed
class CPUCompatibleInternVLChatModel(PreTrainedModel):
    def __init__(self, model, device):
        super().__init__(model.config)
        self.model = model
        self.device = device

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",  # noqa: S107
        IMG_END_TOKEN="</img>",  # noqa: S107
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",  # noqa: S107
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id  # ADDED: for self.model.generate

        conv = get_conv_template("internlm-chat")  # ADDED: to resolve KeyError: 'internlm2-chat'
        conv.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(conv.sep)

        history = [] if history is None else history
        for old_question, old_answer in history:
            conv.append_message(conv.roles[0], old_question)
            conv.append_message(conv.roles[1], old_answer)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)  # ADDED
        attention_mask = model_inputs["attention_mask"].to(self.device)  # ADDED
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.model.generate(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(conv.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            if verbose:
                print(query_to_print, response)
            return response


def download_model(world_size, device, dtype, is_local) -> tuple[AutoTokenizer, AutoModel]:
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

    local_model_path = snapshot_download(
        PARENT_MODEL_PATH,
        local_dir=Path("/") / PRETRAINED_VOLUME / PARENT_MODEL_PATH if not is_local else None,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
    )

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, trust_remote_code=True, use_fast=False
    )

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    def split_model(model_name: str):
        device_map = {}
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

    # ADDED: only use device_map if GPU is available
    device_map = None
    if world_size > 0:
        device_map = split_model(PARENT_MODEL_PATH.split("/")[-1])

    model = AutoModel.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )
        if torch.cuda.is_available()
        else None,
        trust_remote_code=True,
    )

    # ADDED: torch.compile
    model = torch.compile(model)

    # ADDED: for non-gpu compatibility
    if world_size == 0:
        model = model.to(device)
        model = CPUCompatibleInternVLChatModel(model)

    return tokenizer, model


def transform_img(image: Image) -> torch.Tensor:
    def build_transform() -> T.Compose:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        aspect_ratio: float, target_ratios: set[tuple[int, int]], width: int, height: int
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
                if area > 0.5 * INPUT_IMG_SIZE * INPUT_IMG_SIZE * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image: Image, min_num: int = 1, use_thumbnail: bool = False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, MAX_TILES + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= MAX_TILES and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height)

        # calculate the target width and height
        target_width = INPUT_IMG_SIZE * target_aspect_ratio[0]
        target_height = INPUT_IMG_SIZE * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // INPUT_IMG_SIZE)) * INPUT_IMG_SIZE,
                (i // (target_width // INPUT_IMG_SIZE)) * INPUT_IMG_SIZE,
                ((i % (target_width // INPUT_IMG_SIZE)) + 1) * INPUT_IMG_SIZE,
                ((i // (target_width // INPUT_IMG_SIZE)) + 1) * INPUT_IMG_SIZE,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            processed_images.append(thumbnail_img)
        return processed_images

    transform = build_transform()
    images = dynamic_preprocess(image, use_thumbnail=True)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# FT filepaths
PREFIX_PATH = Path(__file__).parent
ARTIFACT_PATH = PREFIX_PATH / "artifacts"
TRAIN_SCRIPT_PATH = PREFIX_PATH / "train.py"

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
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# term-image
warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class COLORS:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
