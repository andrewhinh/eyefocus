import os
from contextlib import suppress
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import login, snapshot_download
from rich import print
from timm.data import create_transform, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import set_jit_fuser, setup_default_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

login(token=os.getenv("HF_TOKEN"), new_session=False)

has_compile = hasattr(torch, "compile")
torchcompile = None  # Enable compilation w/ specified backend (default: inductor).

# Device & distributed
device = "cpu"  # Device (accelerator) to use.
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

has_flash_attn = False
try:
    import flash_attn  # noqa: F401

    has_flash_attn = True
except ImportError:
    pass

has_bnb = False
try:
    import bitsandbytes  # noqa: F401

    has_bnb = True
except ImportError:
    pass
# -----------------------------------------------------------------------------


try:
    from apex import amp  # noqa: F401

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if torch.cuda.amp.autocast is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion  # noqa: F401

    has_functorch = True
except ImportError:
    has_functorch = False

num_gpu = torch.cuda.device_count() if device == "cuda" else 0  # Number of GPUS to use
amp = False  # use Native AMP for mixed precision training
amp_dtype = "float16"  # lower precision AMP dtype (default: float16)

# -----------------------------------------------------------------------------
# Classifier
pretrained = True  # use pre-trained model
channels_last = False  # Use channels_last memory layout
fuser = ""  # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')

## scripting / codegen
torchscript = False  # torch.jit.script the full model
aot_autograd = False  # Enable AOT Autograd support.

## Misc
test_pool = False  # enable test time pool
topk = 1  # Top-k


class_config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool, dict, list, Path, type(None)))
]
class_config = {k: globals()[k] for k in class_config_keys}  # will be useful for logging
class_config = {k: str(v) if isinstance(v, Path) else v for k, v in class_config.items()}  # since Path not serializable


def setup_classifier(model_name, verbose):  # noqa: C901
    setup_default_logging()

    if class_config["device"] == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(class_config["device"])

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if class_config["amp"]:
        assert has_native_amp, "Please update PyTorch to a version with native AMP (or use APEX)."
        assert class_config["amp_dtype"] in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if class_config["amp_dtype"] == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if verbose:
            print("Running inference in mixed precision with native PyTorch AMP.")
    else:
        if verbose:
            print("Running inference in float32. AMP not enabled.")

    if class_config["fuser"]:
        set_jit_fuser(class_config["fuser"])

    # create model
    model = create_model(model_name, pretrained=True)
    if verbose:
        print(f"Model {model_name} created, param count: {sum([m.numel() for m in model.parameters()])}")

    data_config = resolve_data_config(class_config, model=model)
    transforms = create_transform(**data_config, is_training=False)
    if class_config["test_pool"]:
        model, _ = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()
    if class_config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    if class_config["torchscript"]:
        model = torch.jit.script(model)
    elif class_config["torchcompile"]:
        assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        torch._dynamo.reset()
        model = torch.compile(model, backend=class_config["torchcompile"])
    elif class_config["aot_autograd"]:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if class_config["num_gpu"] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(class_config["num_gpu"])))

    return transforms, amp_autocast, model


# -----------------------------------------------------------------------------

# OCR


def setup_ocr(model_path):
    local_model_path = snapshot_download(
        model_path,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
    )

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
        attn_implementation="flash_attention_2" if has_flash_attn else None,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )
        if has_bnb
        else None,
    )
    assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
    torch._dynamo.reset()
    model = torch.compile(model, backend=class_config["torchcompile"])

    return streamer, tokenizer, model


# -----------------------------------------------------------------------------

# LLM
TORCH_DTYPE = torch.bfloat16
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
USE_DOUBLE_QUANT = True

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_NEW_TOKENS = 64
TITLE_PROMPT = "Look at the user's screen text: {text}. Then write a short (2-5 words) title about focusing on work or stopping procrastination, noting the user's screen text."
MESSAGE_PROMPT = "Look at the user's screen text: {text}. Then write a short (max 15 words) message about focusing on work or stopping procrastination, noting the user's screen text."


def setup_llm(model_path):
    local_model_path = snapshot_download(
        model_path,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
    )

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if has_flash_attn else None,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )
        if has_bnb
        else None,
    )
    assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
    torch._dynamo.reset()
    model = torch.compile(model, backend=class_config["torchcompile"])

    def transforms(prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        return model_inputs

    return streamer, transforms, model
