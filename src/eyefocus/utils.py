import os
from contextlib import suppress
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler, NanoLlavaChatHandler

# from llama_cpp.llama_speculative import LlamaPromptLookupDecoding  # not used because models are already small
from rich import print
from timm.data import create_transform, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import set_jit_fuser, setup_default_logging

login(token=os.getenv("HF_TOKEN"), new_session=False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

device = "cpu"  # Device (accelerator) to use.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# -----------------------------------------------------------------------------

# Classifier imports

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

# -----------------------------------------------------------------------------

# Classifier config

pretrained = True  # use pre-trained model
channels_last = False  # Use channels_last memory layout
fuser = ""  # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')

## scripting / codegen
torchscript = False  # torch.jit.script the full model
aot_autograd = False  # Enable AOT Autograd support.

## Device & distributed
num_gpu = torch.cuda.device_count() if device == "cuda" else 0  # Number of GPUS to use
amp = False  # use Native AMP for mixed precision training
amp_dtype = "float16"  # lower precision AMP dtype (default: float16)
has_compile = hasattr(torch, "compile")
torchcompile = None  # Enable compilation w/ specified backend (default: inductor).

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

# LLM

has_flash_attn = False
try:
    import flash_attn  # noqa: F401

    has_flash_attn = True
except ImportError:
    pass


def setup_gguf(model_path, gguf_model_path, n_ctx, clip_model_path=None, verbose=False):
    chat_handler = None
    if clip_model_path:
        local_clip_model_path = hf_hub_download(model_path, clip_model_path)
        if model_path == "vikhyatk/moondream2":
            chat_handler = MoondreamChatHandler(clip_model_path=local_clip_model_path, verbose=verbose)
        elif model_path == "abetlen/nanollava-gguf":
            chat_handler = NanoLlavaChatHandler(clip_model_path=local_clip_model_path, verbose=verbose)

    local_model_path = hf_hub_download(model_path, gguf_model_path)
    llm = Llama(
        model_path=local_model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=-1 if device == "cuda" else 0,
        flash_attn=has_flash_attn,
        verbose=verbose,
        # draft_model=LlamaPromptLookupDecoding(
        #     num_pred_tokens=10 if device == "cuda" else 2
        # ),  # 10 tokens on GPU, 2 tokens on CPU
    )

    return llm
