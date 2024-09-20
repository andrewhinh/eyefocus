import os

import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

SEED = 42
torch.manual_seed(SEED)

CLASSES = ["focused", "distracted"]

TORCH_DTYPE = torch.bfloat16
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
USE_DOUBLE_QUANT = True

MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TITLE_PROMPT = "Create a short, catchy title (2-5 words) about focusing on work or stopping procrastination. The title should be slightly humorous or playful in tone."
MESSAGE_PROMPT = "Create a short, humorous sentence (maximum 15 words) that playfully tells the user to get back to work or stop procrastinating."


def download_llm(model_path, is_local) -> tuple[TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM]:
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

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
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )
        if torch.cuda.is_available()
        else None,
    )
    model = torch.compile(model)

    return streamer, tokenizer, model


def create_inputs(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return model_inputs
