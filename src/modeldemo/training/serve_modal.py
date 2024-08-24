import os
import time
from pathlib import Path
from uuid import uuid4

import modal
import requests
import torch
import yaml
from huggingface_hub import login
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.modeldemo.training.utils import (
    ARTIFACT_PATH,
    IMAGE,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    SERVE_ALLOW_CONCURRENT_INPUTS,
    SERVE_CONFIG_PATH,
    SERVE_CONTAINER_IDLE_TIMEOUT,
    SERVE_TIMEOUT,
    VOLUME_CONFIG,
)

# Modal
GPU_TYPE = "a10g"
GPU_COUNT = 1
GPU_SIZE = None  # options = None, "40GB", "80GB"
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
if GPU_TYPE.lower() == "a100":
    GPU_CONFIG = modal.gpu.A100(count=GPU_COUNT, size=GPU_SIZE)

APP_NAME = "serve_model"
app = modal.App(name=APP_NAME)


# Model
@app.cls(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    gpu=GPU_CONFIG,
    timeout=SERVE_TIMEOUT,
    volumes=VOLUME_CONFIG,
    container_idle_timeout=SERVE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=SERVE_ALLOW_CONCURRENT_INPUTS,
)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    async def download_model(self):
        """Download the model and tokenizer."""
        self.config = yaml.safe_load(open(SERVE_CONFIG_PATH))

        # set up model
        login(token=os.getenv("HF_TOKEN"), new_session=not self.config["is_local"])

        if self.config["is_local"]:
            local_model_path = ARTIFACT_PATH / self.config["model_path"]
        else:
            local_model_path = Path("/") / PRETRAINED_VOLUME / self.config["model_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

        torch.set_float32_matmul_precision("high")  # use tf32

        torch_dtype = torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_8bit=self.config["load_in_8bit"],
            load_in_4bit=self.config["load_in_4bit"],
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        use_quant_config = self.config["load_in_8bit"] or self.config["load_in_4bit"]
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True if use_quant_config else False,
            quantization_config=quant_config if use_quant_config else None,
        )
        # model = torch.compile(model) # optionally torch compile the model

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        print(f"using device: {self.device}")

    @modal.web_endpoint(method="POST", docs=True)
    async def infer(
        self,
        query: str,
        seed: int,
        max_length: int,
    ) -> str:
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        self.model.eval()
        tokens = self.tokenizer.encode(query)
        tokens = torch.tensor(tokens, dtype=torch.long)
        xgen = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(seed)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    outputs = self.model(xgen)  # (B, T, vocab_size)
                    logits = outputs.logits
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # return the generated text
        tokens = xgen[:max_length].tolist()
        decoded = self.tokenizer.decode(tokens)

        print(f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds")

        return decoded


@app.local_entrypoint()
def main(
    twice: bool = True,
):
    model = Model()
    config = yaml.safe_load(open(SERVE_CONFIG_PATH))
    sample = {
        "query": config["query"],
        "seed": config["seed"],
        "max_length": config["max_length"],
    }

    response = requests.post(model.infer.web_url, json=sample)
    assert response.ok, response.status_code

    if twice:
        # second response is faster, because the Function is already running
        response = requests.post(model.infer.web_url, json=sample)
        assert response.ok, response.status_code
