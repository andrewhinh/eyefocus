import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import modal
import numpy as np
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer

from src.modeldemo.training.utils import (
    ARTIFACT_PATH,
    CPU,
    DATA_VOLUME,
    IMAGE,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    TIMEOUT,
    TRAIN_CONFIG_PATH,
    VOLUME_CONFIG,
)


def _tokenize(doc, enc):
    max_length = enc.model_max_length
    eot = enc.eos_token_id

    text = doc["text"]
    estimated_tokens = len(enc.encode(text, max_length=max_length, truncation=False))

    if estimated_tokens > max_length:
        chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]
        tokens = []
        for chunk in chunks:
            tokens.extend(enc.encode(chunk, add_special_tokens=False))
        tokens = [eot] + tokens
    else:
        tokens = [eot] + enc.encode(text)

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint32 = tokens_np.astype(np.uint32)
    return tokens_np_uint32


def download_data(
    dataset_name: str,
    dataset_remote_name: str,
    dataset_split: str,
    model_path: str,
    shard_size: int,
    data_dir: str,
    is_local: bool = False,
):
    if is_local:
        load_dotenv(PREFIX_PATH / ".env")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # create the cache the local directory if it doesn't exist yet
    model_name = model_path.split("/")[-1]
    if is_local:
        data_path = ARTIFACT_PATH / data_dir / model_name
    else:
        data_path = Path("/") / DATA_VOLUME / data_dir / model_name
    os.makedirs(data_path, exist_ok=True)

    # download the dataset
    fw = load_dataset(dataset_name, name=dataset_remote_name, split=dataset_split)

    # load tokenizer
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

    if is_local:
        local_model_path = ARTIFACT_PATH / model_path
    else:
        local_model_path = Path("/") / PRETRAINED_VOLUME / model_path
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path, exist_ok=True)
        snapshot_download(
            model_path,
            local_dir=local_model_path,
            ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
        )

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    tok_fn = partial(_tokenize, enc=tokenizer)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tok_fn, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = data_path / f"{split}_{shard_index:06d}"

                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                np.save(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = data_path / f"{split}_{shard_index:06d}"
            np.save(filename, all_tokens_np[:token_count])
            shard_index += 1


if __name__ == "__main__":
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    download_data(
        config["dataset_name"],
        config["dataset_remote_name"],
        config["dataset_split"],
        config["model_path"],
        config["shard_size"],
        config["data_dir"],
        True,
    )


# Modal
APP_NAME = "load_data"
app = modal.App(name=APP_NAME)


@app.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv(path=PREFIX_PATH)],
    volumes=VOLUME_CONFIG,
    timeout=TIMEOUT,
    cpu=CPU,
)
def run():
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    download_data(
        config["dataset_name"],
        config["dataset_remote_name"],
        config["dataset_split"],
        config["model_path"],
        config["shard_size"],
        config["data_dir"],
    )


@app.local_entrypoint()
def main():
    run.remote()
