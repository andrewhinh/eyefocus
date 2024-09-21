# Reference: https://github.com/karpathy/nanoGPT/blob/master/train.py

import argparse
import inspect
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import datasets
import torch
import torch.nn as nn
import yaml
from huggingface_hub import login, snapshot_download
from term_image.image import AutoImage
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.modeldemo.utils import (
    ARTIFACT_PATH,
    DATA_VOLUME,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    PROMPT,
    RUNS_VOLUME,
    TRAIN_CONFIG_PATH,
    Colors,
)

# load
GRADIENT_ACCUMULATION_STEPS = 2  # used to simulate larger batch sizes
BATCH_SIZE = 4  # if gradient_accumulation_steps > 1, this is the micro-batch size
BLOCK_SIZE = 2048  # to fit image + text tokens, TODO: make this dynamic

# model
MODEL_PATH = "vikhyatk/moondream2"
MODEL_REVISION = None  # https://github.com/vikhyat/moondream/issues/131

## I/O
OUT_DIR = "moondream2-ScreenSpot"  # 'out_dir' + str(time.time())
EVAL_INTERVAL = 1
LOG_INTERVAL = 1
EVAL_ITERS = 30
EVAL_ONLY = False  # if True, script exits right after the first eval
ALWAYS_SAVE_CHECKPOINT = True  # if True, always save a checkpoint after each eval
INIT_FROM = "scratch"  # 'scratch' or 'resume'

## wandb logging
WANDB_LOG = False  # disabled by default
WANDB_PROJECT = "modeldemo"
WANDB_RUN_NAME = "moondream2-ScreenSpot"  # 'wandb_run_name' + str(time.time())

## adamw optimizer
LEARNING_RATE = 1e-5  # max learning rate
MAX_ITERS = 300  # total number of training iterations, ~1 epoch
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-6
GRAD_CLIP = 1.0  # clip gradients at this value, or disable if == 0.0

## learning rate decay settings
DECAY_LR = True  # whether to decay the learning rate
WARMUP_ITERS = 30  # how many steps to warm up for
LR_DECAY_ITERS = 300  # should be ~= max_iters per Chinchilla
MIN_LR = 1e-6  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

## DDP settings
BACKEND = "nccl"  # 'nccl', 'gloo', etc.

## system
COMPILE = True  # use PyTorch 2.0 to compile the model to be faster

## sampling
NUM_SAMPLES = 5
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 5
EARLY_STOPPING = True


# -----------------------------------------------------------------------------
# Data
ANSWER_EOS = "<|endoftext|>"
# Number of tokens used to represent each image.
IMG_TOKENS = 729


class HFDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def collate_fn(batch, tokenizer, model, seq_len):
    images = [sample["image"].convert("RGB") for sample in batch]
    images = torch.stack(model.vision_encoder.preprocess(images))

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        q_t = tokenizer(f"\n\n{PROMPT}\n\nAnswer:", add_special_tokens=False).input_ids
        toks.extend(q_t)
        labs.extend([-100] * len(q_t))

        a_t = tokenizer(f" {sample['response']}{ANSWER_EOS}", add_special_tokens=False).input_ids
        toks.extend(a_t)
        labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = seq_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(la, dtype=torch.long) for la in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


# -----------------------------------------------------------------------------
# Training helper functions


def loss_fn(batch, model, device):
    images, tokens, labels, attn_mask = batch

    images = images.to(device)
    tokens = tokens.to(device)
    labels = labels.to(device)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        img_embs = model.vision_encoder.encoder(images)
        img_embs = model.vision_encoder.projection(img_embs)

    tok_embs = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, ctx, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, batch in enumerate(dataloader):
        if k >= eval_iters:
            break
        with ctx:
            loss = loss_fn(batch, model, device)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def sample_preds(model, dataset, num_samples, tokenizer, num_beams, no_repeat_ngram_size, early_stopping):
    model.eval()
    answers = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        with torch.no_grad():
            answer = model.answer_question(
                model.encode_image(sample["image"]),
                PROMPT,
                tokenizer=tokenizer,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
        answers.append(answer)
    model.train()
    return answers


def get_lr_fn(
    it, learning_rate, warmup_iters, lr_decay_iters, min_lr
):  # learning rate decay scheduler (cosine with warmup)
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------


def train(  # noqa: C901
    config: dict,
    is_local: bool,
    is_sweep: bool,
):
    # various inits, derived attributes, I/O setup
    if is_local:
        from dotenv import load_dotenv

        load_dotenv(PREFIX_PATH / ".env")

    ## Set up DDP (distributed data parallel).
    ### torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config["gradient_accumulation_steps"] % ddp_world_size == 0
        config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process (i.e. vanilla, non-DDP run)
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    tokens_per_iter = (
        config["gradient_accumulation_steps"] * ddp_world_size * config["batch_size"] * config["block_size"]
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    runs_path = ARTIFACT_PATH / RUNS_VOLUME if is_local else Path("/") / RUNS_VOLUME
    out_dir = runs_path / (config["out_dir"] + f"_{time.time():.0f}")
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Starting training run in {out_dir}.")
    torch.manual_seed(config["seed"] + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    dtype = (
        "float32" if device == "cpu" else "float16"  # bfloat16 not supported by safetensors (when converting to gguf)
    )
    ptdtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    login(token=os.getenv("HF_TOKEN"), new_session=not is_local)

    if is_local:
        local_model_path = ARTIFACT_PATH / "models" / config["model_path"]
    else:
        local_model_path = Path("/") / PRETRAINED_VOLUME / config["model_path"]
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
        snapshot_download(
            config["model_path"],
            local_dir=local_model_path,
            revision=config["model_revision"],
            ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Ensure safetensors
        )

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
        device_map={"": device},
    )
    checkpoint = {}
    if config["init_from"] == "resume":
        print(f"Resuming training from {out_dir}")
        checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"))
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    # crop down the model block size if desired, using model surgery
    if config["block_size"] < tokenizer.model_max_length:
        # model surgery to decrease the block size if necessary
        tokenizer.model_max_length = config["block_size"]
        model.text_model.transformer.embd.wte.weight = nn.Parameter(
            model.text_model.transformer.embd.wte.weight[: config["block_size"]]
        )
        for block in model.text_model.transformer.h:
            if hasattr(block.mixer, "bias"):
                block.mixer.bias = block.mixer.bias[:, :, : config["block_size"], : config["block_size"]]
    model.to(device)
    model.text_model.transformer.gradient_checkpointing_enable()

    # load data

    if is_local:
        data_root = ARTIFACT_PATH / "data" / config["data_dir"]
    else:
        data_root = Path("/") / DATA_VOLUME / config["data_dir"]

    train_dataset = datasets.load_from_disk(data_root / "train")
    val_dataset = datasets.load_from_disk(data_root / "test")

    train_loader = DataLoader(
        HFDataset(train_dataset),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, model, config["block_size"]),  # raw_model defined below
    )
    val_loader = DataLoader(
        HFDataset(val_dataset),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, model, config["block_size"]),  # raw_model defined below
    )

    # TODO: https://discuss.huggingface.co/t/attempting-to-unscale-fp16-gradients/91253
    # # initialize a GradScaler. If enabled=False scaler is a no-op
    # scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))

    # optimizer
    ## start with all of the candidate parameters (that require grad)
    param_dict = dict(model.named_parameters())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    ## create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    ## i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    ## Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        fused=use_fused,
    )
    if config["init_from"] == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    if config["wandb_log"] and master_process:
        import wandb

        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            dir=runs_path,
            project=config["wandb_project"],
            name=config["wandb_run_name"] + f"_{time.time():.0f}",
            config=config,
        )

        hyperparam_config = wandb.config if is_sweep else config
        wandb.watch(model, log_freq=hyperparam_config["log_interval"])

    # training loop
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    get_lr = partial(
        get_lr_fn,
        learning_rate=config["learning_rate"],
        warmup_iters=config["warmup_iters"],
        lr_decay_iters=config["lr_decay_iters"],
        min_lr=config["min_lr"],
    )

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config["eval_interval"] == 0 and master_process:
            losses = {
                "train": estimate_loss(raw_model, train_loader, config["eval_iters"], ctx, device),
                "val": estimate_loss(raw_model, val_loader, config["eval_iters"], ctx, device),
            }
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if config["wandb_log"]:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                    }
                )
            if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

                    if is_local:
                        best_model_path = out_dir / f"{iter_num:05d}_{best_val_loss:.4f}.pt"
                    else:
                        best_model_path = out_dir / f"{iter_num:05d}_{best_val_loss:.4f}.pt"

                    raw_model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
        if iter_num == 0 and config["eval_only"]:
            break

        # once in a while generate from the model (except step 0, which is noise)
        if (iter_num > 0 and iter_num % config["eval_interval"] == 0) and (not is_sweep):
            preds = sample_preds(
                raw_model,
                val_dataset,
                config["num_samples"],
                tokenizer,
                config["num_beams"],
                config["no_repeat_ngram_size"],
                config["early_stopping"],
            )
            for sample, pred in zip(val_dataset, preds, strict=False):
                terminal_image = AutoImage(sample["image"])
                terminal_image.draw()
                print(
                    Colors.BOLD,
                    Colors.GREEN,
                    f"Answer: {pred}",
                    Colors.END,
                    sep="",
                )

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step, batch in enumerate(train_loader):
            if micro_step >= config["gradient_accumulation_steps"]:
                break

            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == config["gradient_accumulation_steps"] - 1
            with ctx:
                loss = loss_fn(batch, raw_model, device)
                loss = (
                    loss / config["gradient_accumulation_steps"]
                )  # scale the loss to account for gradient accumulation
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
            # scaler.scale(loss).backward()
        # clip the gradient
        if config["grad_clip"] != 0.0:
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * config["gradient_accumulation_steps"]
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_local", action="store_false", dest="is_local", help="Disable local execution")
    parser.add_argument("--is_sweep", action="store_true", help="Enable sweep")
    args = parser.parse_args()

    train(
        config,
        args.is_local,
        args.is_sweep,
    )
