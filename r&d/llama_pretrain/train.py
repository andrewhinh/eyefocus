import argparse
import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from hellaswag import iterate_examples, render_example
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.modeldemo.training.utils import (
    ARTIFACT_PATH,
    DATA_VOLUME,
    PREFIX_PATH,
    PRETRAINED_VOLUME,
    RUNS_VOLUME,
    TRAIN_CONFIG_PATH,
)

# -----------------------------------------------------------------------------


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int64)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, data_dir, model_name, is_local, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        if is_local:
            data_root = ARTIFACT_PATH / data_dir / model_name
        else:
            data_root = Path("/") / DATA_VOLUME / data_dir / model_name
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
def configure_optimizers(self, lr, betas, eps, wd, device_type, master_process):
    import torch

    # start with all of the candidate parameters (that require grad)
    param_dict = dict(self.named_parameters())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
    return optimizer


def train(  # noqa: C901
    config: dict,
    is_local: bool = False,
    is_sweep: bool = False,
):
    print(f"Starting training run in {RUNS_VOLUME}.")
    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()

    if is_local:
        from dotenv import load_dotenv

        load_dotenv(PREFIX_PATH / ".env")

    # Set up DDP (distributed data parallel).
    ## torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    ## added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Set seed
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    # Set up wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    runs_path = ARTIFACT_PATH / RUNS_VOLUME if is_local else Path("/") / RUNS_VOLUME
    os.makedirs(runs_path, exist_ok=True)
    wandb_run = wandb.init(
        dir=runs_path,
        project=config["project"],
        name=config["run_name"],
    )
    hyperparam_config = wandb.config if is_sweep else config

    ## Log the config file
    if master_process:
        artifact = wandb.Artifact(name="train_config", type="config")
        artifact.add_file(local_path=TRAIN_CONFIG_PATH)
        wandb_run.log_artifact(artifact)

    # Load tokenizer (for sampling)
    if is_local:
        local_model_path = ARTIFACT_PATH / config["model_path"]
    else:
        local_model_path = Path("/") / PRETRAINED_VOLUME / config["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model_name = local_model_path.name

    # Load data
    total_batch_size = hyperparam_config["total_bs"]
    B = hyperparam_config["bs"]
    T = hyperparam_config["seq_len"]
    assert T <= tokenizer.model_max_length, "sequence length too long for model"
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"

    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        data_dir=config["data_dir"],
        model_name=model_name,
        is_local=is_local,
        split="train",
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        data_dir=config["data_dir"],
        model_name=model_name,
        is_local=is_local,
        split="val",
    )

    # Use TF32 for matmuls
    torch.set_float32_matmul_precision("high")

    # Load model
    torch_dtype = torch.bfloat16
    quant_config = BitsAndBytesConfig(
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    use_quant_config = config["load_in_8bit"] or config["load_in_4bit"]
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True if use_quant_config else False,
        quantization_config=quant_config if use_quant_config else None,
    )

    use_compile = False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    if use_quant_config:
        peft_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            r=hyperparam_config["rank"],
            lora_alpha=hyperparam_config["alpha"],
            lora_dropout=hyperparam_config["dropout"],
            use_rslora=hyperparam_config["use_rslora"],
            init_lora_weights=hyperparam_config["init_lora_weights"],
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    wandb.watch(model, log_freq=hyperparam_config["log_freq"])

    # Load optimizer
    optimizer = configure_optimizers(
        model,
        lr=hyperparam_config["max_lr"],
        betas=(hyperparam_config["beta1"], hyperparam_config["beta2"]),
        eps=hyperparam_config["eps"],
        wd=hyperparam_config["wd"],
        device_type=device_type,
        master_process=master_process,
    )

    # Load scheduler
    max_steps = hyperparam_config["max_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hyperparam_config["max_lr"],
        total_steps=max_steps,
        pct_start=hyperparam_config["pct_start"],
        anneal_strategy=hyperparam_config["anneal_strategy"],
        cycle_momentum=hyperparam_config["cycle_momentum"],
        base_momentum=hyperparam_config["base_momentum"],
        max_momentum=hyperparam_config["max_momentum"],
        div_factor=hyperparam_config["div_factor"],
        final_div_factor=hyperparam_config["final_div_factor"],
        three_phase=hyperparam_config["three_phase"],
    )

    # Training Loop
    best_val_loss = float("inf")
    best_model_path = None
    if is_local:
        ckpt_path = ARTIFACT_PATH / "ckpts" / config["run_name"]
    else:
        ckpt_path = Path("/") / RUNS_VOLUME / "ckpts" / config["run_name"]
    os.makedirs(ckpt_path, exist_ok=True)

    eval_interval = hyperparam_config["eval_interval"]
    val_steps = hyperparam_config["val_steps"]
    num_return_sequences = config["num_return_sequences"]

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        ## once in a while evaluate our validation loss
        if step % eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch_dtype):
                        outputs = model(x, labels=y)
                        logits = outputs.logits
                        loss = outputs.loss
                    loss = loss / val_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                wandb.log({"val_loss": val_loss_accum.item(), "step": step})

                if val_loss_accum.item() < best_val_loss:
                    best_val_loss = val_loss_accum.item()
                    if hyperparam_config["log_ckpt"]:
                        # optionally write model checkpoints
                        if is_local:
                            best_model_path = ckpt_path / f"{step:05d}_{val_loss_accum.item():.4f}.pt"
                        else:
                            best_model_path = ckpt_path / f"{step:05d}_{val_loss_accum.item():.4f}.pt"

                        raw_model.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        print(f"new best model saved to {best_model_path}")

        ## once in a while evaluate hellaswag
        if (step % eval_interval == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val", is_local)):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example, tokenizer)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch_dtype):
                        outputs = model(tokens)
                        logits = outputs.logits
                        loss = outputs.loss
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                wandb.log({"hella_acc": acc_norm, "step": step})

        ## once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % eval_interval == 0) or last_step) and (not use_compile) and (not is_sweep):
            model.eval()
            tokens = tokenizer.encode(config["text"])
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(config["seed"] + ddp_rank)
            while xgen.size(1) < config["max_length"]:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch_dtype):
                        outputs = model(xgen)  # (B, T, vocab_size)
                        logits = outputs.logits
                        loss = outputs.loss
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
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, : config["max_length"]].tolist()
                decoded = tokenizer.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        ## do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with torch.autocast(device_type=device_type, dtype=torch_dtype):
                outputs = model(x, labels=y)
                logits = outputs.logits
                loss = outputs.loss
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            wandb.log({"train_loss": loss_accum.item(), "step": step})

    if ddp:
        destroy_process_group()

    # Log the best model
    if best_model_path:
        if master_process:
            wandb_run.log_model(
                path=best_model_path,
                name=config["run_name"],
                aliases=["best"],
            )
    wandb_run.finish()


if __name__ == "__main__":
    config = yaml.safe_load(open(TRAIN_CONFIG_PATH))

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_local", type=bool, default=True)
    parser.add_argument("--is_sweep", type=bool, default=False)
    args = parser.parse_args()

    train(
        config,
        args.is_local,
        args.is_sweep,
    )
