from notifypy import Notify
import mss
import json
import time
from PIL import Image
import traceback
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated
from pathlib import Path

from src.modeldemo.utils import (
    clean_output,
    download_model,
    transform_img,
    TORCH_DTYPE,
    SEED,
    MAX_NEW_TOKENS,
    DO_SAMPLE,
    PROMPT,
)

# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False, "super_verbose": False}


# Model
MODEL_PATH = "OpenGVLab/InternVL2-1B"
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
# TODO: not supported by Internvl2
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     DEVICE = "mps"
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Notifypy
NOTIFICATION_INTERVAL = 8  # seconds
notification = Notify(
    default_application_name="Modeldemo",
    default_notification_urgency="critical",
    default_notification_icon=str(Path(__file__).parent / "icon.png"),
    default_notification_audio=str(Path(__file__).parent / "sound.wav"),
)


# Helper fns
def capture_screenshot() -> Image:
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def predict(img: Image, streamer: TextIteratorStreamer, tokenizer: AutoTokenizer, model: AutoModel) -> str:
    pixel_values = transform_img(img).to(TORCH_DTYPE).to(DEVICE)
    generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE, "streamer": streamer}

    thread = Thread(
        target=model.chat,
        kwargs={
            "tokenizer": tokenizer,
            "pixel_values": pixel_values,
            "question": PROMPT,
            "history": None,
            "return_history": False,
            "generation_config": generation_config,
        },
    )
    thread.start()

    generated_text = ""
    for new_text in streamer:
        if new_text == model.conv_template.sep:
            break
        generated_text += new_text
        if state["super_verbose"]:
            print(new_text, end="", flush=True)
    if state["super_verbose"]:
        print()
    return generated_text


# Typer CLI
def run() -> None:
    torch.manual_seed(SEED)

    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        print(f"Using device: {DEVICE}")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            streamer, tokenizer, model = download_model(MODEL_PATH, WORLD_SIZE, DEVICE, True)
        print("Model downloaded!")
    else:
        streamer, tokenizer, model = download_model(MODEL_PATH, WORLD_SIZE, DEVICE, True)
    model.eval()

    while True:
        img = capture_screenshot()
        pred = predict(img, streamer, tokenizer, model)
        if state["verbose"]:
            print(pred, end="", flush=True)
            print()

        try:
            format_pred = clean_output(pred)
            format_pred = json.loads(format_pred)
            is_distracted, title, message = format_pred["is_distracted"], format_pred["title"], format_pred["message"]
            is_distracted = bool(is_distracted)
        except Exception as e:
            if state["verbose"]:
                print(f"Failed to parse prediction: {e}")
            continue

        if is_distracted:
            notification.title = title
            notification.message = message
            notification.send(block=False)
            time.sleep(NOTIFICATION_INTERVAL)


@app.command(
    help="Stay [bold red]focused.[/bold red]",
    epilog="Made by [bold blue]Andrew Hinh.[/bold blue] :mechanical_arm::person_climbing:",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": True},
)
def main(verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0) -> None:
    try:
        state["verbose"] = verbose > 0
        state["super_verbose"] = verbose > 1
        run()
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print(traceback.format_exc())
            print("\n\nExiting...")
