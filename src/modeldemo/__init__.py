from notifypy import Notify
import time
import timm
import mss
from PIL import Image
import traceback
import torch
from threading import Thread

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated
from pathlib import Path

from .utils import (
    CLASSES,
    download_llm,
    SEED,
    MAX_NEW_TOKENS,
    TITLE_PROMPT,
    MESSAGE_PROMPT,
    create_inputs,
)

# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False, "super_verbose": False}


# Model
CLASSIFIER = "eva_large_patch14_196.in22k_ft_in22k_in1k"
LLM = "Qwen/Qwen2.5-0.5B-Instruct"

# Notifypy
NOTIFICATION_INTERVAL = 8  # seconds
notification = Notify(
    default_application_name="Modeldemo",
    default_notification_urgency="critical",
    default_notification_icon=str(Path(__file__).parent / "icon.png"),
    default_notification_audio=str(Path(__file__).parent / "sound.wav"),
)


# # Helper fns
def capture_screenshot() -> Image:
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def classify(classifier, img) -> str:
    t0 = time.time()
    data_config = timm.data.resolve_model_data_config(classifier)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    output = classifier(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

    top1_probabilities, top1_class_indices = torch.topk(output.softmax(dim=1), k=1)
    pred_prob = top1_probabilities.item()
    predicted_class = CLASSES[top1_class_indices.item()]

    if state["verbose"]:
        print(f"Predicted class: {predicted_class} with probability: {pred_prob}")

    t1 = time.time()
    if state["super_verbose"]:
        print(f"Classify time: {t1 - t0}")
    return predicted_class


def generate(streamer, tokenizer, llm, prompt) -> str:
    t0 = time.time()
    num_tokens = 0
    thread = Thread(
        target=llm.generate,
        kwargs={
            **create_inputs(tokenizer, llm, prompt),
            "max_new_tokens": MAX_NEW_TOKENS,
            "streamer": streamer,
        },
    )
    thread.start()

    generated_text = ""
    for new_text in streamer:
        num_tokens += 1
        generated_text += new_text
        if state["verbose"]:
            print(new_text, end="", flush=True)
    if state["verbose"]:
        print()
    t1 = time.time()
    if state["super_verbose"]:
        print(f"Tok/sec: {num_tokens / (t1 - t0)}")

    return generated_text


# Typer CLI
def run() -> None:
    torch.manual_seed(SEED)

    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading models...", total=None)
            classifier = timm.create_model(CLASSIFIER, pretrained=True, num_classes=len(CLASSES))
            streamer, tokenizer, llm = download_llm(LLM, True)
        print("Models downloaded!")
    else:
        classifier = timm.create_model(CLASSIFIER, pretrained=True, num_classes=len(CLASSES))
        streamer, tokenizer, llm = download_llm(LLM, True)
    classifier.eval()
    llm.eval()

    while True:
        img = capture_screenshot()
        pred = classify(classifier, img)

        if pred == "distracted":
            notification.title = generate(streamer, tokenizer, llm, TITLE_PROMPT)
            notification.message = generate(streamer, tokenizer, llm, MESSAGE_PROMPT)
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
