from notifypy import Notify
import time

from pathlib import Path

import torch
from PIL import Image
import mss
import traceback
from threading import Thread
from ghapi.all import GhApi


from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated

from .utils import (
    MAX_NEW_TOKENS,
    TITLE_PROMPT,
    MESSAGE_PROMPT,
    setup_classifier,
    class_config,
    setup_llm,
    extract_dates,
)

api = GhApi()
per_page = 30
n_pages = 8
org = "EurekaLabsAI"

# -----------------------------------------------------------------------------

# Classifier
CLASSIFIER = "hf_hub:andrewhinh/resnet152-224-Screenspot"

# LLM
LLM = "Qwen/Qwen2.5-0.5B-Instruct"


# -----------------------------------------------------------------------------


# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False}

# -----------------------------------------------------------------------------

# Notifypy
NOTIFICATION_INTERVAL = 8  # seconds
notification = Notify(
    default_application_name="Modeldemo",
    default_notification_urgency="critical",
    default_notification_icon=str(Path(__file__).parent / "icon.png"),
    default_notification_audio=str(Path(__file__).parent / "sound.wav"),
)

# -----------------------------------------------------------------------------


# Helper fns
def capture_screenshot() -> Image:
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def classify(model, transforms, amp_autocast, image: Image):
    t0 = time.time()

    device = torch.device(class_config["device"])
    img_pt = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with amp_autocast():
            output = model(img_pt)
    output = output.softmax(-1)
    output, indices = output.topk(class_config["topk"])
    labels = model.pretrained_cfg["label_names"]
    predictions = [{"label": labels[i], "score": v.item()} for i, v in zip(indices, output, strict=False)]
    preds, probs = [p["label"] for p in predictions], [p["score"] for p in predictions]

    t1 = time.time()
    if state["verbose"]:
        print(f"Prediction: {preds[0]}")
        print(f"Probability: ({probs[0] * 100:.2f}%) in {t1 - t0:.2f} seconds")
    return preds[0]


def generate(streamer, llm_tsfm, llm, prompt) -> str:
    t0 = time.time()
    num_tokens = 0
    thread = Thread(
        target=llm.generate,
        kwargs={
            **llm_tsfm(prompt),
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
    t1 = time.time()
    if state["verbose"]:
        print()
        print(f"Tok/sec: {num_tokens / (t1 - t0):.2f}")

    return generated_text


# -----------------------------------------------------------------------------


# Typer CLI
def run() -> None:
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading models...", total=None)
            cls_tsfm, amp_autocast, classifier = setup_classifier(CLASSIFIER, state["verbose"])
            streamer, llm_tsfm, llm = setup_llm(LLM)
    else:
        cls_tsfm, amp_autocast, classifier = setup_classifier(CLASSIFIER, state["verbose"])
        streamer, llm_tsfm, llm = setup_llm(LLM)
    classifier.eval()
    llm.eval()

    while True:
        img = capture_screenshot()
        pred = classify(classifier, cls_tsfm, amp_autocast, img)

        if pred == "distracted":
            data = api.list_events_parallel(per_page=per_page, n_pages=n_pages, org=org)
            dates = extract_dates(data)
            notification.title = generate(
                streamer, llm_tsfm, llm, TITLE_PROMPT.format(org=org, dates=",".join(map(str, dates)))
            )
            notification.message = generate(
                streamer, llm_tsfm, llm, MESSAGE_PROMPT.format(org=org, dates=",".join(map(str, dates)))
            )
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
        run()
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print(traceback.format_exc())
            print("\n\nExiting...")
