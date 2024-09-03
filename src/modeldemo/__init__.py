import os
import mss
import time
from PIL import Image
from notifypy import Notify

from pathlib import Path

import torch.nn.functional as F
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import torch
import timm
import typer
from typing_extensions import Annotated

app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False, "super_verbose": False}

MODEL_NAME = "convnextv2_huge.fcmae"
CLASSES = ["focused", "distracted"]

NOTIF_INT = 5  # seconds
NOTIF_LEN = 10  # seconds
NOTIF_TITLE = "You there..."
NOTIF_MSG = "Stay focused!"
NOTIF_ICON_PATH = str(Path(__file__).parent / "focus.png")
RETURN_MSG = "[bold green]Good![/bold green]"

NOTIF = Notify(
    default_notification_title=NOTIF_TITLE,
    default_application_name="Modeldemo",
    default_notification_message=NOTIF_MSG,
    default_notification_icon=NOTIF_ICON_PATH,
    enable_logging=state["super_verbose"],
)

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"


# Helper fns
def clear_terminal() -> None:
    if os.name == "nt":  # For Windows
        _ = os.system("cls")
    else:  # For macOS and Linux
        _ = os.system("clear")


def capture_screenshot() -> Image:
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def download_model() -> tuple[torch.nn.Module, timm.data.transforms_factory.transforms]:
    torch.set_float32_matmul_precision("high")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(CLASSES))
    model = model.to(DEVICE)
    model = torch.compile(model)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    if state["verbose"]:
        print("Model downloaded!")
    return model, transforms


def download_model_with_progress() -> tuple[torch.nn.Module, timm.data.transforms_factory.transforms]:
    if state["verbose"]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            return download_model()
    else:
        return download_model()


def capture_and_predict(
    model: torch.nn.Module, transforms: timm.data.transforms_factory.transforms
) -> tuple[float, str]:
    screen = capture_screenshot()
    img_tensor = transforms(screen).unsqueeze(0).to(DEVICE)
    output = model(img_tensor)
    probabilities = F.softmax(output[0], dim=0)
    top1_prob, top1_class = torch.topk(probabilities, 1)
    return top1_prob.item(), CLASSES[top1_class.item()]


# Typer CLI
def run(verbose: int) -> None:
    state["verbose"] = verbose > 0
    state["super_verbose"] = verbose > 1

    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")

    model, transforms = download_model_with_progress()

    if state["verbose"]:
        print(f"Using device: {DEVICE}")
        print("Starting...")

    notif_active = False
    last_check_time = 0

    while True:
        current_time = time.time()
        if current_time - last_check_time >= NOTIF_INT:
            prob, pred = capture_and_predict(model, transforms)

            if state["super_verbose"]:
                print(f"{pred} with probability {prob:.2f}")

            if pred != "focused":
                if not notif_active:
                    if state["verbose"]:
                        print(f"[bold red]{NOTIF_MSG}[/bold red]")
                    NOTIF.send(block=False)
                    notif_active = True
                    notif_start_time = current_time
            else:
                if notif_active:
                    if state["verbose"]:
                        print(RETURN_MSG)
                    notif_active = False

            last_check_time = current_time

        if notif_active and (current_time - notif_start_time >= NOTIF_LEN):
            notif_active = False

        time.sleep(1)  # Small delay to prevent excessive CPU usage


@app.command(
    help="Stay [bold red]focused.[/bold red]",
    epilog="Made by [bold blue]Andrew Hinh.[/bold blue] :mechanical_arm::person_climbing:",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": True},
)
def main(verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0) -> None:
    try:
        run(verbose)
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
        else:
            clear_terminal()
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print("\n\nExiting...")
        else:
            clear_terminal()
