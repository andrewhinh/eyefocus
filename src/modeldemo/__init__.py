from secrets import token_urlsafe
from notifypy import Notify
import time
import base64
import io
import json

from pathlib import Path

import torch
from PIL import Image
import mss
import traceback

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated

from .utils import (
    TITLE_PROMPT,
    MESSAGE_PROMPT,
    setup_classifier,
    class_config,
    setup_gguf,
    OCR_PROMPT,
    SYSTEM_PROMPT,
)

# -----------------------------------------------------------------------------

# Classifier
CLASSIFIER = "hf_hub:andrewhinh/resnet152-224-Screenspot"

# MM-LLM
MM_LLM = "abetlen/nanollava-gguf"
MM_LLM_CLIP = "nanollava-mmproj-f16.gguf"
MM_LLM_GGUF = "nanollava-text-model-f16.gguf"
MM_LLM_CTX = 4096  # img + text tokens
MM_LLM_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"screen_text": {"type": "string"}},
        "required": ["screen_text"],
    },
}

# LLM
LLM = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
LLM_GGUF = "llama-3.2-1b-instruct-q4_k_m.gguf"
LLM_CTX = 1024  # text tokens
LLM_TITLE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
}
LLM_MESSAGE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}

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


def image_to_base64_data_uri(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


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


def generate(llm, prompt, response_format, image=None) -> str:
    t0 = time.time()
    num_tokens = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    if image:
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image)}})

    generated_text = llm.create_chat_completion(
        messages=messages,
        response_format=response_format,
    )
    if state["verbose"]:
        print(f"Generated text: {generated_text}")

    t1 = time.time()
    if state["verbose"]:
        print()
        print(f"Tok/sec: {num_tokens / (t1 - t0):.2f}")

    return generated_text


# -----------------------------------------------------------------------------


# Typer CLI
def run() -> None:
    ## load models
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading models...", total=None)
            cls_tsfm, amp_autocast, classifier = setup_classifier(CLASSIFIER, state["verbose"])
            ocr = setup_gguf(MM_LLM, MM_LLM_GGUF, MM_LLM_CTX, MM_LLM_CLIP, state["verbose"])
            llm = setup_gguf(LLM, LLM_GGUF, LLM_CTX, verbose=state["verbose"])
    else:
        cls_tsfm, amp_autocast, classifier = setup_classifier(CLASSIFIER, state["verbose"])
        ocr = setup_gguf(MM_LLM, MM_LLM_GGUF, MM_LLM_CTX, MM_LLM_CLIP, state["verbose"])
        llm = setup_gguf(LLM, LLM_GGUF, LLM_CTX, verbose=state["verbose"])
    classifier.eval()

    ## main loop
    while True:
        img = capture_screenshot()
        pred = classify(classifier, cls_tsfm, amp_autocast, img)

        if pred == "distracted":
            ocr_out = generate(ocr, OCR_PROMPT, MM_LLM_FORMAT, img)
            # free memory
            ocr.close()
            ocr.chat_handler._exit_stack.close()  # https://github.com/abetlen/llama-cpp-python/issues/1746
            try:
                ocr_out = json.loads(ocr_out)
                screen_text = ocr_out["screen_text"]
            except KeyError:
                if state["verbose"]:
                    print("No screen text detected.")
                continue

            title_out = generate(llm, TITLE_PROMPT.format(description=screen_text), LLM_TITLE_FORMAT)
            message_out = generate(llm, MESSAGE_PROMPT.format(description=screen_text), LLM_MESSAGE_FORMAT)
            llm.close()
            try:
                title_out = json.loads(title_out)
                message_out = json.loads(message_out)
                title = title_out["title"]
                message = message_out["message"]
            except KeyError:
                if state["verbose"]:
                    print("No title or message generated.")
                continue

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
        run()
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print(traceback.format_exc())
            print("\n\nExiting...")
