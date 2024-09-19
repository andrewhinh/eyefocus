from notifypy import Notify
import mss
import json
import time
import os
from huggingface_hub import hf_hub_download

from PIL import Image
import io
import traceback
import torch
import base64
from rich import print
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False, "super_verbose": False}


# Model config
MODEL_PATH = "openbmb/MiniCPM-V-2_6-gguf"
MM_PROJ_FILEPATH = "mmproj-model-f16.gguf"
GGML_FILEPATH = "ggml-model-Q2_K.gguf"
MAX_LEN = 4096

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"

PROMPT = """
Task: Analyze the given computer screenshot to determine if it shows evidence of focused, productive activity or potentially distracting activity, then provide an appropriate titled response.

Instructions:
1. Examine the screenshot carefully.
2. Look for indicators of focused, productive activities including but not limited to:
   - Code editors or IDEs in use
   - Document editing software with substantial text visible
   - Spreadsheet applications with data or formulas
   - Research papers or educational materials being read
   - Professional design or modeling software in use
   - Terminal/command prompt windows with active commands
3. Identify potentially distracting activities including but not limited to:
   - Social media websites
   - Video streaming platforms
   - Unrelated news websites or apps
   - Online shopping sites
   - Music or video players
   - Messaging apps
   - Games or gaming platforms
4. Consider the context: e.g. a coding-related YouTube video might be considered focused activity for a programmer.

Response Format:
Return a single JSON object with the following fields:
- is_distracted (boolean): value (true if the screenshot primarily shows evidence of distraction, false if it shows focused activity)
- title (string): 1-liner snarky title to catch the user's attention (only if is_distracted is true, otherwise an empty string)
- message (string): 1-liner snarky message to encourage the user to refocus (only if is_distracted is true, otherwise an empty string)

Example responses:
{"is_distracted": false, "title": "", "message": ""}
{"is_distracted": true, "title": "Uh-oh!", "message": "Looks like someone's getting a little distracted..."}
"""

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


def pil_to_base64(img: Image) -> str:
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    base64_data = base64.b64encode(img_byte_arr).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


## Reference: https://llama-cpp-python.readthedocs.io/en/stable/#multi-modal-models
def download_model() -> Llama:
    local_mm_proj_path = hf_hub_download(MODEL_PATH, MM_PROJ_FILEPATH)
    local_ggml_path = hf_hub_download(MODEL_PATH, GGML_FILEPATH)

    chat_handler = MiniCPMv26ChatHandler(clip_model_path=local_mm_proj_path)
    llm = Llama(
        model_path=local_ggml_path,
        n_gpu_layers=-1 if DEVICE == "cuda" else 0,
        chat_handler=chat_handler,
        n_ctx=MAX_LEN,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10 if DEVICE == "cuda" else 2),
        verbose=state["super_verbose"],
    )

    return llm


def predict(img: Image, llm: Llama) -> str:
    img_base64 = pil_to_base64(img)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": img_base64}}],
            }
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "is_distracted": {"type": "boolean"},
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["is_distracted"],
            },
        },
    )

    return response["choices"][0]["message"]["content"]


# Typer CLI
def run() -> None:
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        print(f"Using device: {DEVICE}")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            llm = download_model()
        print("Model downloaded!")
    else:
        llm = download_model()

    while True:
        img = capture_screenshot()
        pred = predict(img, llm)
        if state["verbose"]:
            print(pred, end="", flush=True)
            print()
        pred_json = json.loads(pred)
        is_distracted, title, message = pred_json["is_distracted"], pred_json["title"], pred_json["message"]
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
