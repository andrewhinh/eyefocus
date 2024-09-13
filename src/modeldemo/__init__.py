from notifypy import Notify
import mss
import json
import time
from PIL import Image
import traceback
import torch
import transformers
from threading import Thread
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoProcessor
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
from typing_extensions import Annotated
from pathlib import Path


# Typer CLI
app = typer.Typer(
    rich_markup_mode="rich",
)
state = {"verbose": False, "super_verbose": False}


# Model config
MODEL_PATH = "microsoft/Phi-3.5-vision-instruct"
TORCH_DTYPE = torch.bfloat16
USER_PROMPT = "<|user|>\n"
ASSISTANT_PROMPT = "<|assistant|>\n"
PROMPT_SUFFIX = "<|end|>\n"

## BitsAndBytes quantization
quant_config = None
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True
try:
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_8bit=LOAD_IN_8BIT,
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=TORCH_DTYPE,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
except ImportError:
    pass


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0

NUM_CROPS = 16
MAX_NEW_TOKENS = 128
PROMPT = """
Task: Analyze the given computer screenshot to determine if it shows evidence of focused, productive activity or potentially distracting activity, then provide an appropriate titled response.

Instructions:
1. Examine the screenshot carefully.
2. Look for indicators of focused, productive activities such as:
   - Code editors or IDEs in use
   - Document editing software with substantial text visible
   - Spreadsheet applications with data or formulas
   - Research papers or educational materials being read
   - Professional design or modeling software in use
   - Terminal/command prompt windows with active commands
3. Identify potentially distracting activities, including:
   - Social media websites
   - Video streaming platforms
   - Unrelated news websites or apps
   - Online shopping sites
   - Music or video players
   - Messaging apps
   - Games or gaming platforms
4. Consider the context: a coding-related YouTube video might be considered focused activity for a programmer.

Response Format:
Return a single JSON object with the following fields:
- is_distracted (boolean): value (true if the screenshot primarily shows evidence of distraction, false if it shows focused activity)
- title (string): 1-liner snarky title to catch the user's attention (only if is_distracted is true, otherwise an empty string)
- message (string): 1-liner snarky message to encourage the user to refocus (only if is_distracted is true, otherwise an empty string)

Example responses:
{"is_distracted": false, "title": "", "message": ""}
{"is_distracted": true, "title": "Uh-oh!", "message": "Looks like someone's getting a little distracted..."}

Important:
- Only write a title and message if is_distracted is true.
- Provide only one JSON object as your complete response.
- Ensure the JSON is valid and properly formatted.
- Do not include any explanations or additional text outside the JSON object.
- Use true/false for the boolean value, not 1/0.
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


## Fns from: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
def download_model() -> tuple[AutoProcessor, TextIteratorStreamer, AutoModelForCausalLM]:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        num_crops=NUM_CROPS,
    )
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    torch.set_float32_matmul_precision("high")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=DEVICE,
        ).eval()
    except Exception:
        if state["verbose"]:
            print("Flash Attention not available, using eager attention.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=DEVICE,
        ).eval()
    model = torch.compile(model)

    return processor, streamer, model


def predict(img: Image, processor: AutoProcessor, streamer: TextIteratorStreamer, model: AutoModelForCausalLM) -> str:
    prompt = f"{USER_PROMPT}<|image_1|>\n{PROMPT}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    inputs = processor(prompt, img, return_tensors="pt").to(DEVICE)

    thread = Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "streamer": streamer,
        },
    )
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        if state["super_verbose"]:
            print(new_text, end="", flush=True)
    if state["super_verbose"]:
        print()

    return generated_text


# Typer CLI
def run() -> None:
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        print(f"Using device: {DEVICE}")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            processor, streamer, model = download_model()
        print("Model downloaded!")
    else:
        processor, streamer, model = download_model()

    while True:
        img = capture_screenshot()
        pred = predict(img, processor, streamer, model)

        try:
            format_pred = json.loads(pred)
            is_distracted, title, message = format_pred["is_distracted"], format_pred["title"], format_pred["message"]
            is_distracted = bool(is_distracted)
        except Exception as e:
            if state["verbose"]:
                print(f"Failed to parse prediction: {e}")
            continue

        if is_distracted:
            if state["verbose"]:
                print(f"[bold red] {title}\n{message} [/bold red]")
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
        if state["super_verbose"]:
            transformers.logging.set_verbosity_debug()
        elif state["verbose"]:
            transformers.logging.set_verbosity_warning()
        else:
            transformers.logging.set_verbosity_error()
        run()
    except KeyboardInterrupt:
        if state["verbose"]:
            print("\n\nExiting...")
    except Exception as e:
        if state["verbose"]:
            print(f"Failed with error: {e}")
            print(traceback.format_exc())
            print("\n\nExiting...")
