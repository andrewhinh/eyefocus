from notifypy import Notify
import mss
import json
import time
from PIL import Image
import traceback
import torch
from fastchat.conversation import get_conv_template
import transformers
from threading import Thread
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer, PreTrainedModel
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
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
MODEL_PATH = "OpenGVLab/InternVL2-2B"
TORCH_DTYPE = torch.bfloat16

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
# Not supported by InternVL2
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     DEVICE = "mps"
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0

INPUT_IMG_SIZE = 448
MAX_TILES = 12
MAX_NEW_TOKENS = 128
DO_SAMPLE = True
PROMPT = """
<image>
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
4. If multiple windows/tabs are visible, prioritize the active or most prominent one.
5. Consider the context: a coding-related YouTube video might be considered focused activity for a programmer.

Response Format:
Return a single JSON object with the following fields:
- is_distracted: boolean value (true if the screenshot primarily shows evidence of distraction, false if it shows focused activity)
- title: string 1-liner snarky title to catch the user's attention (only if is_distracted is true, otherwise an empty string)
- message: string 1-liner snarky message to encourage the user to refocus (only if is_distracted is true, otherwise an empty string)

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


## Fns from: https://huggingface.co/OpenGVLab/InternVL2
class ExtraCompatibleInternVLChatModel(
    PreTrainedModel
):  # ADDED: needed because original model.chat() contains explicit .cuda() calls
    def __init__(self, model):
        super().__init__(model.config)
        self.model = model

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",  # noqa: S107
        IMG_END_TOKEN="</img>",  # noqa: S107
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",  # noqa: S107
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id  # ADDED: for self.model.generate

        conv = get_conv_template("internlm-chat")  # ADDED: to resolve KeyError: 'internlm2-chat'
        conv.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(conv.sep)

        history = [] if history is None else history
        for old_question, old_answer in history:
            conv.append_message(conv.roles[0], old_question)
            conv.append_message(conv.roles[1], old_answer)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(DEVICE)  # ADDED
        attention_mask = model_inputs["attention_mask"].to(DEVICE)  # ADDED
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.model.generate(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(conv.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            if verbose:
                print(query_to_print, response)
            return response


def download_model() -> tuple[TextIteratorStreamer, AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)

    torch.set_float32_matmul_precision("high")

    def split_model(model_name: str):
        device_map = {}
        num_layers = {
            "InternVL2-1B": 24,
            "InternVL2-2B": 24,
            "InternVL2-4B": 32,
            "InternVL2-8B": 32,
            "InternVL2-26B": 48,
            "InternVL2-40B": 60,
            "InternVL2-Llama3-76B": 80,
        }[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (WORLD_SIZE - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * WORLD_SIZE
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

    # ADDED: only use device_map if GPU is available
    device_map = None
    if WORLD_SIZE > 0:
        device_map = split_model(MODEL_PATH.split("/")[-1])

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        # use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    # ADDED: torch.compile
    model = torch.compile(model)

    # ADDED: for non-gpu compatibility
    if WORLD_SIZE == 0:
        model = model.to(DEVICE)
        model = ExtraCompatibleInternVLChatModel(model)

    return streamer, tokenizer, model


def transform_img(image: Image) -> torch.Tensor:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform() -> T.Compose:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        aspect_ratio: float, target_ratios: set[tuple[int, int]], width: int, height: int
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * INPUT_IMG_SIZE * INPUT_IMG_SIZE * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image: Image, min_num: int = 1, use_thumbnail: bool = False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, MAX_TILES + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= MAX_TILES and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height)

        # calculate the target width and height
        target_width = INPUT_IMG_SIZE * target_aspect_ratio[0]
        target_height = INPUT_IMG_SIZE * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // INPUT_IMG_SIZE)) * INPUT_IMG_SIZE,
                (i // (target_width // INPUT_IMG_SIZE)) * INPUT_IMG_SIZE,
                ((i % (target_width // INPUT_IMG_SIZE)) + 1) * INPUT_IMG_SIZE,
                ((i // (target_width // INPUT_IMG_SIZE)) + 1) * INPUT_IMG_SIZE,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            processed_images.append(thumbnail_img)
        return processed_images

    transform = build_transform()
    images = dynamic_preprocess(image, use_thumbnail=True)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
    if state["verbose"]:
        print("Press Ctrl+C to stop at any time.")
        print(f"Using device: {DEVICE}")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            progress.add_task("Downloading model...", total=None)
            streamer, tokenizer, model = download_model()
        print("Model downloaded!")
    else:
        streamer, tokenizer, model = download_model()

    while True:
        img = capture_screenshot()
        pred = predict(img, streamer, tokenizer, model)

        try:
            format_pred = pred.replace("```json", "").replace("```", "").strip()
            format_pred = json.loads(format_pred)
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
            transformers.logging.set_verbosity_info()
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
