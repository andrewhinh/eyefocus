import warnings
from pathlib import Path, PurePosixPath

import modal

# Constants
PREFIX_PATH = Path(__file__).parent
ARTIFACT_PATH = PREFIX_PATH / "artifacts"
TRAIN_CONFIG_PATH = PREFIX_PATH / "config" / "train.yml"
TRAIN_SCRIPT_PATH = PREFIX_PATH / "train.py"
SWEEP_CONFIG_PATH = PREFIX_PATH / "config" / "sweep.yml"
EVAL_CONFIG_PATH = PREFIX_PATH / "config" / "eval.yml"
SERVE_CONFIG_PATH = PREFIX_PATH / "config" / "serve.yml"

# Modal
CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"
PYTHON_VERSION = "3.11"

PRETRAINED_VOLUME = "pretrained"
DATA_VOLUME = "data"
RUNS_VOLUME = "runs"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(PRETRAINED_VOLUME, create_if_missing=True),
    f"/{DATA_VOLUME}": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
    f"/{RUNS_VOLUME}": modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
}

CPU = 20  # cores (Modal max)

MINUTES = 60  # seconds
TIMEOUT = 24 * 60 * MINUTES

SERVE_TIMEOUT = 2 * MINUTES
SERVE_CONTAINER_IDLE_TIMEOUT = 5 * MINUTES
SERVE_ALLOW_CONCURRENT_INPUTS = 100

IMAGE = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        TAG, add_python=PYTHON_VERSION
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add Python dependencies
        "pillow==10.4.0",
        "torch==2.4.1",
        "accelerate==0.34.2",
        "datasets==3.0.0",
        "einops==0.8.0",
        "python-dotenv==1.0.1",
        "timm==1.0.9",
        "torchvision==0.19.1",
        "hf_transfer==0.1.8",
        "wandb==0.17.7",
        "ninja==1.11.1.1",
        "packaging==24.1",
        "wheel==0.44.0",
        "pydantic==2.8.2",
        "term-image==0.7.2",
    )
    .run_commands(  # add FlashAttention for faster inference using a shell command
        "pip install flash-attn==2.6.3 --no-build-isolation"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

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

warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
