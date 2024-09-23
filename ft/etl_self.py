import math
import os
import random

import mss
import numpy as np
import torch
from PIL import Image

from ft.utils import ARTIFACT_PATH, CLASSES, MAX_NEW_TOKENS, PROMPT, TEMPERATURE, download_model, transform_img

data_dir = ARTIFACT_PATH / "self"
num_imgs = 2000  # number of images to capture
val_split = 0.1  # validation split


def capture_screenshot():
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return np.array(sct_img)


def main():
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    # TODO: not supported by InternVL2
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     DEVICE = "mps"
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer, model = download_model(WORLD_SIZE, DEVICE, TORCH_DTYPE, True)

    train_idxs = random.sample(range(num_imgs), k=math.ceil(num_imgs * (1 - val_split)))

    print("Starting screenshot capture. Press Ctrl+C to stop.")
    print(f"Saving screenshots to: {data_dir}")

    idx = 0
    try:
        while True:
            current_screenshot = capture_screenshot()
            image = Image.fromarray(current_screenshot)
            pixel_vals = [transform_img(image).to(TORCH_DTYPE).to(DEVICE)]
            num_patches_list = [pixel_vals[0].size(0)]
            pixel_values = torch.cat(pixel_vals, dim=0)

            generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}
            questions = [f"<image>\n{PROMPT}"] * len(num_patches_list)
            with torch.no_grad():
                out = model.batch_chat(
                    tokenizer,
                    pixel_values,
                    num_patches_list=num_patches_list,
                    questions=questions,
                    generation_config=generation_config,
                )[0]

            try:
                out = int(out)
            except ValueError:
                out = -1
            pred = CLASSES[out] if out >= 0 else "error"
            print(f"Prediction: {pred}")

            label_dir = data_dir / "train" / pred if idx in train_idxs else data_dir / "validation" / pred
            os.makedirs(label_dir, exist_ok=True)
            img_path = label_dir / f"{idx}.jpg"
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path)

            idx += 1
    except KeyboardInterrupt:
        print("\nScreenshot capture stopped.")


if __name__ == "__main__":
    main()
