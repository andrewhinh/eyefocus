import os
import time

import mss
import numpy as np
from PIL import Image
from tqdm import tqdm

data_dir = "self"
SECONDS = 60
MINUTES = 60
num_imgs = 2 * MINUTES * SECONDS  # number of screenshots to capture
file_format = "jpg"  # file format of the images
interval = 12  #  interval between each screenshot in seconds


def capture_screenshot():
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        return np.array(sct_img)


def main():
    print("Starting screenshot capture. Press Ctrl+C to stop.")
    print(f"Saving screenshots to: {data_dir}")

    os.makedirs(data_dir, exist_ok=True)
    existing_files = [int(f.split(".")[0]) for f in os.listdir(data_dir) if f.endswith(f".{file_format}")]
    idx = max(existing_files) + 1 if existing_files else 0

    for _ in tqdm(range(num_imgs), desc="Capturing Screenshots"):
        try:
            current_screenshot = capture_screenshot()
        except Exception as e:
            print(f"Failed to capture screenshot: {e}")
            continue
        image = Image.fromarray(current_screenshot)
        img_path = os.path.join(data_dir, f"{idx}.{file_format}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path)
        idx += 1
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScreenshot capture stopped.")
