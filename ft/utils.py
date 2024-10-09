from pathlib import Path

# ETL
CLASSES = ["focused", "distracted"]


# FT filepaths
PREFIX_PATH = Path(__file__).parent
ARTIFACT_PATH = PREFIX_PATH / "artifacts"
TRAIN_SCRIPT_PATH = PREFIX_PATH / "train.py"
