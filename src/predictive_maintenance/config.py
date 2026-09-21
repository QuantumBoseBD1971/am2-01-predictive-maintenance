"""Project configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"

TARGET_COLUMN = "machine_failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
