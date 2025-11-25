from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

META_DIR = RAW_DIR / "meta"
TRAIN_ROOT = RAW_DIR / "training"
VAL_ROOT = RAW_DIR / "validation"

MASTER_INDEX_CSV = PROCESSED_DIR / "master_index.csv"
