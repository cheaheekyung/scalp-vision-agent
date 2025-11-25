from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import json

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import MASTER_INDEX_CSV
from src.schemas import ScalpSampleIndex


LabelDict = dict[str, int]


def load_label_json(path: Path) -> LabelDict:
    """
    라벨 JSON에서 value_1 ~ value_6 을 int로 뽑아서 dict로 반환.

    예시 JSON:
    {
        "value_1": "1",
        "value_2": "2",
        "value_3": "3",
        "value_4": "0",
        "value_5": "0",
        "value_6": "0"
    }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels: LabelDict = {}
    for i in range(1, 7):
        key = f"value_{i}"
        raw = data.get(key, "0")
        try:
            labels[key] = int(raw)
        except (TypeError, ValueError):
            labels[key] = 0  # 이상하면 일단 0으로
    return labels


def load_meta_json(path: Path) -> dict:
    """
    메타 JSON은 일단 raw dict로 그대로 반환.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class ScalpDataset(Dataset):
    """
    master_index.csv 기반 두피 이미지 Dataset.

    __getitem__ 반환 형식 예시:
    {
        "image": Tensor 또는 PIL.Image (transforms 여부에 따라),
        "labels": {
            "value_1": 1,
            "value_2": 2,
            ...
        },
        "meta": dict 또는 None,
        "sample_id": "0013_A2LEBJJDE00060O_1602578303771_3_TH",
        "split": "train" 또는 "val"
    }
    """

    def __init__(
        self,
        index_csv: Path | None = None,
        split: Optional[str] = None,  # "train", "val", None(전부)
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if index_csv is None:
            index_csv = MASTER_INDEX_CSV

        self.index_path = Path(index_csv)
        df = pd.read_csv(self.index_path)

        if split is not None:
            df = df[df["split"] == split]

        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def _get_row_index(self, idx: int) -> ScalpSampleIndex:
        row = self.df.iloc[idx]

        return ScalpSampleIndex(
            sample_id=row["sample_id"],
            split=row["split"],
            image_path=Path(row["image_path"]),
            label_path=Path(row["label_path"]) if pd.notna(row["label_path"]) else None,
            meta_path=Path(row["meta_path"]) if pd.notna(row["meta_path"]) else None,
        )

    def __getitem__(self, idx: int):
        sample = self._get_row_index(idx)

        # 1) 이미지 로드
        image = Image.open(sample.image_path).convert("RGB")

        # 2) 라벨 로드
        labels: LabelDict | None = None
        if sample.label_path is not None:
            labels = load_label_json(sample.label_path)

        # 3) 메타 로드
        meta: dict | None = None
        if sample.meta_path is not None and sample.meta_path.exists():
            meta = load_meta_json(sample.meta_path)

        # 4) transforms 적용 
        if self.transforms is not None:
            # 보통 torchvision.transforms.Compose 를 기대
            image = self.transforms(image)

        return {
            "image": image,
            "labels": labels,
            "meta": meta,
            "sample_id": sample.sample_id,
            "split": sample.split,
        }
