from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from src.config import PROCESSED_DIR, MASTER_INDEX_CSV, TRAIN_ROOT, VAL_ROOT, META_DIR

from src.io.meta_parser import user_profile_from_meta


def build_eda_table(limit: int | None = None) -> pd.DataFrame:
    """
    master_index.csv를 기반으로 EDA/ML용 테이블만들기

    각 행(row)은 하나의 샘플이고,
    컬럼에는 다음 정보들이 포함된다:

    - sample_id, split, location
    - gender, age
    - shampoo_frequency, perm_frequency, dye_frequency
    - value_1 ~ value_6 (두피 증상 등급)

    limit가 주어지면, 앞에서부터 최대 limit개까지만 로드한다.
    (테스트용으로 1000개 정도만 먼저 돌려볼 수 있음)
    """
    index_path = MASTER_INDEX_CSV
    df_index = pd.read_csv(index_path)

    records: list[dict[str, Any]] = []

    for i, row in df_index.iterrows():
        if limit is not None and i >= limit:
            break

        sample_id = row["sample_id"]
        split = row["split"]

        # 현재 머신의 TRAIN_ROOT / VAL_ROOT / META_DIR 기준으로 다시 조립한다.
        label_orig = Path(row["label_path"])
        meta_orig = Path(row["meta_path"])

        label_filename = label_orig.name  # 예: 0013_..._TH.json
        label_class_dir = label_orig.parent.name  # 예: [라벨]피지 과다_0.양호

        if split == "train":
            base_dir = TRAIN_ROOT  # data/raw/training
        else:
            base_dir = VAL_ROOT  # data/raw/validation

        label_path = base_dir / label_class_dir / label_filename

        meta_filename = meta_orig.name  # 예: 0013_..._TH_META.json
        meta_path = META_DIR / meta_filename  # data/raw/meta/...

        # JSON 로드
        with label_path.open(encoding="utf-8") as f:
            label = json.load(f)

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        # META → UserProfile
        profile = user_profile_from_meta(meta)

        # location은 META에서 가져오되, 없으면 index row에서 가져오기
        location = meta.get("location") or row.get("location") or None

        # 한 행(row)에 넣을 딕셔너리 구성
        rec: dict[str, Any] = {
            "sample_id": sample_id,
            "split": split,
            "location": location,
            # 프로필 정보
            "gender": profile.gender,
            "age": profile.age,
            "shampoo_frequency": profile.shampoo_frequency,
            "perm_frequency": profile.perm_frequency,
            "dye_frequency": profile.dye_frequency,
            # 라벨 정보 (증상 등급)
            "value_1": int(label["value_1"]),
            "value_2": int(label["value_2"]),
            "value_3": int(label["value_3"]),
            "value_4": int(label["value_4"]),
            "value_5": int(label["value_5"]),
            "value_6": int(label["value_6"]),
        }

        records.append(rec)

    df_eda = pd.DataFrame.from_records(records)

    # 결과를 CSV로도 저장
    out_path = PROCESSED_DIR / "eda_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_eda.to_csv(out_path, index=False)

    print(f"[build_eda_table] Saved EDA table to: {out_path} (rows={len(df_eda)})")

    return df_eda


if __name__ == "__main__":
    # 테스트할 때는 limit를 작게 (예: 1000)
    build_eda_table()
