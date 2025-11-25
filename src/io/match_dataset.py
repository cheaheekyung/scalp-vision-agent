from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import TRAIN_ROOT, VAL_ROOT, META_DIR, MASTER_INDEX_CSV
from src.schemas import ScalpSampleIndex


def _iter_dirs_with_keyword(roots: Iterable[Path], keyword: str) -> list[Path]:
    """
    training/, validation/ 중 
    폴더 이름에 keyword("[원천]", "[라벨]" 등)가 포함된 폴더만 모음
    """
    results: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and keyword in p.name:
                results.append(p)
    return results


def scan_images() -> dict[str, tuple[Path, str]]:
    """
    training/validation 아래 [원천] 폴더 내 이미지 경로 수집

    반환: { sample_id: (이미지 경로, split) }
    """
    mapping: dict[str, tuple[Path, str]] = {}

    roots = [TRAIN_ROOT, VAL_ROOT]
    split_map = {TRAIN_ROOT: "train", VAL_ROOT: "val"}

    for root in roots:
        split = split_map[root]
        source_dirs = _iter_dirs_with_keyword([root], "[원천]")
        for d in source_dirs:
            for img_path in d.rglob("*.jpg"):
                mapping[img_path.stem] = (img_path, split)

    return mapping


def scan_labels() -> dict[str, Path]:
    """training/validation 아래 [라벨] 폴더의 JSON 라벨 수집"""
    mapping: dict[str, Path] = {}
    duplicate_count = 0
    
    roots = [TRAIN_ROOT, VAL_ROOT]
    label_dirs = _iter_dirs_with_keyword(roots, "[라벨]")
    
    for d in label_dirs:
        for lbl_path in d.rglob("*.json"):
            stem = lbl_path.stem
            if stem in mapping:
                duplicate_count += 1
            mapping[lbl_path.stem] = lbl_path
    print(f"라벨 중복 파일 수 : {duplicate_count}")
    return mapping


def scan_meta() -> dict[str, Path]:
    """
    raw/meta 아래 메타데이터 JSON 수집

    파일명 규칙:
      이미지: 0013_A2LEBJJDE00060O_1602578303771_3_TH.jpg
      메타  : 0013_A2LEBJJDE00060O_1602578303771_3_TH_META.json

    => stem에서 "_META"를 떼고 이미지 stem과 매칭.
    """
    mapping: dict[str, Path] = {}
    if META_DIR.exists():
        for meta_path in META_DIR.rglob("*.json"):
            stem = meta_path.stem  # 예: "..._TH_META"
            base_stem = stem.removesuffix("_META")  # 예: "..._TH"

            mapping[base_stem] = meta_path

    print(f"메타 JSON 수집 개수: {len(mapping)}")
    return mapping



def build_master_index(output_csv: Path = MASTER_INDEX_CSV) -> pd.DataFrame:
    image_map = scan_images()
    label_map = scan_labels()
    meta_map = scan_meta()

    records: list[ScalpSampleIndex] = []

    for stem, (img_path, split) in image_map.items():
        lbl_path = label_map.get(stem)
        mta_path = meta_map.get(stem)

        record = ScalpSampleIndex(
            sample_id=stem,
            split=split,
            image_path=img_path,
            label_path=lbl_path,
            meta_path=mta_path,
        )
        records.append(record)

    # Pydantic 모델 리스트 → DataFrame
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)

    # Path 컬럼을 문자열로 변환 (CSV 저장용)
    for col in ["image_path", "label_path", "meta_path"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    total = len(df)
    with_label = df["label_path"].notna().sum()
    with_meta = df["meta_path"].notna().sum()

    print("===== 매칭 결과 =====")
    print(f"총 이미지 수         : {total}")
    print(f"라벨(JSON) 매칭 개수 : {with_label} / {total}")
    print(f"메타(JSON) 매칭 개수 : {with_meta} / {total}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nmaster_index 저장 완료 → {output_csv}")

    return df


if __name__ == "__main__":
    build_master_index()
