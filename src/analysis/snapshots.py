# src/analysis/snapshots.py
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.schemas import (
    ScalpCondition,
    UserProfile,
    ScalpAnalysisResponse,
)

# data/snapshots 디렉터리 자동 생성
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis_snapshot(
    *,
    visit_id: int,
    condition: ScalpCondition,
    profile: UserProfile | None,
    risk_score: float,
    risk_level: str,
    llm_ok: bool,
    llm_error: str | None,
    analysis: ScalpAnalysisResponse,
    report_text: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    분석 요청 1건에 대한 스냅샷을 JSON으로 저장한다.

    - 디렉터리: data/snapshots/
    - 파일명: visit_{visit_id}_{UTC타임스탬프}.json
    """
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    payload: dict[str, Any] = {
        "visit_id": visit_id,
        "created_at_utc": dt.datetime.utcnow().isoformat(),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "llm_ok": llm_ok,
        "llm_error": llm_error,
        "condition": condition.model_dump(),
        "profile": profile.model_dump() if profile is not None else None,
        "analysis": analysis.model_dump(),  # ScalpAnalysisResponse
        "report_text": report_text,
    }

    if extra:
        payload["extra"] = extra

    path = SNAPSHOT_DIR / f"visit_{visit_id}_{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
