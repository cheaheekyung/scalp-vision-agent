from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ScalpSampleIndex(BaseModel):
    """이미지 ↔ 라벨 ↔ 메타 경로를 담는 인덱스 행 단위 스키마"""

    sample_id: str
    split: Literal["train", "val"]
    image_path: Path
    label_path: Path | None = None
    meta_path: Path | None = None


class ScalpCondition(BaseModel):
    """CNN의 결과값 '두피 상태 요약'."""
    sample_id: str
    location: Literal["TH", "LH", "RH", "BH"] | None = None   # 정수리/좌/우/후두부 등

    value_1: int  # 각질 0~3
    value_2: int  # 피지 0~3
    value_3: int  # 모낭 사이 홍반 0~3
    value_4: int  # 모낭 홍반/농포 0~3
    value_5: int  # 비듬 0~3
    value_6: int  # 탈모 0~3

class UserProfile(BaseModel):
    """메타데이터에서 뽑아낸 사용자 생활 습관/프로필"""
    gender: Literal["M", "F", "U"] | None = None
    age: int | None = None

    shampoo_frequency: str | None = None  # 예: "매일", "2~3일에 한 번"
    perm_frequency: str | None = None
    dye_frequency: str | None = None


class ScalpAnalysisRequest(BaseModel):
    """
    두피 리포트 생성에 필요한 입력 전체.
    - condition: CNN + 메타에서 온 두피 상태 요약
    - profile: 사용자의 기본 생활습관/프로필
    """
    condition: ScalpCondition
    profile: UserProfile


class RecommendationItem(BaseModel):
    title: str
    description: str


class ScalpAnalysisResponse(BaseModel):
    """
    LLM/에이전트가 만들어줄 최종 리포트 형태.
    - risk_score: 0~3 정수 점수
    - risk_level: "normal"/"low"/"medium"/"high" 텍스트 레벨
    """
    risk_score: int
    risk_level: Literal["normal", "low", "medium", "high"]
    summary: str                 # 전체 요약
    details: str                 # 증상별 상세 설명
    recommendations: list[RecommendationItem]