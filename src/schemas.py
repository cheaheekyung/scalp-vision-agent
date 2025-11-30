from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, List
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, computed_field, Field


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
    location: Literal["TH", "LH", "RH", "BH"] | None = None  # 정수리/좌/우/후두부 등

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
    title: str = Field(..., description="권장 사항 제목")
    description: str = Field(..., description="권장 사항 상세 설명")


class ScalpAnalysisResponse(BaseModel):
    """
    두피 분석 결과 응답 스키마.

    - risk_score: 0~3 정수 (UI에서 쓰기 좋은 형태)
    - risk_level: normal/low/medium/high 중 하나
    - summary: 한두 문장 요약
    - details: 상세 설명
    - recommendations: 구체적인 권장 사항 리스트
    - history_message: 직전 방문 대비 변화 요약 (없으면 None)
    - plan_text: 향후 1~3개월 관리 플랜 텍스트 (없으면 None)
    """

    risk_score: float = Field(..., ge=0.0, le=3.0)
    risk_level: Literal["normal", "low", "medium", "high"]

    summary: str
    details: str
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="권장 사항 리스트",
    )

    # 신규 필드 (선택적)
    history_message: str | None = Field(
        default=None,
        description="이전 방문 대비 변화에 대한 한 줄 요약",
    )
    plan_text: str | None = Field(
        default=None,
        description="향후 1~3개월 관리 플랜 텍스트",
    )


# -----------------------------
# 1) User (고객)
# -----------------------------
class UserBase(BaseModel):
    name: str
    gender: Optional[str] = None  # "male" / "female" / "unknown"
    birth_date: Optional[date] = None  # YYYY-MM-DD


class UserCreate(UserBase):
    """
    클라이언트 → 서버: 신규 고객 생성 요청에 사용.
    id, created_at은 서버가 채운다.
    """

    pass


class User(UserBase):
    """
    서버 → 클라이언트: 저장된 고객 정보 응답용.
    """

    user_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def age_group(self) -> Optional[str]:
        """
        응답에서만 쓰는 편의 필드.
        birth_date 기준으로 "20s", "30s", "40s", "70s+" 같은 문자열을 계산해서 내려준다.
        """
        if self.birth_date is None:
            return None

        today = date.today()
        age = (
            today.year
            - self.birth_date.year
            - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))
        )
        if age < 0:
            return None

        if age >= 70:
            return "70s+"

        decade = (age // 10) * 10
        return f"{decade}s"


class UserUpdate(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[date] = None


# -----------------------------
# 2) Visit (방문 세션)
# -----------------------------
class VisitBase(BaseModel):
    visit_date: date
    note: Optional[str] = None  # 상담 메모 등


class VisitCreate(VisitBase):
    """
    클라이언트 → 서버: 새 방문 세션 생성 요청에 사용.
    """

    user_id: int


class Visit(VisitBase):
    """
    서버 → 클라이언트: 저장된 방문 세션 정보 응답용.
    """

    visit_id: int
    user_id: int
    created_at: datetime


# -----------------------------
# 3) VisitReport (방문 리포트)
# -----------------------------
class VisitReportBase(BaseModel):
    visit_id: int
    risk_score: float = Field(..., ge=0.0, le=3.0)
    risk_level: str  # "low" / "medium" / "high" 등
    summary: str
    details: str
    history_message: str | None = None
    plan_text: str | None = None
    recommendations_json: str | None = None  # JSON string (list[RecommendationItem])
    report_text: str  # LLM/룰 기반 리포트 본문


class VisitReportCreate(VisitReportBase):
    """
    내부적으로 rule + LLM 결과를 저장할 때 사용.
    (API로 직접 받을 수도 있고, 서버 내부에서만 쓸 수도 있음)
    """

    pass


class VisitReport(VisitReportBase):
    report_id: int
    created_at: datetime


class FullVisitResponse(BaseModel):
    """
    한 번의 호출로 User / Visit / VisitReport 를 모두 내려주는 응답용 스키마.
    report 는 아직 생성되지 않았을 수 있으므로 Optional 처리.
    """

    user: User
    visit: Visit
    report: Optional[VisitReport] = None
