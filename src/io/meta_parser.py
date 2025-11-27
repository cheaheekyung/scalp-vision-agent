from __future__ import annotations

from typing import Any, Mapping

from src.schemas import UserProfile

from src.schemas import ScalpCondition, ScalpAnalysisRequest


def _normalize_gender(raw: Any) -> str:
    """
    AI Hub META의 gender 값을 'M' / 'F' / 'U'로 정규화.
    - '남', 'M', 'm', '1' → 'M'
    - '여', 'F', 'f', '2' → 'F'
    - 그 외 / None        → 'U'
    """
    if raw is None:
        return "U"

    s = str(raw).strip()

    if s in {"남", "M", "m", "1"}:
        return "M"
    if s in {"여", "F", "f", "2"}:
        return "F"

    return "U"


def _parse_age_band(raw: Any) -> int | None:
    """
    '20대', '30대 초반', '40' 같은 문자열에서
    첫 번째 숫자 부분만 뽑아서 정수로 변환.
    - '20대'  → 20
    - '30대초반' → 30
    - '50'   → 50
    실패하면 None.
    """
    if raw is None:
        return None

    s = str(raw)

    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def user_profile_from_meta(meta: Mapping[str, Any]) -> UserProfile:
    """
    AI Hub 두피 META JSON(dict)을 받아서 UserProfile로 변환.

    사용 규칙 (AI Hub 스키마 기준):
    - gender      : '남' / '여' 등 → M / F / U 로 정규화
    - age         : '20대', '30대' 등 → 20, 30 같은 정수로 변환
    - answers1    : 샴푸 사용 빈도        → shampoo_frequency
    - answers2    : 펌 주기              → perm_frequency
    - answers3    : 염색 주기(자가 염색 포함) → dye_frequency
    """

    gender = _normalize_gender(meta.get("gender"))
    age = _parse_age_band(meta.get("age"))

    shampoo_frequency = meta.get("answers1")
    perm_frequency = meta.get("answers2")
    dye_frequency = meta.get("answers3")

    return UserProfile(
        gender=gender,
        age=age,
        shampoo_frequency=shampoo_frequency,
        perm_frequency=perm_frequency,
        dye_frequency=dye_frequency,
    )


def build_analysis_request_from_meta(
    condition: ScalpCondition,
    meta: Mapping[str, Any],
) -> ScalpAnalysisRequest:
    """
    CNN이 반환한 ScalpCondition과 META JSON(dict)을 받아서
    ScalpAnalysisRequest를 만들어준다.

    - UserProfile은 user_profile_from_meta()로 생성
    - LLM/리포트/에이전트는 이 Request 하나만 받으면 동작하도록 맞춰두는 게 목표
    """
    profile = user_profile_from_meta(meta)

    return ScalpAnalysisRequest(
        condition=condition,
        profile=profile,
    )


def build_analysis_request_from_meta(
    condition: ScalpCondition,
    meta: Mapping[str, Any],
) -> ScalpAnalysisRequest:
    """
    CNN이 만든 ScalpCondition과 AI Hub META JSON(dict)을 받아
    ScalpAnalysisRequest를 생성한다.
    """
    profile = user_profile_from_meta(meta)
    return ScalpAnalysisRequest(condition=condition, profile=profile)
