from typing import Tuple

from src.schemas import (
    ScalpCondition,
    UserProfile,
    ScalpAnalysisRequest,
    ScalpAnalysisResponse,
    RecommendationItem,
)


def format_symptom_scores(c: ScalpCondition) -> str:
    """
    value_1~6 각각의 등급을 한국어 라벨과 함께 한 줄로 정리
    """
    labels = {
        "value_1": "각질",
        "value_2": "피지",
        "value_3": "모낭 사이 홍반",
        "value_4": "모낭 홍반/농포",
        "value_5": "비듬",
        "value_6": "탈모",
    }

    parts = [
        f"{labels['value_1']} {c.value_1}",
        f"{labels['value_2']} {c.value_2}",
        f"{labels['value_3']} {c.value_3}",
        f"{labels['value_4']} {c.value_4}",
        f"{labels['value_5']} {c.value_5}",
        f"{labels['value_6']} {c.value_6}",
    ]
    # 예: "각질 2 / 피지 0 / ... / 탈모 1"
    return " / ".join(parts)


def score_risk(condition: ScalpCondition) -> Tuple[int, str]:
    """
    ScalpCondition을 기반으로 0~3 정수 점수와
    'normal'/'low'/'medium'/'high' 텍스트 레벨을 계산.
    """
    # 아주 단순한 예시 점수: 탈모 + 홍반 + 비듬/각질
    hair_loss = condition.value_6
    inflammation = max(condition.value_3, condition.value_4)
    scaling_dandruff = max(condition.value_1, condition.value_5)

    raw = hair_loss * 2 + inflammation + scaling_dandruff

    # 0~3 사이로 압축
    if raw <= 1:
        return 0, "normal"
    elif raw <= 3:
        return 1, "low"
    elif raw <= 5:
        return 2, "medium"
    else:
        return 3, "high"


def simple_rule_based_analysis(
    condition: ScalpCondition,
    profile: UserProfile | None = None,
) -> Tuple[float, str]:
    """
    rule-based risk_score / risk_level 계산.

    - condition: value_1~6 점수 포함
    - profile: 지금은 사용하지 않지만, 나중에 연령/성별 가중치 등에 활용 가능

    반환:
        (risk_score, risk_level)
        risk_score: 0.0 ~ 3.0 사이 float
        risk_level: "normal" | "low" | "medium" | "high"
    """

    # 1) 기본 점수는 탈모(value_6)를 그대로 사용
    base_score = float(condition.value_6)

    # 2) 동반 증상에 따라 가중치 추가 (간단 버전)
    extra = 0.0

    # 홍반/염증이 심하면 약간 가중
    if condition.value_3 >= 2 or condition.value_4 >= 2:
        extra += 0.3

    # 각질/비듬이 심하면 추가 가중
    if condition.value_1 >= 2 or condition.value_5 >= 2:
        extra += 0.2

    # 피지가 과다하면 소폭 가중
    if condition.value_2 >= 2:
        extra += 0.1

    risk_score = min(3.0, base_score + extra)

    # 3) 구간에 따른 리스크 레벨 매핑
    if risk_score < 1.0:
        risk_level = "normal"
    elif risk_score < 2.0:
        risk_level = "low"
    elif risk_score < 2.5:
        risk_level = "medium"
    else:
        risk_level = "high"

    return risk_score, risk_level
