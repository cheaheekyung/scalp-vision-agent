# src/analysis/report_rules.py

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


def simple_rule_based_analysis(req: ScalpAnalysisRequest) -> ScalpAnalysisResponse:
    """
    LLM 없이, 간단한 규칙 기반으로 리포트를 생성하는 함수.
    나중에 LLM/에이전트가 이 자리를 대체하거나 감쌀 예정.
    """
    c = req.condition
    p = req.profile

    risk_score, risk_level = score_risk(c)

    # 요약 문장
    summary = f"현재 두피 전반의 위험도는 '{risk_level}' 수준으로 판단됩니다."

    # 상세 설명 (아주 단순한 버전)
    detail_parts: list[str] = []

    symptom_line = format_symptom_scores(c)
    detail_parts.append(f"증상별 등급: {symptom_line}.")

    if c.value_6 >= 2:
        detail_parts.append("탈모 징후가 뚜렷하게 관찰됩니다.")
    elif c.value_6 == 1:
        detail_parts.append("탈모 초기 단계 가능성이 있어 관찰이 필요합니다.")
    else:
        detail_parts.append("현재 탈모 위험은 크지 않은 편입니다.")

    if max(c.value_3, c.value_4) >= 2:
        detail_parts.append(
            "모낭 주변 홍반/염증 소견이 있어 두피 자극을 줄이는 관리가 필요합니다."
        )

    if max(c.value_1, c.value_5) >= 2:
        detail_parts.append("각질/비듬이 많아 두피 세정과 보습 관리가 중요합니다.")

    # 프로필 기반 한두 문장 추가 (있으면)
    if p.shampoo_frequency:
        detail_parts.append(
            f"현재 샴푸 빈도는 '{p.shampoo_frequency}'로 기입되어 있습니다."
        )

    details = " ".join(detail_parts)

    # 아주 기본적인 추천 항목 1~2개
    recommendations: list[RecommendationItem] = []

    if risk_level in {"medium", "high"}:
        recommendations.append(
            RecommendationItem(
                title="두피 전문 클리닉 또는 병원 상담 권장",
                description="지속적인 탈모/염증 소견이 있는 만큼, 전문의 상담을 통해 정확한 원인을 확인해보는 것을 권장드립니다.",
            )
        )

    recommendations.append(
        RecommendationItem(
            title="두피 자극 줄이기",
            description="과도한 펌/염색, 뜨거운 바람, 잦은 스타일링 제품 사용을 줄이고, 두피에 자극이 적은 샴푸를 사용하는 것이 좋습니다.",
        )
    )

    return ScalpAnalysisResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        summary=summary,
        details=details,
        recommendations=recommendations,
    )
