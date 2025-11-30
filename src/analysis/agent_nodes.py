from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.schemas import (
    ScalpCondition,
    UserProfile,
    ScalpAnalysisResponse,
)
from src.analysis.report_rules import simple_rule_based_analysis
from src.analysis.llm_agent import (
    generate_llm_scalp_report,
    generate_dummy_llm_report,
)

from datetime import datetime
from typing import List

from src.db import models as db_models
from sqlalchemy.orm import Session


@dataclass
class RuleRiskOutput:
    """
    Rule 기반 위험도 노드 출력.
    """

    risk_score: float  # 0.0 ~ 3.0
    risk_level: str  # "normal" / "low" / "medium" / "high"


@dataclass
class LLMReportOutput:
    """
    LLM 리포트 노드 출력.
    """

    analysis: ScalpAnalysisResponse
    report_text: str
    llm_ok: bool
    llm_error: Optional[str]


def run_rule_risk(
    condition: ScalpCondition,
    profile: Optional[UserProfile] = None,
) -> RuleRiskOutput:
    """
    Rule 기반 위험도 계산 노드.

    - float risk_score (0.0~3.0)
    - 문자열 risk_level
    """
    risk_score, risk_level = simple_rule_based_analysis(
        condition=condition,
        profile=profile,
    )
    return RuleRiskOutput(
        risk_score=risk_score,
        risk_level=risk_level,
    )


def run_llm_report(
    condition: ScalpCondition,
    profile: Optional[UserProfile],
    risk_score: float,
    risk_level: str,
) -> LLMReportOutput:
    """
    LLM 리포트 생성 노드.

    - 성공 시: generate_llm_scalp_report 결과 반환
    - 실패 시: rule 기반 fallback ScalpAnalysisResponse + dummy 리포트 생성
    """
    llm_ok = True
    llm_error: Optional[str] = None

    try:
        analysis, report_text = generate_llm_scalp_report(
            condition=condition,
            profile=profile,
            risk_score=risk_score,
            risk_level=risk_level,
        )
        return LLMReportOutput(
            analysis=analysis,
            report_text=report_text,
            llm_ok=llm_ok,
            llm_error=llm_error,
        )
    except Exception as e:  # noqa: BLE001
        llm_ok = False
        llm_error = str(e)

        # risk_score_int = int(round(risk_score))
        fallback_analysis = ScalpAnalysisResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            summary="현재는 LLM 리포트 생성에 오류가 발생하여, 기본 rule 기반 요약만 제공합니다.",
            details="두피 상태는 rule 기반 점수만으로 평가되었습니다. 추후 시스템 안정화 후 다시 분석을 권장드립니다.",
            recommendations=[],
        )
        report_text = generate_dummy_llm_report(
            fallback_analysis,
            language="ko",
        )

        return LLMReportOutput(
            analysis=fallback_analysis,
            report_text=report_text,
            llm_ok=llm_ok,
            llm_error=llm_error,
        )


@dataclass
class HistoryCompareOutput:
    """
    이전 방문 기록과 비교한 결과.

    - prev_visit_id: 직전 방문 ID (없으면 None)
    - message: 요약 설명 (예: "지난 방문보다 탈모 위험도가 상승했습니다.")
    """

    prev_visit_id: int | None
    message: str


@dataclass
class PlanRecommendationOutput:
    """
    관리 플랜 추천 결과.

    - plan_text: 1~3개월 관리 플랜 텍스트
    """

    plan_text: str


def run_history_compare(
    db: Session,
    user_id: int,
    current_visit_id: int,
    current_analysis: ScalpAnalysisResponse,
) -> HistoryCompareOutput:
    """
    HistoryCompare 노드.

    - 같은 user_id에 대한 직전 Visit/VisitReport를 찾고,
    - 현재 분석 결과와 비교한 한 줄 요약을 만든다.
    - 지금은 rule 기반 / 간단 비교로 두고,
      나중에 LLM을 붙여도 됨.
    """
    # 1) 현재 visit 기준, 직전 방문 찾기
    prev_visit: db_models.Visit | None = (
        db.query(db_models.Visit)
        .filter(
            db_models.Visit.user_id == user_id,
            db_models.Visit.visit_id < current_visit_id,
        )
        .order_by(db_models.Visit.visit_id.desc())
        .first()
    )

    if prev_visit is None:
        return HistoryCompareOutput(
            prev_visit_id=None,
            message="이전 방문 기록이 없어, 이번 분석이 첫 방문 기준 리포트입니다.",
        )

    prev_report: db_models.VisitReport | None = (
        db.query(db_models.VisitReport)
        .filter(db_models.VisitReport.visit_id == prev_visit.visit_id)
        .one_or_none()
    )

    if prev_report is None:
        return HistoryCompareOutput(
            prev_visit_id=prev_visit.visit_id,
            message="직전 방문의 리포트 데이터가 없어, 변화 추이를 계산할 수 없습니다.",
        )

    # 2) 아주 간단한 rule 기반 비교 (나중에 LLM으로 교체 가능)
    prev_risk = prev_report.risk_score
    curr_risk = float(current_analysis.risk_score)

    if curr_risk > prev_risk:
        trend_msg = "지난 방문보다 탈모/두피 위험도가 상승했습니다."
    elif curr_risk < prev_risk:
        trend_msg = "지난 방문보다 탈모/두피 위험도가 다소 완화되었습니다."
    else:
        trend_msg = "지난 방문과 유사한 수준의 위험도를 보입니다."

    msg = f"직전 방문({prev_visit.visit_id}) 대비 변화: {trend_msg}"

    return HistoryCompareOutput(
        prev_visit_id=prev_visit.visit_id,
        message=msg,
    )


def run_plan_recommendation(
    analysis: ScalpAnalysisResponse,
    profile: UserProfile | None,
) -> PlanRecommendationOutput:
    """
    PlanGenerator 노드 (1~3개월 관리 플랜 텍스트).

    지금은 rule 기반/템플릿 기반으로 간단하게 작성해 두고,
    나중에 LLM 프롬프트를 붙여 고도화할 수 있다.
    """
    # TODO: 나중에 generate_llm_scalp_report 스타일의
    #       별도 LLM 프롬프트로 교체 가능

    base_plan = "향후 1~3개월 동안 다음과 같은 관리를 권장드립니다.\n"

    if analysis.risk_level in ("high", "medium"):
        plan = (
            base_plan + "- 주 2~3회 이상 두피 전용 샴푸 사용\n"
            "- 열기구/펌/염색 등 자극적인 시술은 최대한 줄이기\n"
            "- 필요 시 전문 의료진과의 상담을 통해 약물/시술 병행 검토\n"
        )
    else:
        plan = (
            base_plan + "- 현재 수준을 유지할 수 있도록 규칙적인 두피 세정\n"
            "- 과도한 열기구/시술은 피하고, 스트레스 관리에 신경쓰기\n"
        )

    # profile이 있으면 간단한 커스터마이징 여지
    if profile is not None and profile.age is not None:
        if profile.age >= 40:
            plan += "- 40대 이상에서는 정기적인 두피 상태 점검을 권장드립니다.\n"

    return PlanRecommendationOutput(plan_text=plan)
