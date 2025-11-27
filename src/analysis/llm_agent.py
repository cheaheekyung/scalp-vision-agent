from __future__ import annotations

from typing import Optional, Literal
from textwrap import dedent

from src.schemas import (
    ScalpCondition,
    UserProfile,
    RecommendationItem,
    ScalpAnalysisResponse,
    ScalpAnalysisRequest,
)


def build_report_prompt(
    analysis: ScalpAnalysisResponse,
    language: Literal["ko", "en"] = "ko",
) -> str:
    """
    ScalpAnalysisResponse(risk_score, risk_level, summary, details, recommendations)를
    기반으로 LLM에 전달할 프롬프트 텍스트를 생성한다.

    ⚠️ NOTE:
    - 더 이상 value_1 ~ value_6 같은 라벨 원본 필드에는 직접 접근하지 않는다.
      (현재 ScalpAnalysisResponse 스키마에는 존재하지 않음)
    """
    # 안전하게 getattr 사용 (혹시 일부 필드가 비어 있어도 처리되도록)
    risk_score = getattr(analysis, "risk_score", None)
    risk_level = getattr(analysis, "risk_level", None)
    summary = getattr(analysis, "summary", "")
    details = getattr(analysis, "details", "")
    recommendations = getattr(analysis, "recommendations", []) or []

    if language == "en":
        rec_lines = []
        for idx, rec in enumerate(recommendations, start=1):
            title = getattr(rec, "title", f"Recommendation {idx}")
            desc = getattr(rec, "description", "")
            rec_lines.append(f"- {title}: {desc}")
        rec_block = (
            "\n".join(rec_lines) if rec_lines else "No specific recommendations."
        )

        prompt = f"""
        You are an AI assistant that writes scalp condition reports for clients.

        Risk score: {risk_score}
        Risk level: {risk_level}

        Summary of scalp condition:
        {summary}

        Detailed analysis:
        {details}

        Recommendations:
        {rec_block}

        Please write a clear and friendly report for the client.
        The report should be written in natural English and include:
        1. A one-paragraph overview.
        2. Key issues to pay attention to.
        3. Concrete care tips for the next 1–3 months.
        """
        return dedent(prompt).strip()

    # 기본: 한국어
    rec_lines = []
    for idx, rec in enumerate(recommendations, start=1):
        title = getattr(rec, "title", f"권장사항 {idx}")
        desc = getattr(rec, "description", "")
        rec_lines.append(f"- {title}: {desc}")
    rec_block = (
        "\n".join(rec_lines)
        if rec_lines
        else "현재 특별히 강조할 관리 권장사항은 없습니다."
    )

    prompt = f"""
    당신은 두피 상태를 설명해주는 전문 상담사입니다.
    아래 분석 결과를 바탕으로, 고객에게 전달할 두피 상태 리포트를 작성하세요.

    [위험 지표]
    - 위험 점수: {risk_score}
    - 위험 등급: {risk_level}

    [요약]
    {summary}

    [상세 분석]
    {details}

    [권장 관리]
    {rec_block}

    작성 가이드:
    1. 전체적으로는 친절하고 차분한 톤으로 작성합니다.
    2. 현재 두피 상태를 솔직하게 설명하되, 불필요한 공포감을 주지 않도록 합니다.
    3. 1~3개월 동안 실천 가능한 구체적인 관리 방법을 제시합니다.
    4. 생활 습관(샴푸 습관, 펌/염색 빈도 등)이 영향을 줄 수 있지만,
       이 분석 결과만으로 '원인'을 단정하지 말고
       '함께 관찰되는 경향' 수준에서 설명합니다.
    """
    return dedent(prompt).strip()


def generate_dummy_llm_report(
    analysis: ScalpAnalysisResponse,
    language: str = "ko",
) -> str:
    """
    실제 LLM 없이도 파이프라인 테스트할 수 있는 더미 리포트 생성기.
    나중에 OpenAI/LangChain 연결할 때 이 함수를 교체하거나 내부에서 LLM 호출 추가.
    """
    prompt = build_report_prompt(analysis, language=language)

    # 지금은 아주 단순한 템플릿 기반 응답으로 대체
    # (필요하면 여기에 간단한 규칙 더 추가해도 됨)
    summary = f"현재 전체 두피 상태는 '{analysis.risk_level}' 위험도이며, 점수는 {analysis.risk_score}입니다."

    tips = [
        "샴푸는 두피에 자극이 덜한 제품을 사용하고, 손톱 대신 손가락 지문으로 부드럽게 마사지하세요.",
        "드라이기 사용 시 너무 뜨거운 바람을 피하고, 두피에서 20cm 이상 떨어뜨려 사용하세요.",
        "펌/염색 주기가 짧다면, 최소 2~3개월 간격으로 늘려 두피 자극을 줄여 보세요.",
        "평소 수면, 스트레스 관리도 탈모 및 두피 건강에 영향을 줄 수 있으니 규칙적인 생활을 유지해보세요.",
    ]

    tips_block = "\n".join([f"- {t}" for t in tips])

    dummy_report = f"""{summary}

(아래는 rule-based 분석과 기본적인 두피 관리 원칙을 바탕으로 한 자동 리포트 초안입니다.)

1) 현재 두피 상태 요약
- 위험도: {analysis.risk_level}
- 점수: {analysis.risk_score}

2) 증상별 코멘트
(자세한 내용은 UI에서 증상별 그래프/아이콘으로 함께 보여줄 수 있습니다.)

3) 생활습관 관점 제안
- 샴푸, 펌/염색 등은 두피 자극을 줄이는 방향으로 조정해 보세요.
- 개인별 상황에 따라 전문의 상담이 필요한 경우도 있습니다.

4) 향후 1~3개월 실천 팁:
{tips_block}
"""
    return dummy_report


def generate_dummy_llm_report(
    analysis: ScalpAnalysisResponse,
    language: Literal["ko", "en"] = "ko",
) -> str:
    """
    실제 LLM 대신, rule-based 분석 결과를 기반으로
    간단한 한국어/영어 리포트 문장을 만들어주는 더미 함수.

    - 지금은 prompt를 생성만 하고, 그 내용을 요약/가공해서 그대로 리포트로 쓴다.
    - 나중에 진짜 LLM을 붙일 때는 이 함수 내부에서 openai / huggingface 호출로 교체 가능.
    """
    prompt = build_report_prompt(analysis, language=language)

    # 🔹 지금은 간단하게, prompt의 핵심 부분을 기반으로 리포트를 구성
    #    (실제 LLM 붙이면 이 부분을 LLM 응답으로 교체)
    risk_line = (
        f"위험 점수 {analysis.risk_score}, 등급은 '{analysis.risk_level}'입니다."
        if language == "ko"
        else f"Risk score is {analysis.risk_score} with level '{analysis.risk_level}'."
    )

    if language == "en":
        base_report = f"""
        Scalp Condition Report

        {risk_line}

        Summary:
        {analysis.summary}

        Details:
        {analysis.details}

        Care Recommendations (1–3 months):
        """
        lines = [base_report.strip()]
        for rec in analysis.recommendations:
            lines.append(f"- {rec.title}: {rec.description}")
        return "\n".join(lines).strip()

    # 기본: 한국어 리포트
    base_report = f"""
    두피 상태 리포트

    {risk_line}

    [요약]
    {analysis.summary}

    [상세 설명]
    {analysis.details}

    [1~3개월 관리 권장사항]
    """
    lines = [dedent(base_report).strip()]
    for rec in analysis.recommendations:
        lines.append(f"- {rec.title}: {rec.description}")
    return "\n".join(lines).strip()


# 만약 실제 OpenAI LLM을 붙이고 싶다면 이런 식으로:
"""
from openai import OpenAI
client = OpenAI()

def generate_llm_report_openai(analysis: ScalpAnalysisResponse) -> str:
    prompt = build_report_prompt(analysis, language="ko")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 두피 케어 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content
"""
