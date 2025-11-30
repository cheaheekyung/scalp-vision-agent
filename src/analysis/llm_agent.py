from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI

from src.config import PROJECT_ROOT
from src.schemas import (
    ScalpCondition,
    UserProfile,
    RecommendationItem,
    ScalpAnalysisResponse,
)

# --------------------------------
# 환경 변수 로드 (.env)
# --------------------------------
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(f"OPENAI_API_KEY is not set. Checked dotenv path: {DOTENV_PATH}")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-5.1-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))

client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------
# 프롬프트 빌드 (value_1~6 + 프로필)
# --------------------------------
def build_scalp_prompt(
    *,
    condition: ScalpCondition,
    profile: UserProfile | None,
    risk_score: float,
    risk_level: str,
) -> str:
    value_desc = (
        f"- value_1(각질): {condition.value_1}\n"
        f"- value_2(피지): {condition.value_2}\n"
        f"- value_3(모낭 사이 홍반): {condition.value_3}\n"
        f"- value_4(모낭 홍반/농포): {condition.value_4}\n"
        f"- value_5(비듬): {condition.value_5}\n"
        f"- value_6(탈모): {condition.value_6}\n"
    )

    if profile:
        profile_desc = (
            f"- 성별: {profile.gender or '알 수 없음'}\n"
            f"- 나이: {profile.age or '알 수 없음'}\n"
            f"- 샴푸 빈도: {profile.shampoo_frequency or '정보 없음'}\n"
            f"- 펌 빈도: {profile.perm_frequency or '정보 없음'}\n"
            f"- 염색 빈도: {profile.dye_frequency or '정보 없음'}\n"
        )
    else:
        profile_desc = "- 프로필: 제공되지 않음\n"

    text = f"""
너는 두피 관리샵/클리닉에서 사용하는 두피 리포트를 작성하는 전문가야.
의학적 진단이나 처방을 내리는 것은 아니고,
고객이 이해하기 쉬운 '상담 리포트'를 작성하는 역할이다.

[두피 상태 점수]
{value_desc}

[종합 위험도]
- risk_score: {risk_score}
- risk_level: {risk_level}  # normal / low / medium / high 중 하나

[고객 프로필]
{profile_desc}

위 정보를 바탕으로 다음 JSON 형식에 맞춰 리포트를 작성해줘.

반환 형식(JSON, 다른 텍스트 금지):

{{
  "summary": "한 줄 요약 (한국어, 1~2문장)",
  "details": "현재 두피 상태를 자세히 설명 (한국어, 3~6문장)",
  "recommendations": [
    {{
      "title": "짧은 추천 제목",
      "description": "구체적인 관리 방법 또는 생활습관 조언 (한국어, 1~2문장)"
    }}
  ],
  "report_text": "상담 카드에 그대로 들어갈 정도로 자연스러운 전체 리포트 (한국어, 5~10문장)"
}}

주의:
- 꼭 위 JSON 형식만 출력하고, 앞뒤에 다른 문장은 쓰지 마.
- 의학적 진단이나 약/치료 처방은 하지 말고, 생활 관리/두피 케어 중심으로 작성해.
"""
    return text.strip()


# --------------------------------
# 실제 LLM 호출 (Chat Completions)
# --------------------------------
def _extract_json_from_text(text: str) -> dict[str, Any]:
    """
    LLM이 ```json ... ``` 형식으로 감싸서 줄 수도 있으니
    그 경우까지 감안해서 JSON만 깔끔하게 파싱.
    """
    stripped = text.strip()

    # ``` 코드블록 제거 시도
    if "```" in stripped:
        # ```json ... ``` 또는 ``` ... ``` 중간 내용만 추출
        parts = stripped.split("```")
        # parts 예: ["", "json\n{...}", ""]
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 'json' 접두어 제거
            if part.lower().startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # 일반적인 경우: 전체가 곧 JSON이라고 가정
    return json.loads(stripped)


def _call_llm(prompt: str) -> dict[str, Any]:
    """
    Chat Completions API를 이용해 JSON 응답을 받는다.
    ⚠️ 현재 사용하는 모델은 temperature 커스텀 값을 지원하지 않으므로
       temperature 파라미터는 전달하지 않는다.
    """
    resp = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 두피 케어 상담 리포트를 작성하는 전문가입니다. "
                    "반드시 유효한 JSON만 반환하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # ❌ temperature는 이 모델에서 1만 허용이거나, 아예 옵션 미지원이라 제거
        # temperature=LLM_TEMPERATURE,
    )

    choice = resp.choices[0]
    text = choice.message.content
    if text is None:
        raise RuntimeError("LLM 응답이 비어 있습니다. (message.content is None)")

    try:
        data = _extract_json_from_text(text)
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise RuntimeError(f"LLM JSON 파싱 실패: {exc}\n원본: {text!r}")

    return data


# --------------------------------
# 메인: LLM 기반 두피 리포트 생성
# --------------------------------
def generate_llm_scalp_report(
    *,
    condition: ScalpCondition,
    profile: UserProfile | None,
    risk_score: float,
    risk_level: str,
) -> tuple[ScalpAnalysisResponse, str]:
    """
    rule 기반 risk_score/level을 받고,
    LLM을 호출해 ScalpAnalysisResponse + report_text(VisitReport용)를 생성한다.
    """
    prompt = build_scalp_prompt(
        condition=condition,
        profile=profile,
        risk_score=risk_score,
        risk_level=risk_level,
    )
    data = _call_llm(prompt)

    rec_items = [
        RecommendationItem(
            title=(item.get("title") or "").strip(),
            description=(item.get("description") or "").strip(),
        )
        for item in data.get("recommendations", [])
    ]
    # risk_score_int = int(round(risk_score))
    response = ScalpAnalysisResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        summary=(data.get("summary") or "").strip(),
        details=(data.get("details") or "").strip(),
        recommendations=rec_items,
    )

    report_text = (data.get("report_text") or "").strip() or response.details

    return response, report_text


# --------------------------------
# (구) 리포트 프롬프트 + 더미 리포트 (fallback용)
# --------------------------------
def build_report_prompt(
    analysis: ScalpAnalysisResponse,
    language: Literal["ko", "en"] = "ko",
) -> str:
    """
    ScalpAnalysisResponse(risk_score, risk_level, summary, details, recommendations)를
    기반으로 LLM에 전달할 프롬프트 텍스트를 생성한다.
    """
    risk_score = getattr(analysis, "risk_score", None)
    risk_level = getattr(analysis, "risk_level", None)
    summary = getattr(analysis, "summary", "")
    details = getattr(analysis, "details", "")
    recommendations = getattr(analysis, "recommendations", []) or []

    if language == "en":
        rec_lines: list[str] = []
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
    rec_lines_ko: list[str] = []
    for idx, rec in enumerate(recommendations, start=1):
        title = getattr(rec, "title", f"권장사항 {idx}")
        desc = getattr(rec, "description", "")
        rec_lines_ko.append(f"- {title}: {desc}")
    rec_block_ko = (
        "\n".join(rec_lines_ko)
        if rec_lines_ko
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
    {rec_block_ko}

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
    language: Literal["ko", "en"] = "ko",
) -> str:
    """
    실제 LLM 대신, rule-based 분석 결과를 기반으로
    간단한 한국어/영어 리포트 문장을 만들어주는 더미 함수.
    """
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
