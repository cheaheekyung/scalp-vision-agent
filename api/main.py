# api/main.py

from fastapi import FastAPI

from src.schemas import (
    ScalpAnalysisRequest,
    ScalpAnalysisResponse,
)
from src.analysis.report_rules import simple_rule_based_analysis


app = FastAPI(title="Scalp Vision Agent API")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/scalp/analyze", response_model=ScalpAnalysisResponse)
def analyze_scalp(req: ScalpAnalysisRequest) -> ScalpAnalysisResponse:
    """
    CNN + 메타데이터결과를 대신해서,
    현재는 rule-based 리포트 생성기로만 리포트를 만들어주는 엔드포인트.
    나중에 여기서 CNN 추론/LLM 호출 로직으로 교체할 예정.
    """
    resp = simple_rule_based_analysis(req)
    return resp
