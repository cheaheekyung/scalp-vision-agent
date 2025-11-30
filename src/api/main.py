from __future__ import annotations

from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from src.db.database import get_db, engine
from src.db import models as db_models
from src.schemas import (
    ScalpCondition,
    UserProfile,
    ScalpAnalysisRequest,
    ScalpAnalysisResponse,
    UserCreate,
    User,
    VisitCreate,
    Visit,
    VisitReport,
    FullVisitResponse,
    UserUpdate,
)

from src.analysis.snapshots import save_analysis_snapshot
from src.analysis.report_rules import simple_rule_based_analysis
from src.analysis.llm_agent import generate_dummy_llm_report, generate_llm_scalp_report
from src.analysis.agent_nodes import run_rule_risk, run_llm_report
from src.inference import predict_condition_from_bytes

import logging
import json

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scalp Vision Agent API",
    version="0.1.0",
)

db_models.Base.metadata.create_all(bind=engine)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계라 * 허용, 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


@app.post("/users", response_model=User)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> User:
    """
    신규 고객 생성 (DB 저장).
    """
    user = db_models.User(
        name=payload.name,
        gender=payload.gender,
        birth_date=payload.birth_date,
    )

    db.add(user)
    db.commit()
    db.refresh(user)  # INSERT된 값(id, created_at 등) 다시 가져오기

    return user


@app.get("/users", response_model=list[User])
def list_users(db: Session = Depends(get_db)):
    users = db.query(db_models.User).order_by(db_models.User.user_id.desc()).all()
    return users


@app.get("/users/search", response_model=List[User])
def search_users(
    name: str = Query(..., description="이름 또는 이름 일부"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    이름으로 유저 검색.

    - 부분 일치 검색 (예: '희' → '김희경', '이희수' 등)
    - 여러 명 나올 수 있으니 리스트로 반환
    """
    # SQLite 기준: ilike도 동작함 (대소문자 구분 없이 부분검색)
    query = (
        db.query(db_models.User)
        .filter(db_models.User.name.ilike(f"%{name}%"))
        .order_by(db_models.User.created_at.desc())
        .limit(limit)
    )

    users = query.all()
    return users


@app.get("/users/{user_id}", response_model=User)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
) -> User:
    """
    고객 한 명 조회.
    """
    user = db.get(db_models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users/{user_id}", response_model=User)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
):
    user = db.query(db_models.User).filter(db_models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 들어온 값만 선택적으로 업데이트
    if payload.name is not None:
        user.name = payload.name
    if payload.gender is not None:
        user.gender = payload.gender
    if payload.birth_date is not None:
        user.birth_date = payload.birth_date

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@app.post("/visits", response_model=Visit)
def create_visit(
    payload: VisitCreate,
    db: Session = Depends(get_db),
) -> Visit:
    """
    방문 세션 생성.
    - user_id가 존재하는 유저인지 확인
    - visit_date, note 저장
    """
    # 1) user_id 존재 여부 체크 (안 하면 외래키 에러 날 수도 있음)
    user = db.get(db_models.User, payload.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 2) Visit 생성
    visit = db_models.Visit(
        user_id=payload.user_id,
        visit_date=payload.visit_date,
        note=payload.note,
    )

    db.add(visit)
    db.commit()
    db.refresh(visit)

    # 3) Pydantic Visit으로 응답
    return visit


@app.get("/visits/{visit_id}", response_model=Visit)
def get_visit(
    visit_id: int,
    db: Session = Depends(get_db),
) -> Visit:
    """
    방문 세션 한 건 조회.
    """
    visit = db.get(db_models.Visit, visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found")
    return visit


@app.post("/visits/{visit_id}/analyze-demo", response_model=ScalpAnalysisResponse)
def analyze_demo(
    visit_id: int,
    payload: ScalpAnalysisRequest,
    db: Session = Depends(get_db),
) -> ScalpAnalysisResponse:
    # 0) 방문 조회
    visit = (
        db.query(db_models.Visit)
        .filter(db_models.Visit.visit_id == visit_id)
        .one_or_none()
    )
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found")

    condition = payload.condition
    profile = payload.profile

    # 1) rule 기반 위험도 계산
    risk_score, risk_level = simple_rule_based_analysis(
        condition=condition,
        profile=profile,
    )
    # risk_score_int = int(round(risk_score))

    # 2) LLM 호출 시도 + 실패 시 폴백
    llm_ok = True
    llm_error: str | None = None

    try:
        llm_response, report_text = generate_llm_scalp_report(
            condition=condition,
            profile=profile,
            risk_score=risk_score,
            risk_level=risk_level,
        )
    except Exception as e:  # noqa: BLE001
        llm_ok = False
        llm_error = str(e)
        logger.exception("LLM 리포트 생성 중 오류 발생: %s", e)

        # rule 기반 정보로만 응답을 한 번 만든 뒤
        fallback = ScalpAnalysisResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            summary=(
                "현재는 LLM 리포트 생성에 오류가 발생하여, "
                "기본 rule 기반 요약만 제공합니다."
            ),
            details=(
                "두피 상태는 rule 기반 점수만으로 평가되었습니다. "
                "추후 시스템 안정화 후 다시 분석을 권장드립니다."
            ),
            recommendations=[],
            report_text="",
        )
        # 기존 dummy 리포트 생성기로 자연어 report_text 생성
        report_text = generate_dummy_llm_report(fallback, language="ko")
        fallback.report_text = report_text
        llm_response = fallback

    # 2.5) 동일 사용자 과거 방문과 비교해서 history_message / plan_text 생성
    history_message: str | None = None
    plan_text: str | None = None

    # 같은 user_id의 다른 방문 중, 가장 최근 방문 하나
    previous_visit = (
        db.query(db_models.Visit)
        .filter(
            db_models.Visit.user_id == visit.user_id,
            db_models.Visit.visit_id != visit_id,
        )
        .order_by(db_models.Visit.visit_date.desc())
        .first()
    )

    if previous_visit and previous_visit.report:
        prev_report = previous_visit.report
        diff = risk_score - float(prev_report.risk_score)

        if abs(diff) < 0.5:
            trend = "이전과 비슷한 수준입니다."
        elif diff < 0:
            trend = "이전보다 위험도가 조금 낮아졌습니다."
        else:
            trend = "이전보다 위험도가 조금 높아졌습니다."

        history_message = (
            f"이전 방문({previous_visit.visit_date})의 위험도는 "
            f"{prev_report.risk_level}({prev_report.risk_score:.1f})였고, "
            f"이번 방문은 {risk_level}({risk_score:.1f})입니다. {trend}"
        )
    else:
        history_message = "이번이 첫 방문이라, 과거 기록과의 비교는 아직 어렵습니다."

    # 위험도 레벨별 간단 관리 플랜
    if risk_level in ("high", "very_high"):
        plan_text = (
            "1~2주 이내에 피부과나 두피 클리닉 상담을 권장드립니다. "
            "그 전까지는 펌/염색 등 자극적인 시술을 피하고, "
            "두피 전용 샴푸 사용과 수면, 스트레스 관리에 특히 신경 써 주세요."
        )
    elif risk_level in ("medium", "moderate"):
        plan_text = (
            "향후 1~3개월 동안 생활 습관 관리에 집중해 보세요. "
            "주 2~3회 저자극 샴푸 사용, 충분한 수면, 균형 잡힌 식단을 유지하면서 "
            "증상 변화가 있으면 사진과 함께 기록해두면 좋습니다."
        )
    else:
        # low / normal 등 비교적 양호한 구간
        plan_text = (
            "현재는 큰 이상 소견은 아니지만, "
            "기본적인 두피 관리 습관을 유지하면서 6개월~1년 간격으로 "
            "정기적으로 상태를 체크해 보길 권장드립니다."
        )

    # 확장된 스키마 필드에 세팅
    llm_response.history_message = history_message
    llm_response.plan_text = plan_text

    # 3) VisitReport upsert
    existing = (
        db.query(db_models.VisitReport)
        .filter(db_models.VisitReport.visit_id == visit_id)
        .one_or_none()
    )
    if existing is not None:
        db.delete(existing)
        db.flush()

    # recommendations를 JSON으로 직렬화
    recommendations_json = (
        json.dumps(
            [r.model_dump() for r in llm_response.recommendations],
            ensure_ascii=False,
        )
        if llm_response.recommendations
        else None
    )

    report = db_models.VisitReport(
        visit_id=visit_id,
        risk_score=risk_score,
        risk_level=risk_level,
        summary=llm_response.summary,
        details=llm_response.details,
        history_message=history_message,
        plan_text=plan_text,
        recommendations_json=recommendations_json,
        report_text=report_text,
    )

    # snapshot 저장
    save_analysis_snapshot(
        visit_id=visit_id,
        condition=condition,
        profile=profile,
        risk_score=risk_score,
        risk_level=risk_level,
        llm_ok=llm_ok,
        llm_error=llm_error,
        analysis=llm_response,
        report_text=report_text,
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    # 4) React Admin으로 보내줄 응답
    return llm_response


@app.get("/visits/{visit_id}/report", response_model=VisitReport)
def get_visit_report(
    visit_id: int,
    db: Session = Depends(get_db),
) -> VisitReport:
    """
    특정 방문 세션에 대한 저장된 리포트 조회.
    """
    report = (
        db.query(db_models.VisitReport)
        .filter(db_models.VisitReport.visit_id == visit_id)
        .one_or_none()
    )
    if report is None:
        raise HTTPException(status_code=404, detail="VisitReport not found")
    return report


@app.get("/users/{user_id}/visits", response_model=list[Visit])
def list_user_visits(
    user_id: int,
    db: Session = Depends(get_db),
):
    """
    특정 User 가 가진 방문(Visit) 목록을 조회하는 엔드포인트.

    - 404: user 가 존재하지 않을 때
    - 200: Visit 리스트 (없으면 빈 리스트)
    """
    user = db.query(db_models.User).filter(db_models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    visits = (
        db.query(db_models.Visit)
        .filter(db_models.Visit.user_id == user_id)
        .order_by(db_models.Visit.visit_date.desc())
        .all()
    )
    return visits


@app.get("/visits/{visit_id}/full", response_model=FullVisitResponse)
def get_full_visit_info(
    visit_id: int,
    db: Session = Depends(get_db),
):
    """
    Visit 하나에 대한 전체 정보를 한 번에 조회하는 엔드포인트.

    - user: 이 방문을 한 사용자 정보 (Pydantic User)
    - visit: 방문 정보 (Pydantic Visit)
    - report: 방문 리포트 (Pydantic VisitReport | None)
    """
    visit_obj = (
        db.query(db_models.Visit).filter(db_models.Visit.visit_id == visit_id).first()
    )
    if visit_obj is None:
        raise HTTPException(status_code=404, detail="Visit not found")

    user_obj = visit_obj.user
    report_obj = visit_obj.report  # 1:1 관계, 없으면 None

    if user_obj is None:
        raise HTTPException(
            status_code=500,
            detail="Inconsistent data: user not found for this visit",
        )

    user_schema = User.model_validate(user_obj, from_attributes=True)
    visit_schema = Visit.model_validate(visit_obj, from_attributes=True)
    report_schema = (
        VisitReport.model_validate(report_obj, from_attributes=True)
        if report_obj is not None
        else None
    )

    return FullVisitResponse(
        user=user_schema,
        visit=visit_schema,
        report=report_schema,
    )


@app.post("/analyze/demo", response_model=ScalpAnalysisResponse)
def analyze_demo_no_visit(request: ScalpAnalysisRequest) -> ScalpAnalysisResponse:
    """
    CNN/LLM 없이:
    - scalp_condition + user_profile을 입력받아서
    - rule-based 분석 결과만 반환하는 간단 데모용 엔드포인트
    """
    condition = request.condition
    profile = request.profile

    # 1) rule-based 분석
    risk_score, risk_level = simple_rule_based_analysis(
        condition=condition,
        profile=profile,
    )
    # risk_score_int = int(round(risk_score))

    # 2) rule 기반 결과만 담은 ScalpAnalysisResponse 구성
    analysis = ScalpAnalysisResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        summary="rule 기반 간단 분석 결과입니다.",
        details="CNN/LLM 없이 rule 기반 점수만으로 평가한 데모 응답입니다.",
        recommendations=[],
    )

    return analysis


@app.post("/visits/{visit_id}/analyze-image", response_model=ScalpAnalysisResponse)
def analyze_visit_from_image(
    visit_id: int,
    file: UploadFile = File(...),
    gender: str = "U",  # "M" / "F" / "U"
    age: int | None = None,
    db: Session = Depends(get_db),
) -> ScalpAnalysisResponse:
    """
    업로드된 두피 이미지 1장을 기반으로:
    1) CNN으로 value_1~6 등급 예측
    2) rule 기반 위험도 계산
    3) LLM 리포트 생성 + DB/스냅샷 저장
    4) ScalpAnalysisResponse 반환
    """

    # 0) visit 존재 여부 확인
    visit = db.get(db_models.Visit, visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found")

    # 1) 이미지 bytes 읽기
    image_bytes = file.file.read()

    # 2) CNN으로 value_1~6 예측
    preds = predict_condition_from_bytes(image_bytes)

    # 3) ScalpCondition 구성
    condition = ScalpCondition(
        sample_id=file.filename or f"visit_{visit_id}_image",
        location="TH",  # TODO: 나중에 프론트에서 실제 촬영 위치를 받아서 교체
        **preds,
    )

    # 4) UserProfile 구성
    profile = UserProfile(
        gender=gender,
        age=age,
        shampoo_frequency=None,
        perm_frequency=None,
        dye_frequency=None,
    )

    # 5) rule 기반 위험도 계산
    risk_score, risk_level = simple_rule_based_analysis(
        condition=condition,
        profile=profile,
    )
    # risk_score_int = int(round(risk_score))

    # 6) LLM 호출 (analyze-demo와 동일 패턴)
    llm_ok = True
    llm_error: str | None = None

    try:
        llm_response, report_text = generate_llm_scalp_report(
            condition=condition,
            profile=profile,
            risk_score=risk_score,
            risk_level=risk_level,
        )
    except Exception as e:  # noqa: BLE001
        llm_ok = False
        llm_error = str(e)
        logger.exception("LLM 리포트 생성 중 오류 발생: %s", e)

        # rule 기반 결과만으로 fallback 응답 구성
        fallback = ScalpAnalysisResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            summary=(
                "현재는 LLM 리포트 생성에 오류가 발생하여, "
                "기본 rule 기반 요약만 제공합니다."
            ),
            details=(
                "두피 상태는 rule 기반 점수만으로 평가되었습니다. "
                "추후 시스템 안정화 후 다시 분석을 권장드립니다."
            ),
            recommendations=[],
            report_text="",
        )
        report_text = generate_dummy_llm_report(fallback, language="ko")
        fallback.report_text = report_text
        llm_response = fallback

    # 7) 과거 방문과 비교해 history_message / plan_text 생성 (analyze-demo 복붙)
    history_message: str | None = None
    plan_text: str | None = None

    previous_visit = (
        db.query(db_models.Visit)
        .filter(
            db_models.Visit.user_id == visit.user_id,
            db_models.Visit.visit_id != visit_id,
        )
        .order_by(db_models.Visit.visit_date.desc())
        .first()
    )

    if previous_visit and previous_visit.report:
        prev_report = previous_visit.report
        diff = risk_score - float(prev_report.risk_score)

        if abs(diff) < 0.5:
            trend = "이전과 비슷한 수준입니다."
        elif diff < 0:
            trend = "이전보다 위험도가 조금 낮아졌습니다."
        else:
            trend = "이전보다 위험도가 조금 높아졌습니다."

        history_message = (
            f"이전 방문({previous_visit.visit_date})의 위험도는 "
            f"{prev_report.risk_level}({prev_report.risk_score:.1f})였고, "
            f"이번 방문은 {risk_level}({risk_score:.1f})입니다. {trend}"
        )
    else:
        history_message = "이번이 첫 방문이라, 과거 기록과의 비교는 아직 어렵습니다."

    if risk_level in ("high", "very_high"):
        plan_text = (
            "1~2주 이내에 피부과나 두피 클리닉 상담을 권장드립니다. "
            "그 전까지는 펌/염색 등 자극적인 시술을 피하고, "
            "두피 전용 샴푸 사용과 수면, 스트레스 관리에 특히 신경 써 주세요."
        )
    elif risk_level in ("medium", "moderate"):
        plan_text = (
            "향후 1~3개월 동안 생활 습관 관리에 집중해 보세요. "
            "주 2~3회 저자극 샴푸 사용, 충분한 수면, 균형 잡힌 식단을 유지하면서 "
            "증상 변화가 있으면 사진과 함께 기록해두면 좋습니다."
        )
    else:
        plan_text = (
            "현재는 큰 이상 소견은 아니지만, "
            "기본적인 두피 관리 습관을 유지하면서 6개월~1년 간격으로 "
            "정기적으로 상태를 체크해 보길 권장드립니다."
        )

    llm_response.history_message = history_message
    llm_response.plan_text = plan_text

    # 8) VisitReport upsert
    existing = (
        db.query(db_models.VisitReport)
        .filter(db_models.VisitReport.visit_id == visit_id)
        .one_or_none()
    )
    if existing is not None:
        db.delete(existing)
        db.flush()

    recommendations_json = (
        json.dumps(
            [r.model_dump() for r in llm_response.recommendations],
            ensure_ascii=False,
        )
        if llm_response.recommendations
        else None
    )

    report = db_models.VisitReport(
        visit_id=visit_id,
        risk_score=risk_score,
        risk_level=risk_level,
        summary=llm_response.summary,
        details=llm_response.details,
        history_message=history_message,
        plan_text=plan_text,
        recommendations_json=recommendations_json,
        report_text=report_text,
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    # 9) snapshot 저장
    save_analysis_snapshot(
        visit_id=visit_id,
        condition=condition,
        profile=profile,
        risk_score=risk_score,
        risk_level=risk_level,
        llm_ok=llm_ok,
        llm_error=llm_error,
        analysis=llm_response,
        report_text=report_text,
    )

    # 10) 최종 응답
    return llm_response
