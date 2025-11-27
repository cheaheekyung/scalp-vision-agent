# src/api/main.py (기존 내용 + 아래 추가)
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
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
from src.analysis.report_rules import simple_rule_based_analysis
from src.analysis.llm_agent import generate_dummy_llm_report
from typing import List, Optional
from fastapi import Query

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


@app.post(
    "/visits/{visit_id}/analyze-demo",
    response_model=ScalpAnalysisResponse,
)
def analyze_visit_demo(
    visit_id: int,
    payload: ScalpAnalysisRequest,
    db: Session = Depends(get_db),
) -> ScalpAnalysisResponse:
    """
    CNN 없이, 증상/프로필을 직접 받아서:
    - rule-based 분석 수행
    - VisitReport 테이블에 리포트 저장
    - ScalpAnalysisResponse 반환

    나중에 CNN을 붙이면:
    - payload.condition 부분을 모델 출력으로 채우거나
    - 아예 body 없이 visit_id만 받고 서버가 CNN + META를 조합해서 만드는 방식으로 확장 가능.
    """

    # 1) visit 존재 여부 확인
    visit = db.get(db_models.Visit, visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found")

    # 2) rule-based 분석 실행 (CNN 없이, payload 기준)
    analysis = simple_rule_based_analysis(payload)

    # 3) VisitReport용 report_text 구성
    report_text = generate_dummy_llm_report(analysis)

    # 4) 기존 리포트가 있으면 덮어쓰기 (visit당 리포트 하나라고 가정)
    #    - 이미 visit에 report가 달려 있다면 삭제 후 새로 생성
    existing_report = (
        db.query(db_models.VisitReport)
        .filter(db_models.VisitReport.visit_id == visit_id)
        .one_or_none()
    )
    if existing_report is not None:
        db.delete(existing_report)
        db.commit()

    # 5) 새 VisitReport 생성 & 저장
    db_report = db_models.VisitReport(
        visit_id=visit_id,
        risk_score=float(analysis.risk_score),
        risk_level=analysis.risk_level,
        report_text=report_text,
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    # 6) 클라이언트에는 분석 결과(ScalpAnalysisResponse) 그대로 반환
    return analysis


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
def analyze_demo(request: ScalpAnalysisRequest) -> ScalpAnalysisResponse:
    """
    CNN 모델 없이:
    - 이미 들어온 scalp_condition + user_profile을 사용해서
    - rule-based 분석 + LLM 스타일 리포트까지 생성해 보는 데모용 엔드포인트
    """
    # 1) rule-based 분석
    analysis = simple_rule_based_analysis(request)

    # 2) LLM(지금은 더미) 리포트 생성
    report_text = generate_dummy_llm_report(analysis)

    # ScalpAnalysisResponse 안에 report_text 비슷한 필드가 있으면 채워주고,
    # 없으면 우선 analysis를 그냥 반환하거나, 스키마를 살짝 확장해도 됨.
    # 여기서는 analysis에 report_text 속성이 있다고 가정하고 예시:
    if hasattr(analysis, "report_text"):
        analysis.report_text = report_text

    return analysis
