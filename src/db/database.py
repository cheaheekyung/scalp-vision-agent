from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


# ---- 1) DB URL 설정 ----
#   - 지금은 로컬 SQLite로 시작 (파일 이름: scalp_vision.db)
#   - 나중에 PostgreSQL로 바꾸고 싶으면 이 부분만 바꾸면 됨.
DATABASE_URL = "sqlite:///./scalp_vision.db"


# ---- 2) 엔진 생성 ----
#   - SQLite에서만 필요한 connect_args, 다른 DB에서는 빼도 됨.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 전용 옵션
)


# ---- 3) 세션 팩토리 ----
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---- 4) Declarative Base ----
class Base(DeclarativeBase):
    """모든 ORM 모델이 상속받을 Base 클래스"""

    pass


# ---- 5) FastAPI에서 사용할 세션 dependency ----
#   (엔드포인트에서: db: Session = Depends(get_db) 형태로 사용할 예정)
def get_db():
    from sqlalchemy.orm import Session

    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
