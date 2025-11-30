from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    String,
    Integer,
    Date,
    DateTime,
    Text,
    Float,
    ForeignKey,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


# -----------------------------
# 1) User (고객)
# -----------------------------
class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # "male"/"female"/"unknown"
    birth_date = mapped_column(Date, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 관계: 한 명의 유저는 여러 방문(Visit)을 가질 수 있음
    visits: Mapped[List["Visit"]] = relationship(
        "Visit",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# -----------------------------
# 2) Visit (방문 세션)
# -----------------------------
class Visit(Base):
    __tablename__ = "visits"

    visit_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    visit_date: Mapped[date] = mapped_column(Date, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 관계: Visit ↔ User (N:1)
    user: Mapped["User"] = relationship(
        "User",
        back_populates="visits",
    )

    # 관계: Visit ↔ VisitReport (1:1)
    report: Mapped[Optional["VisitReport"]] = relationship(
        "VisitReport",
        back_populates="visit",
        uselist=False,
        cascade="all, delete-orphan",
    )


# -----------------------------
# 3) VisitReport (방문 리포트)
# -----------------------------
class VisitReport(Base):
    __tablename__ = "visit_reports"

    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    visit_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("visits.visit_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # 방문당 리포트 하나라고 가정
    )

    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    history_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plan_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    report_text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 관계: VisitReport ↔ Visit (1:1)
    visit: Mapped["Visit"] = relationship(
        "Visit",
        back_populates="report",
    )
