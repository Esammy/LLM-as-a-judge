"""Database schema.

Two tables. Runs hold the aggregate and, crucially, the **rubric fingerprint**
that produced it; judgements hold the per-case detail.

Storing the fingerprint on every row is the point rather than an audit
nicety. Once results live in a database they outlive the rubric file that made
them, and the first thing anybody does with stored eval history is plot it over
time. Without the fingerprint travelling alongside the score there is nothing
stopping that chart from silently splicing together numbers graded under
different rules - which is exactly the failure this project exists to prevent,
reappearing at the storage layer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class RunRow(Base):
    """One execution of a dataset against a rubric."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)

    dataset_name: Mapped[str] = mapped_column(String(200), index=True)
    dataset_version: Mapped[str] = mapped_column(String(50))

    rubric_id: Mapped[str] = mapped_column(String(200), index=True)
    rubric_version: Mapped[str] = mapped_column(String(50))
    rubric_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    """Indexed because "show me every run comparable with this one" is the
    query that makes stored history safe to plot."""

    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(200))

    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    """queued -> running -> completed | failed."""

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    errored: Mapped[int] = mapped_column(Integer, default=0)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    judgements: Mapped[list[JudgementRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        # The history query: this rubric, newest first.
        Index("ix_runs_rubric_created", "rubric_fingerprint", "created_at"),
    )


class JudgementRow(Base):
    """One scored case within a run."""

    __tablename__ = "judgements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("runs.run_id", ondelete="CASCADE"), index=True
    )

    case_id: Mapped[str] = mapped_column(String(200), index=True)
    score: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(20), index=True)
    reasoning: Mapped[str] = mapped_column(Text, default="")

    criterion_scores: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list)

    model: Mapped[str] = mapped_column(String(200), default="")
    family: Mapped[str] = mapped_column(String(50), default="")
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[RunRow] = relationship(back_populates="judgements")

    __table_args__ = (Index("ix_judgements_run_case", "run_id", "case_id", unique=True),)
