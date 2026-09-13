"""Request and response shapes for the HTTP API.

Kept separate from the core models on purpose. ``RunResult`` is an internal
structure that should be free to change; an HTTP response is a contract with
somebody else's code. Collapsing the two makes every internal refactor a
breaking API change.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from judgekit.storage.models import JudgementRow, RunRow


class SubmitRunRequest(BaseModel):
    """Ask for a dataset to be judged."""

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(description="Dataset filename, resolved inside the dataset directory.")
    rubric: str = Field(description="Rubric filename, resolved inside the rubric directory.")
    provider: str | None = Field(default=None, description="Defaults to the configured provider.")
    concurrency: int | None = Field(default=None, gt=0, le=64)


class SubmitRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str


class RubricRefOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    version: str
    fingerprint: str


class JudgementOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    score: float
    verdict: str
    reasoning: str
    flags: tuple[str, ...]
    criterion_scores: dict[str, float]
    elapsed_ms: float
    error: str | None

    @classmethod
    def from_row(cls, row: JudgementRow) -> JudgementOut:
        return cls(
            case_id=row.case_id,
            score=row.score,
            verdict=row.verdict,
            reasoning=row.reasoning,
            flags=tuple(row.flags or ()),
            criterion_scores=dict(row.criterion_scores or {}),
            elapsed_ms=row.elapsed_ms,
            error=row.error,
        )


class RunSummary(BaseModel):
    """A run without its per-case detail, for listings."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    dataset: str
    dataset_version: str
    rubric: RubricRefOut
    provider: str
    model: str
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    mean_score: float
    cost_usd: float
    created_at: datetime
    finished_at: datetime | None
    error: str | None

    @classmethod
    def from_row(cls, row: RunRow) -> RunSummary:
        return cls(
            run_id=row.run_id,
            status=row.status,
            dataset=row.dataset_name,
            dataset_version=row.dataset_version,
            rubric=RubricRefOut(
                id=row.rubric_id,
                version=row.rubric_version,
                fingerprint=row.rubric_fingerprint,
            ),
            provider=row.provider,
            model=row.model,
            total=row.total,
            passed=row.passed,
            failed=row.failed,
            errored=row.errored,
            pass_rate=row.pass_rate,
            mean_score=row.mean_score,
            cost_usd=row.cost_usd,
            created_at=row.created_at,
            finished_at=row.finished_at,
            error=row.error,
        )


class RunDetail(RunSummary):
    """A run with every judgement."""

    judgements: tuple[JudgementOut, ...] = ()

    @classmethod
    def from_row(cls, row: RunRow) -> RunDetail:
        summary = RunSummary.from_row(row)
        return cls(
            **summary.model_dump(),
            judgements=tuple(JudgementOut.from_row(j) for j in row.judgements),
        )


class RunList(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs: tuple[RunSummary, ...]
    count: int


class ComparisonOut(BaseModel):
    """The result of diffing two runs."""

    model_config = ConfigDict(frozen=True)

    baseline_run_id: str
    candidate_run_id: str
    pass_rate_before: float
    pass_rate_after: float
    pass_rate_delta: float
    mean_score_before: float
    mean_score_after: float
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    gate_passed: bool


class HealthOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    version: str


class ReadyOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    database: str
    detail: str | None = None
