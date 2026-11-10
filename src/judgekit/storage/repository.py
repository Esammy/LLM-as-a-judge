"""Reading and writing runs.

Every query that returns more than one run is scoped by rubric fingerprint by
default. That is a deliberate default rather than a convenience: the reason to
store eval history is to look at it over time, and a chart that splices together
scores graded under different rubrics is worse than no chart, because it looks
like evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from judgekit.core.judge import Judgement
from judgekit.core.models import Flag, Verdict
from judgekit.core.rubric import RubricRef
from judgekit.core.runner import RunResult
from judgekit.storage.models import JudgementRow, RunRow


def _judgement_rows(run: RunResult) -> list[JudgementRow]:
    return [
        JudgementRow(
            case_id=j.case_id,
            score=j.score,
            verdict=j.verdict.value,
            reasoning=j.reasoning,
            criterion_scores=dict(j.criterion_scores),
            flags=[f.value for f in j.flags],
            model=j.model,
            family=j.family,
            elapsed_ms=j.elapsed_ms,
            error=j.error,
        )
        for j in run.judgements
    ]


def _apply(row: RunRow, run: RunResult, *, status: str) -> None:
    """Copy a finished run's aggregates onto ``row``.

    Deliberately leaves ``created_at`` alone: that is when the run was
    *submitted*, and history is ordered by it.
    """
    row.dataset_name = run.dataset_name
    row.dataset_version = run.dataset_version
    row.rubric_id = run.rubric.id
    row.rubric_version = run.rubric.version
    row.rubric_fingerprint = run.rubric.fingerprint
    row.provider = run.provider
    row.model = run.model
    row.status = status
    row.error = None
    row.total = run.total
    row.passed = run.passed
    row.failed = run.failed
    row.errored = run.errored
    row.pass_rate = run.pass_rate
    row.mean_score = run.mean_score
    row.prompt_tokens = run.usage.prompt_tokens
    row.completion_tokens = run.usage.completion_tokens
    row.cost_usd = run.usage.cost_usd
    row.started_at = run.started_at
    row.finished_at = run.finished_at


def _to_row(run: RunResult, *, status: str = "completed") -> RunRow:
    row = RunRow(run_id=run.run_id)
    _apply(row, run, status=status)
    row.judgements = _judgement_rows(run)
    return row


def to_result(row: RunRow) -> RunResult:
    """Rebuild a :class:`RunResult` from stored rows.

    The rubric ref is reconstructed in full, fingerprint included, so a run read
    back out of the database is still guarded by ``compare()``.
    """
    return RunResult(
        run_id=row.run_id,
        dataset_name=row.dataset_name,
        dataset_version=row.dataset_version,
        rubric=RubricRef(
            id=row.rubric_id,
            version=row.rubric_version,
            fingerprint=row.rubric_fingerprint,
        ),
        provider=row.provider,
        model=row.model,
        judgements=tuple(
            Judgement(
                case_id=j.case_id,
                score=j.score,
                verdict=Verdict(j.verdict),
                reasoning=j.reasoning,
                criterion_scores=dict(j.criterion_scores or {}),
                rubric=RubricRef(
                    id=row.rubric_id,
                    version=row.rubric_version,
                    fingerprint=row.rubric_fingerprint,
                ),
                flags=tuple(Flag(f) for f in (j.flags or [])),
                model=j.model,
                family=j.family,
                elapsed_ms=j.elapsed_ms,
                error=j.error,
            )
            for j in row.judgements
        ),
        started_at=row.started_at or row.created_at,
        finished_at=row.finished_at,
    )


async def reserve_run(
    session: AsyncSession,
    *,
    run_id: str,
    dataset_name: str,
    rubric_id: str,
    rubric_version: str,
    rubric_fingerprint: str,
    provider: str,
) -> RunRow:
    """Record a queued run before any work starts.

    Written up front so a submitted run is visible while it executes, and so a
    worker that dies mid-run leaves a row stuck in ``running`` rather than
    leaving no trace at all.
    """
    row = RunRow(
        run_id=run_id,
        dataset_name=dataset_name,
        dataset_version="",
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        rubric_fingerprint=rubric_fingerprint,
        provider=provider,
        model="",
        status="queued",
    )
    session.add(row)
    await session.flush()
    return row


async def mark_status(
    session: AsyncSession, run_id: str, status: str, *, error: str | None = None
) -> bool:
    """Move a run to ``status``. Returns False when the run does not exist."""
    row = await session.get(RunRow, run_id)
    if row is None:
        return False
    row.status = status
    row.error = error
    if status == "running":
        row.started_at = datetime.now(UTC)
    elif status in {"completed", "failed"}:
        row.finished_at = datetime.now(UTC)
    return True


async def save_run(session: AsyncSession, run: RunResult, *, status: str = "completed") -> RunRow:
    """Persist a finished run, filling in its reserved placeholder if there is one.

    The placeholder is updated in place rather than deleted and re-inserted, so
    ``created_at`` keeps meaning "when it was submitted" and a run's position in
    history does not move when it finishes.
    """
    row = await session.get(RunRow, run.run_id)
    if row is None:
        row = _to_row(run, status=status)
        session.add(row)
        await session.flush()
        return row

    # Flush removals before adding replacements: the unit of work runs inserts
    # before deletes, which would trip the (run_id, case_id) unique index.
    row.judgements.clear()
    await session.flush()

    _apply(row, run, status=status)
    row.judgements = _judgement_rows(run)
    await session.flush()
    return row


async def get_run(session: AsyncSession, run_id: str) -> RunRow | None:
    return await session.get(RunRow, run_id)


async def list_runs(
    session: AsyncSession,
    *,
    rubric_fingerprint: str | None = None,
    dataset_name: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Sequence[RunRow]:
    """Most recent runs first, optionally narrowed."""
    query = select(RunRow).order_by(RunRow.created_at.desc()).limit(limit).offset(offset)

    if rubric_fingerprint:
        query = query.where(RunRow.rubric_fingerprint == rubric_fingerprint)
    if dataset_name:
        query = query.where(RunRow.dataset_name == dataset_name)
    if status:
        query = query.where(RunRow.status == status)

    return (await session.execute(query)).scalars().all()


async def comparable_history(
    session: AsyncSession, run_id: str, *, limit: int = 50
) -> Sequence[RunRow]:
    """Runs that can legitimately be plotted alongside the given one.

    Scoped to the same rubric fingerprint, so a quality trend cannot silently
    span a rubric change. Anything else belongs on a different chart.
    """
    row = await session.get(RunRow, run_id)
    if row is None:
        return []

    query = (
        select(RunRow)
        .where(
            RunRow.rubric_fingerprint == row.rubric_fingerprint,
            RunRow.dataset_name == row.dataset_name,
            RunRow.status == "completed",
        )
        .order_by(RunRow.created_at.desc())
        .limit(limit)
    )
    return (await session.execute(query)).scalars().all()


async def delete_run(session: AsyncSession, run_id: str) -> bool:
    row = await session.get(RunRow, run_id)
    if row is None:
        return False
    await session.delete(row)
    return True
