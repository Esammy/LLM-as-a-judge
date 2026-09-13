"""The unit of work: judge one dataset and store the result.

One function, called identically whether the run was dispatched to an arq
worker or executed in-process. Keeping a single implementation is what stops
the two paths from drifting - the usual outcome being that the queued path
grows a behaviour the inline one silently lacks, and only production notices.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from judgekit.core.judge import Judge
from judgekit.core.models import Dataset
from judgekit.core.rubric import Rubric
from judgekit.core.runner import Runner
from judgekit.observability import get_metrics
from judgekit.providers.registry import close_provider, create_provider
from judgekit.storage import repository as repo
from judgekit.storage.database import session_scope

log = logging.getLogger(__name__)


def resolve_inside(directory: str | Path, name: str) -> Path:
    """Resolve ``name`` within ``directory``, refusing to escape it.

    The dataset and rubric names arrive over HTTP. Without this, ``"../../etc/
    passwd"`` is a file read, and a symlink or absolute path is the same hole
    wearing a hat. Resolving both sides and comparing is the check that actually
    holds, rather than string-matching for "..".
    """
    root = Path(directory).resolve()
    candidate = (root / name).resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{name!r} resolves outside {directory!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{name!r} not found in {directory!r}")
    return candidate


async def execute_run(
    *,
    run_id: str,
    dataset_path: Path,
    rubric_path: Path,
    provider_name: str,
    concurrency: int,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run a dataset end to end, recording progress in the database.

    Never raises. A failure is written to the run row as ``failed`` with its
    reason, because a job that vanishes without trace is far harder to diagnose
    than one that records why it stopped.
    """
    metrics = get_metrics()

    async with session_scope(session_factory) as session:
        await repo.mark_status(session, run_id, "running")

    provider = None
    try:
        dataset = Dataset.from_file(dataset_path)
        rubric = Rubric.from_file(rubric_path)
        provider = create_provider(provider_name)

        log.info(
            "run started",
            extra={
                "run_id": run_id,
                "dataset": dataset.name,
                "rubric": f"{rubric.id}@{rubric.version}",
                "fingerprint": rubric.fingerprint[:12],
                "provider": provider.name,
                "cases": len(dataset),
            },
        )

        with metrics.time_run():
            result = await Runner(Judge(rubric, provider), concurrency=concurrency).run(dataset)

        result = result.model_copy(update={"run_id": run_id})

        async with session_scope(session_factory) as session:
            await repo.save_run(session, result, status="completed")

        metrics.record_run(result, provider=provider.name)

        log.info(
            "run completed",
            extra={
                "run_id": run_id,
                "pass_rate": round(result.pass_rate, 4),
                "mean_score": round(result.mean_score, 3),
                "errored": result.errored,
                "cost_usd": round(result.usage.cost_usd, 6),
                "duration_s": round(result.duration_seconds, 2),
            },
        )

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.exception("run failed", extra={"run_id": run_id})
        metrics.runs_total.labels(status="failed").inc()

        async with session_scope(session_factory) as session:
            await repo.mark_status(session, run_id, "failed", error=reason)

    finally:
        if provider is not None:
            await close_provider(provider)
