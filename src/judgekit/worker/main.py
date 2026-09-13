"""The arq worker entrypoint.

Run with::

    arq judgekit.worker.main.WorkerSettings

Each pod is stateless: it pulls a job, judges a dataset, writes the result and
takes the next one. That is what makes the pool horizontally scalable, and it
is the reason the Kubernetes manifests carry an HPA rather than a fixed replica
count - judging is slow, I/O-bound and embarrassingly parallel, so a deeper
queue really is answered by more pods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from judgekit.config import get_settings
from judgekit.observability import configure_logging, get_metrics
from judgekit.storage.database import (
    create_engine,
    create_schema,
    create_session_factory,
)
from judgekit.worker.tasks import execute_run

log = logging.getLogger(__name__)


async def run_evaluation(
    ctx: dict[str, Any],
    run_id: str,
    dataset_path: str,
    rubric_path: str,
    provider_name: str,
    concurrency: int,
) -> str:
    """Judge one dataset. The only job this worker knows how to do."""
    await execute_run(
        run_id=run_id,
        dataset_path=Path(dataset_path),
        rubric_path=Path(rubric_path),
        provider_name=provider_name,
        concurrency=concurrency,
        session_factory=ctx["session_factory"],
    )
    get_metrics().queue_depth.dec()
    return run_id


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    engine = create_engine(settings.database_url)
    if settings.is_sqlite:
        await create_schema(engine)

    ctx["engine"] = engine
    ctx["session_factory"] = create_session_factory(engine)
    log.info("worker ready", extra={"provider": settings.provider})


async def shutdown(ctx: dict[str, Any]) -> None:
    engine = ctx.get("engine")
    if engine is not None:
        await engine.dispose()
    log.info("worker stopped")


def _redis_settings() -> Any:
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """arq configuration. Referenced by path on the command line."""

    functions: ClassVar[list[Any]] = [run_evaluation]
    on_startup = startup
    on_shutdown = shutdown

    # One job at a time per pod. Each run already fans out internally via the
    # runner's semaphore, so stacking runs inside a pod would multiply
    # concurrency against the provider's rate limit in a way nothing accounts
    # for. Scale out with pods, not with job slots.
    max_jobs = 1

    job_timeout = 3600
    keep_result = 3600
    max_tries = 2

    @staticmethod
    def redis_settings() -> Any:  # pragma: no cover - needs a live Redis
        return _redis_settings()
