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

from arq.connections import RedisSettings

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


def _redis_settings() -> RedisSettings:
    """Parse ``REDIS_URL`` into arq's connection settings.

    Pure parsing - this opens no socket, which is what makes it safe to call
    while the class body below is being evaluated.
    """
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """arq configuration. Referenced by path on the command line.

    Every value here must be a **plain class attribute**. ``arq`` collects its
    configuration with ``settings_cls.__dict__``, not ``getattr``, so anything
    behind a descriptor - a ``staticmethod``, a ``classmethod``, a ``property``,
    or a ``property`` on a metaclass - is handed to the worker as the descriptor
    object itself rather than the value it would return.

    That failure is silent until the worker dials Redis and something deep in
    ``arq.connections`` asks the "settings" for a ``.host`` it does not have. The
    process then dies at startup, having consumed nothing, while the API happily
    keeps accepting runs that queue forever. ``test_redis_settings_is_a_value``
    pins the type so that stays fixed.
    """

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

    # Evaluated once, when this module is imported - which for the worker is
    # process start, after the environment is set. It cannot be deferred: see
    # the class docstring.
    redis_settings = _redis_settings()
