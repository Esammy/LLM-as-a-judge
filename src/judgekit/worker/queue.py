"""Dispatching work, either to a worker pool or in-process.

Two implementations behind one protocol. The default is ``inline``, for the
same reason the default provider is the stub: ``judgekit-api`` should start and
work with nothing else installed. Someone evaluating the project should not
have to stand up Redis before seeing a run complete.

``arq`` is what production uses, and it is the reason the worker pool scales
horizontally - which is in turn the honest justification for the Kubernetes
manifests. Judging is slow, I/O-bound and embarrassingly parallel; that is a
real case for more pods, not a decorative one.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from judgekit.worker.tasks import execute_run

log = logging.getLogger(__name__)

INLINE = "inline"
ARQ = "arq"


@runtime_checkable
class JobQueue(Protocol):
    """Anything that can get a run executed."""

    async def enqueue(
        self,
        *,
        run_id: str,
        dataset_path: Path,
        rubric_path: Path,
        provider_name: str,
        concurrency: int,
    ) -> None: ...

    async def aclose(self) -> None: ...


class InlineQueue:
    """Runs the job in this process, as a background task.

    Fine for development, a single-container deployment, and the test suite.
    Not fine for anything that needs the work to survive a restart: a process
    that dies takes its in-flight runs with it, which is exactly what the arq
    queue exists to fix.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._tasks: set[asyncio.Task[None]] = set()

    async def enqueue(
        self,
        *,
        run_id: str,
        dataset_path: Path,
        rubric_path: Path,
        provider_name: str,
        concurrency: int,
    ) -> None:
        task = asyncio.create_task(
            execute_run(
                run_id=run_id,
                dataset_path=dataset_path,
                rubric_path=rubric_path,
                provider_name=provider_name,
                concurrency=concurrency,
                session_factory=self._session_factory,
            )
        )
        # Held deliberately: asyncio keeps only a weak reference to a running
        # task, so without this the garbage collector can cancel a job
        # mid-flight for no visible reason.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_idle(self, timeout: float = 30.0) -> None:
        """Block until every in-flight job finishes. Tests and shutdown."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)

    async def aclose(self) -> None:
        await self.wait_for_idle(timeout=5.0)
        for task in list(self._tasks):
            task.cancel()


class ArqQueue:
    """Hands the job to an arq worker pool over Redis."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: Any = None

    async def _connect(self) -> Any:
        if self._pool is None:
            from arq import create_pool
            from arq.connections import RedisSettings

            self._pool = await create_pool(RedisSettings.from_dsn(self._redis_url))
        return self._pool

    async def enqueue(
        self,
        *,
        run_id: str,
        dataset_path: Path,
        rubric_path: Path,
        provider_name: str,
        concurrency: int,
    ) -> None:
        pool = await self._connect()
        await pool.enqueue_job(
            "run_evaluation",
            run_id,
            str(dataset_path),
            str(rubric_path),
            provider_name,
            concurrency,
            _job_id=run_id,  # idempotent: a resubmitted run_id will not double-run
        )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None


def create_queue(
    kind: str,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    redis_url: str,
) -> JobQueue:
    """Build the configured queue, falling back to inline if arq is absent."""
    if kind.strip().lower() == ARQ:
        try:
            import arq  # noqa: F401
        except ImportError:
            log.warning(
                "arq is not installed; falling back to the inline queue. "
                "Install the 'service' extra for the worker pool."
            )
        else:
            return ArqQueue(redis_url)

    return InlineQueue(session_factory)
