"""The HTTP API.

Three things here are load-bearing rather than boilerplate.

**``/healthz`` and ``/readyz`` are different.** Liveness answers "is this
process alive"; readiness answers "can it serve traffic", which means checking
the database. Wiring both to the same handler is the classic way to get
Kubernetes to restart a healthy pod because its database was briefly slow, or
to route traffic at a pod that cannot answer.

**Dataset and rubric names are resolved, not concatenated.** They arrive over
HTTP and name files on disk, so path traversal is the obvious hole.

**Comparison is refused, not fudged.** ``POST /compare`` returns 409 when two
runs were graded under different rubrics, because a diff across a rubric change
is worse than no diff.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from judgekit import __version__
from judgekit.api.schemas import (
    ComparisonOut,
    HealthOut,
    ReadyOut,
    RunDetail,
    RunList,
    RunSummary,
    SubmitRunRequest,
    SubmitRunResponse,
)
from judgekit.config import Settings, get_settings
from judgekit.core.errors import IncomparableScoresError, JudgekitError
from judgekit.core.rubric import Rubric
from judgekit.core.runner import compare
from judgekit.observability import configure_logging, get_metrics
from judgekit.storage import repository as repo
from judgekit.storage.database import (
    create_engine,
    create_schema,
    create_session_factory,
    session_scope,
)
from judgekit.worker.queue import INLINE, create_queue
from judgekit.worker.tasks import resolve_inside

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)

    if settings.is_sqlite:
        # Convenience for local runs and tests. Postgres migrates with Alembic;
        # create_all against a real database would diverge from the migrations.
        await create_schema(engine)

    queue = create_queue(
        getattr(app.state, "queue_kind", INLINE),
        session_factory=factory,
        redis_url=settings.redis_url,
    )

    app.state.engine = engine
    app.state.session_factory = factory
    app.state.queue = queue

    log.info("api started", extra={"database": settings.database_url.split("://")[0]})
    try:
        yield
    finally:
        await queue.aclose()
        await engine.dispose()


def get_session_factory(request: Request) -> Any:
    return request.app.state.session_factory


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with session_scope(request.app.state.session_factory) as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def create_app(settings: Settings | None = None, *, queue_kind: str | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Defaults to the process environment.
        queue_kind: ``inline`` or ``arq``. Defaults to the configured queue,
            which itself defaults to inline so the API runs standalone.
    """
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.api_title,
        version=__version__,
        summary="Measure the judge, not just the model.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.queue_kind = queue_kind or resolved.queue

    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.get("/healthz", response_model=HealthOut, tags=["ops"])
    async def healthz() -> HealthOut:
        """Liveness. Deliberately touches nothing external.

        If this checked the database, a slow query would get a healthy process
        killed and restarted, which helps nobody.
        """
        return HealthOut(status="ok", version=__version__)

    @app.get("/readyz", response_model=ReadyOut, tags=["ops"])
    async def readyz(response: Response, request: Request) -> ReadyOut:
        """Readiness. Can this process actually serve a request?"""
        try:
            async with session_scope(request.app.state.session_factory) as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyOut(
                status="unavailable", database="down", detail=f"{type(exc).__name__}: {exc}"
            )
        return ReadyOut(status="ready", database="up")

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=generate_latest(get_metrics().registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post(
        "/runs",
        response_model=SubmitRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["runs"],
    )
    async def submit_run(
        payload: SubmitRunRequest, request: Request, session: SessionDep
    ) -> SubmitRunResponse:
        """Queue a dataset for judging.

        Returns 202 with a run id. The rubric is loaded here rather than in the
        worker so a bad request fails immediately, and so the fingerprint is
        recorded before any work starts.
        """
        settings: Settings = request.app.state.settings

        try:
            dataset_path = resolve_inside(settings.dataset_dir, payload.dataset)
            rubric_path = resolve_inside(settings.rubric_dir, payload.rubric)
            rubric = Rubric.from_file(rubric_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except JudgekitError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        run_id = uuid.uuid4().hex[:12]
        provider_name = payload.provider or settings.provider

        await repo.reserve_run(
            session,
            run_id=run_id,
            dataset_name=dataset_path.stem,
            rubric_id=rubric.id,
            rubric_version=rubric.version,
            rubric_fingerprint=rubric.fingerprint,
            provider=provider_name,
        )

        await request.app.state.queue.enqueue(
            run_id=run_id,
            dataset_path=dataset_path,
            rubric_path=rubric_path,
            provider_name=provider_name,
            concurrency=payload.concurrency or settings.concurrency,
        )

        get_metrics().queue_depth.inc()
        return SubmitRunResponse(run_id=run_id, status="queued")

    @app.get("/runs", response_model=RunList, tags=["runs"])
    async def list_runs(
        session: SessionDep,
        rubric_fingerprint: Annotated[str | None, Query()] = None,
        dataset: Annotated[str | None, Query()] = None,
        run_status: Annotated[str | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> RunList:
        """Recent runs, newest first.

        Filter by ``rubric_fingerprint`` to get a set that is safe to chart
        together.
        """
        rows = await repo.list_runs(
            session,
            rubric_fingerprint=rubric_fingerprint,
            dataset_name=dataset,
            status=run_status,
            limit=limit,
            offset=offset,
        )
        summaries = tuple(RunSummary.from_row(r) for r in rows)
        return RunList(runs=summaries, count=len(summaries))

    @app.get("/runs/{run_id}", response_model=RunDetail, tags=["runs"])
    async def get_run(run_id: str, session: SessionDep) -> RunDetail:
        row = await repo.get_run(session, run_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {run_id!r}")
        return RunDetail.from_row(row)

    @app.get("/runs/{run_id}/history", response_model=RunList, tags=["runs"])
    async def run_history(run_id: str, session: SessionDep) -> RunList:
        """Runs that can legitimately be plotted alongside this one.

        Same dataset, same rubric fingerprint. Anything else belongs on a
        different chart, and returning it here would invite exactly the
        cross-rubric trend line this project exists to prevent.
        """
        if await repo.get_run(session, run_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {run_id!r}")

        rows = await repo.comparable_history(session, run_id)
        summaries = tuple(RunSummary.from_row(r) for r in rows)
        return RunList(runs=summaries, count=len(summaries))

    @app.post("/compare", response_model=ComparisonOut, tags=["runs"])
    async def compare_runs(
        session: SessionDep,
        baseline: Annotated[str, Query()],
        candidate: Annotated[str, Query()],
        max_drop: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    ) -> ComparisonOut:
        """Diff two stored runs.

        Returns 409 when they are not comparable. That is a refusal, not an
        error: the runs exist and the request was well formed, but answering it
        would produce a number that looks like evidence and is not.
        """
        rows = {}
        for key, run_id in (("baseline", baseline), ("candidate", candidate)):
            row = await repo.get_run(session, run_id)
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {run_id!r}")
            rows[key] = row

        try:
            result = compare(repo.to_result(rows["baseline"]), repo.to_result(rows["candidate"]))
        except IncomparableScoresError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

        return ComparisonOut(
            baseline_run_id=result.baseline_run_id,
            candidate_run_id=result.candidate_run_id,
            pass_rate_before=result.pass_rate_before,
            pass_rate_after=result.pass_rate_after,
            pass_rate_delta=result.pass_rate_delta,
            mean_score_before=result.mean_score_before,
            mean_score_after=result.mean_score_after,
            regressions=tuple(d.case_id for d in result.regressions),
            improvements=tuple(d.case_id for d in result.improvements),
            gate_passed=result.gate(max_pass_rate_drop=max_drop),
        )

    @app.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["runs"])
    async def delete_run(run_id: str, session: SessionDep) -> Response:
        if not await repo.delete_run(session, run_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {run_id!r}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
