"""Configuration, storage, observability and job dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from judgekit.config import Settings, get_settings
from judgekit.core.errors import IncomparableScoresError
from judgekit.core.judge import Judge
from judgekit.core.models import Dataset
from judgekit.core.rubric import Rubric
from judgekit.core.runner import Runner, RunResult, compare
from judgekit.observability import (
    JsonFormatter,
    Metrics,
    configure_logging,
    get_metrics,
    reset_metrics,
)
from judgekit.providers.stub import StubProvider
from judgekit.storage import repository as repo
from judgekit.storage.database import (
    create_engine,
    create_schema,
    create_session_factory,
    drop_schema,
    session_scope,
)
from judgekit.worker.queue import ARQ, INLINE, InlineQueue, create_queue
from judgekit.worker.tasks import execute_run, resolve_inside

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "datasets" / "example.jsonl"
V1_PATH = REPO_ROOT / "rubrics" / "answer-quality.v1.yaml"
V2_PATH = REPO_ROOT / "rubrics" / "answer-quality.v2.yaml"


@pytest.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    yield create_session_factory(engine)
    await drop_schema(engine)
    await engine.dispose()


async def make_run(rubric_path: Path = V2_PATH) -> RunResult:
    dataset = Dataset.from_file(DATASET_PATH)
    rubric = Rubric.from_file(rubric_path)
    return await Runner(Judge(rubric, StubProvider())).run(dataset)


class TestSettings:
    def test_defaults_need_nothing_installed(self) -> None:
        """SQLite, the stub provider and the inline queue: it runs standalone."""
        settings = Settings()
        assert settings.is_sqlite
        assert settings.provider == "stub"
        assert settings.queue == INLINE

    def test_reads_the_environment(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        monkeypatch.setenv("JUDGEKIT_QUEUE", "arq")
        monkeypatch.setenv("JUDGEKIT_CONCURRENCY", "16")
        monkeypatch.setenv("LOG_JSON", "false")
        monkeypatch.setenv("CORS_ORIGINS", "http://a, http://b ,")

        settings = Settings.from_env()

        assert not settings.is_sqlite
        assert settings.queue == "arq"
        assert settings.concurrency == 16
        assert settings.log_json is False
        assert settings.cors_origins == ("http://a", "http://b")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False)],
    )
    def test_boolean_parsing(self, monkeypatch: Any, raw: str, expected: bool) -> None:
        monkeypatch.setenv("LOG_JSON", raw)
        assert Settings.from_env().log_json is expected

    def test_rejects_a_nonsense_concurrency(self) -> None:
        with pytest.raises(ValueError, match="greater_than"):
            Settings(concurrency=0)

    def test_settings_are_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()
        get_settings.cache_clear()


class TestResolveInside:
    def test_resolves_a_plain_name(self) -> None:
        assert resolve_inside(REPO_ROOT / "rubrics", "answer-quality.v2.yaml").is_file()

    @pytest.mark.parametrize(
        "escape",
        ["../pyproject.toml", "../../etc/passwd", "sub/../../pyproject.toml"],
    )
    def test_refuses_to_escape_the_directory(self, escape: str) -> None:
        with pytest.raises(ValueError, match="resolves outside"):
            resolve_inside(REPO_ROOT / "rubrics", escape)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_inside(REPO_ROOT / "rubrics", "absent.yaml")

    def test_a_directory_is_not_a_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_inside(REPO_ROOT, "rubrics")


class TestRepository:
    async def test_saves_and_restores_a_run(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run = await make_run()

        async with session_scope(factory) as session:
            await repo.save_run(session, run)

        async with session_scope(factory) as session:
            row = await repo.get_run(session, run.run_id)
            assert row is not None
            restored = repo.to_result(row)

        assert restored.total == run.total
        assert restored.pass_rate == pytest.approx(run.pass_rate)
        assert len(restored.judgements) == len(run.judgements)

    async def test_the_fingerprint_survives_a_round_trip(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A stored run must still be guarded by compare().

        Once results outlive the rubric file that produced them, the fingerprint
        travelling with them is the only thing preventing a chart from splicing
        together scores graded under different rules.
        """
        run = await make_run()

        async with session_scope(factory) as session:
            await repo.save_run(session, run)

        async with session_scope(factory) as session:
            row = await repo.get_run(session, run.run_id)
            assert row is not None
            restored = repo.to_result(row)

        compare(run, restored)  # must not raise
        assert restored.rubric.fingerprint == run.rubric.fingerprint

    async def test_restored_runs_still_refuse_a_bad_comparison(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        v2, v1 = await make_run(V2_PATH), await make_run(V1_PATH)

        async with session_scope(factory) as session:
            await repo.save_run(session, v2)
            await repo.save_run(session, v1)

        async with session_scope(factory) as session:
            a = repo.to_result(await repo.get_run(session, v2.run_id))  # type: ignore[arg-type]
            b = repo.to_result(await repo.get_run(session, v1.run_id))  # type: ignore[arg-type]

        with pytest.raises(IncomparableScoresError, match="differs in version"):
            compare(a, b)

    async def test_preserves_flags_and_verdicts(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run = await make_run()

        async with session_scope(factory) as session:
            await repo.save_run(session, run)

        async with session_scope(factory) as session:
            restored = repo.to_result(await repo.get_run(session, run.run_id))  # type: ignore[arg-type]

        flagged = {j.case_id for j in restored.judgements if j.flags}
        assert "revenue-correct-but-unverifiable" in flagged

    async def test_reserve_then_save_replaces_the_placeholder(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run = await make_run()

        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id=run.run_id,
                dataset_name="example",
                rubric_id="answer-quality",
                rubric_version="v2",
                rubric_fingerprint=run.rubric.fingerprint,
                provider="stub",
            )

        async with session_scope(factory) as session:
            row = await repo.get_run(session, run.run_id)
            assert row is not None
            assert row.status == "queued"

        async with session_scope(factory) as session:
            await repo.save_run(session, run)

        async with session_scope(factory) as session:
            rows = await repo.list_runs(session)
            assert len(rows) == 1
            assert rows[0].status == "completed"

    async def test_mark_status_records_a_failure_reason(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id="abc",
                dataset_name="d",
                rubric_id="r",
                rubric_version="v1",
                rubric_fingerprint="f",
                provider="stub",
            )

        async with session_scope(factory) as session:
            await repo.mark_status(session, "abc", "failed", error="boom")

        async with session_scope(factory) as session:
            row = await repo.get_run(session, "abc")
            assert row is not None
            assert (row.status, row.error) == ("failed", "boom")
            assert row.finished_at is not None

    async def test_mark_status_on_an_unknown_run_is_a_no_op(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(factory) as session:
            await repo.mark_status(session, "ghost", "failed")

    async def test_comparable_history_excludes_other_rubrics(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        first, second, other = (
            await make_run(V2_PATH),
            await make_run(V2_PATH),
            await make_run(V1_PATH),
        )

        async with session_scope(factory) as session:
            for run in (first, second, other):
                await repo.save_run(session, run)

        async with session_scope(factory) as session:
            history = await repo.comparable_history(session, first.run_id)

        assert {r.rubric_version for r in history} == {"v2"}
        assert len(history) == 2

    async def test_comparable_history_of_an_unknown_run_is_empty(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(factory) as session:
            assert await repo.comparable_history(session, "ghost") == []

    async def test_delete_removes_judgements_too(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run = await make_run()

        async with session_scope(factory) as session:
            await repo.save_run(session, run)

        async with session_scope(factory) as session:
            assert await repo.delete_run(session, run.run_id) is True

        async with session_scope(factory) as session:
            assert await repo.get_run(session, run.run_id) is None

    async def test_deleting_an_unknown_run_reports_false(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(factory) as session:
            assert await repo.delete_run(session, "ghost") is False

    async def test_session_scope_rolls_back_on_error(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run = await make_run()

        async def save_then_fail() -> None:
            async with session_scope(factory) as session:
                await repo.save_run(session, run)
                raise RuntimeError("something went wrong")

        with pytest.raises(RuntimeError):
            await save_then_fail()

        async with session_scope(factory) as session:
            assert await repo.get_run(session, run.run_id) is None


class TestExecuteRun:
    async def test_records_a_completed_run(self, factory: async_sessionmaker[AsyncSession]) -> None:
        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id="r1",
                dataset_name="example",
                rubric_id="answer-quality",
                rubric_version="v2",
                rubric_fingerprint="x",
                provider="stub",
            )

        await execute_run(
            run_id="r1",
            dataset_path=DATASET_PATH,
            rubric_path=V2_PATH,
            provider_name="stub",
            concurrency=4,
            session_factory=factory,
        )

        async with session_scope(factory) as session:
            row = await repo.get_run(session, "r1")
            assert row is not None
            assert row.status == "completed"
            assert row.total == 8

    async def test_a_failure_is_recorded_rather_than_raised(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A job that vanishes is harder to diagnose than one that explains itself."""
        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id="r2",
                dataset_name="example",
                rubric_id="answer-quality",
                rubric_version="v2",
                rubric_fingerprint="x",
                provider="nonesuch",
            )

        await execute_run(
            run_id="r2",
            dataset_path=DATASET_PATH,
            rubric_path=V2_PATH,
            provider_name="nonesuch",
            concurrency=4,
            session_factory=factory,
        )

        async with session_scope(factory) as session:
            row = await repo.get_run(session, "r2")
            assert row is not None
            assert row.status == "failed"
            assert "nonesuch" in (row.error or "")


class TestQueue:
    def test_defaults_to_inline(self, factory: async_sessionmaker[AsyncSession]) -> None:
        queue = create_queue(INLINE, session_factory=factory, redis_url="redis://x")
        assert isinstance(queue, InlineQueue)

    def test_arq_is_selected_when_available(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        queue = create_queue(ARQ, session_factory=factory, redis_url="redis://x")
        assert not isinstance(queue, InlineQueue)

    async def test_inline_queue_runs_the_job(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id="q1",
                dataset_name="example",
                rubric_id="answer-quality",
                rubric_version="v2",
                rubric_fingerprint="x",
                provider="stub",
            )

        queue = InlineQueue(factory)
        await queue.enqueue(
            run_id="q1",
            dataset_path=DATASET_PATH,
            rubric_path=V2_PATH,
            provider_name="stub",
            concurrency=4,
        )
        await queue.wait_for_idle()

        async with session_scope(factory) as session:
            row = await repo.get_run(session, "q1")
            assert row is not None
            assert row.status == "completed"

    async def test_closing_an_idle_queue_is_safe(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        await InlineQueue(factory).aclose()


class TestObservability:
    def test_json_formatter_emits_one_object_per_line(self) -> None:
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        payload = json.loads(JsonFormatter().format(record))

        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "t"

    def test_extras_are_merged_as_fields(self) -> None:
        """The reason for JSON at all: a field name beats a regex over prose."""
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="run completed",
            args=(),
            exc_info=None,
        )
        record.run_id = "abc123"
        record.pass_rate = 0.92

        payload = json.loads(JsonFormatter().format(record))

        assert payload["run_id"] == "abc123"
        assert payload["pass_rate"] == 0.92

    def test_exceptions_are_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="t",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )

        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exception"]

    def test_configure_logging_replaces_handlers(self) -> None:
        configure_logging(level="DEBUG", json_output=True)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
        configure_logging(level="WARNING", json_output=False)
        assert len(logging.getLogger().handlers) == 1

    async def test_metrics_record_a_run(self) -> None:
        metrics = Metrics()
        run = await make_run()

        metrics.record_run(run, provider="stub")

        rendered = _render(metrics)
        assert 'judgekit_runs_total{status="completed"} 1.0' in rendered
        assert 'judgekit_cases_judged_total{verdict="pass"}' in rendered

    def test_timing_a_run_observes_the_histogram(self) -> None:
        metrics = Metrics()
        with metrics.time_run():
            pass
        assert "judgekit_run_duration_seconds_count 1.0" in _render(metrics)

    def test_each_metrics_instance_has_its_own_registry(self) -> None:
        """Otherwise every test would fight duplicate-registration errors."""
        assert Metrics().registry is not Metrics().registry

    def test_singleton_is_stable_until_reset(self) -> None:
        reset_metrics()
        first = get_metrics()
        assert get_metrics() is first
        reset_metrics()
        assert get_metrics() is not first


def _render(metrics: Metrics) -> str:
    from prometheus_client import generate_latest

    return generate_latest(metrics.registry).decode()


class TestWorkerEntrypoint:
    """The arq worker, exercised without Redis.

    ``run_evaluation`` and the lifecycle hooks are ordinary coroutines; only the
    broker connection needs Redis, and that is arq's code rather than ours.
    """

    async def test_startup_builds_a_session_factory_and_shutdown_releases_it(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        from judgekit.config import get_settings as settings_cache
        from judgekit.worker.main import shutdown, startup

        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
        monkeypatch.setenv("LOG_JSON", "false")
        settings_cache.cache_clear()

        ctx: dict[str, Any] = {}
        try:
            await startup(ctx)
            assert "session_factory" in ctx
            await shutdown(ctx)
        finally:
            settings_cache.cache_clear()

    async def test_run_evaluation_judges_and_decrements_the_queue_gauge(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from judgekit.worker.main import run_evaluation

        async with session_scope(factory) as session:
            await repo.reserve_run(
                session,
                run_id="w1",
                dataset_name="example",
                rubric_id="answer-quality",
                rubric_version="v2",
                rubric_fingerprint="x",
                provider="stub",
            )

        reset_metrics()
        get_metrics().queue_depth.inc()

        result = await run_evaluation(
            {"session_factory": factory}, "w1", str(DATASET_PATH), str(V2_PATH), "stub", 4
        )

        assert result == "w1"
        assert "judgekit_queue_depth 0.0" in _render(get_metrics())

        async with session_scope(factory) as session:
            row = await repo.get_run(session, "w1")
            assert row is not None
            assert row.status == "completed"

    async def test_shutdown_without_an_engine_is_safe(self) -> None:
        from judgekit.worker.main import shutdown

        await shutdown({})

    def test_worker_runs_one_job_per_pod(self) -> None:
        """Scale out with pods, not job slots.

        Each run already fans out internally against the provider's rate limit,
        so stacking runs inside one pod would multiply concurrency in a way
        nothing accounts for.
        """
        from judgekit.worker.main import WorkerSettings

        assert WorkerSettings.max_jobs == 1
        assert WorkerSettings.functions[0].__name__ == "run_evaluation"

    def test_redis_settings_is_a_value_not_a_descriptor(self) -> None:
        """arq reads ``__dict__``, so a staticmethod here is never called.

        This shipped broken once: ``redis_settings`` was a ``@staticmethod``,
        which arq passed through verbatim, and the worker died at startup on
        ``'staticmethod' object has no attribute 'host'``. Nothing caught it,
        because the API kept accepting runs - they simply queued forever with
        no consumer. Asserting the *value* rather than the callable is the
        whole point of this test.
        """
        from arq.connections import RedisSettings

        from judgekit.worker.main import WorkerSettings

        assert isinstance(WorkerSettings.__dict__["redis_settings"], RedisSettings)

    def test_every_worker_setting_reaches_arq(self) -> None:
        """A name arq's ``Worker`` does not accept is silently discarded.

        ``get_kwargs`` keeps only the keys matching ``Worker``'s signature, so a
        typo or a renamed upstream parameter drops the setting without warning -
        ``max_jobs`` quietly reverting to arq's default of 10 would multiply
        concurrency against the provider's rate limit.
        """
        import inspect

        from arq.worker import Worker

        from judgekit.worker.main import WorkerSettings

        accepted = set(inspect.signature(Worker).parameters)
        declared = {k for k in WorkerSettings.__dict__ if not k.startswith("_")}
        assert declared <= accepted, f"ignored by arq: {sorted(declared - accepted)}"
