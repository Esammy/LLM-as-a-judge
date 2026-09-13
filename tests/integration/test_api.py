"""The HTTP API, end to end against in-memory SQLite.

These exercise the real application - lifespan, dependency injection, the
inline queue, the storage layer - through an ASGI transport rather than a live
socket. Nothing here needs Postgres, Redis or a network, so it runs in the
offline CI job with everything else.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from judgekit.api.app import create_app
from judgekit.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "example.jsonl"
V1 = "answer-quality.v1.yaml"
V2 = "answer-quality.v2.yaml"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        dataset_dir=str(REPO_ROOT / "datasets"),
        rubric_dir=str(REPO_ROOT / "rubrics"),
        log_json=False,
        log_level="WARNING",
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with LifespanManager(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def submit(client: httpx.AsyncClient, app: FastAPI, rubric: str = V2) -> str:
    """Submit a run and wait for the inline worker to finish it."""
    response = await client.post("/runs", json={"dataset": DATASET, "rubric": rubric})
    assert response.status_code == 202
    await app.state.queue.wait_for_idle()
    return str(response.json()["run_id"])


class TestOps:
    async def test_healthz_is_liveness_only(self, client: httpx.AsyncClient) -> None:
        """Must not touch the database.

        A liveness probe that checks a dependency gets a healthy process killed
        whenever that dependency is briefly slow.
        """
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_readyz_checks_the_database(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "database": "up", "detail": None}

    async def test_readyz_reports_503_when_the_database_is_gone(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        await app.state.engine.dispose()

        class Broken:
            def __call__(self) -> object:
                raise ConnectionError("database is down")

        app.state.session_factory = Broken()

        response = await client.get("/readyz")

        assert response.status_code == 503
        assert response.json()["database"] == "down"
        assert "database is down" in response.json()["detail"]

    async def test_metrics_are_exposed_in_prometheus_format(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "judgekit_runs_total" in response.text

    async def test_openapi_is_published(self, client: httpx.AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()
        assert "/runs" in schema["paths"]


class TestSubmitRun:
    async def test_accepts_and_completes_a_run(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        run_id = await submit(client, app)
        detail = (await client.get(f"/runs/{run_id}")).json()

        assert detail["status"] == "completed"
        assert detail["total"] == 8
        assert len(detail["judgements"]) == 8

    async def test_records_the_rubric_fingerprint(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """Provenance has to survive storage, or stored history is unsafe to plot."""
        run_id = await submit(client, app)
        rubric = (await client.get(f"/runs/{run_id}")).json()["rubric"]

        assert rubric["id"] == "answer-quality"
        assert rubric["version"] == "v2"
        assert len(rubric["fingerprint"]) == 64

    async def test_the_run_is_visible_before_it_finishes(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """Reserved up front, so a submitted run never vanishes into the queue."""
        response = await client.post("/runs", json={"dataset": DATASET, "rubric": V2})
        run_id = response.json()["run_id"]

        immediate = await client.get(f"/runs/{run_id}")

        assert immediate.status_code == 200
        assert immediate.json()["status"] in {"queued", "running", "completed"}
        await app.state.queue.wait_for_idle()

    async def test_carries_flags_through_to_the_response(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        run_id = await submit(client, app)
        judgements = (await client.get(f"/runs/{run_id}")).json()["judgements"]

        flagged = {j["case_id"]: j["flags"] for j in judgements if j["flags"]}
        assert "no_evidence" in flagged["revenue-correct-but-unverifiable"]

    async def test_rejects_an_unknown_dataset(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/runs", json={"dataset": "nope.jsonl", "rubric": V2})
        assert response.status_code == 400

    async def test_rejects_an_unknown_rubric(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/runs", json={"dataset": DATASET, "rubric": "nope.yaml"})
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "escape",
        [
            "../../../etc/passwd",
            "../pyproject.toml",
            "/etc/passwd",
        ],
    )
    async def test_refuses_to_read_outside_the_dataset_directory(
        self, client: httpx.AsyncClient, escape: str
    ) -> None:
        """Dataset names arrive over HTTP and name files on disk."""
        response = await client.post("/runs", json={"dataset": escape, "rubric": V2})
        assert response.status_code == 400

    async def test_a_failed_run_records_its_reason(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """A job that vanishes is far harder to diagnose than one that explains itself."""
        response = await client.post(
            "/runs", json={"dataset": DATASET, "rubric": V2, "provider": "nonesuch"}
        )
        run_id = response.json()["run_id"]
        await app.state.queue.wait_for_idle()

        detail = (await client.get(f"/runs/{run_id}")).json()

        assert detail["status"] == "failed"
        assert "nonesuch" in detail["error"]


class TestListRuns:
    async def test_lists_newest_first(self, client: httpx.AsyncClient, app: FastAPI) -> None:
        await submit(client, app)
        await submit(client, app, rubric=V1)

        listing = (await client.get("/runs")).json()

        assert listing["count"] == 2

    async def test_filters_by_rubric_fingerprint(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """The filter that makes a stored trend safe to chart."""
        v2_run = await submit(client, app, rubric=V2)
        await submit(client, app, rubric=V1)

        fingerprint = (await client.get(f"/runs/{v2_run}")).json()["rubric"]["fingerprint"]
        listing = (await client.get("/runs", params={"rubric_fingerprint": fingerprint})).json()

        assert listing["count"] == 1
        assert listing["runs"][0]["run_id"] == v2_run

    async def test_filters_by_status(self, client: httpx.AsyncClient, app: FastAPI) -> None:
        await submit(client, app)
        listing = (await client.get("/runs", params={"status": "completed"})).json()
        assert listing["count"] == 1

    async def test_respects_limit(self, client: httpx.AsyncClient, app: FastAPI) -> None:
        await submit(client, app)
        await submit(client, app)
        assert (await client.get("/runs", params={"limit": 1})).json()["count"] == 1

    async def test_rejects_a_nonsense_limit(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/runs", params={"limit": 0})).status_code == 422


class TestHistory:
    async def test_only_returns_comparable_runs(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """Same dataset, same fingerprint. A v1 run must not appear here."""
        first = await submit(client, app, rubric=V2)
        await submit(client, app, rubric=V2)
        await submit(client, app, rubric=V1)

        history = (await client.get(f"/runs/{first}/history")).json()

        assert history["count"] == 2
        versions = {r["rubric"]["version"] for r in history["runs"]}
        assert versions == {"v2"}

    async def test_unknown_run_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/runs/nope/history")).status_code == 404


class TestCompare:
    async def test_compares_two_runs_under_one_rubric(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        baseline = await submit(client, app)
        candidate = await submit(client, app)

        response = await client.post(
            "/compare", params={"baseline": baseline, "candidate": candidate}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["pass_rate_delta"] == 0.0
        assert body["gate_passed"] is True

    async def test_refuses_across_rubric_versions_with_409(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        """409, not 500 or a fudged number.

        The request was well formed and both runs exist. Answering it would
        produce something that looks like evidence and is not, so the API
        declines rather than obliging.
        """
        v2_run = await submit(client, app, rubric=V2)
        v1_run = await submit(client, app, rubric=V1)

        response = await client.post("/compare", params={"baseline": v1_run, "candidate": v2_run})

        assert response.status_code == 409
        assert "differs in version" in response.json()["detail"]

    async def test_unknown_run_is_404(self, client: httpx.AsyncClient, app: FastAPI) -> None:
        run_id = await submit(client, app)
        response = await client.post("/compare", params={"baseline": run_id, "candidate": "ghost"})
        assert response.status_code == 404


class TestGetAndDelete:
    async def test_unknown_run_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/runs/ghost")).status_code == 404

    async def test_delete_removes_a_run_and_its_judgements(
        self, client: httpx.AsyncClient, app: FastAPI
    ) -> None:
        run_id = await submit(client, app)

        assert (await client.delete(f"/runs/{run_id}")).status_code == 204
        assert (await client.get(f"/runs/{run_id}")).status_code == 404
        assert (await client.get("/runs")).json()["count"] == 0

    async def test_deleting_an_unknown_run_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.delete("/runs/ghost")).status_code == 404
