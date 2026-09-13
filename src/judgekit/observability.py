"""Structured logging and Prometheus metrics.

Logs are JSON by default because a container's stdout is read by a machine
before it is read by a person, and a regex over free text is a worse contract
than a field name.

The metrics are chosen to answer the questions someone actually asks at 3am:
is the queue moving, is the judge erroring, and how much is this costing. A
metric nobody would page on is a metric worth leaving out.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)  # fmt: skip


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extras merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Anything passed via `extra=` lands as an attribute on the record.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install a single stdout handler, replacing any existing ones."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty and say nothing a request log does not.
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Metrics:
    """The Prometheus collectors, on their own registry.

    A private registry rather than the global default, so tests can build a
    fresh one per case instead of fighting duplicate-registration errors.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.runs_total = Counter(
            "judgekit_runs_total",
            "Runs finished, by outcome.",
            ["status"],
            registry=self.registry,
        )
        self.run_duration = Histogram(
            "judgekit_run_duration_seconds",
            "Wall-clock duration of a run.",
            buckets=(0.5, 1, 5, 15, 60, 300, 900),
            registry=self.registry,
        )
        self.cases_judged = Counter(
            "judgekit_cases_judged_total",
            "Cases judged, by verdict.",
            ["verdict"],
            registry=self.registry,
        )
        self.judge_errors = Counter(
            "judgekit_judge_errors_total",
            "Cases that produced no usable judgement, by provider.",
            ["provider"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "judgekit_tokens_total",
            "Tokens consumed, by provider and direction.",
            ["provider", "direction"],
            registry=self.registry,
        )
        self.cost = Counter(
            "judgekit_cost_usd_total",
            "Estimated spend, by provider.",
            ["provider"],
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "judgekit_queue_depth",
            "Runs queued or running. The worker HPA scales on this.",
            registry=self.registry,
        )

    @contextmanager
    def time_run(self) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.run_duration.observe(time.perf_counter() - started)

    def record_run(self, result: Any, *, provider: str) -> None:
        """Fold a finished :class:`RunResult` into the counters."""
        self.runs_total.labels(status="completed").inc()
        self.cases_judged.labels(verdict="pass").inc(result.passed)
        self.cases_judged.labels(verdict="fail").inc(result.failed)
        self.cases_judged.labels(verdict="error").inc(result.errored)

        if result.errored:
            self.judge_errors.labels(provider=provider).inc(result.errored)

        usage = result.usage
        self.tokens.labels(provider=provider, direction="prompt").inc(usage.prompt_tokens)
        self.tokens.labels(provider=provider, direction="completion").inc(usage.completion_tokens)
        if usage.cost_usd:
            self.cost.labels(provider=provider).inc(usage.cost_usd)


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """The process-wide metrics singleton."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def reset_metrics() -> None:
    """Drop the singleton. Tests only."""
    global _metrics
    _metrics = None
