"""Run a whole dataset through a judge, and compare one run against another.

Two things here are load-bearing beyond the obvious orchestration.

**Bounded concurrency.** Judging is almost entirely waiting on a network call,
so the runner fans out. It fans out through a semaphore rather than an
unbounded ``gather`` because every real provider rate-limits, and an unbounded
fan-out converts a fast run into a cascade of 429s.

**Comparison is guarded.** :func:`compare` refuses to diff two runs graded
under different rubrics. That refusal is the whole point of the project: a
regression number computed across a rubric edit is worse than no number, since
it looks like evidence.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from judgekit.core.judge import Judge, Judgement
from judgekit.core.models import Dataset, Verdict
from judgekit.core.rubric import RubricRef, assert_comparable
from judgekit.providers.base import Usage

ProgressHook = Callable[[Judgement], None]


class DomainStats(BaseModel):
    """Aggregates for one slice of a run."""

    model_config = ConfigDict(frozen=True)

    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    mean_score: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Errors count against the pass rate; a run that crashed did not pass."""
        return self.passed / self.total if self.total else 0.0


class RunResult(BaseModel):
    """Everything produced by judging one dataset once."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    dataset_name: str
    dataset_version: str
    rubric: RubricRef
    provider: str
    model: str
    judgements: tuple[Judgement, ...] = ()
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def total(self) -> int:
        return len(self.judgements)

    @property
    def passed(self) -> int:
        return sum(1 for j in self.judgements if j.verdict is Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for j in self.judgements if j.verdict is Verdict.FAIL)

    @property
    def errored(self) -> int:
        return sum(1 for j in self.judgements if j.verdict is Verdict.ERROR)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """Mean over judgements that produced a real score.

        Errors are excluded rather than counted as zero: an unparseable response
        is a missing measurement, not a bad one, and folding it in as zero would
        make an infrastructure problem look like a quality problem.
        """
        scored = [j.score for j in self.judgements if j.ok]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def usage(self) -> Usage:
        total = Usage()
        for j in self.judgements:
            total = total + j.usage
        return total

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def by_domain(self, domains: dict[str, str | None]) -> dict[str, DomainStats]:
        """Break the run down by case domain.

        Args:
            domains: Map of case id to its domain, since judgements carry only
                the case id.
        """
        buckets: dict[str, list[Judgement]] = defaultdict(list)
        for j in self.judgements:
            buckets[domains.get(j.case_id) or "unassigned"].append(j)

        stats: dict[str, DomainStats] = {}
        for domain, items in sorted(buckets.items()):
            scored = [i.score for i in items if i.ok]
            stats[domain] = DomainStats(
                total=len(items),
                passed=sum(1 for i in items if i.verdict is Verdict.PASS),
                failed=sum(1 for i in items if i.verdict is Verdict.FAIL),
                errored=sum(1 for i in items if i.verdict is Verdict.ERROR),
                mean_score=sum(scored) / len(scored) if scored else 0.0,
            )
        return stats


class Runner:
    """Executes a dataset against a judge with bounded concurrency.

    Args:
        judge: The bound rubric-and-provider pair to grade with.
        concurrency: How many cases may be in flight at once.
        retries: How many times to retry a case that comes back as an error.
            Transient provider failures are common enough that not retrying
            turns a flaky network into a failed build.
        backoff_seconds: Base delay for exponential backoff between retries.
    """

    def __init__(
        self,
        judge: Judge,
        *,
        concurrency: int = 8,
        retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.judge = judge
        self.concurrency = concurrency
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    async def _judge_with_retry(self, case_index: int, dataset: Dataset) -> Judgement:
        case = dataset.cases[case_index]
        judgement = await self.judge.judge(case)

        attempt = 0
        while judgement.verdict is Verdict.ERROR and attempt < self.retries:
            await asyncio.sleep(self.backoff_seconds * (2**attempt))
            judgement = await self.judge.judge(case)
            attempt += 1

        return judgement

    async def run(self, dataset: Dataset, *, on_result: ProgressHook | None = None) -> RunResult:
        """Judge every case, preserving dataset order in the result."""
        started = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(index: int) -> Judgement:
            async with semaphore:
                judgement = await self._judge_with_retry(index, dataset)
                if on_result is not None:
                    on_result(judgement)
                return judgement

        judgements = await asyncio.gather(*(one(i) for i in range(len(dataset.cases))))

        return RunResult(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            rubric=self.judge.rubric.ref,
            provider=self.judge.provider.name,
            model=self.judge.provider.model,
            judgements=tuple(judgements),
            started_at=started,
            finished_at=datetime.now(UTC),
        )


class CaseDelta(BaseModel):
    """How one case moved between two runs."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    before: float
    after: float
    before_verdict: Verdict
    after_verdict: Verdict

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def regressed(self) -> bool:
        return self.before_verdict is Verdict.PASS and self.after_verdict is not Verdict.PASS

    @property
    def improved(self) -> bool:
        return self.before_verdict is not Verdict.PASS and self.after_verdict is Verdict.PASS


class Comparison(BaseModel):
    """The difference between a baseline run and a candidate run."""

    model_config = ConfigDict(frozen=True)

    baseline_run_id: str
    candidate_run_id: str
    pass_rate_before: float
    pass_rate_after: float
    mean_score_before: float
    mean_score_after: float
    deltas: tuple[CaseDelta, ...] = ()
    only_in_baseline: tuple[str, ...] = ()
    only_in_candidate: tuple[str, ...] = ()

    @property
    def pass_rate_delta(self) -> float:
        return self.pass_rate_after - self.pass_rate_before

    @property
    def regressions(self) -> tuple[CaseDelta, ...]:
        return tuple(d for d in self.deltas if d.regressed)

    @property
    def improvements(self) -> tuple[CaseDelta, ...]:
        return tuple(d for d in self.deltas if d.improved)

    def gate(self, *, max_pass_rate_drop: float = 0.0) -> bool:
        """Whether this comparison should be allowed through a CI gate."""
        return self.pass_rate_delta >= -max_pass_rate_drop


def compare(baseline: RunResult, candidate: RunResult) -> Comparison:
    """Diff two runs, refusing when they are not legitimately comparable.

    Raises:
        IncomparableScoresError: If the two runs were graded under different
            rubrics, different rubric versions, or the same version with
            different content.
    """
    assert_comparable(baseline.rubric, candidate.rubric)

    before = {j.case_id: j for j in baseline.judgements}
    after = {j.case_id: j for j in candidate.judgements}
    shared = sorted(before.keys() & after.keys())

    deltas = tuple(
        CaseDelta(
            case_id=case_id,
            before=before[case_id].score,
            after=after[case_id].score,
            before_verdict=before[case_id].verdict,
            after_verdict=after[case_id].verdict,
        )
        for case_id in shared
    )

    return Comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        pass_rate_before=baseline.pass_rate,
        pass_rate_after=candidate.pass_rate,
        mean_score_before=baseline.mean_score,
        mean_score_after=candidate.mean_score,
        deltas=deltas,
        only_in_baseline=tuple(sorted(before.keys() - after.keys())),
        only_in_candidate=tuple(sorted(after.keys() - before.keys())),
    )
