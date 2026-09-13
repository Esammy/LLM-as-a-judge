"""Running a dataset, and diffing one run against another."""

from __future__ import annotations

import asyncio

import pytest

from judgekit.core.errors import IncomparableScoresError
from judgekit.core.judge import Judge, Judgement
from judgekit.core.models import Case, Dataset, Verdict
from judgekit.core.rubric import Rubric
from judgekit.core.runner import Runner, compare
from judgekit.providers.base import Completion, CompletionRequest, Usage
from judgekit.providers.stub import StubProvider


class FlakyProvider:
    """Fails a set number of times, then succeeds. Exercises retry."""

    name = "flaky"
    model = "flaky-v1"
    family = "flaky"

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.attempts = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise TimeoutError("transient")
        return Completion(text='{"score": 5}', model=self.model, usage=Usage())


class SlowProvider:
    """Records peak concurrency, to prove the semaphore actually bounds it."""

    name = "slow"
    model = "slow-v1"
    family = "slow"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return Completion(text='{"score": 5}', model=self.model, usage=Usage())
        finally:
            self.in_flight -= 1


def many_cases(count: int) -> Dataset:
    return Dataset(
        name="many",
        cases=tuple(
            Case(id=f"case-{i}", input="q", output="a", reference="a", domain="d")
            for i in range(count)
        ),
    )


class TestRunner:
    async def test_judges_every_case(self, rubric: Rubric, dataset: Dataset) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        assert result.total == 3
        assert result.errored == 0

    async def test_preserves_dataset_order(self, rubric: Rubric) -> None:
        """Concurrency must not shuffle the results."""
        data = many_cases(20)
        result = await Runner(Judge(rubric, StubProvider()), concurrency=8).run(data)
        assert [j.case_id for j in result.judgements] == [c.id for c in data.cases]

    async def test_records_dataset_rubric_and_provider(
        self, rubric: Rubric, dataset: Dataset
    ) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        assert result.dataset_name == "fixture"
        assert result.dataset_version == "v1"
        assert result.rubric == rubric.ref
        assert result.provider == "stub"
        assert result.model == "stub-judge-v1"

    async def test_bounds_concurrency(self, rubric: Rubric) -> None:
        provider = SlowProvider()
        await Runner(Judge(rubric, provider), concurrency=3).run(many_cases(12))
        assert provider.peak <= 3

    def test_rejects_nonsense_concurrency(self, rubric: Rubric) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Runner(Judge(rubric, StubProvider()), concurrency=0)

    async def test_calls_the_progress_hook_once_per_case(
        self, rubric: Rubric, dataset: Dataset
    ) -> None:
        seen: list[Judgement] = []
        await Runner(Judge(rubric, StubProvider())).run(dataset, on_result=seen.append)
        assert len(seen) == 3

    async def test_retries_a_transient_failure(self, rubric: Rubric) -> None:
        provider = FlakyProvider(failures=2)
        runner = Runner(Judge(rubric, provider), retries=3, backoff_seconds=0.0)
        result = await runner.run(many_cases(1))
        assert result.errored == 0
        assert provider.attempts == 3

    async def test_gives_up_after_the_retry_budget(self, rubric: Rubric) -> None:
        provider = FlakyProvider(failures=99)
        runner = Runner(Judge(rubric, provider), retries=2, backoff_seconds=0.0)
        result = await runner.run(many_cases(1))
        assert result.errored == 1
        assert provider.attempts == 3  # one attempt plus two retries

    async def test_no_retries_means_one_attempt(self, rubric: Rubric) -> None:
        provider = FlakyProvider(failures=99)
        runner = Runner(Judge(rubric, provider), retries=0, backoff_seconds=0.0)
        await runner.run(many_cases(1))
        assert provider.attempts == 1


class TestStatistics:
    async def test_empty_dataset_reports_zeroes_rather_than_dividing_by_zero(
        self, rubric: Rubric
    ) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(Dataset(name="empty"))
        assert result.total == 0
        assert result.pass_rate == 0.0
        assert result.mean_score == 0.0

    async def test_mean_score_excludes_errors(self, rubric: Rubric) -> None:
        """An unparseable response is a missing measurement, not a zero.

        Counting it as zero would make an infrastructure problem look like a
        quality regression.
        """
        provider = FlakyProvider(failures=1)
        runner = Runner(Judge(rubric, provider), retries=0, backoff_seconds=0.0)
        result = await runner.run(many_cases(2))

        assert result.errored == 1
        assert result.mean_score == 5.0

    async def test_pass_rate_counts_errors_against_the_total(self, rubric: Rubric) -> None:
        """A run that crashed did not pass, even though it did not fail either."""
        provider = FlakyProvider(failures=1)
        runner = Runner(Judge(rubric, provider), retries=0, backoff_seconds=0.0)
        result = await runner.run(many_cases(2))
        assert result.pass_rate == 0.5

    async def test_sums_usage_across_the_run(self, rubric: Rubric, dataset: Dataset) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        assert result.usage.total_tokens > 0

    async def test_reports_duration(self, rubric: Rubric, dataset: Dataset) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        assert result.duration_seconds >= 0.0

    def test_duration_is_zero_before_a_run_finishes(self, rubric: Rubric) -> None:
        from judgekit.core.runner import RunResult

        pending = RunResult(
            dataset_name="d", dataset_version="v1", rubric=rubric.ref, provider="p", model="m"
        )
        assert pending.duration_seconds == 0.0

    async def test_breaks_results_down_by_domain(self, rubric: Rubric, dataset: Dataset) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        stats = result.by_domain({c.id: c.domain for c in dataset.cases})
        assert set(stats) == {"finance"}
        assert stats["finance"].total == 3

    async def test_cases_with_no_domain_land_in_unassigned(self, rubric: Rubric) -> None:
        data = Dataset(name="d", cases=(Case(id="a", input="q", output="o"),))
        result = await Runner(Judge(rubric, StubProvider())).run(data)
        stats = result.by_domain({"a": None})
        assert "unassigned" in stats

    async def test_domain_pass_rate(self, rubric: Rubric, dataset: Dataset) -> None:
        result = await Runner(Judge(rubric, StubProvider())).run(dataset)
        stats = result.by_domain({c.id: c.domain for c in dataset.cases})
        assert 0.0 <= stats["finance"].pass_rate <= 1.0


class TestCompare:
    async def _run(self, rubric: Rubric, data: Dataset, **kwargs: object) -> object:
        provider = StubProvider(**kwargs)  # type: ignore[arg-type]
        return await Runner(Judge(rubric, provider)).run(data)

    async def test_refuses_to_compare_across_rubric_versions(
        self, rubric: Rubric, dataset: Dataset
    ) -> None:
        """The refusal that justifies the whole project."""
        v1 = await Runner(Judge(rubric, StubProvider())).run(dataset)
        bumped = rubric.model_copy(update={"version": "v2"})
        v2 = await Runner(Judge(bumped, StubProvider())).run(dataset)

        with pytest.raises(IncomparableScoresError, match="differs in version"):
            compare(v1, v2)

    async def test_refuses_to_compare_across_an_in_place_edit(
        self, rubric: Rubric, dataset: Dataset
    ) -> None:
        before = await Runner(Judge(rubric, StubProvider())).run(dataset)
        edited = rubric.model_copy(update={"instructions": "Grade much more harshly."})
        after = await Runner(Judge(edited, StubProvider())).run(dataset)

        with pytest.raises(IncomparableScoresError, match="edited in place"):
            compare(before, after)

    async def test_compares_two_runs_under_the_same_rubric(
        self, rubric: Rubric, dataset: Dataset
    ) -> None:
        judge = Judge(rubric, StubProvider())
        baseline = await Runner(judge).run(dataset)
        candidate = await Runner(judge).run(dataset)

        result = compare(baseline, candidate)

        assert len(result.deltas) == 3
        assert result.pass_rate_delta == 0.0
        assert result.regressions == ()
        assert result.improvements == ()

    async def test_detects_a_regression(self, rubric: Rubric, dataset: Dataset) -> None:
        good = await Runner(Judge(rubric, StubProvider())).run(dataset)
        bad = await Runner(Judge(rubric, StubProvider(noise=1.0, seed=7))).run(dataset)

        result = compare(good, bad)
        moved = [d for d in result.deltas if d.delta != 0.0]
        assert moved, "expected the noisy run to move at least one score"

    async def test_reports_cases_present_on_only_one_side(self, rubric: Rubric) -> None:
        judge = Judge(rubric, StubProvider())
        baseline = await Runner(judge).run(
            Dataset(name="d", cases=(Case(id="kept", input="q", output="o"),))
        )
        candidate = await Runner(judge).run(
            Dataset(name="d", cases=(Case(id="added", input="q", output="o"),))
        )

        result = compare(baseline, candidate)

        assert result.only_in_baseline == ("kept",)
        assert result.only_in_candidate == ("added",)
        assert result.deltas == ()

    async def test_gate_allows_an_unchanged_run(self, rubric: Rubric, dataset: Dataset) -> None:
        judge = Judge(rubric, StubProvider())
        baseline = await Runner(judge).run(dataset)
        candidate = await Runner(judge).run(dataset)
        assert compare(baseline, candidate).gate()

    def test_gate_blocks_a_drop_beyond_tolerance(self) -> None:
        from judgekit.core.runner import Comparison

        comparison = Comparison(
            baseline_run_id="a",
            candidate_run_id="b",
            pass_rate_before=0.9,
            pass_rate_after=0.7,
            mean_score_before=4.5,
            mean_score_after=3.5,
        )
        assert not comparison.gate()
        assert not comparison.gate(max_pass_rate_drop=0.1)
        assert comparison.gate(max_pass_rate_drop=0.25)

    def test_delta_classifies_movement(self) -> None:
        from judgekit.core.runner import CaseDelta

        regressed = CaseDelta(
            case_id="c",
            before=5.0,
            after=2.0,
            before_verdict=Verdict.PASS,
            after_verdict=Verdict.FAIL,
        )
        assert regressed.regressed
        assert not regressed.improved
        assert regressed.delta == -3.0

        improved = CaseDelta(
            case_id="c",
            before=2.0,
            after=5.0,
            before_verdict=Verdict.FAIL,
            after_verdict=Verdict.PASS,
        )
        assert improved.improved
        assert not improved.regressed

    def test_an_error_becoming_a_pass_counts_as_an_improvement(self) -> None:
        from judgekit.core.runner import CaseDelta

        delta = CaseDelta(
            case_id="c",
            before=1.0,
            after=5.0,
            before_verdict=Verdict.ERROR,
            after_verdict=Verdict.PASS,
        )
        assert delta.improved
