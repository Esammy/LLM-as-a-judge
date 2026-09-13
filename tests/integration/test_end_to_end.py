"""End-to-end behaviour against the rubrics and dataset the project ships.

These tests assert the claims the README makes. If one of them fails, the
README is wrong and needs changing - not the test.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from judgekit.core.errors import IncomparableScoresError
from judgekit.core.judge import Judge
from judgekit.core.models import Dataset, Flag, Verdict
from judgekit.core.rubric import Rubric, load_rubrics, read_lock, verify_lock
from judgekit.core.runner import Runner, RunResult, compare
from judgekit.providers.stub import StubProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = REPO_ROOT / "rubrics"
DATASET_PATH = REPO_ROOT / "datasets" / "example.jsonl"


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset.from_file(DATASET_PATH)


@pytest.fixture(scope="module")
def v1() -> Rubric:
    return Rubric.from_file(RUBRICS_DIR / "answer-quality.v1.yaml")


@pytest.fixture(scope="module")
def v2() -> Rubric:
    return Rubric.from_file(RUBRICS_DIR / "answer-quality.v2.yaml")


async def run(rubric: Rubric, dataset: Dataset, **provider_kwargs: float) -> RunResult:
    provider = StubProvider(**provider_kwargs)  # type: ignore[arg-type]
    return await Runner(Judge(rubric, provider), concurrency=4).run(dataset)


class TestShippedArtifacts:
    def test_the_rubric_lockfile_is_current(self) -> None:
        """Guards the guard.

        If someone edits a shipped rubric without bumping its version, this
        fails - which is the exact protection judgekit offers its users, applied
        to judgekit itself.
        """
        violations = verify_lock(load_rubrics(RUBRICS_DIR), read_lock(RUBRICS_DIR))
        assert violations == [], "\n".join(str(v) for v in violations)

    def test_the_example_dataset_is_fully_labelled(self, dataset: Dataset) -> None:
        """Calibration needs human labels; an unlabelled example teaches nothing."""
        assert len(dataset) == 8
        assert len(dataset.labelled) == len(dataset)

    def test_the_two_rubrics_differ_in_evidence_handling(self, v1: Rubric, v2: Rubric) -> None:
        assert v1.requires_evidence is False
        assert v2.requires_evidence is True
        assert v1.fingerprint != v2.fingerprint


class TestRunning:
    async def test_the_whole_dataset_judges_without_errors(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        result = await run(v2, dataset)
        assert result.total == 8
        assert result.errored == 0

    async def test_results_carry_full_provenance(self, v2: Rubric, dataset: Dataset) -> None:
        result = await run(v2, dataset)
        assert result.rubric.id == "answer-quality"
        assert result.rubric.version == "v2"
        assert result.rubric.fingerprint == v2.fingerprint
        assert all(j.rubric == v2.ref for j in result.judgements)

    async def test_a_run_is_reproducible(self, v2: Rubric, dataset: Dataset) -> None:
        """Two identical runs must agree exactly, or nothing downstream means anything."""
        first = await run(v2, dataset)
        second = await run(v2, dataset)
        assert [j.score for j in first.judgements] == [j.score for j in second.judgements]


class TestEvidenceAwareness:
    async def test_the_invented_figure_scores_lowest_of_the_revenue_cases(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        result = await run(v2, dataset)
        scores = {j.case_id: j.score for j in result.judgements}
        assert scores["revenue-invented-figure"] < scores["revenue-grounded"]
        assert scores["revenue-invented-figure"] < scores["revenue-correct-but-unverifiable"]

    async def test_the_unverifiable_case_is_flagged_not_silently_marked_down(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        """The finding this project exists to surface.

        A correct answer whose evidence was never captured is *unverifiable*.
        Scoring it low without saying why is how an eval suite reports a quality
        problem that is really a logging problem.
        """
        result = await run(v2, dataset)
        judgement = next(
            j for j in result.judgements if j.case_id == "revenue-correct-but-unverifiable"
        )
        assert Flag.NO_EVIDENCE in judgement.flags
        assert judgement.verdict is not Verdict.ERROR

    async def test_only_the_evidenceless_cases_carry_the_evidence_flag(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        result = await run(v2, dataset)
        flagged = {j.case_id for j in result.judgements if Flag.NO_EVIDENCE in j.flags}
        assert flagged == {"revenue-correct-but-unverifiable"}

    async def test_the_case_without_a_reference_is_flagged(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        result = await run(v2, dataset)
        flagged = {j.case_id for j in result.judgements if Flag.NO_REFERENCE in j.flags}
        assert flagged == {"policy-no-reference"}


class TestJudgeQuality:
    async def test_the_default_judge_broadly_agrees_with_the_humans(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        """A sanity floor on the stub, not a claim about real judges.

        If this drops, the stub has stopped being a plausible stand-in and every
        test built on it is measuring the wrong thing.
        """
        result = await run(v2, dataset)
        labels = {c.id: c.human_label for c in dataset.cases}

        human = [labels[j.case_id] or 0.0 for j in result.judgements]
        judged = [j.score for j in result.judgements]

        correlation = statistics.correlation(human, judged, method="ranked")
        assert correlation > 0.7, f"rank agreement fell to {correlation:.3f}"

    async def test_concise_beats_padded_by_default(self, v2: Rubric, dataset: Dataset) -> None:
        """Matches the human labels: 5 for the concise answer, 3 for the padded one."""
        result = await run(v2, dataset)
        scores = {j.case_id: j.score for j in result.judgements}
        assert scores["inventory-concise"] > scores["inventory-padded"]

    async def test_injected_verbosity_bias_inverts_that_ordering(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        """Proves the bias is detectable, which Phase 3's detector depends on."""
        result = await run(v2, dataset, verbosity_bias=0.9)
        scores = {j.case_id: j.score for j in result.judgements}
        assert scores["inventory-padded"] > scores["inventory-concise"]


class TestComparability:
    async def test_two_runs_under_one_rubric_compare_cleanly(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        baseline = await run(v2, dataset)
        candidate = await run(v2, dataset)

        result = compare(baseline, candidate)

        assert len(result.deltas) == 8
        assert result.pass_rate_delta == 0.0
        assert result.gate()

    async def test_comparing_v1_against_v2_is_refused(
        self, v1: Rubric, v2: Rubric, dataset: Dataset
    ) -> None:
        """The headline guarantee, on the shipped rubrics."""
        before = await run(v1, dataset)
        after = await run(v2, dataset)

        with pytest.raises(IncomparableScoresError, match="differs in version"):
            compare(before, after)

    async def test_per_case_regressions_survive_a_flattering_aggregate(
        self, v2: Rubric, dataset: Dataset
    ) -> None:
        """A rising pass rate can still hide cases that got worse.

        Injecting noise moves scores in both directions. On this dataset the
        aggregate pass rate *improves* while ``revenue-grounded`` drops from
        PASS to FAIL - so a gate watching only the headline number would wave
        the change through.

        This is the argument for tracking case-level deltas rather than a single
        percentage, and it is why :class:`Comparison` exposes ``regressions``
        separately from ``pass_rate_delta``.
        """
        baseline = await run(v2, dataset)
        degraded = await run(v2, dataset, noise=1.0)

        result = compare(baseline, degraded)

        moved = [d for d in result.deltas if d.delta != 0.0]
        assert moved, "noise should have moved at least one score"

        assert result.regressions, "expected at least one case to regress"
        assert all(d.before_verdict is Verdict.PASS for d in result.regressions)
        assert all(d.after_verdict is not Verdict.PASS for d in result.regressions)

    def test_a_falling_pass_rate_fails_the_gate(self) -> None:
        from judgekit.core.runner import Comparison

        dropped = Comparison(
            baseline_run_id="a",
            candidate_run_id="b",
            pass_rate_before=0.9,
            pass_rate_after=0.6,
            mean_score_before=4.4,
            mean_score_after=3.2,
        )
        assert not dropped.gate()
        assert dropped.pass_rate_delta == pytest.approx(-0.3)
