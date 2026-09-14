"""Bias detection.

Every detector is tested the same way: run it against a judge with *no* bias
and confirm it reports none, then inject a known amount into the stub provider
and confirm the detector finds it. That only works because the stub is neutral
by construction - see ``tests/unit/test_stub.py::TestNeutrality``.
"""

from __future__ import annotations

import pytest

from judgekit.core.bias import (
    GENERATOR_FAMILY_KEY,
    BiasFinding,
    measure_position_bias,
    measure_self_preference,
    measure_verbosity_bias,
)
from judgekit.core.judge import Judge, Judgement
from judgekit.core.models import Case, Dataset, Verdict
from judgekit.core.rubric import Rubric, Scale
from judgekit.core.runner import Runner
from judgekit.providers.stub import StubProvider

SCALE = Scale(minimum=1, maximum=5, pass_at=4)


def case(
    case_id: str,
    output: str,
    label: float,
    *,
    family: str | None = None,
) -> Case:
    metadata = {GENERATOR_FAMILY_KEY: family} if family else {}
    return Case(
        id=case_id,
        input="q",
        output=output,
        reference="a reference answer",
        human_label=label,
        metadata=metadata,
    )


def judgement(rubric: Rubric, case_id: str, score: float, *, ok: bool = True) -> Judgement:
    return Judgement(
        case_id=case_id,
        score=score,
        verdict=Verdict.PASS if ok else Verdict.ERROR,
        rubric=rubric.ref,
        error=None if ok else "boom",
    )


class TestBiasFinding:
    def test_renders_readably(self) -> None:
        finding = BiasFinding(
            kind="verbosity", magnitude=0.5, unit="correlation", n=10, detail="", concerning=True
        )
        assert str(finding) == "[!] verbosity: +0.500 correlation (n=10)"

    def test_marks_an_acceptable_finding(self) -> None:
        finding = BiasFinding(
            kind="position", magnitude=0.0, unit="scale points", n=4, detail="", concerning=False
        )
        assert str(finding).startswith("[ok]")


class TestVerbosityBias:
    def test_detects_a_judge_that_rewards_length(self, rubric: Rubric) -> None:
        """Residual grows with length: the judge is paying for words."""
        dataset = Dataset(
            name="d",
            cases=(
                case("short", "x" * 10, 3.0),
                case("medium", "x" * 200, 3.0),
                case("long", "x" * 800, 3.0),
            ),
        )
        judgements = [
            judgement(rubric, "short", 3.0),
            judgement(rubric, "medium", 4.0),
            judgement(rubric, "long", 5.0),
        ]

        finding = measure_verbosity_bias(judgements, dataset, scale=SCALE)

        assert finding.magnitude > 0.5
        assert finding.concerning
        assert "more generous" in finding.detail

    def test_detects_a_judge_that_punishes_length(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(
                case("short", "x" * 10, 3.0),
                case("medium", "x" * 200, 3.0),
                case("long", "x" * 800, 3.0),
            ),
        )
        judgements = [
            judgement(rubric, "short", 5.0),
            judgement(rubric, "medium", 4.0),
            judgement(rubric, "long", 3.0),
        ]

        finding = measure_verbosity_bias(judgements, dataset, scale=SCALE)

        assert finding.magnitude < -0.5
        assert "harsher" in finding.detail

    def test_a_calibrated_judge_shows_no_bias(self, rubric: Rubric) -> None:
        """Long answers that humans also rated highly must not trip the detector.

        This is why the measurement uses the residual rather than the raw score:
        correlating length against score alone would flag this judge, which is
        agreeing with its humans perfectly.
        """
        dataset = Dataset(
            name="d",
            cases=(
                case("short", "x" * 10, 2.0),
                case("medium", "x" * 200, 3.0),
                case("long", "x" * 800, 5.0),
            ),
        )
        judgements = [
            judgement(rubric, "short", 2.0),
            judgement(rubric, "medium", 3.0),
            judgement(rubric, "long", 5.0),
        ]

        finding = measure_verbosity_bias(judgements, dataset, scale=SCALE)

        assert not finding.concerning
        assert finding.magnitude == pytest.approx(0.0, abs=1e-9)

    def test_reports_insufficient_data_rather_than_zero(self, rubric: Rubric) -> None:
        """ "Not measured" and "measured as unbiased" are different claims."""
        dataset = Dataset(name="d", cases=(case("a", "x", 3.0),))
        finding = measure_verbosity_bias([judgement(rubric, "a", 3.0)], dataset, scale=SCALE)

        assert finding.n == 1
        assert not finding.concerning
        assert "at least 3" in finding.detail

    def test_handles_no_variation_in_length(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(case("a", "xxx", 3.0), case("b", "yyy", 3.0), case("c", "zzz", 3.0)),
        )
        judgements = [
            judgement(rubric, "a", 3.0),
            judgement(rubric, "b", 4.0),
            judgement(rubric, "c", 5.0),
        ]

        finding = measure_verbosity_bias(judgements, dataset, scale=SCALE)

        assert "no variation" in finding.detail

    async def test_finds_bias_injected_into_the_stub(self) -> None:
        """End to end, against the shipped dataset."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        dataset = Dataset.from_file(repo_root / "datasets" / "example.jsonl")
        v2 = Rubric.from_file(repo_root / "rubrics" / "answer-quality.v2.yaml")

        neutral_run = await Runner(Judge(v2, StubProvider())).run(dataset)
        biased_run = await Runner(Judge(v2, StubProvider(verbosity_bias=0.9))).run(dataset)

        neutral = measure_verbosity_bias(neutral_run.judgements, dataset, scale=v2.scale)
        biased = measure_verbosity_bias(biased_run.judgements, dataset, scale=v2.scale)

        assert not neutral.concerning
        assert biased.concerning
        assert biased.magnitude > neutral.magnitude


class TestPositionBias:
    async def test_a_neutral_judge_shows_no_slot_effect(self, rubric: Rubric) -> None:
        judge = Judge(rubric, StubProvider())
        cases = [case("a", "an answer", 4.0), case("b", "another answer", 3.0)]

        finding = await measure_position_bias(judge, cases)

        assert finding.magnitude == pytest.approx(0.0)
        assert not finding.concerning
        assert "0 of 2 cases moved" in finding.detail

    async def test_detects_an_injected_slot_preference(self, rubric: Rubric) -> None:
        """Identical content, different slot, different score."""
        judge = Judge(rubric, StubProvider(position_bias=0.25))
        cases = [case("a", "an answer", 4.0), case("b", "another answer", 3.0)]

        finding = await measure_position_bias(judge, cases)

        assert finding.magnitude > 0.15
        assert finding.concerning
        assert "favouring the first slot" in finding.detail

    async def test_a_deterministic_judge_has_a_zero_noise_floor(self, rubric: Rubric) -> None:
        """The stub never moves, so the slot comparison stands on its own."""
        judge = Judge(rubric, StubProvider())
        cases = [case("a", "an answer", 4.0), case("b", "another answer", 3.0)]

        finding = await measure_position_bias(judge, cases)

        assert "Noise floor 0.00 points" in finding.detail
        assert "moved 0 of 2 cases with nothing changed" in finding.detail

    async def test_jitter_is_not_reported_as_position_bias(self, rubric: Rubric) -> None:
        """A judge that wobbles at random must not be called slot-biased.

        Hosted models are not deterministic at temperature 0 - batching and
        expert routing move scores between byte-identical calls. Measured on
        groq/openai/gpt-oss-120b, 2 of 8 cases drifted by up to 0.57 scale
        points across identical runs, which was exactly the movement the slot
        comparison had been attributing to position. This judge has no slot
        preference whatsoever and only jitters, so the finding must not be
        concerning however much it moves.
        """
        from itertools import cycle

        from judgekit.providers.base import Completion

        class Jittery:
            """Returns 5, then 4, then 3 for every case, ignoring the slot.

            Per case that is first=5, second=4, replicate=3: a slot delta of
            1.0 point against a noise floor of 2.0. The delta clears the 0.15
            threshold on its own, so without the replicate this judge would be
            reported as position-biased - which it provably is not, since it
            never reads the slot marker at all.
            """

            name = "jittery"
            model = "jittery-v1"
            family = "jittery"

            def __init__(self) -> None:
                self._scores = cycle((5, 4, 3))

            async def complete(self, request: object) -> Completion:
                score = next(self._scores)
                return Completion(text=f'{{"score": {score}, "reasoning": "r"}}', model=self.model)

        finding = await measure_position_bias(
            Judge(rubric, Jittery()),
            [case("a", "an answer", 4.0), case("b", "another answer", 3.0)],
        )

        assert finding.n == 2
        # The raw slot delta is well past the threshold that would flag it.
        assert finding.magnitude == pytest.approx(1.0)
        assert abs(finding.magnitude) > 0.15
        # But it is half the judge's own variance, so it is not a finding.
        assert not finding.concerning, finding.detail
        assert "Noise floor 2.00 points" in finding.detail
        assert "within the judge's own variance" in finding.detail

    async def test_no_cases_reports_insufficient(self, rubric: Rubric) -> None:
        finding = await measure_position_bias(Judge(rubric, StubProvider()), [])
        assert finding.n == 0
        assert "no cases supplied" in finding.detail

    async def test_unreadable_responses_are_reported_not_silently_dropped(
        self, rubric: Rubric
    ) -> None:
        class Gibberish:
            name = "gibberish"
            model = "gibberish-v1"
            family = "gibberish"

            async def complete(self, request: object) -> object:
                from judgekit.providers.base import Completion

                return Completion(text="I would rather not say", model=self.model)

        finding = await measure_position_bias(
            Judge(rubric, Gibberish()),  # type: ignore[arg-type]
            [case("a", "an answer", 4.0)],
        )

        assert finding.n == 1
        assert "unreadable" in finding.detail


class TestSelfPreference:
    def test_detects_a_judge_favouring_its_own_family(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(
                case("own-1", "answer", 3.0, family="stub"),
                case("own-2", "answer", 3.0, family="stub"),
                case("other-1", "answer", 3.0, family="rival"),
                case("other-2", "answer", 3.0, family="rival"),
            ),
        )
        judgements = [
            judgement(rubric, "own-1", 5.0),
            judgement(rubric, "own-2", 5.0),
            judgement(rubric, "other-1", 3.0),
            judgement(rubric, "other-2", 3.0),
        ]

        finding = measure_self_preference(judgements, dataset, judge_family="stub")

        assert finding.magnitude == pytest.approx(2.0)
        assert finding.concerning
        assert "scores its own output" in finding.detail

    def test_no_preference_when_families_are_treated_alike(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(
                case("own", "answer", 3.0, family="stub"),
                case("other", "answer", 3.0, family="rival"),
            ),
        )
        judgements = [judgement(rubric, "own", 3.0), judgement(rubric, "other", 3.0)]

        finding = measure_self_preference(judgements, dataset, judge_family="stub")

        assert finding.magnitude == pytest.approx(0.0)
        assert not finding.concerning

    def test_untagged_cases_say_so_rather_than_reporting_zero(self, rubric: Rubric) -> None:
        """Silently returning 0.0 would read as "no self-preference found"."""
        dataset = Dataset(name="d", cases=(case("a", "answer", 3.0),))
        finding = measure_self_preference(
            [judgement(rubric, "a", 3.0)], dataset, judge_family="stub"
        )

        assert finding.magnitude == 0.0
        assert not finding.concerning
        assert GENERATOR_FAMILY_KEY in finding.detail
        assert "found 0 own and 0 other" in finding.detail

    def test_only_one_family_present_reports_insufficient(self, rubric: Rubric) -> None:
        dataset = Dataset(name="d", cases=(case("a", "answer", 3.0, family="stub"),))
        finding = measure_self_preference(
            [judgement(rubric, "a", 3.0)], dataset, judge_family="stub"
        )
        assert "found 1 own and 0 other" in finding.detail
