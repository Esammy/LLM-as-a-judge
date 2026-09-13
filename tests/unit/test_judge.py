"""The judge: prompt construction, flagging, aggregation and failure handling."""

from __future__ import annotations

import json

import pytest

from judgekit.core.judge import Judge
from judgekit.core.models import Case, Evidence, Flag, ToolCall, Verdict
from judgekit.core.promptfmt import (
    TAG_CRITERIA,
    TAG_EVIDENCE,
    TAG_INPUT,
    TAG_OUTPUT,
    TAG_REFERENCE,
    TAG_SCALE,
    read_section,
)
from judgekit.core.rubric import Criterion, Rubric, Scale
from judgekit.providers.base import Completion, CompletionRequest, Usage
from judgekit.providers.stub import StubProvider


class ScriptedProvider:
    """Returns a fixed response, so judge behaviour can be tested in isolation."""

    def __init__(self, text: str, *, truncated: bool = False) -> None:
        self._text = text
        self._truncated = truncated
        self.calls: list[CompletionRequest] = []

    name = "scripted"
    model = "scripted-v1"
    family = "scripted"

    async def complete(self, request: CompletionRequest) -> Completion:
        self.calls.append(request)
        return Completion(
            text=self._text,
            model=self.model,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            truncated=self._truncated,
        )


class ExplodingProvider:
    """Raises, to prove a provider outage does not take the run down with it."""

    name = "exploding"
    model = "exploding-v1"
    family = "exploding"

    async def complete(self, request: CompletionRequest) -> Completion:
        raise ConnectionError("provider is unreachable")


class TestBuildPrompt:
    def test_includes_input_and_output(self, rubric: Rubric, grounded_case: Case) -> None:
        prompt = Judge(rubric, StubProvider()).build_prompt(grounded_case)
        assert read_section(prompt, TAG_INPUT) == grounded_case.input
        assert read_section(prompt, TAG_OUTPUT) == grounded_case.output

    def test_includes_evidence_when_the_rubric_wants_it(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        prompt = Judge(rubric, StubProvider()).build_prompt(grounded_case)
        evidence = read_section(prompt, TAG_EVIDENCE)
        assert evidence is not None
        assert "get_revenue" in evidence
        assert "48200" in evidence

    def test_omits_evidence_when_the_rubric_does_not(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        blind = rubric.model_copy(update={"requires_evidence": False})
        prompt = Judge(blind, StubProvider()).build_prompt(grounded_case)
        assert read_section(prompt, TAG_EVIDENCE) is None

    def test_includes_the_reference_when_present(self, rubric: Rubric, grounded_case: Case) -> None:
        prompt = Judge(rubric, StubProvider()).build_prompt(grounded_case)
        assert read_section(prompt, TAG_REFERENCE) == grounded_case.reference

    def test_says_so_explicitly_when_there_is_no_reference(self, rubric: Rubric) -> None:
        """Silence would let the judge assume a missing gold answer means failure."""
        case = Case(id="c", input="q", output="an answer")
        prompt = Judge(rubric, StubProvider()).build_prompt(case)
        assert read_section(prompt, TAG_REFERENCE) is None
        assert "No reference answer was supplied" in prompt

    def test_lists_the_criteria(self, rubric: Rubric, grounded_case: Case) -> None:
        criteria = read_section(
            Judge(rubric, StubProvider()).build_prompt(grounded_case), TAG_CRITERIA
        )
        assert criteria is not None
        assert "groundedness" in criteria
        assert "correctness" in criteria

    def test_states_the_scale_as_json(self, rubric: Rubric, grounded_case: Case) -> None:
        raw = read_section(Judge(rubric, StubProvider()).build_prompt(grounded_case), TAG_SCALE)
        assert raw is not None
        assert json.loads(raw)["pass_at"] == 4

    def test_renders_empty_evidence_rather_than_omitting_it(self, rubric: Rubric) -> None:
        case = Case(id="c", input="q", output="a", reference="a")
        prompt = Judge(rubric, StubProvider()).build_prompt(case)
        assert read_section(prompt, TAG_EVIDENCE) == "(no evidence supplied)"


class TestFlags:
    async def test_flags_a_missing_reference(self, rubric: Rubric) -> None:
        case = Case(id="c", input="q", output="an answer")
        judgement = await Judge(rubric, StubProvider()).judge(case)
        assert Flag.NO_REFERENCE in judgement.flags

    async def test_flags_missing_evidence_when_the_rubric_requires_it(
        self, rubric: Rubric, evidenceless_case: Case
    ) -> None:
        judgement = await Judge(rubric, StubProvider()).judge(evidenceless_case)
        assert Flag.NO_EVIDENCE in judgement.flags

    async def test_does_not_flag_evidence_the_rubric_never_asked_for(
        self, rubric: Rubric, evidenceless_case: Case
    ) -> None:
        blind = rubric.model_copy(update={"requires_evidence": False})
        judgement = await Judge(blind, StubProvider()).judge(evidenceless_case)
        assert Flag.NO_EVIDENCE not in judgement.flags

    async def test_flags_a_truncated_response(self, rubric: Rubric, grounded_case: Case) -> None:
        provider = ScriptedProvider('{"score": 5}', truncated=True)
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert Flag.TRUNCATED in judgement.flags

    async def test_flags_a_recovered_parse(self, rubric: Rubric, grounded_case: Case) -> None:
        provider = ScriptedProvider('Sure thing!\n```json\n{"score": 5}\n```')
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert Flag.PARSE_RECOVERED in judgement.flags
        assert judgement.score == 5.0

    async def test_clean_json_is_not_flagged_as_recovered(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider('{"score": 5}')
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert Flag.PARSE_RECOVERED not in judgement.flags


class TestAggregation:
    async def test_weights_criteria_as_the_rubric_specifies(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        # groundedness weighs 3, correctness 2 -> (5*3 + 1*2) / 5 = 3.4
        provider = ScriptedProvider(
            json.dumps({"criterion_scores": {"groundedness": 5, "correctness": 1}})
        )
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == pytest.approx(3.4)

    async def test_ignores_criteria_the_rubric_does_not_define(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider(
            json.dumps({"criterion_scores": {"groundedness": 4, "invented": 1}})
        )
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == pytest.approx(4.0)

    async def test_falls_back_to_a_plain_mean_for_unknown_criteria(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider(json.dumps({"criterion_scores": {"whatever": 2, "other": 4}}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == pytest.approx(3.0)

    async def test_falls_back_to_a_top_level_score(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        """Real models often ignore the per-criterion request. Handle it."""
        provider = ScriptedProvider(json.dumps({"score": 4.5}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == 4.5

    async def test_clamps_a_score_outside_the_scale(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider(json.dumps({"score": 99}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == 5.0

    async def test_discards_non_numeric_criterion_scores(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider(
            json.dumps({"criterion_scores": {"groundedness": "excellent"}, "score": 3})
        )
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == 3.0


class TestVerdict:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(5.0, Verdict.PASS), (4.0, Verdict.PASS), (3.99, Verdict.FAIL), (1.0, Verdict.FAIL)],
    )
    async def test_applies_the_pass_threshold(
        self, rubric: Rubric, grounded_case: Case, score: float, expected: Verdict
    ) -> None:
        provider = ScriptedProvider(json.dumps({"score": score}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.verdict is expected


class TestFailureHandling:
    async def test_a_provider_outage_becomes_an_error_judgement(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        """One dead call must not lose the other 599 results in a run."""
        judgement = await Judge(rubric, ExplodingProvider()).judge(grounded_case)
        assert judgement.verdict is Verdict.ERROR
        assert not judgement.ok
        assert judgement.error is not None
        assert "ConnectionError" in judgement.error

    async def test_unparseable_output_becomes_an_error_judgement(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        judgement = await Judge(rubric, ScriptedProvider("I would rather not")).judge(grounded_case)
        assert judgement.verdict is Verdict.ERROR
        assert judgement.error is not None
        assert "no JSON object" in judgement.error

    async def test_json_without_any_score_becomes_an_error_judgement(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        provider = ScriptedProvider(json.dumps({"reasoning": "it was fine"}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.verdict is Verdict.ERROR
        assert judgement.error is not None
        assert "neither usable criterion_scores nor a score" in judgement.error

    async def test_a_json_array_becomes_an_error_judgement(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        judgement = await Judge(rubric, ScriptedProvider("[1, 2, 3]")).judge(grounded_case)
        assert judgement.verdict is Verdict.ERROR

    async def test_error_judgements_still_carry_their_flags(
        self, rubric: Rubric, evidenceless_case: Case
    ) -> None:
        judgement = await Judge(rubric, ExplodingProvider()).judge(evidenceless_case)
        assert Flag.NO_EVIDENCE in judgement.flags


class TestProvenance:
    async def test_records_the_rubric_it_graded_under(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        judgement = await Judge(rubric, StubProvider()).judge(grounded_case)
        assert judgement.rubric == rubric.ref
        assert judgement.rubric.fingerprint == rubric.fingerprint

    async def test_records_the_model_and_family(self, rubric: Rubric, grounded_case: Case) -> None:
        judgement = await Judge(rubric, StubProvider(family="anthropic")).judge(grounded_case)
        assert judgement.model == "stub-judge-v1"
        assert judgement.family == "anthropic"

    async def test_records_usage_and_elapsed_time(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        judgement = await Judge(rubric, ScriptedProvider('{"score": 4}')).judge(grounded_case)
        assert judgement.usage.total_tokens == 15
        assert judgement.elapsed_ms >= 0.0

    async def test_passes_temperature_zero_by_default(
        self, rubric: Rubric, grounded_case: Case
    ) -> None:
        """A judge that is not reproducible is not a measurement."""
        provider = ScriptedProvider('{"score": 4}')
        await Judge(rubric, provider).judge(grounded_case)
        assert provider.calls[0].temperature == 0.0

    async def test_forwards_the_system_prompt(self, rubric: Rubric, grounded_case: Case) -> None:
        provider = ScriptedProvider('{"score": 4}')
        await Judge(rubric, provider).judge(grounded_case)
        system = provider.calls[0].system
        assert system is not None
        assert "Length is not quality" in system


class TestEvidenceAwareness:
    async def test_the_same_answer_scores_higher_with_evidence_than_without(
        self, rubric: Rubric, grounded_case: Case, evidenceless_case: Case
    ) -> None:
        """The project's central claim, asserted end to end.

        Both cases state the same correct figure. The only difference is whether
        the tool call behind it was captured.
        """
        judge = Judge(rubric, StubProvider())
        with_evidence = await judge.judge(grounded_case)
        without = await judge.judge(evidenceless_case)

        assert with_evidence.score > without.score
        assert Flag.NO_EVIDENCE in without.flags
        assert Flag.NO_EVIDENCE not in with_evidence.flags

    async def test_an_invented_figure_scores_below_a_grounded_one(
        self, rubric: Rubric, grounded_case: Case, ungrounded_case: Case
    ) -> None:
        judge = Judge(rubric, StubProvider())
        assert (await judge.judge(grounded_case)).score > (await judge.judge(ungrounded_case)).score

    async def test_evidence_reaches_the_prompt_verbatim(self, rubric: Rubric) -> None:
        case = Case(
            id="c",
            input="q",
            output="a",
            evidence=Evidence(
                tool_calls=(ToolCall(name="lookup", arguments={"k": "v"}, result=[1, 2]),),
                context=("a retrieved passage",),
            ),
        )
        provider = ScriptedProvider('{"score": 4}')
        await Judge(rubric, provider).judge(case)

        prompt = provider.calls[0].prompt
        assert "lookup" in prompt
        assert "a retrieved passage" in prompt


class TestConfiguration:
    async def test_honours_a_custom_scale(self, grounded_case: Case) -> None:
        rubric = Rubric(
            id="wide",
            version="v1",
            instructions="grade",
            scale=Scale(minimum=0, maximum=10, pass_at=7),
            criteria=(Criterion(id="quality", description="how good"),),
        )
        provider = ScriptedProvider(json.dumps({"criterion_scores": {"quality": 8}}))
        judgement = await Judge(rubric, provider).judge(grounded_case)
        assert judgement.score == 8.0
        assert judgement.verdict is Verdict.PASS

    async def test_forwards_max_tokens(self, rubric: Rubric, grounded_case: Case) -> None:
        provider = ScriptedProvider('{"score": 4}')
        await Judge(rubric, provider, max_tokens=77).judge(grounded_case)
        assert provider.calls[0].max_tokens == 77
