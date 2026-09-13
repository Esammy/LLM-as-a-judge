"""The deterministic stub provider.

The stub underpins every other test, so its own properties matter more than
usual. Two in particular:

* it must be **reproducible**, or nothing built on it can be;
* it must be **unbiased by default**, because the bias detectors are tested by
  injecting a known amount of bias into it. A provider that is already biased
  makes those measurements meaningless.
"""

from __future__ import annotations

import json

import pytest

from judgekit.core.promptfmt import (
    TAG_CRITERIA,
    TAG_EVIDENCE,
    TAG_OUTPUT,
    TAG_REFERENCE,
    TAG_SCALE,
    section,
)
from judgekit.providers.base import CompletionRequest, Provider
from judgekit.providers.stub import StubProvider


def prompt_for(
    output: str,
    *,
    reference: str | None = None,
    evidence: str | None = None,
    criteria: list[str] | None = None,
    scale: tuple[int, int] = (1, 5),
    extra: str = "",
) -> str:
    parts = [section(TAG_OUTPUT, output)]
    if reference is not None:
        parts.append(section(TAG_REFERENCE, reference))
    if evidence is not None:
        parts.append(section(TAG_EVIDENCE, evidence))
    if criteria:
        parts.append(section(TAG_CRITERIA, "\n".join(f"- {c}: description" for c in criteria)))
    parts.append(
        section(TAG_SCALE, json.dumps({"minimum": scale[0], "maximum": scale[1], "pass_at": 4}))
    )
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


async def score(provider: StubProvider, prompt: str) -> float:
    completion = await provider.complete(CompletionRequest(prompt=prompt))
    return float(json.loads(completion.text)["score"])


class TestProtocol:
    def test_satisfies_the_provider_protocol(self) -> None:
        assert isinstance(StubProvider(), Provider)

    def test_reports_its_identity(self) -> None:
        provider = StubProvider(model="custom", family="fake-family")
        assert provider.name == "stub"
        assert provider.model == "custom"
        assert provider.family == "fake-family"


class TestDeterminism:
    async def test_identical_prompts_give_identical_scores(self) -> None:
        provider = StubProvider(noise=0.5)
        prompt = prompt_for("an answer", reference="an answer")
        assert await score(provider, prompt) == await score(provider, prompt)

    async def test_different_seeds_give_different_jitter(self) -> None:
        prompt = prompt_for("partly right", reference="a fully correct answer here")
        a = await score(StubProvider(seed=1, noise=0.5), prompt)
        b = await score(StubProvider(seed=2, noise=0.5), prompt)
        assert a != b

    async def test_no_noise_means_no_jitter(self) -> None:
        prompt = prompt_for("an answer", reference="an answer")
        a = await score(StubProvider(seed=1), prompt)
        b = await score(StubProvider(seed=999), prompt)
        assert a == b


class TestNeutrality:
    async def test_prefers_a_concise_answer_over_a_padded_one(self) -> None:
        """Regression test for a structural verbosity bias in this stub.

        Scoring by recall against the reference alone rewarded length: a padded
        answer covers more reference tokens simply by saying more, so the stub
        ranked padding above precision. Since the verbosity detector is tested
        by injecting known bias *into* this provider, a provider biased by
        construction would have invalidated that whole measurement.

        The fix was an F1 over content tokens, where padding costs precision.
        """
        reference = "There are 312 units of SKU-4471 in stock."
        evidence = "get_stock({}) -> {'units': 312}"

        concise = await score(
            StubProvider(), prompt_for("312 units.", reference=reference, evidence=evidence)
        )
        padded = await score(
            StubProvider(),
            prompt_for(
                "That is a great question and inventory is important to watch closely. "
                "Having reviewed the current position across the various warehouses, I "
                "can confirm the figure you want is 312 units, though do note that stock "
                "levels fluctuate over time with incoming and outgoing orders.",
                reference=reference,
                evidence=evidence,
            ),
        )

        assert concise > padded


class TestInjectedBias:
    async def test_verbosity_bias_rewards_length(self) -> None:
        reference = "There are 312 units in stock."
        long_output = "312 units. " + ("Additional context. " * 40)
        neutral = await score(StubProvider(), prompt_for(long_output, reference=reference))
        biased = await score(
            StubProvider(verbosity_bias=0.9), prompt_for(long_output, reference=reference)
        )
        assert biased > neutral

    # A middling answer, so the score has headroom. A perfect answer saturates
    # the top of the scale and no amount of bias can move it, which would make
    # the assertion pass or fail for the wrong reason.
    MEDIOCRE = "revenue rose"
    REFERENCE = "revenue rose sharply in the third quarter"

    async def test_position_bias_applies_only_to_the_first_candidate(self) -> None:
        plain = prompt_for(self.MEDIOCRE, reference=self.REFERENCE)
        pairwise = prompt_for(self.MEDIOCRE, reference=self.REFERENCE, extra="<CANDIDATE_A>")

        provider = StubProvider(position_bias=0.3)
        assert await score(provider, pairwise) > await score(provider, plain)

    async def test_no_position_bias_leaves_pairwise_prompts_alone(self) -> None:
        plain = prompt_for(self.MEDIOCRE, reference=self.REFERENCE)
        pairwise = prompt_for(self.MEDIOCRE, reference=self.REFERENCE, extra="<CANDIDATE_A>")
        provider = StubProvider()
        assert await score(provider, pairwise) == await score(provider, plain)


class TestGroundedness:
    async def test_supported_figures_score_above_unsupported_ones(self) -> None:
        reference = "Revenue was 48200 USD."
        supported = await score(
            StubProvider(),
            prompt_for("Revenue was 48200 USD.", reference=reference, evidence="total 48200"),
        )
        invented = await score(
            StubProvider(),
            prompt_for("Revenue was 99999 USD.", reference=reference, evidence="total 48200"),
        )
        assert supported > invented

    async def test_an_answer_with_no_figures_is_not_penalised(self) -> None:
        result = await score(
            StubProvider(),
            prompt_for("Revenue grew.", reference="Revenue grew.", evidence="some evidence"),
        )
        assert result >= 4.0

    async def test_missing_evidence_lowers_a_numeric_answer(self) -> None:
        """Mirrors the real judge behaviour this project exists to expose."""
        reference = "Revenue was 48200 USD."
        with_evidence = await score(
            StubProvider(),
            prompt_for("Revenue was 48200 USD.", reference=reference, evidence="total 48200"),
        )
        without = await score(
            StubProvider(), prompt_for("Revenue was 48200 USD.", reference=reference)
        )
        assert without < with_evidence


class TestPromptParsing:
    async def test_honours_the_scale_in_the_prompt(self) -> None:
        result = await score(
            StubProvider(),
            prompt_for("an answer", reference="an answer", scale=(0, 100)),
        )
        assert result > 5.0

    async def test_falls_back_to_a_default_scale(self) -> None:
        completion = await StubProvider().complete(
            CompletionRequest(prompt=section(TAG_OUTPUT, "an answer"))
        )
        assert 1.0 <= json.loads(completion.text)["score"] <= 5.0

    @pytest.mark.parametrize("bad", ["not json", '{"minimum": "low"}', "[]"])
    async def test_survives_an_unusable_scale_section(self, bad: str) -> None:
        prompt = section(TAG_OUTPUT, "an answer") + "\n\n" + section(TAG_SCALE, bad)
        completion = await StubProvider().complete(CompletionRequest(prompt=prompt))
        assert 1.0 <= json.loads(completion.text)["score"] <= 5.0

    async def test_emits_a_score_for_every_criterion(self) -> None:
        completion = await StubProvider().complete(
            CompletionRequest(prompt=prompt_for("an answer", criteria=["groundedness", "clarity"]))
        )
        assert set(json.loads(completion.text)["criterion_scores"]) == {
            "groundedness",
            "clarity",
        }

    async def test_emits_no_criterion_scores_when_none_were_given(self) -> None:
        completion = await StubProvider().complete(
            CompletionRequest(prompt=prompt_for("an answer"))
        )
        assert json.loads(completion.text)["criterion_scores"] == {}


class TestCompletion:
    async def test_reports_usage_and_costs_nothing(self) -> None:
        completion = await StubProvider().complete(CompletionRequest(prompt=prompt_for("answer")))
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        assert completion.usage.cost_usd == 0.0

    async def test_says_it_is_a_heuristic_not_a_model(self) -> None:
        completion = await StubProvider().complete(CompletionRequest(prompt=prompt_for("answer")))
        assert "not a model" in json.loads(completion.text)["reasoning"]
