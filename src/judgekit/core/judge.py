"""The judge: turn a case plus a rubric into a defensible score.

The design decision that matters here is what the judge is *shown*. A judge
asked whether a figure was invented cannot answer honestly unless it can see
the tool call the figure came from. Given only the answer, it guesses, and it
guesses conservatively - so correct figures get marked as fabrications and the
suite reports a quality problem that does not exist.

So evidence is rendered into the prompt as a first-class section, and a case
that states figures with no evidence behind them is *flagged* rather than
silently scored low.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from judgekit.core.errors import JudgeParseError
from judgekit.core.models import Case, Flag, Verdict
from judgekit.core.promptfmt import (
    TAG_CRITERIA,
    TAG_EVIDENCE,
    TAG_INPUT,
    TAG_OUTPUT,
    TAG_REFERENCE,
    TAG_SCALE,
    extract_json,
    section,
)
from judgekit.core.rubric import Rubric, RubricRef
from judgekit.providers.base import CompletionRequest, Provider, Usage

SYSTEM_PROMPT = """\
You are a careful evaluator. You grade one answer against a rubric and return \
JSON only.

Three rules govern everything you do:

1. Judge only what you were given. Do not reward an answer for information you \
happen to know but were not shown.
2. A figure is grounded if it traces back to the supplied evidence. If evidence \
is supplied and a figure does not appear in it, that is a real problem. If no \
evidence is supplied at all, do not treat the figures as invented - say the \
answer is unverifiable instead.
3. Length is not quality. A short correct answer outscores a long vague one.
"""

RESPONSE_FORMAT = """\
Return a single JSON object and nothing else:

{
  "criterion_scores": {"<criterion id>": <number>, ...},
  "reasoning": "<two sentences at most, naming the deciding factor>"
}
"""


class Judgement(BaseModel):
    """One scored case, carrying everything needed to defend the number."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    score: float
    verdict: Verdict
    reasoning: str = ""
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    rubric: RubricRef
    flags: tuple[Flag, ...] = ()
    model: str = ""
    family: str = ""
    usage: Usage = Usage()
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.verdict is not Verdict.ERROR


class Judge:
    """Scores cases against one rubric using one provider.

    A judge is bound to a single rubric on purpose. Scores produced under
    different rubrics are not comparable, so letting one judge instance swap
    rubrics mid-run would quietly produce a meaningless aggregate.
    """

    def __init__(
        self,
        rubric: Rubric,
        provider: Provider,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.rubric = rubric
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(self, case: Case) -> str:
        """Render a case into the delimited prompt the judge reads."""
        parts = [
            self.rubric.instructions.strip(),
            "",
            section(TAG_INPUT, case.input),
            section(TAG_OUTPUT, case.output),
        ]

        if case.reference:
            parts.append(section(TAG_REFERENCE, case.reference))
        else:
            parts.append(
                "No reference answer was supplied. Grade on intrinsic quality "
                "and do not penalise the answer for differing from a gold text."
            )

        if self.rubric.requires_evidence:
            parts.append(section(TAG_EVIDENCE, case.evidence.render()))

        if self.rubric.criteria:
            criteria = "\n".join(f"- {c.id}: {c.description}" for c in self.rubric.criteria)
            parts.append(section(TAG_CRITERIA, criteria))

        parts.append(section(TAG_SCALE, json.dumps(self.rubric.scale.model_dump())))
        parts.append(RESPONSE_FORMAT)

        return "\n\n".join(parts)

    def _detect_flags(self, case: Case) -> list[Flag]:
        flags: list[Flag] = []
        if case.reference is None:
            flags.append(Flag.NO_REFERENCE)
        if self.rubric.requires_evidence and case.evidence.is_empty:
            flags.append(Flag.NO_EVIDENCE)
        return flags

    def _aggregate(self, payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Combine per-criterion scores into one number, weighted by the rubric.

        Falls back to a top-level ``score`` when the judge did not break its
        answer down, which real models do often enough to be worth handling.
        """
        raw = payload.get("criterion_scores")
        criterion_scores: dict[str, float] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, int | float):
                    criterion_scores[str(key)] = float(value)

        if criterion_scores and self.rubric.criteria:
            weighted = 0.0
            total = 0.0
            for criterion in self.rubric.criteria:
                if criterion.id in criterion_scores:
                    weighted += criterion_scores[criterion.id] * criterion.weight
                    total += criterion.weight
            if total:
                return self.rubric.scale.clamp(weighted / total), criterion_scores

        if criterion_scores:
            mean = sum(criterion_scores.values()) / len(criterion_scores)
            return self.rubric.scale.clamp(mean), criterion_scores

        fallback = payload.get("score")
        if isinstance(fallback, int | float):
            return self.rubric.scale.clamp(float(fallback)), criterion_scores

        raise JudgeParseError(
            "judge response contained neither usable criterion_scores nor a score"
        )

    async def judge(self, case: Case) -> Judgement:
        """Score one case. Never raises: failures come back as ERROR judgements.

        Returning rather than raising is what lets a 600-case run survive one
        malformed response instead of losing the whole batch.
        """
        started = time.perf_counter()
        flags = self._detect_flags(case)
        prompt = self.build_prompt(case)

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000.0

        try:
            completion = await self.provider.complete(
                CompletionRequest(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
        except Exception as exc:
            return Judgement(
                case_id=case.id,
                score=float(self.rubric.scale.minimum),
                verdict=Verdict.ERROR,
                rubric=self.rubric.ref,
                flags=tuple(flags),
                model=self.provider.model,
                family=self.provider.family,
                elapsed_ms=elapsed(),
                error=f"{type(exc).__name__}: {exc}",
            )

        if completion.truncated:
            flags.append(Flag.TRUNCATED)

        try:
            payload = json.loads(completion.text)
            if not isinstance(payload, dict):
                raise TypeError("not an object")
        except (json.JSONDecodeError, TypeError):
            try:
                payload = extract_json(completion.text)
            except JudgeParseError as exc:
                return Judgement(
                    case_id=case.id,
                    score=float(self.rubric.scale.minimum),
                    verdict=Verdict.ERROR,
                    rubric=self.rubric.ref,
                    flags=tuple(flags),
                    model=completion.model,
                    family=self.provider.family,
                    usage=completion.usage,
                    elapsed_ms=elapsed(),
                    error=str(exc),
                )
            flags.append(Flag.PARSE_RECOVERED)

        try:
            score, criterion_scores = self._aggregate(payload)
        except JudgeParseError as exc:
            return Judgement(
                case_id=case.id,
                score=float(self.rubric.scale.minimum),
                verdict=Verdict.ERROR,
                rubric=self.rubric.ref,
                flags=tuple(flags),
                model=completion.model,
                family=self.provider.family,
                usage=completion.usage,
                elapsed_ms=elapsed(),
                error=str(exc),
            )

        reasoning = payload.get("reasoning")
        return Judgement(
            case_id=case.id,
            score=score,
            verdict=Verdict.PASS if self.rubric.scale.is_pass(score) else Verdict.FAIL,
            reasoning=str(reasoning) if reasoning is not None else "",
            criterion_scores=criterion_scores,
            rubric=self.rubric.ref,
            flags=tuple(flags),
            model=completion.model,
            family=self.provider.family,
            usage=completion.usage,
            elapsed_ms=elapsed(),
        )
