"""Measuring judge bias, rather than only mitigating it.

Most write-ups on LLM-as-judge name the three well-known biases and prescribe a
mitigation: swap the order, normalise for length, use a judge from a different
family. That advice is correct and incomplete, because it never tells you how
big the problem was. A mitigation you cannot measure is a mitigation you cannot
justify keeping, tuning, or dropping.

So every function here returns a **magnitude** in a stated unit, not a verdict.

The measurements:

**Verbosity bias** is estimated against the *residual* - judge score minus human
label - rather than against the raw score. This matters. Long answers are often
genuinely better, so correlating length with score just rediscovers that. What
you want to know is whether the judge rewards length *beyond* what the humans
thought it was worth, and only the residual answers that.

**Position bias** is measured by scoring identical content twice, once presented
as the first candidate and once as the second. Nothing about the answer changes
between the two calls, so any difference in score is attributable to slot order
and nothing else.

**Self-preference** compares residuals on outputs the judge's own model family
generated against everything else, which needs cases to record who produced
them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from judgekit.core.errors import JudgeParseError
from judgekit.core.judge import Judge, Judgement
from judgekit.core.models import Case, Dataset
from judgekit.core.promptfmt import extract_json
from judgekit.core.rubric import Scale
from judgekit.providers.base import CompletionRequest

GENERATOR_FAMILY_KEY = "generator_family"
"""Case metadata key naming the model family that produced ``Case.output``."""

SLOT_FIRST = "<CANDIDATE_A>"
SLOT_SECOND = "<CANDIDATE_B>"

# Above this, a bias is worth acting on rather than noting. Chosen to sit near
# the low end of published verbosity-bias measurements rather than derived from
# anything - it is a default, and callers should override it with judgement.
CONCERN_THRESHOLD = 0.15


class BiasFinding(BaseModel):
    """One measured bias, with its size and how it was arrived at."""

    model_config = ConfigDict(frozen=True)

    kind: str
    magnitude: float
    unit: str
    n: int
    detail: str
    concerning: bool

    def __str__(self) -> str:
        flag = "!" if self.concerning else "ok"
        return f"[{flag}] {self.kind}: {self.magnitude:+.3f} {self.unit} (n={self.n})"


def _insufficient(kind: str, unit: str, n: int, detail: str) -> BiasFinding:
    """A finding that says the question could not be asked.

    Returned rather than raising, and never as a silent zero: "not measured" and
    "measured as unbiased" are very different claims about a judge.
    """
    return BiasFinding(kind=kind, magnitude=0.0, unit=unit, n=n, detail=detail, concerning=False)


def _residuals(judgements: Sequence[Judgement], dataset: Dataset) -> tuple[list[Case], np.ndarray]:
    """Cases with both a human label and a usable judgement, plus judge minus human."""
    labels = {c.id: c for c in dataset.cases if c.has_human_label}
    cases: list[Case] = []
    residuals: list[float] = []

    for judgement in judgements:
        case = labels.get(judgement.case_id)
        if case is None or not judgement.ok or case.human_label is None:
            continue
        cases.append(case)
        residuals.append(judgement.score - case.human_label)

    return cases, np.asarray(residuals, dtype=float)


def measure_verbosity_bias(
    judgements: Sequence[Judgement],
    dataset: Dataset,
    *,
    scale: Scale,
    threshold: float = CONCERN_THRESHOLD,
) -> BiasFinding:
    """How much the judge rewards length beyond what the humans did.

    The magnitude is the correlation between answer length and the judge's
    residual. Positive means the judge is more generous to long answers than the
    humans were; negative means it is harsher.

    Measuring against the residual rather than the raw score is the whole
    method. Long answers frequently *are* better, so a length-to-score
    correlation mostly measures that, and would flag a perfectly calibrated
    judge as biased.
    """
    cases, residuals = _residuals(judgements, dataset)
    if len(cases) < 3:
        return _insufficient(
            "verbosity",
            "correlation",
            len(cases),
            "need at least 3 labelled cases to estimate a correlation",
        )

    lengths = np.asarray([len(c.output) for c in cases], dtype=float)
    if np.std(lengths) == 0.0 or np.std(residuals) == 0.0:
        return _insufficient(
            "verbosity",
            "correlation",
            len(cases),
            "no variation in answer length or in residuals",
        )

    correlation = float(np.corrcoef(lengths, residuals)[0, 1])

    # Also express it in something physical: scale points per 1000 characters.
    slope = float(np.polyfit(lengths, residuals, 1)[0]) * 1000.0
    direction = "more generous" if correlation > 0 else "harsher"

    return BiasFinding(
        kind="verbosity",
        magnitude=correlation,
        unit="correlation",
        n=len(cases),
        detail=(
            f"length correlates {correlation:+.2f} with judge-minus-human residual; "
            f"the judge is {direction} to longer answers by roughly "
            f"{abs(slope):.2f} points per 1000 characters "
            f"on a {scale.minimum}-{scale.maximum} scale"
        ),
        concerning=abs(correlation) > threshold,
    )


def measure_self_preference(
    judgements: Sequence[Judgement],
    dataset: Dataset,
    *,
    judge_family: str,
    threshold: float = CONCERN_THRESHOLD,
) -> BiasFinding:
    """Whether the judge scores its own family's output above everyone else's.

    Needs cases to record who wrote the answer, in
    ``metadata["generator_family"]``. Without that the question cannot be asked,
    and this says so rather than returning a misleading zero.
    """
    cases, residuals = _residuals(judgements, dataset)

    own: list[float] = []
    other: list[float] = []
    for case, residual in zip(cases, residuals, strict=True):
        family = case.metadata.get(GENERATOR_FAMILY_KEY)
        if family is None:
            continue
        (own if family == judge_family else other).append(residual)

    if not own or not other:
        return _insufficient(
            "self_preference",
            "scale points",
            len(own) + len(other),
            f"needs cases from the judge's own family ({judge_family}) and from "
            f"others, tagged in metadata[{GENERATOR_FAMILY_KEY!r}]; "
            f"found {len(own)} own and {len(other)} other",
        )

    difference = float(np.mean(own) - np.mean(other))
    return BiasFinding(
        kind="self_preference",
        magnitude=difference,
        unit="scale points",
        n=len(own) + len(other),
        detail=(
            f"judge family {judge_family!r} scores its own output "
            f"{difference:+.2f} points relative to others, after subtracting "
            f"human labels ({len(own)} own vs {len(other)} other)"
        ),
        concerning=abs(difference) > threshold,
    )


def _score_from(text: str, scale: Scale) -> float | None:
    """Pull a numeric score out of a judge response, or ``None`` if unreadable."""
    try:
        payload = extract_json(text)
    except JudgeParseError:
        return None

    raw = payload.get("score")
    if isinstance(raw, int | float):
        return scale.clamp(float(raw))

    criteria = payload.get("criterion_scores")
    if isinstance(criteria, dict):
        values = [float(v) for v in criteria.values() if isinstance(v, int | float)]
        if values:
            return scale.clamp(sum(values) / len(values))
    return None


async def measure_position_bias(
    judge: Judge,
    cases: Sequence[Case],
    *,
    threshold: float = CONCERN_THRESHOLD,
) -> BiasFinding:
    """Whether identical content scores differently depending on its slot.

    Each case is scored twice from the same prompt, distinguished only by a
    marker saying whether the answer was presented first or second. The content,
    the rubric, the evidence and the temperature are all held constant, so any
    difference in score is attributable to slot order alone.

    The magnitude is the mean of ``first - second`` in scale points. Positive
    means the judge favours whatever it sees first, which is the direction
    reported in the literature. 0.0 means slot order made no difference.

    An earlier version of this tried to read a ``winner`` field from a pairwise
    response. That silently defaulted to "A" for any judge that returns scores
    rather than a verdict, producing a 100% flip rate on every input - a
    measurement that looked alarming and meant nothing. Scoring identical
    content in both slots avoids depending on a response shape at all.
    """
    if not cases:
        return _insufficient("position", "scale points", 0, "no cases supplied")

    scale = judge.rubric.scale
    deltas: list[float] = []
    unreadable = 0

    for case in cases:
        base = judge.build_prompt(case)
        scores: list[float | None] = []
        for marker in (SLOT_FIRST, SLOT_SECOND):
            completion = await judge.provider.complete(
                CompletionRequest(prompt=f"{base}\n\n{marker}", temperature=0.0)
            )
            scores.append(_score_from(completion.text, scale))

        first, second = scores
        if first is None or second is None:
            unreadable += 1
            continue
        deltas.append(first - second)

    if not deltas:
        return _insufficient(
            "position",
            "scale points",
            len(cases),
            f"no case produced two readable scores ({unreadable} unreadable)",
        )

    mean_delta = float(np.mean(deltas))
    moved = sum(1 for d in deltas if abs(d) > 1e-9)
    favoured = "first" if mean_delta > 0 else "second"

    detail = (
        f"the same answer scored {mean_delta:+.2f} points differently between "
        f"slots; {moved} of {len(deltas)} cases moved at all"
    )
    if moved:
        detail += f", favouring the {favoured} slot"
    if unreadable:
        detail += f" ({unreadable} case(s) unreadable)"

    return BiasFinding(
        kind="position",
        magnitude=mean_delta,
        unit="scale points",
        n=len(deltas),
        detail=detail,
        concerning=abs(mean_delta) > threshold,
    )
