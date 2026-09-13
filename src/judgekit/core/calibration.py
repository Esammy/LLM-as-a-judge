"""Measuring the judge against human labels.

This is the module the project is named for. Everything else grades answers;
this grades *the grader*.

The motivating failure is mundane and common: a team stands up an LLM judge,
watches a number, and ships against it for months without ever checking whether
that number tracks what their own reviewers would have said. When somebody
finally sits down and scores fifty outputs by hand, agreement turns out to be
poor - and every decision taken on the strength of the old number was taken on
the strength of nothing.

Four things worth knowing about what is computed here:

**Quadratic weighted kappa is the headline, not plain Cohen's kappa.** Plain
kappa treats every disagreement as equally bad, so a judge that says 5 where a
human said 4 is penalised exactly as hard as one that says 5 where a human said
1. On an ordinal quality scale that is the wrong model, and it makes good judges
look broken.

**Agreement and accuracy are different questions.** A judge can track human
rankings almost perfectly while sitting a full point low. Spearman would look
excellent, kappa poor. Both are reported, plus the systematic offset, because
the fix differs: a ranking problem needs a better rubric, an offset needs a
recalibrated threshold.

**Chance-corrected metrics punish narrow label distributions.** If every human
label is 4 or 5, agreeing by accident is easy, so kappa is harsh. That is
correct behaviour, not a bug, but it does mean a dataset with no bad answers in
it cannot tell you much about a judge.

**None of these thresholds are laws.** The interpretation bands are conventions
from the inter-rater reliability literature, useful for orientation and no
substitute for looking at where the disagreements actually fall.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from judgekit.core.judge import Judgement
from judgekit.core.models import Dataset
from judgekit.core.rubric import Scale

# Landis & Koch (1977). Widely used, entirely conventional, and worth treating
# as orientation rather than as a standard anybody agreed to.
_BANDS: tuple[tuple[float, str], ...] = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.01, "none or worse than chance"),
)


def interpret(kappa: float) -> str:
    """Describe an agreement coefficient in the conventional bands."""
    for floor, label in _BANDS:
        if kappa >= floor:
            return label
    return "none or worse than chance"  # pragma: no cover - unreachable given the bands


class ConfusionMatrix(BaseModel):
    """Human label against judge score, bucketed to whole points."""

    model_config = ConfigDict(frozen=True)

    categories: tuple[int, ...]
    counts: tuple[tuple[int, ...], ...]
    """``counts[human][judge]`` - rows are human labels, columns judge scores."""

    def render(self) -> str:
        """A small fixed-width table, for terminals and reports."""
        header = "human\\judge " + " ".join(f"{c:>5}" for c in self.categories)
        rows = [
            f"{cat:>11} " + " ".join(f"{n:>5}" for n in row)
            for cat, row in zip(self.categories, self.counts, strict=True)
        ]
        return "\n".join([header, *rows])


class AgreementReport(BaseModel):
    """How well a judge agrees with the humans who labelled the same cases."""

    model_config = ConfigDict(frozen=True)

    n: int

    cohens_kappa: float
    """Chance-corrected agreement on exact matches. Harsh on ordinal scales."""

    quadratic_kappa: float
    """Chance-corrected agreement that weights near-misses lightly. The headline."""

    krippendorff_alpha: float
    """Interval-scale reliability. Handles the ordinal case without bucketing."""

    spearman: float
    """Rank correlation: does the judge order cases the way humans do?"""

    pearson: float
    mean_absolute_error: float

    systematic_offset: float
    """Judge mean minus human mean. Positive means the judge is generous."""

    judge_mean: float
    human_mean: float
    exact_agreement: float
    within_one: float
    confusion: ConfusionMatrix

    @property
    def interpretation(self) -> str:
        return interpret(self.quadratic_kappa)

    def meets(self, *, min_kappa: float) -> bool:
        """Whether the judge clears an agreement floor."""
        return self.quadratic_kappa >= min_kappa

    def summary(self) -> str:
        return (
            f"n={self.n} quadratic kappa={self.quadratic_kappa:.3f} "
            f"({self.interpretation}), Spearman={self.spearman:.3f}, "
            f"offset={self.systematic_offset:+.2f}"
        )


def _confusion(
    human: np.ndarray, judge: np.ndarray, categories: Sequence[int]
) -> tuple[np.ndarray, ConfusionMatrix]:
    index = {c: i for i, c in enumerate(categories)}
    size = len(categories)
    counts = np.zeros((size, size), dtype=np.int64)

    # Bucket to whole points, then clip: a human label outside the rubric scale
    # is a data error, but it should land in the nearest bucket rather than
    # raising a KeyError halfway through a calibration run.
    low, high = categories[0], categories[-1]
    human_bucket = np.clip(np.rint(human), low, high).astype(np.int64).tolist()
    judge_bucket = np.clip(np.rint(judge), low, high).astype(np.int64).tolist()

    for h, j in zip(human_bucket, judge_bucket, strict=True):
        counts[index[h], index[j]] += 1

    return counts, ConfusionMatrix(
        categories=tuple(categories),
        counts=tuple(tuple(int(v) for v in row) for row in counts),
    )


def _kappa(counts: np.ndarray, *, quadratic: bool) -> float:
    """Cohen's kappa over a confusion matrix, optionally quadratically weighted.

    Returns 1.0 when there is no possible disagreement to correct for - a single
    category used by both raters - because calling perfect agreement "undefined"
    is less useful than calling it perfect.
    """
    n = counts.sum()
    if n == 0:
        return 0.0

    observed = counts / n
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)

    size = counts.shape[0]
    if size == 1:
        return 1.0

    if quadratic:
        idx = np.arange(size)
        weights = ((idx[:, None] - idx[None, :]) ** 2) / ((size - 1) ** 2)
    else:
        weights = 1.0 - np.eye(size)

    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        # Both raters put everything in one category: no expected disagreement,
        # so agreement is perfect by construction.
        return 1.0
    return float(1.0 - (weights * observed).sum() / denominator)


def _krippendorff_alpha_interval(human: np.ndarray, judge: np.ndarray) -> float:
    """Krippendorff's alpha for interval data with two raters.

    Observed disagreement is the mean squared difference per unit. Expected
    disagreement comes from the pooled spread of every rating either rater gave,
    which is what makes the coefficient chance-corrected.
    """
    n = len(human)
    if n < 2:
        return 0.0

    observed = float(np.mean((human - judge) ** 2))

    pooled = np.concatenate([human, judge])
    size = len(pooled)
    variance = float(np.var(pooled))
    expected = 2.0 * size * variance / (size - 1)

    if expected == 0.0:
        # Every rating identical everywhere: no spread to disagree within.
        return 1.0 if observed == 0.0 else 0.0
    return float(1.0 - observed / expected)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort the correlation."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    # Average the ranks within each group of equal values.
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 when either side is constant."""
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calibrate(
    human_labels: Sequence[float],
    judge_scores: Sequence[float],
    *,
    scale: Scale,
) -> AgreementReport:
    """Compare a judge's scores against human labels for the same cases.

    Args:
        human_labels: Ground-truth scores, in case order.
        judge_scores: The judge's scores, in the same order.
        scale: The rubric scale both sets of scores live on.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if len(human_labels) != len(judge_scores):
        raise ValueError(
            f"got {len(human_labels)} human labels but {len(judge_scores)} judge scores"
        )
    if not human_labels:
        raise ValueError("cannot calibrate against an empty set of labels")

    human = np.asarray(human_labels, dtype=float)
    judge = np.asarray(judge_scores, dtype=float)

    categories = list(range(scale.minimum, scale.maximum + 1))
    counts, matrix = _confusion(human, judge, categories)

    differences = np.abs(human - judge)

    return AgreementReport(
        n=len(human),
        cohens_kappa=_kappa(counts, quadratic=False),
        quadratic_kappa=_kappa(counts, quadratic=True),
        krippendorff_alpha=_krippendorff_alpha_interval(human, judge),
        spearman=_correlation(_rank(human), _rank(judge)),
        pearson=_correlation(human, judge),
        mean_absolute_error=float(np.mean(differences)),
        systematic_offset=float(np.mean(judge) - np.mean(human)),
        judge_mean=float(np.mean(judge)),
        human_mean=float(np.mean(human)),
        exact_agreement=float(np.mean(np.round(human) == np.round(judge))),
        within_one=float(np.mean(differences <= 1.0)),
        confusion=matrix,
    )


def calibrate_run(
    judgements: Sequence[Judgement],
    dataset: Dataset,
    *,
    scale: Scale,
) -> AgreementReport:
    """Calibrate a run against the human labels carried by its dataset.

    Cases without a human label are skipped, and errored judgements are skipped
    too - a judgement that never happened is not a disagreement.

    Raises:
        ValueError: If no case has both a human label and a usable judgement.
    """
    labels = {c.id: c.human_label for c in dataset.cases if c.has_human_label}

    paired = [
        (labels[j.case_id], j.score)
        for j in judgements
        if j.ok and j.case_id in labels and labels[j.case_id] is not None
    ]
    if not paired:
        raise ValueError(
            "no case has both a human label and a successful judgement, so there "
            "is nothing to calibrate against"
        )

    human = [h for h, _ in paired if h is not None]
    scores = [s for h, s in paired if h is not None]
    return calibrate(human, scores, scale=scale)
