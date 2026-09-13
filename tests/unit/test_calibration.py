"""Calibration maths.

Several of these assert against values computed by hand or taken from the
inter-rater reliability literature, because a statistics module that is subtly
wrong is worse than one that is obviously missing: it produces numbers people
will act on.
"""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.core.calibration import (
    _kappa,
    calibrate,
    calibrate_run,
    interpret,
)
from judgekit.core.judge import Judgement
from judgekit.core.models import Case, Dataset, Verdict
from judgekit.core.rubric import Rubric, Scale

SCALE = Scale(minimum=1, maximum=5, pass_at=4)


def judgement(rubric: Rubric, case_id: str, score: float, *, ok: bool = True) -> Judgement:
    return Judgement(
        case_id=case_id,
        score=score,
        verdict=Verdict.PASS if ok else Verdict.ERROR,
        rubric=rubric.ref,
        error=None if ok else "boom",
    )


class TestKappaAgainstKnownValues:
    def test_textbook_two_by_two(self) -> None:
        """A worked example: p_o = 0.70, p_e = 0.50, kappa = 0.40.

        20 + 15 agreements out of 50 gives observed agreement 0.70; the
        marginals give expected agreement 0.50; (0.70 - 0.50) / (1 - 0.50) = 0.40.
        """
        counts = np.array([[20, 5], [10, 15]])
        assert _kappa(counts, quadratic=False) == pytest.approx(0.4)

    def test_perfect_agreement_is_one(self) -> None:
        assert _kappa(np.array([[10, 0], [0, 10]]), quadratic=False) == pytest.approx(1.0)

    def test_a_single_category_counts_as_perfect(self) -> None:
        """No disagreement was possible, so calling it undefined helps nobody."""
        assert _kappa(np.array([[7]]), quadratic=False) == 1.0

    def test_everything_in_one_cell_of_a_larger_matrix(self) -> None:
        counts = np.zeros((5, 5), dtype=np.int64)
        counts[2, 2] = 9
        assert _kappa(counts, quadratic=True) == 1.0

    def test_empty_matrix_is_zero(self) -> None:
        assert _kappa(np.zeros((3, 3), dtype=np.int64), quadratic=False) == 0.0


class TestQuadraticWeighting:
    def test_near_misses_are_punished_less_than_far_ones(self) -> None:
        """The reason quadratic kappa is the headline metric.

        Both judges below get five of six cases exactly right and miss one. The
        only difference is the size of the miss: one point versus four. Exact
        agreement cannot tell them apart, because it only asks whether the
        raters matched. On an ordinal quality scale that is the wrong model -
        being one point out is not the same mistake as being four points out -
        and quadratic weighting is what separates them.
        """
        human = [1, 2, 3, 4, 5, 5]
        near = calibrate(human, [1, 2, 3, 4, 5, 4], scale=SCALE)
        far = calibrate(human, [1, 2, 3, 4, 5, 1], scale=SCALE)

        assert near.exact_agreement == far.exact_agreement
        assert near.quadratic_kappa > far.quadratic_kappa
        assert near.mean_absolute_error < far.mean_absolute_error


class TestAgreementVersusAccuracy:
    def test_a_constant_offset_keeps_ranking_but_loses_agreement(self) -> None:
        """Different problems, different fixes.

        A judge one point low ranks everything correctly. Spearman says it is
        excellent, kappa says it is mediocre, and both are right. The fix is a
        recalibrated threshold, not a rewritten rubric.
        """
        report = calibrate([1, 2, 3, 4], [2, 3, 4, 5], scale=SCALE)

        assert report.spearman == pytest.approx(1.0)
        assert report.quadratic_kappa < 1.0
        assert report.systematic_offset == pytest.approx(1.0)

    def test_perfect_agreement(self) -> None:
        report = calibrate([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], scale=SCALE)
        assert report.quadratic_kappa == pytest.approx(1.0)
        assert report.krippendorff_alpha == pytest.approx(1.0)
        assert report.exact_agreement == 1.0
        assert report.systematic_offset == 0.0
        assert report.mean_absolute_error == 0.0

    def test_inverted_ranking_is_negative(self) -> None:
        report = calibrate([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], scale=SCALE)
        assert report.spearman == pytest.approx(-1.0)
        assert report.quadratic_kappa < 0


class TestReportFields:
    def test_within_one_is_more_forgiving_than_exact(self) -> None:
        report = calibrate([3, 3, 3, 3], [3, 4, 4, 4], scale=SCALE)
        assert report.exact_agreement == 0.25
        assert report.within_one == 1.0

    def test_means_are_reported(self) -> None:
        report = calibrate([2, 4], [3, 5], scale=SCALE)
        assert report.human_mean == 3.0
        assert report.judge_mean == 4.0

    def test_meets_applies_the_floor_to_quadratic_kappa(self) -> None:
        report = calibrate([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], scale=SCALE)
        assert report.meets(min_kappa=0.9)
        assert not report.meets(min_kappa=1.1)

    def test_summary_names_the_headline_numbers(self) -> None:
        summary = calibrate([1, 3, 5], [1, 3, 5], scale=SCALE).summary()
        assert "quadratic kappa" in summary
        assert "Spearman" in summary

    def test_ties_do_not_break_rank_correlation(self) -> None:
        report = calibrate([3, 3, 5], [3, 3, 5], scale=SCALE)
        assert report.spearman == pytest.approx(1.0)

    def test_a_constant_judge_correlates_at_zero(self) -> None:
        """No variance means no correlation, rather than a divide-by-zero."""
        report = calibrate([1, 2, 3, 4], [3, 3, 3, 3], scale=SCALE)
        assert report.spearman == 0.0
        assert report.pearson == 0.0


class TestConfusionMatrix:
    def test_rows_are_human_and_columns_are_judge(self) -> None:
        report = calibrate([1, 1, 5], [5, 5, 1], scale=SCALE)
        counts = report.confusion.counts
        categories = report.confusion.categories

        human_one = categories.index(1)
        judge_five = categories.index(5)
        assert counts[human_one][judge_five] == 2

    def test_renders_a_readable_table(self) -> None:
        rendered = calibrate([3], [3], scale=SCALE).confusion.render()
        assert "human\\judge" in rendered
        assert len(rendered.splitlines()) == 6  # header plus five categories

    def test_out_of_range_labels_are_clipped_not_fatal(self) -> None:
        """A label outside the scale is a data error, not a reason to crash mid-run."""
        report = calibrate([9.0, 1.0], [5.0, 1.0], scale=SCALE)
        assert report.n == 2

    def test_fractional_scores_are_bucketed(self) -> None:
        report = calibrate([4.0], [3.6], scale=SCALE)
        categories = report.confusion.categories
        assert report.confusion.counts[categories.index(4)][categories.index(4)] == 1


class TestValidation:
    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="3 human labels but 2 judge scores"):
            calibrate([1, 2, 3], [1, 2], scale=SCALE)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty set of labels"):
            calibrate([], [], scale=SCALE)


class TestInterpret:
    @pytest.mark.parametrize(
        ("kappa", "expected"),
        [
            (0.95, "almost perfect"),
            (0.70, "substantial"),
            (0.50, "moderate"),
            (0.30, "fair"),
            (0.10, "slight"),
            (-0.50, "none or worse than chance"),
        ],
    )
    def test_conventional_bands(self, kappa: float, expected: str) -> None:
        assert interpret(kappa) == expected


class TestCalibrateRun:
    def test_pairs_judgements_with_labels(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(
                Case(id="a", input="q", output="o", human_label=4.0),
                Case(id="b", input="q", output="o", human_label=2.0),
            ),
        )
        judgements = [judgement(rubric, "a", 4.0), judgement(rubric, "b", 2.0)]

        report = calibrate_run(judgements, dataset, scale=SCALE)

        assert report.n == 2
        assert report.exact_agreement == 1.0

    def test_skips_unlabelled_cases(self, rubric: Rubric) -> None:
        dataset = Dataset(
            name="d",
            cases=(
                Case(id="a", input="q", output="o", human_label=4.0),
                Case(id="b", input="q", output="o"),
            ),
        )
        judgements = [judgement(rubric, "a", 4.0), judgement(rubric, "b", 1.0)]
        assert calibrate_run(judgements, dataset, scale=SCALE).n == 1

    def test_skips_errored_judgements(self, rubric: Rubric) -> None:
        """A judgement that never happened is not a disagreement."""
        dataset = Dataset(
            name="d",
            cases=(
                Case(id="a", input="q", output="o", human_label=4.0),
                Case(id="b", input="q", output="o", human_label=4.0),
            ),
        )
        judgements = [judgement(rubric, "a", 4.0), judgement(rubric, "b", 1.0, ok=False)]

        assert calibrate_run(judgements, dataset, scale=SCALE).n == 1

    def test_raises_when_nothing_can_be_paired(self, rubric: Rubric) -> None:
        dataset = Dataset(name="d", cases=(Case(id="a", input="q", output="o"),))
        with pytest.raises(ValueError, match="nothing to calibrate against"):
            calibrate_run([judgement(rubric, "a", 4.0)], dataset, scale=SCALE)

    def test_ignores_judgements_for_unknown_cases(self, rubric: Rubric) -> None:
        dataset = Dataset(name="d", cases=(Case(id="a", input="q", output="o", human_label=4.0),))
        judgements = [judgement(rubric, "a", 4.0), judgement(rubric, "ghost", 1.0)]
        assert calibrate_run(judgements, dataset, scale=SCALE).n == 1
