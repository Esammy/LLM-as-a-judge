"""HTML and JSON reporting."""

from __future__ import annotations

from judgekit.core.judge import Judgement
from judgekit.core.models import Flag, Verdict
from judgekit.core.report import load_json, render_html, render_json
from judgekit.core.rubric import Rubric
from judgekit.core.runner import RunResult


def make_run(rubric: Rubric, *judgements: Judgement) -> RunResult:
    return RunResult(
        dataset_name="fixture",
        dataset_version="v1",
        rubric=rubric.ref,
        provider="stub",
        model="stub-judge-v1",
        judgements=judgements,
    )


def judgement(
    rubric: Rubric,
    *,
    case_id: str = "case-1",
    score: float = 4.0,
    verdict: Verdict = Verdict.PASS,
    reasoning: str = "looked fine",
    flags: tuple[Flag, ...] = (),
    error: str | None = None,
) -> Judgement:
    return Judgement(
        case_id=case_id,
        score=score,
        verdict=verdict,
        reasoning=reasoning,
        rubric=rubric.ref,
        flags=flags,
        error=error,
    )


class TestHtml:
    def test_renders_a_complete_document(self, rubric: Rubric) -> None:
        html = render_html(make_run(rubric, judgement(rubric)))
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")

    def test_is_entirely_self_contained(self, rubric: Rubric) -> None:
        """No CDN, no webfont, no analytics.

        Eval output routinely contains customer data, so a report that phones
        out cannot be attached to a CI artifact or opened in a regulated
        environment without a review nobody wants to do.
        """
        html = render_html(make_run(rubric, judgement(rubric)))
        assert "http://" not in html
        assert "https://" not in html
        assert "<script" not in html.lower()

    def test_leads_with_rubric_provenance(self, rubric: Rubric) -> None:
        html = render_html(make_run(rubric, judgement(rubric)))
        assert rubric.fingerprint in html
        assert "test-quality" in html

    def test_shows_flags(self, rubric: Rubric) -> None:
        run = make_run(rubric, judgement(rubric, flags=(Flag.NO_EVIDENCE,)))
        assert "no_evidence" in render_html(run)

    def test_explains_errored_cases_rather_than_hiding_them(self, rubric: Rubric) -> None:
        run = make_run(
            rubric,
            judgement(rubric, case_id="ok"),
            judgement(
                rubric,
                case_id="broken",
                verdict=Verdict.ERROR,
                error="ConnectionError: unreachable",
                score=1.0,
            ),
        )

        html = render_html(run)

        assert "1 case(s) errored" in html
        assert "excluded from the mean score" in html
        assert "ConnectionError: unreachable" in html

    def test_shows_the_error_instead_of_empty_reasoning(self, rubric: Rubric) -> None:
        run = make_run(
            rubric,
            judgement(rubric, verdict=Verdict.ERROR, reasoning="", error="boom", score=1.0),
        )
        assert "boom" in render_html(run)

    def test_includes_a_domain_breakdown_when_domains_exist(self, rubric: Rubric) -> None:
        run = make_run(rubric, judgement(rubric, case_id="a"))
        html = render_html(run, domains={"a": "finance"})
        assert "By domain" in html
        assert "finance" in html

    def test_omits_the_domain_section_for_an_empty_run(self, rubric: Rubric) -> None:
        assert "By domain" not in render_html(make_run(rubric))

    def test_escapes_case_data(self, rubric: Rubric) -> None:
        """Case ids and judge reasoning are untrusted input.

        They come from datasets and model output, both of which can contain
        anything at all. Interpolating them raw into HTML would turn an eval
        report into a script-injection vector against whoever opens it.
        """
        hostile = judgement(
            rubric,
            case_id="<script>alert('xss')</script>",
            reasoning="<img src=x onerror=alert(1)>",
        )

        html = render_html(make_run(rubric, hostile))

        assert "<script>alert" not in html
        assert "<img src=x" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img" in html


class TestJson:
    def test_roundtrips(self, rubric: Rubric) -> None:
        original = make_run(rubric, judgement(rubric, case_id="a", score=4.25))
        restored = load_json(render_json(original))

        assert restored.run_id == original.run_id
        assert restored.rubric == original.rubric
        assert len(restored.judgements) == 1
        assert restored.judgements[0].score == 4.25

    def test_preserves_the_fingerprint_so_comparison_stays_guarded(self, rubric: Rubric) -> None:
        """A saved run must carry enough to refuse a bad comparison later."""
        restored = load_json(render_json(make_run(rubric, judgement(rubric))))
        assert restored.rubric.fingerprint == rubric.fingerprint

    def test_preserves_flags_and_verdicts(self, rubric: Rubric) -> None:
        run = make_run(
            rubric,
            judgement(rubric, verdict=Verdict.FAIL, flags=(Flag.NO_EVIDENCE, Flag.TRUNCATED)),
        )
        restored = load_json(render_json(run))
        assert restored.judgements[0].verdict is Verdict.FAIL
        assert set(restored.judgements[0].flags) == {Flag.NO_EVIDENCE, Flag.TRUNCATED}
