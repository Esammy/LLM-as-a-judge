"""Deterministic checks."""

from __future__ import annotations

import pytest

from judgekit.core.checks import (
    GroundedFigures,
    MaxLength,
    MustContain,
    MustNotContain,
    NonEmpty,
    NotARefusal,
    ValidJson,
    all_passed,
    run_checks,
)
from judgekit.core.models import Case, Evidence, ToolCall


def case(output: str, *, evidence: Evidence | None = None) -> Case:
    return Case(id="c", input="q", output=output, evidence=evidence or Evidence())


class TestNonEmpty:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_fails_blank_output(self, text: str) -> None:
        result = NonEmpty()(case(text))
        assert not result
        assert result.detail == "output is empty"

    def test_passes_real_output(self) -> None:
        assert NonEmpty()(case("an answer"))


class TestMaxLength:
    def test_passes_within_budget(self) -> None:
        assert MaxLength(10)(case("short"))

    def test_fails_over_budget_and_says_by_how_much(self) -> None:
        result = MaxLength(3)(case("far too long"))
        assert not result
        assert "12 characters exceeds the 3 limit" in result.detail

    def test_rejects_a_nonsense_limit(self) -> None:
        with pytest.raises(ValueError, match="limit must be positive"):
            MaxLength(0)

    def test_id_is_overridable(self) -> None:
        assert MaxLength(5, check_id="brevity").id == "brevity"


class TestMustContain:
    def test_passes_when_all_present(self) -> None:
        assert MustContain(["alpha", "beta"])(case("alpha and beta"))

    def test_lists_what_was_missing(self) -> None:
        result = MustContain(["alpha", "gamma"])(case("alpha only"))
        assert not result
        assert "missing: gamma" in result.detail

    def test_is_case_insensitive_by_default(self) -> None:
        assert MustContain(["ALPHA"])(case("alpha"))

    def test_case_sensitivity_is_opt_in(self) -> None:
        assert not MustContain(["ALPHA"], case_sensitive=True)(case("alpha"))

    def test_regex_mode(self) -> None:
        assert MustContain([r"\d{4} USD"], regex=True)(case("that is 4820 USD"))

    def test_regex_mode_respects_case_sensitivity(self) -> None:
        assert not MustContain(["ALPHA"], regex=True, case_sensitive=True)(case("alpha"))


class TestMustNotContain:
    def test_passes_when_absent(self) -> None:
        assert MustNotContain(["Traceback"])(case("a clean answer"))

    def test_lists_what_leaked(self) -> None:
        result = MustNotContain(["Traceback", "api_key"])(case("Traceback (most recent call)"))
        assert not result
        assert "found: Traceback" in result.detail

    def test_has_its_own_default_id(self) -> None:
        assert MustNotContain([]).id == "must_not_contain"


class TestGroundedFigures:
    def test_passes_when_every_figure_is_supported(self) -> None:
        evidence = Evidence(tool_calls=(ToolCall(name="f", arguments={}, result={"total": 48200}),))
        assert GroundedFigures()(case("Revenue was 48200 USD.", evidence=evidence))

    def test_fails_and_names_the_unsupported_figure(self) -> None:
        evidence = Evidence(tool_calls=(ToolCall(name="f", arguments={}, result={"total": 48200}),))
        result = GroundedFigures()(case("Revenue was 99999 USD.", evidence=evidence))
        assert not result
        assert "99999" in result.detail

    def test_normalises_thousands_separators(self) -> None:
        """48,200 and 48200 are the same figure and must compare equal."""
        evidence = Evidence(context=("total was 48200",))
        assert GroundedFigures()(case("Revenue was 48,200 USD.", evidence=evidence))

    def test_normalises_trailing_decimals(self) -> None:
        evidence = Evidence(context=("total was 48200.00",))
        assert GroundedFigures()(case("Revenue was 48200 USD.", evidence=evidence))

    def test_passes_trivially_with_no_figures(self) -> None:
        result = GroundedFigures()(case("Revenue grew substantially."))
        assert result
        assert result.detail == "no figures stated"

    def test_skips_rather_than_fails_when_no_evidence_exists(self) -> None:
        """Unverifiable and wrong are different findings.

        Failing here would make a correct-but-unlogged answer indistinguishable
        from an invented one, which is exactly the conflation this project
        exists to remove.
        """
        result = GroundedFigures()(case("Revenue was 48200 USD."))
        assert result
        assert "unverifiable" in result.detail


class TestNotARefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "I cannot access that.",
            "I can't help with this.",
            "As an AI, I have limits.",
            "I do not have access to those records.",
        ],
    )
    def test_detects_refusals(self, text: str) -> None:
        result = NotARefusal()(case(text))
        assert not result
        assert "looks like a refusal" in result.detail

    def test_passes_a_real_answer(self) -> None:
        assert NotARefusal()(case("Revenue was 48200 USD."))


class TestValidJson:
    def test_passes_valid_json(self) -> None:
        assert ValidJson()(case('{"ok": true}'))

    def test_fails_invalid_json_with_the_reason(self) -> None:
        result = ValidJson()(case("{not json"))
        assert not result
        assert "invalid JSON" in result.detail


class TestRunChecks:
    def test_runs_every_check_even_after_one_fails(self) -> None:
        """A report should list all problems in one pass, not one per build."""
        results = run_checks(case(""), [NonEmpty(), MaxLength(5), ValidJson()])
        assert len(results) == 3
        assert [r.id for r in results] == ["non_empty", "max_length", "valid_json"]

    def test_all_passed_summarises(self) -> None:
        assert all_passed(run_checks(case("fine"), [NonEmpty()]))
        assert not all_passed(run_checks(case(""), [NonEmpty()]))

    def test_no_checks_passes_vacuously(self) -> None:
        assert all_passed(run_checks(case("anything"), []))
