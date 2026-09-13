"""The command line.

Exit codes get as much attention as output here, because these commands are
meant to run in CI and a gate that reports a problem while exiting 0 is worse
than no gate at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgekit.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets" / "example.jsonl"
RUBRIC_DIR = REPO_ROOT / "rubrics"
V1 = RUBRIC_DIR / "answer-quality.v1.yaml"
V2 = RUBRIC_DIR / "answer-quality.v2.yaml"

runner = CliRunner()


@pytest.fixture
def rubric_copy(tmp_path: Path) -> Path:
    """A writable copy of the shipped rubrics, with a current lockfile."""
    target = tmp_path / "rubrics"
    shutil.copytree(RUBRIC_DIR, target)
    return target


class TestValidate:
    def test_reports_what_is_in_a_dataset(self) -> None:
        result = runner.invoke(app, ["validate", str(DATASET)])
        assert result.exit_code == 0
        assert "8 cases" in result.stdout
        assert "8 with human labels" in result.stdout

    def test_missing_dataset_exits_one(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path / "absent.jsonl")])
        assert result.exit_code == 1

    def test_notes_when_a_dataset_cannot_calibrate(self, tmp_path: Path) -> None:
        path = tmp_path / "unlabelled.jsonl"
        path.write_text(json.dumps({"id": "a", "input": "q", "output": "o"}) + "\n", "utf-8")

        result = runner.invoke(app, ["validate", str(path)])

        assert result.exit_code == 0, "unlabelled is a note, not a failure"
        assert "cannot measure this judge" in result.stdout


class TestRun:
    def test_judges_a_dataset(self) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2)])
        assert result.exit_code == 0
        assert "passed" in result.stdout

    def test_always_prints_the_rubric_fingerprint(self) -> None:
        """A score without its rubric is a number, not a measurement."""
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2)])
        assert "answer-quality@v2" in result.stdout

    def test_writes_a_self_contained_html_report(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--out", str(out)])

        assert result.exit_code == 0
        html = out.read_text(encoding="utf-8")
        assert html.startswith("<!doctype html>")
        # No CDN, no webfont, no analytics: eval output contains customer data.
        assert "http://" not in html
        assert "https://" not in html

    def test_report_shows_provenance_and_flags(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--out", str(out)])
        html = out.read_text(encoding="utf-8")

        assert "answer-quality" in html
        assert "no_evidence" in html
        assert "fingerprint" in html

    def test_saved_run_can_be_read_back(self, tmp_path: Path) -> None:
        saved = tmp_path / "run.json"
        runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--save", str(saved)])

        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["rubric"]["version"] == "v2"
        assert len(payload["judgements"]) == 8

    def test_filters_by_domain(self) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--domain", "inventory"])
        assert result.exit_code == 0
        assert "2 cases" in result.stdout

    def test_an_empty_domain_exits_one(self) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--domain", "nope"])
        assert result.exit_code == 1

    def test_min_pass_rate_gate_fails_a_bad_run(self) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--min-pass-rate", "0.99"])
        assert result.exit_code == 1

    def test_min_pass_rate_gate_allows_a_good_run(self) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(V2), "--min-pass-rate", "0.0"])
        assert result.exit_code == 0

    def test_a_missing_rubric_exits_one(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(tmp_path / "absent.yaml")])
        assert result.exit_code == 1


class TestRubricCommands:
    def test_list_shows_fingerprints(self) -> None:
        result = runner.invoke(app, ["rubric", "list", "-d", str(RUBRIC_DIR)])
        assert result.exit_code == 0
        assert "answer-quality@v1" in result.stdout
        assert "answer-quality@v2" in result.stdout

    def test_verify_passes_on_the_shipped_rubrics(self) -> None:
        result = runner.invoke(app, ["rubric", "verify", "-d", str(RUBRIC_DIR)])
        assert result.exit_code == 0

    def test_lock_then_verify_roundtrips(self, tmp_path: Path) -> None:
        target = tmp_path / "rubrics"
        target.mkdir()
        shutil.copy(V2, target / "v2.yaml")

        assert runner.invoke(app, ["rubric", "verify", "-d", str(target)]).exit_code == 1
        assert runner.invoke(app, ["rubric", "lock", "-d", str(target)]).exit_code == 0
        assert runner.invoke(app, ["rubric", "verify", "-d", str(target)]).exit_code == 0

    def test_verify_catches_an_in_place_edit(self, rubric_copy: Path) -> None:
        """The gate the whole project exists to provide."""
        target = rubric_copy / "answer-quality.v2.yaml"
        text = target.read_text(encoding="utf-8")
        target.write_text(text.replace("score of 4 or above", "score of 5 or above"), "utf-8")

        result = runner.invoke(app, ["rubric", "verify", "-d", str(rubric_copy)])

        assert result.exit_code == 1
        assert "answer-quality@v2" in result.output

    def test_the_violation_kind_survives_rich_markup(self, rubric_copy: Path) -> None:
        """Regression test: Rich parses square brackets as style tags.

        ``[edited-without-bump]`` was being read as markup and silently dropped,
        removing the single most useful word from the CI log at exactly the
        moment somebody is reading it to work out what broke.
        """
        target = rubric_copy / "answer-quality.v2.yaml"
        text = target.read_text(encoding="utf-8")
        target.write_text(text.replace("score of 4 or above", "score of 5 or above"), "utf-8")

        result = runner.invoke(app, ["rubric", "verify", "-d", str(rubric_copy)])

        assert "edited-without-bump" in result.output

    def test_verify_flags_a_rubric_missing_from_disk(self, rubric_copy: Path) -> None:
        (rubric_copy / "answer-quality.v1.yaml").unlink()
        result = runner.invoke(app, ["rubric", "verify", "-d", str(rubric_copy)])
        assert result.exit_code == 1
        assert "missing" in result.output

    def test_verify_on_a_missing_directory_exits_one(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["rubric", "verify", "-d", str(tmp_path / "nowhere")])
        assert result.exit_code == 1

    def test_diff_reports_identical_rubrics(self) -> None:
        result = runner.invoke(app, ["rubric", "diff", str(V2), str(V2)])
        assert result.exit_code == 0
        assert "identical" in result.stdout

    def test_diff_explains_why_two_versions_are_incomparable(self) -> None:
        result = runner.invoke(app, ["rubric", "diff", str(V1), str(V2)])
        assert result.exit_code == 0
        assert "not comparable" in result.stdout
        assert "requires_evidence" in result.stdout

    def test_diff_criteria_lists_are_not_eaten_by_markup(self) -> None:
        """Python list reprs contain brackets too, so they need escaping."""
        result = runner.invoke(app, ["rubric", "diff", str(V1), str(V2)])
        assert "groundedness" in result.stdout


class TestCompare:
    def _save(self, tmp_path: Path, name: str, rubric: Path) -> Path:
        path = tmp_path / name
        result = runner.invoke(app, ["run", str(DATASET), "-r", str(rubric), "--save", str(path)])
        assert result.exit_code == 0
        return path

    def test_compares_two_runs_under_one_rubric(self, tmp_path: Path) -> None:
        a = self._save(tmp_path, "a.json", V2)
        b = self._save(tmp_path, "b.json", V2)

        result = runner.invoke(app, ["compare", str(a), str(b)])

        assert result.exit_code == 0
        assert "pass rate" in result.stdout

    def test_refuses_to_compare_across_rubric_versions(self, tmp_path: Path) -> None:
        """Exits 1 rather than printing a meaningless delta."""
        v1_run = self._save(tmp_path, "v1.json", V1)
        v2_run = self._save(tmp_path, "v2.json", V2)

        result = runner.invoke(app, ["compare", str(v1_run), str(v2_run)])

        assert result.exit_code == 1
        assert "differs in version" in result.output

    def test_unreadable_runs_exit_one(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["compare", str(bad), str(bad)])
        assert result.exit_code == 1


class TestHelp:
    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "Measure the judge" in result.stdout

    def test_every_command_is_listed(self) -> None:
        result = runner.invoke(app, ["--help"])
        for command in ("run", "validate", "compare", "rubric"):
            assert command in result.stdout
