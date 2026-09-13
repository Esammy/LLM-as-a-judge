"""Data model: evidence rendering, dataset loading and dataset invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgekit.core.models import Case, Dataset, Evidence, Flag, ToolCall, Verdict


class TestToolCall:
    def test_renders_name_arguments_and_result(self) -> None:
        call = ToolCall(name="get_revenue", arguments={"month": "2026-03"}, result={"total": 48200})
        rendered = call.render()
        assert "get_revenue" in rendered
        assert "2026-03" in rendered
        assert "48200" in rendered

    def test_renders_arguments_deterministically(self) -> None:
        """Key order must not vary, or identical evidence would hash differently."""
        a = ToolCall(name="f", arguments={"b": 2, "a": 1}, result=None)
        b = ToolCall(name="f", arguments={"a": 1, "b": 2}, result=None)
        assert a.render() == b.render()

    def test_renders_unserialisable_results_without_crashing(self) -> None:
        call = ToolCall(name="f", arguments={}, result={"when": object()})
        assert "f(" in call.render()


class TestEvidence:
    def test_empty_evidence_says_so(self) -> None:
        evidence = Evidence()
        assert evidence.is_empty
        assert evidence.render() == "(no evidence supplied)"

    def test_renders_tool_calls_and_context_together(self) -> None:
        evidence = Evidence(
            tool_calls=(ToolCall(name="lookup", arguments={}, result=7),),
            context=("a retrieved passage",),
        )
        rendered = evidence.render()
        assert not evidence.is_empty
        assert "Tool calls:" in rendered
        assert "Retrieved context:" in rendered
        assert "a retrieved passage" in rendered

    def test_context_only_evidence_is_not_empty(self) -> None:
        assert not Evidence(context=("something",)).is_empty


class TestCase:
    def test_defaults_to_no_evidence_and_no_label(self) -> None:
        case = Case(id="c", input="q", output="a")
        assert case.evidence.is_empty
        assert not case.has_human_label

    def test_reports_a_human_label_of_zero(self) -> None:
        """Zero is a real score; only None means unlabelled."""
        assert Case(id="c", input="q", output="a", human_label=0.0).has_human_label

    def test_is_immutable(self, grounded_case: Case) -> None:
        with pytest.raises(ValueError, match="frozen"):
            grounded_case.output = "tampered"


class TestDataset:
    def test_rejects_duplicate_case_ids(self) -> None:
        case = Case(id="same", input="q", output="a")
        with pytest.raises(ValueError, match="duplicate case ids"):
            Dataset(name="d", cases=(case, case))

    def test_supports_len_and_iteration(self, dataset: Dataset) -> None:
        assert len(dataset) == 3
        assert [c.id for c in dataset] == ["grounded", "ungrounded", "evidenceless"]

    def test_labelled_selects_only_scored_cases(self) -> None:
        data = Dataset(
            name="d",
            cases=(
                Case(id="a", input="q", output="o", human_label=3.0),
                Case(id="b", input="q", output="o"),
            ),
        )
        assert [c.id for c in data.labelled] == ["a"]

    def test_filter_by_domain(self, dataset: Dataset) -> None:
        assert len(dataset.filter(domain="finance")) == 3
        assert len(dataset.filter(domain="nothing-here")) == 0

    def test_filter_by_tags_requires_all_of_them(self) -> None:
        data = Dataset(
            name="d",
            cases=(
                Case(id="both", input="q", output="o", tags=("x", "y")),
                Case(id="one", input="q", output="o", tags=("x",)),
            ),
        )
        assert [c.id for c in data.filter(tags=["x"])] == ["both", "one"]
        assert [c.id for c in data.filter(tags=["x", "y"])] == ["both"]

    def test_filter_returns_a_new_dataset(self, dataset: Dataset) -> None:
        narrowed = dataset.filter(domain="nothing")
        assert len(dataset) == 3
        assert len(narrowed) == 0
        assert narrowed.name == dataset.name


class TestDatasetLoading:
    def test_loads_the_shipped_example(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        data = Dataset.from_file(repo_root / "datasets" / "example.jsonl")
        assert len(data) == 8
        assert len(data.labelled) == 8
        assert data.name == "example"

    def test_jsonl_takes_its_name_from_the_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "my-suite.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "q", "output": "o"}) + "\n", encoding="utf-8"
        )
        assert Dataset.from_file(path).name == "my-suite"

    def test_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        body = json.dumps({"id": "a", "input": "q", "output": "o"})
        path.write_text(f"{body}\n\n  \n", encoding="utf-8")
        assert len(Dataset.from_file(path)) == 1

    def test_structured_json_carries_name_and_version(self, tmp_path: Path) -> None:
        path = tmp_path / "d.json"
        path.write_text(
            json.dumps({"name": "named", "version": "v7", "cases": []}), encoding="utf-8"
        )
        data = Dataset.from_file(path)
        assert (data.name, data.version) == ("named", "v7")

    def test_bare_json_list_is_treated_as_cases(self, tmp_path: Path) -> None:
        path = tmp_path / "d.json"
        path.write_text(json.dumps([{"id": "a", "input": "q", "output": "o"}]), encoding="utf-8")
        assert len(Dataset.from_file(path)) == 1

    def test_yaml_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "d.yaml"
        path.write_text(
            "name: y\ncases:\n  - id: a\n    input: q\n    output: o\n", encoding="utf-8"
        )
        assert Dataset.from_file(path).name == "y"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dataset not found"):
            Dataset.from_file(tmp_path / "absent.jsonl")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("id,input\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported dataset format"):
            Dataset.from_file(path)


class TestEnums:
    def test_verdict_values_are_stable(self) -> None:
        """These land in stored results, so renaming one is a breaking change."""
        assert [v.value for v in Verdict] == ["pass", "fail", "error"]

    def test_flag_values_are_stable(self) -> None:
        assert Flag.NO_EVIDENCE.value == "no_evidence"
        assert Flag.NO_REFERENCE.value == "no_reference"
