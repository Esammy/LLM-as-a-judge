"""Rubric fingerprinting, comparability and locking.

These are the most important tests in the project. Everything judgekit claims
rests on the guarantee that a score cannot be silently compared against a score
produced under different grading rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgekit.core.errors import IncomparableScoresError, RubricError, RubricLockError
from judgekit.core.rubric import (
    Criterion,
    Rubric,
    Scale,
    assert_comparable,
    load_rubrics,
    read_lock,
    verify_lock,
    write_lock,
)


class TestScale:
    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="must be below maximum"):
            Scale(minimum=5, maximum=1)

    def test_rejects_pass_threshold_outside_range(self) -> None:
        with pytest.raises(ValueError, match="must fall within"):
            Scale(minimum=1, maximum=5, pass_at=9)

    def test_normalise_maps_to_unit_interval(self) -> None:
        scale = Scale(minimum=1, maximum=5)
        assert scale.normalise(1) == 0.0
        assert scale.normalise(5) == 1.0
        assert scale.normalise(3) == 0.5

    def test_clamp_bounds_out_of_range_scores(self) -> None:
        scale = Scale(minimum=1, maximum=5)
        assert scale.clamp(-4) == 1.0
        assert scale.clamp(99) == 5.0

    def test_is_pass_uses_the_threshold(self) -> None:
        scale = Scale(minimum=1, maximum=5, pass_at=4)
        assert scale.is_pass(4.0)
        assert not scale.is_pass(3.99)


class TestFingerprint:
    """The fingerprint must react to grading changes and ignore cosmetic ones."""

    def test_is_stable_across_identical_rubrics(self, rubric: Rubric) -> None:
        twin = rubric.model_copy(deep=True)
        assert twin.fingerprint == rubric.fingerprint

    def test_ignores_description(self, rubric: Rubric) -> None:
        redocumented = rubric.model_copy(update={"description": "a totally new explanation"})
        assert redocumented.fingerprint == rubric.fingerprint

    def test_ignores_version(self, rubric: Rubric) -> None:
        """The version labels the hash; it must not be part of it.

        Otherwise bumping a version would change the fingerprint even when the
        grading rules are identical, and the edited-in-place detector could
        never distinguish a real edit from a renumbering.
        """
        bumped = rubric.model_copy(update={"version": "v99"})
        assert bumped.fingerprint == rubric.fingerprint

    def test_ignores_surrounding_whitespace_in_instructions(self, rubric: Rubric) -> None:
        padded = rubric.model_copy(update={"instructions": f"\n\n{rubric.instructions}\n  "})
        assert padded.fingerprint == rubric.fingerprint

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("instructions", "Grade it completely differently."),
            ("requires_evidence", False),
            ("scale", Scale(minimum=0, maximum=10, pass_at=7)),
            ("criteria", (Criterion(id="only", description="one thing"),)),
        ],
    )
    def test_changes_when_grading_changes(self, rubric: Rubric, field: str, value: object) -> None:
        altered = rubric.model_copy(update={field: value})
        assert altered.fingerprint != rubric.fingerprint

    def test_reacts_to_criterion_weight(self, rubric: Rubric) -> None:
        """Reweighting silently changes every aggregate score, so it must count."""
        reweighted = rubric.model_copy(
            update={
                "criteria": tuple(
                    c.model_copy(update={"weight": c.weight * 2}) for c in rubric.criteria
                )
            }
        )
        assert reweighted.fingerprint != rubric.fingerprint


class TestRubricValidation:
    def test_rejects_duplicate_criterion_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate criterion ids"):
            Rubric(
                id="dupes",
                version="v1",
                instructions="x",
                criteria=(
                    Criterion(id="same", description="a"),
                    Criterion(id="same", description="b"),
                ),
            )

    def test_rejects_non_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="greater_than"):
            Criterion(id="c", description="d", weight=0.0)


class TestAssertComparable:
    """Three distinct failures, because each has a different remedy."""

    def test_accepts_identical_refs(self, rubric: Rubric) -> None:
        assert_comparable(rubric.ref, rubric.ref)

    def test_rejects_different_rubric_ids(self, rubric: Rubric) -> None:
        other = rubric.model_copy(update={"id": "something-else"})
        with pytest.raises(IncomparableScoresError, match="do not measure the same thing"):
            assert_comparable(rubric.ref, other.ref)

    def test_rejects_different_versions(self, rubric: Rubric) -> None:
        other = rubric.model_copy(update={"version": "v2"})
        with pytest.raises(IncomparableScoresError, match="differs in version"):
            assert_comparable(rubric.ref, other.ref)

    def test_rejects_same_version_edited_in_place(self, rubric: Rubric) -> None:
        """The dangerous case: the version says these match and the content does not."""
        edited = rubric.model_copy(update={"instructions": "Grade much more harshly."})
        assert edited.version == rubric.version
        with pytest.raises(IncomparableScoresError, match="edited in place"):
            assert_comparable(rubric.ref, edited.ref)

    def test_ref_renders_readably(self, rubric: Rubric) -> None:
        assert str(rubric.ref).startswith("test-quality@v1 (")


class TestLoading:
    def test_loads_the_shipped_rubrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        rubrics = load_rubrics(repo_root / "rubrics")
        assert set(rubrics) == {"answer-quality@v1", "answer-quality@v2"}
        assert rubrics["answer-quality@v1"].requires_evidence is False
        assert rubrics["answer-quality@v2"].requires_evidence is True

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RubricError, match="rubric not found"):
            Rubric.from_file(tmp_path / "nope.yaml")

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RubricError, match="directory not found"):
            load_rubrics(tmp_path / "nowhere")

    def test_unparseable_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("id: [unclosed", encoding="utf-8")
        with pytest.raises(RubricError, match="could not parse"):
            Rubric.from_file(bad)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(RubricError, match="must be a mapping"):
            Rubric.from_file(bad)

    def test_invalid_schema_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "invalid.yaml"
        bad.write_text("id: x\nversion: v1\n", encoding="utf-8")
        with pytest.raises(RubricError, match="invalid rubric"):
            Rubric.from_file(bad)

    def test_duplicate_version_across_files_raises(self, tmp_path: Path, rubric: Rubric) -> None:
        for name in ("a.yaml", "b.yaml"):
            (tmp_path / name).write_text(
                json.dumps(rubric.model_dump(mode="json")), encoding="utf-8"
            )
        with pytest.raises(RubricError, match="versions must be unique"):
            load_rubrics(tmp_path)

    def test_json_rubrics_load(self, tmp_path: Path, rubric: Rubric) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps(rubric.model_dump(mode="json")), encoding="utf-8")
        assert Rubric.from_file(path).fingerprint == rubric.fingerprint


class TestLockfile:
    def test_roundtrip(self, tmp_path: Path, rubric: Rubric) -> None:
        rubrics = {"test-quality@v1": rubric}
        write_lock(tmp_path, rubrics)
        assert read_lock(tmp_path) == {"test-quality@v1": rubric.fingerprint}
        assert verify_lock(rubrics, read_lock(tmp_path)) == []

    def test_absent_lockfile_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_lock(tmp_path) == {}

    def test_unparseable_lockfile_raises(self, tmp_path: Path) -> None:
        (tmp_path / "rubrics.lock.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(RubricLockError, match="could not parse"):
            read_lock(tmp_path)

    def test_non_mapping_lockfile_raises(self, tmp_path: Path) -> None:
        (tmp_path / "rubrics.lock.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(RubricLockError, match="must be a mapping"):
            read_lock(tmp_path)

    def test_flags_an_unlocked_rubric(self, rubric: Rubric) -> None:
        violations = verify_lock({"test-quality@v1": rubric}, {})
        assert [v.kind for v in violations] == ["unlocked"]
        assert "judgekit rubric lock" in violations[0].detail

    def test_flags_a_rubric_edited_without_a_version_bump(self, rubric: Rubric) -> None:
        """The violation the lockfile exists for."""
        lock = {"test-quality@v1": rubric.fingerprint}
        edited = rubric.model_copy(update={"instructions": "Grade much more harshly."})

        violations = verify_lock({"test-quality@v1": edited}, lock)

        assert [v.kind for v in violations] == ["edited-without-bump"]
        assert "Bump the version" in violations[0].detail

    def test_flags_a_rubric_that_vanished(self, rubric: Rubric) -> None:
        violations = verify_lock({}, {"test-quality@v1": rubric.fingerprint})
        assert [v.kind for v in violations] == ["missing"]

    def test_reports_every_violation_at_once(self, rubric: Rubric) -> None:
        """One build should surface all the problems, not one per run."""
        edited = rubric.model_copy(update={"instructions": "harsher"})
        fresh = rubric.model_copy(update={"id": "brand-new"})

        violations = verify_lock(
            {"test-quality@v1": edited, "brand-new@v1": fresh},
            {"test-quality@v1": rubric.fingerprint, "gone@v1": "deadbeef"},
        )

        assert {v.kind for v in violations} == {"edited-without-bump", "unlocked", "missing"}

    def test_violation_renders_readably(self, rubric: Rubric) -> None:
        violation = verify_lock({"test-quality@v1": rubric}, {})[0]
        assert str(violation).startswith("[unlocked] test-quality@v1:")
