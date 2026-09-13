"""Rubrics, and the machinery that keeps their scores comparable.

The central idea: **a score is only meaningful next to the rubric that produced
it.** Editing a scoring prompt silently invalidates every number already
recorded, and the usual result is a quality metric that drifts for reasons
nobody can reconstruct months later.

judgekit makes that impossible to do by accident:

* every rubric has a ``fingerprint`` - a hash over exactly the fields that
  affect grading (scale, criteria, instructions), and nothing else;
* every judgement records the rubric id, version and fingerprint it was made
  under;
* comparing results across differing rubrics raises rather than returns;
* a lockfile pins ``version -> fingerprint``, so editing a rubric without
  bumping its version fails CI the way an unexpected dependency change does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from judgekit.core.errors import IncomparableScoresError, RubricError, RubricLockError

LOCKFILE_NAME = "rubrics.lock.json"


class Scale(BaseModel):
    """The numeric range a rubric scores on, and where good enough sits."""

    model_config = ConfigDict(frozen=True)

    minimum: int = 1
    maximum: int = 5
    pass_at: int = 4
    """Lowest score that counts as a pass."""

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError(
                f"scale minimum ({self.minimum}) must be below maximum ({self.maximum})"
            )
        if not self.minimum <= self.pass_at <= self.maximum:
            raise ValueError(
                f"pass_at ({self.pass_at}) must fall within [{self.minimum}, {self.maximum}]"
            )
        return self

    @property
    def span(self) -> int:
        return self.maximum - self.minimum

    def normalise(self, score: float) -> float:
        """Map a raw score onto 0.0-1.0, for comparing across different scales."""
        return (score - self.minimum) / self.span

    def clamp(self, score: float) -> float:
        return max(float(self.minimum), min(float(self.maximum), score))

    def is_pass(self, score: float) -> bool:
        return score >= self.pass_at


class Criterion(BaseModel):
    """One dimension the judge scores against.

    Decomposing a rubric into explicit criteria improves judge reliability over
    asking for a single holistic number, and it makes a low score explainable
    rather than merely low.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    weight: float = Field(default=1.0, gt=0.0)


class RubricRef(BaseModel):
    """An immutable pointer to the exact rubric a score was produced under."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: str
    fingerprint: str

    def __str__(self) -> str:
        return f"{self.id}@{self.version} ({self.fingerprint[:12]})"


class Rubric(BaseModel):
    """A versioned, content-addressed scoring specification."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: str
    instructions: str
    """The body of the judge prompt: what good and bad look like."""

    scale: Scale = Scale()
    criteria: tuple[Criterion, ...] = ()
    requires_evidence: bool = True
    """Whether the judge is shown, and told to rely on, the case evidence."""

    description: str | None = None
    """Free text for humans. Deliberately excluded from the fingerprint."""

    @model_validator(mode="after")
    def _check_criteria(self) -> Self:
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(f"rubric {self.id!r} has duplicate criterion ids")
        return self

    @property
    def fingerprint(self) -> str:
        """SHA-256 over exactly the fields that change how a case is graded.

        ``description`` and ``version`` are excluded on purpose: renaming or
        re-documenting a rubric must not invalidate historical scores, and the
        version is the label *for* this hash rather than part of it.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "instructions": self.instructions.strip(),
            "scale": self.scale.model_dump(),
            "criteria": [c.model_dump() for c in self.criteria],
            "requires_evidence": self.requires_evidence,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def ref(self) -> RubricRef:
        return RubricRef(id=self.id, version=self.version, fingerprint=self.fingerprint)

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria) or 1.0

    @classmethod
    def from_file(cls, path: str | Path) -> Rubric:
        """Load a rubric from a YAML or JSON file."""
        p = Path(path)
        if not p.is_file():
            raise RubricError(f"rubric not found: {p}")
        text = p.read_text(encoding="utf-8")
        try:
            raw = (
                yaml.safe_load(text) if p.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
            )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise RubricError(f"could not parse rubric {p}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RubricError(f"rubric {p} must be a mapping, got {type(raw).__name__}")
        try:
            return cls.model_validate(raw)
        except ValueError as exc:
            raise RubricError(f"invalid rubric {p}: {exc}") from exc


def assert_comparable(left: RubricRef, right: RubricRef) -> None:
    """Raise unless two results were graded under the identical rubric.

    Three distinct failures, each with its own remedy:

    * different rubric ids - unrelated measurements, nothing to compare;
    * different versions - an intentional change, so re-run the baseline;
    * same version, different fingerprint - a rubric was **edited in place**.
      This is the dangerous one, because nothing else in the system would
      notice, and every score on either side of the edit is quietly
      incompatible.
    """
    if left.id != right.id:
        raise IncomparableScoresError(
            f"different rubrics: {left.id!r} and {right.id!r} do not measure the same thing"
        )
    if left.version != right.version:
        raise IncomparableScoresError(
            f"rubric {left.id!r} differs in version ({left.version} vs {right.version}). "
            "Re-run the baseline under one version before comparing."
        )
    if left.fingerprint != right.fingerprint:
        raise IncomparableScoresError(
            f"rubric {left.id}@{left.version} was edited in place: fingerprints "
            f"{left.fingerprint[:12]} and {right.fingerprint[:12]} disagree. "
            "Bump the version instead of editing a published rubric."
        )


def load_rubrics(directory: str | Path) -> dict[str, Rubric]:
    """Load every rubric in a directory, keyed by ``id@version``."""
    d = Path(directory)
    if not d.is_dir():
        raise RubricError(f"rubric directory not found: {d}")

    rubrics: dict[str, Rubric] = {}
    for path in sorted(d.iterdir()):
        if path.name == LOCKFILE_NAME or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        rubric = Rubric.from_file(path)
        key = f"{rubric.id}@{rubric.version}"
        if key in rubrics:
            raise RubricError(f"two files define rubric {key}; versions must be unique")
        rubrics[key] = rubric
    return rubrics


class LockViolation(BaseModel):
    """One disagreement between the lockfile and the rubrics on disk."""

    model_config = ConfigDict(frozen=True)

    key: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.key}: {self.detail}"


def verify_lock(rubrics: dict[str, Rubric], lock: dict[str, str]) -> list[LockViolation]:
    """Compare rubrics against a lockfile, returning every disagreement.

    Returns violations rather than raising, so a caller can report all of them
    at once instead of one per run.
    """
    violations: list[LockViolation] = []

    for key, rubric in rubrics.items():
        expected = lock.get(key)
        if expected is None:
            violations.append(
                LockViolation(
                    key=key,
                    kind="unlocked",
                    detail="not in the lockfile; run 'judgekit rubric lock' to record it",
                )
            )
        elif expected != rubric.fingerprint:
            violations.append(
                LockViolation(
                    key=key,
                    kind="edited-without-bump",
                    detail=(
                        f"content changed since it was locked "
                        f"({expected[:12]} -> {rubric.fingerprint[:12]}). "
                        "Bump the version rather than editing a published rubric."
                    ),
                )
            )

    for key in lock:
        if key not in rubrics:
            violations.append(
                LockViolation(key=key, kind="missing", detail="locked but no longer on disk")
            )

    return violations


def read_lock(directory: str | Path) -> dict[str, str]:
    """Read the lockfile from a rubric directory; empty if there is none."""
    p = Path(directory) / LOCKFILE_NAME
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RubricLockError(f"could not parse {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RubricLockError(f"{p} must be a mapping of 'id@version' to fingerprint")
    return {str(k): str(v) for k, v in raw.items()}


def write_lock(directory: str | Path, rubrics: dict[str, Rubric]) -> Path:
    """Write the lockfile for a rubric directory and return its path."""
    p = Path(directory) / LOCKFILE_NAME
    payload = {key: rubric.fingerprint for key, rubric in sorted(rubrics.items())}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
