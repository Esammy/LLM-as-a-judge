"""Core data model.

The shapes here are the contract between datasets on disk, the judge, and the
report. Two of them carry most of the design intent:

``Evidence``
    What actually produced an answer - the tool calls and retrieved context.
    A judge asked "is this figure invented?" cannot answer honestly without it,
    so evidence is a first-class field rather than something stuffed into a
    prompt by the caller.

``Case.human_label``
    The optional ground-truth score from a person. It is what makes the judge
    measurable, and it is why calibration is possible at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(StrEnum):
    """Outcome of judging a single case."""

    PASS = "pass"  # noqa: S105 - a verdict, not a credential
    FAIL = "fail"
    ERROR = "error"


class Flag(StrEnum):
    """Machine-readable notes a judgement can carry.

    These exist so a report can explain *why* a score is untrustworthy, rather
    than silently folding the problem into the number.
    """

    NO_EVIDENCE = "no_evidence"
    """Output contains figures but the case supplied no evidence to check them against."""

    NO_REFERENCE = "no_reference"
    """Scored without a reference answer; the judge graded on intrinsic quality alone."""

    PARSE_RECOVERED = "parse_recovered"
    """The judge's response was malformed and was recovered by a fallback parse."""

    TRUNCATED = "truncated"
    """The provider stopped on a length limit, so the judgement may be incomplete."""


class ToolCall(BaseModel):
    """A single tool invocation that contributed to an answer."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None

    def render(self) -> str:
        """Render for inclusion in a judge prompt."""
        args = json.dumps(self.arguments, ensure_ascii=False, sort_keys=True, default=str)
        result = json.dumps(self.result, ensure_ascii=False, sort_keys=True, default=str)
        return f"{self.name}({args}) -> {result}"


class Evidence(BaseModel):
    """Everything the system consulted in order to answer.

    This is the field that removes the most common false negative in production
    eval suites: a judge marking a correct figure as fabricated purely because
    it could not see where the figure came from.
    """

    model_config = ConfigDict(frozen=True)

    tool_calls: tuple[ToolCall, ...] = ()
    context: tuple[str, ...] = ()
    """Retrieved documents or passages, in the order they were supplied."""

    @property
    def is_empty(self) -> bool:
        return not self.tool_calls and not self.context

    def render(self) -> str:
        """Render as a plain-text block for a judge prompt."""
        if self.is_empty:
            return "(no evidence supplied)"
        parts: list[str] = []
        if self.tool_calls:
            calls = "\n".join(f"  {i + 1}. {c.render()}" for i, c in enumerate(self.tool_calls))
            parts.append(f"Tool calls:\n{calls}")
        if self.context:
            docs = "\n".join(f"  [{i + 1}] {d}" for i, d in enumerate(self.context))
            parts.append(f"Retrieved context:\n{docs}")
        return "\n\n".join(parts)


class Case(BaseModel):
    """One thing to be judged.

    A case pairs an input with the answer a system gave, plus whatever is needed
    to grade it: a reference answer, the evidence behind the output, and - when
    it exists - a human's own score.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    input: str
    """The question or instruction the system was given."""

    output: str
    """The answer under test."""

    reference: str | None = None
    """A gold answer, where one exists. Absent means intrinsic-quality grading."""

    evidence: Evidence = Evidence()
    domain: str | None = None
    tags: tuple[str, ...] = ()

    human_label: float | None = None
    """A person's score for this case, on the rubric's scale. Drives calibration."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_human_label(self) -> bool:
        return self.human_label is not None


class Dataset(BaseModel):
    """A named, versioned collection of cases.

    Versioning is deliberate. An eval number is only comparable against another
    number produced from the same dataset version, in the same way that it is
    only comparable under the same rubric version.
    """

    name: str
    version: str = "v1"
    description: str | None = None
    cases: tuple[Case, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicate_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen:
                duplicates.append(case.id)
            seen.add(case.id)
        if duplicates:
            listed = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"dataset {self.name!r} has duplicate case ids: {listed}")
        return self

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:  # type: ignore[override]
        return iter(self.cases)

    @property
    def labelled(self) -> tuple[Case, ...]:
        """The subset carrying human labels - the only cases calibration can use."""
        return tuple(c for c in self.cases if c.has_human_label)

    def filter(
        self,
        *,
        domain: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> Dataset:
        """Return a new dataset narrowed to matching cases."""
        wanted = set(tags or ())
        cases = tuple(
            c
            for c in self.cases
            if (domain is None or c.domain == domain)
            and (not wanted or wanted.issubset(set(c.tags)))
        )
        return self.model_copy(update={"cases": cases})

    @classmethod
    def from_file(cls, path: str | Path) -> Dataset:
        """Load a dataset from ``.jsonl``, ``.json``, ``.yaml`` or ``.yml``.

        JSONL is treated as a bare list of cases; the dataset name is taken from
        the filename. The structured formats may supply ``name``/``version``
        alongside ``cases``.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"dataset not found: {p}")

        suffix = p.suffix.lower()
        if suffix == ".jsonl":
            cases = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return cls.model_validate({"name": p.stem, "cases": cases})

        if suffix in {".yaml", ".yml"}:
            raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
        elif suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"unsupported dataset format {suffix!r}: use .jsonl, .json or .yaml")

        if isinstance(raw, list):
            return cls.model_validate({"name": p.stem, "cases": raw})
        return cls.model_validate(raw)
