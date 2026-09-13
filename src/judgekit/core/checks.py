"""Deterministic checks - the things you should never pay a model to decide.

An LLM judge is expensive, slow and slightly non-deterministic. Plenty of what
an eval suite needs to assert is none of those things: whether the answer is
empty, whether it leaked a stack trace, whether every figure it quotes actually
appears in the evidence.

Running these first is not only cheaper, it is more trustworthy. A check that
can be written as code should never be delegated to a judge, because the judge
introduces variance into a question that had an exact answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from judgekit.core.models import Case

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Phrases a model uses when it declines. Worth detecting explicitly: a refusal
# scored as a low-quality answer hides the fact that it answered nothing at all.
_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "i do not have access",
    "i don't have access",
)


class CheckResult(BaseModel):
    """The outcome of one deterministic check."""

    model_config = ConfigDict(frozen=True)

    id: str
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


@runtime_checkable
class Check(Protocol):
    """Anything that can pass or fail a case without calling a model."""

    @property
    def id(self) -> str: ...

    def __call__(self, case: Case) -> CheckResult: ...


def _numbers(text: str) -> set[str]:
    """Numeric literals, normalised so 1,234.0 and 1234 compare equal."""
    found = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        found.add(f"{value:g}")
    return found


class NonEmpty:
    """Fails an answer that is blank or whitespace."""

    id = "non_empty"

    def __call__(self, case: Case) -> CheckResult:
        ok = bool(case.output.strip())
        return CheckResult(id=self.id, passed=ok, detail="" if ok else "output is empty")


class MaxLength:
    """Fails an answer longer than a character budget."""

    def __init__(self, limit: int, *, check_id: str = "max_length") -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._id = check_id

    @property
    def id(self) -> str:
        return self._id

    def __call__(self, case: Case) -> CheckResult:
        length = len(case.output)
        ok = length <= self.limit
        return CheckResult(
            id=self.id,
            passed=ok,
            detail="" if ok else f"{length} characters exceeds the {self.limit} limit",
        )


class MustContain:
    """Requires every pattern to appear in the answer."""

    def __init__(
        self,
        patterns: Sequence[str],
        *,
        regex: bool = False,
        case_sensitive: bool = False,
        check_id: str = "must_contain",
    ) -> None:
        self.patterns = tuple(patterns)
        self.regex = regex
        self.case_sensitive = case_sensitive
        self._id = check_id

    @property
    def id(self) -> str:
        return self._id

    def _present(self, pattern: str, haystack: str) -> bool:
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return re.search(pattern, haystack, flags) is not None
        if self.case_sensitive:
            return pattern in haystack
        return pattern.lower() in haystack.lower()

    def __call__(self, case: Case) -> CheckResult:
        missing = [p for p in self.patterns if not self._present(p, case.output)]
        return CheckResult(
            id=self.id,
            passed=not missing,
            detail="" if not missing else f"missing: {', '.join(missing)}",
        )


class MustNotContain(MustContain):
    """Requires none of the patterns to appear. Useful for leak and PII checks."""

    def __init__(
        self,
        patterns: Sequence[str],
        *,
        regex: bool = False,
        case_sensitive: bool = False,
        check_id: str = "must_not_contain",
    ) -> None:
        super().__init__(patterns, regex=regex, case_sensitive=case_sensitive, check_id=check_id)

    def __call__(self, case: Case) -> CheckResult:
        present = [p for p in self.patterns if self._present(p, case.output)]
        return CheckResult(
            id=self.id,
            passed=not present,
            detail="" if not present else f"found: {', '.join(present)}",
        )


class GroundedFigures:
    """Requires every figure in the answer to appear in the evidence.

    This is the deterministic half of evidence-aware judging. Where the numbers
    are checkable by string comparison, checking them in code is strictly better
    than asking a model whether they look invented.

    An answer with no figures passes trivially. A case with no evidence is
    *skipped* rather than failed, because "unverifiable" and "wrong" are
    different findings and collapsing them loses information.
    """

    id = "grounded_figures"

    def __call__(self, case: Case) -> CheckResult:
        stated = _numbers(case.output)
        if not stated:
            return CheckResult(id=self.id, passed=True, detail="no figures stated")

        if case.evidence.is_empty:
            return CheckResult(
                id=self.id,
                passed=True,
                detail="skipped: no evidence supplied, so figures are unverifiable",
            )

        supported = _numbers(case.evidence.render())
        unsupported = sorted(stated - supported)
        return CheckResult(
            id=self.id,
            passed=not unsupported,
            detail=("" if not unsupported else f"not found in evidence: {', '.join(unsupported)}"),
        )


class NotARefusal:
    """Fails an answer that declines rather than answering."""

    id = "not_a_refusal"

    def __call__(self, case: Case) -> CheckResult:
        lowered = case.output.lower()
        hit = next((m for m in _REFUSAL_MARKERS if m in lowered), None)
        return CheckResult(
            id=self.id,
            passed=hit is None,
            detail="" if hit is None else f"looks like a refusal: {hit!r}",
        )


class ValidJson:
    """Requires the answer to parse as JSON. For structured-output evaluation."""

    id = "valid_json"

    def __call__(self, case: Case) -> CheckResult:
        try:
            json.loads(case.output)
        except json.JSONDecodeError as exc:
            return CheckResult(id=self.id, passed=False, detail=f"invalid JSON: {exc.msg}")
        return CheckResult(id=self.id, passed=True)


def run_checks(case: Case, checks: Sequence[Check]) -> tuple[CheckResult, ...]:
    """Run every check against a case and return all results.

    All of them run even after one fails, so a report can list every problem in
    a single pass rather than revealing them one build at a time.
    """
    return tuple(check(case) for check in checks)


def all_passed(results: Sequence[CheckResult]) -> bool:
    return all(r.passed for r in results)
