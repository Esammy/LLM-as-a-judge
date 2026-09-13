"""Shared fixtures.

Everything here is offline and deterministic. No fixture may reach the network,
because the suite's central promise is that a fresh clone runs green with no API
key - and a fixture is exactly where that promise would quietly break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from judgekit.core.models import Case, Dataset, Evidence, ToolCall
from judgekit.core.rubric import Criterion, Rubric, Scale
from judgekit.providers.stub import StubProvider

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scale() -> Scale:
    return Scale(minimum=1, maximum=5, pass_at=4)


@pytest.fixture
def rubric(scale: Scale) -> Rubric:
    """A small evidence-aware rubric with weighted criteria."""
    return Rubric(
        id="test-quality",
        version="v1",
        instructions="Grade the answer against the criteria.",
        scale=scale,
        requires_evidence=True,
        criteria=(
            Criterion(id="groundedness", description="Do figures trace to evidence?", weight=3.0),
            Criterion(id="correctness", description="Is it right?", weight=2.0),
        ),
    )


@pytest.fixture
def grounded_case() -> Case:
    """A correct answer whose figure is supported by a tool call."""
    return Case(
        id="grounded",
        input="What was revenue last month?",
        output="Revenue was 48200 USD.",
        reference="Revenue was 48200 USD.",
        evidence=Evidence(
            tool_calls=(
                ToolCall(
                    name="get_revenue",
                    arguments={"month": "2026-03"},
                    result={"total_usd": 48200},
                ),
            )
        ),
        domain="finance",
        human_label=5.0,
    )


@pytest.fixture
def ungrounded_case() -> Case:
    """The same question answered with a figure the evidence contradicts."""
    return Case(
        id="ungrounded",
        input="What was revenue last month?",
        output="Revenue was 99999 USD.",
        reference="Revenue was 48200 USD.",
        evidence=Evidence(
            tool_calls=(
                ToolCall(
                    name="get_revenue",
                    arguments={"month": "2026-03"},
                    result={"total_usd": 48200},
                ),
            )
        ),
        domain="finance",
        human_label=1.0,
    )


@pytest.fixture
def evidenceless_case() -> Case:
    """Correct, but nothing was captured to verify it against.

    The case that motivates the whole project: a judge shown no evidence tends
    to treat a correct figure as invented.
    """
    return Case(
        id="evidenceless",
        input="What was revenue last month?",
        output="Revenue was 48200 USD.",
        reference="Revenue was 48200 USD.",
        domain="finance",
        human_label=4.0,
    )


@pytest.fixture
def dataset(grounded_case: Case, ungrounded_case: Case, evidenceless_case: Case) -> Dataset:
    return Dataset(
        name="fixture",
        version="v1",
        cases=(grounded_case, ungrounded_case, evidenceless_case),
    )


@pytest.fixture
def provider() -> StubProvider:
    return StubProvider()
