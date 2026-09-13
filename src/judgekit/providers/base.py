"""The provider port.

A provider is the only part of judgekit that talks to a model. Everything else
depends on this ``Protocol`` and never on a concrete SDK, which is what makes
swapping Gemini for Groq - or for the deterministic stub - a configuration
change rather than a refactor.

Structural typing is deliberate: an adapter does not subclass anything, it just
has to have the right shape. A user can pass their own object with a
``complete`` coroutine and it will work.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Token and cost accounting for a single call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class Completion(BaseModel):
    """What a provider returns for one request."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: Usage = Usage()
    truncated: bool = False
    """True when the provider stopped on a length limit rather than finishing.

    Surfaced rather than swallowed: a judgement cut off mid-JSON is a different
    problem from a judgement the model genuinely could not make.
    """


class CompletionRequest(BaseModel):
    """One request to a model."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    system: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    """Judging defaults to 0.0. A judge that is not reproducible is not a measurement."""

    max_tokens: int = Field(default=1024, gt=0)
    seed: int | None = None


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a prompt into text."""

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``stub``, ``gemini``, ``groq``."""
        ...

    @property
    def model(self) -> str:
        """The specific model in use, recorded on every run for reproducibility."""
        ...

    @property
    def family(self) -> str:
        """Model family, used to detect self-preference bias.

        A judge scoring its own family's output higher is a known and measurable
        failure. Comparing across families is only possible if a provider says
        which family it belongs to.
        """
        ...

    async def complete(self, request: CompletionRequest) -> Completion:
        """Run one completion."""
        ...
