"""A deterministic judge that needs no API key and no network.

This is the **default** provider, and that is a deliberate design choice rather
than a convenience. It buys three things:

* ``pytest`` runs green on a fresh clone, offline, in milliseconds - which is
  what makes strangers actually try the project;
* CI is free, hermetic and reproducible, so a red build means a real
  regression rather than a rate limit;
* the bias machinery in :mod:`judgekit.core.bias` can be tested against a judge
  whose bias is *known*, because this one can be told to exhibit exactly as
  much position or verbosity bias as a test needs.

It is a test double, not a model. It scores by lexical overlap with the
reference and by whether figures in the output trace back to the evidence. That
is enough to exercise every layer above it, and it is honest about being a
heuristic.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from judgekit.core.promptfmt import (
    TAG_CRITERIA,
    TAG_EVIDENCE,
    TAG_OUTPUT,
    TAG_REFERENCE,
    TAG_SCALE,
    read_section,
)
from judgekit.providers.base import Completion, CompletionRequest, Usage

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Judging a long answer that merely restates the question should not beat a
# short correct one. This caps how much length alone can move a score.
_MAX_VERBOSITY_SHIFT = 0.25

# Function words carry no information about whether an answer is correct, and
# leaving them in wrecks the similarity metric: a reference sentence is mostly
# stopwords, so a terse correct answer looks like it covered almost none of it.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _tokens(text: str) -> set[str]:
    """Content tokens: lowercase words with function words removed."""
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


class StubProvider:
    """A deterministic, offline stand-in for a judge model.

    Args:
        seed: Fixes the jitter. The same prompt and seed always produce the
            same judgement.
        position_bias: How strongly to favour whichever answer is presented
            first, as a fraction of the scale. ``0.0`` is unbiased; ``0.2``
            simulates a judge with a substantial position preference. Used to
            prove the bias detector actually detects something.
        verbosity_bias: How strongly to reward longer answers regardless of
            quality. Published measurements put real judges in the 0.15-0.30
            range, which is why that is the interesting band to simulate.
        noise: Deterministic pseudo-random jitter, as a fraction of the scale.
            Nonzero values let calibration tests exercise imperfect agreement.
        family: Reported model family. Set it to match a candidate's family to
            simulate self-preference bias.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        position_bias: float = 0.0,
        verbosity_bias: float = 0.0,
        noise: float = 0.0,
        family: str = "stub",
        model: str = "stub-judge-v1",
    ) -> None:
        self._seed = seed
        self._position_bias = position_bias
        self._verbosity_bias = verbosity_bias
        self._noise = noise
        self._family = family
        self._model = model

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return self._model

    @property
    def family(self) -> str:
        return self._family

    def _jitter(self, prompt: str) -> float:
        """Deterministic pseudo-noise in ``[-1.0, 1.0]`` derived from the prompt."""
        if self._noise <= 0.0:
            return 0.0
        digest = hashlib.sha256(f"{self._seed}:{prompt}".encode()).digest()
        return (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0

    def _quality(self, output: str, reference: str | None, evidence: str | None) -> float:
        """Estimate answer quality in ``[0.0, 1.0]``.

        Two signals, weighted: how much of the reference the output covers, and
        whether the figures it states trace back to the evidence.
        """
        out_tokens = _tokens(output)

        if reference:
            ref_tokens = _tokens(reference)
            coverage = self._f1(out_tokens, ref_tokens)
        else:
            # With no reference there is nothing to cover, so start from neutral
            # and let groundedness carry the score.
            coverage = 0.5

        grounded = self._groundedness(output, evidence)
        return max(0.0, min(1.0, 0.6 * coverage + 0.4 * grounded))

    @staticmethod
    def _f1(output_tokens: set[str], reference_tokens: set[str]) -> float:
        """Harmonic mean of precision and recall over content tokens.

        Recall alone would make this stub structurally verbosity-biased: a long
        answer covers more of the reference simply by saying more, and a terse
        correct answer scores badly for being terse. That would be fatal here,
        because :mod:`judgekit.core.bias` tests its verbosity detector by
        injecting a *known* amount of bias into this provider - which only works
        if the provider is unbiased to begin with.

        Adding precision fixes it. Padding dilutes the overlap and pulls the
        score down, so brevity is rewarded rather than punished.
        """
        if not reference_tokens or not output_tokens:
            return 0.5
        overlap = len(output_tokens & reference_tokens)
        if not overlap:
            return 0.0
        precision = overlap / len(output_tokens)
        recall = overlap / len(reference_tokens)
        return 2 * precision * recall / (precision + recall)

    def _groundedness(self, output: str, evidence: str | None) -> float:
        """Fraction of figures in the output that appear in the evidence.

        This is what makes the stub useful for testing evidence-aware judging:
        remove the evidence and a numeric answer's score drops, exactly as it
        would with a real judge that cannot see where a number came from.
        """
        stated = _numbers(output)
        if not stated:
            return 1.0
        if not evidence:
            return 0.0
        supported = _numbers(evidence)
        return len(stated & supported) / len(stated)

    def _scale(self, prompt: str) -> tuple[float, float]:
        raw = read_section(prompt, TAG_SCALE)
        if raw:
            try:
                parsed: Any = json.loads(raw)
                return float(parsed["minimum"]), float(parsed["maximum"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return 1.0, 5.0

    def _criteria(self, prompt: str) -> list[str]:
        raw = read_section(prompt, TAG_CRITERIA)
        if not raw:
            return []
        ids: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip().lstrip("-*").strip()
            if ":" in stripped:
                ids.append(stripped.split(":", 1)[0].strip())
        return ids

    async def complete(self, request: CompletionRequest) -> Completion:
        """Score the case embedded in the prompt and return judge-shaped JSON."""
        prompt = request.prompt
        output = read_section(prompt, TAG_OUTPUT) or ""
        reference = read_section(prompt, TAG_REFERENCE)
        evidence = read_section(prompt, TAG_EVIDENCE)
        minimum, maximum = self._scale(prompt)
        span = maximum - minimum

        quality = self._quality(output, reference, evidence)

        if self._verbosity_bias:
            # Saturating, so a very long answer does not score arbitrarily high.
            length_factor = min(1.0, len(output) / 600.0)
            quality += self._verbosity_bias * length_factor * _MAX_VERBOSITY_SHIFT / 0.25

        if self._position_bias and "<CANDIDATE_A>" in prompt:
            quality += self._position_bias

        quality += self._jitter(prompt) * self._noise
        quality = max(0.0, min(1.0, quality))

        score = round(minimum + quality * span, 2)
        criteria = self._criteria(prompt)
        criterion_scores = dict.fromkeys(criteria, score)

        reasoning = (
            f"Stub judge: lexical coverage and groundedness give {quality:.2f} "
            f"on a normalised scale, which maps to {score} on [{minimum:g}, {maximum:g}]. "
            "This is a deterministic heuristic, not a model."
        )

        body = json.dumps(
            {"score": score, "criterion_scores": criterion_scores, "reasoning": reasoning}
        )

        return Completion(
            text=body,
            model=self._model,
            usage=Usage(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(body) // 4,
                cost_usd=0.0,
            ),
        )
