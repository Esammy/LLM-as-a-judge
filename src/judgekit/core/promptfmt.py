"""The wire format between the judge prompt and anything that reads it back.

The judge renders a case into delimited sections, and the response comes back
as JSON. Both shapes live here rather than being inlined as string literals,
for two reasons:

* the deterministic stub provider parses these same sections in order to
  behave like a plausible judge without a network call, and a magic string
  duplicated in two modules is a bug waiting to happen;
* a change to the prompt layout is a change to the rubric fingerprint's
  meaning, so it should be visible in one place during review.
"""

from __future__ import annotations

import json
import re
from typing import Any

from judgekit.core.errors import JudgeParseError

TAG_INPUT = "INPUT"
TAG_OUTPUT = "OUTPUT"
TAG_REFERENCE = "REFERENCE"
TAG_EVIDENCE = "EVIDENCE"
TAG_CRITERIA = "CRITERIA"
TAG_SCALE = "SCALE"

ALL_TAGS = (
    TAG_INPUT,
    TAG_OUTPUT,
    TAG_REFERENCE,
    TAG_EVIDENCE,
    TAG_CRITERIA,
    TAG_SCALE,
)


def section(tag: str, body: str) -> str:
    """Render one delimited section of a judge prompt."""
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def read_section(text: str, tag: str) -> str | None:
    """Extract one section's body, or ``None`` when the tag is absent."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1) if match else None


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose and fences no matter how firmly the prompt asks
    them not to, so three strategies are tried in order of confidence: parse
    the whole response, parse the contents of a fenced block, then fall back to
    the outermost brace-delimited span.

    Raises:
        JudgeParseError: when no strategy yields a JSON object.
    """
    candidates: list[str] = [text.strip()]

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text[:200].replace("\n", " ")
    raise JudgeParseError(f"no JSON object found in judge response: {preview!r}")
