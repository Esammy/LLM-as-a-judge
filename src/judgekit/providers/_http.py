"""Shared HTTP machinery for real providers.

judgekit talks to providers over plain REST rather than through their SDKs.
Each of these APIs is a single POST, `httpx` is already a dependency, and the
Groq endpoint is OpenAI-shaped so one adapter covers a whole family. Two large
transitive dependency trees for two HTTP calls is a poor trade, and it makes the
adapters testable with a mock transport instead of SDK stubs.

What lives here is the part that is easy to get wrong and identical everywhere:

**Rate limiting.** Free tiers are tight - roughly 15 requests per minute on
Gemini, 30 on Groq - and the runner fans out to many cases at once. Without a
limiter the first thing a new user sees is a wall of 429s, and they conclude the
tool is broken rather than that they are being throttled.

**Backoff that honours ``Retry-After``.** Guessing a delay when the server has
told you exactly how long to wait is both ruder and slower.

**Refusing to retry what will not succeed.** A 401 is not transient. Retrying it
three times turns an obvious "your key is wrong" into a slow, confusing failure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from judgekit.core.errors import ProviderError

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3

# Retried with backoff. Everything else in the 4xx range is a request problem
# that will fail identically however many times it is sent.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class Pricing(BaseModel):
    """Cost per million tokens, so a run can report what it spent."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1_000_000


class RateLimiter:
    """A small async token bucket.

    Shared across the whole fan-out, so raising ``--concurrency`` cannot quietly
    exceed the provider's per-minute allowance.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        self.capacity = float(requests_per_minute)
        self._tokens = float(requests_per_minute)
        self._refill_per_second = requests_per_minute / 60.0
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request may be sent."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self._refill_per_second
                )
                self._updated = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait = (1.0 - self._tokens) / self._refill_per_second

            await asyncio.sleep(wait)


def _retry_after(response: httpx.Response, fallback: float) -> float:
    """How long to wait, preferring the server's own instruction."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            # The HTTP-date form is legal but rare; the fallback is fine there.
            pass
    return fallback


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = 1.0,
    limiter: RateLimiter | None = None,
    provider: str = "provider",
) -> dict[str, Any]:
    """POST JSON and return the decoded body, retrying what is worth retrying.

    Raises:
        ProviderError: On a non-retryable status, on exhausted retries, or when
            the response is not a JSON object.
    """
    last_error = ""

    for attempt in range(retries + 1):
        if limiter is not None:
            await limiter.acquire()

        try:
            response = await client.post(url, json=payload, headers=headers, params=params)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                break
            await asyncio.sleep(backoff_seconds * (2**attempt))
            continue

        if response.status_code < 400:
            try:
                body = response.json()
            except ValueError as exc:
                raise ProviderError(
                    f"{provider} returned a non-JSON body: {response.text[:200]!r}"
                ) from exc
            if not isinstance(body, dict):
                raise ProviderError(
                    f"{provider} returned {type(body).__name__}, expected an object"
                )
            return body

        detail = response.text[:300]

        if response.status_code not in RETRYABLE_STATUS:
            raise ProviderError(f"{provider} returned {response.status_code}: {detail}")

        last_error = f"HTTP {response.status_code}: {detail}"
        if attempt == retries:
            break

        await asyncio.sleep(_retry_after(response, backoff_seconds * (2**attempt)))

    raise ProviderError(f"{provider} failed after {retries + 1} attempts - {last_error}")


def require_key(key: str | None, *, provider: str, env_var: str) -> str:
    """Return an API key or explain precisely how to supply one."""
    if key:
        return key
    raise ProviderError(
        f"no API key for {provider}. Set {env_var}, or pass api_key=..., or use the "
        "default 'stub' provider, which needs no key and no network."
    )
