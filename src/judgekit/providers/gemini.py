"""Google Gemini, over the REST API.

Free-tier friendly: the default model is a Flash variant and the default rate
limit sits under the free allowance, so a first run throttles rather than
failing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from judgekit.core.errors import ProviderError
from judgekit.providers._http import (
    DEFAULT_TIMEOUT,
    Pricing,
    RateLimiter,
    post_json,
    require_key,
)
from judgekit.providers.base import Completion, CompletionRequest, Usage

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
ENV_VAR = "GEMINI_API_KEY"
# Hosted model ids are not stable. The Groq default in this package was
# llama-3.3-70b-versatile until it was withdrawn, which surfaces as a 404 that
# reads like a broken install rather than a retired model. This one has NOT been
# checked against a live key; list what yours can reach with
#   curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
# and pass --model to override without editing this file.
DEFAULT_MODEL = "gemini-2.0-flash"

# Free tier is roughly 15 requests per minute. Defaulting under it means a new
# user's first run is slow rather than a wall of 429s they read as a bug.
DEFAULT_RPM = 12

PRICING: dict[str, Pricing] = {
    "gemini-2.0-flash": Pricing(input_per_mtok=0.10, output_per_mtok=0.40),
    "gemini-2.0-flash-lite": Pricing(input_per_mtok=0.075, output_per_mtok=0.30),
    "gemini-2.5-flash": Pricing(input_per_mtok=0.30, output_per_mtok=2.50),
    "gemini-2.5-pro": Pricing(input_per_mtok=1.25, output_per_mtok=10.00),
}


class GeminiProvider:
    """Judge backed by Google Gemini.

    Args:
        api_key: Defaults to ``$GEMINI_API_KEY``.
        model: Model id. Pricing is looked up by this name; unknown models
            simply report zero cost rather than guessing.
        requests_per_minute: Client-side throttle, shared across the fan-out.
        client: Inject an ``httpx.AsyncClient`` to test without a network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        requests_per_minute: int = DEFAULT_RPM,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = require_key(
            api_key or os.environ.get(ENV_VAR), provider="gemini", env_var=ENV_VAR
        )
        self._model = model
        self._limiter = RateLimiter(requests_per_minute)
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._pricing = PRICING.get(model, Pricing())

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    @property
    def family(self) -> str:
        return "google"

    async def aclose(self) -> None:
        """Close the HTTP client, if this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                # Judging returns JSON. Asking for it at the API level is far more
                # reliable than asking for it in prose and parsing the fallout.
                "responseMimeType": "application/json",
            },
        }
        if request.system:
            payload["systemInstruction"] = {"parts": [{"text": request.system}]}
        return payload

    async def complete(self, request: CompletionRequest) -> Completion:
        body = await post_json(
            self._client,
            f"{API_ROOT}/models/{self._model}:generateContent",
            payload=self._payload(request),
            params={"key": self._api_key},
            limiter=self._limiter,
            provider="gemini",
        )

        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            # A blocked prompt comes back with no candidates and a reason. Say so,
            # rather than reporting an empty answer the judge will score as bad.
            feedback = body.get("promptFeedback", {})
            reason = feedback.get("blockReason", "no candidates returned")
            raise ProviderError(f"gemini returned no candidates: {reason}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))

        usage_meta = body.get("usageMetadata", {})
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0))
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0))

        return Completion(
            text=text,
            model=self._model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=self._pricing.cost(prompt_tokens, completion_tokens),
            ),
            truncated=candidate.get("finishReason") == "MAX_TOKENS",
        )
