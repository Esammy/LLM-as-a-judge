"""Groq, over its OpenAI-compatible REST API.

The wire format here is the OpenAI chat-completions shape, so this adapter is
really an OpenAI-compatible adapter that happens to default to Groq's host.
Point ``base_url`` elsewhere - Together, Fireworks, OpenRouter, vLLM, Ollama,
OpenAI itself - and it keeps working. That is the practical argument for
speaking REST rather than importing a vendor SDK.
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

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
ENV_VAR = "GROQ_API_KEY"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Groq's free tier is generous on requests and tight on tokens per minute.
DEFAULT_RPM = 25

PRICING: dict[str, Pricing] = {
    "llama-3.3-70b-versatile": Pricing(input_per_mtok=0.59, output_per_mtok=0.79),
    "llama-3.1-8b-instant": Pricing(input_per_mtok=0.05, output_per_mtok=0.08),
    "openai/gpt-oss-20b": Pricing(input_per_mtok=0.10, output_per_mtok=0.50),
    "openai/gpt-oss-120b": Pricing(input_per_mtok=0.15, output_per_mtok=0.75),
}


class GroqProvider:
    """Judge backed by any OpenAI-compatible endpoint, defaulting to Groq.

    Args:
        api_key: Defaults to ``$GROQ_API_KEY``.
        model: Model id, also used to look up pricing.
        base_url: Point this at any OpenAI-compatible host.
        requests_per_minute: Client-side throttle, shared across the fan-out.
        family: Reported model family, which the self-preference bias check
            uses. Override it when pointing at a non-Groq host so that check
            stays meaningful.
        client: Inject an ``httpx.AsyncClient`` to test without a network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        requests_per_minute: int = DEFAULT_RPM,
        timeout: float = DEFAULT_TIMEOUT,
        family: str = "groq",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = require_key(
            api_key or os.environ.get(ENV_VAR), provider="groq", env_var=ENV_VAR
        )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._family = family
        self._limiter = RateLimiter(requests_per_minute)
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._pricing = PRICING.get(model, Pricing())

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    @property
    def family(self) -> str:
        return self._family

    async def aclose(self) -> None:
        """Close the HTTP client, if this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            # Judging returns JSON. Constraining it at the API level beats
            # asking politely in the prompt and parsing whatever comes back.
            "response_format": {"type": "json_object"},
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    async def complete(self, request: CompletionRequest) -> Completion:
        body = await post_json(
            self._client,
            f"{self._base_url}/chat/completions",
            payload=self._payload(request),
            headers={"Authorization": f"Bearer {self._api_key}"},
            limiter=self._limiter,
            provider="groq",
        )

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("groq returned no choices")

        choice = choices[0]
        text = choice.get("message", {}).get("content") or ""

        usage = body.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        return Completion(
            text=text,
            model=str(body.get("model", self._model)),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=self._pricing.cost(prompt_tokens, completion_tokens),
            ),
            truncated=choice.get("finish_reason") == "length",
        )
