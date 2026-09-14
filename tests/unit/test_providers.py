"""Real providers, exercised entirely through a mock HTTP transport.

Not one test here opens a socket. ``httpx.MockTransport`` intercepts at the
transport layer, so these run in the offline CI job alongside everything else -
which is the point of speaking REST rather than importing an SDK that would have
needed mocking at the library level instead.

Tests that genuinely need a provider are marked ``live`` and skipped unless a
key is present.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import pytest

from judgekit.core.errors import ProviderError
from judgekit.providers._http import (
    Pricing,
    RateLimiter,
    post_json,
    require_key,
)
from judgekit.providers.base import CompletionRequest, Provider
from judgekit.providers.gemini import GeminiProvider
from judgekit.providers.groq import GroqProvider
from judgekit.providers.registry import (
    available,
    close_provider,
    create_provider,
)
from judgekit.providers.stub import StubProvider

REQUEST = CompletionRequest(prompt="grade this", system="you are a judge", max_tokens=256)


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def gemini_body(text: str = '{"score": 4}', *, finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": finish},
        ],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
    }


def groq_body(text: str = '{"score": 4}', *, finish: str = "stop") -> dict[str, Any]:
    return {
        "model": "llama-3.3-70b-versatile",
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class TestPricing:
    def test_costs_nothing_by_default(self) -> None:
        assert Pricing().cost(1000, 1000) == 0.0

    def test_prices_input_and_output_separately(self) -> None:
        pricing = Pricing(input_per_mtok=1.0, output_per_mtok=10.0)
        assert pricing.cost(1_000_000, 0) == pytest.approx(1.0)
        assert pricing.cost(0, 1_000_000) == pytest.approx(10.0)


class TestRateLimiter:
    def test_rejects_a_nonsense_rate(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RateLimiter(0)

    async def test_lets_the_initial_burst_straight_through(self) -> None:
        limiter = RateLimiter(60)
        started = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        assert time.monotonic() - started < 0.1

    async def test_throttles_once_the_bucket_empties(self) -> None:
        """A request past the allowance waits for the bucket to refill.

        Uses a fast refill rate (600/min is ten per second) so the assertion
        costs a tenth of a second rather than a whole one. The behaviour under
        test is the wait, not its duration.
        """
        limiter = RateLimiter(600)
        limiter._tokens = 0.0

        started = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - started

        assert elapsed >= 0.05
        assert elapsed < 1.0

    async def test_is_safe_under_concurrency(self) -> None:
        limiter = RateLimiter(600)
        await asyncio.gather(*(limiter.acquire() for _ in range(20)))
        assert limiter._tokens <= limiter.capacity


class TestPostJson:
    async def test_returns_a_decoded_body(self) -> None:
        async with transport(lambda _: httpx.Response(200, json={"ok": True})) as client:
            assert await post_json(client, "https://x/y", payload={}) == {"ok": True}

    async def test_retries_a_429_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
            return httpx.Response(200, json={"ok": True})

        async with transport(handler) as client:
            assert await post_json(client, "https://x/y", payload={}, backoff_seconds=0.0)
        assert calls["n"] == 2

    async def test_honours_retry_after(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0.3"})
            return httpx.Response(200, json={"ok": True})

        started = time.monotonic()
        async with transport(handler) as client:
            await post_json(client, "https://x/y", payload={}, backoff_seconds=0.0)

        # backoff_seconds is 0, so any delay at all came from the header.
        assert time.monotonic() - started >= 0.25

    async def test_ignores_an_unparseable_retry_after(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
            return httpx.Response(200, json={"ok": True})

        async with transport(handler) as client:
            await post_json(client, "https://x/y", payload={}, backoff_seconds=0.0)
        assert calls["n"] == 2

    async def test_does_not_retry_an_auth_failure(self) -> None:
        """A 401 is not transient. Retrying it makes a clear error slow and murky."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, text="invalid key")

        async with transport(handler) as client:
            with pytest.raises(ProviderError, match="returned 401"):
                await post_json(client, "https://x/y", payload={}, backoff_seconds=0.0)

        assert calls["n"] == 1

    async def test_gives_up_after_the_retry_budget(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="unavailable")

        async with transport(handler) as client:
            with pytest.raises(ProviderError, match="failed after 3 attempts"):
                await post_json(client, "https://x/y", payload={}, retries=2, backoff_seconds=0.0)

        assert calls["n"] == 3

    async def test_retries_a_transport_error(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("no route")
            return httpx.Response(200, json={"ok": True})

        async with transport(handler) as client:
            await post_json(client, "https://x/y", payload={}, backoff_seconds=0.0)
        assert calls["n"] == 2

    async def test_a_persistent_transport_error_becomes_a_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route")

        async with transport(handler) as client:
            with pytest.raises(ProviderError, match="ConnectError"):
                await post_json(client, "https://x/y", payload={}, retries=1, backoff_seconds=0.0)

    async def test_rejects_a_non_json_body(self) -> None:
        async with transport(lambda _: httpx.Response(200, text="<html>nope")) as client:
            with pytest.raises(ProviderError, match="non-JSON body"):
                await post_json(client, "https://x/y", payload={})

    async def test_rejects_a_json_array(self) -> None:
        async with transport(lambda _: httpx.Response(200, json=[1, 2])) as client:
            with pytest.raises(ProviderError, match="expected an object"):
                await post_json(client, "https://x/y", payload={})


class TestRequireKey:
    def test_returns_a_present_key(self) -> None:
        assert require_key("abc", provider="p", env_var="P_KEY") == "abc"

    @pytest.mark.parametrize("missing", [None, ""])
    def test_explains_how_to_supply_one(self, missing: str | None) -> None:
        with pytest.raises(ProviderError) as exc:
            require_key(missing, provider="gemini", env_var="GEMINI_API_KEY")

        message = str(exc.value)
        assert "GEMINI_API_KEY" in message
        assert "stub" in message


class TestGemini:
    async def test_satisfies_the_provider_protocol(self) -> None:
        async with transport(lambda _: httpx.Response(200, json=gemini_body())) as client:
            provider = GeminiProvider(api_key="k", client=client)
            assert isinstance(provider, Provider)
            assert (provider.name, provider.family) == ("gemini", "google")

    async def test_sends_the_prompt_and_system_instruction(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = __import__("json").loads(request.content)
            seen["url"] = str(request.url)
            return httpx.Response(200, json=gemini_body())

        async with transport(handler) as client:
            await GeminiProvider(api_key="secret", client=client).complete(REQUEST)

        payload = seen["payload"]
        assert payload["contents"][0]["parts"][0]["text"] == "grade this"
        assert payload["systemInstruction"]["parts"][0]["text"] == "you are a judge"
        assert payload["generationConfig"]["maxOutputTokens"] == 256
        # Judging returns JSON; asking at the API level beats asking in prose.
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert "key=secret" in seen["url"]

    async def test_omits_the_system_instruction_when_absent(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=gemini_body())

        async with transport(handler) as client:
            await GeminiProvider(api_key="k", client=client).complete(CompletionRequest(prompt="x"))

        assert "systemInstruction" not in seen["payload"]

    async def test_reads_text_usage_and_cost(self) -> None:
        async with transport(lambda _: httpx.Response(200, json=gemini_body())) as client:
            completion = await GeminiProvider(
                api_key="k", model="gemini-2.0-flash", client=client
            ).complete(REQUEST)

        assert completion.text == '{"score": 4}'
        assert completion.usage.prompt_tokens == 100
        assert completion.usage.completion_tokens == 20
        assert completion.usage.cost_usd == pytest.approx((100 * 0.10 + 20 * 0.40) / 1_000_000)
        assert not completion.truncated

    async def test_an_unknown_model_reports_zero_cost_rather_than_guessing(self) -> None:
        async with transport(lambda _: httpx.Response(200, json=gemini_body())) as client:
            completion = await GeminiProvider(
                api_key="k", model="gemini-from-the-future", client=client
            ).complete(REQUEST)

        assert completion.usage.cost_usd == 0.0

    async def test_flags_a_truncated_response(self) -> None:
        body = gemini_body(finish="MAX_TOKENS")
        async with transport(lambda _: httpx.Response(200, json=body)) as client:
            completion = await GeminiProvider(api_key="k", client=client).complete(REQUEST)
        assert completion.truncated

    async def test_a_blocked_prompt_raises_rather_than_scoring_empty(self) -> None:
        """An empty answer would be scored as a bad one. It is a refusal, not a score."""
        body = {"promptFeedback": {"blockReason": "SAFETY"}}
        async with transport(lambda _: httpx.Response(200, json=body)) as client:
            with pytest.raises(ProviderError, match="SAFETY"):
                await GeminiProvider(api_key="k", client=client).complete(REQUEST)

    async def test_joins_multipart_responses(self) -> None:
        body = {
            "candidates": [{"content": {"parts": [{"text": "ab"}, {"text": "cd"}]}}],
            "usageMetadata": {},
        }
        async with transport(lambda _: httpx.Response(200, json=body)) as client:
            completion = await GeminiProvider(api_key="k", client=client).complete(REQUEST)
        assert completion.text == "abcd"

    def test_a_missing_key_is_explained_at_construction(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
            GeminiProvider()

    def test_reads_the_key_from_the_environment(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert GeminiProvider().model  # constructed without raising


class TestGroq:
    async def test_sends_openai_shaped_messages(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = __import__("json").loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            seen["url"] = str(request.url)
            return httpx.Response(200, json=groq_body())

        async with transport(handler) as client:
            await GroqProvider(api_key="secret", client=client).complete(REQUEST)

        payload = seen["payload"]
        # The caller's own messages are passed through untouched and in order,
        # after the JSON precondition message - see
        # test_adds_the_json_precondition_only_when_needed.
        assert payload["messages"][-2:] == [
            {"role": "system", "content": "you are a judge"},
            {"role": "user", "content": "grade this"},
        ]
        assert payload["response_format"] == {"type": "json_object"}
        assert seen["auth"] == "Bearer secret"
        assert seen["url"].endswith("/chat/completions")

    async def test_omits_the_system_message_when_absent(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=groq_body())

        async with transport(handler) as client:
            await GroqProvider(api_key="k", client=client).complete(
                CompletionRequest(prompt="return json")
            )

        assert [m["role"] for m in seen["payload"]["messages"]] == ["user"]

    async def test_adds_the_json_precondition_only_when_needed(self) -> None:
        """Groq 400s on response_format=json_object unless a message says "json".

        The adapter is what asks for a JSON response, so the adapter has to
        satisfy the precondition. Leaving it to the caller produced a provider
        that worked for the built-in judge prompt - which happens to say JSON -
        and returned 400 for anyone passing a prompt of their own.
        """
        seen: list[list[dict[str, str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(__import__("json").loads(request.content)["messages"])
            return httpx.Response(200, json=groq_body())

        async with transport(handler) as client:
            provider = GroqProvider(api_key="k", client=client)
            await provider.complete(CompletionRequest(prompt="grade this"))
            await provider.complete(CompletionRequest(prompt="reply as JSON please"))
            await provider.complete(CompletionRequest(prompt="grade this", system="answer in json"))

        without, in_prompt, in_system = seen

        # Nothing mentions JSON, so one is prepended.
        assert without[0]["role"] == "system"
        assert "json" in without[0]["content"].lower()
        assert without[-1] == {"role": "user", "content": "grade this"}

        # Already satisfied, in either message: left exactly as the caller sent it.
        assert [m["role"] for m in in_prompt] == ["user"]
        assert [m["role"] for m in in_system] == ["system", "user"]

    async def test_forwards_a_seed_when_given(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=groq_body())

        async with transport(handler) as client:
            await GroqProvider(api_key="k", client=client).complete(
                CompletionRequest(prompt="x", seed=42)
            )

        assert seen["payload"]["seed"] == 42

    async def test_reads_text_usage_and_cost(self) -> None:
        async with transport(lambda _: httpx.Response(200, json=groq_body())) as client:
            completion = await GroqProvider(
                api_key="k", model="llama-3.3-70b-versatile", client=client
            ).complete(REQUEST)

        assert completion.text == '{"score": 4}'
        assert completion.usage.total_tokens == 120
        assert completion.usage.cost_usd > 0

    async def test_flags_a_length_stop(self) -> None:
        body = groq_body(finish="length")
        async with transport(lambda _: httpx.Response(200, json=body)) as client:
            completion = await GroqProvider(api_key="k", client=client).complete(REQUEST)
        assert completion.truncated

    async def test_no_choices_raises(self) -> None:
        async with transport(lambda _: httpx.Response(200, json={"choices": []})) as client:
            with pytest.raises(ProviderError, match="no choices"):
                await GroqProvider(api_key="k", client=client).complete(REQUEST)

    async def test_a_null_content_becomes_empty_text(self) -> None:
        body = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "usage": {}}
        async with transport(lambda _: httpx.Response(200, json=body)) as client:
            completion = await GroqProvider(api_key="k", client=client).complete(REQUEST)
        assert completion.text == ""

    async def test_points_at_any_openai_compatible_host(self) -> None:
        """The practical payoff of speaking REST instead of importing an SDK."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json=groq_body())

        async with transport(handler) as client:
            provider = GroqProvider(
                api_key="k",
                base_url="http://localhost:11434/v1/",
                family="local",
                client=client,
            )
            await provider.complete(REQUEST)

        assert seen["url"] == "http://localhost:11434/v1/chat/completions"
        assert provider.family == "local"


class TestRegistry:
    def test_lists_what_can_be_requested(self) -> None:
        assert available() == ("gemini", "groq", "stub")

    def test_defaults_to_the_offline_stub(self, monkeypatch: Any) -> None:
        """Nothing reaches the network unless somebody asked for it by name."""
        monkeypatch.delenv("JUDGEKIT_PROVIDER", raising=False)
        assert create_provider().name == "stub"

    def test_reads_the_default_from_the_environment(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("JUDGEKIT_PROVIDER", "stub")
        assert create_provider().name == "stub"

    def test_is_case_and_whitespace_insensitive(self) -> None:
        assert create_provider("  STUB ").name == "stub"

    def test_an_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(ProviderError, match="Available: gemini, groq, stub"):
            create_provider("nonesuch")

    def test_forwards_constructor_arguments(self) -> None:
        provider = create_provider("stub", family="pretend")
        assert provider.family == "pretend"

    def test_builds_a_real_provider_when_a_key_is_present(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert create_provider("gemini").name == "gemini"

    def test_surfaces_a_missing_key(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="GROQ_API_KEY"):
            create_provider("groq")

    async def test_closing_the_stub_is_a_no_op(self) -> None:
        await close_provider(StubProvider())

    async def test_closes_a_provider_that_holds_a_client(self) -> None:
        provider = GeminiProvider(api_key="k")
        await close_provider(provider)
        assert provider._client.is_closed

    async def test_does_not_close_an_injected_client(self) -> None:
        """The caller owns what the caller supplied."""
        async with transport(lambda _: httpx.Response(200, json=gemini_body())) as client:
            await close_provider(GeminiProvider(api_key="k", client=client))
            assert not client.is_closed


@pytest.mark.live
class TestLiveProviders:
    """Only run when a real key is present. Skipped by default, including in CI.

    These deliberately do not cap ``max_tokens``. Current judge models reason
    before they answer - openai/gpt-oss-120b spends about 220 tokens getting to
    a twelve-character reply - so a tight budget truncates the response
    mid-object and the provider rejects it as malformed JSON rather than
    reporting a length problem. An earlier version of these tests passed 64 and
    failed against every reasoning model for reasons that looked like a bug in
    the adapter.
    """

    async def test_gemini_returns_a_judgement(self) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        provider = GeminiProvider()
        try:
            completion = await provider.complete(
                CompletionRequest(
                    prompt='Return exactly {"score": 4} and nothing else.',
                )
            )
        finally:
            await provider.aclose()

        assert "score" in completion.text
        assert completion.usage.total_tokens > 0

    async def test_groq_returns_a_judgement(self) -> None:
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

        provider = GroqProvider()
        try:
            completion = await provider.complete(
                CompletionRequest(
                    prompt='Return exactly {"score": 4} and nothing else.',
                )
            )
        finally:
            await provider.aclose()

        assert "score" in completion.text
        assert completion.usage.total_tokens > 0
