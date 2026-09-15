"""Selecting a provider by name.

One place that turns ``"gemini"`` into a configured provider, so the CLI, the
worker and a library caller all resolve them identically.

The default is always ``stub``. That is a deliberate safety property rather than
a convenience: nothing in judgekit reaches the network unless somebody asked for
a real provider by name, so a mistyped flag or a missing environment variable
degrades to a deterministic offline judge instead of quietly spending money.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from judgekit.core.errors import ProviderError
from judgekit.providers.base import Provider
from judgekit.providers.stub import StubProvider

DEFAULT_PROVIDER = "stub"
ENV_VAR = "JUDGEKIT_PROVIDER"


def _stub(**kwargs: Any) -> Provider:
    return StubProvider(**kwargs)


def _gemini(**kwargs: Any) -> Provider:
    # Imported lazily so that `import judgekit` never touches provider config,
    # and so a missing key raises only when that provider is actually wanted.
    from judgekit.providers.gemini import GeminiProvider

    return GeminiProvider(**kwargs)


def _groq(**kwargs: Any) -> Provider:
    from judgekit.providers.groq import GroqProvider

    return GroqProvider(**kwargs)


_FACTORIES: dict[str, Callable[..., Provider]] = {
    "stub": _stub,
    "gemini": _gemini,
    "groq": _groq,
}


def available() -> tuple[str, ...]:
    """Every provider name that can be requested."""
    return tuple(sorted(_FACTORIES))


def resolve_name(name: str | None = None) -> str:
    """The provider name that :func:`create_provider` would use.

    Exposed so callers can ask which provider they are about to get without
    building one - the CLI needs it to decide whether a ``--model`` is
    meaningful - rather than re-deriving the precedence rule and drifting from
    it.
    """
    return (name or os.environ.get(ENV_VAR) or DEFAULT_PROVIDER).strip().lower()


def create_provider(name: str | None = None, **kwargs: Any) -> Provider:
    """Build a provider by name.

    Args:
        name: One of :func:`available`. Defaults to ``$JUDGEKIT_PROVIDER``, and
            then to ``stub``.
        **kwargs: Passed through to the provider's constructor.

    Raises:
        ProviderError: If the name is unknown, or the provider cannot be built
            (most often a missing API key).
    """
    chosen = resolve_name(name)

    factory = _FACTORIES.get(chosen)
    if factory is None:
        raise ProviderError(f"unknown provider {chosen!r}. Available: {', '.join(available())}")

    return factory(**kwargs)


async def close_provider(provider: Provider) -> None:
    """Release a provider's resources, if it holds any.

    Providers are not required to be closeable - the stub holds nothing - so
    this is a no-op for anything without an ``aclose``.
    """
    closer = getattr(provider, "aclose", None)
    if closer is not None:
        await closer()
