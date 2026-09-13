# Contributing

```bash
uv sync --all-extras
uv run pre-commit install
make check          # ruff, mypy --strict, pytest, coverage gate
.\make.ps1 check    # same, on Windows
```

## The rules that are not negotiable

**The test suite must run with no API key and no network.** This is the project's
central promise, and CI proves it by blackholing DNS and running the suite again.
Anything needing a real provider is marked `@pytest.mark.live` and deselected by
default.

**Never edit a published rubric in place.** Bump the version and add a file.
`judgekit rubric verify` fails CI otherwise, which is the point — that guard is
what the project sells. After adding a rubric, run `judgekit rubric lock`.

**mypy runs in strict mode over `src` and `tests`.** Tests are code.

**Coverage floor is 85%.** Not because the number means much on its own, but
because it stops a large untested module arriving unnoticed.

## Conventions

- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rename something instead.
- Tests are named as sentences: `test_refuses_to_compare_across_rubric_versions`.
- A test for a surprising behaviour gets a docstring explaining the surprise.
- Errors say what to do next. `"no API key for gemini. Set GEMINI_API_KEY, or
  pass api_key=..., or use the default 'stub' provider"` beats `"auth failed"`.

## Adding a provider

Implement the `Provider` protocol in `judgekit/providers/`, register it in
`registry.py`, and test it through `httpx.MockTransport`. Do not add a vendor
SDK — these APIs are one POST each, and `httpx` is already a dependency.

If the endpoint is OpenAI-compatible, you probably do not need a new adapter at
all: point `GroqProvider` at it with `base_url` and set `family`.

## Adding a bias detector

Return a **magnitude in a stated unit**, never a verdict. Return an
"insufficient data" finding rather than a zero when the question cannot be
asked: "not measured" and "measured as unbiased" are different claims about a
judge, and collapsing them launders ignorance into reassurance.

Validate it by injecting a known bias into `StubProvider` and confirming it is
found. That only works because the stub is neutral by construction — keep it
that way.
