"""Runtime configuration, read once from the environment.

Deliberately plain: a frozen model built from ``os.environ``, with no settings
library behind it. The service has under a dozen knobs, and a dependency whose
whole job is reading environment variables is not worth the supply chain.

The database default is SQLite so that ``judgekit-api`` starts and works with
nothing installed. Production points ``DATABASE_URL`` at Postgres; the SQLAlchemy
layer above does not care which, which is also what lets the test suite cover
storage without a database running in CI.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./judgekit.db"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"

ENV_FILENAME = ".env"
DISABLE_ENV_FILE = "JUDGEKIT_DISABLE_ENV_FILE"


def _strip_value(value: str) -> str:
    """Unquote a .env value and drop any trailing comment.

    A quoted value ends at its closing quote and anything after it is a comment,
    so a ``#`` inside the quotes is kept. Unquoted, a ``#`` only starts a comment
    when whitespace precedes it - which keeps values that legitimately contain
    one, such as a URL fragment or a password, intact.

    The trailing-comment case is not hypothetical: a line reading
    ``JUDGEKIT_PROVIDER=groq  # stub | gemini | groq`` is exactly what people
    write, and without this the provider name included the comment and the
    registry rejected it as unknown.
    """
    value = value.strip()

    if value[:1] in ("'", '"'):
        quote = value[0]
        closing = value.find(quote, 1)
        if closing != -1:
            # Everything after the closing quote is a comment, so a value like
            # 'a#b'  # note keeps the # that is inside the quotes and drops the
            # one that is not. The closing quote is not always the last
            # character, which is what a naive first==last check gets wrong.
            return value[1:closing]
        # Unterminated: treat the quote as part of the value rather than
        # guessing where the author meant it to end.

    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1].isspace():
            return value[:i].rstrip()
    return value


def load_env_file(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read ``.env`` into ``os.environ`` and return what it set.

    Hand-rolled for the same reason the settings below are: this reads a file of
    ``KEY=value`` lines, and taking a dependency for that is not a trade worth
    making.

    **A real environment variable always wins.** Exporting something in your
    shell to override the file, and finding the file silently beat you, is a
    confusing half-hour; ``override=True`` is there for tests that need the
    opposite.

    Setting ``JUDGEKIT_DISABLE_ENV_FILE`` skips the file entirely. The test
    suite sets it, and needs to: without it, running pytest in a checkout that
    has a .env picked up the developer's own provider and key, so the suite
    reached the network and spent money while claiming to be hermetic. CI never
    noticed, because CI has no .env to find.

    Understands what people actually write in these files: blank lines, comments
    both on their own line and trailing a value, a leading ``export``, matched
    single or double quotes, and values containing ``=`` or ``#``. It does not do
    interpolation or multi-line values - if a value needs either, it belongs in
    the shell rather than here.
    """
    if os.environ.get(DISABLE_ENV_FILE):
        return {}

    target = path or Path.cwd() / ENV_FILENAME
    if not target.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = _strip_value(value)
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


class Settings(BaseModel):
    """Everything the service reads from its environment."""

    model_config = ConfigDict(frozen=True)

    database_url: str = DEFAULT_DATABASE_URL
    redis_url: str = DEFAULT_REDIS_URL

    rubric_dir: str = "rubrics"
    dataset_dir: str = "datasets"

    provider: str = "stub"
    queue: str = "inline"
    """``inline`` runs jobs in the API process; ``arq`` dispatches to workers.

    Defaults to inline for the same reason the provider defaults to the stub:
    `judgekit-api` should start and work with nothing else installed.
    """

    concurrency: int = Field(default=8, gt=0)

    log_level: str = "INFO"
    log_json: bool = True
    """Structured logs by default. A container's stdout is read by a machine."""

    api_title: str = "judgekit"
    cors_origins: tuple[str, ...] = ()

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @classmethod
    def from_env(cls) -> Settings:
        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        origins = os.environ.get("CORS_ORIGINS", "")

        return cls(
            database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            redis_url=os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
            rubric_dir=os.environ.get("JUDGEKIT_RUBRIC_DIR", "rubrics"),
            dataset_dir=os.environ.get("JUDGEKIT_DATASET_DIR", "datasets"),
            provider=os.environ.get("JUDGEKIT_PROVIDER", "stub"),
            queue=os.environ.get("JUDGEKIT_QUEUE", "inline"),
            concurrency=int(os.environ.get("JUDGEKIT_CONCURRENCY", "8")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            log_json=flag("LOG_JSON", default=True),
            api_title=os.environ.get("JUDGEKIT_API_TITLE", "judgekit"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return Settings.from_env()
