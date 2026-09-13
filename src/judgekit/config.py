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

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./judgekit.db"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


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
