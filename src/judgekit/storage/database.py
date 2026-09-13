"""Engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from judgekit.storage.models import Base


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build an async engine for Postgres or SQLite.

    SQLite gets ``check_same_thread=False`` because the async driver hands
    connections between threads, and an in-memory URL additionally needs a
    ``StaticPool`` so every session sees the same database rather than each
    connection getting its own empty one - the classic way an in-memory test
    suite reports "no such table".
    """
    if database_url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        in_memory = ":memory:" in database_url
        return create_async_engine(
            database_url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if in_memory else None,
        )

    return create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def create_schema(engine: AsyncEngine) -> None:
    """Create tables directly, without Alembic.

    For tests and a first local run. Production migrates with Alembic; calling
    this against a real database would silently diverge from the migration
    history.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def drop_schema(engine: AsyncEngine) -> None:
    """Drop every table. Tests only."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A session that commits on success and rolls back on any exception."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
