"""
SQLAlchemy 2.0 async engine + session factory and declarative base.
All ORM models must inherit from `Base`.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://single_pass_3d:devpassword@localhost:5432/single_pass_3d_dev",
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def build_engine(url: str = DATABASE_URL, echo: bool = False) -> AsyncEngine:
    return create_async_engine(url, echo=echo, pool_pre_ping=True)


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(url: str = DATABASE_URL, echo: bool = False) -> AsyncEngine:
    global _engine, _session_factory
    _engine = build_engine(url, echo)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async DB session."""
    if _session_factory is None:
        raise RuntimeError("Database engine not initialised. Call init_engine() first.")
    async with _session_factory() as session:
        yield session
