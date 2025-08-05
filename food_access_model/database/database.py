from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from food_access_model.settings import settings


class Base(DeclarativeBase, AsyncAttrs):
    __mapper_args__ = {"eager_defaults": True}


class AsyncDatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, Any] = {}):
        self._engine: AsyncEngine | None = create_async_engine(host, **engine_kwargs)
        self._sessionmaker: async_sessionmaker | None = async_sessionmaker(
            autocommit=False, bind=self._engine, expire_on_commit=False
        )

    async def close(self):
        if self._engine is None:
            raise Exception("AsyncDatabaseSessionManager is not initialized.")
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise Exception("AsyncDatabaseSessionManager is not initialized.")
        async with self._engine.begin() as conn:
            try:
                yield conn
            except Exception:
                await conn.rollback()
                raise

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("AsyncDatabaseSessionManager is not initialized.")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self, connection: AsyncConnection):
        await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self, connection: AsyncConnection):
        await connection.run_sync(Base.metadata.drop_all)


engine_kwargs = {
    "echo": False,
    "future": True,
    "pool_size": 20,
    "max_overflow": 20,
    "pool_recycle": 3600,
}

postgres_uri = (
    settings.POSTGRES_URI
    if settings.POSTGRES_URI
    else f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)
sessionmanager = AsyncDatabaseSessionManager(
    host=postgres_uri, engine_kwargs=engine_kwargs
)


# Used by api to get session
async def get_session():
    async with sessionmanager.session() as session:
        yield session