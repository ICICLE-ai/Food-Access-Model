"""Tests for lazy database-pool creation (``ensure_pool``).

Regression coverage for the Oct 2026 outage: fassdb's node died, the backend
restarted while the database was down, and because the pool was only created
at startup every request failed with ``'NoneType' object has no attribute
'acquire'`` until the backend was restarted by hand -- even after the database
came back. No Postgres needed: ``asyncpg.create_pool`` is stubbed so each test
controls whether the "database" is reachable.
"""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

import food_access_model.api.routes as routes
from food_access_model.main import app


class _FakeConnection:
    async def fetch(self, query: str) -> list:
        return [{"?column?": 1}]


class _FakeAcquire:
    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection()

    async def __aexit__(self, *exc_info) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.closed = False

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()

    async def close(self) -> None:
        self.closed = True


class _FakeDatabase:
    """Stands in for ``asyncpg.create_pool``; ``reachable`` flips the outage on and off."""

    def __init__(self, reachable: bool) -> None:
        self.reachable = reachable
        self.create_calls = 0

    async def create_pool(self, *args, **kwargs) -> _FakePool:
        self.create_calls += 1
        await asyncio.sleep(0)  # yield so concurrent callers actually interleave
        if not self.reachable:
            raise ConnectionRefusedError(111, "Connect call failed")
        return _FakePool()


@pytest.fixture
def database(monkeypatch):
    """Start every test with no pool (the post-outage state) and a stubbed create_pool."""
    def make(reachable: bool) -> _FakeDatabase:
        fake = _FakeDatabase(reachable)
        monkeypatch.setattr(routes.asyncpg, "create_pool", fake.create_pool)
        return fake

    monkeypatch.setattr(routes, "pool", None)
    return make


async def test_creates_pool_when_missing(database):
    fake = database(reachable=True)

    await routes.ensure_pool()

    assert isinstance(routes.pool, _FakePool)
    assert fake.create_calls == 1


async def test_existing_pool_is_reused(database):
    fake = database(reachable=True)
    await routes.ensure_pool()
    first_pool = routes.pool

    await routes.ensure_pool()

    assert routes.pool is first_pool
    assert fake.create_calls == 1


async def test_unreachable_database_raises_503_and_leaves_pool_unset(database):
    database(reachable=False)

    with pytest.raises(HTTPException) as exc_info:
        await routes.ensure_pool()

    assert exc_info.value.status_code == 503
    assert routes.pool is None


async def test_recovers_after_startup_failure_once_database_is_back(database):
    """The outage scenario end to end: startup can't connect, requests 503 while
    the database is down, then the first request after it returns succeeds."""
    fake = database(reachable=False)

    await routes.startup()  # must not raise
    assert routes.pool is None

    with pytest.raises(HTTPException) as exc_info:
        await routes.ensure_pool()
    assert exc_info.value.status_code == 503

    fake.reachable = True
    await routes.ensure_pool()

    assert isinstance(routes.pool, _FakePool)
    assert await routes.health_check() == {"status": "healthy", "message": "API is operational"}
    assert fake.create_calls == 3


async def test_concurrent_requests_create_only_one_pool(database):
    fake = database(reachable=True)

    await asyncio.gather(*(routes.ensure_pool() for _ in range(10)))

    assert fake.create_calls == 1
    assert isinstance(routes.pool, _FakePool)


async def test_shutdown_without_pool_does_not_raise(database):
    database(reachable=False)

    await routes.shutdown()

    assert routes.pool is None


async def test_shutdown_closes_existing_pool(database):
    database(reachable=True)
    await routes.ensure_pool()
    created_pool = routes.pool

    await routes.shutdown()

    assert created_pool.closed is True


def test_every_api_route_runs_ensure_pool():
    """Guards the wiring: the retry only helps if FastAPI runs it before each
    handler. Checks every registered /api route rather than just the router
    config, so a route added some other way can't silently skip it."""
    api_routes = [route for route in app.routes if isinstance(route, APIRoute) and route.path.startswith("/api")]
    missing = [
        route.path
        for route in api_routes
        if routes.ensure_pool not in {dependency.call for dependency in route.dependant.dependencies}
    ]

    assert len(api_routes) == 15
    assert missing == []
