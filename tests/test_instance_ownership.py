"""Tests for optional bearer-token user scoping on instance access (issue #102).

No Postgres needed: a tiny fake pool answers the one ``SELECT 1`` the ownership
predicate executes. Token parsing is a pure function and is exercised directly
without fakes. The ownership check runs the exact predicate the production
dependency uses, so the five-scenario matrix from the issue is covered
end-to-end.
"""
import uuid

import pytest
from fastapi import HTTPException

import food_access_model.api.routes as routes


VALID_ID = str(uuid.UUID("11111111-1111-4111-8111-111111111111"))


# --- token parsing (pure, no DB) --------------------------------------------
#
# Parsing is tested independently from verification: these tests stub
# ``verify_token`` so the Bearer-header parser can be exercised without
# standing up a JWKS endpoint. The verification path itself is covered by
# tests/test_token_verification.py.


@pytest.fixture
def stub_verify_token(monkeypatch):
    """Make verify_token echo the raw token so parsing tests can assert on
    the token the parser extracted, not the verified user id."""
    monkeypatch.setattr(routes, "verify_token", lambda token: token or None)


def test_no_header_returns_none():
    # Short-circuits before verify_token is called, so no stub needed.
    assert routes.get_current_user_id(authorization=None) is None


def test_empty_header_returns_none():
    assert routes.get_current_user_id(authorization="") is None


def test_bearer_token_is_extracted(stub_verify_token):
    assert routes.get_current_user_id(authorization="Bearer user-abc") == "user-abc"


def test_bearer_scheme_is_case_insensitive(stub_verify_token):
    """Clients vary on casing; accept the common forms rather than coupling to one."""
    assert routes.get_current_user_id(authorization="bearer user-abc") == "user-abc"
    assert routes.get_current_user_id(authorization="BEARER user-abc") == "user-abc"


def test_bearer_token_with_surrounding_whitespace_is_trimmed(stub_verify_token):
    assert routes.get_current_user_id(authorization="Bearer   user-abc  ") == "user-abc"


def test_non_bearer_scheme_returns_none():
    """A Basic/Digest header is not an error -- it's just "no bearer token here",
    which the no-token path handles the same way as an absent header."""
    assert routes.get_current_user_id(authorization="Basic abcdef==") is None


def test_bearer_with_empty_token_returns_none():
    assert routes.get_current_user_id(authorization="Bearer   ") is None


def test_extract_user_id_from_token_delegates_to_verify_token(monkeypatch):
    """Regression guard: this port replaces trust-as-is with a verify_token
    delegation. If someone reverts _extract_user_id_from_token to pass the token
    through unchanged, this test fires."""
    recorded = {}

    def fake_verify(token):
        recorded["token"] = token
        return "verified-user"

    monkeypatch.setattr(routes, "verify_token", fake_verify)
    assert routes._extract_user_id_from_token("some-token") == "verified-user"
    assert recorded["token"] == "some-token"


# --- ownership check (fake pool for the single SELECT 1) --------------------


class _OwnershipConn:
    """Answers the ownership SELECT 1 against an in-memory row set.

    Rows are (instance_id, owner_id) tuples; owner_id of None models the shared
    public pool. The predicate (`owner_id IS NOT DISTINCT FROM $2::text`) is
    reproduced in Python here, mirroring the SQL exactly so the fake tracks
    the production comparison and not something looser.
    """

    def __init__(self, rows):
        self._rows = rows

    async def fetchval(self, query, *args):
        assert "simulation_instances" in query, f"unexpected fetchval: {query}"
        instance_id, user_id = args
        for row_instance, row_owner in self._rows:
            if row_instance == instance_id and row_owner == user_id:
                return 1
        return None


class _OwnershipPool:
    def __init__(self, rows):
        self._rows = rows

    def acquire(self):
        rows = self._rows

        class _Acquire:
            async def __aenter__(self):
                return _OwnershipConn(rows)

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


@pytest.fixture
def seed_pool(monkeypatch):
    def _seed(*rows):
        pool = _OwnershipPool(set(rows))
        monkeypatch.setattr(routes, "pool", pool)
        return pool

    return _seed


# The matrix from the acceptance criteria. Each row is one scenario.

async def test_no_token_public_instance_is_allowed(seed_pool):
    seed_pool((VALID_ID, None))
    # Should not raise.
    await routes.authorize_instance_access(VALID_ID, user_id=None)


async def test_no_token_owned_instance_is_404(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    with pytest.raises(HTTPException) as exc:
        await routes.authorize_instance_access(VALID_ID, user_id=None)
    assert exc.value.status_code == 404


async def test_token_own_instance_is_allowed(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    await routes.authorize_instance_access(VALID_ID, user_id="owner-1")


async def test_token_other_owner_instance_is_404(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    with pytest.raises(HTTPException) as exc:
        await routes.authorize_instance_access(VALID_ID, user_id="owner-2")
    assert exc.value.status_code == 404


async def test_token_public_instance_is_404(seed_pool):
    """A token means "my scenarios only" and does NOT include the shared public
    pool, so logged-in users don't see it."""
    seed_pool((VALID_ID, None))
    with pytest.raises(HTTPException) as exc:
        await routes.authorize_instance_access(VALID_ID, user_id="owner-1")
    assert exc.value.status_code == 404


async def test_nonexistent_instance_is_404(seed_pool):
    """Shape-wise identical to the "not yours" 404; both deliberately look the
    same so the response doesn't leak whether a given id belongs to someone else."""
    seed_pool()  # empty -- no rows at all
    with pytest.raises(HTTPException) as exc:
        await routes.authorize_instance_access(VALID_ID, user_id=None)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await routes.authorize_instance_access(VALID_ID, user_id="owner-1")
    assert exc.value.status_code == 404


# --- the authorized_* dependencies compose parse + authorize ----------------


async def test_authorized_instance_id_path_returns_id_when_accessible(seed_pool):
    seed_pool((VALID_ID, None))
    result = await routes.authorized_instance_id_path(instance_id=VALID_ID, user_id=None)
    assert result == VALID_ID


async def test_authorized_instance_id_path_404s_when_inaccessible(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    with pytest.raises(HTTPException) as exc:
        await routes.authorized_instance_id_path(instance_id=VALID_ID, user_id="other")
    assert exc.value.status_code == 404


async def test_authorized_instance_id_path_400s_on_malformed_id(seed_pool):
    """Format validation runs before ownership so a bad id can't be used to
    probe which ids exist."""
    seed_pool()
    with pytest.raises(HTTPException) as exc:
        await routes.authorized_instance_id_path(instance_id="not-a-uuid", user_id=None)
    assert exc.value.status_code == 400


async def test_authorized_instance_id_query_returns_id_when_accessible(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    result = await routes.authorized_instance_id_query(simulation_instance_id=VALID_ID, user_id="owner-1")
    assert result == VALID_ID


async def test_authorized_instance_id_query_404s_when_inaccessible(seed_pool):
    seed_pool((VALID_ID, "owner-1"))
    with pytest.raises(HTTPException) as exc:
        await routes.authorized_instance_id_query(simulation_instance_id=VALID_ID, user_id=None)
    assert exc.value.status_code == 404


async def test_authorized_instance_id_query_400s_on_malformed_id(seed_pool):
    seed_pool()
    with pytest.raises(HTTPException) as exc:
        await routes.authorized_instance_id_query(simulation_instance_id="not-a-uuid", user_id=None)
    assert exc.value.status_code == 400


# --- add_store validates ownership on body-supplied instance id -------------


async def test_add_store_404s_when_instance_not_accessible(seed_pool):
    """POST /stores takes the instance id in the body (StoreInput) rather than
    via a dependency, so ownership must be enforced explicitly in the handler.
    Guards against a regression where that call is removed."""
    from food_access_model.api.helpers import StoreInput

    seed_pool((VALID_ID, "owner-1"))
    store = StoreInput(
        name="Test Market",
        category="supermarket",
        longitude="-89.4",
        latitude="43.07",
        simulation_instance_id=VALID_ID,
        simulation_step=0,
    )
    with pytest.raises(HTTPException) as exc:
        await routes.add_store(store, user_id="other")
    assert exc.value.status_code == 404
