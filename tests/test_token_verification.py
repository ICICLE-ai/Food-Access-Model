"""Tests for JWKS-backed bearer token verification (issue #102, ports FEAST-backend #219).

A real RSA keypair is generated once per session and used to sign the test
tokens; a stub ``PyJWKClient`` returns the matching public key so verification
runs the production code path end to end without a network dependency on a
live JWKS endpoint. "Bad signature" is modelled by signing with a *second*
keypair so the token is syntactically valid but the signature doesn't verify
against the stub's key -- same shape as a real forged token.

Env vars are set per-test via monkeypatch so a stray export in the operator's
shell can't make the suite pass or fail against the "real" verifier config.
"""
import time
from functools import lru_cache

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from food_access_model.api import token_verification


@lru_cache(maxsize=2)
def _rsa_keypair(seed=0):
    # Deterministic cache keyed by `seed` so the "wrong key" test gets a
    # genuinely different key, while repeated calls with the same seed
    # reuse the same instance -- RSA generation is slow (~0.5s) and would
    # otherwise dominate the suite runtime.
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def _clear_jwks_client_cache():
    """``_jwks_client`` memoizes per URL via lru_cache. Clear it between tests
    so a stub swapped in by one test doesn't leak into the next."""
    token_verification._jwks_client.cache_clear()


@pytest.fixture
def signing_keypair():
    return _rsa_keypair(0)


@pytest.fixture
def wrong_keypair():
    return _rsa_keypair(1)


@pytest.fixture
def configure_verifier(monkeypatch, signing_keypair):
    """Apply the standard JWT_* config for a verifier that will accept tokens
    signed by ``signing_keypair``. Returns a mutable dict so individual tests
    can tweak values before calling ``verify_token``."""
    config = {
        "JWT_JWKS_URL": "https://example.test/.well-known/jwks.json",
        "JWT_ISSUER": "https://example.test/",
        "JWT_AUDIENCE": "feast-backend",
    }

    for key, value in config.items():
        monkeypatch.setenv(key, value)

    class _StubJWK:
        def __init__(self, key):
            self.key = key

    class _StubClient:
        def __init__(self, key):
            self._key = key

        def get_signing_key_from_jwt(self, token):
            return _StubJWK(self._key)

    public_key = signing_keypair.public_key()
    monkeypatch.setattr(token_verification, "_jwks_client",
                        lambda url: _StubClient(public_key))
    return config


def _issue_token(private_key, iss="https://example.test/", aud="feast-backend",
                 sub="user-123", exp_offset=60, extra=None):
    now = int(time.time())
    payload = {"iss": iss, "aud": aud, "sub": sub,
               "iat": now, "exp": now + exp_offset}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, private_key, algorithm="RS256")


# --- happy path -------------------------------------------------------------


def test_valid_token_returns_sub(configure_verifier, signing_keypair):
    token = _issue_token(signing_keypair)
    assert token_verification.verify_token(token) == "user-123"


def test_valid_token_uses_configured_user_claim(configure_verifier, signing_keypair, monkeypatch):
    """JWT_USER_CLAIM lets a deployment pick a non-``sub`` claim (Tapis uses
    ``username``) without having to patch the verifier."""
    monkeypatch.setenv("JWT_USER_CLAIM", "username")
    token = _issue_token(signing_keypair, extra={"username": "jdoe"})
    assert token_verification.verify_token(token) == "jdoe"


# --- verification failures all return None ---------------------------------


def test_expired_token_returns_none(configure_verifier, signing_keypair):
    token = _issue_token(signing_keypair, exp_offset=-1)
    assert token_verification.verify_token(token) is None


def test_bad_signature_returns_none(configure_verifier, wrong_keypair):
    """Token signed by a key that is NOT in the JWKS -- same shape as a forged
    token. Must not be accepted even though the token is well-formed."""
    token = _issue_token(wrong_keypair)
    assert token_verification.verify_token(token) is None


def test_unknown_issuer_returns_none(configure_verifier, signing_keypair):
    token = _issue_token(signing_keypair, iss="https://attacker.example/")
    assert token_verification.verify_token(token) is None


def test_wrong_audience_returns_none(configure_verifier, signing_keypair):
    token = _issue_token(signing_keypair, aud="some-other-service")
    assert token_verification.verify_token(token) is None


def test_missing_exp_claim_returns_none(configure_verifier, signing_keypair):
    """``options={"require": ["exp"]}`` is a defence-in-depth check: a provider
    that emits non-expiring tokens is a security smell, so refuse rather than
    accept the first `sub` we see."""
    payload = {"iss": "https://example.test/", "aud": "feast-backend", "sub": "user-123"}
    token = jwt.encode(payload, signing_keypair, algorithm="RS256")
    assert token_verification.verify_token(token) is None


def test_malformed_token_returns_none(configure_verifier):
    assert token_verification.verify_token("not-a-jwt") is None


def test_token_without_user_claim_returns_none(configure_verifier, signing_keypair, monkeypatch):
    monkeypatch.setenv("JWT_USER_CLAIM", "username")
    # Token has a sub but no username; verification succeeds, extraction fails.
    token = _issue_token(signing_keypair)
    assert token_verification.verify_token(token) is None


# --- unconfigured verifier short-circuits ----------------------------------


def test_unset_jwks_url_disables_verification(monkeypatch):
    """A dev deployment that leaves JWT_JWKS_URL unset must NOT trust any
    token: the request falls through to public-pool behavior rather than
    accepting the raw token as the user id (the trust-as-is behavior this
    port explicitly removes)."""
    monkeypatch.delenv("JWT_JWKS_URL", raising=False)
    assert token_verification.verify_token("anything-at-all") is None


def test_empty_jwks_url_is_treated_as_unset(monkeypatch):
    """An operator that clears the env var to the empty string should get the
    "disabled" behavior, not a verifier that tries to fetch from ``""``."""
    monkeypatch.setenv("JWT_JWKS_URL", "")
    assert token_verification.verify_token("anything-at-all") is None


# --- issuer/audience are optional ------------------------------------------


def test_issuer_optional_when_jwt_issuer_unset(monkeypatch, signing_keypair):
    """If the operator didn't set JWT_ISSUER, tokens with any (or no) ``iss``
    are accepted -- audience-pinning is similarly optional. Keeps the verifier
    usable against providers that don't emit stable issuer URLs without
    silently relaxing checks the operator did configure."""
    monkeypatch.setenv("JWT_JWKS_URL", "https://example.test/.well-known/jwks.json")
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    class _StubJWK:
        def __init__(self, key):
            self.key = key

    class _StubClient:
        def __init__(self, key):
            self._key = key

        def get_signing_key_from_jwt(self, token):
            return _StubJWK(self._key)

    monkeypatch.setattr(token_verification, "_jwks_client",
                        lambda url: _StubClient(signing_keypair.public_key()))

    token = _issue_token(signing_keypair, iss="whatever", aud="whatever")
    assert token_verification.verify_token(token) == "user-123"
