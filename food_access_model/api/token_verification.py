"""Bearer token verification for optional per-request user scoping (#102).

Ports FEAST-backend #219's JWKS-backed JWT verifier onto this ICICLE
deployment. Any client-supplied token has to pass signature + expiry checks
(plus optional issuer/audience pinning) before its claim is accepted as the
user id.

The implementation is deliberately generic: it will work for any provider
that exposes a JWKS URL. The ICICLE deployment sets ``JWT_JWKS_URL`` to a
Tapis JWKS; other deployments can point it at their own OIDC provider.
Deployment-specific pieces (JWKS URL, expected issuer / audience, which claim
carries the user id) live in env vars, not code.

Two distinct "no user id" outcomes are intentional:

* ``JWT_JWKS_URL`` unset -> ``verify_token`` returns None. The verifier is
  disabled; every request falls through to the shared public pool. Lets a
  local dev operator skip standing up an OIDC provider.
* Verifier enabled but the token itself fails -> ``verify_token`` raises
  ``TokenVerificationError``. The caller (``get_current_user_id`` /
  ``_extract_user_id_from_token``) maps this to HTTP 401 rather than
  silently downgrading the request to anonymous, which would let a
  logged-in user with an expired Tapis token accidentally create or mutate
  public-pool rows. See PR review on this change (#104).

A trust-as-is dev fallback was considered and rejected: dev and prod should
be on the same verification code path to avoid regressions slipping through
dev review.
"""
import logging
import os
from functools import lru_cache

import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError

logger = logging.getLogger(__name__)


class TokenVerificationError(Exception):
    """Raised when a bearer token is present but cannot be verified.

    Separate from ``verify_token`` returning None (which signals "verifier
    disabled"): callers translate this exception into HTTP 401 so a
    present-but-invalid token does not silently fall through to anonymous.
    """


# RS256 by default: every OIDC-conformant JWKS uses asymmetric keys
# (RSA / EC). HS256-style shared-secret algorithms are not fetched via JWKS,
# so there's no sensible default for them here -- operators that need a
# different algorithm override via JWT_ALGORITHMS.
_DEFAULT_ALGORITHMS = "RS256"
_DEFAULT_USER_CLAIM = "sub"
_JWKS_FETCH_TIMEOUT_SECONDS = 5
_JWKS_CACHE_LIFESPAN_SECONDS = 3600


def _env(name, default=None):
    """Return a non-empty env var or the default. Treats "" as absent so an
    operator who clears a var in their shell gets the fall-through behavior
    rather than a mysterious failure deeper in.
    """
    value = os.getenv(name)
    return value if value else default


@lru_cache(maxsize=4)
def _jwks_client(url):
    """Memoized per-URL so the HTTP JWKS fetch happens once per process, not
    per request. PyJWKClient also caches signing keys internally after the
    first fetch; the ``lifespan`` here governs when it will re-fetch to pick
    up a key rotation.
    """
    return PyJWKClient(
        url,
        cache_keys=True,
        lifespan=_JWKS_CACHE_LIFESPAN_SECONDS,
        timeout=_JWKS_FETCH_TIMEOUT_SECONDS,
    )


def verify_token(token):
    """Verify a bearer token and return its user id.

    Returns None when ``JWT_JWKS_URL`` is unset (verifier disabled; caller
    falls through to the public pool). Raises ``TokenVerificationError`` when
    the verifier is enabled but the token is bad -- bad signature, expired,
    wrong issuer/audience, unknown key id, missing ``exp``, malformed token,
    JWKS fetch failure, or verified-but-missing user claim. Callers map that
    exception to HTTP 401 rather than silently downgrading to anonymous.

    Config is read at call time (not import time) so a server that boots
    before its ``.env`` is populated still picks up the config on the first
    real request.
    """
    jwks_url = _env("JWT_JWKS_URL")
    if not jwks_url:
        # Verification deliberately disabled; see module docstring.
        return None

    algorithms_raw = _env("JWT_ALGORITHMS", _DEFAULT_ALGORITHMS)
    algorithms = [a.strip() for a in algorithms_raw.split(",") if a.strip()]
    issuer = _env("JWT_ISSUER")
    audience = _env("JWT_AUDIENCE")
    user_claim = _env("JWT_USER_CLAIM", _DEFAULT_USER_CLAIM)

    # Issuer / audience pinning is opt-in: an operator who hasn't set
    # JWT_ISSUER / JWT_AUDIENCE gets a verifier that validates the signature
    # and expiry but doesn't enforce iss/aud. For audience specifically PyJWT
    # defaults to *requiring* audience whenever the token carries an ``aud``
    # claim, so we have to explicitly disable that check to match the
    # intent ("opt-in only"). Issuer validation works correctly by omission.
    decode_kwargs = {"algorithms": algorithms, "options": {"require": ["exp"]}}
    if issuer:
        decode_kwargs["issuer"] = issuer
    if audience:
        decode_kwargs["audience"] = audience
    else:
        decode_kwargs["options"]["verify_aud"] = False

    try:
        signing_key = _jwks_client(jwks_url).get_signing_key_from_jwt(token)
        payload = jwt.decode(token, signing_key.key, **decode_kwargs)
    except PyJWKClientError as exc:
        # JWKS fetch / parse failure is an ops signal: a misconfigured
        # JWT_JWKS_URL in prod (wrong Tapis URL, Tapis down, etc.) would
        # otherwise silently 401 every logged-in user and the only evidence
        # would be a debug log nobody is tailing.
        logger.warning("JWKS client error (%s): %s", type(exc).__name__, exc)
        raise TokenVerificationError("JWKS client error") from exc
    except InvalidTokenError as exc:
        logger.debug("Invalid token (%s): %s", type(exc).__name__, exc)
        raise TokenVerificationError("Invalid token") from exc

    user_id = payload.get(user_claim)
    if not user_id:
        # Signature + expiry + iss/aud all passed, but the claim the
        # deployment is configured to scope on isn't in the token. From the
        # user's perspective the token is effectively invalid for this
        # service, so 401 rather than downgrade to anonymous.
        logger.debug("Verified token missing %s claim", user_claim)
        raise TokenVerificationError("Token missing " + user_claim + " claim")
    return str(user_id)
