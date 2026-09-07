"""
Auth0 RS256 JWT verifier with JWKS caching.
Extracts claims and maps to local User records in PostgreSQL.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional
from uuid import UUID

import jwt
from jwt import PyJWT, PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

AUTH0_DOMAIN: str = os.environ.get("AUTH0_DOMAIN", "your-tenant.auth0.com")
AUTH0_AUDIENCE: str = os.environ.get("AUTH0_AUDIENCE", "https://api.single-pass-3d.io")
JWKS_URL: str = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
ALGORITHMS: list[str] = ["RS256"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# JWKS client caches signing keys automatically (PyJWT >= 2.6)
_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(JWKS_URL, cache_jwk_set=True, lifespan=600)
    return _jwks_client


def verify_token(token: str) -> dict[str, Any]:
    """
    Verify an Auth0 RS256 Bearer JWT.

    Raises:
        HTTPException 401 – token missing, expired, malformed, or wrong audience/issuer.

    Returns:
        Decoded JWT claims dict.
    """
    try:
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience.")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer.")
    except jwt.DecodeError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token decode error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token verification failed: {exc}")


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> dict[str, Any]:
    """
    FastAPI dependency that:
    1. Verifies the Bearer JWT with Auth0 JWKS.
    2. Returns the decoded claims dict.
       (Caller can use claims["sub"] as the Auth0 subject identifier.)

    DB lookup / creation is deferred to service-layer helpers so this
    dependency stays fast and stateless.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(token)
