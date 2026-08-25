"""
Cloudflare Access JWT verification (defense-in-depth).

Cloudflare Access already blocks unauthenticated requests at the edge before
they ever reach this box. This middleware is a second layer: if a request
reaches uvicorn directly — e.g. over Tailscale, LAN, or because a hostname
was misconfigured — without a valid `Cf-Access-Jwt-Assertion` header signed
by the team's identity provider, it gets rejected too.

Only active when settings.CF_ACCESS_ENABLED is true. No-op otherwise, so
local dev (NO auth) is unaffected.
"""
import time

import httpx
import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

_JWKS_CACHE: dict = {"keys": None, "fetched_at": 0.0}
_JWKS_TTL_SECONDS = 3600


def _certs_url() -> str:
    return f"https://{settings.CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs"


async def _get_signing_key(token: str):
    now = time.time()
    if _JWKS_CACHE["keys"] is None or now - _JWKS_CACHE["fetched_at"] > _JWKS_TTL_SECONDS:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_certs_url())
            resp.raise_for_status()
        _JWKS_CACHE["keys"] = resp.json()
        _JWKS_CACHE["fetched_at"] = now

    header = jwt.get_unverified_header(token)
    for key in _JWKS_CACHE["keys"].get("keys", []):
        if key.get("kid") == header.get("kid"):
            return jwt.PyJWK(key).key
    raise jwt.InvalidKeyError("No matching JWKS key for token")


class CloudflareAccessMiddleware(BaseHTTPMiddleware):
    """Rejects requests without a valid Cloudflare Access JWT."""

    EXEMPT_PATHS = {"/health"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        token = request.headers.get("Cf-Access-Jwt-Assertion")
        if not token:
            token = request.cookies.get("CF_Authorization")
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing Cloudflare Access token"})

        try:
            signing_key = await _get_signing_key(token)
            jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=settings.CF_ACCESS_AUD,
            )
        except Exception as exc:
            return JSONResponse(status_code=401, content={"detail": f"Invalid Cloudflare Access token: {exc}"})

        return await call_next(request)
