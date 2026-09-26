"""Shared OAuth2 + REST plumbing for the HTTP connectors (blueprint 6.2/6.5).

One implementation of the boring, dangerous parts - PKCE, token refresh,
rate limiting, retry, cursor extraction - so each provider file is reduced to
"endpoints + normalizer", which is what the blueprint asks for.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
from typing import Any, Callable, Mapping

import httpx

from omnilinker.connectors.contract import (
    AuthorizationRequest,
    AuthorizationResult,
    RawEnvelope,
    TokenGrant,
)
from omnilinker.connectors.ratelimit import AdaptiveThrottle, RateLimiter, RetryPolicy

USER_AGENT = "OmniLinker/0.1 (+https://localhost; local-first data indexer)"


def new_pkce_pair() -> tuple[str, str]:
    """RFC 7636. verifier: 43-128 chars from the unreserved set."""
    verifier = base64.urlsafe_b64encode(os.urandom(64)).decode().rstrip("=")[:96]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    return verifier, challenge


def new_state() -> str:
    return base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip("=")


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


def _explain_exchange_failure(descriptor, body: str) -> str:
    """Turn a token-endpoint rejection into a sentence about what to do.

    A raw `invalid_client` invites the conclusion "my secret is wrong", and for
    a Slack app that is usually the *wrong* conclusion - a provider console shows
    four similar-looking values (Client ID, Client Secret, Verification Token,
    Signing Secret) and the Verification Token in the Client Secret slot produces
    exactly this error. Naming the specific mistake turns a dead end into a
    thirty-second fix.
    """
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body[:300]
    if not isinstance(parsed, dict):
        return body[:300]

    code = str(parsed.get("error") or "")
    detail = str(parsed.get("error_description") or "")

    if code in ("invalid_client", "invalid_client_id", "unauthorized_client"):
        return (
            f"{code}: the provider rejected the client id and secret for "
            f"{descriptor.id}. Either the secret is wrong, or it is a different "
            f"credential than you think - check that you used the Client Secret "
            f"and not the Verification Token or the Signing Secret. Set it as "
            f"OMNI_OAUTH_{descriptor.id.upper()}_CLIENT_SECRET and restart. "
            f"({detail})" if detail else
            f"{code}: the provider rejected the client id and secret for "
            f"{descriptor.id}. Either the secret is wrong, or it is a different "
            f"credential than you think - check that you used the Client Secret "
            f"and not the Verification Token or the Signing Secret. Set it as "
            f"OMNI_OAUTH_{descriptor.id.upper()}_CLIENT_SECRET and restart."
        )
    if code in ("invalid_code", "bad_verification_code"):
        return (
            f"{code}: the authorization code was rejected. The code is single-use "
            f"and short-lived, so this usually means the callback was replayed, "
            f"or the state was already consumed. Start the Connect flow again."
        )
    if code in ("redirect_uri_mismatch", "invalid_redirect_uri"):
        return (
            f"{code}: the redirect URI sent to the provider does not match the "
            f"one registered with the app. Register "
            f"{descriptor.id}'s callback URL exactly, including scheme, port and "
            f"trailing path. ({detail})" if detail else
            f"{code}: the redirect URI sent to the provider does not match the "
            f"one registered with the app. Register the callback URL exactly, "
            f"including scheme, port and trailing path."
        )
    if code in ("invalid_scope", "invalid_scope_configuration"):
        return (
            f"{code}: the app has not requested the scopes this connector needs. "
            f"Add the missing scopes in the provider's console and reinstall the "
            f"app to the workspace. ({detail})" if detail else
            f"{code}: the app has not requested the scopes this connector needs. "
            f"Add the missing scopes in the provider's console and reinstall the "
            f"app to the workspace."
        )
    if code in ("access_denied", "user_denied"):
        return f"{code}: access was declined in the provider's approval screen."
    if code:
        return f"{code}: {detail}" if detail else code
    return body[:300]


class BaseRestConnector:
    """Mixin-style base for HTTP connectors.

    Subclasses set `descriptor` and implement `discover_streams`, `pull` and
    `normalize`. They get auth, throttling and retries for free.
    """

    descriptor: Any

    def __init__(self, *, timeout: float = 20.0, limiter: RateLimiter | None = None) -> None:
        self.timeout = timeout
        d = self.descriptor
        self.limiter = limiter or RateLimiter(d.rate_limit_per_sec, d.rate_limit_burst)
        self.throttle = AdaptiveThrottle(d.rate_limit_per_sec)
        self.retry = RetryPolicy()

    # -- auth ---------------------------------------------------------
    def _default_scopes(self) -> list[str]:
        return [s.name for s in self.descriptor.scopes]
    def _default_scopes(self) -> list[str]:
        return [s.name for s in self.descriptor.scopes]

    def _credentials(self) -> tuple[str, str]:
        """The client id and secret to present, from the one resolver.

        This used to read `{ID}_CLIENT_ID` straight out of the environment and
        fall back to the literal string `"demo-client-id"`. That was silently
        fatal: the *authorize* step went through `connections.client_credentials`
        and therefore sent the real configured id, so the app looked correctly
        configured, but the *token exchange* - the step that actually
        authenticates - sent `demo-client-id` and failed with `invalid_client`.
        The user would have been told their credentials were wrong when the
        credentials had never been read. Three call sites, one resolver, so the
        two halves of a handshake cannot disagree about who the client is.
        """
        from omnilinker.connectors.connections import client_credentials

        creds = client_credentials(self.descriptor.id)
        return creds.client_id, creds.client_secret

    def authorize(self, req: AuthorizationRequest) -> AuthorizationResult:
        d = self.descriptor
        if not d.authorize_url:
            raise RuntimeError(f"{d.id}: no authorization URL (auth_flow={d.auth_flow})")
        verifier = req.code_verifier or new_pkce_pair()[0]
        client_id, _ = self._credentials()
        params = {
            "client_id": client_id,
            "redirect_uri": req.redirect_uri,
            "response_type": "code",
            "state": req.state,
            "scope": " ".join(req.scopes or self._default_scopes()),
        }
        if d.pkce_required:
            params["code_challenge"] = _challenge(verifier)
            params["code_challenge_method"] = "S256"
        if req.login_hint:
            params["login_hint"] = req.login_hint
        sep = "&" if "?" in d.authorize_url else "?"
        return AuthorizationResult(
            authorization_url=f"{d.authorize_url}{sep}{urllib.parse.urlencode(params)}",
            state=req.state,
        )

    def exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> TokenGrant:
        d = self.descriptor
        if not d.token_url:
            raise RuntimeError(f"{d.id}: no token URL")
        client_id, client_secret = self._credentials()
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        if d.pkce_required:
            data["code_verifier"] = code_verifier
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                d.token_url, data=data,
                headers={"User-Agent": USER_AGENT,
                         "Accept": "application/json"},
            )
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _explain_exchange_failure(d, resp.text))
        return _grant_from_json(resp.json())

    def refresh_tokens(self, grant: TokenGrant) -> TokenGrant:
        d = self.descriptor
        if not d.refresh_url or not grant.refresh_token:
            return grant
        client_id, client_secret = self._credentials()
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": grant.refresh_token,
            "grant_type": "refresh_token",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(d.refresh_url, data=data, headers={"User-Agent": USER_AGENT})
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text)
        fresh = _grant_from_json(resp.json())
        # Providers that rotate refresh tokens omit the old one; keep it.
        return TokenGrant(
            access_token=fresh.access_token,
            refresh_token=fresh.refresh_token or grant.refresh_token,
            expires_at=fresh.expires_at,
            scopes=fresh.scopes or grant.scopes,
            extra=fresh.extra,
        )

    def revoke(self, grant: TokenGrant) -> None:
        return None

    # -- HTTP ---------------------------------------------------------
    def _request(
        self,
        grant: TokenGrant,
        method: str,
        url: str,
        *,
        limiter_key: str = "default",
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        """One HTTP call with throttling, retry and error typing."""
        attempt = 0
        while True:
            attempt += 1
            wait = self.throttle.wait_time()
            if wait:
                time.sleep(wait)
            if not self.limiter.acquire(limiter_key):
                raise ApiError(429, "local rate limiter exhausted (app-level budget)")

            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Authorization": f"Bearer {grant.access_token}",
            }
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.request(method, url, params=params, json=json_body,
                                          headers=headers)
            except httpx.TransportError as exc:
                if self.retry.should_retry(attempt, None, retryable_exc=True):
                    time.sleep(self.retry.delay(attempt))
                    continue
                raise ApiError(0, f"transport error: {exc}") from exc

            if resp.status_code < 400:
                self.throttle.reward()
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return {"_raw": resp.text}

            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            if self.retry.should_retry(attempt, resp.status_code):
                self.throttle.penalize(retry_after)
                time.sleep(self.retry.delay(attempt, retry_after))
                continue
            raise ApiError(resp.status_code, resp.text, retry_after)

    def _get_paginated(
        self,
        grant: TokenGrant,
        url: str,
        *,
        items_key: str,
        next_key: str,
        params: Mapping[str, Any] | None = None,
        limiter_key: str = "default",
        max_pages: int = 50,
    ) -> tuple[list[Any], str | None]:
        """Walk a cursor/page-token paginated collection. Returns (items, next)."""
        items: list[Any] = []
        cursor: str | None = None
        for _ in range(max_pages):
            page_params = dict(params or {})
            if cursor:
                page_params[next_key] = cursor
            body = self._request(grant, "GET", url, params=page_params, limiter_key=limiter_key)
            chunk = body.get(items_key) or []
            if isinstance(chunk, dict):  # Slack-style {"messages": [[...]]}
                chunk = chunk.get("messages", [])
            items.extend(chunk)
            cursor = body.get(next_key)
            if not cursor:
                break
        return items, cursor


def _challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:  # HTTP-date form
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(value)
            import datetime as _dt

            now = _dt.datetime.now(_dt.timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None


def _grant_from_json(body: Mapping[str, Any]) -> TokenGrant:
    expires_in = body.get("expires_in")
    return TokenGrant(
        access_token=body.get("access_token", ""),
        refresh_token=body.get("refresh_token"),
        expires_at=(time.time() + float(expires_in)) if expires_in else None,
        scopes=tuple((body.get("scope") or "").split()) if body.get("scope") else (),
        token_type=body.get("token_type", "Bearer"),
        extra={k: v for k, v in body.items()
               if k in {"id_token", "instance_url", "user_id", "workspace_id", "sub"}},
    )
