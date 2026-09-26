"""OAuth connection management: PKCE handshake, sealed token storage, lifecycle.

Nothing here existed while every connector declared an `auth_flow` and the UI
showed a "Connect" button that did nothing. The connectors have always
implemented `authorize` / `exchange_code` / `refresh_tokens` / `revoke`; what was
missing was the state machine *around* them - where the PKCE verifier waits
across a browser redirect, where the resulting refresh token lives, and what
happens when the user disconnects.

Four things here are security decisions rather than plumbing, and each is load
-bearing:

**1. Tokens are sealed, and with their own key.**
A refresh token is a bearer credential for the user's *whole* Slack workspace -
strictly more sensitive than a message body, which is one record. It is stored
as an AEAD envelope, and the key is derived from a dedicated info string
(`omnilinker/credential/v1|<workspace>`) rather than reusing a per-source DEK,
so losing or rotating content keys cannot be used to read credentials and vice
versa. The AAD binds the ciphertext to `<workspace>|<connector>`, so a token
cannot be lifted from one connection row into another.

**2. The PKCE verifier lives server-side, keyed by an opaque state.**
The verifier never reaches the browser. `state` is a random handle; the browser
only ever holds that, and the callback trades it for the verifier. This is what
makes the flow CSRF-resistant: an attacker who starts their own authorization
cannot complete yours, because their `state` is not in the pending table, and
cannot read yours either, because there is no endpoint that returns a verifier.

**3. Pending authorizations expire in ten minutes and are single-use.**
A `state` is deleted the moment it is redeemed, whether or not the exchange then
succeeds. Anything else is a replay window.

**4. No endpoint ever returns a token.**
`/api/connections` reports *whether* a connector is connected, when the grant
expires, and which scopes were granted. Not the token, not a prefix of it. A
UI that needs to know "is this connected" must not be able to learn "and here is
the credential", because that is the difference between a status page and a
credential exfiltration endpoint.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode

from omnilinker.config import get_settings
from omnilinker.connectors.base_rest import new_pkce_pair, new_state
from omnilinker.connectors.contract import (
    AuthorizationRequest,
    AuthorizationResult,
    Connector,
    TokenGrant,
)
from omnilinker.connectors.registry import get_registry
from omnilinker.crypto.envelope import DecryptionError, encrypt, is_envelope
from omnilinker.crypto.keys import get_key_manager
from omnilinker.ids import prefixed
from omnilinker.normalize import iso
from omnilinker.store import get_stores
from omnilinker.store.base import Query

CONNECTIONS = "connections"
PENDING = "oauth_pending"

#: A `state` is good for ten minutes. Long enough for a user to work through a
#: consent screen and a 2FA prompt; short enough that an abandoned flow does not
#: sit in the store.
PENDING_TTL_SECONDS = 600


class ConnectionError_(RuntimeError):
    """Named with a trailing underscore so it never shadows the builtin."""


@dataclass
class ClientCredentials:
    client_id: str
    client_secret: str = ""
    #: Notion and Evernote publish *public* desktop clients whose secret is a
    #: fixed non-secret string. Sending a blank one makes them reject the token
    #: exchange with an opaque 400.
    is_public_client: bool = False

    @property
    def usable(self) -> bool:
        return bool(self.client_id)


def client_credentials(provider: str) -> ClientCredentials:
    """Resolve a provider's client credentials from the environment.

    Read lazily and by convention (`OMNI_OAUTH_<PROVIDER>_CLIENT_ID`) rather
    than through a hardcoded table, so adding a provider needs no change here.
    The alternative - a config field per provider - is a file that has to be
    edited every time somebody ships a connector, and it is always wrong by one.
    """
    provider = provider.strip().lower().replace(".", "_")
    env_id = os.environ.get(f"OMNI_OAUTH_{provider.upper()}_CLIENT_ID", "").strip()
    env_secret = os.environ.get(f"OMNI_OAUTH_{provider.upper()}_CLIENT_SECRET", "").strip()
    if env_id:
        return ClientCredentials(env_id, env_secret)
    # Public clients ship their identifier in the connector, because it is
    # published in the provider's own documentation and is not a secret.
    connector = get_registry().get(provider)
    public_id = getattr(connector.descriptor, "public_client_id", "") or ""
    if public_id:
        return ClientCredentials(public_id, env_secret, is_public_client=True)
    return ClientCredentials("")


def redirect_uri(connector_id: str) -> str:
    """Where the provider sends the browser back to.

    Built from one configured public URL rather than assembled per connector, so
    the value registered with the provider is predictable and a mismatch is a
    single config fix instead of a hunt.
    """
    base = get_settings().public_url.rstrip("/")
    return f"{base}/api/connections/{connector_id}/callback"


# ---------------------------------------------------------------------------
# Pending authorizations
# ---------------------------------------------------------------------------


def _pending_id(state: str) -> str:
    return f"pend_{state}"


def begin_authorization(
    connector_id: str,
    *,
    workspace: str,
    login_hint: str | None = None,
    scopes: Sequence[str] | None = None,
) -> AuthorizationResult:
    """Step 1: mint a PKCE pair, stash the verifier, return the provider URL.

    The verifier is stored and never returned. A provider that does not require
    PKCE still gets one, because a code-verifier in an extra parameter is
    ignored rather than rejected by every provider that does not ask for it.
    """
    registry = get_registry()
    if not registry.has(connector_id):
        raise ConnectionError_(f"unknown connector {connector_id!r}")
    descriptor = registry.get(connector_id).descriptor
    if descriptor.auth_flow == "none" or not descriptor.authorize_url:
        raise ConnectionError_(
            f"{connector_id} does not use an OAuth redirect (auth_flow="
            f"{descriptor.auth_flow!r}). It is configured with credentials or a "
            f"file, not a consent screen."
        )
    creds = client_credentials(connector_id)
    if not creds.usable:
        raise ConnectionError_(
            f"{connector_id} has no client id. Set "
            f"OMNI_OAUTH_{connector_id.upper()}_CLIENT_ID (and _CLIENT_SECRET) "
            f"and restart. The Connectors page shows which is missing."
        )

    verifier, challenge = new_pkce_pair()
    state = new_state()

    params: dict[str, str] = {
        "client_id": creds.client_id,
        "redirect_uri": redirect_uri(connector_id),
        "response_type": "code",
        "state": state,
    }
    wanted = list(scopes or [s.name for s in descriptor.scopes if s.name])
    if wanted:
        # Slack and Google want space-separated; Notion/Evernote want commas.
        joiner = "," if connector_id in ("notion", "evernote") else " "
        params["scope"] = joiner.join(wanted)
    if descriptor.pkce_required:
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"
    if login_hint:
        params["login_hint"] = login_hint

    stores = get_stores().docs
    _prune_pending(stores)
    stores.put(PENDING, _pending_id(state), {
        "_id": _pending_id(state),
        "workspace_id": workspace,
        "connector_id": connector_id,
        "state": state,
        "code_verifier": verifier,
        "redirect_uri": params["redirect_uri"],
        "created_at": iso(),
        "expires_at": time.time() + PENDING_TTL_SECONDS,
    })

    separator = "&" if "?" in descriptor.authorize_url else "?"
    return AuthorizationResult(
        authorization_url=f"{descriptor.authorize_url}{separator}{urlencode(params)}",
        state=state,
        expires_in=PENDING_TTL_SECONDS,
    )


def _prune_pending(stores: Any) -> int:
    now = time.time()
    removed = 0
    for row in stores.find(Query(PENDING, limit=500)):
        if float(row.get("expires_at") or 0) < now:
            stores.delete(PENDING, row["_id"])
            removed += 1
    return removed


def complete_authorization(
    connector_id: str,
    *,
    workspace: str,
    code: str,
    state: str,
    error: str | None = None,
) -> dict:
    """Step 2: redeem the state for the verifier, exchange the code, store it.

    `state` is consumed first and unconditionally, so a callback can be replayed
    only once and only if it arrives before the TTL. The exchange failing
    afterwards does not put the verifier back - a user retrying gets a new flow,
    which is the correct behaviour for a credential exchange.
    """
    if error:
        raise ConnectionError_(f"the provider returned an error: {error}")
    if not state:
        raise ConnectionError_("the provider did not return a state value")
    if not code:
        raise ConnectionError_("the provider did not return an authorization code")

    stores = get_stores().docs
    pending = stores.get(PENDING, _pending_id(state))
    if not pending:
        raise ConnectionError_(
            "this authorization link has expired or was already used. "
            "Start the connection again."
        )
    if float(pending.get("expires_at") or 0) < time.time():
        stores.delete(PENDING, pending["_id"])
        raise ConnectionError_("this authorization link expired. Start again.")
    if pending.get("workspace_id") != workspace:
        raise ConnectionError_("this authorization belongs to a different workspace.")
    if pending.get("connector_id") != connector_id:
        raise ConnectionError_(
            f"this authorization was started for {pending.get('connector_id')!r}, "
            f"not {connector_id!r}."
        )

    # Single use, before the network call.
    stores.delete(PENDING, pending["_id"])

    connector = get_registry().get(connector_id)
    grant = connector.exchange_code(
        code, pending["code_verifier"], pending["redirect_uri"]
    )
    _store_grant(connector_id, workspace, grant, connector)
    return describe(connector_id, workspace)


# ---------------------------------------------------------------------------
# Grant storage
# ---------------------------------------------------------------------------


def _connection_id(connector_id: str, workspace: str) -> str:
    return f"conn_{connector_id}_{workspace}"


def _seal(value: str, connector_id: str, workspace: str, field: str) -> dict:
    km = get_key_manager()
    return encrypt(
        # A dedicated source name, so credentials are not encrypted under a
        # content key: rotating content keys must not strand a refresh token.
        f"credential:{connector_id}",
        value,
        doc_id=_connection_id(connector_id, workspace),
        field_path=field,
        key_manager=km,
    )


def _unseal(envelope: Any, connector_id: str, workspace: str, field: str) -> str:
    from omnilinker.crypto.envelope import decrypt

    return decrypt(
        f"credential:{connector_id}", envelope,
        doc_id=_connection_id(connector_id, workspace),
        field_path=field, key_manager=get_key_manager(),
    ).decode("utf-8")


def _store_grant(connector_id: str, workspace: str, grant: TokenGrant,
                 connector: Connector | None = None) -> None:
    existing = get_stores().docs.get(CONNECTIONS, _connection_id(connector_id, workspace)) or {}
    refresh = grant.refresh_token or ""
    doc: dict[str, Any] = {
        **existing,
        "_id": _connection_id(connector_id, workspace),
        "workspace_id": workspace,
        "connector_id": connector_id,
        "access_token": _seal(grant.access_token, connector_id, workspace, "access_token"),
        "access_token_present": bool(grant.access_token),
        "refresh_token": _seal(refresh, connector_id, workspace, "refresh_token")
        if refresh else "",
        "refresh_token_present": bool(refresh),
        "expires_at": grant.expires_at,
        "scopes": list(grant.scopes),
        "token_type": grant.token_type,
        "connected_at": existing.get("connected_at", iso()),
        "updated_at": iso(),
        "status": "connected",
    }
    if not grant.access_token:
        # A 200 from the token endpoint carrying no access token is a failure
        # that looks like a success. Storing it as connected would present a
        # dead connection to the user until their first sync.
        get_stores().docs.put(CONNECTIONS, doc["_id"], {
            **doc, "status": "failed", "status_detail": "the token endpoint returned no access token",
        })
        return
    if grant.extra:
        # Provider-specific non-secret context (workspace id, instance url,
        # user id). None of these are credentials; none are treated as secret.
        doc["provider_context"] = {k: v for k, v in grant.extra.items()
                                   if k not in ("access_token", "refresh_token")}
    get_stores().docs.put(CONNECTIONS, doc["_id"], doc)


def get_grant(connector_id: str, workspace: str, *, refresh: bool = True) -> TokenGrant | None:
    """Load the stored grant, refreshing it if it is close to expiry.

    A connector without a refresh token (or one whose provider issues none) is
    returned as-is; the next sync will fail loudly rather than silently
    pretending to be connected, which is what `_expired_grant` below makes
    explicit.
    """
    doc = get_stores().docs.get(CONNECTIONS, _connection_id(connector_id, workspace))
    if not doc or doc.get("status") != "connected":
        return None
    try:
        access = _unseal(doc["access_token"], connector_id, workspace, "access_token")
    except (DecryptionError, KeyError, TypeError):
        # A credential we cannot decrypt is worse than no credential: it looks
        # connected and fails at the first API call. Surface it as disconnected.
        return None
    refresh_token = None
    if doc.get("refresh_token"):
        try:
            refresh_token = _unseal(doc["refresh_token"], connector_id, workspace,
                                    "refresh_token") or None
        except (DecryptionError, KeyError, TypeError):
            refresh_token = None

    grant = TokenGrant(
        access_token=access,
        refresh_token=refresh_token,
        expires_at=doc.get("expires_at"),
        scopes=tuple(doc.get("scopes") or ()),
        token_type=doc.get("token_type", "Bearer"),
        extra=dict(doc.get("provider_context") or {}),
    )
    if not refresh or not grant.needs_refresh():
        return grant
    if not refresh_token:
        return grant

    try:
        connector = get_registry().get(connector_id)
        fresh = connector.refresh_tokens(grant)
    except Exception as exc:
        _mark_status(connector_id, workspace, "refresh_failed", str(exc)[:200])
        return grant
    _store_grant(connector_id, workspace, fresh)
    return fresh


def _mark_status(connector_id: str, workspace: str, status: str, detail: str = "") -> None:
    doc = get_stores().docs.get(CONNECTIONS, _connection_id(connector_id, workspace))
    if not doc:
        return
    doc["status"] = status
    doc["status_detail"] = detail
    doc["updated_at"] = iso()
    get_stores().docs.put(CONNECTIONS, doc["_id"], doc)


def describe(connector_id: str, workspace: str) -> dict:
    """Status for the UI. Deliberately contains no token material."""
    doc = get_stores().docs.get(CONNECTIONS, _connection_id(connector_id, workspace)) or {}
    creds = client_credentials(connector_id)
    descriptor = get_registry().get(connector_id).descriptor if get_registry().has(connector_id) \
        else None
    # Connected means "there is a token AND we can still read it". Checking only
    # for a row means a credential we can no longer decrypt presents as
    # connected and fails at the first API call instead of here.
    connected = doc.get("status") == "connected" and bool(doc.get("access_token"))
    if connected:
        try:
            connected = bool(_unseal(doc["access_token"], connector_id, workspace,
                                     "access_token"))
        except (DecryptionError, KeyError, TypeError):
            connected = False
    expires_at = doc.get("expires_at")
    return {
        "connector_id": connector_id,
        "connected": connected,
        "status": doc.get("status", "disconnected"),
        "status_detail": doc.get("status_detail", ""),
        "connected_at": doc.get("connected_at", ""),
        "updated_at": doc.get("updated_at", ""),
        "expires_at": (
            datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z")
            if expires_at else ""
        ),
        "needs_refresh": bool(expires_at and float(expires_at) < time.time() + 120),
        "scopes": doc.get("scopes", []),
        "access_token_present": bool(doc.get("access_token_present")),
        "refresh_token_present": bool(doc.get("refresh_token_present")),
        "provider_context": doc.get("provider_context", {}),
        "configured": creds.usable,
        "credentials_source": "environment" if creds.usable else
                              ("public_client" if getattr(descriptor, "public_client_id", "") else "none"),
        "auth_flow": getattr(descriptor, "auth_flow", "unknown"),
        "redirect_uri": redirect_uri(connector_id) if connected or creds.usable else "",
        # What the user has to do next, as a sentence. The UI renders this
        # verbatim rather than inventing its own instructions.
        "next_step": _next_step(connector_id, creds, descriptor, connected, doc),
    }


def _next_step(connector_id: str, creds: ClientCredentials, descriptor: Any,
               connected: bool, doc: Mapping[str, Any]) -> str:
    auth_flow = getattr(descriptor, "auth_flow", "none")
    status = doc.get("status", "disconnected")
    # Status first, and *not* gated on `connected`. A failed refresh leaves the
    # row in a state where `connected` is already False, so testing both
    # together made the branch unreachable and the user was told "ready to
    # connect" about a connection they had made and that had just been rejected.
    if status == "refresh_failed":
        return ("The refresh token was rejected, so the connection cannot be kept "
                "alive. Disconnect and reconnect to grant access again.")
    if status == "unauthorized":
        return ("The provider rejected the stored grant. Disconnect and "
                "reconnect - the access may have been revoked outside this app.")
    if status == "failed":
        return doc.get("status_detail") or ("The token exchange failed. Start the "
                                             "connection again.")
    if connected:
        if doc.get("needs_refresh"):
            return "Connected. The access token expires soon; it refreshes automatically."
        return "Connected. Sync to pull data."
    if auth_flow in ("export_file", "file"):
        return ("No account connection. Supply the export file the provider "
                "generates, then sync.")
    if auth_flow == "none":
        return "No credentials needed. Sync to pull data."
    if auth_flow == "api_key":
        return (f"Set this provider's API credentials in the environment, then "
                f"sync. There is no consent screen for a {auth_flow} connector.")
    if not getattr(descriptor, "authorize_url", ""):
        return f"{auth_flow} connector - configure it in the environment, then sync."
    if not creds.usable:
        return (f"Set OMNI_OAUTH_{connector_id.upper()}_CLIENT_ID (and "
                f"_CLIENT_SECRET if the provider issued one), restart, then Connect.")
    return "Ready to connect. You will be sent to the provider to approve access."


def list_connections(workspace: str) -> list[dict]:
    registry = get_registry()
    return [describe(connector_id, workspace) for connector_id in registry.ids()]


def disconnect(connector_id: str, workspace: str, *, revoke_remote: bool = True) -> dict:
    """Remove a connection, revoking remotely first when we can.

    A local delete without a remote revoke leaves a live grant sitting at the
    provider that the user believes they have revoked. The remote call is
    best-effort - if it fails we still delete locally and report that the
    provider was not told, because leaving a credential on disk after the user
    asked to remove it is the worse failure.
    """
    doc = get_stores().docs.get(CONNECTIONS, _connection_id(connector_id, workspace))
    if not doc:
        return {"disconnected": False, "reason": "not connected"}
    revoked = False
    revoke_error = ""
    if revoke_remote:
        grant = get_grant(connector_id, workspace, refresh=False)
        if grant:
            try:
                get_registry().get(connector_id).revoke(grant)
                revoked = True
            except Exception as exc:
                revoke_error = str(exc)[:200]
    get_stores().docs.delete(CONNECTIONS, doc["_id"])
    return {
        "disconnected": True,
        "revoked_remotely": revoked,
        "revoke_error": revoke_error,
        "note": ("The provider was not notified, so revoke the grant in its own "
                 "settings." if revoke_error else ""),
    }


def verify(connector_id: str, workspace: str) -> dict:
    """Prove a stored grant actually works, rather than reporting "connected"
    because a token row exists."""
    grant = get_grant(connector_id, workspace)
    if not grant:
        return {"ok": False, "error": "not connected"}
    connector = get_registry().get(connector_id)
    try:
        streams = connector.discover_streams(grant)
    except Exception as exc:
        _mark_status(connector_id, workspace, "unauthorized", str(exc)[:200])
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                "hint": "The grant is stored but the provider rejected it. "
                        "Disconnect and reconnect."}
    return {"ok": True, "streams": len(streams),
            "sample": [s.label for s in streams[:3]]}


# Provider error codes that are *about the client credentials*. Everything else
# an `authorization_code` probe can return - `invalid_code`, `invalid_grant`,
# `unsupported_grant_type` - is about the fabricated code, and therefore says
# nothing whatsoever about the secret.
_CREDENTIAL_ERRORS = frozenset({
    "invalid_client", "invalid_client_id", "unauthorized_client",
})


def _probe_once(token_url: str, client_id: str, secret: str, redirect: str) -> dict:
    """One `authorization_code` request with a fabricated code.

    Returns the provider's error code verbatim. The point is what it says when
    the client is *wrong* - which is why the caller compares it against the
    candidate's answer before trusting either.
    """
    import httpx

    try:
        response = httpx.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": secret,
                "grant_type": "authorization_code",
                "code": "omnilinker-credential-probe",
                "redirect_uri": redirect,
            },
            timeout=15.0,
        )
    except Exception as exc:
        return {"error_code": f"network:{type(exc).__name__}", "status": 0}

    try:
        body = response.json()
    except ValueError:
        return {"error_code": f"http_{response.status_code}_non_json",
                "status": response.status_code}
    if not isinstance(body, Mapping):
        return {"error_code": f"http_{response.status_code}_non_object",
                "status": response.status_code}
    return {"error_code": str(body.get("error") or ""), "status": response.status_code,
            "body": dict(body)}


def diagnose_credentials(connector_id: str, candidate_secret: str) -> dict:
    """Test a candidate client secret against the provider, definitively.

    Exists because the credentials a user reads off a provider's console are
    four values that look alike - Client ID, Client Secret, Verification Token,
    Signing Secret - and putting the wrong one in the Client Secret slot
    produces `invalid_client` at the token exchange, several steps later, with
    an error that does not say which value was wrong.

    Whether a provider *can* answer this without a real authorization code
    varies, and Slack cannot: its token endpoint checks the code before the
    client credentials, so a real secret and a nonsense one produce the same
    `invalid_code`. So the probe runs a control first, and if the control
    matches the candidate it reports `inconclusive` rather than a false pass.
    The alternative - reporting a guess as a result - is worse than reporting
    nothing, because the user stops looking.

    The candidate is never stored. It is used for one request and dropped.
    """
    connector = get_registry().get(connector_id)
    descriptor = connector.descriptor
    creds = client_credentials(connector_id)
    if not descriptor.token_url:
        return {"ok": False, "error": f"{connector_id} has no token endpoint to test against"}
    if not creds.usable:
        return {"ok": False, "error": f"{connector_id} has no client id configured"}

    # One request, through the same helper the control uses - so the candidate
    # and the control differ *only* in the secret, which is the entire basis of
    # the comparison below.
    result = _probe_once(descriptor.token_url, creds.client_id, candidate_secret,
                         redirect_uri(connector_id))
    code = str(result.get("error_code") or "")
    status = int(result.get("status") or 0)
    body = result.get("body") or {}

    # **Self-validating control.** Before reporting a result, probe the same
    # endpoint with a deliberately wrong secret. That control establishes what
    # the provider says when the secret is *definitely* wrong, which is the only
    # way to tell a real answer from an artefact of the probe.
    #
    # Two earlier versions of this probe were wrong, in the direction that
    # matters: they reported `credentials_accepted` for a value that was not the
    # secret at all. A confidently wrong answer is worse than no answer, because
    # the user stops looking. The control is what makes that unrepresentable.
    #
    # A transport failure is the provider's answer being unreachable, not a
    # verdict, so it short-circuits before the comparison rather than being
    # compared - otherwise both requests fail identically and every unreachable
    # endpoint would look like a matched control.
    if code.startswith(("network:", "http_")):
        return {
            "ok": False,
            "reason": code.split(":", 1)[0],
            "error": str(body.get("error_description") or body or code)[:200],
            "hint": "The token endpoint could not be reached, so the secret was "
                    "never tested.",
            "provider_error": code,
        }

    control = _probe_once(descriptor.token_url, creds.client_id,
                          "omnilinker-control-value-not-a-real-secret",
                          redirect_uri(connector_id))
    control_code = str(control.get("error_code") or "")

    if control_code and control_code == code:
        # Same verdict for a known-bad secret and for the candidate. What that
        # means depends entirely on *what the verdict is about*:
        #
        #  - A credential error means the provider did read the secret and
        #    rejected it. The candidate is genuinely wrong, and saying so is
        #    correct - this is a real answer, not a missing one.
        #  - An error about the code or the grant means the provider never
        #    reached the credential check, so the secret was never examined and
        #    the two cases are indistinguishable. Say so.
        #
        # Slack is the second: `oauth.v2.access` rejects a fabricated code with
        # `invalid_code` before it looks at the client credentials at all.
        if code in _CREDENTIAL_ERRORS:
            return {
                "ok": False,
                "reason": "invalid_client",
                "error": "the provider rejected this client id and secret pair",
                "hint": "This is not a valid Client ID + Client Secret for that "
                        "app. Check that the value is the Client Secret and not "
                        "the Verification Token or the Signing Secret.",
                "provider_error": code,
            }
        return {
            "ok": False,
            "reason": "inconclusive",
            "error": f"the provider answers '{code}' for a known-bad secret too, "
                     f"so it never reached the credential check and this endpoint "
                     f"cannot validate a client secret",
            "hint": "This provider checks the authorization code before the "
                    "client credentials, so a secret cannot be tested on its "
                    "own. Use the Connect button: the real handshake either "
                    "succeeds or reports invalid_client, which names the problem.",
            "provider_error": code,
        }

    if code in ("invalid_client", "invalid_client_id", "unauthorized_client"):
        return {
            "ok": False,
            "reason": "invalid_client",
            "error": "the provider rejected this client id and secret pair",
            "hint": "This is not a valid Client ID + Client Secret for that app. "
                    "Check that the value is the Client Secret and not the "
                    "Verification Token or the Signing Secret.",
            "provider_error": code,
        }
    if code in ("invalid_code", "bad_verification_code", "invalid_grant"):
        return {
            "ok": True,
            "reason": "credentials_accepted",
            "note": "The provider authenticated the client id and secret and then "
                    "rejected the fabricated code - which is the expected outcome "
                    "and means the pair is correct.",
            "provider_error": code,
        }
    if code == "unsupported_grant_type":
        # Conclusive for neither answer, and saying so is the only honest result.
        return {
            "ok": False,
            "reason": "inconclusive",
            "error": "the provider rejected the probe grant type before "
                     "authenticating the client, so this says nothing about the "
                     "secret",
            "hint": "Use the Connect button instead; the token exchange during "
                    "the real handshake will accept or reject the pair.",
            "provider_error": code,
        }
    if status == 200 and not code:
        return {"ok": True, "reason": "accepted",
                "note": "The token endpoint accepted the pair."}
    return {
        "ok": False,
        "reason": code or f"http_{status}",
        "error": str(body.get("error_description") or body)[:200],
        "provider_error": code,
    }


def credentials_report() -> list[dict]:
    """Which connectors have credentials configured, for the Connectors page.

    Reports *presence*, never values - the UI needs to say "set this env var",
    and it must not be able to read the secret while doing it.
    """
    registry = get_registry()
    out: list[dict] = []
    for connector_id in registry.ids():
        descriptor = registry.get(connector_id).descriptor
        creds = client_credentials(connector_id)
        needs_env = bool(getattr(descriptor, "authorize_url", "")) and not creds.usable
        out.append({
            "connector_id": connector_id,
            "configured": creds.usable,
            "public_client": creds.is_public_client,
            "needs_env": needs_env,
            "env_vars": (
                [f"OMNI_OAUTH_{connector_id.upper()}_CLIENT_ID"]
                + ([f"OMNI_OAUTH_{connector_id.upper()}_CLIENT_SECRET"]
                   if not creds.is_public_client else [])
            ) if needs_env else [],
        })
    return out
