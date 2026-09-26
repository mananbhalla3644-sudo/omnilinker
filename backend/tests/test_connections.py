"""OAuth connection lifecycle (blueprint 6.2 / 7.1).

These tests cannot reach a real provider - there are no credentials and no
network in CI. What they *can* do is exercise everything that is ours, which is
where the security properties live, using a fake provider registered through the
same gate as a real one.

The properties asserted here are the ones a real handshake would not reveal:
the verifier never leaves the server, a state is single-use and expiring, a token
is sealed on disk, and no endpoint returns a token.
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from omnilinker.connectors import base_rest
from omnilinker.connectors.connections import (
    CONNECTIONS,
    PENDING,
    ConnectionError_,
    _connection_id,
    begin_authorization,
    client_credentials,
    complete_authorization,
    describe,
    disconnect,
    list_connections,
    redirect_uri,
)
from omnilinker.connectors.contract import (
    AuthorizationRequest,
    AuthorizationResult,
    Connector,
    ConnectorDescriptor,
    PullPage,
    RawEnvelope,
    Scope,
    StreamDescriptor,
    TokenGrant,
)
from omnilinker.connectors.registry import get_registry, register_builtins, registry
from omnilinker.crypto.envelope import is_envelope
from omnilinker.store.base import Query

REDIRECT = "https://provider.test/oauth/authorize"


class FakeOAuthConnector(Connector):
    """A provider that completes a handshake without a network call.

    Registered through the real registry, so it has to pass the same five
    static checks as everything else - a test double that bypasses the gate
    would be testing nothing.
    """

    descriptor = ConnectorDescriptor(
        id="fakeoauth",
        display_name="Fake OAuth provider",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url=REDIRECT,
        token_url="https://provider.test/oauth/token",
        base_url="https://api.provider.test",
        docs_url="https://provider.test/docs",
        scopes=(
            Scope("read", "Read the user's documents", "sensitive"),
            Scope("metadata", "Read file names and sizes", "normal"),
        ),
        content_kinds=("message",),
        # The registry gate (check C3) refused this double on the first attempt
        # for exactly the reason it would refuse a real connector: an
        # incremental connector that names no cursor field. Worth keeping the
        # double honest - a test double that bypasses the gate tests nothing.
        incremental_cursor=True,
        cursor_field="page_token",
        rate_limit_per_sec=10.0,
        rate_limit_burst=5,
        backfill_strategy="reconcile",
        notes="Test double. Deletes the authorization code so replay is observable.",
    )

    issued: list[dict] = []
    revoked: list[str] = []

    def authorize(self, req: AuthorizationRequest) -> AuthorizationResult:  # pragma: no cover
        raise AssertionError("the framework must build the URL, not the connector")

    def exchange_code(self, code: str, code_verifier: str,
                      redirect_uri: str) -> TokenGrant:
        FakeOAuthConnector.issued.append({
            "code": code, "verifier": code_verifier, "redirect_uri": redirect_uri,
        })
        if code == "explode":
            raise RuntimeError("token endpoint returned 502")
        if code == "stale":
            # A 200 with no token: looks like success, is not.
            return TokenGrant(access_token="", refresh_token=None)
        return TokenGrant(
            access_token="access-secret-value",
            refresh_token="refresh-secret-value",
            expires_at=time.time() + 3600,
            scopes=("read", "metadata"),
            token_type="Bearer",
            extra={"workspace_id": "T123"},
        )

    def revoke(self, grant: TokenGrant) -> None:
        FakeOAuthConnector.revoked.append(grant.access_token)

    def discover_streams(self, grant: TokenGrant | None = None,
                         user_ref: str | None = None) -> list[StreamDescriptor]:
        if grant is None:
            raise ConnectionError_("not connected")
        return [StreamDescriptor(key="s1", kind="mailbox", label="Inbox", meta={})]

    def pull(self, grant, stream, cursor, page=0) -> PullPage:  # pragma: no cover
        return PullPage(records=[], next_cursor=None, has_more=False)

    def normalize(self, raw: RawEnvelope) -> list:  # pragma: no cover
        return []


def get_registry_scopes(connector_id: str):
    return get_registry().get(connector_id).descriptor.scopes


@pytest.fixture
def fake_provider(monkeypatch):
    """Register the double and give it credentials."""
    FakeOAuthConnector.issued = []
    FakeOAuthConnector.revoked = []
    register_builtins()
    if not registry.has("fakeoauth"):
        registry.register("fakeoauth", FakeOAuthConnector)
    monkeypatch.setenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_ID", "client-abc")
    monkeypatch.setenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_SECRET", "secret-xyz")
    return FakeOAuthConnector


class TestAuthorizationUrl:
    def test_url_carries_pkce_and_state(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        query = parse_qs(urlparse(result.authorization_url).query)
        assert query["client_id"] == ["client-abc"]
        assert query["response_type"] == ["code"]
        assert query["code_challenge_method"] == ["S256"]
        assert query["code_challenge"][0]
        assert query["state"] == [result.state]
        assert query["scope"] == ["read metadata"]

    def test_challenge_is_the_sha256_of_the_stored_verifier(
        self, fake_provider
    ) -> None:
        """If the challenge does not match the stored verifier, the provider
        rejects the exchange and the flow fails at the last step with an opaque
        400. Asserting the pair is the only way to catch it early."""
        import base64
        import hashlib

        result = begin_authorization("fakeoauth", workspace="ws_test")
        stored = fake_provider and None
        from omnilinker.store import get_stores

        pending = get_stores().docs.get(PENDING, f"pend_{result.state}")
        verifier = pending["code_verifier"]
        challenge = parse_qs(urlparse(result.authorization_url).query)["code_challenge"][0]
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).decode().rstrip("=")
        assert challenge == expected
        assert stored is None

    def test_verifier_never_appears_in_the_url(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        from omnilinker.store import get_stores

        pending = get_stores().docs.get(PENDING, f"pend_{result.state}")
        assert pending["code_verifier"] not in result.authorization_url

    def test_redirect_uri_is_built_from_one_config_value(self, fake_provider) -> None:
        """Derived from OMNI_PUBLIC_URL, not assembled per connector, so moving
        the app is one config change rather than a hunt. Asserted against the
        configured value so a port change does not silently break it."""
        from omnilinker.config import get_settings

        expected = f"{get_settings().public_url.rstrip('/')}" \
                   "/api/connections/fakeoauth/callback"
        assert redirect_uri("fakeoauth") == expected
        assert "8000" not in expected, \
            "the default public URL must not collide with a model API port"

    def test_space_separated_scopes_in_the_url(self, fake_provider) -> None:
        """Slack, Google and Discord want space-separated scopes; Notion and
        Evernote want commas. Sending the wrong one is ignored by some providers
        and hard-rejected by others, so the joiner is per-provider."""
        result = begin_authorization("fakeoauth", workspace="ws_test")
        scope = parse_qs(urlparse(result.authorization_url).query)["scope"][0]
        assert scope == "read metadata", "space-separated providers must not get commas"

    @pytest.mark.parametrize("connector_id,joiner", [
        ("notion", ","), ("evernote", ","), ("slack", " "), ("gmail", " "),
    ])
    def test_every_oauth_provider_gets_its_own_scope_joiner(
        self, monkeypatch, connector_id: str, joiner: str
    ) -> None:
        monkeypatch.setenv(f"OMNI_OAUTH_{connector_id.upper()}_CLIENT_ID", "id-for-test")
        result = begin_authorization(connector_id, workspace="ws_test")
        scope = parse_qs(urlparse(result.authorization_url).query)["scope"][0]
        names = [s.name for s in get_registry().get(connector_id).descriptor.scopes
                 if s.name]
        assert scope == joiner.join(names)
        if joiner == ",":
            assert "," in scope or len(names) == 1

class TestAuthorizationFailures:
    def test_missing_credentials_say_which_variable(
        self, fake_provider, monkeypatch
    ) -> None:
        """The error is the UI, so it has to name the fix."""
        monkeypatch.delenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_SECRET", raising=False)
        with pytest.raises(ConnectionError_, match="OMNI_OAUTH_FAKEOAUTH_CLIENT_ID"):
            begin_authorization("fakeoauth", workspace="ws_test")

    def test_non_oauth_connector_is_refused_clearly(self) -> None:
        with pytest.raises(ConnectionError_, match="does not use an OAuth redirect"):
            begin_authorization("whatsapp", workspace="ws_test")

    def test_unknown_connector(self) -> None:
        with pytest.raises(ConnectionError_, match="unknown connector"):
            begin_authorization("nope", workspace="ws_test")


class TestCallback:
    def test_happy_path_stores_a_sealed_grant(self, fake_provider) -> None:
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        status = complete_authorization("fakeoauth", workspace="ws_test",
                                       code="the-code", state=result.state)
        assert status["connected"] is True

        doc = get_stores().docs.get(CONNECTIONS, _connection_id("fakeoauth", "ws_test"))
        assert is_envelope(doc["access_token"]), "access token stored in the clear"
        assert is_envelope(doc["refresh_token"]), "refresh token stored in the clear"
        assert doc["access_token_present"] is True
        assert doc["refresh_token_present"] is True
        # ...and the ciphertext is not the plaintext.
        assert "access-secret-value" not in str(doc)
        assert "refresh-secret-value" not in str(doc)

    def test_the_verifier_is_passed_to_the_exchange(self, fake_provider) -> None:
        """This is the whole point of keeping it server-side."""
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        pending = get_stores().docs.get(PENDING, f"pend_{result.state}")
        verifier = pending["code_verifier"]
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="the-code", state=result.state)
        assert FakeOAuthConnector.issued[-1]["verifier"] == verifier
        assert FakeOAuthConnector.issued[-1]["code"] == "the-code"
        assert FakeOAuthConnector.issued[-1]["redirect_uri"] == pending["redirect_uri"]

    def test_state_is_single_use(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="the-code", state=result.state)
        # A replayed callback must fail, even with a fresh code.
        with pytest.raises(ConnectionError_, match="expired or was already used"):
            complete_authorization("fakeoauth", workspace="ws_test",
                                   code="replayed", state=result.state)
        assert len(FakeOAuthConnector.issued) == 1, "a replay reached the token endpoint"

    def test_state_is_consumed_before_the_network_call(self, fake_provider) -> None:
        """If the exchange fails, the verifier is not put back - a retry gets a
        new flow, which is the correct behaviour for a credential exchange."""
        result = begin_authorization("fakeoauth", workspace="ws_test")
        with pytest.raises(RuntimeError, match="502"):
            complete_authorization("fakeoauth", workspace="ws_test",
                                   code="explode", state=result.state)
        from omnilinker.store import get_stores

        assert get_stores().docs.get(PENDING, f"pend_{result.state}") is None

    def test_expired_state_is_rejected(self, fake_provider) -> None:
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        key = f"pend_{result.state}"
        row = get_stores().docs.get(PENDING, key)
        row["expires_at"] = time.time() - 1
        get_stores().docs.put(PENDING, key, row)
        with pytest.raises(ConnectionError_, match="expired"):
            complete_authorization("fakeoauth", workspace="ws_test",
                                   code="c", state=result.state)

    def test_state_from_another_workspace_is_refused(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_a")
        with pytest.raises(ConnectionError_, match="different workspace"):
            complete_authorization("fakeoauth", workspace="ws_b",
                                   code="c", state=result.state)

    def test_state_for_another_connector_is_refused(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        with pytest.raises(ConnectionError_, match="was started for"):
            complete_authorization("gmail", workspace="ws_test",
                                   code="c", state=result.state)

    def test_missing_state_is_refused(self, fake_provider) -> None:
        begin_authorization("fakeoauth", workspace="ws_test")
        with pytest.raises(ConnectionError_, match="did not return a state"):
            complete_authorization("fakeoauth", workspace="ws_test", code="c", state="")

    def test_provider_error_is_surfaced(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        with pytest.raises(ConnectionError_, match="access_denied"):
            complete_authorization("fakeoauth", workspace="ws_test", code="",
                                   state=result.state, error="access_denied")

    def test_empty_token_marks_the_connection_unusable(self, fake_provider) -> None:
        """A 200 from the token endpoint that carries no access token must not
        present as "connected" - otherwise the user finds out at their first
        sync, from the provider, with no explanation."""
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        status = complete_authorization("fakeoauth", workspace="ws_test",
                                       code="stale", state=result.state)
        assert status["connected"] is False
        assert status["status"] == "failed"
        row = get_stores().docs.get(CONNECTIONS, _connection_id("fakeoauth", "ws_test"))
        assert row["status"] == "failed"
        assert "no access token" in row["status_detail"]


class TestTokenHandling:
    def test_grant_round_trips_through_storage(self, fake_provider) -> None:
        from omnilinker.connectors.connections import get_grant

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        grant = get_grant("fakeoauth", "ws_test")
        assert grant is not None
        assert grant.access_token == "access-secret-value"
        assert grant.refresh_token == "refresh-secret-value"
        assert "workspace_id" in grant.extra

    def test_undecryptable_credential_reads_as_disconnected(self, fake_provider) -> None:
        """A credential we cannot decrypt is worse than none: it looks connected
        and fails at the first API call."""
        from omnilinker.connectors.connections import get_grant
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        key = _connection_id("fakeoauth", "ws_test")
        doc = get_stores().docs.get(CONNECTIONS, key)
        doc["access_token"] = {**doc["access_token"], "ct": "AAAA"}
        get_stores().docs.put(CONNECTIONS, key, doc)
        assert get_grant("fakeoauth", "ws_test") is None
        assert describe("fakeoauth", "ws_test")["connected"] is False

    def test_refresh_happens_when_expiring(self, fake_provider, monkeypatch) -> None:
        from omnilinker.connectors.connections import get_grant
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        key = _connection_id("fakeoauth", "ws_test")
        doc = get_stores().docs.get(CONNECTIONS, key)
        doc["expires_at"] = time.time() + 10  # inside the 120s skew
        get_stores().docs.put(CONNECTIONS, key, doc)

        calls: list[TokenGrant] = []

        def fake_refresh(self, grant: TokenGrant) -> TokenGrant:
            calls.append(grant)
            return TokenGrant(access_token="fresh-access",
                              refresh_token=grant.refresh_token,
                              expires_at=time.time() + 3600)

        monkeypatch.setattr(FakeOAuthConnector, "refresh_tokens", fake_refresh)
        grant = get_grant("fakeoauth", "ws_test")
        assert len(calls) == 1
        assert calls[0].refresh_token == "refresh-secret-value"
        assert grant.access_token == "fresh-access"
        # ...and the new token is sealed too.
        stored = get_stores().docs.get(CONNECTIONS, key)
        assert is_envelope(stored["access_token"])
        assert "fresh-access" not in str(stored)

    def test_refresh_failure_marks_the_row(self, fake_provider, monkeypatch) -> None:
        from omnilinker.connectors.connections import get_grant

        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        key = _connection_id("fakeoauth", "ws_test")
        doc = get_stores().docs.get(CONNECTIONS, key)
        doc["expires_at"] = time.time() + 10  # inside the refresh skew
        get_stores().docs.put(CONNECTIONS, key, doc)

        def boom(self, grant):
            raise RuntimeError("invalid_grant")

        monkeypatch.setattr(FakeOAuthConnector, "refresh_tokens", boom)
        # The token is still returned - a failed refresh should degrade, not
        # throw, so an in-flight sync can finish on the last valid token.
        assert get_grant("fakeoauth", "ws_test") is not None
        status = describe("fakeoauth", "ws_test")
        assert status["status"] == "refresh_failed"
        assert "invalid_grant" in status["status_detail"]
        assert "reconnect" in status["next_step"].lower()

    def test_a_fresh_grant_is_not_refreshed(self, fake_provider, monkeypatch) -> None:
        from omnilinker.connectors.connections import get_grant

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        monkeypatch.setattr(
            FakeOAuthConnector, "refresh_tokens",
            lambda self, g: pytest.fail("a valid token was refreshed"),
        )
        assert get_grant("fakeoauth", "ws_test") is not None


class TestDisconnect:
    def test_disconnect_revokes_remotely_then_deletes(self, fake_provider) -> None:
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        outcome = disconnect("fakeoauth", "ws_test")
        assert outcome["disconnected"] is True
        assert outcome["revoked_remotely"] is True
        assert FakeOAuthConnector.revoked == ["access-secret-value"]
        assert get_stores().docs.get(CONNECTIONS, _connection_id("fakeoauth", "ws_test")) is None

    def test_local_delete_happens_even_if_revocation_fails(self, fake_provider,
                                                           monkeypatch) -> None:
        """Leaving a live credential on disk after the user asked to remove it is
        the worse failure, so the delete is unconditional."""
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        monkeypatch.setattr(FakeOAuthConnector, "revoke",
                            lambda self, g: (_ for _ in ()).throw(RuntimeError("boom")))
        outcome = disconnect("fakeoauth", "ws_test")
        assert outcome["disconnected"] is True
        assert outcome["revoked_remotely"] is False
        assert outcome["revoke_error"]
        assert "revoke" in outcome["note"].lower()
        assert get_stores().docs.get(CONNECTIONS, _connection_id("fakeoauth", "ws_test")) is None

    def test_disconnect_when_not_connected(self, fake_provider) -> None:
        assert disconnect("fakeoauth", "ws_test")["disconnected"] is False


class TestNoTokenInAnyStatus:
    def test_describe_never_includes_token_material(self, fake_provider) -> None:
        result = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test",
                               code="c", state=result.state)
        blob = repr(describe("fakeoauth", "ws_test"))
        assert "access-secret-value" not in blob
        assert "refresh-secret-value" not in blob
        assert "access_token\"" not in blob, "a field named access_token leaked"
        assert describe("fakeoauth", "ws_test")["access_token_present"] is True

    def test_list_connections_covers_every_connector(self, fake_provider) -> None:
        rows = list_connections("ws_test")
        assert len(rows) >= 10
        assert all({"connector_id", "connected", "next_step", "configured"} <= set(r)
                   for r in rows)

    def test_credentials_report_gives_names_not_values(self, fake_provider) -> None:
        from omnilinker.connectors.connections import credentials_report

        report = {r["connector_id"]: r for r in credentials_report()}
        assert report["fakeoauth"]["configured"] is True
        assert report["fakeoauth"]["needs_env"] is False
        # An unconfigured provider is told which variables to set.
        slack = report["slack"]
        assert slack["needs_env"] is True
        assert "OMNI_OAUTH_SLACK_CLIENT_ID" in slack["env_vars"]


class TestCredentialProbeRefusesToGuess:
    """The probe exists to settle the Client ID / Client Secret / Verification
    Token ambiguity, which a provider console makes unavoidable.

    Its dangerous failure mode is a false positive: reporting "your secret is
    fine" for a value that is not, because the user then stops looking. Every
    test here is about the probe declining to answer when it cannot.
    """

    @staticmethod
    def _stub_token_endpoint(monkeypatch, reply):
        """Make the token endpoint answer with a fixed error code.

        `reply` maps a client_secret to the provider's error string, so a test
        can model a provider that discriminates or one that does not.
        """
        import httpx

        def fake_post(url, data=None, timeout=None, **kwargs):
            secret = (data or {}).get("client_secret", "")
            code = reply(secret)
            return httpx.Response(200, json={"error": code})

        monkeypatch.setattr(httpx, "post", fake_post)

    def test_control_mismatch_lets_a_real_answer_through(
        self, fake_provider, monkeypatch
    ) -> None:
        """A provider that distinguishes a good secret from a bad one works."""
        from omnilinker.connectors.connections import diagnose_credentials

        self._stub_token_endpoint(
            monkeypatch,
            lambda s: "invalid_code" if s == "secret-xyz" else "invalid_client",
        )
        result = diagnose_credentials("fakeoauth", "secret-xyz")
        assert result["ok"] is True
        assert result["reason"] == "credentials_accepted"

    def test_control_match_forces_inconclusive(self, fake_provider, monkeypatch) -> None:
        """The same answer for a known-bad secret means no answer at all.

        This is Slack's real behaviour: `oauth.v2.access` checks the code before
        the client credentials, so a correct secret and a nonsense one both come
        back `invalid_code`. The probe must say `inconclusive`, not "accepted".
        """
        from omnilinker.connectors.connections import diagnose_credentials

        self._stub_token_endpoint(monkeypatch, lambda s: "invalid_code")
        result = diagnose_credentials("fakeoauth", "secret-xyz")
        assert result["ok"] is False
        assert result["reason"] == "inconclusive"
        assert "cannot validate" in result["error"]
        assert "Connect" in result["hint"]

    def test_wrong_secret_is_reported_as_invalid_client(
        self, fake_provider, monkeypatch
    ) -> None:
        from omnilinker.connectors.connections import diagnose_credentials

        self._stub_token_endpoint(
            monkeypatch, lambda s: "invalid_client" if s != "secret-xyz" else "ok"
        )
        result = diagnose_credentials("fakeoauth", "verification-token-by-mistake")
        assert result["ok"] is False
        assert result["reason"] == "invalid_client"
        assert "Client Secret" in result["hint"]

    def test_candidate_secret_is_never_persisted(
        self, fake_provider, monkeypatch
    ) -> None:
        """Probing a secret must not turn into storing it.

        Checked against a workspace that already holds a real connection, so the
        assertion is about the probe not adding to stored state rather than
        about the collection happening to be empty.
        """
        from omnilinker.connectors.connections import (
            begin_authorization,
            complete_authorization,
            diagnose_credentials,
        )
        from omnilinker.store import get_stores

        pending = begin_authorization("fakeoauth", workspace="ws_test")
        complete_authorization("fakeoauth", workspace="ws_test", code="c",
                               state=pending.state)

        self._stub_token_endpoint(monkeypatch, lambda s: "invalid_code")
        diagnose_credentials("fakeoauth", "a-very-distinctive-candidate-value")

        rows = get_stores().docs.find(Query(collection=CONNECTIONS))
        assert rows, "expected the completed connection to be stored"
        assert "a-very-distinctive-candidate-value" not in repr(rows)

    def test_probe_needs_a_configured_client_id(self, fake_provider, monkeypatch) -> None:
        from omnilinker.connectors.connections import diagnose_credentials

        monkeypatch.delenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_ID", raising=False)
        result = diagnose_credentials("fakeoauth", "secret-xyz")
        assert result["ok"] is False
        assert "client id" in result["error"]

    def test_network_failure_is_reported_not_swallowed(
        self, fake_provider, monkeypatch
    ) -> None:
        import httpx

        def boom(*args, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "post", boom)
        from omnilinker.connectors.connections import diagnose_credentials

        result = diagnose_credentials("fakeoauth", "secret-xyz")
        assert result["ok"] is False
        assert result["reason"] == "network"
        assert "ConnectError" in result["error"]


class TestHandshakeUsesTheSameCredentialsInBothSteps:
    """Regression cover for a bug that made Connect impossible while looking
    like a credentials problem.

    `authorize` resolved the client id through `connections.client_credentials`
    (`OMNI_OAUTH_<PROVIDER>_CLIENT_ID`) but `exchange_code` and
    `refresh_tokens` read `{PROVIDER}_CLIENT_ID` and fell back to the literal
    string `"demo-client-id"`. So the app reported itself correctly configured,
    the browser was sent to the provider with the real client id, and then the
    one step that actually authenticates the client sent a placeholder and got
    `invalid_client` - which reads exactly like "your secret is wrong".
    """

    def test_exchange_sends_the_configured_credentials(
        self, fake_provider, monkeypatch
    ) -> None:
        import httpx
        from omnilinker.connectors.base_rest import BaseRestConnector

        monkeypatch.setenv("OMNI_OAUTH_SLACK_CLIENT_ID", "slack-client-real")
        monkeypatch.setenv("OMNI_OAUTH_SLACK_CLIENT_SECRET", "slack-secret-real")
        sent: dict = {}

        # `httpx.Client.post` is patched as an unbound method, so `self` arrives
        # as the first positional argument and has to be absorbed here.
        def capture(self, url, data=None, **kwargs):
            sent.update(data or {})
            return httpx.Response(200, json={
                "access_token": "at", "refresh_token": "rt", "token_type": "Bearer",
            })

        monkeypatch.setattr(httpx.Client, "post", capture)
        connector = get_registry().get("slack")
        assert isinstance(connector, BaseRestConnector), "slack must be a rest connector"
        connector.exchange_code("the-code", "the-verifier", "http://127.0.0.1/cb")

        assert sent["client_id"] == "slack-client-real"
        assert sent["client_secret"] == "slack-secret-real"
        assert "demo-client-id" not in sent["client_id"]
        assert "demo-secret" not in sent["client_secret"]

    def test_omni_oauth_prefix_is_the_only_convention_that_counts(
        self, fake_provider, monkeypatch
    ) -> None:
        """The old, unprefixed variable must not be picked up.

        Otherwise a stale `SLACK_CLIENT_ID` in a shell would silently shadow the
        value the app actually uses everywhere else.
        """
        import httpx
        from omnilinker.connectors.base_rest import BaseRestConnector

        monkeypatch.setenv("OMNI_OAUTH_SLACK_CLIENT_ID", "slack-client-real")
        monkeypatch.setenv("SLACK_CLIENT_ID", "wrong-unprefixed-value")
        sent: dict = {}

        def capture(self, url, data=None, **kwargs):
            sent.update(data or {})
            return httpx.Response(200, json={"access_token": "at", "token_type": "Bearer"})

        monkeypatch.setattr(httpx.Client, "post", capture)
        connector = get_registry().get("slack")
        assert isinstance(connector, BaseRestConnector)
        connector.exchange_code("c", "v", "http://127.0.0.1/cb")
        assert sent["client_id"] == "slack-client-real"

    def test_authorize_and_exchange_agree_on_the_client(
        self, fake_provider, monkeypatch
    ) -> None:
        """The two halves of a handshake must not disagree about the client."""
        import httpx
        from omnilinker.connectors.base_rest import BaseRestConnector
        from urllib.parse import parse_qs, urlparse

        monkeypatch.setenv("OMNI_OAUTH_SLACK_CLIENT_ID", "slack-client-real")
        monkeypatch.setenv("OMNI_OAUTH_SLACK_CLIENT_SECRET", "slack-secret-real")
        sent: dict = {}

        def capture(self, url, data=None, **kwargs):
            sent.update(data or {})
            return httpx.Response(200, json={"access_token": "at", "token_type": "Bearer"})

        monkeypatch.setattr(httpx.Client, "post", capture)
        connector = get_registry().get("slack")
        assert isinstance(connector, BaseRestConnector)

        pending = begin_authorization("slack", workspace="ws_test")
        connector.exchange_code("c", "v", "http://127.0.0.1/cb")
        from_id = parse_qs(urlparse(pending.authorization_url).query)["client_id"][0]
        assert from_id == "slack-client-real"
        assert from_id == sent["client_id"]


class TestExchangeFailureExplainsItself:
    """A raw `invalid_client` sends people hunting the wrong credential."""

    @staticmethod
    def _descriptor():
        return get_registry().get("fakeoauth").descriptor

    def _explain(self, payload: str) -> str:
        from omnilinker.connectors.base_rest import _explain_exchange_failure

        return _explain_exchange_failure(self._descriptor(), payload)

    def test_invalid_client_names_the_verification_token_mixup(self) -> None:
        text = self._explain('{"error":"invalid_client"}')
        assert "Client Secret" in text
        assert "Verification Token" in text
        assert "OMNI_OAUTH_FAKEOAUTH_CLIENT_SECRET" in text

    def test_redirect_uri_mismatch_names_the_exact_cause(self) -> None:
        text = self._explain('{"error":"redirect_uri_mismatch"}')
        assert "redirect URI" in text
        assert "registered" in text

    def test_invalid_scope_points_at_the_app_console(self) -> None:
        text = self._explain('{"error":"invalid_scope","error_description":"missing users:read"}')
        assert "scopes" in text
        assert "users:read" in text

    def test_invalid_code_says_start_over(self) -> None:
        """A replayed callback should not read as a credential problem."""
        text = self._explain('{"error":"invalid_code"}')
        assert "single-use" in text
        assert "Verification Token" not in text

    def test_access_denied_is_reported_as_a_decline(self) -> None:
        assert "declined" in self._explain('{"error":"access_denied"}')

    def test_non_json_body_is_passed_through(self) -> None:
        assert "gateway" in self._explain("502 Bad Gateway").lower()

    def test_unknown_error_keeps_the_description(self) -> None:
        text = self._explain('{"error":"server_error","error_description":"try later"}')
        assert "server_error" in text
        assert "try later" in text


class TestProviderEndpointShapes:
    """A wrong `authorize_url` is invisible to every other test.

    The app reports the connector as correctly configured, the Connect button
    opens a window, and the provider answers with an error page that never comes
    back to us - so there is no exception, no failed assertion, and nothing in
    our own logs. Slack's authorize URL was
    `https://slack.com/api/oauth.v2/authorize`, which is wrong twice over: the
    browser-facing consent page is not under `/api`, and the method is
    `oauth/v2/authorize`, not `oauth.v2/authorize`. Slack answers that with an
    opaque "There's been a glitch…" page, and Connect simply did nothing.

    These assertions encode each provider's *documented* URL shape, which is
    checkable offline. `tests/check_endpoints_live.py` confirms them against the
    real providers; this keeps the suite honest between runs.
    """

    def test_authorize_urls_match_each_provider(self) -> None:
        from omnilinker.connectors.registry import get_registry, register_builtins

        register_builtins()
        expected = {
            "slack": "https://slack.com/oauth/v2/authorize",
            "discord": "https://discord.com/oauth2/authorize",
            "notion": "https://api.notion.com/v1/oauth/authorize",
            "gdrive": "https://accounts.google.com/o/oauth2/v2/auth",
            "gmail": "https://accounts.google.com/o/oauth2/v2/auth",
            "onedrive": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "dropbox": "https://www.dropbox.com/oauth2/authorize",
            "youtube": "https://accounts.google.com/o/oauth2/v2/auth",
        }
        registry = get_registry()
        for connector_id, url in expected.items():
            assert registry.get(connector_id).descriptor.authorize_url == url, (
                f"{connector_id} authorize_url is wrong; a bad consent URL makes "
                f"the Connect button dead with no error anywhere"
            )

    def test_slack_consent_page_is_not_a_web_api_method(self) -> None:
        """The specific regression, stated as a rule.

        Slack's consent page is `slack.com/oauth/v2/authorize`; only the token
        exchange is a Web API method at `slack.com/api/oauth.v2.access`. Mixing
        the two produces a URL that resolves and then fails.
        """
        from omnilinker.connectors.registry import get_registry, register_builtins

        register_builtins()
        descriptor = get_registry().get("slack").descriptor
        assert not descriptor.authorize_url.startswith("https://slack.com/api/"), (
            "Slack's authorize page is not a Web API method and is not under /api"
        )
        assert "/api/oauth.v2.access" in descriptor.token_url

    def test_every_oauth_connector_has_both_endpoints(self) -> None:
        from omnilinker.connectors.registry import get_registry, register_builtins

        register_builtins()
        registry = get_registry()
        for connector_id in registry.ids():
            descriptor = registry.get(connector_id).descriptor
            if descriptor.auth_flow != "oauth2_code":
                continue
            assert descriptor.authorize_url, f"{connector_id} has no authorize_url"
            assert descriptor.token_url, f"{connector_id} has no token_url"
            assert descriptor.authorize_url.startswith("https://"), (
                f"{connector_id} authorize_url is not https"
            )

    def test_authorize_and_token_use_different_hosts_or_paths(self) -> None:
        """Catches the class of bug where one was copied from the other.

        Not a rule about correctness - Google legitimately uses two hosts - but
        a consent page and a token endpoint being byte-identical is almost always
        one field pointing at the other's value.
        """
        from omnilinker.connectors.registry import get_registry, register_builtins

        register_builtins()
        registry = get_registry()
        for connector_id in registry.ids():
            descriptor = registry.get(connector_id).descriptor
            if descriptor.auth_flow != "oauth2_code":
                continue
            if descriptor.authorize_url == descriptor.token_url:
                # Evernote publishes one URL for both; that is legitimate.
                assert connector_id == "evernote", (
                    f"{connector_id} points authorize and token at the same URL"
                )


class TestEnvFileLoading:
    """`.env` shipped documented but unread; now it is read, and now that it is
    read the tests must not depend on it."""

    def test_env_file_populates_settings(self, tmp_path, monkeypatch) -> None:
        from omnilinker import config

        # OMNI_WORKSPACE is already in the environment - the autouse fixture sets
        # it - so it is used here to demonstrate the real-environment-wins rule
        # rather than the plain load.
        monkeypatch.delenv("OMNI_SKIP_ENV_FILE", raising=False)
        env = tmp_path / ".env"
        env.write_text("OMNI_INDEX_K=7\n# a comment\n\nOMNI_EMBED_MODEL=hash-x\n")
        loaded = config.load_env_file(env)
        assert loaded["OMNI_INDEX_K"] == "7"
        assert loaded["OMNI_EMBED_MODEL"] == "hash-x"
        assert os.environ["OMNI_INDEX_K"] == "7"
        assert "OMNI_WORKSPACE" not in loaded, "must not overwrite a real var"

    def test_real_environment_wins_over_the_file(self, tmp_path, monkeypatch) -> None:
        """`OMNI_PORT=9000 ./run.sh` must beat a checked-in `.env`."""
        from omnilinker import config

        monkeypatch.delenv("OMNI_SKIP_ENV_FILE", raising=False)
        monkeypatch.setenv("OMNI_INDEX_K", "3")
        env = tmp_path / ".env"
        env.write_text("OMNI_INDEX_K=99\n")
        assert config.load_env_file(env) == {}
        assert os.environ["OMNI_INDEX_K"] == "3"

    def test_quotes_are_stripped(self, tmp_path, monkeypatch) -> None:
        from omnilinker import config

        monkeypatch.delenv("OMNI_SKIP_ENV_FILE", raising=False)
        env = tmp_path / ".env"
        env.write_text('OMNI_MERGE_T="0.75"\nOMNI_SUGGEST_T=\'0.5\'\n')
        loaded = config.load_env_file(env)
        assert loaded["OMNI_MERGE_T"] == "0.75"
        assert loaded["OMNI_SUGGEST_T"] == "0.5"

    def test_reset_removes_only_what_it_added(self, tmp_path, monkeypatch) -> None:
        """`reset_env_file` must undo its own writes without touching keys that
        belonged to the caller, or it would silently delete a developer's real
        environment on the way out of a test run."""
        from omnilinker import config

        monkeypatch.delenv("OMNI_SKIP_ENV_FILE", raising=False)
        monkeypatch.setenv("OMNI_INDEX_K", "5")
        monkeypatch.delenv("OMNI_VECTOR", raising=False)
        env = tmp_path / ".env"
        env.write_text("OMNI_INDEX_K=1\nOMNI_VECTOR=1\nOMNI_EMBED_MODEL=x\n")
        config.load_env_file(env)
        assert os.environ["OMNI_VECTOR"] == "1"
        config.reset_env_file()
        assert "OMNI_VECTOR" not in os.environ
        assert os.environ["OMNI_INDEX_K"] == "5"

    def test_skip_flag_disables_loading_entirely(self, tmp_path, monkeypatch) -> None:
        from omnilinker import config

        monkeypatch.setenv("OMNI_SKIP_ENV_FILE", "1")
        monkeypatch.delenv("OMNI_VECTOR", raising=False)
        env = tmp_path / ".env"
        env.write_text("OMNI_VECTOR=1\n")
        assert config.load_env_file(env) == {}
        assert "OMNI_VECTOR" not in os.environ

    def test_missing_file_is_not_an_error(self, tmp_path, monkeypatch) -> None:
        from omnilinker import config

        monkeypatch.delenv("OMNI_SKIP_ENV_FILE", raising=False)
        assert config.load_env_file(tmp_path / "nope.env") == {}

    def test_suite_is_hermetic_regardless_of_a_local_env_file(self) -> None:
        """The guard that keeps a developer's real credentials out of the run."""
        assert os.environ.get("OMNI_SKIP_ENV_FILE") == "1"


class TestPendingPruning:
    def test_expired_pending_rows_are_pruned(self, fake_provider) -> None:
        from omnilinker.store import get_stores

        result = begin_authorization("fakeoauth", workspace="ws_test")
        from omnilinker.connectors.connections import begin_authorization as begin2

        result2 = begin2("fakeoauth", workspace="ws_test")
        key = f"pend_{result.state}"
        row = get_stores().docs.get(PENDING, key)
        row["expires_at"] = time.time() - 1
        get_stores().docs.put(PENDING, key, row)
        begin2("fakeoauth", workspace="ws_test")  # triggers the prune
        assert get_stores().docs.get(PENDING, key) is None


class TestApiSurface:
    @pytest.fixture
    def client(self, ingested, fake_provider):
        from fastapi.testclient import TestClient

        from omnilinker.api.app import create_app

        with TestClient(create_app()) as test_client:
            yield test_client

    def test_connections_endpoint(self, client) -> None:
        body = client.get("/api/connections").json()
        assert body["ok"] is True
        assert body["connections"]
        assert body["credentials"]

    def test_authorize_endpoint_returns_a_url(self, client) -> None:
        body = client.post("/api/connections/fakeoauth/authorize").json()
        assert body["ok"] is True
        assert body["authorization_url"].startswith(REDIRECT)
        assert body["state"]
        assert body["redirect_uri"].endswith("/api/connections/fakeoauth/callback")

    def test_authorize_endpoint_reports_missing_credentials(self, client,
                                                            monkeypatch) -> None:
        monkeypatch.delenv("OMNI_OAUTH_FAKEOAUTH_CLIENT_ID", raising=False)
        response = client.post("/api/connections/fakeoauth/authorize")
        assert response.status_code == 409
        assert "OMNI_OAUTH_FAKEOAUTH_CLIENT_ID" in str(response.json())

    def test_callback_returns_html_not_json(self, client) -> None:
        """This is a top-level browser navigation; a JSON body here renders as
        raw text in a popup and looks like a crash to whoever just approved."""
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        response = client.get(
            f"/api/connections/fakeoauth/callback?code=c&state={state}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "omni:connection" in response.text

    def test_callback_failure_renders_an_error_page(self, client) -> None:
        response = client.get("/api/connections/fakeoauth/callback?code=c&state=bogus")
        assert response.status_code == 400
        assert "Could not connect" in response.text

    def test_callback_writes_an_audit_entry(self, client) -> None:
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        audit = client.get("/api/audit").json()["entries"]
        assert any(a.get("action") == "oauth_callback" and a.get("ok") for a in audit)

    def test_verify_endpoint(self, client) -> None:
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        body = client.post("/api/connections/fakeoauth/verify").json()
        assert body["ok"] is True
        assert body["streams"] == 1

    def test_verify_reports_not_connected(self, client) -> None:
        response = client.post("/api/connections/fakeoauth/verify")
        assert response.status_code == 409
        assert response.json()["ok"] is False

    def test_sync_of_an_unconnected_oauth_connector_is_actionable(self, client) -> None:
        """Not a bare 500 from the provider: the response says what to do and
        where to do it."""
        response = client.post("/api/sync", json={"connector_id": "fakeoauth"})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["next_step"]
        assert detail["authorize"].endswith("/authorize")

    def test_sync_uses_the_stored_grant(self, client) -> None:
        """A connected provider must not be rejected for lacking credentials."""
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        # It reaches the connector and finds no streams to pull, rather than
        # 409-ing at the gate.
        response = client.post("/api/sync", json={"connector_id": "fakeoauth"})
        assert response.status_code == 200
        assert response.json()["ingest"]["status"] in ("ok", "partial")

    def test_disconnect_endpoint(self, client) -> None:
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        body = client.post("/api/connections/fakeoauth/disconnect").json()
        assert body["disconnected"] is True
        assert client.get("/api/connections/fakeoauth").json()["connection"]["connected"] is False

    def test_no_endpoint_returns_a_token(self, client) -> None:
        """Walks every connection endpoint. A status page that can return a
        credential is a credential exfiltration endpoint."""
        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        for path in ["/api/connections", "/api/connections/fakeoauth",
                     "/api/passport", "/api/system/info"]:
            blob = client.get(path).text
            assert "access-secret-value" not in blob, f"{path} leaked a token"
            assert "refresh-secret-value" not in blob, f"{path} leaked a refresh token"

    def test_sealed_token_survives_a_reload(self, client) -> None:
        """The credential is as durable as the content, and as unreadable."""
        from omnilinker.store import flush_stores, get_stores

        state = client.post("/api/connections/fakeoauth/authorize").json()["state"]
        client.get(f"/api/connections/fakeoauth/callback?code=c&state={state}")
        flush_stores()
        key = _connection_id("fakeoauth", "ws_test")
        assert (get_stores().data_root / f"{CONNECTIONS}.json").exists()
        blob = (get_stores().data_root / f"{CONNECTIONS}.json").read_text("utf-8")
        assert "refresh-secret-value" not in blob
        assert "AES-256-GCM" in blob
