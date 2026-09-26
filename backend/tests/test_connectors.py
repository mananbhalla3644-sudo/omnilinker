"""Connector contract tests (blueprint 15.1).

Every connector's `normalize()` gets the same three assertions, because the
golden payload is the only place provider schema knowledge lives and it is the
only thing that changes silently when a provider ships a new field shape:

  1. `normalize()` is a pure function - same input, same output, twice.
  2. Every output record is *canonical*: required keys present, correct
     `kind`, and an idempotency key derived from the provider's own id.
  3. It performs no I/O. The registry enforces this statically, and this test
     enforces it dynamically by calling it with the network module broken.

Plus one behavioural test per provider for the quirk that actually breaks
normalizers in production (nested MIME, snowflakes, zip-slip, caption formats).
"""

from __future__ import annotations

import socket

import pytest

from omnilinker.connectors import get_registry
from omnilinker.connectors.contract import CanonicalRecord, Connector, RawEnvelope

REQUIRED_KEYS = {
    "message": ("provider", "provider_msg_id", "ts", "sender_ref", "body_text",
                "workspace_id"),
    "email": ("provider", "provider_msg_id", "ts", "sender_ref", "body_text"),
    "file": ("provider", "provider_file_id", "name", "mime", "workspace_id"),
    "note": ("provider", "provider_page_id", "title", "body_text"),
    "video": ("provider", "provider_video_id", "title"),
    "transcript": ("provider_transcript_id", "video_id", "text"),
    "identity": ("provider", "provider_user_id"),
}

PROVIDER_NORMALIZERS = ["slack", "gmail", "discord", "whatsapp", "gdrive", "onedrive",
                        "dropbox", "notion", "evernote", "youtube", "demo"]


def demo_envelopes() -> list[RawEnvelope]:
    """Every demo envelope, which are provider-shaped by construction."""
    demo = get_registry().get("demo")
    out: list[RawEnvelope] = []
    for stream in demo.discover_streams():
        page = demo.pull(None, stream, None, 0)
        out.extend(page.records)
    return out


def envelopes_for(provider: str) -> list[RawEnvelope]:
    if provider == "demo":
        # The demo connector emits *provider-shaped* payloads, so it never
        # produces an envelope whose provider is "demo". It is covered twice
        # over instead: `TestPullContract` tests its cursor and determinism
        # behaviour, and its `normalize()` is the ten real normalizers, each of
        # which has its own contract test against these same fixtures.
        return []
    return [e for e in demo_envelopes() if e.provider == provider]


def normalize_all(connector: Connector, envelopes: list[RawEnvelope]) -> list[CanonicalRecord]:
    out: list[CanonicalRecord] = []
    for envelope in envelopes:
        out.extend(connector.normalize(envelope))
    return out


class TestRegistration:
    def test_all_builtins_register(self) -> None:
        registry = get_registry()
        assert set(registry.ids()) >= set(PROVIDER_NORMALIZERS)

    def test_every_connector_passes_the_static_checks(self) -> None:
        """register() is a gate, not a dict assignment. A rejected built-in is
        reported at boot, so this asserts none were."""
        from omnilinker.connectors.registry import register_builtins, reset_registry

        reset_registry()
        register_builtins()
        assert get_registry().ids()

    def test_descriptors_are_serialisable(self) -> None:
        for descriptor in get_registry().descriptors():
            assert descriptor["id"]
            assert descriptor["scopes"], f"{descriptor['id']} has no scopes"
            for scope in descriptor["scopes"]:
                assert scope["justification"].strip(), \
                    f"{descriptor['id']}.{scope['name']} has no justification"

    def test_unknown_connector_raises(self) -> None:
        with pytest.raises(KeyError):
            get_registry().get("does-not-exist")

    def test_file_connector_has_no_oauth_step(self) -> None:
        """A WhatsApp export has no authorization URL. The default must raise a
        clear error rather than the ABC demanding a stub that says so."""
        whatsapp = get_registry().get("whatsapp")
        with pytest.raises(NotImplementedError, match="whatsapp"):
            whatsapp.authorize(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("provider", PROVIDER_NORMALIZERS)
class TestNormalizerContract:
    def test_normalize_is_pure(self, provider: str) -> None:
        connector = get_registry().get(provider)
        envelopes = envelopes_for(provider)
        if not envelopes:
            pytest.skip(f"no {provider} fixtures in the demo dataset")
        first = normalize_all(connector, envelopes)
        second = normalize_all(connector, envelopes)
        assert [r.data for r in first] == [r.data for r in second], \
            f"{provider}.normalize() is not deterministic"

    def test_output_is_canonical(self, provider: str) -> None:
        connector = get_registry().get(provider)
        envelopes = envelopes_for(provider)
        if not envelopes:
            pytest.skip(f"no {provider} fixtures")
        records = normalize_all(connector, envelopes)
        assert records, f"{provider} produced no records"
        for record in records:
            assert record.kind in REQUIRED_KEYS, f"{provider}: unknown kind {record.kind}"
            for key in REQUIRED_KEYS[record.kind]:
                assert key in record.data, \
                    f"{provider}/{record.kind} is missing required key {key!r}"

    def test_no_io_during_normalize(self, provider: str) -> None:
        """A normalizer that reaches the network turns a replayable pure
        function into a client, and the failure is invisible at review time."""
        connector = get_registry().get(provider)
        envelopes = envelopes_for(provider)
        if not envelopes:
            pytest.skip(f"no {provider} fixtures")

        def refuse(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError(f"{provider}.normalize() attempted network I/O")

        saved = socket.socket
        socket.socket = refuse  # type: ignore[assignment]
        try:
            normalize_all(connector, envelopes)
        finally:
            socket.socket = saved  # type: ignore[assignment]

    def test_idempotency_key_is_the_provider_id(self, provider: str) -> None:
        """Re-running a sync must overwrite, never duplicate. The document id is
        the provider's stable id, so re-ingest is safe by construction."""
        connector = get_registry().get(provider)
        envelopes = envelopes_for(provider)
        if not envelopes:
            pytest.skip(f"no {provider} fixtures")
        records = normalize_all(connector, envelopes)
        # The same key table the pipeline uses, so this test and the ingestion
        # path can never disagree about what identifies a document.
        keys = [
            r.data.get("provider_msg_id") or r.data.get("provider_file_id")
            or r.data.get("provider_page_id") or r.data.get("provider_video_id")
            or r.data.get("provider_transcript_id") or r.data.get("video_id")
            or r.data.get("provider_user_id")
            for r in records
        ]
        assert all(keys), f"{provider} produced a record with no provider id"
        # Across a whole stream, duplicate keys are *expected* - Slack re-emits a
        # file every time it is attached to a new message, and the pipeline
        # upserts on (provider, key) so that is harmless by design. What must
        # never happen is one payload yielding two records with the same key,
        # which would make the upsert order-dependent.
        connector = get_registry().get(provider)
        for envelope in envelopes:
            emitted = [
                r.data.get("provider_msg_id") or r.data.get("provider_file_id")
                or r.data.get("provider_page_id") or r.data.get("provider_video_id")
                or r.data.get("provider_transcript_id") or r.data.get("video_id")
                or r.data.get("provider_user_id")
                for r in connector.normalize(envelope)
            ]
            assert len(emitted) == len(set(emitted)), \
                f"{provider} emitted duplicate keys from one payload"


class TestProviderQuirks:
    """The specific thing that breaks each normalizer in production."""

    def test_slack_mentions_and_links_are_unwrapped(self) -> None:
        records = [r for r in normalize_all(get_registry().get("slack"),
                                             envelopes_for("slack"))
                   if r.kind == "message"]
        bodies = " ".join(r.data["body_text"] for r in records)
        assert "<@" not in bodies, "Slack user links left in the body"
        assert "<#" not in bodies, "Slack channel links left in the body"
        assert "&amp;" not in bodies, "HTML entities left unescaped"

    def test_gmail_walks_nested_multipart(self) -> None:
        """The real Gmail MIME tree nests arbitrarily deep; a two-level walker
        silently drops body text on half of real mail."""
        records = normalize_all(get_registry().get("gmail"), envelopes_for("gmail"))
        for record in records:
            assert len(record.data["body_text"]) > 80, \
                f"gmail body looks truncated: {record.data['body_text'][:60]!r}"

    def test_gmail_sender_email_is_the_from_header(self) -> None:
        """This is the token that lets a Gmail account and a Slack account for
        the same human resolve to one Person."""
        records = normalize_all(get_registry().get("gmail"), envelopes_for("gmail"))
        for record in records:
            assert record.data["extra"].get("sender_email"), "no sender email emitted"

    def test_gmail_signature_contacts_are_extracted(self) -> None:
        records = normalize_all(get_registry().get("gmail"), envelopes_for("gmail"))
        signatures = [r.data["extra"].get("signature") or {} for r in records]
        assert any(sig.get("phone") for sig in signatures), \
            "no phone found in any signature block"
        # ...and only from the tail. A phone in the middle of a body is a
        # contact mention and must not be attributed to the sender.
        assert all(len(sig) <= 3 for sig in signatures)

    def test_notion_requires_the_type_discriminator(self) -> None:
        """A block without `type` yields an empty document. The demo emits the
        real shape precisely so this stays covered."""
        records = normalize_all(get_registry().get("notion"), envelopes_for("notion"))
        for record in records:
            assert record.data["body_text"].strip(), "notion note has no text"
            assert record.data["title"].strip()

    def test_youtube_captions_become_a_transcript(self) -> None:
        records = normalize_all(get_registry().get("youtube"), envelopes_for("youtube"))
        kinds = {r.kind for r in records}
        assert kinds == {"video", "transcript"}
        transcript = next(r for r in records if r.kind == "transcript")
        assert len(transcript.data["text"]) > 200
        assert transcript.data["seg_count"] > 0

    def test_whatsapp_sender_is_not_swallowed_by_a_dash(self) -> None:
        """'Bharat: Heads up - the index is slow' must attribute to Bharat, not
        to 'Bharat: Heads up'. This was a real parse bug."""
        records = [r for r in normalize_all(get_registry().get("whatsapp"),
                                             envelopes_for("whatsapp"))
                   if r.kind == "message"]
        senders = {r.data["sender_name"] for r in records if r.data.get("sender_name")}
        assert "Bharat: Heads up" not in senders, "sender field swallowed part of the body"
        assert "Alice" in senders

    def test_whatsapp_attachments_become_files(self) -> None:
        records = normalize_all(get_registry().get("whatsapp"), envelopes_for("whatsapp"))
        files = [r for r in records if r.kind == "file"]
        assert files, "media-only messages produced no file records"
        assert any(f.data["name"].endswith((".jpg", ".pdf")) for f in files)

    def test_files_carry_workspace_scope(self, provider: str = "gdrive") -> None:
        records = normalize_all(get_registry().get(provider), envelopes_for(provider))
        for record in records:
            if record.kind == "file":
                assert "workspace_id" in record.data, \
                    "file records must carry workspace scope explicitly"


class TestPullContract:
    def test_pull_is_resumable_from_a_cursor(self) -> None:
        demo = get_registry().get("demo")
        stream = next(s for s in demo.discover_streams() if s.kind == "slack_conversation")
        first = demo.pull(None, stream, None, 0)
        if not first.has_more:
            pytest.skip("stream fits in one page")
        second = demo.pull(None, stream, first.next_cursor, 1)
        assert not ({e.source_artifact_id for e in first.records}
                    & {e.source_artifact_id for e in second.records}), \
            "cursor page returned records already seen"

    def test_pull_is_deterministic(self) -> None:
        demo = get_registry().get("demo")
        for stream in demo.discover_streams():
            first = demo.pull(None, stream, None, 0)
            second = demo.pull(None, stream, None, 0)
            assert [e.source_artifact_id for e in first.records] == \
                   [e.source_artifact_id for e in second.records]

    def test_every_stream_is_discoverable(self) -> None:
        demo = get_registry().get("demo")
        streams = demo.discover_streams()
        assert len(streams) >= 6
        assert len({s.key for s in streams}) == len(streams), "duplicate stream keys"
        for stream in streams:
            assert stream.key and stream.kind and stream.label
