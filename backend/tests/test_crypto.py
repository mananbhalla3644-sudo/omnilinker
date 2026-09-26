"""Crypto: envelope sealing, AAD binding, deterministic tokens (blueprint 10.2).

The AAD tests are the important ones. "It round-trips" is a weak property; "a
ciphertext cannot be moved between documents" is the property that makes
envelope encryption worth doing at all, and it is the one that fails silently
if someone changes the AAD format.
"""

from __future__ import annotations

import pytest

from omnilinker.crypto.envelope import (
    DecryptionError,
    build_aad,
    decrypt,
    encrypt,
    is_envelope,
    seal_fields,
    sealed_fields_for,
    unseal_fields,
)
from omnilinker.crypto.keys import get_key_manager
from omnilinker.crypto.tokens import (
    normalize_email,
    normalize_phone,
    token_for_email,
    token_for_name,
    token_for_phone,
)


class TestEnvelope:
    def test_roundtrip(self) -> None:
        envelope = encrypt("slack", "hello", doc_id="slack:1", field_path="body_text")
        assert is_envelope(envelope)
        assert decrypt("slack", envelope, doc_id="slack:1", field_path="body_text") == b"hello"

    def test_ciphertext_is_not_the_plaintext(self) -> None:
        envelope = encrypt("slack", "a very secret message", doc_id="slack:1",
                           field_path="body_text")
        assert "secret" not in str(envelope)

    def test_nonce_is_fresh_per_write(self) -> None:
        """Same plaintext, same key, two writes -> two different envelopes.

        A fixed nonce under one key is catastrophic for GCM: it leaks the XOR of
        plaintexts and destroys authentication. This is the single most
        important property of the format, so it is asserted directly.
        """
        first = encrypt("slack", "same", doc_id="d", field_path="body_text")
        second = encrypt("slack", "same", doc_id="d", field_path="body_text")
        assert first["nonce"] != second["nonce"]
        assert first["ct"] != second["ct"]

    def test_cannot_move_ciphertext_to_another_document(self) -> None:
        envelope = encrypt("slack", "secret", doc_id="slack:1", field_path="body_text")
        with pytest.raises(DecryptionError, match="AAD mismatch"):
            decrypt("slack", envelope, doc_id="slack:2", field_path="body_text")

    def test_cannot_move_ciphertext_to_another_field(self) -> None:
        envelope = encrypt("slack", "secret", doc_id="slack:1", field_path="body_text")
        with pytest.raises(DecryptionError):
            decrypt("slack", envelope, doc_id="slack:1", field_path="extracted_text")

    def test_cannot_move_ciphertext_to_another_workspace(self) -> None:
        envelope = encrypt("slack", "secret", doc_id="slack:1", field_path="body_text")
        with pytest.raises(DecryptionError):
            decrypt("gmail", envelope, doc_id="slack:1", field_path="body_text")

    def test_tampered_ciphertext_fails_authentication(self) -> None:
        import base64

        envelope = encrypt("slack", "secret", doc_id="d", field_path="body_text")
        raw = bytearray(base64.b64decode(envelope["ct"]))
        raw[2] ^= 0xFF  # flip a ciphertext bit
        envelope["ct"] = base64.b64encode(bytes(raw)).decode("ascii")
        with pytest.raises(DecryptionError):
            decrypt("slack", envelope, doc_id="d", field_path="body_text")

    def test_aad_contains_every_component(self) -> None:
        aad = build_aad("ws", "doc", "field")
        assert aad == "ws|doc|field|v1"

    def test_not_an_envelope_rejected(self) -> None:
        assert not is_envelope({"v": 1, "alg": "AES-256-GCM"})
        with pytest.raises(DecryptionError):
            decrypt("slack", {"nope": True}, doc_id="d", field_path="f")


class TestSealFields:
    def test_seals_only_named_fields(self) -> None:
        record = {"body_text": "secret", "provider": "slack", "ts": "2026-01-01T00:00:00Z"}
        sealed = seal_fields("slack", record, ("body_text",), doc_id="slack:1")
        assert is_envelope(sealed["body_text"])
        assert sealed["provider"] == "slack"
        assert sealed["ts"] == record["ts"]
        assert unseal_fields("slack", sealed, ("body_text",), doc_id="slack:1")["body_text"] \
            == "secret"

    def test_unseal_is_safe_on_mixed_records(self) -> None:
        """A record written with encryption off must not explode on read."""
        record = {"body_text": "plaintext because encryption was disabled"}
        out = unseal_fields("slack", record, sealed_fields_for("message"), doc_id="x")
        assert out["body_text"] == record["body_text"]

    def test_subjects_and_titles_are_deliberately_not_sealed(self) -> None:
        """Titles are the primary retrieval key; sealing them makes documents
        unfindable by the words a user already knows."""
        assert "subject" not in sealed_fields_for("email")
        assert "title" not in sealed_fields_for("note")
        assert "body_text" in sealed_fields_for("message")
        assert "text" in sealed_fields_for("transcript")


class TestTokens:
    def test_email_token_is_deterministic(self) -> None:
        pepper = b"p" * 32
        assert token_for_email(pepper, "a@b.com") == token_for_email(pepper, "A@B.com")

    def test_gmail_dot_plus_folding(self) -> None:
        """Gmail ignores dots and everything after a plus in the local part, so
        a.b+search@ and ab@ are the same mailbox. Without folding, one person
        with two Gmail aliases is two people forever."""
        pepper = b"p" * 32
        assert token_for_email(pepper, "a.b+news@Gmail.com") == \
            token_for_email(pepper, "ab@gmail.com")

    def test_non_gmail_aliases_are_distinct(self) -> None:
        pepper = b"p" * 32
        assert token_for_email(pepper, "a.b@corp.com") != \
            token_for_email(pepper, "ab@corp.com")

    def test_different_peppers_give_different_tokens(self) -> None:
        """The reason the identity pepper is workspace-wide and not per-source."""
        assert token_for_email(b"a" * 32, "x@y.com") != token_for_email(b"b" * 32, "x@y.com")

    def test_phone_normalisation(self) -> None:
        for variant in ["+44 7700 900123", "+447700900123", "44-7700-900123"]:
            assert normalize_phone(variant), variant
        assert token_for_phone(b"p" * 32, "+44 7700 900123") == \
            token_for_phone(b"p" * 32, "+447700900123")

    def test_short_numbers_are_not_phones(self) -> None:
        """A 4-digit figure is a quantity, not a telephone number. Treating it as
        one creates bogus identity matches across unrelated documents."""
        assert normalize_phone("1234") == ""
        assert normalize_phone("2026") == ""

    def test_name_token_ignores_case_and_punctuation(self) -> None:
        pepper = b"p" * 32
        assert token_for_name(pepper, "Alice Chen") == token_for_name(pepper, "alice chen")

    def test_email_normalisation_lowercases_and_trims(self) -> None:
        assert normalize_email("  Alice.Chen@Example.COM ") == "alice.chen@example.com"


class TestKeyManager:
    def test_keys_differ_per_source(self) -> None:
        km = get_key_manager()
        assert km.dek("slack").dek != km.dek("gmail").dek

    def test_same_source_is_stable(self) -> None:
        km = get_key_manager()
        assert km.dek("slack").dek == km.dek("slack").dek

    def test_identity_pepper_is_workspace_wide(self) -> None:
        """Deliberate, and load-bearing: cross-source resolution is impossible if
        the token pepper differs by provider. It must NOT be the DEK pepper."""
        km = get_key_manager()
        assert km.identity_pepper != km.dek("slack").pepper
        assert km.identity_pepper != km.dek("gmail").pepper

    def test_fingerprint_is_stable_and_short(self) -> None:
        km = get_key_manager()
        assert km.fingerprint() == km.fingerprint()
        assert len(km.fingerprint()) == 16

    def test_key_ids_are_human_readable(self) -> None:
        assert get_key_manager().dek("slack").kid == "dek_slack_v1"
