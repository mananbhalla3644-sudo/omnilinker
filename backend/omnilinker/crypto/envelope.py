"""AEAD envelope encryption with AAD binding (blueprint section 10.2).

Format stored in the document store:

    {
      "v": 1,
      "alg": "AES-256-GCM",
      "kid": "dek_slack_v1",
      "nonce": "<12 bytes, b64>",
      "ct": "<ciphertext||tag, b64>",
      "aad": "<sha256 of the binding string, hex>"
    }

Why AAD matters: the binding string is
    `<workspace>|<doc_id>|<field_path>|<schema_version>`
and AES-GCM authenticates it. An attacker who can *write* to the database
cannot move a ciphertext from one document (or one field) to another - the
tag check fails. This blocks ciphertext-substitution attacks, which are the
realistic threat when the attacker has DB write but not the key.

Algorithm note (blueprint 10.2 deviation, recorded as ADR-021): the blueprint
specifies XChaCha20-Poly1305. We use AES-256-GCM because it is available in
`cryptography` everywhere including PyPy/WASM builds, has hardware
acceleration on all target platforms, and gets a 96-bit nonce which is safe
under our "one key per source, random nonce per write" model. Switching to
XChaCha20 is a one-function change in this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Iterable, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from omnilinker.crypto.keys import KeyManager, get_key_manager

ENVELOPE_VERSION = 1
ALG = "AES-256-GCM"
NONCE_LEN = 12
SCHEMA_VERSION = "v1"


class DecryptionError(RuntimeError):
    """Raised when a ciphertext fails authentication. Never swallow this."""


def build_aad(workspace: str, doc_id: str, field_path: str) -> str:
    return f"{workspace}|{doc_id}|{field_path}|{SCHEMA_VERSION}"


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def is_envelope(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("v") == ENVELOPE_VERSION
        and value.get("alg") == ALG
        and "ct" in value
        and "nonce" in value
    )


def encrypt(
    source: str,
    plaintext: bytes | str,
    *,
    doc_id: str,
    field_path: str,
    key_manager: KeyManager | None = None,
) -> dict:
    """Encrypt one value. `field_path` is the dotted path inside the document."""
    km = key_manager or get_key_manager()
    keys = km.dek(source)
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    nonce = os.urandom(NONCE_LEN)
    aad = build_aad(km.workspace, doc_id, field_path)
    ct = AESGCM(keys.dek).encrypt(nonce, plaintext, aad.encode("utf-8"))
    return {
        "v": ENVELOPE_VERSION,
        "alg": ALG,
        "kid": keys.kid,
        "nonce": _b64e(nonce),
        "ct": _b64e(ct),
        "aad": hashlib.sha256(aad.encode("utf-8")).hexdigest(),
    }


def decrypt(
    source: str,
    envelope: Mapping,
    *,
    doc_id: str,
    field_path: str,
    key_manager: KeyManager | None = None,
) -> bytes:
    """Decrypt and authenticate. Raises DecryptionError on tampering."""
    if not is_envelope(envelope):
        raise DecryptionError(f"value at {field_path} is not a v{ENVELOPE_VERSION} envelope")
    km = key_manager or get_key_manager()
    if envelope.get("kid") != km.dek(source).kid:
        raise DecryptionError(f"key id mismatch: envelope wants {envelope.get('kid')!r}")
    aad = build_aad(km.workspace, doc_id, field_path)
    if hashlib.sha256(aad.encode("utf-8")).hexdigest() != envelope.get("aad"):
        # AAD mismatch means the ciphertext was moved to a different document or
        # field. Fail closed and say exactly that - it is a diagnostic, not noise.
        raise DecryptionError(
            f"AAD mismatch for {doc_id}.{field_path}: ciphertext was moved or "
            f"the document id/field path changed after encryption."
        )
    try:
        return AESGCM(km.dek(source).dek).decrypt(
            _b64d(envelope["nonce"]), _b64d(envelope["ct"]), aad.encode("utf-8")
        )
    except InvalidTag as exc:  # pragma: no cover - exercised in tests
        raise DecryptionError(f"authentication failed for {doc_id}.{field_path}") from exc


# ---------------------------------------------------------------------------
# Object helpers. The pipeline uses these so it never hand-rolls JSON.
# ---------------------------------------------------------------------------


def encrypt_obj(
    source: str,
    obj: Any,
    *,
    doc_id: str,
    field_path: str = "$",
    key_manager: KeyManager | None = None,
) -> dict:
    return encrypt(
        source,
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        doc_id=doc_id,
        field_path=field_path,
        key_manager=key_manager,
    )


def decrypt_obj(
    source: str,
    envelope: Mapping,
    *,
    doc_id: str,
    field_path: str = "$",
    key_manager: KeyManager | None = None,
) -> Any:
    raw = decrypt(source, envelope, doc_id=doc_id, field_path=field_path, key_manager=key_manager)
    return json.loads(raw.decode("utf-8"))


def seal_fields(
    source: str,
    record: Mapping,
    fields: Iterable[str],
    *,
    doc_id: str,
    key_manager: KeyManager | None = None,
) -> dict:
    """Return a copy of `record` with the named top-level fields replaced by
    envelopes. Everything else stays queryable in the clear.

    This is the deliberate trade from blueprint 10.2 "Why not full field-level
    E2EE": filters, sorts and facets need plaintext metadata, so only the
    high-value content fields (message bodies, note text, transcripts, doc
    text) are sealed.
    """
    out = dict(record)
    for name in fields:
        if name in out and out[name] not in (None, ""):
            out[name] = encrypt(
                source,
                out[name] if isinstance(out[name], str) else json.dumps(out[name], ensure_ascii=False),
                doc_id=doc_id,
                field_path=name,
                key_manager=key_manager,
            )
    return out


def unseal_fields(
    source: str,
    record: Mapping,
    fields: Iterable[str],
    *,
    doc_id: str,
    key_manager: KeyManager | None = None,
) -> dict:
    """Inverse of `seal_fields`. Non-envelope values pass through unchanged, so
    this is safe to call on a mixed record (e.g. after a decrypt-disabled run).
    """
    out = dict(record)
    for name in fields:
        value = out.get(name)
        if is_envelope(value):
            plain = decrypt(
                source, value, doc_id=doc_id, field_path=name, key_manager=key_manager
            )
            out[name] = plain.decode("utf-8")
    return out


#: Content fields sealed per artifact kind (blueprint Class A / B).
#:
#: **Subject lines and note titles are deliberately NOT sealed.** This is a
#: considered position, not an oversight:
#:
#: * A title is the primary retrieval key. Sealing it makes a document
#:   unfindable by the words the user already knows, which is the single worst
#:   failure mode a search product can have.
#: * A title is metadata that already appears in provider UIs, notification
#:   previews and email headers. Sealing it protects nothing that is not
#:   already exposed.
#: * The body is where the actual private content lives, and that is sealed.
#:
#: The cost, stated plainly: a `search_docs` row for an email carries its
#: subject in the clear, so read access to the derived store reveals subject
#: lines. That is a real, documented exposure, and it is the same one every
#: mail provider accepts.
#: `thread_subject` is included on messages deliberately. It is a *derived*
#: field - a sentence extracted from the body by `ai.enrich.derive_subject` -
#: so leaving it in the clear is the system choosing to un-seal content, which
#: is a materially different act from a provider handing us a subject line
#: (those are deliberately not sealed; see above). Ranking is unaffected: the
#: index receives the decrypted title in memory at build time, exactly as it
#: receives the body.
SEALED_FIELDS: dict[str, tuple[str, ...]] = {
    "message": ("body_text", "body_html", "thread_subject"),
    "email": ("body_text", "body_html"),
    "note": ("body_text",),
    "file": ("extracted_text",),
    # `segments` carries the same words as `text`, one caption cue at a time.
    # Sealing only `text` and leaving the segments in the clear would store the
    # transcript twice, once encrypted.
    "transcript": ("text", "segments"),
    "document": ("text",),
    "summary": ("text",),
}


def sealed_fields_for(kind: str) -> tuple[str, ...]:
    return SEALED_FIELDS.get(kind, ("body_text",))
