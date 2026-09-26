"""Ingestion pipeline (blueprint 6.6): raw -> canonical -> sealed -> stored.

The pipeline is the only place that knows the full order of operations:

    1. pull            provider page -> RawEnvelope (immutable)
    2. raw artifact    store the provider payload verbatim, unencrypted-at-field
    3. normalize       provider payload -> CanonicalRecord (pure)
    4. derive          entity extraction, importance, conversation keys
    5. seal            encrypt content fields with the source DEK
    6. persist         idempotent upsert into the document store
    7. cursor          advance the per-stream cursor, only on success
    8. audit           append-only record of what happened

Two invariants the rest of the system depends on:

  * **Idempotency.** The document id is `f"{provider}:{provider_key}"`. Re-running
    a sync over an overlapping window overwrites in place and creates nothing
    new. This is what makes at-least-once delivery safe, which is the only kind
    of delivery a provider can honestly offer.
  * **Cursors advance only on full success.** A partial failure leaves the
    cursor where it was, so the next run re-reads the window. Re-reading is
    free (upsert), skipping data is not.

Order matters too: raw is written *before* normalization, so if a normalizer
crashes on a payload shape we have never seen, the evidence survives and the
bug is reproducible from the stored artifact.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable

from omnilinker.config import get_settings
from omnilinker.connectors.contract import CanonicalRecord, Connector, RawEnvelope
from omnilinker.connectors.registry import get_registry
from omnilinker.crypto.envelope import (
    SEALED_FIELDS,
    encrypt,
    is_envelope,
    seal_fields,
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
from omnilinker.ids import prefixed
from omnilinker.normalize import content_hash, iso
from omnilinker.store import get_stores
from omnilinker.store.base import Filter, Query

#: canonical kind -> document collection
COLLECTION_FOR_KIND = {
    "message": "messages",
    "email": "messages",
    "file": "files",
    "note": "notes",
    "video": "videos",
    "transcript": "transcripts",
    "identity": "identities",
    "membership": "memberships",
}

#: The field holding the provider's own id for each kind. This is the
#: idempotency key, so it must be the provider's stable identifier - never a
#: hash of mutable content, and never a local autoincrement.
NATURAL_KEY_FOR_KIND = {
    "message": "provider_msg_id",
    "email": "provider_msg_id",
    "file": "provider_file_id",
    "note": "provider_page_id",
    "video": "provider_video_id",
    # Deliberately NOT the video id: transcripts and videos become graph nodes,
    # and a shared id makes one silently overwrite the other.
    "transcript": "provider_transcript_id",
    "identity": "provider_user_id",
    "membership": "provider_membership_id",
}


@dataclass
class Counters:
    envelopes: int = 0
    records: int = 0
    created: int = 0
    updated: int = 0
    sealed: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "envelopes": self.envelopes, "records": self.records,
            "created": self.created, "updated": self.updated,
            "sealed": self.sealed, "skipped": self.skipped, "failed": self.failed,
        }


@dataclass
class IngestRun:
    ingest_run_id: str
    connector_id: str
    workspace: str
    started_at: str
    trigger: str = "manual"
    status: str = "running"
    counters: Counters = field(default_factory=Counters)
    streams: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    finished_at: str = ""
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "ingest_run_id": self.ingest_run_id,
            "connector_id": self.connector_id,
            "workspace": self.workspace,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "trigger": self.trigger,
            "status": self.status,
            "counters": self.counters.as_dict(),
            "streams": self.streams,
            "errors": self.errors[:50],
            "error_count": len(self.errors),
            "duration_ms": self.duration_ms,
        }


class IngestPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.stores = get_stores()
        self.km = get_key_manager()
        self.workspace = self.settings.default_workspace

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def sync_connector(
        self,
        connector_id: str,
        *,
        grant: Any = None,
        stream_filter: str | None = None,
        max_streams: int | None = None,
        trigger: str = "manual",
        seed_identity: bool = True,
    ) -> IngestRun:
        run = IngestRun(
            ingest_run_id=prefixed("run"),
            connector_id=connector_id,
            workspace=self.workspace,
            started_at=iso(),
            trigger=trigger,
        )
        t0 = time.perf_counter()
        connector = get_registry().get(connector_id)

        try:
            streams = connector.discover_streams(grant)
        except Exception as exc:
            run.status = "failed"
            run.errors.append({"stage": "discover_streams", "error": str(exc),
                               "trace": traceback.format_exc()[-1200:]})
            self._finish(run, t0)
            return run

        if stream_filter:
            streams = [s for s in streams
                       if s.key == stream_filter or s.label == stream_filter]
        if max_streams is not None:
            streams = streams[:max_streams]

        if seed_identity and connector_id == "demo":
            self._seed_people(run)

        for stream in streams:
            self._sync_stream(connector, run, grant, stream)

        self._persist_run(run)
        self._finish(run, t0)
        return run

    # ------------------------------------------------------------------
    # Per stream
    # ------------------------------------------------------------------
    def _sync_stream(self, connector: Connector, run: IngestRun, grant: Any, stream) -> None:
        cursor = self.get_cursor(run.workspace, run.connector_id, stream.key)
        stats = {"stream": stream.key, "label": stream.label, "from_cursor": cursor,
                 "envelopes": 0, "records": 0, "failed": 0, "next_cursor": None}
        page = 0
        while True:
            try:
                result = connector.pull(grant, stream, cursor, page)
            except Exception as exc:
                run.status = "partial" if run.status == "running" else run.status
                run.errors.append({"stage": "pull", "stream": stream.key,
                                   "error": str(exc),
                                   "trace": traceback.format_exc()[-1200:]})
                stats["failed"] += 1
                break

            run.counters.envelopes += len(result.records)
            stats["envelopes"] += len(result.records)

            for envelope in result.records:
                envelope.workspace_id = run.workspace
                run.counters.envelopes
                try:
                    self._process_envelope(connector, run, envelope)
                    stats["records"] += 1
                except Exception as exc:
                    run.counters.failed += 1
                    stats["failed"] += 1
                    run.errors.append({
                        "stage": "process", "stream": stream.key,
                        "artifact": envelope.source_artifact_id,
                        "error": str(exc), "trace": traceback.format_exc()[-1200:],
                    })

            if result.partial_failure:
                run.errors.append({"stage": "partial_failure", "stream": stream.key,
                                   **result.partial_failure})

            if result.next_cursor:
                cursor = result.next_cursor
            stats["next_cursor"] = cursor

            # Only advance the stored cursor when the whole page succeeded.
            if stats["failed"] == 0 and result.next_cursor:
                self.set_cursor(run.workspace, run.connector_id, stream.key, cursor)
            if not result.has_more:
                break
            page += 1
            if page > 200:  # hard stop; a cursor loop must not spin forever
                run.errors.append({"stage": "pull", "stream": stream.key,
                                   "error": "page cap reached (200)"})
                break

        run.streams.append(stats)

    # ------------------------------------------------------------------
    # Per envelope
    # ------------------------------------------------------------------
    def _process_envelope(self, connector: Connector, run: IngestRun,
                          envelope: RawEnvelope) -> None:
        # 1. raw artifact, written first so a crash leaves evidence behind
        self._store_raw(envelope, run)

        # 2. normalize (pure)
        records = connector.normalize(envelope)
        run.counters.records += len(records)

        # 3. persist each
        for record in records:
            self._persist_record(run, record)

        # 4. identities implied by the raw payload
        self._collect_identities(run, envelope, records)

    def _store_raw(self, envelope: RawEnvelope, run: IngestRun) -> None:
        raw_id = f"{envelope.provider}:{envelope.stream}:{envelope.source_artifact_id}"
        existing = self.stores.docs.get("raw_artifacts", raw_id)
        payload = envelope.payload
        # Serialise once, then hash *and* store those exact bytes. Hashing
        # `str(payload)` while storing `json.dumps(payload)` produces a hash
        # that can never be verified against the stored artefact - so the check
        # silently stops meaning anything, which is worse than not having one.
        try:
            stored_payload: Any = json.dumps(payload, ensure_ascii=False,
                                             separators=(",", ":"), sort_keys=True,
                                             default=str)
        except (TypeError, ValueError):
            stored_payload = str(payload)
        payload_hash = content_hash(stored_payload)

        if existing and existing.get("payload_hash") == payload_hash:
            return  # byte-identical re-fetch: nothing to do
        # The raw payload is the same bytes as the message body, and it is a
        # *larger* copy - headers, reactions, edit history, everything. Leaving
        # it in the clear next to a sealed canonical record would make the whole
        # encryption claim false, and it is not a trade for reproducibility:
        # `payload_hash` proves the fetch is unchanged without revealing the
        # text, and the payload is decrypted on demand for a normalizer bug.
        if self.settings.enable_encryption:
            stored_payload = encrypt(
                envelope.provider,
                stored_payload, doc_id=raw_id, field_path="payload", key_manager=self.km,
            )
        self.stores.docs.put("raw_artifacts", raw_id, {
            "provider": envelope.provider,
            "stream": envelope.stream,
            "kind": envelope.kind,
            "source_artifact_id": envelope.source_artifact_id,
            "payload": stored_payload,
            "payload_sealed": bool(self.settings.enable_encryption),
            "payload_hash": payload_hash,
            "provider_meta": envelope.provider_meta,
            "fetched_at": envelope.fetched_at,
            "ingest_run_id": run.ingest_run_id,
            "workspace_id": run.workspace,
        })

    # ------------------------------------------------------------------
    # Per record
    # ------------------------------------------------------------------
    def _persist_record(self, run: IngestRun, record: CanonicalRecord) -> None:
        collection = COLLECTION_FOR_KIND.get(record.kind, "records")
        key_field = NATURAL_KEY_FOR_KIND.get(record.kind, "provider_msg_id")
        natural = str(record.data.get(key_field) or record.data.get("_id"))
        if not natural or natural == "None":
            run.counters.skipped += 1
            return

        doc_id = f"{record.provider}:{natural}"
        previous = self.stores.docs.get(collection, doc_id)
        doc = dict(record.data)
        doc["_id"] = doc_id
        doc.setdefault("created_at", iso())
        # Connectors deliberately do not stamp a creation time: it is ingest
        # metadata, and a wall-clock default inside normalize() would make the
        # function impure (two calls on one payload would differ).
        doc.setdefault("created_ts", iso())
        doc["ingest_run_id"] = run.ingest_run_id
        doc["source_artifact_id"] = record.source_artifact_id
        doc["stream"] = record.stream
        doc["lineage"] = {
            "source_artifact_id": record.source_artifact_id,
            "ingest_run_id": run.ingest_run_id,
            "provider": record.provider,
            "raw_ref": f"raw_artifacts:{record.provider}:{record.stream}:{record.source_artifact_id}",
        }

        if previous:
            doc["created_at"] = previous.get("created_at", doc["created_at"])
            doc["first_seen_at"] = previous.get("first_seen_at", doc["created_at"])

        # Seal content fields.
        if self.settings.enable_encryption:
            fields = SEALED_FIELDS.get(record.kind, ())
            sealed_any = [f for f in fields if f in doc and doc[f] not in (None, "")]
            if sealed_any:
                doc = seal_fields(record.provider, doc, sealed_any,
                                  doc_id=doc_id, key_manager=self.km)
                run.counters.sealed += 1
        # The plaintext search projection is a *derived* store. It is dropped
        # and rebuilt by `omnilinker.search`; it is never the source of truth.
        self.stores.docs.put(collection, doc_id, doc)
        if previous:
            run.counters.updated += 1
        else:
            run.counters.created += 1

        if record.kind in ("message", "email"):
            self._ensure_conversation(run, record, doc)

    def _ensure_conversation(self, run: IngestRun, record: CanonicalRecord, doc: dict) -> None:
        conv_key = f"{record.provider}:{record.stream or doc.get('conversation_id')}"
        conv_id = f"conv_{record.provider}_{_slug(doc.get('conversation_id') or record.stream)}"
        existing = self.stores.docs.get("conversations", conv_id)
        if not existing:
            self.stores.docs.put("conversations", conv_id, {
                "_id": conv_id,
                "workspace_id": run.workspace,
                "conversation_key": conv_key,
                "provider": record.provider,
                "stream": record.stream,
                "conversation_id": doc.get("conversation_id", ""),
                "label": _conversation_label(record, doc),
                "kind": _conversation_kind(record),
                "first_ts": doc.get("ts", ""),
                "last_ts": doc.get("ts", ""),
                "message_count": 1,
                "member_refs": [],
                "created_at": iso(),
            })
        else:
            patch = {"last_ts": max(existing.get("last_ts", ""), doc.get("ts", "")),
                     "first_ts": min(existing.get("first_ts", doc.get("ts", "")),
                                     doc.get("ts", "")),
                     "message_count": existing.get("message_count", 0) + 1}
            existing.update(patch)
            self.stores.docs.put("conversations", conv_id, existing)

    # ------------------------------------------------------------------
    # Identities
    # ------------------------------------------------------------------
    def _collect_identities(self, run: IngestRun, envelope: RawEnvelope,
                            records: Iterable[CanonicalRecord]) -> None:
        """Mint an Identity node source for every `provider:ref` we see.

        Identity records are the raw, per-source view of a human. The resolver
        (section 7) decides which of these are the same person; it never trusts
        the provider's own claim that two handles are the same human.
        """
        refs: dict[str, dict] = {}
        for record in records:
            if record.kind not in ("message", "email", "file", "note", "video"):
                continue
            data = record.data
            extra = data.get("extra") or {}
            for ref_field, name, email, phone in (
                ("sender_ref", data.get("sender_name", ""),
                 extra.get("sender_email", ""), extra.get("sender_phone", "")),
                # The owner's name comes from the provider's owner record, never
                # from the artifact. Using `data["name"]` here made every file
                # its owner's "identity", and produced a Person called
                # "Brightfold-MSA-signed.pdf" - which then merged, correctly but
                # uselessly, with the human who owned it.
                ("owner_ref", extra.get("owner_name", ""),
                 extra.get("owner_email", ""), extra.get("owner_phone", "")),
                ("author_ref", extra.get("author_name", "") or data.get("author_name", ""),
                 extra.get("author_email", ""), extra.get("author_phone", "")),
            ):
                ref = data.get(ref_field)
                if not ref or not isinstance(ref, str) or ":" not in ref:
                    continue
                entry = refs.setdefault(ref, {"ref": ref, "name": name, "email": email,
                                              "phone": phone, "count": 0})
                entry["count"] += 1
                if not entry["name"] and name:
                    entry["name"] = name
                if not entry["email"] and email:
                    entry["email"] = email
                if not entry["phone"] and phone:
                    entry["phone"] = phone
                if extra.get("is_self"):
                    entry["is_self"] = True
            # A signature block belongs to the *sender*, and is one of the few
            # places a phone number is reliably attached to the right human.
            # Attribution is scoped to the sender's own message - a phone number
            # found in someone else's message is a contact, never the sender's.
            signature = extra.get("signature") or {}
            sender_ref = data.get("sender_ref")
            if isinstance(signature, dict) and sender_ref:
                entry = refs.get(str(sender_ref))
                if entry is not None:
                    if not entry.get("phone") and signature.get("phone"):
                        entry["phone"] = signature["phone"]
                    if not entry.get("email") and signature.get("email"):
                        entry["email"] = signature["email"]

            # Contact details mentioned in bodies are attributed to a Person, not
            # promoted to one. See `IdentityResolver._attribute_contacts`.
            for addr in (data.get("entities", {}) or {}).get("emails", []) or []:
                ref = f"contact:{record.provider}:{addr}"
                entry = refs.setdefault(ref, {"ref": ref, "name": "", "email": addr,
                                              "phone": "", "count": 0})
                entry["count"] += 1

        for entry in refs.values():
            self._upsert_identity(run, **entry)

    def _upsert_identity(self, run: IngestRun, *, ref: str, name: str = "",
                         email: str = "", phone: str = "", count: int = 1,
                         is_self: bool = False) -> None:
        provider = ref.split(":", 1)[0] if ":" in ref else "unknown"
        local = ref.split(":", 1)[1] if ":" in ref else ref
        doc_id = f"idt_{provider}_{_slug(local)}"
        existing = self.stores.docs.get("identities", doc_id) or {}
        is_self = bool(is_self or existing.get("is_self"))
        # The longest name wins across sightings. "clara" from a Slack @mention
        # and "Clara Nowak" from a profile are the same person, and storing the
        # truncated form is what makes a merged Person display as "clara".
        previous_name = str(existing.get("display_name") or "")
        name = name if len(name) >= len(previous_name) else previous_name
        email = email or existing.get("email", "")
        phone = phone or existing.get("phone", "")

        # Workspace-wide identity pepper, NOT the per-source DEK pepper: tokens
        # must be comparable across sources or cross-source resolution cannot
        # work at all. See KeyManager.identity_pepper for why that distinction
        # is load-bearing and what it costs.
        pepper = self.km.identity_pepper
        doc = {
            "_id": doc_id,
            "workspace_id": run.workspace,
            "provider": provider,
            "provider_user_id": local,
            "ref": ref,
            "display_name": name,
            "email": email,
            "phone": phone,
            "mention_count": int(existing.get("mention_count") or 0) + count,
            # Deterministic tokens: equality across sources without ever
            # storing or decrypting the address itself.
            "email_token": token_for_email(pepper, email) if email else "",
            "phone_token": token_for_phone(pepper, phone) if phone else "",
            "name_token": token_for_name(pepper, name) if name else "",
            "first_seen": existing.get("first_seen", iso()),
            "last_seen": iso(),
            "person_id": existing.get("person_id", ""),
            "is_self": is_self,
        }
        self.stores.docs.put("identities", doc_id, doc)

    def _seed_people(self, run: IngestRun) -> None:
        """Give the demo cast real identity records (names + contact details),
        which is what gives the identity resolver something non-trivial to do."""
        from omnilinker.connectors.providers.demo import PEOPLE

        for p in PEOPLE:
            for ref, email, phone, name in (
                (f"slack:{p['slack_id']}", "", "", p["name"].split()[0].lower()),
                (f"gmail:{p['gmail']}", p["gmail"], "", p["name"]),
                (f"whatsapp:{p['wa']}", "", p.get("phone", ""), p["name"]),
                (f"discord:{p['discord_id']}", p["gmail"], "", p["name"]),
            ):
                if not (email or phone or name):
                    continue
                self._upsert_identity(run, ref=ref, name=name, email=email,
                                      phone=phone, count=1)

    # ------------------------------------------------------------------
    # Cursors + run bookkeeping
    # ------------------------------------------------------------------
    def cursor_id(self, connector_id: str, stream: str) -> str:
        return f"cur_{connector_id}_{_slug(stream)}"

    def get_cursor(self, workspace: str, connector_id: str, stream: str) -> str | None:
        doc = self.stores.docs.get("cursors", self.cursor_id(connector_id, stream))
        return doc.get("cursor") if doc else None

    def set_cursor(self, workspace: str, connector_id: str, stream: str, cursor: str) -> None:
        cid = self.cursor_id(connector_id, stream)
        existing = self.stores.docs.get("cursors", cid) or {}
        self.stores.docs.put("cursors", cid, {
            **existing,
            "_id": cid,
            "workspace_id": workspace,
            "connector_id": connector_id,
            "stream": stream,
            "cursor": cursor,
            "updated_at": iso(),
        })

    def _persist_run(self, run: IngestRun) -> None:
        run_id = run.ingest_run_id
        existing = self.stores.docs.get("ingest_runs", run_id) or {}
        self.stores.docs.put("ingest_runs", run_id, {**existing, **run.as_dict()})
        # Keep the run log bounded: keep the newest 200.
        all_runs = self.stores.docs.find(Query("ingest_runs", sort=[("started_at", -1)],
                                                limit=200))
        for stale in all_runs[200:]:
            self.stores.docs.delete("ingest_runs", stale["_id"])

    def _finish(self, run: IngestRun, t0: float) -> None:
        run.finished_at = iso()
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        if run.status == "running":
            run.status = "failed" if run.counters.failed and not run.counters.records else "ok"
        self._persist_run(run)
        self.audit({
            "action": "sync",
            "connector_id": run.connector_id,
            "ingest_run_id": run.ingest_run_id,
            "status": run.status,
            "counters": run.counters.as_dict(),
        })

    def audit(self, entry: dict) -> None:
        entry_id = prefixed("aud")
        self.stores.docs.put("audit", entry_id, {**entry, "_id": entry_id, "ts": iso()})

    # ------------------------------------------------------------------
    # Read helpers used by the API and the indexer
    # ------------------------------------------------------------------
    def decrypt_record(self, doc: dict) -> dict:
        """Return a copy of a stored document with content fields decrypted."""
        if not self.settings.enable_encryption:
            return dict(doc)
        provider = doc.get("provider", "")
        kind = _kind_of(doc)
        fields = SEALED_FIELDS.get(kind, ())
        return unseal_fields(provider, doc, fields, doc_id=doc["_id"], key_manager=self.km)

    def count(self, collection: str, filters: list[Filter] | None = None) -> int:
        return self.stores.docs.count(collection, filters or [])


# --------------------------------------------------------------------------


def _slug(value: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(value or "unknown"))
    return out.strip("_")[:64] or "unknown"


def _kind_of(doc: dict) -> str:
    kind = doc.get("kind")
    if kind:
        return str(kind)
    provider = doc.get("provider", "")
    if provider == "gmail":
        return "email"
    return "message"


def _conversation_kind(record: CanonicalRecord) -> str:
    return {
        "slack": "channel",
        "discord": "channel",
        "gmail": "email_thread",
        "whatsapp": "chat",
        "notion": "workspace",
    }.get(record.provider, "stream")


def _conversation_label(record: CanonicalRecord, doc: dict) -> str:
    extra = doc.get("extra", {}) or {}
    return (
        extra.get("channel_label")
        or extra.get("channel")
        or extra.get("chat_label")
        or extra.get("subject")
        or doc.get("conversation_id")
        or record.stream
    )
