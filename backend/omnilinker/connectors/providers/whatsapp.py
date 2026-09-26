"""WhatsApp connector (blueprint 6.4 #1).

WhatsApp has no personal-history cloud API, so this connector is a *file*
connector: the user exports a chat and drops the archive on disk or uploads
it through the API. That is the honest architecture, and the UI says so.

Two formats are supported:

  1. iOS/Android export zip containing per-chat `chat.txt` files, optionally
     with media in a sibling folder.
  2. A single `chat.txt`.

Line grammar (both platforms, the de-facto standard):
    [15/01/2026, 10:32:11] Alice Chen: message text
    15/01/2026, 10:32 - Alice: text            (Android, dash separator)
    15/01/26, 10:32 - Messages and calls are end-to-end encrypted   (system)
    ‎15/01/2026, 10:32 - Alice: ‎<attached: 0001-PHOTO.jpg>      (bidi marks)

Incremental sync: the cursor is `(chat_id, last_message_index, last_timestamp)`.
Re-importing a longer export of the same chat is a no-op for the overlapping
prefix, because message identity is (chat_id, index) and the pipeline upserts
idempotently.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from omnilinker.connectors.contract import (
    CanonicalRecord,
    Connector,
    ConnectorDescriptor,
    PullPage,
    RawEnvelope,
    Scope,
    StreamDescriptor,
    TokenGrant,
)
from omnilinker.normalize import iso, new_message

# Bracketed: [15/01/2026, 10:32:11] Alice: text
RE_BRACKET = re.compile(
    r"^\[?(?P<d1>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})[,.]?\s+"
    r"(?P<t>\d{1,2}:\d{2}(?::\d{2})?)\s*\]?\s*(?P<sep>[-–—])\s*(?P<rest>.*)$"
)
RE_BRACKET_COLON = re.compile(
    r"^\[?(?P<d1>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})[,.]?\s+"
    r"(?P<t>\d{1,2}:\d{2}(?::\d{2})?)\s*\]\s*(?P<rest>.*)$"
)
RE_ATTACH = re.compile(r"<attached:\s*([^>]+)>")
RE_MEDIA = re.compile(r"^(?:image|video|audio|document|sticker)\s+(?:omitted|not included)", re.I)
_BIDI = re.compile(r"[‎‏؜]")


def _clean(text: str) -> str:
    return _BIDI.sub("", text or "").strip()


def _parse_date(raw: str) -> str:
    """Normalize the many export date orders to ISO-8601.

    Ambiguity rule: if the first component is > 12 it is a day; otherwise the
    order is the platform default (d/m/yyyy for WhatsApp exports, which is
    d/m/yyyy even in en-US installs).
    """
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            from datetime import datetime

            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _parse_ts(date_raw: str, time_raw: str) -> str:
    from datetime import datetime

    day = _parse_date(date_raw)
    parts = (time_raw.count(":") == 1) and "00:00" or time_raw
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return iso(datetime.strptime(f"{day} {parts}", fmt))
        except ValueError:
            continue
    return iso()


def parse_chat_text(text: str, chat_label: str) -> list[dict]:
    """Parse one chat.txt into message dicts. Pure function - unit tested
    against fixtures in tests/test_whatsapp.py."""
    messages: list[dict] = []
    pending: dict | None = None
    idx = 0
    for line in (text or "").splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = RE_BRACKET_COLON.match(line) or RE_BRACKET.match(line)
        if m:
            if pending:
                messages.append(pending)
            rest = _clean(m.group("rest"))
            # Pick the separator that appears FIRST, not by platform preference.
            #
            # Checking for " - " first looks safe and is not: the message body
            # is far more likely to contain " - " than the sender line is, so
            # "Bharat: Heads up - the index is slow" parses with the sender as
            # "Bharat: Heads up". That silently corrupts the identity of every
            # such message, and identity corruption is exactly the class of
            # bug that is invisible downstream - the message still appears, just
            # under a person who does not exist.
            dash = rest.find(" - ")
            colon = rest.find(": ")
            if dash == -1 and colon == -1:
                sender, body = "", rest
            elif colon == -1 or (dash != -1 and dash < colon):
                sender, _, body = rest.partition(" - ")
            else:
                sender, _, body = rest.partition(": ")
            idx += 1
            pending = {
                "index": idx,
                "ts": _parse_ts(m.group("d1"), m.group("t")),
                "sender": _clean(sender),
                "body": body,
                "raw": line,
            }
        elif pending is not None:
            # continuation line (multi-line message, or a wrapped long message)
            pending["body"] += "\n" + line
            pending["raw"] += "\n" + line
    if pending:
        messages.append(pending)
    return messages


class WhatsAppConnector(Connector):
    """Export-file connector. No OAuth, no network."""

    descriptor = ConnectorDescriptor(
        id="whatsapp",
        display_name="WhatsApp (export)",
        version="1.0.0",
        auth_flow="export_file",
        base_url="file://",
        scopes=(
            Scope(
                "local_file_read",
                "Read the export archive the user explicitly placed in OmniLinker",
                "normal",
            ),
        ),
        realtime_webhooks=False,
        incremental_cursor=True,
        cursor_field="message_index",
        supports_attachments=True,
        content_kinds=("message", "file"),
        rate_limit_per_sec=1000.0,  # no provider to throttle
        rate_limit_burst=1000,
        backfill_strategy="batch",
        notes="No first-party personal-history API. Real-time is unavailable for "
              "personal WhatsApp by design; the UI states this plainly.",
    )

    def __init__(self, export_dir: Path | None = None) -> None:
        self.export_dir = Path(export_dir) if export_dir else None

    # -- streams ------------------------------------------------------
    def discover_streams(self, grant: TokenGrant | None = None,
                         user_ref: str | None = None) -> list[StreamDescriptor]:
        if not self.export_dir or not self.export_dir.exists():
            return []
        streams: list[StreamDescriptor] = []
        for path in sorted(self.export_dir.rglob("*.txt")):
            label = path.stem
            # WhatsApp puts media in a sibling folder whose name contains the
            # chat name; find it so attachments can be resolved to real bytes.
            media = None
            for sibling in path.parent.glob(f"{path.stem}*"):
                if sibling.is_dir() and "media" in sibling.name.lower():
                    media = sibling
            streams.append(StreamDescriptor(
                key=f"wa:{label}", kind="chat", label=label,
                meta={"path": str(path), "media_dir": str(media) if media else "",
                      "platform": _detect_platform(path.read_text("utf-8", errors="replace")[:4000])},
            ))
        return streams

    def pull(self, grant: TokenGrant | None, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        path = Path(stream.meta["path"])
        start = int(cursor) if cursor and cursor.isdigit() else 0
        messages = parse_chat_text(path.read_text("utf-8", errors="replace"), stream.label)
        batch = messages[start: start + 1000]
        records = [
            RawEnvelope(
                source_artifact_id=f"{stream.key}:{m['index']}",
                provider="whatsapp", stream=stream.key, kind="message",
                payload={**m, "_chat_label": stream.label, "_media_dir":
                         stream.meta.get("media_dir", "")},
                provider_meta={"index": m["index"]}, fetched_at=iso(), workspace_id="",
            )
            for m in batch
        ]
        nxt = start + len(batch)
        return PullPage(records=records, next_cursor=str(nxt),
                        has_more=nxt < len(messages))

    # -- normalize ----------------------------------------------------
    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        sender = p.get("sender") or ""
        body = p.get("body", "")
        out: list[CanonicalRecord] = []

        # `is_system` is about the *sender line*, not the body. A message whose
        # whole body is "<attached: x>" is a real message with a real sender and
        # an attachment; a message with no sender at all is the export's own
        # bookkeeping ("Alice created this group", "Messages are end-to-end
        # encrypted"). Conflating them threw away the attachment events, which
        # are exactly the artifacts worth having.
        is_system = not bool(sender)
        media_only = bool(RE_MEDIA.match(body.strip()))
        attachments: list[dict] = []
        for att in RE_ATTACH.findall(body):
            name = att.strip()
            attachments.append({
                "provider_file_id": f"{raw.source_artifact_id}:{name}",
                "name": name.split("-", 1)[-1] if "-" in name else name,
                "mime": _guess_mime(name),
                "size_bytes": 0,
                "local_ref": name,
            })
            body = body.replace(f"<attached: {name}>", "")

        phone_hint = sender.split()[0] if sender else ""
        msg = new_message(
            provider="whatsapp",
            provider_msg_id=raw.source_artifact_id,
            conversation_id=raw.stream,
            sender_ref=f"whatsapp:{sender or 'system'}",
            ts=p.get("ts", iso()),
            body=body,
            workspace=raw.workspace_id,
            sender_name=sender,
            flavour="whatsapp",
            attachments=attachments,
            direction="unknown",
            extra={"chat_label": p.get("_chat_label"), "index": p.get("index"),
                   "is_system": is_system, "media_only": media_only,
                   "phone_hint": phone_hint},
        )
        out.append(CanonicalRecord("message", msg, raw.source_artifact_id, "whatsapp", raw.stream))

        for att in attachments:
            from omnilinker.normalize import new_file

            f = new_file(provider="whatsapp", provider_file_id=att["provider_file_id"],
                         name=att["name"], mime=att["mime"], workspace=raw.workspace_id,
                         modified_ts=msg["ts"], owner_ref=msg["sender_ref"],
                         path=att.get("local_ref", ""))
            f["_sent_by"] = msg["sender_ref"]
            f["_conversation_id"] = raw.stream
            out.append(CanonicalRecord("file", f, att["provider_file_id"], "whatsapp", raw.stream))
        return out


def _detect_platform(head: str) -> str:
    return "ios" if "] " in head.split("\n", 1)[0] and " - " not in head.split("\n", 1)[0] else "android"


def _guess_mime(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".pdf": "application/pdf", ".opus": "audio/ogg", ".m4a": "audio/mp4",
        ".mp4": "video/mp4", ".3gp": "video/3gpp", ".webp": "image/webp",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(ext, "application/octet-stream")


def import_export_zip(zip_path: Path, dest: Path) -> list[Path]:
    """Extract a WhatsApp export zip, skipping macOS resource forks."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or name.startswith("."):
                continue
            if not name.lower().endswith((".txt", ".jpg", ".jpeg", ".png", ".pdf",
                                          ".opus", ".m4a", ".mp4", ".webp", ".3gp")):
                continue
            # Guard against zip-slip: reject absolute or traversing paths.
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            written.append(target)
    return written
