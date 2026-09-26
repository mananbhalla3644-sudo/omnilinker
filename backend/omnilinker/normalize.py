"""Canonical schema + normalization helpers (blueprint 5.3.3, 6.6).

Everything downstream of `normalize()` is provider-agnostic. This module owns:
  * text cleanup (HTML -> text, mrkdwn -> text, de-quoting)
  * entity extraction (emails, phones, handles, URLs, dates, deadlines)
  * canonical record builders, which mint the ULIDs and attach lineage

Entity extraction is regex/rule based here on purpose: it must run offline,
be deterministic and be testable without a model. The blueprint's LLM-based
NER is an *upgrade* layered on top (ai/enrich.py), not a prerequisite.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from omnilinker.normalize_dates import extract_dates, extract_deadlines

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_MRKDWN_BOLD = re.compile(r"\*([^*\n]{1,200})\*")
_MRKDWN_ITALIC = re.compile(r"(?<!\*)\b_([^_\n]{1,200})_\b(?!\*)")
_SLACK_LINK = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")
_SLACK_CHANNEL = re.compile(r"<#([A-Z0-9]+)\|([^>]*)>")
_SLACK_USER = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]*))?>")
_MD_BOLD = re.compile(r"\*\*([^*\n]{1,300})\*\*")
_QUOTE = re.compile(r"^\s*(?:>+\s?)+", re.M)


def strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = _SCRIPT.sub(" ", raw)
    text = _TAG.sub(" ", text)
    return html.unescape(text)


def clean_text(raw: str, *, flavour: str = "plain") -> str:
    """Provider markup -> normalized plain text.

    flavour: plain | slack | markdown | whatsapp
    """
    if not raw:
        return ""
    text = raw
    if flavour == "slack":
        text = _SLACK_USER.sub(lambda m: m.group(2) or f"@{m.group(1)}", text)
        text = _SLACK_CHANNEL.sub(lambda m: f"#{m.group(2) or m.group(1)}", text)
        text = _SLACK_LINK.sub(lambda m: f"{m.group(2)} ({m.group(1)})", text)
    elif flavour == "markdown":
        text = _MD_BOLD.sub(r"\1", text)
    elif flavour == "whatsapp":
        text = re.sub(r"‎|‏|؜", "", text)  # bidi/control marks
    else:
        text = strip_html(text)
    text = _MRKDWN_BOLD.sub(r"\1", text)
    text = _MRKDWN_ITALIC.sub(r"\1", text)
    text = _QUOTE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def snippet(text: str, query: str = "", width: int = 220) -> str:
    """Highlight window for a search hit: centred on the first query term."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if not query:
        return flat[:width] + ("..." if len(flat) > width else "")
    low = flat.lower()
    pos = -1
    for term in query.split():
        pos = low.find(term.lower())
        if pos >= 0:
            break
    if pos < 0:
        return flat[:width] + ("..." if len(flat) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    out = flat[start:end]
    return ("..." if start else "") + out + ("..." if end < len(flat) else "")


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
RE_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{4}(?![\w.])")
RE_HANDLE = re.compile(r"(?<![\w@])@([A-Za-z0-9._-]{2,30})\b")
RE_URL = re.compile(r"https?://[^\s<>\"')\]]+")
RE_MONEY = re.compile(r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?[kKmM]?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|INR|usd|eur|gbp|inr)\b)")
RE_FILE_REF = re.compile(r"\b[\w][\w .\-]{0,60}\.(pdf|docx?|xlsx?|pptx?|csv|md|txt|key|numbers|pages|zip|json|py|ts|tsx|go|sql)\b", re.I)


#: How many trailing lines of an email count as a signature block. Four, not
#: one: real signatures are "Alice Chen / Head of Product / northwind.io /
#: +44 7700 900123", and a window of one line misses the phone entirely.
SIGNATURE_LINES = 4

#: A signature is usually set off by a greeting or an explicit sign-off. When
#: one is present we trust the split; otherwise we only scan the tail.
_SIGN_OFF = re.compile(
    r"^\s*(?:thanks?|thank you|regards|best|cheers|sincerely|kind regards|"
    r"best regards|all the best|talk soon|--\s*|-\s*$)\b", re.I | re.M
)


def extract_signature(text: str) -> dict[str, str]:
    """Pull contact details out of an email signature block.

    Scoped narrowly on purpose, because the output is attributed to the
    *sender* of the message - a phone number found in the middle of someone's
    email is a contact mention, not their number. Two guards make that safe
    enough to act on:

    * only the last `SIGNATURE_LINES` non-empty lines are considered;
    * if the body opens with a greeting or sign-off marker, everything above
      that marker is discarded first, so quoted threads and earlier paragraphs
      cannot contribute.

    It is still a heuristic. Callers must treat the result as evidence to be
    confirmed by a second source, which is exactly what the resolver's scoring
    does - a phone alone does not clear the merge threshold, a phone *and* a
    name does.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    mark = _SIGN_OFF.search(text or "")
    tail_start = 0
    if mark:
        tail_start = text[: mark.start()].count("\n")
    lines = lines[max(tail_start, len(lines) - SIGNATURE_LINES):]
    window = "\n".join(lines)
    emails = [m.group(0).lower() for m in RE_EMAIL.finditer(window)]
    phones = [m.group(0).strip() for m in RE_PHONE.finditer(window)
              if sum(c.isdigit() for c in m.group(0)) >= 9]
    out: dict[str, str] = {}
    if emails:
        out["email"] = emails[0]
    if phones:
        out["phone"] = phones[0]
    if not out:
        return {}
    out["lines"] = str(len(lines))
    return out


def extract_entities(text: str) -> dict[str, list[str]]:
    """Cheap, deterministic NER. Precision over recall by design."""
    if not text:
        return {"emails": [], "phones": [], "handles": [], "urls": [],
                "money": [], "file_refs": []}
    emails = sorted({m.group(0).lower() for m in RE_EMAIL.finditer(text)})
    handles = sorted({m.group(1) for m in RE_HANDLE.finditer(text)} - set(emails))
    urls = sorted({m.group(0).rstrip(".,);") for m in RE_URL.finditer(text)})
    money = sorted({m.group(0).strip() for m in RE_MONEY.finditer(text)})
    file_refs = sorted({m.group(0).strip() for m in RE_FILE_REF.finditer(text)})
    phones = sorted(
        {m.group(0).strip() for m in RE_PHONE.finditer(text)
         if sum(c.isdigit() for c in m.group(0)) >= 9}
    )
    return {"emails": emails, "phones": phones[:5], "handles": handles[:10],
            "urls": urls[:10], "money": money[:10], "file_refs": file_refs[:10]}


def mentions_of(text: str, known_names: dict[str, str]) -> list[dict]:
    """Match known display names in text -> EntityMention records (7.2 T4).

    `known_names` maps a lowercase name to a person_id. Longest-first so
    "Alex Kim" wins over "Alex".
    """
    if not text or not known_names:
        return []
    out: list[dict] = []
    low = text.lower()
    for name in sorted(known_names, key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", low):
            out.append({
                "text": text[m.start():m.end()],
                "type": "PERSON",
                "normalized": name,
                "span_start": m.start(),
                "span_end": m.end(),
                "resolved_person_id": known_names[name],
                "confidence": 0.9 if len(name) > 3 else 0.7,
            })
    return out


# ---------------------------------------------------------------------------
# Canonical record builders
# ---------------------------------------------------------------------------


def lineage(
    *,
    source_artifact_id: str,
    ingest_run_id: str,
    connector_version: str,
    provider: str,
    raw_ref: str | None = None,
) -> dict:
    return {
        "source_artifact_id": source_artifact_id,
        "ingest_run_id": ingest_run_id,
        "connector_version": connector_version,
        "provider": provider,
        "raw_ref": raw_ref,
        "transform_hash": content_hash(f"{source_artifact_id}|{connector_version}"),
    }


def iso(ts: float | datetime | None = None) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_message(
    *,
    provider: str,
    provider_msg_id: str,
    conversation_id: str,
    sender_ref: str,
    ts: str,
    body: str,
    workspace: str,
    sender_name: str = "",
    thread_root_id: str = "",
    reply_to_id: str = "",
    flavour: str = "plain",
    attachments: Iterable[dict] | None = None,
    reactions: list[dict] | None = None,
    direction: str = "unknown",
    extra: dict | None = None,
) -> dict:
    text = clean_text(body, flavour=flavour)
    ents = extract_entities(text)
    return {
        "kind": "message",
        "workspace_id": workspace,
        "provider": provider,
        "provider_msg_id": str(provider_msg_id),
        "conversation_id": conversation_id,
        "thread_root_id": thread_root_id,
        "reply_to_id": reply_to_id,
        "sender_ref": sender_ref,
        "sender_name": sender_name,
        "ts": ts,
        "body_text": text,
        "body_hash": content_hash(text),
        "lang": "en",
        "direction": direction,
        "attachments": list(attachments or []),
        "reactions": reactions or [],
        "entities": {
            "emails": ents["emails"],
            "phones": ents["phones"],
            "handles": ents["handles"],
            "urls": ents["urls"],
            "money": ents["money"],
            "file_refs": ents["file_refs"],
            "person_ids": [],
            "dates": extract_dates(text, anchor=ts),
            "deadlines": extract_deadlines(text, anchor=ts),
            "mentions": [],
        },
        "importance": 0.0,  # filled in by pipeline scoring
        "extra": extra or {},
    }


def new_file(
    *,
    provider: str,
    provider_file_id: str,
    name: str,
    workspace: str,
    owner_ref: str = "",
    mime: str = "",
    size_bytes: int = 0,
    modified_ts: str = "",
    created_ts: str = "",
    path: str = "",
    folder_path: list[str] | None = None,
    shared_with: list[str] | None = None,
    checksum: str = "",
    extra: dict | None = None,
) -> dict:
    return {
        "kind": "file",
        "workspace_id": workspace,
        "provider": provider,
        "provider_file_id": str(provider_file_id),
        "name": name,
        "mime": mime,
        "size_bytes": int(size_bytes or 0),
        "modified_ts": modified_ts or iso(),
        # No wall-clock fallback: see new_file. The pipeline stamps ingest time.
        "path": path,
        "folder_path": list(folder_path or []),
        "owner_ref": owner_ref,
        "shared_with": list(shared_with or []),
        "checksum": checksum or content_hash(f"{provider}:{provider_file_id}"),
        "entities": {"person_ids": [], "urls": []},
        "extra": extra or {},
    }


def new_note(
    *,
    provider: str,
    provider_page_id: str,
    title: str,
    body_text: str,
    workspace: str,
    last_edited: str = "",
    created_ts: str = "",
    parent_id: str = "",
    url: str = "",
    block_count: int = 0,
    tags: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    return {
        "kind": "note",
        "workspace_id": workspace,
        "provider": provider,
        "provider_page_id": str(provider_page_id),
        "title": title,
        "body_text": body_text,
        "body_hash": content_hash(body_text),
        "url": url,
        "parent_id": parent_id,
        "block_count": block_count,
        "tags": tags or [],
        "last_edited": last_edited or iso(),
        # No wall-clock fallback: see new_file. The pipeline stamps ingest time.
        "entities": {"person_ids": [], "urls": [], "file_refs": []},
        "extra": extra or {},
    }


def new_identity(
    *,
    provider: str,
    provider_user_id: str,
    workspace: str,
    display_name: str = "",
    email: str = "",
    phone: str = "",
    username: str = "",
    is_self: bool = False,
    kind: str = "human",
) -> dict:
    return {
        "kind": "identity",
        "workspace_id": workspace,
        "provider": provider,
        "provider_user_id": str(provider_user_id),
        "display_name": display_name,
        "email": email,
        "phone": phone,
        "username": username,
        "kind": kind,
        "is_self": is_self,
        "first_seen": iso(),
        "last_seen": iso(),
    }
