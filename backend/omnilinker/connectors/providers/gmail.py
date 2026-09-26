"""Gmail connector (blueprint 6.4 #4).

Cursor: Gmail's opaque, monotonic `historyId` - the correct incremental key.
Realtime: `users.watch` -> Pub/Sub push (a Google-owned relay, so it needs no
public ingress) with a 60s `history.list` poll as fallback. The historyId is
valid ~7 days, hence the nightly reconcile sweep in the sync orchestrator.
"""

from __future__ import annotations

import base64
import re
from email.header import decode_header, make_header
from typing import Any

from omnilinker.connectors.base_rest import BaseRestConnector
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
from omnilinker.normalize import extract_signature, clean_text, iso, new_file, new_message

API = "https://gmail.googleapis.com/gmail/v1/users/me"
_SUBJECT_DATE = re.compile(r"^(?P<subject>.*?)\s*\((?P<date>[^)]+)\)$", re.S)


def _header(msg: dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            try:
                return str(make_header(decode_header(h.get("value", ""))))
            except Exception:
                return h.get("value", "")
    return ""


def _b64url_decode(data: str) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
    except Exception:
        return ""


def walk_parts(part: dict) -> list[dict]:
    """Flatten a MIME tree into leaf parts."""
    if not part:
        return []
    mime = part.get("mimeType", "")
    body = part.get("body", {}) or {}
    if part.get("parts"):
        out: list[dict] = []
        for child in part["parts"]:
            out.extend(walk_parts(child))
        return out
    if body.get("data"):
        return [{**part, "_data": body["data"], "_size": body.get("size", 0)}]
    if mime.startswith("text/") and part.get("filename") is None:
        return []
    return []


class GmailConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="gmail",
        display_name="Gmail",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        base_url=API,
        docs_url="https://developers.google.com/gmail/api/reference/rest",
        scopes=(
            Scope(
                "https://www.googleapis.com/auth/gmail.readonly",
                "Read message bodies, headers and threads for search",
                "sensitive",
            ),
            Scope(
                "https://www.googleapis.com/auth/gmail.settings.basic",
                "Read labels and filters so threads group correctly",
                "normal",
            ),
        ),
        # Deliberately NOT requesting gmail.modify / gmail.send: this product
        # never writes to a mailbox (blueprint non-goal N1).
        realtime_webhooks=True,
        incremental_cursor=True,
        cursor_field="historyId",
        supports_tombstones=True,
        content_kinds=("email", "file"),
        rate_limit_per_sec=20.0,  # 15k requests/user/min quota
        rate_limit_burst=100,
        max_page_size=500,
        notes="Pub/Sub push for realtime; 60s history.list poll as fallback.",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        profile = self._request(grant, "GET", f"{API}/profile", limiter_key="profile")
        return [StreamDescriptor(key=profile.get("emailAddress", "me"),
                                 kind="mailbox",
                                 label=profile.get("emailAddress", "me"),
                                 meta={"messages_total": profile.get("messagesTotal"),
                                       "threads_total": profile.get("threadsTotal")})]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        if cursor:
            body = self._request(grant, "GET", f"{API}/history",
                                 params={"startHistoryId": cursor, "historyTypes": "messageAdded"},
                                 limiter_key="history.list")
            ids = [h["message"]["id"] for h in body.get("history", [])
                   if h.get("messagesAdded")]
            next_cursor = body.get("historyId") or cursor
        else:
            listing = self._request(grant, "GET", f"{API}/messages",
                                    params={"maxResults": 500, "includeSpamTrash": "false"},
                                    limiter_key="messages.list")
            ids = [m["id"] for m in listing.get("messages", [])]
            next_cursor = listing.get("nextPageToken")

        records: list[RawEnvelope] = []
        for chunk in _chunks(ids, 100):
            body = self._request(grant, "GET", f"{API}/messages/batchGet",
                                 params={"format": "full"}, json_body={"ids": chunk},
                                 limiter_key="messages.get")
            for msg in body.get("messages", []):
                records.append(RawEnvelope(
                    source_artifact_id=msg["id"], provider="gmail", stream=stream.key,
                    kind="email", payload=msg,
                    provider_meta={"threadId": msg.get("threadId"),
                                   "historyId": msg.get("historyId"),
                                   "labelIds": msg.get("labelIds", [])},
                    fetched_at=iso(), workspace_id="",
                ))
        return PullPage(records=records, next_cursor=next_cursor, has_more=False)

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        internal = p.get("internalDate")
        ts = iso(int(internal) / 1000) if internal else iso()
        headers = p.get("payload", {}).get("headers", [])
        subject = _header(p, "Subject")
        sender = _header(p, "From")
        to = _header(p, "To")
        cc = _header(p, "Cc")
        msg_id = _header(p, "Message-Id")
        in_reply_to = _header(p, "In-Reply-To")

        body_text, body_html = "", ""
        attachments: list[dict] = []
        for part in walk_parts(p.get("payload", {})):
            mime = part.get("mimeType", "")
            if mime == "text/plain" and not body_text:
                body_text = _b64url_decode(part["_data"])
            elif mime == "text/html" and not body_html:
                body_html = _b64url_decode(part["_data"])
            elif part.get("filename"):
                attachments.append({
                    "provider_file_id": f"{p.get('id')}:{part.get('partId')}",
                    "name": part.get("filename"),
                    "mime": mime,
                    "size_bytes": part.get("_size", 0),
                    "attachment_id": part.get("body", {}).get("attachmentId"),
                })

        sent_match = re.search(r"<([^>]+)>", sender)
        sender_addr = (sent_match.group(1) if sent_match else sender).strip()
        sender_name = sender.split("<")[0].strip().strip('"') if "<" in sender else sender

        record = new_message(
            provider="gmail",
            provider_msg_id=p.get("id", ""),
            conversation_id=raw.provider_meta.get("threadId") or p.get("id", ""),
            sender_ref=f"gmail:{sender_addr}",
            ts=ts,
            body=body_text or body_html or "",
            workspace=raw.workspace_id,
            sender_name=sender_name,
            reply_to_id=in_reply_to,
            flavour="plain",
            attachments=attachments,
            direction="outbound" if raw.provider_meta.get("labelIds") and
            "SENT" in (raw.provider_meta.get("labelIds") or []) else "inbound",
            extra={"subject": subject, "to": to, "cc": cc, "message_id_header": msg_id,
                   "labels": raw.provider_meta.get("labelIds", []),
                   "snippet": p.get("snippet", ""),
                   # The From header is the account's own address, so it is the
                   # one email in the body we can attribute with certainty. It
                   # is the token that lets Gmail and Slack accounts for the same
                   # human resolve to one Person.
                   "sender_email": sender_addr,
                   # A signature block is a heuristic, and the resolver scores it
                   # as evidence rather than as fact.
                   "signature": extract_signature(body_text or body_html or ""),},
        )
        record["kind"] = "email"
        out = [CanonicalRecord("email", record, raw.source_artifact_id, "gmail", raw.stream)]

        for att in attachments:
            f = new_file(provider="gmail", provider_file_id=att["provider_file_id"],
                         name=att["name"], mime=att["mime"], size_bytes=att["size_bytes"],
                         workspace=raw.workspace_id,
                         modified_ts=ts, owner_ref=record["sender_ref"],
                         extra={"message_id": record["_id"], "subject": subject,
                                "attachment_id": att.get("attachment_id")})
            f["_sent_by"] = record["sender_ref"]
            f["_thread_id"] = record["conversation_id"]
            out.append(CanonicalRecord("file", f, att["provider_file_id"], "gmail", raw.stream))
        return out


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i: i + size] for i in range(0, len(items), size)]
