"""Discord connector (blueprint 6.4 #2).

Cursor: snowflake ids, which are monotonic, so `after=<last_snowflake>` is a
correct incremental key. Limit is 100 messages per request.

Policy note, and it is a hard one: we use bot tokens for guilds the user has
explicitly added, or the user's own OAuth grant for self-authored DMs. We do
NOT implement self-bots. That is a ToS boundary, not a technical preference.
"""

from __future__ import annotations

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
from omnilinker.normalize import iso, new_identity, new_message

API = "https://discord.com/api/v10"


def snowflake_ts(snowflake: str) -> float:
    """Discord snowflake: (ms since 2015-01-01) << 22."""
    try:
        return (int(snowflake) >> 22) + 1_420_070_400_000
    except (TypeError, ValueError):
        return 0.0


class DiscordConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="discord",
        display_name="Discord",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://discord.com/oauth2/authorize",
        token_url="https://discord.com/api/v10/oauth2/token",
        base_url=API,
        docs_url="https://discord.com/developers/docs/reference",
        scopes=(
            Scope("identify", "Resolve the account behind the grant", "normal"),
            Scope("guilds", "Enumerate guilds and their channels", "normal"),
            Scope("messages.read", "Read message text for search and graph", "sensitive"),
        ),
        realtime_webhooks=True,  # Gateway with resume
        incremental_cursor=True,
        cursor_field="snowflake",
        supports_reactions=True,
        supports_edits=False,
        content_kinds=("message", "identity"),
        rate_limit_per_sec=45.0,  # global 50/min -> conservative per-second avg
        rate_limit_burst=20,
        max_page_size=100,
        backfill_strategy="reconcile",
        notes="Gateway WebSocket for realtime; REST backfill with snowflake cursors. "
              "No self-bots (ToS). The messages endpoint returns no deletion tombstone, "
              "so a nightly reconcile sweep diffs known ids against the live channel.",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        me = self._request(grant, "GET", f"{API}/users/@me", limiter_key="users")
        streams = [StreamDescriptor(key=str(me.get("id")), kind="dm_root",
                                    label=me.get("username", "me"),
                                    meta={"is_self": True})]
        guilds = self._request(grant, "GET", f"{API}/users/@me/guilds",
                               limiter_key="guilds")
        # Both Discord list endpoints return bare JSON arrays, which _request
        # passes through unchanged, so normalise defensively to a list.
        for guild in (guilds if isinstance(guilds, list) else []):
            channels = self._request(grant, "GET", f"{API}/guilds/{guild['id']}/channels",
                                     limiter_key="channels")
            for ch in (channels if isinstance(channels, list) else []):
                if ch.get("type") in (0, 5):  # GUILD_TEXT, GUILD_ANNOUNCEMENT
                    streams.append(StreamDescriptor(
                        key=ch["id"], kind="channel", label=ch.get("name", ch["id"]),
                        meta={"guild_id": guild["id"], "guild_name": guild.get("name"),
                              "channel_type": ch.get("type")},
                    ))
        return streams

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        params: dict[str, Any] = {"limit": 100}
        if cursor:
            params["after"] = cursor
        if stream.kind == "channel":
            url = f"{API}/channels/{stream.key}/messages"
        else:
            url = f"{API}/channels/{stream.key}/messages"  # DM channel
        body = self._request(grant, "GET", url, params=params, limiter_key="messages")
        messages = body if isinstance(body, list) else []
        records = [
            RawEnvelope(
                source_artifact_id=m["id"], provider="discord", stream=stream.key,
                kind="message",
                payload={**m, "_stream_label": stream.label,
                         "_guild_name": stream.meta.get("guild_name", "")},
                provider_meta={"edited": m.get("edited_timestamp")},
                fetched_at=iso(), workspace_id="",
            )
            for m in messages or []
            if m.get("type") == 0 and not m.get("author", {}).get("bot")
        ]
        newest = max((m["id"] for m in messages or []), key=lambda s: int(s), default=None)
        return PullPage(records=records, next_cursor=newest or cursor, has_more=bool(messages))

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        author = p.get("author", {})
        attachments = [{
            "provider_file_id": a.get("id"),
            "name": a.get("filename") or "attachment",
            "mime": a.get("content_type", "application/octet-stream"),
            "size_bytes": a.get("size", 0),
            "url": a.get("url"),
        } for a in p.get("attachments", [])]

        reactions: dict[str, dict] = {}
        for r in p.get("reactions", []):
            e = reactions.setdefault(r.get("name", "?"), {"emoji": r.get("name"), "count": 0,
                                                         "users": []})
            e["count"] += 1
            e["users"].append(r.get("user_id"))

        msg = new_message(
            provider="discord",
            provider_msg_id=p["id"],
            conversation_id=raw.stream,
            sender_ref=f"discord:{author.get('id')}",
            ts=iso(snowflake_ts(p["id"]) / 1000),
            body=p.get("content", ""),
            workspace=raw.workspace_id,
            sender_name=(author.get("global_name") or author.get("username") or ""),
            flavour="markdown",
            attachments=attachments,
            reactions=list(reactions.values()),
            direction="inbound",
            extra={"guild": p.get("_guild_name", ""), "channel": p.get("_stream_label", ""),
                   "edited": bool(raw.provider_meta.get("edited"))},
        )
        msg["reply_to_id"] = (p.get("message_reference") or {}).get("message_id", "")
        return [CanonicalRecord("message", msg, raw.source_artifact_id, "discord", raw.stream)]
