"""Slack connector (blueprint 6.4 #3).

Read-only scopes only, each with a written justification - the connector
registry refuses any scope without one (6.2.1 REGISTER static checks).

Streams: one per channel/DM. Cursor: the newest `ts` seen in that channel.
Slack's `ts` is a monotonic "sec.microsec" string, so incremental sync is a
single `conversations.history?oldest=<cursor>` call.
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
from omnilinker.normalize import iso, new_file, new_identity, new_message

API = "https://slack.com/api"
# Slack's *authorization* endpoint is a browser-facing page, not a Web API
# method, so it does not live under /api and is not a dotted method name:
#     correct: https://slack.com/oauth/v2/authorize
#     wrong:   https://slack.com/api/oauth.v2/authorize
# The wrong one answers a browser GET with an opaque "There's been a glitch…"
# page, so a Connect button pointing at it looks completely dead - no error, no
# redirect, nothing in our own logs, because the request never came back to us.
# Only the token exchange is a real Web API method (`/api/oauth.v2.access`).
SLACK_AUTHORIZE = "https://slack.com/oauth/v2/authorize"


class SlackConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="slack",
        display_name="Slack",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url=SLACK_AUTHORIZE,
        token_url=f"{API}/oauth.v2.access",
        base_url=API,
        docs_url="https://api.slack.com/methods",
        scopes=(
            # `users:read.email` is the scope the whole product rests on and it
            # is easy to miss: without it Slack returns no address, `sender_email`
            # is empty, and a Slack account can never resolve to the same Person
            # as the person's Gmail account. Cross-source identity resolution
            # then silently degrades to "N separate humans with the same first
            # name" and nothing says so.
            Scope("users:read.email",
                  "Link a Slack account to the same person's other accounts",
                  "sensitive"),
            Scope("channels:history", "Read message text for search and graph edges", "normal"),
            Scope("groups:history", "Private channels the user belongs to", "sensitive"),
            Scope("im:history", "Direct messages, where most 1:1 context lives", "normal"),
            Scope("mpim:history", "Group DMs", "normal"),
            Scope("channels:read", "Enumerate channels to build the stream list", "normal"),
            Scope("groups:read", "Enumerate private channels", "sensitive"),
            Scope("users:read", "Resolve user ids to names for entity resolution", "normal"),
            Scope("files:read", "Index shared files so Drive/chat links resolve", "normal"),
            Scope("reactions:read", "Reactions are a cheap importance signal", "normal"),
        ),
        realtime_webhooks=True,
        incremental_cursor=True,
        cursor_field="ts",
        supports_reactions=True,
        content_kinds=("message", "file", "identity", "membership"),
        rate_limit_per_sec=1.0,  # tier-dependent; 1/s per method is the safe floor
        rate_limit_burst=5,
        max_page_size=200,
        notes="Events API (HTTP request mode) is the realtime path; reconcile sweeps cover gaps.",
    )

    # -- streams ------------------------------------------------------
    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        streams: list[StreamDescriptor] = []
        for kind, method in (("channel", "users.conversations"),
                            ("group", "users.conversations"),
                            ("im", "users.conversations"),
                            ("mpim", "users.conversations")):
            types = {"channel": "public_channel", "group": "private_channel",
                     "im": "im", "mpim": "mpim"}[kind]
            cursor = None
            while True:
                body = self._request(grant, "GET", f"{API}/users.conversations",
                                     params={"types": types, "limit": 200, "cursor": cursor or "",
                                             "exclude_archived": "true"},
                                     limiter_key=f"users.conversations")
                for ch in body.get("channels", []):
                    streams.append(StreamDescriptor(
                        key=ch["id"], kind="conversation",
                        label=ch.get("name") or ch.get("name_normalized") or ch["id"],
                        meta={"channel_type": kind, "is_private": ch.get("is_private", False),
                              "topic": (ch.get("purpose") or {}).get("value", "")},
                    ))
                cursor = (body.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        return streams

    # -- pull ---------------------------------------------------------
    #: Slack caps `users.info` at Tier 3 (roughly 50 requests/minute), so the
    #: profile cache is bounded and every lookup goes through the shared
    #: rate limiter. Unbounded, a 500k-message workspace turns a sync into tens
    #: of thousands of user lookups and gets the app rate-limited.
    _profile_cache: dict[str, dict] = {}
    _profile_cache_cap = 800

    def _hydrate_users(self, grant: TokenGrant, messages: list[dict]) -> None:
        """Attach `users.info` profiles to the raw messages.

        Cached per process and capped, because the answer is per-user and
        repeated across every channel they post in. A miss falls back to the
        raw id, which is honest: the message still appears, attributed to
        `U01ALICE`, and the identity resolver treats an opaque id as weak
        evidence rather than inventing a name.
        """
        wanted = {
            str(m["user"]) for m in messages
            if m.get("user") and str(m["user"]) not in self._profile_cache
        }
        budget = self._profile_cache_cap - len(self._profile_cache)
        for user_id in sorted(wanted)[:max(0, budget)]:
            try:
                body = self._request(
                    grant, "GET", f"{API}/users.info",
                    params={"user": user_id},
                    limiter_key="users.info",
                )
            except Exception:
                # One unavailable profile must not fail the page of messages.
                continue
            if not body.get("ok"):
                continue
            user = body.get("user") or {}
            profile = user.get("profile") or {}
            self._profile_cache[user_id] = {
                "id": user_id,
                "name": user.get("name", ""),
                "real_name": profile.get("real_name") or user.get("real_name", ""),
                "email": profile.get("email", ""),
                "phone": profile.get("phone", ""),
                "is_bot": bool(user.get("is_bot")),
            }

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        params: dict[str, Any] = {"channel": stream.key, "limit": 200, "inclusive": "true"}
        if cursor:
            # Slack requires a float-looking ts; our cursor is already that.
            params["oldest"] = cursor
        body = self._request(grant, "GET", f"{API}/conversations.history",
                             params=params, limiter_key="conversations.history")
        if not body.get("ok"):
            return PullPage(records=[], has_more=False,
                            partial_failure={"reason": body.get("error", "unknown"),
                                             "retryable": body.get("error") in {"ratelimited", "internal_error"}})
        messages = body.get("messages", [])
        self._hydrate_users(grant, messages)
        records = [
            RawEnvelope(
                source_artifact_id=m["ts"],
                provider="slack",
                stream=stream.key,
                kind="message",
                payload={**m,
                         "_user_profile": self._profile_cache.get(str(m.get("user")), {}),
                         "_is_self": str(m.get("user")) == self._self_user_id(grant),
                         "_stream_label": stream.label,
                         "_channel_type": stream.meta.get("channel_type", "channel")},
                provider_meta={"ts": m.get("ts"), "thread_ts": m.get("thread_ts")},
                fetched_at=iso(),
            )
            for m in messages
            if m.get("type") == "message" and not m.get("subtype") in {"channel_join",
                                                                        "channel_leave", "bot_message"}
        ]
        newest = max((m.get("ts", "0") for m in messages), default=None)
        return PullPage(records=records, next_cursor=newest or cursor,
                        has_more=bool(body.get("has_more")))

    def _self_user_id(self, grant: TokenGrant) -> str:
        """The account that authorized this connection.

        "Sent by you" and "mentions you" drive the importance score, and they
        need to know which user id is the workspace owner rather than assuming
        the first message seen is theirs. Cached per grant, one `auth.test` call
        for the whole sync.
        """
        cached = getattr(grant, "extra", {}).get("_self_user_id")
        if cached:
            return str(cached)
        try:
            body = self._request(grant, "GET", f"{API}/auth.test",
                                 limiter_key="auth.test")
        except Exception:
            return ""
        user_id = str(body.get("user_id") or "") if body.get("ok") else ""
        if user_id:
            try:
                grant.extra["_self_user_id"] = user_id
            except Exception:
                pass
        return user_id

    # -- normalize ----------------------------------------------------
    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        out: list[CanonicalRecord] = []

        # user cache from the message payload
        user_id = p.get("user") or p.get("bot_id") or "unknown"
        user_profile = p.get("_user_profile") or {}

        attachments = []
        for f in p.get("files", []) or []:
            attachments.append({
                "provider_file_id": f.get("id"),
                "name": f.get("name") or f.get("title") or "file",
                "mime": f.get("mimetype", "application/octet-stream"),
                "size_bytes": f.get("size", 0),
                "url_private": f.get("url_private_download") or f.get("url_private"),
                "permalink": f.get("permalink"),
            })

        reactions = [
            {"emoji": r.get("name"), "count": r.get("count", 0),
             "users": r.get("users", [])[:20]}
            for r in (p.get("reactions") or [])
        ]

        msg = new_message(
            provider="slack",
            provider_msg_id=p["ts"],
            conversation_id=raw.stream,
            sender_ref=f"slack:{user_id}",
            ts=iso(float(p["ts"])),
            body=p.get("text", ""),
            workspace=raw.workspace_id,
            sender_name=user_profile.get("real_name") or user_profile.get("name") or user_id,
            thread_root_id=p.get("thread_ts", ""),
            reply_to_id=p.get("thread_ts", "") if p.get("reply_count") else "",
            flavour="slack",
            attachments=attachments,
            reactions=reactions,
            direction="outbound" if p.get("_is_self") else "inbound",
            extra={"channel_type": p.get("_channel_type"),
                   "channel_label": p.get("_stream_label"),
                   "subtype": p.get("subtype", ""),
                   # Slack's user profile carries the workspace email. Emitting
                   # it is what lets a Slack account and a Gmail account
                   # resolve to the same person: without a shared identifier
                   # the only thing linking them is a first name, which is not
                   # evidence. `users.info` returns this field unless the user
                   # restricted it, in which case it is simply absent.
                   "sender_email": user_profile.get("email", ""),
                   "sender_phone": user_profile.get("phone", "")},
        )
        # threads: link replies to their root so the graph gets THREAD_ROOT
        if p.get("thread_ts") and p.get("thread_ts") != p["ts"]:
            msg["reply_to_id"] = p["thread_ts"]
        out.append(CanonicalRecord("message", msg, raw.source_artifact_id, "slack", raw.stream))

        if attachments:
            for att in attachments:
                f = new_file(
                    provider="slack",
                    provider_file_id=att["provider_file_id"],
                    name=att["name"], mime=att["mime"], size_bytes=att["size_bytes"],
                    workspace=raw.workspace_id,
                    modified_ts=msg["ts"], owner_ref=msg["sender_ref"],
                    path=att.get("permalink", ""), extra={"conversation_id": raw.stream},
                )
                f["_conversation_id"] = raw.stream
                f["_sent_by"] = msg["sender_ref"]
                out.append(CanonicalRecord("file", f, att["provider_file_id"], "slack", raw.stream))
        return out
