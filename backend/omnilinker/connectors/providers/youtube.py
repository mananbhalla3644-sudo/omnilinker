"""YouTube connector (blueprint 6.4 #11).

The valuable payload is the transcript, not the metadata. So the priority is
captions: `captions.list` -> `captions.download` (SRT/VTT) -> parsed into
30-second segments that are timestamp-indexed and embedded like any other
chunk, so "find that part about negotiation" resolves to a timecode.

Quota discipline: `search.list` costs 100 units while `playlistItems.list`
costs 1, so the walk is ordered to spend quota on the cheap call. Default
daily quota is 10k units; the limiter plus this ordering keeps a heavy user
well inside it.
"""

from __future__ import annotations

import re
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
from omnilinker.ids import prefixed
from omnilinker.normalize import content_hash, iso

API = "https://www.googleapis.com/youtube/v3"
SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(text: str, video_id: str, language: str = "en") -> dict:
    """SRT/VTT -> {segments:[{idx,start_ms,end_ms,text,start_s}]}."""
    segments: list[dict] = []
    for block in re.split(r"\n\s*\n", text or ""):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = None
        text_start = 0
        for idx, line in enumerate(lines[:2]):
            m = SRT_TIME.search(line)
            if m:
                text_start = idx + 1
                break
        if not m or text_start >= len(lines):
            continue
        g = [int(x) for x in m.groups()]
        start_ms = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end_ms = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        body = " ".join(lines[text_start:])
        if not body:
            continue
        segments.append({
            "idx": len(segments),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_s": round(start_ms / 1000, 1),
            "text": body,
        })
    joined = " ".join(s["text"] for s in segments)
    return {
        "kind": "transcript",
        "video_id": video_id,
        "language": language,
        "seg_count": len(segments),
        "text": joined,
        "text_hash": content_hash(joined),
        "segments": segments,
    }


class YouTubeConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="youtube",
        display_name="YouTube",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        base_url=API,
        docs_url="https://developers.google.com/youtube/v3/docs",
        scopes=(
            Scope("https://www.googleapis.com/auth/youtube.readonly",
                  "Read subscriptions, playlists and watch history", "sensitive"),
        ),
        realtime_webhooks=False,
        incremental_cursor=True,
        cursor_field="nextPageToken",
        content_kinds=("video", "transcript"),
        rate_limit_per_sec=2.0,
        rate_limit_burst=5,
        max_page_size=50,
        backfill_strategy="reconcile",
        notes="No realtime API. Transcripts are the payload; search.list (100 quota "
              "units) is used last. Videos removed from a playlist leave no tombstone, "
              "so removals come from a reconcile diff of playlist membership.",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        streams: list[StreamDescriptor] = []
        # Cheap call first (1 unit).
        subs = self._request(grant, "GET", f"{API}/subscriptions",
                             params={"part": "snippet,contentDetails", "maxResults": 50,
                                     "order": "alphabetical"},
                             limiter_key="subscriptions")
        for item in subs.get("items", []):
            snippet = item.get("snippet", {})
            cd = item.get("contentDetails", {})
            streams.append(StreamDescriptor(
                key=cd.get("playlistId", ""), kind="playlist",
                label=snippet.get("title", "subscription"),
                meta={"channel_title": snippet.get("channelTitle", ""),
                      "position": snippet.get("position", 0)},
            ))
        # Expensive call, deliberately last.
        history = self._request(grant, "GET", f"{API}/search",
                                params={"part": "snippet", "type": "video",
                                        "eventType": "watch", "maxResults": 50},
                                limiter_key="search.list")
        if history.get("items"):
            streams.append(StreamDescriptor(key="watch_history", kind="history",
                                            label="Watch history",
                                            meta={"count": len(history["items"])}))
        return streams

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> list[PullPage] | PullPage:
        if stream.kind == "history":
            return self._pull_history(grant, stream, cursor)
        params: dict[str, Any] = {"part": "snippet,contentDetails", "maxResults": 50,
                                  "playlistId": stream.key}
        if cursor:
            params["pageToken"] = cursor
        body = self._request(grant, "GET", f"{API}/playlistItems", params=params,
                             limiter_key="playlistItems")
        videos = [i for i in body.get("items", []) if i.get("snippet", {}).get("resourceId")]
        records = [RawEnvelope(
            source_artifact_id=i["snippet"]["resourceId"]["videoId"],
            provider="youtube", stream=stream.key, kind="video",
            payload={**i, "_playlist_label": stream.label},
            provider_meta={"position": i.get("snippet", {}).get("position", 0)},
            fetched_at=iso(), workspace_id="") for i in videos]
        return PullPage(records=records, next_cursor=body.get("nextPageToken"),
                        has_more=bool(body.get("nextPageToken")))

    def _pull_history(self, grant: TokenGrant, stream: StreamDescriptor,
                      cursor: str | None) -> PullPage:
        body = self._request(grant, "GET", f"{API}/search",
                             params={"part": "snippet", "type": "video", "eventType": "watch",
                                     "maxResults": 50, "pageToken": cursor or ""},
                             limiter_key="search.list")
        records = [RawEnvelope(source_artifact_id=i["id"]["videoId"], provider="youtube",
                               stream="watch_history", kind="video", payload=i,
                               fetched_at=iso(), workspace_id="")
                   for i in body.get("items", [])]
        return PullPage(records=records, next_cursor=body.get("nextPageToken"),
                        has_more=bool(body.get("nextPageToken")))

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        snippet = p.get("snippet", p if "title" in p else {})
        video_id = p.get("id", {}).get("videoId") if isinstance(p.get("id"), dict) \
            else raw.source_artifact_id
        video_id = video_id or raw.source_artifact_id
        duration = _parse_duration(snippet.get("duration", ""))
        video = {
            "kind": "video",
                "workspace_id": raw.workspace_id,
            "provider": "youtube",
            "provider_video_id": video_id,
            "title": snippet.get("title", ""),
            "description": (snippet.get("description") or "")[:2000],
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_ts": snippet.get("publishedAt") or iso(),
            "duration_s": duration,
            "tags": snippet.get("tags", []),
            "playlist_label": p.get("_playlist_label", ""),
            "thumbnails": list((snippet.get("thumbnails") or {}).keys()),
            "entities": {"person_ids": [], "urls": [], "file_refs": []},
        }
        out = [CanonicalRecord("video", video, raw.source_artifact_id, "youtube", raw.stream)]

        # A caption track may already be attached (the demo/import path does
        # this); the live path fetches it in `attach_transcript`.
        caption = p.get("_caption_srt")
        if caption:
            tr = parse_srt(caption, video_id, p.get("_caption_lang", "en"))
            tr["workspace_id"] = raw.workspace_id
            # Link by the provider's own video id, not by a locally-minted one.
            # A local id is regenerated on every normalize() call, so the edge
            # would point at a different node after each sync.
            tr["_video_id"] = video.get("provider_video_id", "")
            # A transcript needs its own document key. Reusing `video_id` gave
            # it the same id as the video it belongs to, and since both become
            # graph nodes the video node silently won: the transcript node was
            # dropped by the projector and HAS_TRANSCRIPT became a self-loop.
            tr["provider_transcript_id"] = f"{video_id}:{p.get('_caption_lang', 'en')}"
            tr["title"] = video["title"]
            out.append(CanonicalRecord("transcript", tr, tr["provider_transcript_id"],
                                       "youtube", raw.stream))
        return out


def _parse_duration(raw: str) -> int:
    if not raw:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", raw)
    if not m:
        return 0
    h, mm, ss = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mm * 60 + ss
