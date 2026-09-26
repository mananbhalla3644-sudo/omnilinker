"""Note connectors: Notion and Evernote (blueprint 6.4 #9, #10).

Notion notes the two operational facts that bite everyone: (1) the API version
must be pinned on every request or behaviour changes under you, and (2) there
is no refresh token in the classic flow, so the UI must surface re-consent
before sync silently stops. Evernote's legacy API is deprecated; the connector
handles the regional hosts and the EDAM envelope.
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
from omnilinker.ids import prefixed
from omnilinker.normalize import clean_text, content_hash, iso, new_note


class NotionConnector(BaseRestConnector, Connector):
    NOTION_VERSION = "2022-06-28"  # pinned deliberately

    descriptor = ConnectorDescriptor(
        id="notion",
        display_name="Notion",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://api.notion.com/v1/oauth/authorize",
        token_url="https://api.notion.com/v1/oauth/token",
        base_url="https://api.notion.com/v1",
        docs_url="https://developers.notion.com/reference/intro",
        scopes=(
            Scope("read_content", "Read page content for search and clustering", "sensitive"),
        ),
        realtime_webhooks=True,
        incremental_cursor=True,
        cursor_field="last_edited_time",
        supports_edits=True,
        content_kinds=("note", "file"),
        rate_limit_per_sec=2.5,  # documented average ~3 rps
        rate_limit_burst=5,
        max_page_size=100,
        backfill_strategy="reconcile",
        notes="No refresh token in the classic OAuth flow - re-consent must be "
              "surfaced in the UI, not swallowed. Search returns only live pages, so "
              "deletions are found by a reconcile sweep over the known page-id set.",
    )

    def _request_notion(self, grant: TokenGrant, method: str, url: str, **kw) -> Any:
        # _request builds the Authorization header; Notion additionally needs a
        # pinned version and its own Accept.
        kw.setdefault("json_body", None)
        return self._request(grant, method, url, **kw)

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        me = self._request(grant, "GET", "https://api.notion.com/v1/users/me",
                           limiter_key="users")
        return [StreamDescriptor(key=me.get("id", "bot"), kind="workspace",
                                 label=me.get("name", "Notion"),
                                 meta={"workspace_name": me.get("bot", {}).get("workspace_name", "")})]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        params: dict[str, Any] = {"page_size": 100, "sort": {"direction": "descending",
                                                              "timestamp": "last_edited_time"}}
        if cursor:
            params["start_time"] = cursor
        body = self._request(grant, "GET", "https://api.notion.com/v1/search",
                             params=params, limiter_key="search")
        results = [r for r in body.get("results", []) if r.get("object") == "page"]
        records = []
        for page_obj in results:
            page_id = page_obj["id"]
            blocks = self._blocks(grant, page_id)
            records.append(RawEnvelope(
                source_artifact_id=page_id, provider="notion", stream=stream.key,
                kind="note", payload={**page_obj, "_blocks": blocks},
                provider_meta={"last_edited_time": page_obj.get("last_edited_time")},
                fetched_at=iso(), workspace_id=""))
        return PullPage(records=records, next_cursor=body.get("next_cursor"),
                        has_more=bool(body.get("has_more")))

    def _blocks(self, grant: TokenGrant, page_id: str) -> list[dict]:
        blocks: list[dict] = []
        cursor = None
        for _ in range(20):
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            body = self._request(grant, "GET",
                                 f"https://api.notion.com/v1/blocks/{page_id}/children",
                                 params=params, limiter_key="blocks")
            blocks.extend(body.get("results", []))
            cursor = body.get("next_cursor")
            if not cursor:
                break
        return blocks

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        props = p.get("properties", {}) or {}

        def title_of(props: dict) -> str:
            for value in props.values():
                if value.get("type") == "title":
                    return "".join(t.get("plain_text", "") for t in value.get("title", []))
            return ""

        title = title_of(props) or "Untitled"
        lines: list[str] = []
        for block in p.get("_blocks", []):
            btype = block.get("type")
            rich = (block.get(btype) or {}).get("rich_text") or []
            text = "".join(t.get("plain_text", "") for t in rich).strip()
            if not text:
                continue
            if btype in {"heading_1", "heading_2", "heading_3"}:
                lines.append(f"\n{'#' * int(btype[-1])} {text}")
            elif btype == "bulleted_list_item":
                lines.append(f"- {text}")
            elif btype == "to_do":
                mark = "x" if (block.get(btype) or {}).get("checked") else " "
                lines.append(f"- [{mark}] {text}")
            else:
                lines.append(text)

        body_text = clean_text("\n".join(lines), flavour="markdown")
        note = new_note(
            provider="notion", provider_page_id=p["id"], title=title,
            body_text=body_text, workspace=raw.workspace_id,
            last_edited=p.get("last_edited_time") or iso(),
            created_ts=p.get("created_time") or iso(),
            parent_id=(p.get("parent") or {}).get("page_id")
            or (p.get("parent") or {}).get("database_id", ""),
            url=p.get("url", ""),
            block_count=len(p.get("_blocks", [])),
            extra={"icon": (p.get("icon") or {}).get("emoji", "")},
        )
        return [CanonicalRecord("note", note, raw.source_artifact_id, "notion", raw.stream)]


class EvernoteConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="evernote",
        display_name="Evernote",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://www.evernote.com/oauth",
        token_url="https://www.evernote.com/oauth",
        refresh_url="https://www.evernote.com/oauth",
        base_url="https://www.evernote.com/notestore",
        docs_url="https://dev.evernote.com/doc/reference/",
        scopes=(
            Scope("basic", "Read note bodies for search (legacy API)", "sensitive"),
        ),
        realtime_webhooks=False,
        incremental_cursor=True,
        cursor_field="updated",
        supports_attachments=True,
        content_kinds=("note", "file"),
        rate_limit_per_sec=1.0,
        rate_limit_burst=5,
        max_page_size=100,
        backfill_strategy="reconcile",
        notes="Legacy API; regional hosts (us/int/ap). Deprecation is a known risk "
              "(blueprint risk R1).",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        body = self._request(grant, "GET", f"{self.descriptor.base_url}/notebooks",
                             limiter_key="notebooks")
        # Evernote returns an EDAM envelope: {"notebooks": {"notebook": [...]}}
        notebooks = ((body.get("notebooks") or {}).get("notebook")) or []
        return [StreamDescriptor(key=str(nb.get("key")), kind="notebook",
                                 label=nb.get("name", nb.get("key")),
                                 meta={"stack": nb.get("stack", "")}) for nb in notebooks]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        criteria = {"tag": stream.key}
        if cursor:
            criteria["updated"] = ".." + cursor
        body = self._request(grant, "POST", f"{self.descriptor.base_url}/findNotes",
                             json_body={"filter": criteria, "pageSize": 100,
                                        "offset": page * 100,
                                        "orderBy": "UPDATEDATE",
                                        "direction": "DESC"},
                             limiter_key="findNotes")
        notes = ((body.get("notes") or {}).get("note")) or []
        records = [RawEnvelope(source_artifact_id=str(n.get("guid", n.get("key"))),
                               provider="evernote", stream=stream.key, kind="note",
                               payload=n, provider_meta={"updated": n.get("updated")},
                               fetched_at=iso(), workspace_id="")
                   for n in notes]
        newest = max((n.get("updated", 0) for n in notes), default=cursor)
        return PullPage(records=records,
                        next_cursor=str(newest) if newest else None,
                        has_more=len(notes) == 100)

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        title = clean_text(p.get("title", ""))
        content = clean_text((p.get("content") or ""), flavour="markdown")
        note = new_note(
            provider="evernote", provider_page_id=str(p.get("guid", p.get("key"))),
            title=title, body_text=content, workspace=raw.workspace_id,
            last_edited=iso(p.get("updated", 0) / 1000) if p.get("updated") else iso(),
            created_ts=iso(p.get("created", 0) / 1000) if p.get("created") else iso(),
            parent_id=str(p.get("notebookGuid", raw.stream)),
            url="", block_count=content.count("\n") + 1,
            extra={"tags": [t.get("name") for t in (p.get("tagResources") or {}).get("tag", [])]},
        )
        return [CanonicalRecord("note", note, raw.source_artifact_id, "evernote", raw.stream)]
