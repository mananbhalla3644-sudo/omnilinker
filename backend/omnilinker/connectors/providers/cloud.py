"""Cloud storage connectors: Google Drive, OneDrive, Dropbox (blueprint 6.4 #6-8).

All three are "file" connectors, so they share one base class: the only real
differences are the endpoint, the cursor token (pageToken / deltaLink /
cursor) and the item shape. The normalize methods are genuinely per-provider
because the folder/share metadata differs, and that metadata is what powers
the unified file explorer's "shared with" and "where else does this live"
features.
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
from omnilinker.normalize import iso, new_file

from pathlib import Path

GDRIVE = "https://www.googleapis.com/drive/v3"
GRAPH = "https://graph.microsoft.com/v1.0"
DROPBOX = "https://api.dropboxapi.com/2"

#: Dropbox's list_folder entries carry no mime type, so derive it from the name.
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv": "text/csv", ".md": "text/markdown", ".txt": "text/plain",
    ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".zip": "application/zip", ".key": "application/octet-stream",
}


def _mime_for(name: str) -> str:
    return _MIME_BY_EXT.get(Path(name).suffix.lower(), "application/octet-stream")


class GoogleDriveConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="gdrive",
        display_name="Google Drive",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        base_url=GDRIVE,
        docs_url="https://developers.google.com/drive/api/reference/rest/v3",
        scopes=(
            Scope("https://www.googleapis.com/auth/drive.readonly",
                  "Enumerate and read file metadata to build the unified explorer", "sensitive"),
        ),
        realtime_webhooks=False,
        incremental_cursor=True,
        cursor_field="startPageToken",
        supports_tombstones=True,
        content_kinds=("file",),
        rate_limit_per_sec=100.0,  # 12k queries / 100s / user
        rate_limit_burst=50,
        max_page_size=1000,
        notes="change feed via files.list pageToken, polled at 60s.",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        about = self._request(grant, "GET", f"{GDRIVE}/about",
                              params={"fields": "storageQuota,user"}, limiter_key="about")
        return [StreamDescriptor(key="root", kind="drive", label="My Drive",
                                 meta={"quota": (about.get("storageQuota") or {}).get("limit")})]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        params: dict[str, Any] = {
            "pageSize": 1000,
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,createdTime,"
                      "parents,owners(emailAddress,displayName),shared,trashed,md5Checksum,"
                      "webViewLink,driveId,shortcutDetails,videoMediaMetadata)",
            "corpora": "user",
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
            "q": "trashed = false",
        }
        if cursor:
            params["pageToken"] = cursor
        body = self._request(grant, "GET", f"{GDRIVE}/files", params=params,
                             limiter_key="files.list")
        files = body.get("files", [])
        records = [RawEnvelope(source_artifact_id=f["id"], provider="gdrive", stream="root",
                               kind="file", payload=f,
                               provider_meta={"parents": f.get("parents", []),
                                              "shared": f.get("shared", False)},
                               fetched_at=iso(), workspace_id="")
                   for f in files]
        return PullPage(records=records, next_cursor=body.get("nextPageToken"),
                        has_more=bool(body.get("nextPageToken")))

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        owners = p.get("owners") or []
        owner_email = owners[0].get("emailAddress", "") if owners else ""
        owner = f"gdrive:{owner_email}" if owner_email else "gdrive:unknown"
        f = new_file(
            provider="gdrive", provider_file_id=p["id"], name=p.get("name", ""),
            mime=p.get("mimeType", "application/octet-stream"),
            size_bytes=int(p.get("size") or 0), workspace=raw.workspace_id,
            modified_ts=p.get("modifiedTime") or iso(),
            created_ts=p.get("createdTime") or iso(),
            owner_ref=owner,
            path=p.get("webViewLink", ""),
            folder_path=[p.get("parents", ["root"])[0]] if p.get("parents") else [],
            shared_with=[],
            checksum=(f"md5:{p['md5Checksum']}" if p.get("md5Checksum") else ""),
            extra={"shared": p.get("shared", False), "owner_email": owner_email,
                   "owner_name": owners[0].get("displayName", "") if owners else "",
                   "duration_s": (p.get("videoMediaMetadata") or {}).get("durationMillis", 0) // 1000},
        )
        return [CanonicalRecord("file", f, raw.source_artifact_id, "gdrive", raw.stream)]


class OneDriveConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="onedrive",
        display_name="OneDrive",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        refresh_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        base_url=GRAPH,
        docs_url="https://learn.microsoft.com/graph/api/resources/drive",
        scopes=(
            Scope("offline_access", "Refresh tokens for unattended sync", "normal"),
            Scope("Files.Read.All", "Enumerate and read file metadata", "sensitive"),
            Scope("User.Read", "Identify the signed-in account", "normal"),
        ),
        realtime_webhooks=True,
        incremental_cursor=True,
        cursor_field="deltaLink",
        supports_tombstones=True,
        content_kinds=("file",),
        rate_limit_per_sec=20.0,
        rate_limit_burst=50,
        max_page_size=200,
        notes="/me/drive/root/delta walks the whole tree in one call - the cheapest "
              "backfill of any connector.",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        me = self._request(grant, "GET", f"{GRAPH}/me",
                           params={"$select": "id,displayName,userPrincipalName"},
                           limiter_key="me")
        return [StreamDescriptor(key=me.get("id", "me"), kind="drive",
                                 label=me.get("displayName", "OneDrive"),
                                 meta={"upn": me.get("userPrincipalName", "")})]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        url = cursor or f"{GRAPH}/me/drive/root/delta"
        headers_extra = {"Prefer": "odata.maxpagesize=200"}
        params = {"$select": "id,name,size,lastModifiedDateTime,createdDateTime,parentReference,"
                             "file,mimeType,folder,@deleted,@removed,eTag,shared"}
        body = self._request(grant, "GET", url, params=params, limiter_key="delta")
        items = body.get("value", [])
        records = []
        for it in items:
            kind = "file"
            if it.get("folder"):
                kind = "folder"
            records.append(RawEnvelope(
                source_artifact_id=it["id"], provider="onedrive", stream=stream.key,
                kind=kind, payload=it,
                provider_meta={"deleted": bool(it.get("@deleted"))},
                fetched_at=iso(), workspace_id=""))
        return PullPage(records=records, next_cursor=body.get("@odata.deltaLink"),
                        has_more=bool(body.get("@odata.nextLink")))

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        if p.get("folder") or raw.kind == "folder":
            return []  # folders become graph Folder nodes in the pipeline
        f = new_file(
            provider="onedrive", provider_file_id=p["id"], name=p.get("name", ""),
            mime=(p.get("file") or {}).get("mimeType", "application/octet-stream"),
            size_bytes=int(p.get("size") or 0), workspace=raw.workspace_id,
            modified_ts=p.get("lastModifiedDateTime") or iso(),
            created_ts=p.get("createdDateTime") or iso(),
            owner_ref="onedrive:me",
            folder_path=[(p.get("parentReference") or {}).get("id", "")],
            extra={"deleted": raw.provider_meta.get("deleted", False),
                   "e_tag": p.get("eTag", "")},
        )
        if p.get("@deleted"):
            f["deleted"] = True
        return [CanonicalRecord("file", f, raw.source_artifact_id, "onedrive", raw.stream)]


class DropboxConnector(BaseRestConnector, Connector):
    descriptor = ConnectorDescriptor(
        id="dropbox",
        display_name="Dropbox",
        version="1.0.0",
        auth_flow="oauth2_code",
        authorize_url="https://www.dropbox.com/oauth2/authorize",
        token_url="https://api.dropboxapi.com/oauth2/token",
        base_url=DROPBOX,
        docs_url="https://www.dropbox.com/developers/documentation/http/documentation",
        scopes=(
            Scope("files.metadata.read", "List files and folders", "normal"),
            Scope("files.content.read", "Download file content for indexing", "sensitive"),
            Scope("account_info.read", "Identify the account", "normal"),
        ),
        realtime_webhooks=True,
        incremental_cursor=True,
        cursor_field="cursor",
        supports_tombstones=True,
        content_kinds=("file",),
        rate_limit_per_sec=4.0,
        rate_limit_burst=20,
        max_page_size=2000,
        notes="Webhook is a content-free ping; we continue the delta longpoll to "
              "discover what changed (blueprint 6.3).",
    )

    def discover_streams(self, grant: TokenGrant, user_ref: str | None = None) -> list[StreamDescriptor]:
        body = self._request(grant, "POST", f"{DROPBOX}/users/get_current_account",
                             json_body=None, limiter_key="account")
        return [StreamDescriptor(key=str(body.get("account_id", "account")), kind="account",
                                 label=body.get("email", "Dropbox"),
                                 meta={"name": body.get("name", {}).get("display_name", "")})]

    def pull(self, grant: TokenGrant, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        body = self._request(grant, "POST", f"{DROPBOX}/files/list_folder/continue"
                             if cursor else f"{DROPBOX}/files/list_folder",
                             json_body={"cursor": cursor} if cursor else
                             {"path": "", "recursive": True, "include_deleted": True,
                              "limit": 2000},
                             limiter_key="list_folder")
        entries = body.get("entries", [])
        records = [RawEnvelope(source_artifact_id=f"{e.get('tag')}:{e.get('id')}",
                               provider="dropbox", stream=stream.key,
                               kind="folder" if e.get(".tag") == "folder" else "file",
                               payload=e, provider_meta={"rev": e.get("rev")},
                               fetched_at=iso(), workspace_id="")
                   for e in entries]
        return PullPage(records=records, next_cursor=body.get("cursor"),
                        has_more=bool(body.get("has_more")))

    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        p: dict[str, Any] = raw.payload
        if p.get(".tag") == "folder":
            return []
        path = p.get("path_display", "")
        f = new_file(
            provider="dropbox", provider_file_id=p.get("id", ""),
            name=p.get("name", ""), size_bytes=int(p.get("size") or 0),
            workspace=raw.workspace_id, mime=_mime_for(p.get("name", "")),
            modified_ts=p.get("server_modified") or iso(),
            owner_ref="dropbox:me", path=path,
            folder_path=[seg for seg in path.split("/")[:-1] if seg],
            checksum=(f"dropbox:{p['rev']}" if p.get("rev") else ""),
            extra={"rev": p.get("rev", ""), "deleted": p.get(".tag") == "deleted"},
        )
        return [CanonicalRecord("file", f, raw.source_artifact_id, "dropbox", raw.stream)]
