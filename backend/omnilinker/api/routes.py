"""HTTP API (blueprint section 12).

Route surface, and the reason for each group:

    /api/system/*      health, capabilities, and the recorded deviations.
                        A system that quietly differs from its spec is worse
                        than one that is loudly different.
    /api/connectors    what can be connected, and what each one can do.
    /api/sync          ingest + recompute every derived store.
    /api/search        search, with facets, filters and the parsed query echoed
                        back so the UI can show what it did.
    /api/nl2query      question -> plan -> result, plan included.
    /api/graph/*       ego, path, and neighbourhood traversal.
    /api/timeline      chronological view across sources.
    /api/files         file inventory.
    /api/documents/*   one document, decrypted on read.
    /api/persons/*     resolved people, and the evidence for each merge.
    /api/insights      detector output; accept/dismiss is the only state
                        transition, and it is user-action-only.
    /api/predictions   deadlines, unanswered questions, resurfacing.
    /api/passport      the data-passport view: what is held, where, sealed how.

Two conventions worth knowing before reading the code:

**Decrypted on read, never in bulk.** `GET /api/documents/{id}` decrypts that
one document. There is no endpoint that returns the corpus in the clear,
because the first time someone adds one, the system has a plaintext export
endpoint wearing a search UI.

**Every derived field says where it came from.** Search results carry their
scores and the parsed query; suggestions carry their evidence; predictions
carry the sentence they were extracted from. A user can always ask "why am I
looking at this" and get an answer, which is the difference between a system
they trust and a system they tolerate.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query as Q
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from omnilinker.ai.predict import run_predictions
from omnilinker.config import get_settings
from omnilinker.connectors.connections import (
    ConnectionError_,
    begin_authorization,
    complete_authorization,
    credentials_report,
    describe,
    disconnect,
    diagnose_credentials,
    get_grant,
    list_connections,
    verify,
)
from omnilinker.connectors.registry import get_registry
from omnilinker.engine import CONTENT_COLLECTIONS, get_engine
from omnilinker.ids import prefixed
from omnilinker.normalize import iso
from omnilinker.search import NL2QueryCompiler, SearchRequest, get_search_service
from omnilinker.store import flush_stores
from omnilinker.store.base import Filter, Node, Query

router = APIRouter(prefix="/api")

DEFAULT_LIMIT = 20
MAX_LIMIT = 200


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SyncRequest(BaseModel):
    connector_id: str = Field("demo", description="Connector id, e.g. 'demo', 'slack'")
    stream: str | None = Field(None, description="Limit to one stream key")
    max_streams: int | None = Field(None, ge=1, le=200)
    rebuild_graph: bool = True


class NL2QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    execute: bool = True


class AcceptInsightRequest(BaseModel):
    source_id: str
    target_id: str
    kind: str = "unknown"


class _Meta(BaseModel):
    took_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine():
    return get_engine()


def _limit(value: int | None, default: int = DEFAULT_LIMIT) -> int:
    return max(1, min(MAX_LIMIT, value or default))


def _ok(**payload: Any) -> dict:
    return {"ok": True, **payload}


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@router.get("/system/info", summary="Capabilities, store stats and recorded deviations")
def system_info() -> dict:
    return _ok(**_engine().system_info())


@router.get("/system/health", summary="Liveness plus per-stage engine state")
def health() -> JSONResponse:
    engine = _engine()
    state = engine.state
    healthy = state.stage != "error" and not state.error
    body = {
        "ok": healthy,
        "status": "ok" if healthy else "degraded",
        "stage": state.stage,
        "last_run_at": state.last_run_at,
        "last_run_ms": state.last_run_ms,
        "stages": state.stages,
        "error": state.error,
        "mode": get_settings().mode,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@router.get("/passport", summary="Data passport: what is held, from where, sealed how")
def passport() -> dict:
    """The blueprint's transparency surface (10.4).

    Answers "what do you have on me, where did it come from, and can I prove
    it has not been written to?" from the data itself rather than from config,
    so the answer cannot drift from reality.
    """
    engine = _engine()
    docs = engine.stores.docs
    by_provider: dict[str, dict[str, int]] = {}
    for collection in CONTENT_COLLECTIONS:
        for row in docs.find(Query(collection, limit=200_000)):
            provider = str(row.get("provider") or "unknown")
            entry = by_provider.setdefault(provider, {})
            entry[collection] = entry.get(collection, 0) + 1

    identity_count = docs.count("identities")
    sealed = docs.count("sealed_fields")
    return _ok(
        passport={
            "workspace": get_settings().default_workspace,
            "mode": get_settings().mode,
            "sources": dict(sorted(by_provider.items())),
            "identities_held": identity_count,
            "sealed_field_records": sealed,
            "key_fingerprint": engine.pipeline.km.fingerprint(),
            "keystore": engine.pipeline.km.describe(),
            "write_back": {
                "enabled": False,
                "reason": "Blueprint non-goal N1. No code path in this system "
                          "issues a write to a provider.",
            },
            "derived_stores": {
                "search_docs": docs.count("search_docs"),
                "persons": docs.count("persons"),
                "insights": docs.count("insights"),
                "predictions": docs.count("predictions"),
                "note": "Every derived store is droppable and rebuildable from "
                        "the source documents. POST /api/admin/drop-derived "
                        "destroys them so this can be verified rather than "
                        "asserted.",
            },
            "raw_artifacts_retained": docs.count("raw_artifacts"),
            "generated_at": iso(),
        }
    )


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------


@router.get("/connectors", summary="Registered connectors, their scopes and limits")
def connectors() -> dict:
    registry = get_registry()
    return _ok(connectors=registry.descriptors(), capabilities=registry.capabilities())


@router.get("/connectors/{connector_id}", summary="One connector's descriptor")
def connector(connector_id: str) -> dict:
    registry = get_registry()
    if not registry.has(connector_id):
        raise HTTPException(404, f"unknown connector {connector_id!r}; "
                                 f"known: {registry.ids()}")
    instance = registry.get(connector_id)
    return _ok(descriptor=instance.descriptor.to_dict(),
               capabilities=instance.capabilities())


# ---------------------------------------------------------------------------
# Connections (OAuth lifecycle)
# ---------------------------------------------------------------------------


@router.get("/connections", summary="Connection status for every connector")
def connections() -> dict:
    """Status only. No endpoint under /api/connections returns a token, a token
    prefix, or anything derived from one - a status page must not be able to
    become a credential exfiltration endpoint."""
    workspace = get_settings().default_workspace
    return _ok(connections=list_connections(workspace),
               credentials=credentials_report())


class DiagnoseRequest(BaseModel):
    candidate_secret: str = Field(..., min_length=4, max_length=200)


@router.post("/connections/{connector_id}/diagnose",
             summary="Test a candidate client secret against the provider")
def diagnose_connection(connector_id: str, req: DiagnoseRequest) -> dict:
    """Resolves the Client ID / Client Secret / Verification Token ambiguity.

    A provider console shows four similar-looking credentials, and swapping the
    Client Secret for the Verification Token fails at the token exchange with
    `invalid_client` and no indication of which value was wrong. One probe call
    settles it. The candidate is used for that one request and never stored.
    """
    if not get_registry().has(connector_id):
        raise HTTPException(404, f"unknown connector {connector_id!r}")
    result = diagnose_credentials(connector_id, req.candidate_secret)
    _engine().pipeline.audit({
        "action": "diagnose_credentials",
        "connector_id": connector_id,
        "ok": result.get("ok", False),
        "reason": result.get("reason", ""),
    })
    if not result.get("ok"):
        return JSONResponse({"ok": False, "diagnosis": result}, status_code=200)
    return _ok(diagnosis=result)


@router.get("/connections/{connector_id}", summary="One connector's connection status")
def connection(connector_id: str) -> dict:
    if not get_registry().has(connector_id):
        raise HTTPException(404, f"unknown connector {connector_id!r}")
    return _ok(connection=describe(connector_id, get_settings().default_workspace))


@router.post("/connections/{connector_id}/authorize",
             summary="Start an OAuth flow; returns the provider consent URL")
def authorize_connection(connector_id: str, login_hint: str | None = None) -> dict:
    """Step one of the handshake.

    Returns the URL rather than redirecting, so the UI can open it in a popup and
    keep the surrounding page alive. The PKCE verifier stays server-side; the
    browser only ever sees an opaque `state`.
    """
    try:
        result = begin_authorization(
            connector_id, workspace=get_settings().default_workspace,
            login_hint=login_hint)
    except ConnectionError_ as exc:
        raise HTTPException(409, str(exc)) from exc
    return _ok(authorization_url=result.authorization_url, state=result.state,
               expires_in=result.expires_in,
               redirect_uri=describe(connector_id,
                                    get_settings().default_workspace)["redirect_uri"])


@router.get("/connections/{connector_id}/callback",
            summary="OAuth redirect target; exchanges the code and stores the grant")
def connection_callback(
    connector_id: str,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> Any:
    """Where the provider sends the browser.

    Returns HTML rather than JSON, because this is a top-level browser
    navigation - a JSON body here renders as raw text in a popup and looks like a
    crash to the user who just approved access. The page posts the outcome to
    its opener so the Connectors view can update without a manual refresh, and
    links to the API for anyone who arrived here directly.
    """
    detail = error_description or error or ""
    try:
        if not state:
            raise ConnectionError_("the provider did not return a state value")
        result = complete_authorization(
            connector_id, workspace=get_settings().default_workspace,
            code=code or "", state=state, error=detail or None)
        ok, title, message = True, f"{connector_id} connected", result["next_step"]
    except ConnectionError_ as exc:
        ok, title, message = False, f"Could not connect {connector_id}", str(exc)
    except Exception as exc:  # a provider returning something unexpected
        ok, title, message = False, f"Could not connect {connector_id}", \
            f"{type(exc).__name__}: {exc}"[:300]

    _engine().pipeline.audit({
        "action": "oauth_callback",
        "connector_id": connector_id,
        "ok": ok,
        "detail": message[:200],
    })
    from fastapi.responses import HTMLResponse

    return HTMLResponse(_callback_html(ok, connector_id, title, message),
                        status_code=200 if ok else 400)


@router.post("/connections/{connector_id}/disconnect", summary="Remove a connection")
def disconnect_connection(connector_id: str, revoke_remote: bool = True) -> dict:
    if not get_registry().has(connector_id):
        raise HTTPException(404, f"unknown connector {connector_id!r}")
    result = disconnect(connector_id, get_settings().default_workspace,
                        revoke_remote=revoke_remote)
    return _ok(**result)


@router.post("/connections/{connector_id}/verify",
              summary="Prove a stored grant actually works")
def verify_connection(connector_id: str) -> dict:
    """Existence is not validity. A token row can exist and be rejected - revoked
    out of band, or issued for scopes that were later withdrawn - and the honest
    answer is to call the provider and find out."""
    if not get_registry().has(connector_id):
        raise HTTPException(404, f"unknown connector {connector_id!r}")
    result = verify(connector_id, get_settings().default_workspace)
    if not result.get("ok"):
        return JSONResponse({"ok": False, "error": result}, status_code=409)
    return _ok(**result)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@router.post("/sync", summary="Ingest a connector and recompute every derived store")
def sync(req: SyncRequest) -> dict:
    registry = get_registry()
    if not registry.has(req.connector_id):
        raise HTTPException(404, f"unknown connector {req.connector_id!r}")

    # A stored grant, when there is one. Whether a connector *needs* one is
    # derived from its descriptor rather than assumed, so an API-key or
    # file-based connector syncs without one and an OAuth connector fails with
    # an actionable message instead of an opaque provider 401.
    workspace = get_settings().default_workspace
    grant = get_grant(req.connector_id, workspace)
    descriptor = registry.get(req.connector_id).descriptor
    needs_grant = descriptor.auth_flow == "oauth2_code" and bool(descriptor.authorize_url)
    if needs_grant and grant is None and req.connector_id != "demo":
        status = describe(req.connector_id, workspace)
        raise HTTPException(409, {
            "message": f"{req.connector_id} is not connected.",
            "next_step": status["next_step"],
            "authorize": f"/api/connections/{req.connector_id}/authorize",
        })

    t0 = time.perf_counter()
    result = _engine().sync(req.connector_id, stream_filter=req.stream,
                            max_streams=req.max_streams, trigger="api", grant=grant)
    run = result.get("ingest", {})
    if req.rebuild_graph and run.get("status") == "ok":
        result["graph"] = _engine().rebuild_graph()

    # Flush before answering, not at exit. `atexit` covers a clean shutdown
    # only, so without this a crash - or a `docker stop` with a short grace
    # period, or a laptop lid - loses every ingested record. The write is
    # atomic per collection and costs a few milliseconds at this scale; losing
    # a user's workspace is not a trade worth making for that.
    written = flush_stores()

    return _ok(
        took_ms=int((time.perf_counter() - t0) * 1000),
        files_written=written,
        ingest={k: run.get(k) for k in
                ("ingest_run_id", "status", "counters", "error_count", "duration_ms",
                 "streams")},
        derived={k: v for k, v in result.items() if k != "ingest"},
    )


@router.get("/sync/runs", summary="Recent ingest runs")
def sync_runs(limit: int = Q(20, ge=1, le=100)) -> dict:
    rows = _engine().stores.docs.find(
        Query("ingest_runs", sort=[("started_at", -1)], limit=limit))
    return _ok(runs=[{
        "ingest_run_id": r.get("ingest_run_id"),
        "connector_id": r.get("connector_id"),
        "status": r.get("status"),
        "started_at": r.get("started_at"),
        "duration_ms": r.get("duration_ms"),
        "counters": r.get("counters"),
        "error_count": r.get("error_count", 0),
    } for r in rows])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.get("/search", summary="Search across every connected source")
def search(
    q: str = Q("", max_length=500),
    kinds: str = Q("", description="Comma separated: message,email,file,note,video,transcript"),
    providers: str = Q(""),
    person_id: str = Q(""),
    conversation_id: str = Q(""),
    after: str = Q(""),
    before: str = Q(""),
    has_attachment: bool | None = Q(None),
    limit: int = Q(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Q(0, ge=0),
) -> dict:
    request = SearchRequest(
        q=q,
        kinds=[k for k in kinds.split(",") if k],
        providers=[p for p in providers.split(",") if p],
        person_ids=[p for p in person_id.split(",") if p],
        conversation_ids=[c for c in conversation_id.split(",") if c],
        after=after, before=before, has_attachment=has_attachment,
        limit=limit, offset=offset,
    )
    response = get_search_service().search(request)
    return _ok(**response.to_dict())


@router.get("/search/suggest", summary="Typeahead over titles, senders and channels")
def suggest(prefix: str = Q("", min_length=1, max_length=80)) -> dict:
    """Cheap: a prefix scan over the content-free search documents, never a
    full ranking pass. A typeahead that runs BM25 on every keystroke is the
    reason search boxes feel slow in most products."""
    docs = _engine().stores.docs
    needle = prefix.lower()
    seen: dict[str, dict] = {}
    for row in docs.find(Query("search_docs", limit=200_000)):
        for kind, field in (("title", "title"), ("person", "sender_name"),
                            ("channel", "channel"), ("file", "name")):
            value = str(row.get(field) or "")
            if not value or needle not in value.lower():
                continue
            key = f"{kind}:{value.lower()}"
            entry = seen.setdefault(key, {"kind": kind, "value": value, "count": 0})
            entry["count"] += 1
    ranked = sorted(seen.values(), key=lambda e: (-e["count"], len(e["value"])))[:12]
    return _ok(suggestions=ranked)


# ---------------------------------------------------------------------------
# NL2Query
# ---------------------------------------------------------------------------


@router.post("/nl2query", summary="Question -> typed query plan -> result")
def nl2query(req: NL2QueryRequest) -> dict:
    compiler = NL2QueryCompiler()
    plan = compiler.compile(req.question, limit=req.limit)
    if not req.execute:
        return _ok(plan=plan.to_dict())
    outcome = compiler.compile_and_run(req.question, limit=req.limit)
    return _ok(**outcome)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@router.get("/graph/ego", summary="Ego network around a person")
def graph_ego(
    person_id: str = Q(""),
    depth: int = Q(1, ge=1, le=3),
    limit: int = Q(120, ge=1, le=1000),
    kinds: str = Q("Person,Identity,Conversation,Message,File,Note,Event"),
) -> dict:
    graph = _engine().stores.graph
    wanted = {k.strip() for k in kinds.split(",") if k.strip()}

    if not person_id:
        top = graph.top_nodes("Person", limit=1)
        if not top:
            raise HTTPException(404, "graph has no Person nodes; run a sync first")
        person_id = str(top[0].id)

    subgraph = graph.ego(person_id, hops=depth, node_labels=wanted, limit=limit)
    if not subgraph.nodes:
        raise HTTPException(404, f"no person {person_id!r} in the graph")
    return _ok(**_subgraph_payload(subgraph))


@router.get("/graph/path", summary="Shortest path between two people")
def graph_path(
    source_id: str = Q(...),
    target_id: str = Q(...),
    max_depth: int = Q(4, ge=1, le=6),
) -> dict:
    graph = _engine().stores.graph
    if not graph.get_node(source_id):
        raise HTTPException(404, f"unknown node {source_id!r}")
    if not graph.get_node(target_id):
        raise HTTPException(404, f"unknown node {target_id!r}")
    paths = graph.path(source_id, target_id, max_hops=max_depth, limit=1)
    if not paths:
        return _ok(found=False, hops=None, paths=[],
                   note=f"no path within {max_depth} hops. In this model most "
                        f"people are 2-3 hops apart via shared conversations; a "
                        f"longer max_depth will not help if no shared "
                        f"conversation exists.")
    hydrated = []
    for ids in paths:
        nodes = [graph.node(node_id) for node_id in ids]
        edges = []
        for a, b in zip(ids, ids[1:]):
            for edge in graph.neighbors(a, limit=400):
                if edge.dst == b:
                    edges.append({"source": edge.src, "target": edge.dst,
                                  "type": edge.type, "props": edge.props})
        hydrated.append({
            "hops": len(ids) - 1,
            "nodes": [_node_payload(n) for n in nodes if n],
            "edges": edges,
        })
    return _ok(found=True, paths=hydrated, hops=hydrated[0]["hops"])


@router.get("/graph/stats", summary="Node and edge census")
def graph_stats() -> dict:
    return _ok(stats=_engine().stores.graph.stats())


def _node_payload(node: Node) -> dict:
    return {"id": node.id, "label": node.label, "props": node.props}


def _subgraph_payload(subgraph: Any) -> dict:
    """Edges are filtered to those whose endpoints are both present.

    Without that, a degree-limited ego graph ships edges pointing at nodes the
    client never received, and the renderer silently drops them - which looks
    like a rendering bug and is actually a serialization one.
    """
    node_ids = {n.id for n in subgraph.nodes}
    edges = [e for e in subgraph.edges if e.src in node_ids and e.dst in node_ids]
    return {
        "nodes": [_node_payload(n) for n in subgraph.nodes],
        "edges": [{"source": e.src, "target": e.dst, "type": e.type,
                   "props": e.props, "key": list(e.key)} for e in edges],
        "node_count": len(subgraph.nodes),
        "edge_count": len(edges),
    }


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@router.get("/timeline", summary="Chronological activity across all sources")
def timeline(
    after: str = Q(""),
    before: str = Q(""),
    providers: str = Q(""),
    kinds: str = Q(""),
    person_id: str = Q(""),
    limit: int = Q(100, ge=1, le=500),
) -> dict:
    """One chronological view across every source.

    This is the view that makes cross-source product real: a user does not
    think in "Slack" and "email", they think in "what happened", and any UI
    that partitions by provider makes them do the joining themselves.
    """
    engine = _engine()
    filters: list[Filter] = []
    if providers:
        filters.append(Filter(field="provider", op="in",
                              value=[p for p in providers.split(",") if p]))
    if kinds:
        filters.append(Filter(field="kind", op="in",
                              value=[k for k in kinds.split(",") if k]))
    if person_id:
        filters.append(Filter(field="person_ids", op="in", value=[person_id]))
    if after:
        filters.append(Filter(field="ts", op="gte", value=after))
    if before:
        filters.append(Filter(field="ts", op="lte", value=before))

    events: list[dict] = []
    for collection in CONTENT_COLLECTIONS:
        rows = engine.stores.docs.find(
            Query(collection, filters=filters, sort=[("ts", 1)], limit=limit))
        for row in rows:
            # Decrypt for display. `thread_subject` is a sealed envelope at rest
            # (it is a sentence lifted out of the body), so reading it straight
            # off the stored document hands the client an object where a string
            # belongs - which is a crash in the UI, not a cosmetic bug.
            extra = row.get("extra") or {}
            subject = extra.get("subject") or row.get("title") or row.get("name")
            if not subject:
                try:
                    subject = engine.pipeline.decrypt_record(row).get("thread_subject")
                except Exception:
                    subject = ""
                if isinstance(subject, dict):
                    subject = ""
            title = str(subject or "")
            events.append({
                "doc_id": row.get("_id"),
                "ts": row.get("ts") or row.get("modified_ts") or row.get("last_edited") or "",
                "provider": row.get("provider"),
                "kind": row.get("kind") or collection[:-1],
                "title": title,
                "sender_name": row.get("sender_name", ""),
                "importance": round(float(row.get("importance") or 0.0), 3),
                "has_attachment": bool(row.get("attachments")),
                "deadline_count": len((row.get("entities") or {}).get("deadlines") or []),
                "person_ids": row.get("person_ids", []),
            })
    events.sort(key=lambda e: e["ts"] or "")
    return _ok(events=events[:limit], count=len(events[:limit]))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@router.get("/files", summary="File inventory with owner and references")
def files(
    providers: str = Q(""),
    q: str = Q(""),
    has_owner: bool | None = Q(None),
    limit: int = Q(100, ge=1, le=500),
    offset: int = Q(0, ge=0),
) -> dict:
    engine = _engine()
    filters: list[Filter] = []
    if providers:
        filters.append(Filter(field="provider", op="in",
                              value=[p for p in providers.split(",") if p]))
    if q:
        filters.append(Filter(field="name", op="contains", value=q))
    if has_owner is not None:
        filters.append(Filter(field="person_ids", op="exists", value=has_owner))

    rows = engine.stores.docs.find(
        Query("files", filters=filters, sort=[("modified_ts", -1)],
              limit=limit, skip=offset))
    total = engine.stores.docs.count("files", filters)

    # Reference counts come from the message index, not from the file document,
    # so a file is not "orphaned" merely because nobody has opened it.
    referencing: dict[str, int] = {}
    for row in engine.stores.docs.find(Query("messages", limit=200_000)):
        for att in row.get("attachments") or []:
            key = att.get("provider_file_id")
            if key:
                referencing[str(key)] = referencing.get(str(key), 0) + 1

    items = []
    for row in rows:
        items.append({
            "doc_id": row.get("_id"),
            "name": row.get("name", ""),
            "mime": row.get("mime", ""),
            "size_bytes": row.get("size_bytes", 0),
            "provider": row.get("provider", ""),
            "modified_ts": row.get("modified_ts", ""),
            "folder_path": row.get("folder_path", []),
            "owner_person_id": row.get("person_id", ""),
            "references": referencing.get(str(row.get("provider_file_id")), 0),
        })
    return _ok(files=items, total=total, has_more=offset + len(items) < total)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@router.get("/documents/{doc_id:path}", summary="One document, decrypted on read")
def document(doc_id: str, include_body: bool = Q(True)) -> dict:
    engine = _engine()
    row = engine._find_document(doc_id)  # noqa: SLF001 - intentional single accessor
    if not row:
        raise HTTPException(404, f"unknown document {doc_id!r}")
    plain = engine.pipeline.decrypt_record(row)
    if not include_body:
        for field in ("body_text", "body_html", "text", "extracted_text"):
            plain.pop(field, None)
    related = _related_documents(engine, doc_id)
    return _ok(document=plain, related=related)


def _related_documents(engine: Any, doc_id: str) -> list[dict]:
    """Same conversation, most recent first, excluding the document itself.

    Cheap and useful: "what else is in this thread" is the single most common
    next click after a search result, and it needs no new machinery.
    """
    row = engine._find_document(doc_id)  # noqa: SLF001
    if not row:
        return []
    conversation = str(row.get("conversation_id") or "")
    if not conversation:
        return []
    siblings = []
    for collection in ("messages",):
        for other in engine.stores.docs.find(
            Query(collection, filters=[Filter(field="conversation_id", op="eq",
                                              value=conversation)],
                  sort=[("ts", -1)], limit=20)
        ):
            if other["_id"] == doc_id:
                continue
            extra = other.get("extra") or {}
            title = extra.get("subject") or other.get("title") or ""
            if not title:
                # `thread_subject` is sealed at rest, so it is an envelope here,
                # not a string. Decrypt for the label, and fall back to empty
                # rather than shipping the object to the client.
                try:
                    title = engine.pipeline.decrypt_record(other).get("thread_subject") or ""
                except Exception:
                    title = ""
                if isinstance(title, (dict, list)):
                    title = ""
            siblings.append({
                "doc_id": other["_id"],
                "ts": other.get("ts", ""),
                "provider": other.get("provider", ""),
                "sender_name": other.get("sender_name", ""),
                "title": str(title),
            })
    return siblings[:10]


@router.post("/documents/{doc_id:path}/summarize", summary="Extractive summary")
def summarize(doc_id: str, max_sentences: int = Body(3, embed=True, ge=1, le=10)) -> dict:
    result = _engine().summarize(doc_id, max_sentences=max_sentences)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return _ok(**result)


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------


@router.get("/persons", summary="Resolved people")
def persons(
    cross_source_only: bool = Q(False),
    providers: str = Q(""),
    limit: int = Q(100, ge=1, le=500),
) -> dict:
    engine = _engine()
    rows = engine.stores.docs.find(Query("persons", sort=[("message_count", -1)],
                                         limit=limit))
    if cross_source_only:
        rows = [r for r in rows if r.get("cross_source")]
    if providers:
        wanted = {p.strip() for p in providers.split(",") if p.strip()}
        rows = [r for r in rows if wanted & set(r.get("providers") or [])]
    return _ok(persons=[_person_payload(r) for r in rows], count=len(rows))


@router.get("/persons/{person_id}", summary="One person with the evidence for their merges")
def person(person_id: str) -> dict:
    engine = _engine()
    row = engine.stores.docs.get("persons", person_id)
    if not row:
        raise HTTPException(404, f"unknown person {person_id!r}")

    identities = []
    evidence: list[dict] = []
    for iid in row.get("identity_ids", []):
        identity = engine.stores.docs.get("identities", iid)
        if identity:
            identities.append({
                "identity_id": iid, "provider": identity.get("provider"),
                "display_name": identity.get("display_name"),
                "email_present": bool(identity.get("email")),
                "phone_present": bool(identity.get("phone")),
                "mention_count": identity.get("mention_count", 0),
            })
    for suggestion in engine.stores.docs.find(Query("link_suggestions", limit=500)):
        if person_id in (suggestion.get("identity_a"), suggestion.get("identity_b")):
            evidence.append(suggestion)

    recent = engine.stores.docs.find(
        Query("messages", filters=[Filter(field="person_ids", op="in", value=[person_id])],
              sort=[("ts", -1)], limit=10))
    # Same rule as everywhere else: a sealed field read off the stored document
    # is an envelope object, not a label. Decrypt for display, coerce to a
    # string, and never hand the envelope to the client.
    recent_rows = []
    for doc in recent:
        try:
            label = engine.pipeline.decrypt_record(doc).get("thread_subject")
        except Exception:
            label = ""
        if isinstance(label, (dict, list)):
            label = ""
        recent_rows.append({
            "doc_id": doc["_id"],
            "ts": doc.get("ts", ""),
            "title": str((doc.get("extra") or {}).get("subject") or label or ""),
            "provider": doc.get("provider", ""),
        })
    return _ok(person=_person_payload(row), identities=identities,
               pending_suggestions=evidence, recent=recent_rows)


@router.get("/suggestions", summary="Identity links held for user confirmation")
def suggestions(status: str = Q("pending"), limit: int = Q(50, ge=1, le=200)) -> dict:
    rows = _engine().stores.docs.find(
        Query("link_suggestions", filters=[Filter(field="status", op="eq", value=status)],
              sort=[("score", -1)], limit=limit))
    return _ok(suggestions=rows, count=len(rows))


def _person_payload(row: dict) -> dict:
    return {
        "person_id": row.get("person_id"),
        "display_name": row.get("display_name", "Unknown"),
        "aliases": row.get("aliases", []),
        "providers": row.get("providers", []),
        "cross_source": row.get("cross_source", False),
        "message_count": row.get("message_count", 0),
        "identity_count": len(row.get("identity_ids", [])),
        "contact_count": len(row.get("contact_identity_ids", [])),
        "last_seen": row.get("last_seen", ""),
    }


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


@router.get("/insights", summary="Hidden-connection detector output")
def insights(
    detector: str = Q(""),
    state: str = Q("advisory"),
    min_confidence: float = Q(0.0, ge=0.0, le=1.0),
    limit: int = Q(50, ge=1, le=200),
) -> dict:
    filters: list[Filter] = [Filter(field="state", op="eq", value=state)]
    if detector:
        filters.append(Filter(field="detector", op="eq", value=detector))
    rows = _engine().stores.docs.find(
        Query("insights", filters=filters, sort=[("confidence", -1)], limit=limit))
    rows = [r for r in rows if float(r.get("confidence") or 0) >= min_confidence]
    return _ok(insights=rows, count=len(rows))


@router.post("/insights/{insight_id}/accept", summary="Accept an insight (user action)")
def accept_insight(insight_id: str, req: AcceptInsightRequest) -> dict:
    """The only state transition in the system that promotes a machine
    inference to a fact, and the only writer of `state: "accepted"`."""
    row = _engine().stores.docs.get("insights", insight_id)
    if not row:
        raise HTTPException(404, f"unknown insight {insight_id!r}")
    ok = _engine().projector.accept_insight(insight_id, req.kind or str(row.get("kind")),
                                            req.source_id, req.target_id)
    if not ok:
        raise HTTPException(409, "one or both endpoints are not nodes in the graph")
    row["state"] = "accepted"
    row["accepted_by"] = "user"
    row["accepted_at"] = iso()
    _engine().stores.docs.put("insights", insight_id, row)
    return _ok(insight=row, edge={"source": req.source_id, "target": req.target_id,
                                  "type": "LINKED_TO", "state": "accepted"})


@router.post("/insights/{insight_id}/dismiss", summary="Dismiss an insight (user action)")
def dismiss_insight(insight_id: str, req: AcceptInsightRequest) -> dict:
    row = _engine().stores.docs.get("insights", insight_id)
    if not row:
        raise HTTPException(404, f"unknown insight {insight_id!r}")
    _engine().projector.dismiss_insight(req.source_id, req.target_id)
    row["state"] = "dismissed"
    row["dismissed_at"] = iso()
    _engine().stores.docs.put("insights", insight_id, row)
    return _ok(insight=row)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


@router.get("/predictions", summary="Deadlines, unanswered questions, resurfacing")
def predictions(
    kind: str = Q(""),
    limit: int = Q(30, ge=1, le=200),
) -> dict:
    engine = _engine()
    filters: list[Filter] = []
    if kind:
        filters.append(Filter(field="kind", op="eq", value=kind))
    rows = engine.stores.docs.find(
        Query("predictions", filters=filters, sort=[("score", -1)], limit=limit))
    return _ok(predictions=rows, count=len(rows),
               disclaimer="Advisory only. Nothing here is written back to a provider.")


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@router.post("/admin/drop-derived", summary="Destroy every derived store and rebuild")
def drop_derived(rebuild: bool = Q(True)) -> dict:
    """The privacy claim, as an operation rather than a promise."""
    result = _engine().drop_everything()
    if rebuild:
        result["rebuilt"] = _engine().rebuild(full=True)
    result["files_written"] = flush_stores()
    return _ok(**result)


@router.get("/audit", summary="Append-only audit log")
def audit(limit: int = Q(50, ge=1, le=500)) -> dict:
    rows = _engine().stores.docs.find(
        Query("audit", sort=[("ts", -1)], limit=limit))
    return _ok(entries=rows)


def _callback_html(ok: bool, connector_id: str, title: str, message: str) -> str:
    colour = "#1f7a4d" if ok else "#b3261e"
    glyph = "&#10003;" if ok else "&#10005;"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  body{{font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       max-width:32rem;margin:14vh auto;padding:0 1.5rem;color:#1c1c1a;background:#fbfbfa}}
  .box{{background:#fff;border:1px solid #e5e5e2;border-radius:10px;padding:1.6rem 1.8rem;
        box-shadow:0 4px 18px rgba(0,0,0,.06)}}
  .mark{{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;
        color:#fff;background:{colour};font-size:19px;margin-bottom:.9rem}}
  h1{{font-size:1.1rem;margin:0 0 .5rem}}
  p{{color:#6b6b64;margin:0 0 1rem}}
  a{{color:#2f5fd0}}
  .mono{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
        background:#f4f4f2;padding:.1em .35em;border-radius:3px}}
</style></head>
<body><div class="box" id="result" data-ok="{str(ok).lower()}"
     data-connector="{connector_id}" data-message="{message[:300]}">
  <div class="mark">{glyph}</div>
  <h1>{title}</h1>
  <p>{message}</p>
  <p style="font-size:12.5px">You can close this window. The Connectors page updates
     automatically. Or <a href="/api/connections/{connector_id}">check the status via the API</a>.</p>
</div>
<script>
  // Tell the opener, so the Connectors view does not need a manual refresh.
  // A popup opened directly (not from our UI) has no opener, so the guard
  // matters - window.opener is null rather than undefined in that case.
  if (window.opener && !window.opener.closed) {{
    try {{
      window.opener.postMessage({{
        type: 'omni:connection',
        connector: '{connector_id}',
        ok: {str(ok).lower()},
        message: {message!r}.slice(0, 300)
      }}, '*');
    }} catch (e) {{ /* cross-origin opener: the user can close this manually */ }}
    setTimeout(function () {{ window.close(); }}, 1200);
  }}
</script>
</body></html>"""
