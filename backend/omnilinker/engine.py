"""Workspace engine: the orchestration the API talks to.

One object, one job: take documents and bring every derived store up to date,
in the right order.

    ingest -> identities -> persons -> graph -> insights -> search -> AI

The order is a dependency graph, not a preference:

1. **identities first.** They are produced by ingestion but *consumed* by
   resolution.
2. **persons before the graph.** The projector emits `(Message)-[:SENT_BY]->(Person)`
   edges, so persons must exist or every message would be wired to a placeholder.
3. **the graph before insights.** `write_insight_edges` only materializes a
   detector's finding as an edge if both endpoints already exist as nodes.
4. **search after enrichment.** The index must contain `importance` and
   `thread_subject`, or the first query after a sync is missing them and the
   second one silently disagrees.
5. **predictions last.** They read decrypted bodies and the resolved people.

Everything here is idempotent. Running it twice is a no-op; running it after a
connector is added is a fix-up, not a migration. That property is what makes
"recompute" a safe response to "the results look wrong", and it is why there is
no separate backfill code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.ai.enrich import enrich_documents
from omnilinker.ai.predict import run_predictions
from omnilinker.ai.summarize import get_summarizer
from omnilinker.config import get_settings
from omnilinker.crypto.envelope import encrypt
from omnilinker.identity import IdentityResolver, run_all as run_detectors
from omnilinker.ids import prefixed
from omnilinker.normalize import iso
from omnilinker.pipeline import IngestPipeline
from omnilinker.projection import GraphProjector
from omnilinker.search import get_search_service
from omnilinker.store import get_stores
from omnilinker.store.base import Query

#: collections read as "content documents" for indexing, enrichment, detection
CONTENT_COLLECTIONS = ("messages", "notes", "files", "videos", "transcripts")


@dataclass
class EngineState:
    stage: str = "idle"
    last_run_at: str = ""
    last_run_ms: int = 0
    stages: dict[str, dict] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "last_run_at": self.last_run_at,
            "last_run_ms": self.last_run_ms,
            "stages": self.stages,
            "error": self.error,
        }


class WorkspaceEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.stores = get_stores()
        self.pipeline = IngestPipeline()
        self.projector = GraphProjector()
        self.state = EngineState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sync(self, connector_id: str = "demo", **kwargs: Any) -> dict:
        """Ingest one connector, then bring every derived store up to date."""
        self.state.stage = "ingest"
        t_all = time.perf_counter()
        run = self.pipeline.sync_connector(connector_id, **kwargs)
        self._record("ingest", {
            "connector_id": connector_id,
            "status": run.status,
            **run.counters.as_dict(),
        })
        derived = self.rebuild(full=run.counters.created > 0)
        derived["ingest"] = run.as_dict()
        # Persist now rather than at exit. `atexit` only fires on a clean
        # shutdown, so relying on it means a crash loses the workspace - which
        # for a local-first product is losing the user's data.
        from omnilinker.store import flush_stores

        derived["files_written"] = flush_stores()
        derived["total_ms"] = int((time.perf_counter() - t_all) * 1000)
        self.state.last_run_at = iso()
        self.state.last_run_ms = derived["total_ms"]
        self.state.stage = "idle"
        return derived

    def rebuild(self, *, full: bool = False) -> dict:
        """Run every derived stage. Idempotent."""
        out: dict[str, Any] = {}
        for name, fn in (
            ("identities", self.resolve_identities),
            ("graph", self.rebuild_graph if full else self.project_graph),
            ("insights", self.detect),
            ("enrichment", self.enrich),
            ("search", self.reindex),
            ("predictions", self.predict),
        ):
            t0 = time.perf_counter()
            self.state.stage = name
            try:
                result = fn()
            except Exception as exc:
                # A failed derived store must not leave the rest of the system
                # unusable: report it, keep going, and surface it in /system/info.
                self.state.error = f"{name}: {exc}"
                out[name] = {"ok": False, "error": str(exc)}
                self._record(name, {"ok": False, "error": str(exc)})
                continue
            out[name] = result
            self._record(name, {"ok": True, "ms": int((time.perf_counter() - t0) * 1000),
                                **_headline(result)})
        self.state.stage = "idle"
        return out

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def resolve_identities(self, *, merge_threshold: float | None = None,
                           suggest_threshold: float | None = None) -> dict:
        """Block, score, cluster. Writes `persons` and `link_suggestions`."""
        docs = self.stores.docs
        identities = docs.find(Query("identities", limit=200_000))
        documents = self.load_documents()

        resolver = IdentityResolver(
            merge_threshold=merge_threshold or self.settings.merge_threshold,
            suggest_threshold=suggest_threshold
            or self.settings.suggest_threshold,
        )
        result = resolver.resolve(identities, documents)

        # Replace persons wholesale. The resolver is the only writer, and a
        # partial update would leave persons from a previous threshold behind.
        docs.purge(["persons"])
        if result.persons:
            docs.put_many("persons", result.persons, "_id")

        # Persist the identity -> person assignment, and stamp it back onto the
        # documents so the graph projector and the search facets can filter on
        # resolved people without a second join.
        by_id = {i["_id"]: i for i in identities}
        person_of: dict[str, str] = {}
        for person in result.persons:
            for iid in person["identity_ids"]:
                person_of[iid] = str(person["person_id"])
        for suggestion in result.suggestions:
            for key in ("identity_a", "identity_b"):
                target = by_id.get(suggestion[key])
                if target and target.get("person_id"):
                    person_of[suggestion[key]] = str(target["person_id"])

        for iid, person_id in person_of.items():
            doc = docs.get("identities", iid)
            if doc:
                doc["person_id"] = person_id
                docs.put("identities", iid, doc)

        docs.purge(["link_suggestions"])
        if result.suggestions:
            docs.put_many("link_suggestions", result.suggestions, "_id")

        self._stamp_persons(documents, person_of)
        return {
            "persons": len(result.persons),
            "identities": len(identities),
            "suggestions": len(result.suggestions),
            "cross_source_persons": sum(1 for p in result.persons if p["cross_source"]),
            **result.stats,
        }

    def project_graph(self) -> dict:
        return self.projector.project(self.settings.default_workspace)

    def rebuild_graph(self) -> dict:
        return self.projector.rebuild(self.settings.default_workspace)

    def detect(self) -> dict:
        documents = self.load_documents()
        files = self.stores.docs.find(Query("files", limit=200_000))
        persons = self.stores.docs.find(Query("persons", limit=200_000))

        ref_to_person = {}
        for identity in self.stores.docs.find(Query("identities", limit=200_000)):
            if identity.get("person_id") and identity.get("ref"):
                ref_to_person.setdefault(str(identity["ref"]), str(identity["person_id"]))

        result = run_detectors(documents=documents, files=files, persons=persons,
                               ref_to_person=ref_to_person)
        docs = self.stores.docs
        docs.purge(["insights"])
        records = [i.to_dict() for i in result.insights]
        if records:
            docs.put_many("insights", records, "_id")
        edges = self.projector.write_insight_edges(result.insights)
        return {"insights": len(records), "insight_edges": edges, **result.as_dict()}

    def enrich(self) -> dict:
        documents = self.load_documents()
        self_refs = self._self_refs()
        result = enrich_documents(documents, self_refs=self_refs)

        docs = self.stores.docs
        importance = result["importance"]
        patched = 0
        for collection in CONTENT_COLLECTIONS:
            for doc_id, patch in importance.items():
                if not doc_id.startswith(_PREFIX_FOR.get(collection, "\0")):
                    continue
                current = docs.get(collection, doc_id)
                if not current:
                    continue
                current.update(patch)
                docs.put(collection, doc_id, current)
                patched += 1

        # Enrichment runs *after* the pipeline sealed the record, so writing a
        # field here bypasses the seal unless we apply it explicitly. The
        # derived subject is a sentence lifted from the sealed body, so leaving
        # it in the clear would undo the sealing of the field it came from.
        subjects = 0
        for doc_id, subject in result["thread_subjects"].items():
            for collection in CONTENT_COLLECTIONS:
                current = docs.get(collection, doc_id)
                if not current or current.get("thread_subject") == subject:
                    continue
                if self.settings.enable_encryption and current.get("_id"):
                    subject = encrypt(
                        current.get("provider", "unknown"), subject,
                        doc_id=current["_id"], field_path="thread_subject",
                        key_manager=self.pipeline.km,
                    )
                current["thread_subject"] = subject
                docs.put(collection, doc_id, current)
                subjects += 1
                break

        return {"scored": patched, "thread_subjects": subjects,
                "documents": result["documents"]}

    def reindex(self) -> dict:
        return get_search_service().reindex(persist=self.settings.persist_index)

    def predict(self) -> dict:
        documents = self.load_documents()
        result = run_predictions(documents)
        docs = self.stores.docs
        docs.purge(["predictions"])
        records = [p.to_dict() for p in result.predictions]
        if records:
            docs.put_many("predictions", records, "_id")
        return {"predictions": len(records), **result.stats}

    def summarize(self, doc_id: str, *, max_sentences: int = 3) -> dict:
        doc = self._find_document(doc_id)
        if not doc:
            return {"error": f"unknown document {doc_id!r}"}
        plain = self.pipeline.decrypt_record(doc)
        body = "\n".join(str(plain.get(k) or "") for k in
                         ("body_text", "title", "text", "name") if plain.get(k))
        summary = get_summarizer(self.settings.ai_mode).summarize(
            body, max_sentences=max_sentences)
        return {"doc_id": doc_id, "provider": doc.get("provider", ""), **summary.to_dict()}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_documents(self) -> list[dict]:
        """Every content document, decrypted.

        The whole set is held in memory on purpose: the identity resolver, the
        detectors, the summariser and the predictor all need broad access, and
        four separate filtered queries would be both slower and easier to get
        subtly inconsistent. At 200k documents this is the point where a
        streaming redesign becomes necessary, and the limit is explicit.
        """
        out: list[dict] = []
        for collection in CONTENT_COLLECTIONS:
            for doc in self.stores.docs.find(Query(collection, limit=200_000)):
                try:
                    out.append(self.pipeline.decrypt_record(doc))
                except Exception:
                    # One undecryptable row must not take down resolution for
                    # the entire workspace. Keep the document, drop the content.
                    out.append({**doc, "body_text": "", "_undecryptable": True})
        out.sort(key=lambda d: (str(d.get("ts") or ""), str(d.get("_id") or "")))
        return out

    def _find_document(self, doc_id: str) -> dict | None:
        for collection in CONTENT_COLLECTIONS:
            doc = self.stores.docs.get(collection, doc_id)
            if doc:
                return doc
        return None

    def _self_refs(self) -> set[str]:
        """Refs belonging to the workspace owner. Used for "sent by you" and
        "mentions you" in importance scoring."""
        refs: set[str] = set()
        for identity in self.stores.docs.find(Query("identities", limit=200_000)):
            if identity.get("is_self"):
                ref = identity.get("ref")
                if ref:
                    refs.add(str(ref))
        return refs

    def _stamp_persons(self, documents: Sequence[dict],
                       person_of: Mapping[str, str]) -> None:
        """Write `person_id` onto documents and into `entities.person_ids`.

        The doc write is the one that matters: search filters and the projector
        both read it, and keeping it in `entities` as well is a convenience for
        the API that costs one small array.
        """
        docs = self.stores.docs
        ref_to_person: dict[str, str] = {}
        for identity in docs.find(Query("identities", limit=200_000)):
            if identity.get("ref") and identity.get("person_id"):
                ref_to_person[str(identity["ref"])] = str(identity["person_id"])

        touched = 0
        for collection in CONTENT_COLLECTIONS:
            for doc in docs.find(Query(collection, limit=200_000)):
                refs: list[str] = []
                for key in ("sender_ref", "owner_ref", "author_ref"):
                    ref = doc.get(key)
                    if ref and str(ref) in ref_to_person:
                        refs.append(str(ref))
                if not refs:
                    continue
                person_ids = sorted({ref_to_person[r] for r in refs})
                if doc.get("person_ids") == person_ids and not refs:
                    continue
                doc["person_ids"] = person_ids
                doc["person_id"] = person_ids[0]
                doc["sender_refs"] = refs
                entities = dict(doc.get("entities") or {})
                entities["person_ids"] = person_ids
                doc["entities"] = entities
                docs.put(collection, doc["_id"], doc)
                touched += 1
        self._last_stamped = touched

    # ------------------------------------------------------------------
    def drop_everything(self) -> dict:
        """Destroy every derived store. Proves the source documents still
        reconstruct everything (blueprint 10.4 / ADR 'derived stores are
        rebuildable at any time')."""
        get_search_service().drop_index()
        removed = self.projector.rebuild(self.settings.default_workspace)
        for collection in ("persons", "link_suggestions", "insights", "predictions",
                           "search_docs"):
            self.stores.docs.purge([collection])
        self.state.stage = "idle"
        return {"graph": removed, "purged": ["persons", "link_suggestions", "insights",
                                              "predictions", "search_docs"]}

    def system_info(self) -> dict:
        stats = self.stores.stats()
        return {
            "mode": self.settings.mode,
            "workspace": self.settings.default_workspace,
            "stores": stats,
            "engine": self.state.to_dict(),
            "capabilities": {
                "encryption": self.settings.enable_encryption,
                "vector_search": self.settings.enable_vector,
                "ai_mode": self.settings.ai_mode,
                "search_engine": "hybrid" if self.settings.enable_vector else "lexical",
                "resolver": {
                    "merge_threshold": self.settings.merge_threshold,
                    "suggest_threshold": self.settings.suggest_threshold,
                },
                "rrf_k": self.settings.rrf_k,
                "result_limit": self.settings.result_limit,
            },
            "search": get_search_service().ensure_index().stats(),
            "deviations": DEVIATIONS,
        }

    def _record(self, stage: str, data: dict) -> None:
        self.state.stages[stage] = {**self.state.stages.get(stage, {}), **data,
                                    "at": iso()}


_PREFIX_FOR = {"messages": ("slack:", "gmail:", "discord:", "whatsapp:", "demo:"),
               "files": ("gdrive:", "onedrive:", "dropbox:", "slack:", "whatsapp:", "gmail:"),
               "notes": ("notion:", "evernote:"),
               "videos": ("youtube:",), "transcripts": ("youtube:",)}


#: Recorded, not hidden. Every one of these is a place where the running system
#: differs from the blueprint, with the reason.
DEVIATIONS: list[dict] = [
    {
        "id": "ADR-020",
        "area": "stores",
        "blueprint": "Neo4j + MongoDB, dockerised",
        "actual": "DocStore/GraphStore interfaces with two backends: embedded "
                  "(JSON + in-memory adjacency, the default) and docker "
                  "(motor + neo4j)",
        "reason": "The product must be runnable with one command and no "
                  "infrastructure. The interfaces are identical, so the Mongo/"
                  "Neo4j path is a configuration change, not a rewrite.",
        "risk": "The embedded backend is single-process and not tuned for "
                "millions of documents. It is a development and single-user "
                "profile, and the docker profile is the production one.",
    },
    {
        "id": "ADR-021",
        "area": "crypto",
        "blueprint": "XChaCha20-Poly1305",
        "actual": "AES-256-GCM",
        "reason": "Available in the `cryptography` build everywhere including "
                  "wasm, hardware accelerated on all target platforms, and a "
                  "96-bit nonce is safe under this system's one-key-per-source, "
                  "random-nonce-per-write model. The change is one function in "
                  "crypto/envelope.py.",
        "risk": "A future provider fan-out that reuses a DEK across more than "
                "~2^32 writes per key would need the 192-bit nonce. Bounded by "
                "the per-source key hierarchy today.",
    },
    {
        "id": "ADR-022",
        "area": "frontend",
        "blueprint": "Next.js (SSR)",
        "actual": "Vite + React + TypeScript SPA, served as static files by FastAPI",
        "reason": "The client holds the keys, so there is no server to render. "
                  "There is no SEO requirement for a private workspace and no "
                  "benefit from SSR, while Vite gives a faster dev loop and a "
                  "single-command setup.",
        "risk": "No SSR means no server-rendered first paint. Acceptable for an "
                "authenticated tool; would need revisiting for a public "
                "marketing surface.",
    },
    {
        "id": "ADR-023",
        "area": "search",
        "blueprint": "dense vector index (HNSW) over neural embeddings",
        "actual": "BM25 inverted index (field weighted) fused via RRF with a "
                  "hashed bag-of-words embedding, off by default",
        "reason": "The fusion, the filters, the facets and the evaluation harness "
                  "are the parts that carry the precision@10 target, and they "
                  "are identical whichever vector source feeds them. A hashed "
                  "embedding keeps the whole system runnable offline.",
        "risk": "The default embedding has no synonymy: 'shard key' and "
                "'partitioning strategy' are orthogonal to it. This is the "
                "single largest retrieval quality gap and it is the obvious "
                "first swap.",
    },
    {
        "id": "ADR-024",
        "area": "ai",
        "blueprint": "LLM summarisation, prediction and entity extraction",
        "actual": "TextRank extraction, rule-based prediction, rule-based entity "
                  "and date extraction",
        "reason": "Every output remains a verbatim span of a real document, so "
                  "every claim is clickable to its source. A model is an "
                  "upgrade behind the same interface, not a prerequisite.",
        "risk": "Extractive summaries cannot merge facts across documents, and "
                "rule-based extraction misses implicit entities. Both are "
                "measured against the same evaluation set.",
    },
    {
        "id": "ADR-025",
        "area": "search/privacy",
        "blueprint": "k-anonymized token projection as the persisted search store",
        "actual": "Persisted `search_docs` is content-free (filters, facets, "
                  "titles only). The inverted index is in-memory by default; "
                  "when persisted it applies a k-anonymity floor.",
        "reason": "A title is needed to display a result and is already exposed "
                  "in provider UIs; a body must be decrypted to be displayed, so "
                  "it is decrypted per request for the page being shown rather "
                  "than kept as a plaintext copy.",
        "risk": "Persisting the index exposes the corpus vocabulary for terms "
                "shared by >= k documents. k=2 is weak. The default is not to "
                "persist it.",
    },
    {
        "id": "ADR-026",
        "area": "connectors",
        "blueprint": "per-provider OAuth credential configuration",
        "actual": "One resolver, `connections.client_credentials`, reading "
                  "`OMNI_OAUTH_<PROVIDER>_CLIENT_ID` / `_CLIENT_SECRET`. The "
                  "authorize step, the token exchange and the refresh all call "
                  "it. A diagnostic endpoint tests a candidate secret against "
                  "the provider using a known-bad control, and reports "
                  "`inconclusive` when the provider cannot distinguish.",
        "reason": "The two halves of the handshake previously resolved "
                  "credentials independently, and the token exchange - the only "
                  "step that authenticates the client - fell back to the "
                  "literal string `demo-client-id`. The app looked correctly "
                  "configured and the flow reached the approval screen, then "
                  "failed with `invalid_client`, which reads as a wrong secret "
                  "when the secret was never read.",
        "risk": "The `OMNI_OAUTH_` prefix is a convention, not a schema, so a "
                "typo in a variable name fails closed and invisibly until a "
                "connection is attempted. Slack cannot validate a secret in "
                "isolation at all - it checks the authorization code first - so "
                "for it the real handshake is the only test.",
    },
]


def _headline(result: Mapping[str, Any]) -> dict:
    """A few scalars from a stage result, for the engine state readout.

    Deliberately lossy: `/system/info` reports health, not a dump of every
    derived store, and truncating here means a stage that starts returning large
    structures cannot quietly bloat the health endpoint.
    """
    scalarish = ("persons", "identities", "suggestions", "cross_source_persons",
                 "insights", "insight_edges", "nodes", "edges", "scored",
                 "thread_subjects", "search_docs", "terms", "predictions")
    return {k: v for k, v in result.items()
            if k in scalarish and isinstance(v, (int, float, str, bool))}


_engine: WorkspaceEngine | None = None


def get_engine() -> WorkspaceEngine:
    global _engine
    if _engine is None:
        _engine = WorkspaceEngine()
    return _engine


def reset_engine() -> None:
    """Test hook."""
    global _engine
    _engine = None
