"""Search service: index build, hybrid retrieval, RRF fusion, snippets, facets.

The privacy-relevant decision in this file: **the search index stores no
content.** It stores only the fields the ranking function needs (kind,
provider, timestamp, titles, names, person ids). Snippets are produced by
decrypting only the ~20 documents actually being displayed, at request time.

That is why `search_docs` is a safe derived store even when the rest of the
content is sealed: it cannot leak message text, because it does not contain any.
The alternative - keeping a plaintext copy of every body for highlighting - is
what turns a search index into the largest plaintext store in the system, and
it is the mistake that makes "we encrypt your data" untrue.

Retrieval is hybrid (blueprint 8.3):
    lexical (BM25)  +  semantic (hashed embeddings, optional)
fused with Reciprocal Rank Fusion, which needs no score calibration between
the two systems. That matters: BM25 scores are unbounded and cosine scores are
in [-1,1], and any weighted-sum fusion of the two is a tuning exercise that
silently degrades. RRF only uses *ranks*, so it cannot break that way.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.config import get_settings
from omnilinker.ids import prefixed
from omnilinker.normalize import snippet as make_snippet
from omnilinker.search.index import LexicalIndex, tokenize
from omnilinker.search.query import SearchRequest, parse
from omnilinker.store import get_stores
from omnilinker.store.base import Filter, Query

SEARCH_DOC_COLLECTION = "search_docs"
RRF_K = 60

#: Semantic admission gate. See `_semantic_search` for why this is statistical
#: rather than a fixed threshold.
SEMANTIC_Z = 2.5
SEMANTIC_ABS_FLOOR = 0.15

#: How many fused candidates to retrieve before paginating. Also the ceiling
#: on the reported `total`; if the window saturates we say so rather than
#: reporting a fake exact count.
CANDIDATE_WINDOW = 500


# ---------------------------------------------------------------------------
# Search document projection
# ---------------------------------------------------------------------------

#: kind -> which collections feed the index
SOURCE_COLLECTIONS = {
    "message": ["messages"],
    "email": ["messages"],
    "file": ["files"],
    "note": ["notes"],
    "video": ["videos"],
    "transcript": ["transcripts"],
}


def project_search_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The derived, content-free projection used for ranking.

    Deliberately excluded: message bodies, note text, file extracted text,
    transcript text, email addresses, phone numbers. If a field is not needed
    to *rank* or *filter*, it does not belong here.
    """
    kind = str(doc.get("kind") or "message")
    extra = doc.get("extra") or {}
    entities = doc.get("entities") or {}
    return {
        "_id": doc["_id"],
        "workspace_id": doc.get("workspace_id", ""),
        "kind": kind,
        "provider": doc.get("provider", ""),
        "ts": doc.get("ts") or doc.get("modified_ts") or doc.get("last_edited") or "",
        "title": _title_of(doc, extra),
        "subject": extra.get("subject", "") if kind == "email" else "",
        "sender_name": doc.get("sender_name", ""),
        "display_name": doc.get("display_name", ""),
        "name": doc.get("name", ""),
        "channel": extra.get("channel_label") or extra.get("channel") or "",
        "labels": extra.get("labels", []) if isinstance(extra.get("labels"), list) else [],
        "file_name": doc.get("name", ""),
        "conversation_id": doc.get("conversation_id", ""),
        "person_ids": list(entities.get("person_ids") or []) + ([str(doc["person_id"])]
                                                                 if doc.get("person_id") else []),
        "body_hash": doc.get("body_hash", ""),
        "attachment_count": len(doc.get("attachments") or []),
        "has_attachment": bool(doc.get("attachments")),
        "deadline_count": len(entities.get("deadlines") or []),
        "importance": float(doc.get("importance") or 0.0),
        "lineage_provider": doc.get("provider", ""),
    }


def _title_of(doc: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
    kind = str(doc.get("kind") or "message")
    if kind == "note":
        return str(doc.get("title", ""))
    if kind == "file":
        return str(doc.get("name", ""))
    if kind == "video":
        return str(doc.get("title", ""))
    if kind == "message":
        subject = extra.get("subject")
        if subject:
            return str(subject)
        # Deliberately empty, and deliberately NOT the channel name.
        #
        # Two reasons. `thread_subject` is sealed, so reading it here would copy
        # an envelope into the content-free store - which `str()`s into the index
        # and makes every message match every query. And a channel name is a
        # *location*, not a title: falling back to it meant every Slack result
        # read "engineering" or "incident-4417" while the derived subject - a
        # real, specific label - sat unused. The channel is still shown in the
        # result's meta line; it just is not pretending to be a title.
        return ""
    return str(doc.get("title") or doc.get("name") or "")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    doc_id: str
    score: float
    kind: str
    provider: str
    ts: str
    title: str
    sender_name: str = ""
    snippet: str = ""
    highlight_terms: list[str] = field(default_factory=list)
    person_ids: list[str] = field(default_factory=list)
    conversation_id: str = ""
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    rrf_score: float = 0.0
    has_attachment: bool = False
    deadline_count: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 5),
            "rrf_score": round(self.rrf_score, 5),
            "kind": self.kind,
            "provider": self.provider,
            "ts": self.ts,
            "title": self.title,
            "sender_name": self.sender_name,
            "snippet": self.snippet,
            "highlight_terms": self.highlight_terms,
            "person_ids": self.person_ids,
            "conversation_id": self.conversation_id,
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
            "has_attachment": self.has_attachment,
            "deadline_count": self.deadline_count,
        }


@dataclass
class SearchResponse:
    hits: list[SearchHit]
    total: int
    total_is_exact: bool
    took_ms: int
    facets: dict[str, list[dict]]
    parsed: dict
    engine: str
    index_stats: dict
    degraded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "total": self.total,
            "total_is_exact": self.total_is_exact,
            "took_ms": self.took_ms,
            "facets": self.facets,
            "parsed_query": self.parsed,
            "engine": self.engine,
            "index": self.index_stats,
            "degraded": self.degraded,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SearchService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.stores = get_stores()
        self._index: LexicalIndex | None = None
        self._index_version: str | None = None
        self._vectors: dict[str, list[float]] | None = None

    # -- index lifecycle ------------------------------------------------
    def index_version(self) -> str:
        """Cheap change detector. The index is rebuilt when any source moves.

        Counts only. It will miss an in-place content edit that does not change
        a row count, which is the trade for not walking every document on every
        query; `reindex()` is called explicitly after a sync for that reason.
        """
        docs = self.stores.docs
        collections = sorted({c for cs in SOURCE_COLLECTIONS.values() for c in cs})
        return "|".join(f"{name}:{docs.count(name)}" for name in collections)

    def reindex(self, *, persist: bool = False) -> dict[str, Any]:
        """Rebuild both derived stores from documents.

        This is the only place content is decrypted in bulk, and it decrypts
        each document exactly once - the text is handed to the lexical index
        and the vector builder, then dropped. Query time never decrypts for
        ranking, only for the ~20 snippets on screen.

        `persist` is False by default. The inverted index is plaintext-bearing
        (it must be, to rank), so writing it to disk would create a second
        plaintext store; see `LexicalIndex.save` for the k-anonymity trade.
        """
        t0 = time.perf_counter()
        docs = self.stores.docs

        # 1. content-free projection: filters, facets, display fields. Safe to
        #    persist, safe to serve to the browser, contains no message text.
        written = 0
        projected: dict[str, dict] = {}
        for collection in sorted({c for cs in SOURCE_COLLECTIONS.values() for c in cs}):
            for row in docs.find(Query(collection, limit=200_000)):
                doc = project_search_doc(row)
                projected[doc["_id"]] = doc
        if projected:
            docs.put_many(SEARCH_DOC_COLLECTION, list(projected.values()), "_id")
            written = len(projected)
        # Drop projections for documents that no longer exist, or a deleted
        # message would keep appearing in search results forever.
        live = set(projected)
        for stale in docs.find(Query(SEARCH_DOC_COLLECTION, limit=200_000)):
            if stale["_id"] not in live:
                docs.delete(SEARCH_DOC_COLLECTION, stale["_id"])

        # 2. content-bearing structures, built in memory from one decrypt pass.
        index = LexicalIndex()
        vectors: dict[str, list[float]] = {}
        dim = _embed_dim(self.settings.embed_model)
        want_vectors = self.settings.enable_vector
        for doc_id, row in projected.items():
            body, title = self._decrypted_view(doc_id)
            index.add_document(row, body=body, title=title)
            if want_vectors:
                vectors[doc_id] = self._embed_row(row, body, dim, title)
        index.finalize()

        self._index = index
        self._vectors = vectors if want_vectors else None
        self._index_version = self.index_version()

        stats = {
            "search_docs": written,
            "build_ms": int((time.perf_counter() - t0) * 1000),
            "vectors": len(vectors) if want_vectors else 0,
            **index.stats(),
        }
        if persist or self.settings.persist_index:
            try:
                stats["persisted"] = index.save(
                    self.settings.data_dir / "index" / "lexical.json",
                    k_anonymity=self.settings.index_k_anonymity,
                )
            except OSError as exc:  # a read-only data dir must not break search
                stats["persist_error"] = str(exc)
        return stats

    def _embed_row(self, row: Mapping[str, Any], body: str, dim: int,
                   extra_title: str = "") -> list[float]:
        """One document's vector. Title text is repeated because it is a better
        summary of the document than its first paragraph; sender and channel
        twice, because "who said it where" is half the signal in a chat corpus.
        """
        from omnilinker.embeddings import hash_embed

        title = " ".join(str(row.get(k) or "") for k in ("title", "subject", "name"))
        if extra_title:
            title = f"{title} {extra_title}".strip()
        sender = str(row.get("sender_name") or "")
        channel = str(row.get("channel") or "")
        return hash_embed(" ".join([title, title, title, sender, sender, channel, body]), dim)

    def ensure_index(self) -> LexicalIndex:
        if self._index is None or self._index_version != self.index_version():
            self.reindex(persist=False)
        assert self._index is not None
        return self._index

    def drop_index(self) -> None:
        """Explicitly destroy the derived stores. Part of the privacy story
        (blueprint 10.4) and the fastest way to prove the data is still there."""
        self._index = None
        self._vectors = None
        self.stores.docs.purge([SEARCH_DOC_COLLECTION])
        path = self.settings.data_dir / "index" / "lexical.json"
        path.unlink(missing_ok=True)
        return {"dropped": True, "collections": [SEARCH_DOC_COLLECTION]}

    # -- retrieval ------------------------------------------------------
    def search(
        self,
        request: SearchRequest,
        *,
        query_text: str | None = None,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        parsed = parse(query_text if query_text is not None else request.q)
        index = self.ensure_index()
        degraded: list[str] = []

        allowed = self._allowed_ids(request, parsed)
        if allowed is not None and not allowed:
            return SearchResponse([], 0, True,
                                  int((time.perf_counter() - t0) * 1000), {},
                                  parsed.as_dict(), "lexical", index.stats(),
                                  ["filters excluded everything"])

        window = max(CANDIDATE_WINDOW, request.limit * 20)
        # Phrases and exclusions are meaningful in both query styles; bare
        # terms depend on whether this is a question or a keyword query.
        lex_query = f"{' '.join(parsed.phrases)} {' '.join(parsed.retrieval_terms)}".strip()
        # -- lexical ------------------------------------------------------
        lexical = index.search(
            lex_query, limit=window, allowed_docs=allowed,
        )
        lexical = [(doc_id, score) for doc_id, score in lexical
                   if not _excluded(index.doc_meta.get(doc_id, {}), parsed.exclude)]
        for rank, (doc_id, _) in enumerate(lexical, start=1):
            index.doc_meta.setdefault(doc_id, {})["_lexical_rank"] = rank

        # -- semantic (optional) -------------------------------------------
        semantic: list[tuple[str, float]] = []
        if self.settings.enable_vector:
            try:
                semantic = self._semantic_search(lex_query, limit=window, allowed=allowed)
            except Exception as exc:  # a broken vector path must not kill search
                degraded.append(f"semantic unavailable: {exc}")
        if not semantic and self.settings.enable_vector:
            degraded.append("vector index empty; lexical only")

        # -- fusion --------------------------------------------------------
        fused = reciprocal_rank_fusion(
            {doc_id: score for doc_id, score in lexical},
            {doc_id: score for doc_id, score in semantic} if semantic else None,
            k=RRF_K if semantic else None,  # no second list -> plain lexical order
        )

        # -- build hits ----------------------------------------------------
        rows = self.stores.docs.get_many(SEARCH_DOC_COLLECTION, [d for d, _ in fused])
        by_id = {r["_id"]: r for r in rows}
        hits: list[SearchHit] = []
        highlight_terms = parsed.terms + parsed.phrases
        for doc_id, (score, lex_rank, sem_rank) in fused[: request.limit + request.offset]:
            row = by_id.get(doc_id)
            if not row:
                continue
            body, derived_title = self._decrypted_view(doc_id)
            hit = SearchHit(
                doc_id=doc_id,
                score=score,
                rrf_score=score,
                kind=row.get("kind", ""),
                provider=row.get("provider", ""),
                ts=row.get("ts", ""),
                # Preference order: a title the projection could hold, then the
                # decrypted derived subject. A Slack message has no
                # provider-given title, so without the second fallback every
                # result row would read as a channel name.
                title=(row.get("title") or row.get("subject") or row.get("name")
                       or derived_title or row.get("channel") or ""),
                sender_name=row.get("sender_name", ""),
                snippet=make_snippet(body, " ".join(highlight_terms)) if body else "",
                highlight_terms=highlight_terms,
                person_ids=list(row.get("person_ids") or []),
                conversation_id=row.get("conversation_id", ""),
                lexical_rank=lex_rank,
                semantic_rank=sem_rank,
                has_attachment=bool(row.get("has_attachment")),
                deadline_count=int(row.get("deadline_count") or 0),
            )
            hits.append(hit)
        page = hits[request.offset: request.offset + request.limit]

        # `total` is the fused candidate count, not the page length. It is exact
        # while the window is unsaturated and explicitly flagged when it is not,
        # because a silently-wrong total is what breaks infinite scroll.
        saturated = len(fused) >= window
        return SearchResponse(
            hits=page,
            total=len(fused),
            total_is_exact=not saturated,
            took_ms=int((time.perf_counter() - t0) * 1000),
            facets=self._facets([h.doc_id for h in hits], parsed),
            parsed=parsed.as_dict(),
            engine="hybrid" if semantic else "lexical",
            index_stats=index.stats(),
            degraded=degraded,
        )

    # -- helpers --------------------------------------------------------
    def _allowed_ids(self, request: SearchRequest, parsed) -> set[str] | None:
        filters: list[Filter] = [Filter.from_dict(d) for d in request.filter_dicts()]
        for key, values in parsed.filters.items():
            if key in ("after", "before"):
                field_name, op = ("ts", "gte") if key == "after" else ("ts", "lte")
                filters.append(Filter(field=field_name, op=op, value=values[0]))
            elif key == "has":
                filters.append(Filter(field="has_attachment", op="eq",
                                      value=values[0] in ("file", "files", "attachment")))
            elif key == "type":
                filters.append(Filter(field="kind", op="in", value=values))
            elif key == "in":
                filters.append(Filter(field="conversation_id", op="in", value=values))
            elif key == "person":
                filters.append(Filter(field="person_ids", op="in", value=values))
            elif key == "provider":
                filters.append(Filter(field="provider", op="in", value=values))
        if not filters:
            return None
        return {
            r["_id"] for r in self.stores.docs.find(
                Query(SEARCH_DOC_COLLECTION, filters=filters, limit=200_000)
            )
        }

    def _decrypted_view(self, doc_id: str) -> tuple[str, str]:
        """Decrypt one document into (body, title).

        The title is the derived thread subject, which is sealed, and the index
        needs it to rank Slack messages - which have no provider-given subject
        and are otherwise nearly unrankable by title. It is decrypted here, in
        memory, for the build, and never written to the content-free store.
        """
        from omnilinker.pipeline import IngestPipeline

        for collection in ("messages", "notes", "files", "transcripts", "videos"):
            doc = self.stores.docs.get(collection, doc_id)
            if not doc:
                continue
            try:
                plain = IngestPipeline().decrypt_record(doc)
            except Exception:
                # One undecryptable row must degrade to "no body", not fail the
                # whole reindex.
                return "", ""
            chunks: list[str] = []
            for key in ("title", "subject", "name", "body_text", "text", "description"):
                value = plain.get(key)
                # A still-sealed value is a Mapping, not a string. Coercing it
                # with str() would push envelope JSON into the index and make
                # the search match on ciphertext - i.e. match everything.
                if isinstance(value, str) and value.strip():
                    chunks.append(value)
                elif isinstance(value, (list, tuple)):
                    chunks.extend(str(v) for v in value
                                  if isinstance(v, (str, int, float)))
            body = "\n".join(chunks)
            title = str(plain.get("thread_subject") or "")
            return body, title
        return "", ""

    def _decrypted_text(self, doc_id: str) -> str:
        """Decrypt exactly one document, for snippet purposes only."""
        from omnilinker.pipeline import IngestPipeline

        for collection in ("messages", "notes", "files", "transcripts", "videos"):
            doc = self.stores.docs.get(collection, doc_id)
            if doc:
                try:
                    plain = IngestPipeline().decrypt_record(doc)
                except Exception:
                    # A single undecryptable row (rotated key, corrupted AAD)
                    # must degrade to "no snippet", not fail the search.
                    return ""
                chunks: list[str] = []
                for key in ("title", "subject", "name", "body_text", "text", "description"):
                    value = plain.get(key)
                    # A still-sealed value is a Mapping, not a string. Coercing
                    # it with str() would push envelope JSON into the index and
                    # make the search match on ciphertext - which is how you get
                    # a "result" for every document in the corpus.
                    if isinstance(value, str) and value.strip():
                        chunks.append(value)
                    elif isinstance(value, (list, tuple)):
                        chunks.extend(str(v) for v in value if isinstance(v, (str, int, float)))
                return "\n".join(chunks)
        return ""

    def _facets(self, doc_ids: Sequence[str], parsed) -> dict[str, list[dict]]:
        if not doc_ids:
            return {}
        rows = self.stores.docs.get_many(SEARCH_DOC_COLLECTION, doc_ids)
        buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            buckets["provider"][str(row.get("provider") or "unknown")] += 1
            buckets["kind"][str(row.get("kind") or "message")] += 1
            if row.get("has_attachment"):
                buckets["has_attachment"]["with_files"] += 1
            year = str(row.get("ts") or "")[:4]
            if year.isdigit():
                buckets["year"][year] += 1
        return {
            name: [{"value": value, "count": count}
                   for value, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
            for name, counter in buckets.items()
        }

    # -- semantic (optional) --------------------------------------------
    def _build_vectors(self, doc_ids: Sequence[str]) -> dict[str, list[float]]:
        """Hashed bag-of-words embeddings over the *content*.

        Built at reindex time, which is the only moment decryption is cheap
        enough to do in bulk: one pass, one decrypt per document, then the
        vectors are held in memory. At query time nothing is decrypted for
        ranking - only the ~20 snippets are.

        Not a neural model, and not pretending to be one. See
        `omnilinker.embeddings` for exactly what this does and does not
        capture. `AI_MODE` / a real encoder replaces it wholesale: the
        interface is a list of floats and nothing downstream branches on which
        one produced it.
        """
        dim = _embed_dim(self.settings.embed_model)
        vectors: dict[str, list[float]] = {}
        for doc_id in doc_ids:
            row = self.stores.docs.get(SEARCH_DOC_COLLECTION, doc_id)
            if row:
                body, title = self._decrypted_view(doc_id)
                vectors[doc_id] = self._embed_row(row, body, dim, title)
        return vectors

    def _semantic_search(self, text: str, *, limit: int,
                         allowed: set[str] | None) -> list[tuple[str, float]]:
        from omnilinker.embeddings import hash_embed

        if self._vectors is None:
            self._vectors = self._build_vectors(
                [r["_id"] for r in self.stores.docs.find(
                    Query(SEARCH_DOC_COLLECTION, limit=200_000))]
            )
        if not self._vectors:
            return []
        dim = len(next(iter(self._vectors.values())))
        query_vec = hash_embed(text, dim)
        scored: list[tuple[str, float]] = []
        for doc_id, vec in self._vectors.items():
            if allowed is not None and doc_id not in allowed:
                continue
            scored.append((doc_id, _cosine(query_vec, vec)))
        if not scored:
            return []
        # **Statistical admission gate, not a fixed threshold.**
        #
        # A hashed embedding space has an irreducible noise floor: two unrelated
        # documents collide into the same buckets, and the cosine of two random
        # unit vectors is ~N(0, 1/sqrt(dim)). At dim=1024 that is sigma ~= 0.031,
        # so a hard-coded cutoff is wrong twice: too permissive and a nonsense
        # query returns noise, too strict and a good query on a large corpus
        # discards real matches. Worse, the floor moves with corpus size and
        # dimensionality, so any constant is only correct for one deployment.
        #
        # So the gate is computed from the score distribution itself: keep only
        # documents at least SEMANTIC_Z standard deviations above the mean. A
        # nonsense query produces a flat noise distribution and nothing clears
        # the bar; a real query produces a long right tail and the genuine
        # matches clear it comfortably. It needs no tuning per deployment.
        scores_only = [sc for _, sc in scored]
        mean = sum(scores_only) / len(scores_only)
        variance = sum((sc - mean) ** 2 for sc in scores_only) / max(1, len(scores_only) - 1)
        sigma = math.sqrt(variance)
        floor = max(SEMANTIC_ABS_FLOOR, mean + SEMANTIC_Z * sigma)

        kept = [(d, sc) for d, sc in scored if sc >= floor]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        return kept[:limit]


def _embed_dim(model: str) -> int:
    for part in str(model).split("-"):
        if part.isdigit():
            return int(part)
    return 1024


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _excluded(meta: Mapping[str, Any], exclude: Sequence[str]) -> bool:
    if not exclude:
        return False
    haystack = " ".join(str(meta.get(k) or "") for k in ("title", "sender_name")).lower()
    return any(term in haystack for term in exclude)


def reciprocal_rank_fusion(
    lexical: Mapping[str, float],
    semantic: Mapping[str, float] | None = None,
    *,
    k: int | None = RRF_K,
) -> list[tuple[str, tuple[float, int | None, int | None]]]:
    """Fuse ranked lists by rank position only.

        rrf(d) = sum_over_lists  1 / (k + rank_i(d))

    With one list, RRF degenerates to a monotonic transform of that list, so we
    skip it and keep the original BM25 scores - the numbers a user would expect
    to see in a debug panel.
    """
    if not semantic or k is None:
        ordered = sorted(lexical.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(doc_id, (score, rank, None)) for rank, (doc_id, score) in enumerate(ordered, 1)]

    lex_order = [d for d, _ in sorted(lexical.items(), key=lambda kv: (-kv[1], kv[0]))]
    sem_order = [d for d, _ in sorted(semantic.items(), key=lambda kv: (-kv[1], kv[0]))]
    lex_rank = {d: i for i, d in enumerate(lex_order, 1)}
    sem_rank = {d: i for i, d in enumerate(sem_order, 1)}

    scores: dict[str, float] = defaultdict(float)
    for doc_id in set(lex_rank) | set(sem_rank):
        if doc_id in lex_rank:
            scores[doc_id] += 1.0 / (k + lex_rank[doc_id])
        if doc_id in sem_rank:
            scores[doc_id] += 1.0 / (k + sem_rank[doc_id])

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(d, (s, lex_rank.get(d), sem_rank.get(d))) for d, s in ordered]


_service: SearchService | None = None


def get_search_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


def reset_search_service() -> None:
    """Test hook."""
    global _service
    _service = None
