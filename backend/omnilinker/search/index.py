"""Lexical retrieval: BM25 over a field-weighted inverted index (blueprint 8).

Why BM25 and not plain TF-IDF: message corpora are brutally skewed. One
person sends 4,000 messages about a project. Plain TF-IDF rewards exactly the
documents you do not want, and BM25's length normalisation plus its saturation
term (k1) is what stops that. It is also the same ranking function the backend
used in the blueprint's evaluation, so the measured precision@10 numbers mean
something.

Field weighting matters as much as the ranking function in this product:
gmail subjects and Notion titles are dense, short and highly informative;
WhatsApp bodies are noisy. Without weights, a long noisy body beats a precise
title and the results look broken to a user even though the math is right.

`stopwords` is deliberately small. Natural-language questions are half made of
stopwords ("where did we", "what is the"), and a stopword list tuned for
keyword search destroys them.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

K1 = 1.5
B = 0.75

#: field -> (weight, is_searchable)
FIELDS: dict[str, tuple[float, bool]] = {
    "title": (3.0, True),
    "subject": (3.0, True),
    "sender_name": (1.6, True),
    "display_name": (2.0, True),
    "labels": (1.2, True),
    "name": (2.0, True),
    "channel": (1.4, True),
    "body": (1.0, True),
    "file_name": (1.8, True),
    "snippet": (0.6, True),
    "provider": (0.8, True),
}

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "what", "when", "where", "which", "who", "whom", "how", "did",
    "does", "do", "was", "were", "have", "has", "had", "i", "we", "you", "me",
    "my", "our", "us", "about", "from", "to", "can", "could", "should",
    "would", "there", "here", "any", "all", "some", "just", "also",
})

_TOKEN = re.compile(r"[A-Za-z0-9_@.'+-]+")
_PHRASE = re.compile(r'"([^"]+)"')


def tokenize(text: str, *, keep_stopwords: bool = True) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN.findall((text or "").lower()):
        token = raw.strip(".'-")
        if not token or len(token) < 2:
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        out.append(stem(token))
    return out


_SUFFIXES = (
    ("ational", "ate"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("ization", "ize"), ("tional", "tion"),
    ("biliti", "ble"), ("ement", ""), ("ments", ""), ("ing", ""),
    ("edly", ""), ("ed", ""), ("ies", "y"), ("ied", "y"), ("sses", "ss"),
    ("es", ""), ("s", ""),
)


def stem(token: str) -> str:
    """A deliberately crude suffix stripper.

    Full Porter would be more correct and less predictable: "us" -> "u",
    "gas" -> "ga". Recall matters more than linguistic elegance here, and the
    index is rebuilt on every model change anyway.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


@dataclass
class Posting:
    doc_id: str
    tf: float
    #: per-field term frequency, so field weights apply at scoring time
    field_tf: dict[str, float] = field(default_factory=dict)
    length: int = 0


@dataclass
class LexicalIndex:
    """Field-weighted inverted index with BM25 scoring.

    **This structure is plaintext-bearing, and that is unavoidable.** You cannot
    rank on content you cannot read, so an inverted index necessarily contains
    the vocabulary of the corpus. The consequences are handled rather than
    ignored:

    * the index lives in **memory only** by default and is rebuilt from
      documents on boot, so it is never a second plaintext store on disk;
    * `save()` is opt-in and applies a **k-anonymity floor** - any term
      appearing in fewer than `k` documents is dropped, because a term that
      appears in exactly one document is a verbatim fragment of that document;
    * `drop_index()` destroys it, and the search *documents* collection (the
      filters, facets and display titles) is content-free by construction.

    See `omnilinker.search.service` for the split between the two derived
    stores and why only one of them can be safely persisted.
    """

    postings: dict[str, dict[str, Posting]] = field(default_factory=lambda: defaultdict(dict))
    doc_lengths: dict[str, dict[str, int]] = field(default_factory=dict)
    avg_length: dict[str, float] = field(default_factory=dict)
    doc_meta: dict[str, dict] = field(default_factory=dict)
    doc_count: int = 0
    k_anonymity: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- build ---------------------------------------------------------
    def add_document(self, doc: Mapping[str, Any], *, body: str = "",
                     title: str = "") -> None:
        """Index one document. `body` is used for term counting only and is
        never retained - it is not stored in `doc_meta`, not written by
        `save()`, and not reachable from this object after this call returns."""
        doc_id = str(doc["_id"])
        with self._lock:
            self.doc_meta[doc_id] = {
                "provider": doc.get("provider", ""),
                "kind": doc.get("kind", ""),
                "ts": doc.get("ts") or doc.get("modified_ts") or doc.get("last_edited") or "",
                "title": doc.get("title") or doc.get("subject") or doc.get("name") or "",
                "sender_name": doc.get("sender_name", ""),
                "person_ids": doc.get("person_ids", []),
                "conversation_id": doc.get("conversation_id", ""),
                "workspace_id": doc.get("workspace_id", ""),
            }
            # Remove any previous posting so re-adding is a replace, not a merge.
            for term, bucket in self.postings.items():
                bucket.pop(doc_id, None)
            self.doc_lengths[doc_id] = {}
            self.doc_count = len(self.doc_meta)
            for field_name, (weight, searchable) in FIELDS.items():
                if not searchable:
                    continue
                value = doc.get(field_name)
                if isinstance(value, (list, tuple)):
                    value = " ".join(str(v) for v in value)
                if field_name == "body":
                    # Content arrives decrypted, in memory, for this call only.
                    value = body
                elif field_name == "title" and title:
                    # A derived thread subject, also sealed at rest. Supplied
                    # here so a Slack message - which has no provider-given
                    # subject and is otherwise unrankable by title - is not
                    # invisible to the ranker.
                    value = f"{value} {title}".strip()
                tokens = tokenize(str(value or ""))
                if not tokens:
                    continue
                self.doc_lengths[doc_id][field_name] = len(tokens)
                for token in tokens:
                    bucket = self.postings[token]
                    posting = bucket.get(doc_id)
                    if posting is None:
                        posting = Posting(doc_id=doc_id, tf=0.0, field_tf={},
                                          length=len(tokens))
                        bucket[doc_id] = posting
                    posting.tf += 1.0
                    posting.field_tf[field_name] = posting.field_tf.get(field_name, 0.0) + 1.0
                    posting.length = len(tokens)

    def finalize(self) -> None:
        """Compute per-field average document lengths. Required for BM25's
        length normalisation - skipping it is the classic silent bug that makes
        a BM25 implementation rank short documents far too highly."""
        with self._lock:
            self.doc_count = max(1, len(self.doc_meta))
            for field_name in FIELDS:
                lengths = [d.get(field_name, 0) for d in self.doc_lengths.values()]
                self.avg_length[field_name] = (sum(lengths) / len(lengths)) if lengths else 1.0
                if self.avg_length[field_name] <= 0:
                    self.avg_length[field_name] = 1.0
            self.postings = {t: b for t, b in self.postings.items() if b}

    # -- scoring -------------------------------------------------------
    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        if n == 0:
            return 0.0
        # BM25 idf, floored at a small positive so a term present in every
        # document contributes ~0 rather than going negative and *subtracting*
        # relevance, which is the classic Robertson/Sparck-Jones idf bug.
        return max(0.05, math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5)))

    def score_document(self, doc_id: str, terms: Sequence[str],
                       boosts: Mapping[str, float] | None = None) -> float:
        boosts = boosts or {}
        total = 0.0
        lengths = self.doc_lengths.get(doc_id, {})
        for term in terms:
            posting = self.postings.get(term, {}).get(doc_id)
            if posting is None:
                continue
            score = 0.0
            for field_name, (weight, _) in FIELDS.items():
                tf = posting.field_tf.get(field_name)
                if not tf:
                    continue
                length = lengths.get(field_name, 0) or 1
                avg = self.avg_length.get(field_name, 1.0) or 1.0
                norm = 1.0 - B + B * (length / avg)
                score += weight * ((tf * (K1 + 1.0)) / (tf + K1 * norm))
            if score:
                total += self.idf(term) * score
        # Phrase and field boosts are multiplicative: they re-rank, they do not
        # compete with the BM25 term contributions on the same scale.
        total *= boosts.get("phrase", 1.0)
        total *= boosts.get("field", 1.0)
        return total

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        allowed_docs: set[str] | None = None,
        boosts: Mapping[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        terms = tokenize(query)
        phrases = _PHRASE.findall(query or "")
        if not terms and not phrases:
            return []
        # Candidates are the union of postings - we never scan all documents.
        candidates: set[str] = set()
        for term in terms:
            candidates |= set(self.postings.get(term, {}).keys())
        scored: list[tuple[str, float]] = []
        for doc_id in candidates:
            if allowed_docs is not None and doc_id not in allowed_docs:
                continue
            doc_boosts = dict(boosts or {})
            if phrases:
                doc_boosts["phrase"] = doc_boosts.get("phrase", 1.0) * self._phrase_boost(
                    doc_id, phrases
                )
            score = self.score_document(doc_id, terms, doc_boosts)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:limit]

    def _phrase_boost(self, doc_id: str, phrases: Sequence[str]) -> float:
        """Adjacency is a strong signal. Approximated from the metadata title
        plus a stored snippet rather than full body, because the body is
        encrypted and the index is a derived store that must stay cheap."""
        meta = self.doc_meta.get(doc_id, {})
        haystack = " ".join(
            str(meta.get(k) or "") for k in ("title", "sender_name")
        ).lower()
        for phrase in phrases:
            if phrase and phrase.lower() in haystack:
                return 2.5
        return 1.0

    # -- introspection --------------------------------------------------
    def stats(self) -> dict[str, Any]:
        terms = len(self.postings)
        total_postings = sum(len(b) for b in self.postings.values())
        return {
            "documents": len(self.doc_meta),
            "terms": terms,
            "postings": total_postings,
            "avg_terms_per_doc": round(total_postings / len(self.doc_meta), 2)
            if self.doc_meta else 0,
            "k1": K1,
            "b": B,
        }

    def terms_for(self, doc_id: str) -> list[str]:
        return [t for t, bucket in self.postings.items() if doc_id in bucket]

    def save(self, path: Path, *, k_anonymity: int = 2) -> dict[str, Any]:
        """Persist the index, dropping terms too rare to be safe.

        **Opt-in, and lossy on purpose.** The default posture (rebuild in
        memory) is better; this exists for deployments that would rather pay
        disk than rebuild latency.

        `k_anonymity` is a privacy floor, not a tuning knob: a term present in
        fewer than k documents is, in effect, a verbatim quote from one
        document. At k=2 a single-document term still leaks that the document
        contains a rare word, so the honest options are k=2 (low latency) or
        not persisting at all (recommended). The caller is told exactly how
        many terms were dropped so the trade is visible rather than assumed.
        """
        self.k_anonymity = max(0, k_anonymity)
        kept: dict[str, dict[str, Any]] = {}
        dropped = 0
        for term, bucket in self.postings.items():
            if self.k_anonymity and len(bucket) < self.k_anonymity:
                dropped += 1
                continue
            kept[term] = {doc_id: {"tf": p.tf, "field_tf": p.field_tf}
                          for doc_id, p in bucket.items()}
        payload = {
            "v": 2,
            "k_anonymity": self.k_anonymity,
            "doc_meta": self.doc_meta,
            "doc_lengths": self.doc_lengths,
            "avg_length": self.avg_length,
            "postings": kept,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), "utf-8")
        tmp.replace(path)
        return {
            "bytes": path.stat().st_size,
            "terms_kept": len(kept),
            "terms_dropped_for_k_anonymity": dropped,
            "k_anonymity": self.k_anonymity,
        }

    def load(self, path: Path) -> bool:
        """Restore a persisted index. Returns False if the file is absent,
        stale or unreadable - the caller then rebuilds, which is always safe."""
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(payload, dict) or payload.get("v") != 2:
            return False
        self.doc_meta = payload.get("doc_meta", {})
        self.doc_lengths = payload.get("doc_lengths", {})
        self.avg_length = payload.get("avg_length", {})
        self.k_anonymity = int(payload.get("k_anonymity", 0))
        self.postings = defaultdict(dict)
        for term, bucket in payload.get("postings", {}).items():
            self.postings[term] = {
                doc_id: Posting(doc_id=doc_id, tf=entry.get("tf", 0.0),
                                field_tf=entry.get("field_tf", {}))
                for doc_id, entry in bucket.items()
            }
        self.doc_count = max(1, len(self.doc_meta))
        return True
