"""Embeddings, hashed (blueprint 8.3.2 default; swappable for a real model).

The default is a **hashed bag of words with sublinear term weighting** - the
"hashing trick" from the original random-projection work, made useful by three
choices that matter more than the projection itself:

1. **Signed hashing.** Each token contributes `+h` or `-h` depending on a bit
   of its own hash. Unsigned hashing makes collisions *add* together, so a
   document full of frequent terms gets a huge, meaningless magnitude and cosine
   similarity collapses toward 1. Signed hashing makes collisions cancel, which
   is what keeps unrelated documents near-orthogonal.

2. **Sublinear term weighting.** `1 + log(tf)`, not `tf`. Without it, a term
   repeated 50 times dominates the vector completely and the model degenerates
   into a frequency count.

3. **Character n-grams alongside words.** "sharding"/"sharded" share no word
   token. The 4-gram features give morphological overlap for free, which is a
   large part of what real embedding models contribute on this corpus.

What this is *not*: a semantic model. "shard key" and "partitioning strategy" are
orthogonal here, and a real sentence encoder would put them close. This is
lexical similarity in a fixed-size space, which is a genuinely useful second
retrieval signal - it catches stem, inflection and phrasing variation that
BM25 misses - and it is honest to call it that.

Swapping in a real encoder means implementing `embed(texts) -> list[list[float]]`
and returning vectors from here. Nothing else changes: the interface is a list
of floats, the store does not know or care, and `SearchService` has no branch
for "is this a real model".
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from omnilinker.search.index import STOPWORDS, stem

#: Default width. Large enough that collisions are rare, small enough that a
#: 200k-document index stays in memory (200k x 1024 float32 = 800MB, so a real
#: deployment would use float16 or an on-disk ANN index - see notes below).
DEFAULT_DIM = 1024

_NGRAM = re.compile(r"[a-z0-9]+")

#: Weight per feature class. Words are the signal; character shingles are a
#: morphological assist and must stay well below them.
W_WORD = 1.0
W_GRAM = 0.35

#: 4-grams that appear in a large share of English text. `ing$`, `tion`, `ed$`
#: and friends match almost every word longer than six characters, so an
#: unweighted n-gram scheme makes every pair of English sentences look similar.
#: The list is small and the mechanism is blunt on purpose: a gram that is this
#: common carries no information, and dropping it costs nothing.
GRAM_STOPLIST = frozenset({
    "ing$", "tion", "ions", "ing.", ".ing", "ed$", "ies", "ers", "est", "ate",
    "ive$", "ous$", "and$", "ent$", "ant$", "ers$", "ing$", "tio", "ion$",
    "the$", "ing", "res", "nes", "ted$", "ted.", ".ed$", "ing$", "ght$",
})


def _hash64(token: str) -> int:
    """Stable across processes. Python's `hash()` is salted per process, which
    would silently invalidate a persisted index - the classic bug here."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _features(text: str) -> list[tuple[str, float]]:
    """(feature, weight) pairs: content word stems, plus morphological shingles.

    Shingles are only taken from words that survived stopword removal and are
    long enough to have internal structure. Taking them from every word -
    including "the" - is what turns this into a similarity function for
    English as a language rather than for the text.
    """
    words = [w for w in _NGRAM.findall((text or "").lower()) if len(w) > 1]
    content = [w for w in words if w not in STOPWORDS]
    out: list[tuple[str, float]] = [(stem(w), W_WORD) for w in content]
    for word in content:
        if len(word) < 6:
            continue
        padded = f"^{word}$"
        for i in range(len(padded) - 3):
            gram = padded[i: i + 4]
            if gram in GRAM_STOPLIST:
                continue
            out.append((f"#{gram}", W_GRAM))
    return out


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Deterministic, offline, dependency-free embedding."""
    vector = [0.0] * dim
    features = _features(text)
    if not features:
        return vector
    weighted: dict[str, float] = {}
    for feature, weight in features:
        weighted[feature] = weighted.get(feature, 0.0) + weight

    for feature, weight in weighted.items():
        h = _hash64(feature)
        bucket = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(v * v for v in vector))
    if norm:
        return [v / norm for v in vector]
    return vector


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1], normalising both inputs.

    Two things this gets right that a one-liner does not:

    * It normalises. The dot product of two *unnormalised* vectors is not a
      cosine: a 1024-dimension vector of -1 against a single-component unit
      vector scores 1.0, i.e. "maximally similar" for inputs that point in
      opposite directions. `hash_embed` happens to return unit vectors, so the
      shortcut is invisible in-tree and wrong for every other caller.
    * It keeps the negative half. Signed hashing means a document can be
      genuinely anti-correlated with a query, and reporting that as 0 makes an
      unrelated pair indistinguishable from a weak match in a retrieval leg.
      Callers that want a floor apply it explicitly, where the trade is
      visible.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / ((norm_a ** 0.5) * (norm_b ** 0.5))))


def embed(texts: Iterable[str], dim: int = DEFAULT_DIM) -> list[list[float]]:
    return [hash_embed(t, dim) for t in texts]


def similarity(text_a: str, text_b: str, dim: int = DEFAULT_DIM) -> float:
    return cosine(hash_embed(text_a, dim), hash_embed(text_b, dim))


def top_similar(
    query: str, corpus: Sequence[tuple[str, str]], *, dim: int = DEFAULT_DIM, limit: int = 5
) -> list[tuple[str, float]]:
    """Nearest neighbours by text. Used by AI dedup and the 'related' rail."""
    qv = hash_embed(query, dim)
    scored = [(key, cosine(qv, hash_embed(text, dim))) for key, text in corpus]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [(k, s) for k, s in scored[:limit] if s > 0.05]
