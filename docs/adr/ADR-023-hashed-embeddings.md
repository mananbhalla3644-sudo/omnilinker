# ADR-023: hashed embeddings as the second retrieval leg

**Status:** accepted · **Area:** search · **Reversible:** yes, configuration

## The blueprint specifies

A dense vector index (HNSW) over neural sentence embeddings, fused with BM25.

## The default is

A hashed bag-of-words embedding — signed hashing, 1024 dimensions, sublinear tf,
word stems plus morphological 4-grams — exhaustively scanned for cosine
similarity. `OMNI_VECTOR=0` turns the second leg off entirely.

## Why

The parts of the blueprint's retrieval design that carry the precision@10 target
are the *fusion*, the *filters*, the *facets* and the *evaluation harness* —
and all four are identical whichever vector source feeds them. What a neural
model adds is synonymy. A hashed embedding adds morphological and phrasing
variation, which is a smaller but real gain, and it needs no model, no GPU, no
network and no API key.

Three details in the implementation are what make it useful rather than
decorative:

- **Signed hashing.** Unsigned hashing makes collisions *add*, so a document full
  of frequent terms gets a huge magnitude and cosine collapses toward 1. Signed
  hashing makes collisions cancel, which is what keeps unrelated documents near
  orthogonal.
- **Sublinear tf** (`1 + log(tf)`), because repeated words add information with
  diminishing returns and raw tf lets one term outvote all content.
- **A stoplist for the most common 4-grams.** `ing$`, `tion`, `ed$` match almost
  every word longer than six characters, so an unweighted n-gram scheme makes
  every pair of English sentences look similar. The list is small and the
  mechanism is blunt on purpose: a gram this common carries no information.

## The admission gate is statistical, not a threshold

A hashed space has an irreducible noise floor — two unrelated documents collide
into the same buckets, and the cosine of two random unit vectors is
~N(0, 1/√dim) ≈ 0.031 at dim 1024. So a hard-coded cutoff is wrong twice: too
permissive and a nonsense query returns noise, too strict and a good query on a
large corpus discards real matches. Worse, the floor moves with corpus size.

Instead the gate is computed from the score distribution itself: keep documents
at least 2.5σ above the mean. A nonsense query produces a flat noise
distribution and nothing clears the bar; a real query produces a long right tail
and the genuine matches clear it comfortably. No per-deployment tuning, and
`test_search.py::test_nonsense_query_returns_nothing` pins the behaviour.

## Residual risk

**No synonymy, and this is the largest retrieval-quality gap in the system.**
`shard key` and `partitioning strategy` are orthogonal to it. A real encoder
would close that gap and nothing else in the architecture would need to change.

**Exhaustive scan, not HNSW.** O(n) per query, fine to ~50k documents, not to
5M. This is the reason the blueprint's p95 ≤ 2 s target is *architecturally*
reachable here but not *measured* at that scale.

**The swap is a one-function change.** `embeddings.embed()` returns
`list[list[float]]`, `SearchService` has no branch on which model produced a
vector, and the store does not know embeddings exist. The interface is a list of
floats.
