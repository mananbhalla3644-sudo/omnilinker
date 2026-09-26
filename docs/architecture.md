# Architecture

How the pieces fit, and why the boundaries are where they are.

## The dependency rule

```
api  →  engine  →  { pipeline, projection, identity, search, ai }
                     ↓                    ↓
                   store                crypto
```

Nothing below the `api` layer knows that HTTP exists, and nothing below
`engine` knows what a request is. That is not architectural purity for its own
sake — it is what makes the pipeline, the resolver and the indexer testable as
plain functions, which is why the suite runs in three seconds with no event loop
and no mocks.

The one exception is documented where it occurs: `MongoDocStore` and
`Neo4jGraphStore` own an asyncio loop internally, because the alternative was
dragging async through every layer above them for no benefit — ingestion is
sequential and bounded by provider rate limits anyway.

## The three-layer pipeline

### 1. Connectors

One interface, eleven implementations:

```python
class Connector(ABC):
    def discover_streams(self, grant, user_ref) -> list[StreamDescriptor]
    def pull(self, grant, stream, cursor, page) -> PullPage
    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]   # pure
```

Only those three are abstract. The OAuth methods have defaults that raise,
because a WhatsApp export has no authorization URL and forcing every file
connector to write a stub that says "not applicable" is noise.

`normalize()` is the only method that needs a golden payload corpus, because it
is the only place provider schema knowledge exists. It is also the only method
the registry inspects for purity (check C4), because a normalizer that makes an
HTTP call is invisible at review time and turns a replayable function into a
network client.

`register()` is a gate, not a dict assignment. Five checks, each for a failure
that has caused a real incident:

| | Check | Failure it prevents |
|---|---|---|
| C1 | every scope has a justification | a permission list nobody can evaluate |
| C2 | rate limit declared and plausible | self-inflicted bans |
| C3 | incremental ⇒ cursor named | every sync is a full re-ingest |
| C4 | `normalize` declares no I/O | a "pure" function that is a network client |
| C5 | semver, bumped on normalizer change | silent schema drift |

### 2. The pipeline

```
pull → raw artifact → normalize → derive → seal → persist → cursor → audit
```

Two invariants everything else depends on:

**Idempotency.** The document id is `f"{provider}:{provider_key}"` where
`provider_key` is the provider's own stable identifier. Re-running a sync over
an overlapping window overwrites in place and creates nothing new. This is what
makes at-least-once delivery safe, which is the only kind of delivery a provider
can honestly offer.

**Cursors advance only on full success.** A partial failure leaves the cursor
where it was, so the next run re-reads the window. Re-reading is free (upsert);
skipping data is not.

Order matters: the raw artifact is written *before* normalization, so if a
normalizer crashes on a payload shape nobody has seen, the evidence survives and
the bug is reproducible from the stored artifact.

### 3. Derived stores

`engine.rebuild()` runs six stages in a dependency order that is a DAG, not a
preference:

```
identities → persons → graph → insights → search → predictions
```

- **identities first** — produced by ingestion, consumed by resolution.
- **persons before the graph** — the projector emits `SENT_BY → Person`, so
  persons must exist or every message is wired to a placeholder.
- **graph before insights** — `write_insight_edges` only materializes a
  detector's finding if both endpoints already exist as nodes.
- **search after enrichment** — the index must contain `importance` and
  `thread_subject`, or the first query after a sync disagrees with the second.
- **predictions last** — they read decrypted bodies and resolved people.

Every stage is idempotent and independently re-runnable. A failed stage is
reported and the rest continue: one broken derived store must not leave the
system unusable, and `/api/system/health` returns 503 so a monitor sees it.

## Why the graph is derived

`projection.py` treats the graph as a pure function of the documents. That one
decision buys three things:

1. **Re-resolution is safe.** Change the merge threshold and you re-project,
   rather than surgery on live edges.
2. **Bugs are repairable.** A wrong edge is fixed by rebuilding, not by a
   migration.
3. **The privacy claim is executable.** `POST /api/admin/drop-derived` destroys
   every derived store and rebuilds it, which turns "your data is still there"
   from a claim into an operation.

The cost is that the graph cannot be the source of truth for anything, and the
projector has to be able to retract an edge it previously wrote — which is why
`delete_edge` is on the `GraphStore` interface rather than being an embedded-only
convenience.

## Entity resolution

Four stages, each of which exists to make the next one tractable:

**Blocking** (`identity/blocking.py`) — compares only pairs sharing a cheap
high-precision key. On 500k identities, all-pairs is 1.25 × 10¹¹ comparisons;
blocking reduces it to a few thousand. Two details matter more than the
algorithm:

- Blocks larger than `MAX_BLOCK_SIZE` are **reported, not processed**. An
  oversize block almost always means a parser bug — every message attributed to
  a shared bot account — and processing it would silently turn blocking into
  all-pairs while still reporting a healthy reduction ratio.
- There is deliberately no `role:human` key. It looks like a harmless partition
  and it is the exact opposite: every non-contact identity shares it, so the
  block holds all of them. Contact identities are kept apart where it belongs,
  in the scorer, before any feature is considered.

**Scoring** (`identity/resolver.py`) — transparent weighted features, not a
learned classifier, for three reasons: explainability is a product requirement
(the consequence of a merge is a stranger's messages in your profile), the
decisive feature must be *provable* (a shared email token is a fact, not a
probability), and there is no labelled corpus.

Three rules in the scorer are load-bearing and each has a test:

- **Positive evidence saturates at 1.0.** Several weak signals must not add up
  to a confident merge. Only a decisive feature crosses the bar alone.
- **A shared email token is decisive and suppresses weaker conflicts.** Different
  providers having different handles is what every multi-provider person looks
  like, not evidence against. Treating it as a conflict cancelled the merge it
  was meant to qualify, and cross-source resolution returned zero merges while
  still looking entirely healthy.
- **Generic names never merge.** "Admin" on two providers is two admins. This
  is the single most damaging false merge available in this domain.

**Clustering** — union-find over accepted edges, in deterministic order. The
output must be reproducible, or a re-run produces a different clustering and
every downstream test becomes flaky.

**Attribution** — a contact identity (an email address found in a body) is
attached to the Person that owns the token and never becomes a Person itself.
Promoting it would fill the graph with stubs.

## Detectors

Six, each answering one question the user cannot answer by reading a single
source. All output is `state: advisory`.

| | Detector | Guard that keeps it honest |
|---|---|---|
| D1 | dormant relationship | needs ≥ 4 interactions; two messages six months apart is not a relationship that went quiet |
| D2 | collaboration burst | reports a *window*, never "they are close"; confidence capped at 0.85 |
| D3 | hidden introducer | two people who never spoke with ≥ 2 mutual contacts; confidence capped at 0.55, mutual contacts listed so a user can judge in two seconds |
| D4 | single-source person | framed as a *coverage gap*, not a fact about the person |
| D5 | decision divergence | requires an acceptance cue and a rejection cue on the same signature across ≥ 2 sources |
| D6 | orphan file | owner unresolved *and* referenced in conversation |

D3 is the highest-value detector and the most likely to be wrong, which is why
its cap is hard and its evidence is explicit.

## Search

Two legs, fused with Reciprocal Rank Fusion.

**Why RRF and not a weighted sum.** BM25 scores are unbounded; cosine scores are
in [−1, 1]. Any weighted sum of the two is a tuning exercise that silently
degrades as the corpus changes, and there is no principled way to pick the
weight. RRF uses only *ranks*, so it cannot break that way. With a single list
it degenerates to a monotonic transform, so the code keeps the original BM25
scores instead — the numbers a user expects to see in a debug panel.

**Query analysis is two-mode.** A keyword search for "to be" should find "to be";
a question is mostly function words and scoring on them ranks documents by
coincidence. So the parser detects natural language — a question opener, or ≥ 40%
stopwords over ≥ 3 tokens — and uses the right term set for each, reporting what
it dropped. When in doubt it keeps the keyword reading, because that never
silently loses a term the user typed on purpose.

**NL2Query compiles, it does not translate.** The output is a typed
`QueryPlan` whose every field maps onto something the store can execute. If a
question needs something the system cannot express, the plan says so in
`unsupported` rather than quietly dropping the clause — because a generator that
silently discards the part it cannot handle answers a *different* question than
the one asked, and returns confident wrong results. "Show me the budget
spreadsheet Alice sent in March" degrading to a full-text search for "budget"
returns the budget *discussion*: plausible, and not the answer.

The plan is returned to the client alongside the results, and the UI renders it.
Showing the plan is not a debug affordance; it is how a user learns to trust a
system that guessed at what they meant.

## Crypto

```
UMK (file, 0600, never leaves the trust boundary)
  └─ workspace KEK         = HKDF(UMK, "omnilinker/wkek/v1|ws")
       ├─ per-source DEK   = HKDF(WKEK, "omnilinker/dek/v1|<source>")   → content
       └─ identity pepper  = HKDF(WKEK, "omnilinker/identity-pepper/v1|ws")
                                                                   → identity tokens
```

Every derivation is bound to a versioned `info` string, so changing the scheme
later means a new string and old keys keep working.

The two second-level keys have different jobs and must stay separate:

- **Per-source DEKs** encrypt content. Sharing one across sources would mean a
  single compromised provider exposes every source.
- **The identity pepper is workspace-wide**, because comparison across sources is
  the entire purpose and tokens derived from a per-source pepper can never match
  across sources. It never encrypts anything, and losing it costs only the
  ability to re-derive tokens.

**AAD** is `<workspace>|<doc_id>|<field_path>|<schema_version>`, authenticated by
GCM. An attacker with database write but not the key cannot move a ciphertext
between documents, fields or workspaces — the tag check fails.

## Privacy boundaries, in one place

| Data | At rest | In the derived store | In the API |
|---|---|---|---|
| message / note / transcript body | sealed | never | decrypted per document, on read |
| file extracted text | sealed | never | decrypted per document, on read |
| email subject, note title | **plaintext** (ADR-025) | plaintext | plaintext |
| sender name, provider, ts, kind | plaintext | plaintext | plaintext |
| email / phone / name tokens | keyed HMAC | never | presence only |
| identity pepper | in the keystore | never | fingerprint only |

The one row that is a judgement call is the subject line, and
[ADR-025](adr/ADR-025-search-store-privacy.md) states the cost rather than
minimising it.
