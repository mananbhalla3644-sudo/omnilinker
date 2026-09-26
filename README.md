# OmniLinker

Unified search, a relationship graph, and entity resolution across every source
you own — with field-level envelope encryption on the content and no write-back
to any provider.

This is a working implementation of `OmniLinker_Blueprint.txt`, built to run.
One command, no Docker, no credentials, no external services.

```bash
./run.sh          # macOS / Linux
.\run.ps1         # Windows
```

Then open <http://127.0.0.1:8900> and seed the synthetic workspace from the
Connectors page (or `POST /api/sync` with `{"connector_id": "demo"}`).

---

## What actually works

Everything below is implemented and covered by the test suite. Where something
is a deliberate simplification rather than a finished feature, it says so.

| Area | Status | Notes |
|---|---|---|
| **Connectors** | 11 providers, full contract-tested | Slack, Gmail, Discord, WhatsApp, Google Drive, OneDrive, Dropbox, Notion, Evernote, YouTube, demo. OAuth2 + PKCE, per-provider throttling, retry, incremental cursors. |
| **Ingestion** | working | Immutable raw artifacts → canonical records → sealed → stored. Idempotent, cursor-safe, auditable. |
| **Encryption** | working | AES-256-GCM with AAD bound to `workspace\|doc_id\|field\|schema`. Blocks ciphertext substitution. |
| **Identity resolution** | working | Blocking → weighted scoring → clustering. 4/5 demo people resolved across 5 providers each; ambiguous pairs held as suggestions. |
| **Graph** | working | 9 node types, 12 edge types, bounded traversal, observed vs. inferred edges. |
| **Search** | working | Field-weighted BM25 + hashed embeddings, fused with RRF. Facets, filters, phrase boost, p95 single-digit ms. |
| **NL2Query** | working | Question → typed query plan → result, with the plan shown to the user. |
| **Hidden connections** | 6 detectors | Dormant, burst, hidden introducer, single-source, decision divergence, orphan file. |
| **AI** | working, offline | TextRank summarisation (every sentence is a verbatim span), rule-based predictions, linear importance. |
| **Privacy** | working | Data passport, droppable derived stores, no write-back path in the codebase. |
| **Frontend** | working | Vite + React + TS + D3. Dashboard, search, graph, timeline, people, files, connectors. |
| **Docker profile** | written, untested here | `docker-compose.yml` + Mongo/Neo4j backends implement the same interfaces. Docker is not installed on this machine. |

**Not implemented, and named rather than hidden:**

- No LLM-backed summarisation or NL rewriting. `AI_MODE=local`/`cloud` are the
  upgrade path; the interfaces are unchanged and the deterministic path stays
  available as the baseline the model is measured against.
- No vector ANN index (HNSW). The second retrieval leg is an exhaustive cosine
  scan over in-memory vectors — fine to ~50k documents, not to 5M.
- No multi-user auth. The workspace is single-tenant by design; `OMNI_WORKSPACE`
  namespaces data but there are no accounts.
- No realtime Gateway/WebSocket ingestion. Connectors poll on the cursor path;
  the descriptor declares `realtime_webhooks` for each provider that has it.

### The two recorded deviations

Both are documented in `docs/adr/` and surfaced live at `/api/system/info`:

- **ADR-021** — AES-256-GCM instead of the blueprint's XChaCha20-Poly1305.
  Available everywhere `cryptography` builds, hardware accelerated, and a 96-bit
  nonce is safe under this system's one-key-per-source / random-nonce-per-write
  model. One function in `crypto/envelope.py` to change.
- **ADR-022** — Vite SPA instead of Next.js. The client holds the keys, so
  there is no server to render and no SEO requirement for a private workspace.

Four more (embedded store default, hashed embeddings, extractive AI, the
search-store privacy posture) are in `docs/adr/`, along with
[ADR-026](docs/adr/ADR-026-one-credential-resolver.md) on why every step of an
OAuth handshake resolves the provider's client credentials through one function.

### Checking provider endpoints

```bash
cd backend
python -m tests.check_endpoints_live
```

A wrong `authorize_url` is invisible to the test suite. The app reports the
connector as correctly configured, the Connect button opens a window, and the
provider answers with an error page that never comes back — so there is no
exception, no failed assertion, and nothing in our own logs. Slack's consent URL
was `slack.com/api/oauth.v2/authorize`, wrong twice over: the consent page is
not under `/api`, and the method is `oauth/v2/authorize`. Connect did nothing
at all.

This script makes real requests and reports each provider's verdict. It is not
part of `pytest`, because reaching the internet is not a unit-test concern; the
results are pinned into `tests/test_connections.py` so the suite stays honest
between runs. All nine OAuth connectors currently pass.

---

## Quick tour

```bash
# seed the synthetic workspace: 10 providers, ~90 records, 4 resolved people
curl -XPOST localhost:8900/api/sync \
     -H 'content-type: application/json' -d '{"connector_id":"demo"}'

curl 'localhost:8900/api/search?q=what+did+we+decide+about+the+shard+key&limit=3'
curl 'localhost:8900/api/search?q=shard+key+provider:notion'
curl 'localhost:8900/api/graph/ego?depth=2'
curl -XPOST localhost:8900/api/nl2query \
     -H 'content-type: application/json' \
     -d '{"question":"show me the budget file Alice sent"}'
curl localhost:8900/api/passport
curl localhost:8900/api/docs        # interactive OpenAPI
```

The demo dataset is built to be a *test*, not a toy. The same four humans appear
across five providers with deliberately inconsistent names — Alice is `U01ALICE`
on Slack, `alice.chen@northwind.io` on Gmail and "Alice" in a WhatsApp export —
so the identity resolver has real ambiguity to resolve, and the remaining
ambiguity surfaces as suggestions rather than silent wrong merges.

---

## Architecture

```
                    ┌───────────────────────────────────────────┐
   connectors  ───► │  pipeline.py                             │
   (11 providers)    │  pull → raw → normalize → seal → store   │
                    └──────────────┬────────────────────────────┘
                                   │ documents (the source of truth)
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  identity/                   search/                      ai/
  blocking → scoring        BM25 + hashed                 TextRank,
  → clustering → persons    embeddings → RRF              predictions,
  → detectors               → facets, snippets            importance
        │                          │                          │
        └──────────────┬───────────┴──────────────┬───────────┘
                       ▼                          ▼
                 projection.py              (derived stores)
                 graph nodes + edges        all droppable
                       │                          │
                       └────────────┬─────────────┘
                                    ▼
                              api/  ──►  frontend/  (served from the same process)
```

Three decisions carry most of the weight:

**1. The graph is a pure function of the documents.** `projection.py` can
rebuild it from scratch at any time. That is what makes re-resolving identities
safe: you drop and re-project rather than surgery on live data, and a bug in
either is repairable by rebuilding. It is also why `POST /api/admin/drop-derived`
exists — the privacy claim is an operation you can run, not a promise.

**2. Identity tokens use a workspace-wide pepper, not the per-source DEK
pepper.** This is the one bug in this codebase that looked completely healthy
while doing nothing: `token_for_email(dek("slack").pepper, …) !=
token_for_email(dek("gmail").pepper, …)` by construction, so every token matched
only within its own provider. Blocking ran, pairs were scored, Persons were
emitted — and cross-source resolution returned exactly zero merges. The two
secrets have different jobs (encrypt content vs. compare identities) so they are
different keys; see `KeyManager.identity_pepper`.

**3. The search index stores no content.** `search_docs` holds only what
ranking and filtering need — kind, provider, timestamp, titles, person ids — and
never a body, never an email address. Bodies are decrypted for the ~20 results
on screen and nowhere else. The inverted index *is* plaintext-bearing (it has
to be, to rank), which is why it lives in memory by default and applies a
k-anonymity floor when you explicitly persist it.

### Data model

```
(Workspace)
(Person)  -[:HAS_IDENTITY]->      (Identity)
(Message) -[:SENT_BY]->           (Person)
(Person)  -[:PARTICIPATED_IN]->   (Conversation)
(Message) -[:IN_CONVERSATION]->   (Conversation)
(Message) -[:REPLY_TO|THREAD_ROOT]-> (Message)
(Message) -[:ATTACHED_TO]->       (File)
(Message) -[:MENTIONS]->          (Person)
(Message) -[:PREDICTED_DUE]->     (Event)
(File)    -[:OWNS]->              (Person)
(Note)    -[:AUTHORED]->          (Person)
(Note)    -[:PART_OF]->           (Note)
(Video)   -[:HAS_TRANSCRIPT]->    (Transcript)
(Person)  -[:LINKED_TO]->         (Person)   state: advisory
```

`state: advisory` on `LINKED_TO` is the privacy contract from blueprint §11.5. A
machine may *suggest* a relationship; only a human may *assert* one. The
asymmetry is enforced structurally: `write_insight_edges` is the only writer of
Person-Person edges and it hard-codes the state, and `accept_insight` — reachable
only from an explicit user action in the API — is the only thing that flips it.

---

## Repository layout

```
OmniLinker/
├── run.sh / run.ps1              one-command start (no Docker needed)
├── docker-compose.yml            the real Mongo + Neo4j profile
├── backend/
│   ├── omnilinker/
│   │   ├── connectors/           contract, registry, ratelimit, 11 providers
│   │   ├── crypto/               key hierarchy, AEAD envelope, identity tokens
│   │   ├── store/                DocStore/GraphStore: embedded | mongo | neo4j
│   │   ├── identity/             blocking, scorer, clustering, detectors
│   │   ├── search/               BM25 index, query grammar, hybrid, NL2Query
│   │   ├── ai/                   TextRank, predictions, importance, subjects
│   │   ├── api/                  FastAPI app + routes
│   │   ├── pipeline.py           raw → canonical → sealed → stored
│   │   ├── projection.py         graph as a pure function of documents
│   │   └── engine.py             the orchestration the API talks to
│   ├── tests/                    298 tests
│   └── requirements.txt
├── frontend/                     Vite + React + TS + D3
├── docs/adr/                     why the running system differs from the spec
└── deploy/nginx.conf             split frontend/backend deployment
```

---

## Testing

```bash
cd backend
python -m pip install --only-binary :all: -r requirements.txt
python -m pytest tests/ -q          # 298 passed
```

The suite is organised around the properties that are easy to lose and hard to
notice, and a good half of it is negative — anyone can build a resolver that
merges aggressively; the product risk is the opposite error.

- **`test_connectors.py`** — the contract tests blueprint §15.1 requires. For
  every one of the 11 providers: `normalize()` is pure (asserted dynamically by
  breaking `socket`), output is canonical, the idempotency key is the provider's
  own id, and one payload never yields two records with the same key. Plus the
  quirk that actually breaks each normalizer in production — nested MIME
  trees, snowflakes, zip-slip, `type`-discriminated blocks, SRT captions, and
  the WhatsApp `Name: text - with a dash` sender-split bug.
- **`test_crypto.py`** — AAD binding, fresh nonce per write, and the four
  directions a ciphertext cannot be moved in.
- **`test_pipeline.py`** — idempotency (a second sync creates nothing), cursors
  that only advance on full success, sealing, lineage that points at a real raw
  artifact.
- **`test_identity.py`** — that a decisive email match survives a handle
  mismatch, that a name alone never merges, that "Admin" never merges, that a
  phone alone never merges, and that resolution is deterministic.
- **`test_search.py`** — BM25 idf never goes negative, field weights beat length,
  k1 saturation stops keyword stuffing, a term that appears only in a *body* is
  findable, and the content-free projection cannot smuggle a body in.
- **`test_ai.py`** — every summary sentence exists verbatim in its source; every
  prediction names the message it came from; importance explains itself.
- **`test_stores_and_api.py`** — both store contracts, the API envelope, the
  drop-and-rebuild cycle, and the on-disk sealing audit that scans every file
  for plaintext that should be ciphertext.

Two of the tests in there exist because they caught a real bug during the build:
`test_merge_actually_happens` (a resolver that scored pairs but never merged, and
looked completely healthy) and `test_oversize_blocks_are_reported` / the
reduction-ratio assertion (a single `role:human` blocking key put every identity
into one block, reducing nothing while reporting a healthy reduction ratio).

---

## Performance, honestly

Measured on the 90-record demo workspace, single process, no GPU:

| Operation | Time |
|---|---|
| Full sync + all derived stages | ~120 ms |
| BM25 search, p95 | 6–8 ms |
| Hybrid search, p95 | 8–12 ms |
| Identity resolution (27 identities) | ~15 ms |
| Ego graph, 2 hops | <5 ms |

These are small numbers on a small corpus and should not be read as a claim
about 5M documents. The blueprint's p95 ≤ 2 s target is about the *architecture*
— bounded traversal, an inverted index rather than a scan, RRF instead of score
calibration — all of which is in place. What is not in place for that scale is
the ANN vector index, the streaming resolver, and the query planner. The
embedded store is explicitly a single-user development profile.

---

## Privacy posture

- **Content is sealed, and that is audited rather than asserted.** Message
  bodies, note text, file extracted text, transcript text *and* its per-caption
  segments, the derived thread subject, and the raw provider payloads are all
  AES-256-GCM envelopes bound to
  `workspace|doc_id|field_path|schema_version`. Moving a ciphertext between
  documents, fields or workspaces fails authentication instead of decrypting.

  A test scans every file on disk for a phrase that exists in a sealed field.
  It found three plaintext copies during the build — the raw payload, the
  derived subject, and the caption segments — each written by a component that
  had no reason to think about sealing. See
  [ADR-025](docs/adr/ADR-025-search-store-privacy.md).
- **Identity attributes and person names are plaintext**, because a resolution
  result nobody can read is not a result. Tokens make the comparison work
  without decryption; the addresses make it usable.
- **Subjects and titles are deliberately *not* sealed.** A title is the primary
  retrieval key; sealing it makes a document unfindable by the words the user
  already knows, and a title is already exposed in every provider UI. The cost —
  a readable subject in the derived store — is documented in
  `crypto/envelope.py` and in the data passport.
- **Identity tokens are deterministic and keyed.** Equality across sources
  without ever storing or decrypting an address. Gmail dot/plus folding is
  applied, because otherwise one person with two Gmail aliases is two people
  forever.
- **No write-back.** There is no code path in this system that issues a write to
  a provider. `test_search.py::test_nothing_is_written_back_to_a_provider`
  greps the route table to keep it that way.
- **Derived stores are droppable.** `POST /api/admin/drop-derived` destroys the
  search projection, persons, insights and predictions, then rebuilds them from
  the source documents. The data passport reports this as an operation.
- **Advisory ≠ accepted.** Every detector output and every identity suggestion
  starts as a lead. Only a user action promotes it.

## Configuration

Every setting has a working default; see `.env.example` for the full list. The
ones that change behaviour deliberately:

| Variable | Default | Effect |
|---|---|---|
| `OMNI_PORT` | `8900` | 8000 is conventionally an OpenAI-compatible server; colliding there shows up as a 502 from a proxy, not a clear error. |
| `OMNI_PUBLIC_URL` | `http://127.0.0.1:8900` | OAuth redirect target. Must match the URI registered in each provider's console. |
| `OMNI_OAUTH_<PROVIDER>_CLIENT_ID` | — | Per-provider OAuth client. The Connectors page names the exact variable for each. |
| `OMNI_MODE` | `embedded` | `docker` switches to Mongo + Neo4j. |
| `OMNI_ENCRYPT` | `1` | `0` stores bodies in the clear. |
| `OMNI_VECTOR` | `1` | Second retrieval leg on/off. |
| `OMNI_PERSIST_INDEX` | `0` | Write the inverted index to disk. |
| `OMNI_MERGE_T` | `0.90` | Auto-merge threshold. |
| `AI_MODE` | `extractive` | `lead` selects the baseline. |
