# ADR-025: what the search store is allowed to contain

**Status:** accepted · **Area:** search / privacy · **Reversible:** partly

## The blueprint specifies

A k-anonymized token projection as the persisted search store.

## What we do

Two derived stores with deliberately different postures.

**`search_docs` — persisted, content-free.** Filters, facets, sort keys,
display titles, person ids, and *no* body, no email address, no phone number, no
note text. `project_search_doc()` is the single place that decides this, and
`test_search.py::test_search_doc_projection_excludes_content` pins it — including
the failure mode where a still-sealed envelope gets `str()`'d into the index and
makes every document match every query.

**The inverted index — in memory by default.** An index has to contain the
corpus vocabulary in order to rank; that is not an implementation detail, it is
what a text index *is*. So writing it to disk would create a second plaintext
store, and the honest response is to not write it.

## Where the blueprint's k-anonymity went

Into `LexicalIndex.save()`, which is opt-in and lossy on purpose:

```python
index.save(path, k_anonymity=2)
# {'terms_kept': 812, 'terms_dropped_for_k_anonymity': 47, 'k_anonymity': 2}
```

A term present in fewer than *k* documents is, in effect, a verbatim quote from
one document. At k=2 a single-document term still leaks that the document
contains a rare word, so the honest options are *k*=2 (low latency) or **do not
persist at all**, which is the default and the recommendation. The caller is
told how many terms were dropped, so the trade is visible rather than assumed.

## The one place content is decrypted at query time

Snippets. When a search returns 20 results, those 20 documents are decrypted,
a snippet is extracted around the match terms, and the plaintext is discarded.
The bound is the page being displayed, which is why there is no endpoint
anywhere in this system that returns the corpus in the clear — the first time
someone adds one, the system has a plaintext export endpoint wearing a search
UI.

## The sealing audit, and what it found

Writing the test that scans **every file on disk** for a phrase that exists in a
sealed field turned up three plaintext copies of content we had already claimed
to protect. Each looked correct in isolation and each made the headline claim
untrue:

| Leak | Why it happened | Fix |
|---|---|---|
| `raw_artifacts.payload` | the immutable provider payload is the same bytes as the message body, and a *larger* copy — headers, reactions, edit history | sealed; reproducibility preserved by `payload_hash`, which is now taken over the exact stored bytes |
| `thread_subject` | a derived sentence lifted from a sealed body, written by the enrichment pass *after* the pipeline had sealed the record | added to `SEALED_FIELDS`; the index receives the decrypted title in memory at build time |
| `transcripts[].segments` | per-caption text stored beside a sealed `text` field — the same transcript, twice | sealed together with `text` |
| `importance_features[].detail` | quoted the matched decision cue out of the body | reports the feature and its weight instead; the fragment was never needed to explain the score |

The lesson generalises: **a sealing boundary is only as good as the audit of
everything that touches the data after it.** Each of these was written by a
component that had no reason to think about sealing — a projector, an enrichment
pass, a normalizer emitting segments. The check now lives in
`test_stores_and_api.py::test_no_plaintext_from_a_sealed_field_reaches_disk`.

## The cost we are choosing to pay

**Identity attributes are readable.** Names, email addresses and phone numbers
in `identities` are plaintext, and Person display names are plaintext in the
graph. This is structural rather than incidental: cross-source resolution
compares deterministic tokens, but a person has to be *displayable*, and the
whole product is a list of people. The tokens are what make comparison work
without decryption; the addresses are what make the result usable. A deployment
that cannot accept this would have to encrypt the identity store, at the cost of
being unable to show anyone who anyone is.

**Titles and subjects are readable in the derived store.** So they are in
`search_docs`, so a read-only compromise of the derived store reveals the
subject line of every indexed email.

That is stated rather than minimised. The reasoning: a title is the primary
retrieval key, so sealing it makes a document unfindable by the words the user
already knows; and a subject line is already exposed in every provider UI,
every notification preview and every email header. Sealing it protects nothing
that is not already exposed, while breaking the single most important search
affordance. The bodies — where the actual private content is — are sealed.

This is the same reasoning behind removing `subject` and `title` from
`SEALED_FIELDS`, and it is the one place in this codebase where the encryption
boundary is a judgement call rather than a maximum.
