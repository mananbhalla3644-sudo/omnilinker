# Architecture Decision Records

Every place the running system differs from `OmniLinker_Blueprint.txt`, and why.
A system that quietly differs from its spec is worse than one that is loudly
different, so these are also served live at `GET /api/system/info` and rendered
on the dashboard.

| ADR | Area | Deviation | Reversible? |
|---|---|---|---|
| [ADR-020](ADR-020-embedded-store-default.md) | stores | Embedded backend is the default | yes, config |
| [ADR-021](ADR-021-aes-gcm.md) | crypto | AES-256-GCM, not XChaCha20-Poly1305 | yes, one function |
| [ADR-022](ADR-022-spa-frontend.md) | frontend | Vite SPA, not Next.js | yes, config |
| [ADR-023](ADR-023-hashed-embeddings.md) | search | Hashed embeddings, not a neural index | yes, config |
| [ADR-024](ADR-024-deterministic-ai.md) | ai | Extractive/rule-based, not LLM | yes, config |
| [ADR-025](ADR-025-search-store-privacy.md) | search/privacy | Content-free `search_docs`; in-memory index | partly |
| [ADR-026](ADR-026-one-credential-resolver.md) | connectors | One credential resolver per OAuth handshake | yes, one function |

## The one that is not a deviation

None of these change a *product* commitment. The blueprint's non-goals hold: no
write-back to any provider (ADR and a test enforce it), field-level encryption
rather than full-record (§10.2, as specified), bounded graph traversal (§5.2.4),
and a data passport (§10.4).

What the ADRs change is *how* each commitment is met, and each one states the
residual risk it introduces. Where a choice has a cost that is only acceptable
because of a specific property — the 96-bit nonce is safe *because* there is one
key per source and a random nonce per write — that dependency is written down,
because it is the thing that would break first if someone changed the key
hierarchy.
