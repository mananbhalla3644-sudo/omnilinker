# ADR-021: AES-256-GCM instead of XChaCha20-Poly1305

**Status:** accepted · **Area:** crypto · **Reversible:** yes, one function

## The blueprint specifies

XChaCha20-Poly1305, with the extended nonce specifically to remove the
"random 96-bit nonces must never repeat" constraint.

## We use

AES-256-GCM, with a random 96-bit nonce per write and a per-source key.

## Why

**Availability.** `cryptography` ships AES-GCM in every build — CPython,
PyPy, Pyodide, WASM. XChaCha20-Poly1305 requires either the `chacha20poly1305`
extra or a Rust extension. That is a real constraint for a product whose pitch
includes running locally, and it converts "install a native extension" into a
support question.

**Performance.** AES-NI makes AES-GCM roughly an order of magnitude faster than
a portable ChaCha20 implementation on the target platforms. Sealing happens on
every field of every record during ingest, so this is on the hot path.

**The extended nonce is buying insurance we do not need.** The reason to want a
192-bit nonce is to tolerate nonce reuse from a *random* source across a very
large number of messages. Our scheme is:

```
UMK (file, 0600)
  └─ workspace KEK        = HKDF(UMK, "wkek|ws")
       └─ per-source DEK  = HKDF(WKEK, "dek|<source>")
            └─ nonce      = os.urandom(12), fresh per write
```

One key per source, and a write consumes a nonce exactly once. GCM's failure
mode on nonce reuse under one key is catastrophic (it leaks the XOR of
plaintexts and forfeits authentication), so this is not a property to be casual
about — but the *structural* guarantee is that a DEK is used for a bounded number
of writes and then rotated by re-deriving with a new version string. The random
nonce is a second line of defence, not the only one.

## Residual risk

**The safety argument depends on the key hierarchy staying one-key-per-source.**
If a future change derives a single global content DEK, a 96-bit random nonce
gives birthday-bound collision at roughly 2³² writes under that key. At a
realistic ingest rate that is far away, but the margin is gone and the ADR must
be revisited at that point rather than discovered.

**Switching costs one function.** `crypto/envelope.py` contains all the AEAD
usage. XChaCha20-Poly1305 needs a 24-byte nonce and a 16-byte tag; both formats
and the `DecryptionError` contract are already isolated there.

## What we did not compromise

The AAD binding is unaffected and is the part that actually matters:

```
AAD = "<workspace>|<doc_id>|<field_path>|<schema_version>"
```

AES-GCM authenticates the AAD, so an attacker with database *write* access —
the realistic threat when they do not have the key — cannot move a ciphertext
from one document, field or workspace to another. The tag check fails. This is
`test_crypto.py::TestEnvelope::test_cannot_move_ciphertext_to_another_document`
and its three siblings.
