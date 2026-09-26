"""Blocking (blueprint 7.2 stage 1).

Resolution over 500k identities is quadratic if you compare everything to
everything. Blocking reduces it to comparing only pairs that share at least
one cheap, high-precision key - so the expensive scorer runs on a few thousand
candidate pairs instead of 125 billion.

Four blocking strategies, in the order they are applied:

  B1 exact token   email / phone / name token equality
  B2 name prefix   first 4 chars of the sorted name token + length bucket.
                   Catches "clara" vs "clara n" without a similarity index.
  B3 handle        the provider handle, lowercased
  B4 co-occurrence identities that appear in the same conversation within a
                   time window. This is the fallback that finds people whose
                   contact details changed completely - they still show up in
                   the same Slack thread.

Blocking only ever *proposes* pairs. It never decides. Recall is what matters
here; precision is enforced in the scorer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping, Sequence

Pair = tuple[str, str]

#: Records fewer than this are ignored as blocks: a block of size 1 proposes
#: nothing, and a block containing every identity in the workspace would
#: silently turn blocking into an all-pairs comparison. Blocks above
#: `MAX_BLOCK_SIZE` are skipped and reported, because they usually mean a
#: parser bug (e.g. every message attributed to a shared bot account).
MIN_BLOCK_SIZE = 2
MAX_BLOCK_SIZE = 400


def block_keys(identity: Mapping[str, Any]) -> list[str]:
    """Blocking keys for one identity. Empty list = unblockable, which is
    legitimate (a handle-less, name-less identity has no cheap join key)."""
    keys: list[str] = []
    provider = identity.get("provider", "")

    if identity.get("email_token"):
        keys.append(f"em:{identity['email_token']}")
    if identity.get("phone_token"):
        keys.append(f"ph:{identity['phone_token']}")
    if identity.get("name_token"):
        name_token = identity["name_token"]
        keys.append(f"nm:{name_token}")
        # B2: a prefix+length bucket catches truncated display names
        # ("clara" from Slack vs "Clara Nowak" from Gmail) with no
        # similarity index. The length bucket keeps the block small.
        if len(name_token) >= 8:
            keys.append(f"np:{name_token[:8]}:{len(name_token) // 8}")
    handle = (identity.get("provider_user_id") or "").strip().lower()
    if handle and provider not in {"contact", "whatsapp"}:
        keys.append(f"hd:{provider}:{handle}")
    # NOTE: there is deliberately no "role:human" / "role:contact" key here.
    #
    # It looks like a harmless partition and it is the exact opposite: every
    # non-contact identity in the workspace shares that one key, so the block
    # holds all of them and proposes every pair - which is all-pairs
    # comparison with extra bookkeeping. On 200 identities the candidate count
    # was exactly the all-pairs count, i.e. blocking reduced nothing while
    # reporting a healthy reduction ratio in every other respect.
    #
    # Contact identities do need to be kept apart from real people, and that is
    # enforced where it belongs: `PairScorer.score` returns `distinct` for any
    # pair with a contact on either side, before any feature is considered.
    return keys


def build_blocks(identities: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[dict]]:
    """Returns (blocks, oversize_blocks)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for identity in identities:
        iid = str(identity.get("_id") or identity.get("ref"))
        for key in block_keys(identity):
            buckets[key].append(iid)

    oversize: list[dict] = []
    blocks: dict[str, list[str]] = {}
    for key, members in buckets.items():
        unique = sorted(set(members))
        if len(unique) < MIN_BLOCK_SIZE:
            continue
        if len(unique) > MAX_BLOCK_SIZE:
            oversize.append({"key": key, "size": len(unique)})
            continue
        blocks[key] = unique
    return blocks, oversize


def pairs_from_blocks(blocks: Mapping[str, Sequence[str]]) -> list[Pair]:
    """All unordered pairs inside each block, deduplicated across blocks.

    Returns pairs sorted for determinism - the resolver's output must be
    reproducible, otherwise a re-run produces a different clustering and every
    downstream test becomes flaky.
    """
    seen: set[Pair] = set()
    for members in blocks.values():
        ordered = sorted(members)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                seen.add((a, b))
    return sorted(seen)


def cooccurrence_pairs(
    documents: Sequence[Mapping[str, Any]],
    *,
    window_days: int = 7,
) -> list[Pair]:
    """B4: identities that appear in the same conversation close together.

    `documents` are message/email/note docs carrying a `sender_ref` and a
    timestamp. This is deliberately a small sliding window over per-conversation
    sorted ref lists: it is O(n log n) and it is the only signal that survives
    a person changing their email address.
    """
    from datetime import datetime, timedelta, timezone

    by_conversation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc in documents:
        ref = doc.get("sender_ref")
        ts = doc.get("ts")
        if not ref or not ts:
            continue
        try:
            when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        by_conversation[str(doc.get("conversation_id") or ref)].append((ref, when.isoformat()))

    pairs: set[Pair] = set()
    horizon = timedelta(days=window_days)
    for entries in by_conversation.values():
        refs: list[tuple[datetime, str]] = []
        for ref, when in entries:
            try:
                refs.append((datetime.fromisoformat(when), ref))
            except ValueError:
                continue
        refs.sort()
        for i, (when_a, ref_a) in enumerate(refs):
            for when_b, ref_b in refs[i + 1:]:
                if when_b - when_a > horizon:
                    break
                if ref_a != ref_b:
                    pairs.add(tuple(sorted((ref_a, ref_b))))  # type: ignore[arg-type]
    return sorted(pairs)


def blocking_stats(identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks, oversize = build_blocks(identities)
    pairs = pairs_from_blocks(blocks)
    n = len(identities)
    all_pairs = n * (n - 1) // 2
    return {
        "identities": n,
        "blocks": len(blocks),
        "candidate_pairs": len(pairs),
        "all_pairs": all_pairs,
        # The whole point of blocking. On a 500k workspace this is 7 orders of
        # magnitude, which is the difference between "runs" and "never".
        "reduction_ratio": round(1 - (len(pairs) / all_pairs), 4) if all_pairs else 0.0,
        "oversize_blocks": oversize,
    }
