"""ULID implementation (blueprint ADR-013).

Why ULID over UUIDv4: lexicographically sortable by creation time, so range
scans on `ts`-adjacent ids stay index-friendly, URL-safe, and they can be
minted client-side (the browser can mint an id for an optimistic row without
a server round-trip).

Layout (48 bits ms timestamp | 80 bits randomness), Crockford base32, 26 chars.
"""

from __future__ import annotations

import os
import threading
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32, no I L O U
_DECODE = {c: i for i, c in enumerate(_ALPHABET)}
_lock = threading.Lock()
_last_ms = 0
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def _decode(value: str) -> int:
    total = 0
    for ch in value.upper():
        if ch not in _DECODE:
            raise ValueError(f"invalid ULID character: {ch!r}")
        total = (total << 5) | _DECODE[ch]
    return total


def new_ulid(ts_ms: int | None = None) -> str:
    """Mint a new ULID. Monotonic within a process (same-ms values increment)."""
    global _last_ms, _last_rand
    with _lock:
        now = int(time.time() * 1000) if ts_ms is None else ts_ms
        if now == _last_ms:
            _last_rand = (_last_rand + 1) & ((1 << 80) - 1)
        else:
            _last_ms = now
            _last_rand = int.from_bytes(os.urandom(10), "big")
        ms, rand = _last_ms, _last_rand
    return _encode(ms, 10) + _encode(rand, 16)


def ulid_timestamp(uid: str) -> int:
    """Extract the embedded millisecond timestamp."""
    if len(uid) != 26:
        raise ValueError("ULID must be 26 characters")
    return _decode(uid[:10])


def prefixed(prefix: str) -> str:
    """Domain-prefixed id, e.g. msg_01J... - keeps ids self-describing in logs."""
    return f"{prefix}_{new_ulid()}"


def is_ulid(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 26:
        return False
    try:
        _decode(value)
    except ValueError:
        return False
    return True
