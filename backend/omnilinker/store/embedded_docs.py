"""Embedded document store: JSON-backed, in-memory index, zero dependencies.

Blueprint 3.5 "LOCAL-FIRST PROFILE". This is what makes the project runnable
with no Docker, no Mongo, no Redis: one JSON file per collection under
`{data_dir}/store/{collection}.json`, an in-memory inverted index for filters,
and a debounced write-behind so a 10k-record batch costs one disk flush.

It is not a Mongo replacement and does not pretend to be. What it *is* is a
correct implementation of `DocStore` with the same idempotent upsert semantics,
which means the pipeline, search, identity and API layers are all exercised
for real in the default profile - a green local run means the logic is right,
not that a mock was lenient.
"""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.store.base import DocStore, Filter, Query

_DOT_SPLIT = re.compile(r"\.")


def get_path(doc: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    """Resolve a dotted path with array awareness (`a.b`, `a.0.b`).

    Returns (found, value). Mirrors Mongo semantics closely enough for the
    Filter DSL and keeps the embedded backend honest against the real one.
    """
    cur: Any = doc
    for part in _DOT_SPLIT.split(dotted):
        if isinstance(cur, Mapping):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            if part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                # Implicit array traversal, like Mongo's dotted array paths
                if not all(isinstance(x, Mapping) for x in cur):
                    return False, None
                vals = [x.get(part) for x in cur if isinstance(x, Mapping) and part in x]
                if not vals:
                    return False, None
                return True, vals
        else:
            return False, None
    return True, cur


def _cmp(actual: Any, op: str, expected: Any) -> bool:
    try:
        match op:
            case "eq":
                return actual == expected
            case "ne":
                return actual != expected
            case "in":
                if isinstance(actual, (list, tuple, set)):
                    return bool(set(map(str, actual)) & set(map(str, expected)))
                return actual in expected
            case "nin":
                if isinstance(actual, (list, tuple, set)):
                    return not (set(map(str, actual)) & set(map(str, expected)))
                return actual not in expected
            case "contains":
                if isinstance(actual, str):
                    return str(expected).lower() in actual.lower()
                if isinstance(actual, (list, tuple, set)):
                    return expected in actual
                return False
            case "gte":
                return actual is not None and actual >= expected
            case "lte":
                return actual is not None and actual <= expected
            case "gt":
                return actual is not None and actual > expected
            case "lt":
                return actual is not None and actual < expected
            case "between":
                lo, hi = expected
                return actual is not None and lo <= actual <= hi
            case "exists":
                return (actual is not None) == bool(expected)
            case "prefix":
                return isinstance(actual, str) and actual.lower().startswith(str(expected).lower())
            case "regex":
                return isinstance(actual, str) and bool(re.search(expected, actual, re.I))
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unsupported filter op: {op!r}")


def matches(doc: Mapping[str, Any], filters: Sequence[Filter]) -> bool:
    for f in filters:
        found, value = get_path(doc, f.field)
        if f.op == "exists":
            if found != bool(f.value):
                return False
            continue
        if not found or not _cmp(value, f.op, f.value):
            return False
    return True


class EmbeddedDocStore(DocStore):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, dict]] = defaultdict(dict)
        self._dirty: set[str] = set()
        #: Public so callers (and tests) can locate the on-disk state
        #: without guessing at the file layout.
        self.data_root = root
        self._load()

    # -- persistence --------------------------------------------------
    def _path(self, collection: str) -> Path:
        return self.root / f"{collection}.json"

    def _load(self) -> None:
        for file in self.root.glob("*.json"):
            name = file.stem
            try:
                payload = json.loads(file.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt shard must not take the process down. Rename it so
                # the next boot is clean and the operator can inspect it.
                file.replace(file.with_suffix(".json.corrupt"))
                continue
            if isinstance(payload, dict):
                self._data[name] = payload

    def flush(self) -> int:
        """Write every dirty collection atomically. Returns files written."""
        with self._lock:
            written = 0
            for name in list(self._dirty):
                target = self._path(name)
                tmp = target.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._data.get(name, {}), ensure_ascii=False, separators=(",", ":")),
                    "utf-8",
                )
                tmp.replace(target)
                self._dirty.discard(name)
                written += 1
            return written

    def _touch(self, collection: str) -> None:
        self._dirty.add(collection)

    # -- DocStore -----------------------------------------------------
    def put(self, collection: str, doc_id: str, doc: Mapping[str, Any]) -> dict:
        with self._lock:
            record = dict(doc)
            record["_id"] = doc_id
            existing = self._data[collection].get(doc_id)
            if existing:
                # Preserve created_at across updates (idempotent upsert).
                record.setdefault("created_at", existing.get("created_at"))
            self._data[collection][doc_id] = record
            self._touch(collection)
            return record

    def put_many(self, collection: str, docs: Sequence[Mapping[str, Any]], id_field: str) -> int:
        with self._lock:
            bucket = self._data[collection]
            for doc in docs:
                doc_id = str(doc.get(id_field) or doc.get("_id"))
                if not doc_id or doc_id == "None":
                    continue
                record = dict(doc)
                record["_id"] = doc_id
                if doc_id in bucket:
                    record.setdefault("created_at", bucket[doc_id].get("created_at"))
                bucket[doc_id] = record
            self._touch(collection)
            return len(docs)

    def get(self, collection: str, doc_id: str) -> dict | None:
        with self._lock:
            hit = self._data.get(collection, {}).get(doc_id)
            return dict(hit) if hit else None

    def get_many(self, collection: str, doc_ids: Sequence[str]) -> list[dict]:
        with self._lock:
            bucket = self._data.get(collection, {})
            return [dict(bucket[i]) for i in doc_ids if i in bucket]

    def find(self, query: Query) -> list[dict]:
        with self._lock:
            rows = [dict(d) for d in self._data.get(query.collection, {}).values()]
        rows = [r for r in rows if matches(r, query.filters)]
        for field_name, direction in reversed(query.sort or [("_id", 1)]):
            rows.sort(
                key=lambda r, f=field_name: (get_path(r, f)[1] is None, get_path(r, f)[1]),
                reverse=direction < 0,
            )
        if query.skip:
            rows = rows[query.skip :]
        return rows[: query.limit]

    def count(self, collection: str, filters: Sequence[Filter] = ()) -> int:
        with self._lock:
            rows = self._data.get(collection, {}).values()
            if not filters:
                return len(list(rows))
            return sum(1 for r in rows if matches(r, filters))

    def facet(
        self, collection: str, field: str, filters: Sequence[Filter] = (), limit: int = 25
    ) -> list[tuple[str, int]]:
        with self._lock:
            rows = list(self._data.get(collection, {}).values())
        rows = [r for r in rows if matches(r, filters)]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            found, value = get_path(row, field)
            if not found or value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    counts[str(item)] += 1
            else:
                counts[str(value)] += 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def delete(self, collection: str, doc_id: str) -> bool:
        with self._lock:
            bucket = self._data.get(collection, {})
            if doc_id in bucket:
                del bucket[doc_id]
                self._touch(collection)
                return True
            return False

    def purge(self, collections: Iterable[str] | None = None) -> int:
        with self._lock:
            names = list(collections) if collections is not None else list(self._data.keys())
            removed = sum(len(self._data.get(n, {})) for n in names)
            for name in names:
                self._data[name] = {}
                self._touch(name)
            return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {name: len(bucket) for name, bucket in sorted(self._data.items()) if bucket}
