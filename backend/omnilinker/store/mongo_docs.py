"""MongoDB document store (the `docker` / `cloud` profile).

Implements the same `DocStore` contract as the embedded backend. Two rules make
that possible:

1. **The store never sees a query string.** Every call arrives as a typed
   `Filter` / `Query`, and translation to a Mongo document happens here, in
   exactly one function. There is no string to inject and no caller that could
   build one.

2. **Dot paths are resolved the same way in both backends.** `embedded_docs`
   defines the semantics; `_mongo_field` mirrors them. A filter that works in
   one profile must work in the other, or "just use the docker profile" becomes
   a debugging exercise.

`a` vs `a.b` is the one genuinely ambiguous case - a scalar and an object
cannot both live under the same key in Mongo, and `a` matches the object
loosely. Both backends resolve the *more specific* path first, so
`entities.person_ids` is checked before `entities`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.store.base import DocStore, Filter, Query

_DOT_SPLIT = re.compile(r"\.")


def _mongo_field(dotted: str) -> str:
    """`a.b.0.c` -> `a.b.0.c`; arrays are addressed by index, matching the
    embedded resolver's `a.0.b`."""
    return dotted.replace(".", "\x00.").replace("\x00.0.", ".0.").replace("\x00.", ".")


def _mongo_filter(f: Filter) -> dict[str, Any]:
    path = _mongo_field(f.field)
    op, value = f.op, f.value
    if op == "eq":
        return {path: {"$eq": value}}
    if op == "ne":
        return {path: {"$ne": value}}
    if op == "gt":
        return {path: {"$gt": value}}
    if op == "gte":
        return {path: {"$gte": value}}
    if op == "lt":
        return {path: {"$lt": value}}
    if op == "lte":
        return {path: {"$lte": value}}
    if op == "in":
        return {path: {"$in": list(value or [])}}
    if op == "nin":
        return {path: {"$nin": list(value or [])}}
    if op == "contains":
        # Case-insensitive substring. A regex here is anchored to the caller
        # only via re.escape, so a user-supplied value cannot inject a pattern.
        return {path: {"$regex": re.escape(str(value)), "$options": "i"}}
    if op == "startswith":
        return {path: {"$regex": "^" + re.escape(str(value)), "$options": "i"}}
    if op == "endswith":
        return {path: {"$regex": re.escape(str(value)) + "$", "$options": "i"}}
    if op == "exists":
        if value:
            return {path: {"$exists": True}}
        return {path: {"$exists": False}}
    raise ValueError(f"unsupported operator {op!r}")


def _combine(filters: Sequence[Filter]) -> dict[str, Any]:
    """All filters are ANDed. The embedded store does the same, so a query
    cannot mean one thing in one profile and something else in the other."""
    parts = [_mongo_filter(f) for f in filters]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


class MongoDocStore(DocStore):
    """Async motor client, driven synchronously.

    The rest of the codebase is synchronous by design - the pipeline, the
    resolver and the indexer are all plain functions, which is what makes them
    testable without an event loop. This adapter runs each motor coroutine on a
    dedicated loop rather than dragging asyncio through every layer above it.
    The alternative (an async pipeline) would buy nothing here: ingestion is
    sequential and bounded by provider rate limits anyway.
    """

    def __init__(self, uri: str, database: str = "omnilinker", loop=None) -> None:
        import asyncio

        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "OMNI_MODE=docker needs motor. Install requirements-docker.txt."
            ) from exc

        self._loop = loop or asyncio.new_event_loop()
        self._client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[database]
        self._collections: dict[str, Any] = {}
        self._indexes_ready = False

    def _run(self, coro: Any) -> Any:
        import asyncio

        if self._loop.is_running():  # pragma: no cover - defensive
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return self._loop.run_until_complete(coro)

    def _collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._db[name]
        return self._collections[name]

    def _ensure_indexes(self) -> None:
        """Created once, lazily. The index set is the query set: every filter
        the API can issue has a matching index, so a facet or a timeline over
        5M documents is a range scan rather than a collection scan."""
        if self._indexes_ready:
            return

        async def build() -> None:
            specs = {
                "messages": [("provider", 1), ("ts", -1), ("conversation_id", 1),
                             ("person_ids", 1), ("kind", 1)],
                "files": [("provider", 1), ("modified_ts", -1), ("name", 1),
                          ("person_ids", 1)],
                "notes": [("provider", 1), ("last_edited", -1)],
                "videos": [("provider", 1), ("published_ts", -1)],
                "transcripts": [("provider", 1), ("video_id", 1)],
                "conversations": [("provider", 1), ("stream", 1)],
                "identities": [("provider", 1), ("email_token", 1), ("phone_token", 1),
                               ("name_token", 1), ("person_id", 1)],
                "persons": [("message_count", -1), ("cross_source", 1)],
                "search_docs": [("provider", 1), ("kind", 1), ("ts", -1),
                                ("person_ids", 1)],
                "cursors": [("connector_id", 1), ("stream", 1)],
                "ingest_runs": [("started_at", -1)],
                "audit": [("ts", -1)],
                "insights": [("state", 1), ("detector", 1), ("confidence", -1)],
                "predictions": [("kind", 1), ("score", -1)],
                "link_suggestions": [("status", 1), ("score", -1)],
                "raw_artifacts": [("provider", 1), ("stream", 1), ("payload_hash", 1)],
            }
            for name, keys in specs.items():
                for keys_ in keys:
                    try:
                        await self._collection(name).create_index(keys_)
                    except Exception:
                        pass  # an index that already exists is not an error

        self._run(build())
        self._indexes_ready = True

    # -- DocStore ------------------------------------------------------
    def put(self, collection: str, doc_id: str, doc: Mapping[str, Any]) -> dict:
        self._ensure_indexes()
        payload = {**doc, "_id": doc_id}

        async def write() -> dict:
            result = await self._collection(collection).replace_one(
                {"_id": doc_id}, payload, upsert=True
            )
            return {"created": result.upserted_id is not None, "id": doc_id}

        return self._run(write())

    def put_many(self, collection: str, docs: Sequence[Mapping[str, Any]],
                 id_field: str) -> int:
        if not docs:
            return 0
        self._ensure_indexes()
        rows = [{**d, "_id": d[id_field]} for d in docs]

        async def write() -> int:
            from pymongo import ReplaceOne

            operations = [ReplaceOne({"_id": row["_id"]}, row, upsert=True) for row in rows]
            result = await self._collection(collection).bulk_write(operations, ordered=False)
            return result.upserted_count + result.modified_count

        return self._run(write())

    def get(self, collection: str, doc_id: str) -> dict | None:
        return self._run(self._collection(collection).find_one({"_id": doc_id}))

    def get_many(self, collection: str, doc_ids: Sequence[str]) -> list[dict]:
        ids = list(doc_ids)
        if not ids:
            return []
        return list(self._run(self._collection(collection).find({"_id": {"$in": ids}})))

    def find(self, query: Query) -> list[dict]:
        self._ensure_indexes()
        spec = _combine(query.filters)
        if query.projection:
            spec = {"$and": [spec, {"projection": {"$slice": len(query.projection)}}]} \
                if False else spec
        cursor = self._collection(query.collection).find(
            spec or {},
            projection={field: 1 for field in query.projection} if query.projection else None,
        )
        if query.sort:
            cursor = cursor.sort(query.sort)
        cursor = cursor.skip(query.skip).limit(query.limit)
        return list(self._run(cursor.to_list(length=query.limit)))

    def count(self, collection: str, filters: Sequence[Filter] = ()) -> int:
        spec = _combine(filters)
        return int(self._run(self._collection(collection).count_documents(spec or {})))

    def facet(self, collection: str, field: str,
              filters: Sequence[Filter] = (), limit: int = 25) -> list[tuple[str, int]]:
        spec = _combine([*filters, Filter(field, "exists", True)])
        path = _mongo_field(field)
        pipeline = [
            {"$match": spec},
            {"$group": {"_id": f"${path}", "n": {"$sum": 1}}},
            {"$sort": {"n": -1, "_id": 1}},
            {"$limit": limit},
        ]
        rows = self._run(self._collection(collection).aggregate(pipeline).to_list(length=limit))
        return [(str(row["_id"]), int(row["n"])) for row in rows]

    def delete(self, collection: str, doc_id: str) -> bool:
        result = self._run(self._collection(collection).delete_one({"_id": doc_id}))
        return bool(result.deleted_count)

    def purge(self, collections: Iterable[str] | None = None) -> int:
        names = list(collections) if collections is not None else list(self._collections)
        total = 0
        for name in names:
            result = self._run(self._collection(name).delete_many({}))
            total += result.deleted_count
        return total

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in self._collections:
            out[name] = int(self._run(self._collection(name).estimated_document_count()))
        return out

    def close(self) -> None:
        self._run(self._client.close())
        self._loop.close()
