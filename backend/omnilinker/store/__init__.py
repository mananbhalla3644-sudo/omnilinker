"""Store factory + the store handle used across the app.

Blueprint rule: services never branch on the deployment profile. They call
`get_stores()` and get whatever the profile provides.
"""

from __future__ import annotations

import atexit
import threading

from omnilinker.config import get_settings
from omnilinker.store.base import Stores
from omnilinker.store.embedded_docs import EmbeddedDocStore
from omnilinker.store.embedded_graph import EmbeddedGraphStore

_lock = threading.Lock()
_stores: Stores | None = None


def build_stores() -> Stores:
    s = get_settings()
    if s.uses_embedded_store:
        root = s.data_dir / "store"
        return Stores(
            docs=EmbeddedDocStore(root),
            graph=EmbeddedGraphStore(root),
            backend="embedded",
            data_root=root,
        )

    # docker / cloud profile. Imported lazily so the embedded path never needs
    # motor or the neo4j driver installed.
    try:
        from omnilinker.store.mongo_docs import MongoDocStore
        from omnilinker.store.neo4j_graph import Neo4jGraphStore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"OMNI_MODE={s.mode!r} needs the optional drivers. Run:\n"
            "    python -m pip install -r requirements-docker.txt"
        ) from exc

    docs = MongoDocStore(s.mongo_uri, s.mongo_db)
    graph = Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    graph.ensure_schema()
    return Stores(docs=docs, graph=graph, backend=s.mode)


def get_stores() -> Stores:
    global _stores
    if _stores is not None:
        return _stores
    with _lock:
        if _stores is None:
            _stores = build_stores()
            atexit.register(flush_stores)
    return _stores


def flush_stores() -> int:
    """Persist dirty state. Returns the number of files written.

    Called after every sync and admin mutation, not only at exit: `atexit`
    fires on a clean shutdown, so relying on it alone means a crash, a
    `docker stop` with a short grace period, or a closed laptop lid loses the
    user's entire workspace. The write is atomic per collection and costs a few
    milliseconds, which is not a trade worth making the other way.
    """
    if _stores is None:
        return 0
    written = 0
    for store in (_stores.docs, _stores.graph):
        fn = getattr(store, "flush", None)
        if callable(fn):
            try:
                written += int(fn() or 0)
            except Exception:  # pragma: no cover - shutdown must not raise
                pass
    return written


def reset_stores() -> None:
    """Test hook."""
    global _stores
    with _lock:
        _stores = None
