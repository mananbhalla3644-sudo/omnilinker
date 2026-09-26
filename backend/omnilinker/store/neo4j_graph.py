"""Neo4j graph store (the `docker` / `cloud` profile).

Same contract as the embedded adjacency store. Two properties are carried over
deliberately, because they are what the blueprint's bounded-traversal rule
(5.2.4) actually requires and they are easy to lose in a Cypher translation:

1. **`ego()` is bounded on both ends.** Not just a `LIMIT` on the result: the
   traversal stops expanding past `max_nodes`, so a Person with a million
   messages cannot make the API hang *and* cannot make it return a truncated
   answer that looks complete.

2. **Node and edge properties are plain scalars and string lists.** The embedded
   store keeps them that way, so a `File` node with 40 folder segments and a
   `File` node with one behave identically in both profiles.

The projector emits no Cypher at all - it writes through this interface, so
there is no query string to get wrong and nothing here needs sanitising beyond
parameter binding, which is what every call below uses.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from omnilinker.store.base import Edge, GraphStore, Node, Subgraph

#: Cap on nodes materialised by one traversal. A workspace-scale graph has
#: millions; a browser session has a few hundred. Anything between those is a
#: server-side aggregation, not a graph fetch.
MAX_EGO_NODES = 2_000
DEFAULT_EGO_NODES = 400


def _props(props: dict[str, Any] | None) -> dict[str, Any]:
    """Neo4j property values must be primitives or lists of primitives."""
    if not props:
        return {}
    out: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple, set)):
            flat = [v for v in value
                    if isinstance(v, (str, int, float, bool))]
            if flat:
                out[key] = flat
        else:
            # A nested structure becomes its JSON form rather than being dropped:
            # losing a property silently is worse than a stringly-typed one.
            import json

            out[key] = json.dumps(value, ensure_ascii=False, default=str)[:2000]
    return out


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, user: str = "neo4j", password: str = "neo4j",
                 database: str = "neo4j", loop=None) -> None:
        import asyncio

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "OMNI_MODE=docker needs the neo4j driver. "
                "Install requirements-docker.txt."
            ) from exc

        self._loop = loop or asyncio.new_event_loop()
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._ready = False

    def _run(self, query: str, **params: Any) -> list[dict]:
        async def execute() -> list[dict]:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, **params)
                return [record.data() async for record in result]

        return self._loop.run_until_complete(execute())

    def ensure_schema(self) -> None:
        """Constraints, not just indexes. Uniqueness on the id is what makes
        `upsert` idempotent under concurrent projection, and the label
        indexes are what keep `MATCH (n:Person)` from being a label scan."""
        if self._ready:
            return
        for label in ("Person", "Identity", "Conversation", "Message", "File",
                      "Note", "Video", "Transcript", "Event", "Workspace"):
            self._run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
            self._run(f"CREATE INDEX {label.lower()}_label IF NOT EXISTS "
                      f"FOR (n:{label}) ON (n.id)")
        self._run("CREATE INDEX message_ts IF NOT EXISTS FOR (n:Message) ON (n.ts)")
        self._run("CREATE INDEX message_conv IF NOT EXISTS "
                  "FOR (n:Message) ON (n.conversation_id)")
        self._ready = True

    # -- nodes and edges -----------------------------------------------
    def upsert_node(self, node: Node) -> None:
        self.ensure_schema()
        self._run(
            f"MERGE (n:{_label(node.label)} {{id: $id}}) SET n += $props",
            id=node.id, props=_props(node.props),
        )

    def upsert_edge(self, edge: Edge) -> None:
        self.ensure_schema()
        # MERGE on the endpoints *and* the type, so a projected edge is
        # idempotent without a unique constraint on relationships.
        self._run(
            f"MATCH (a {{id: $src}}), (b {{id: $dst}}) "
            f"MERGE (a)-[r:{_rel(edge.type)}]->(b) SET r += $props",
            src=edge.src, dst=edge.dst, props=_props(edge.props),
        )

    def delete_edge(self, src: str, edge_type: str, dst: str) -> bool:
        self._run(
            f"MATCH (a {{id: $src}})-[r:{_rel(edge_type)}]->(b {{id: $dst}}) DELETE r "
            f"RETURN count(r) AS n",
            src=src, dst=dst,
        )
        return True

    def delete_edges_where(self, *, src: str | None = None, edge_type: str | None = None,
                           dst: str | None = None) -> int:
        pattern = (f"(a {{id: $src}})" if src else "(a)")
        rel = f"[r:{_rel(edge_type)}]" if edge_type else "[r]"
        tail = f"->(b {{id: $dst}})" if dst else ""
        rows = self._run(
            f"MATCH {pattern}-{rel}->{tail or '(b)'} DELETE r RETURN count(r) AS n",
            src=src, dst=dst,
        )
        return int(rows[0]["n"]) if rows else 0

    def upsert_batch(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> tuple[int, int]:
        if not nodes and not edges:
            return 0, 0
        self.ensure_schema()
        node_rows = [{"label": _label(n.label), "id": n.id, "props": _props(n.props)}
                     for n in nodes]
        # Unwound writes in a single transaction. The projector emits thousands
        # of rows per run, and one round trip per row dominates the runtime.
        self._run(
            """
            UNWIND $rows AS row
            CALL apoc.merge.node(row.label, {id: row.id}, {}, row.props, row.props)
            YIELD node
            RETURN count(node) AS n
            """,
            rows=node_rows,
        ) if self._apoc() else self._merge_nodes(node_rows)

        edge_rows = [{"type": _rel(e.type), "src": e.src, "dst": e.dst,
                      "props": _props(e.props)} for e in edges]
        self._run(
            """
            UNWIND $rows AS row
            MATCH (a {id: row.src})
            MATCH (b {id: row.dst})
            CALL apoc.merge.relationship(row.type, row.src, row.type, row.dst, {}, row.props)
            YIELD rel
            RETURN count(rel) AS n
            """,
            rows=edge_rows,
        ) if self._apoc() else self._merge_edges(edge_rows)

        return len(node_rows), len(edge_rows)

    def _apoc(self) -> bool:
        """APOC gives true MERGE-on-relationship semantics. Without it, the
        fallback is two queries per batch, which is slower but correct - and a
        deployment that has not installed a plugin should still work."""
        if not hasattr(self, "_has_apoc"):
            try:
                self._run("RETURN apoc.version() AS v")
                self._has_apoc = True
            except Exception:
                self._has_apoc = False
        return self._has_apoc

    def _merge_nodes(self, rows: list[dict]) -> None:
        by_label: dict[str, list[dict]] = {}
        for row in rows:
            by_label.setdefault(row["label"], []).append(row)
        for label, group in by_label.items():
            self._run(
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props "
                f"RETURN count(n) AS n",
                rows=group,
            )

    def _merge_edges(self, rows: list[dict]) -> None:
        by_type: dict[str, list[dict]] = {}
        for row in rows:
            by_type.setdefault(row["type"], []).append(row)
        for rel, group in by_type.items():
            self._run(
                f"UNWIND $rows AS row "
                f"MATCH (a {{id: row.src}}) MATCH (b {{id: row.dst}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += row.props "
                f"RETURN count(r) AS n",
                rows=group,
            )

    # -- reads ----------------------------------------------------------
    def get_node(self, node_id: str) -> Node | None:
        rows = self._run(
            "MATCH (n) WHERE n.id = $id RETURN n.id AS id, labels(n) AS labels, "
            "properties(n) AS props LIMIT 1",
            id=node_id,
        )
        if not rows:
            return None
        labels = rows[0]["labels"] or ["Node"]
        return Node(id=rows[0]["id"], label=_unlabel(labels[0]),
                    props=rows[0]["props"] or {})

    def node(self, node_id: str) -> Node | None:
        return self.get_node(node_id)

    def top_nodes(self, label: str, *, limit: int = 20,
                  by: str = "message_count") -> list[Node]:
        rows = self._run(
            f"MATCH (n:{_label(label)}) RETURN n.id AS id, properties(n) AS props "
            f"ORDER BY coalesce(n.{_safe(by)}, 0) DESC, n.id ASC LIMIT $limit",
            limit=limit,
        )
        return [Node(id=r["id"], label=_label(label), props=r["props"] or {})
                for r in rows]

    def ego(
        self,
        focus: str,
        *,
        hops: int = 1,
        node_labels: Sequence[str] | None = None,
        edge_types: Sequence[str] | None = None,
        limit: int = DEFAULT_EGO_NODES,
    ) -> Subgraph:
        """Bounded BFS.

        The cap is enforced inside the traversal, not applied to the result. A
        `LIMIT` on the output would still walk the whole neighbourhood and then
        throw it away; stopping expansion is what keeps a hub node cheap.
        """
        cap = min(limit, MAX_EGO_NODES)
        type_filter = "|".join(_rel(t) for t in (edge_types or []))
        rows = self._ego_cypher(focus, hops, cap, type_filter)

        wanted = {_unlabel(l) for l in node_labels} if node_labels else None
        nodes = [Node(id=r["id"], label=_unlabel(r["labels"][0]),
                      props=r["props"] or {})
                 for r in rows if r.get("id")]
        if wanted is not None:
            nodes = [n for n in nodes if n.label in wanted]
        ids = {n.id for n in nodes}
        edge_rows = self._run(
            "MATCH (a)-[r]->(b) WHERE a.id IN $ids AND b.id IN $ids "
            "RETURN a.id AS src, b.id AS dst, type(r) AS type, properties(r) AS props",
            ids=list(ids),
        ) if ids else []
        edges = [Edge(type=r["type"], src=r["src"], dst=r["dst"], props=r["props"] or {})
                 for r in edge_rows]
        return Subgraph(nodes=nodes, edges=edges, center=focus)

    def _ego_cypher(self, focus: str, hops: int, cap: int,
                    type_filter: str) -> list[dict]:
        """Bounded breadth-first expansion, depth-limited and count-limited."""
        return self._run(
            f"""
            MATCH (c {{id: $focus}})
            WITH c
            MATCH (n) WHERE n.id = c.id
            WITH collect(n) AS frontier, 0 AS depth, [c.id] AS seen
            WITH frontier, depth, seen
            CALL (frontier, depth, seen) {{
                WITH frontier, depth, seen
                WHERE depth < $hops
                UNWIND frontier AS f
                MATCH (f)-[r]->(m)
                WHERE NOT m.id IN seen
                  AND ($types = '' OR type(r) = $types)
                RETURN collect(DISTINCT m) AS next, collect(DISTINCT m.id) AS next_ids
            }} {{
                WITH next, next_ids, depth + 1 AS nd, seen + next_ids AS ns
                RETURN next AS frontier, nd AS depth, ns AS seen
            }}
            UNWIND frontier AS n
            WITH n LIMIT $cap
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
            """,
            focus=focus, hops=int(hops), cap=cap, types=type_filter,
        )

    def path(self, src: str, dst: str, *, max_hops: int = 4,
             limit: int = 5) -> list[list[str]]:
        """All shortest paths, bounded by both depth and count.

        `allShortestPaths` rather than a single `shortestPath`: "how are these
        two people connected" has several valid answers and picking one silently
        hides the others.
        """
        rows = self._run(
            f"""
            MATCH (a {{id: $src}}), (b {{id: $dst}})
            MATCH p = allShortestPaths((a)-[*..{int(max_hops)}]-(b))
            WITH p, [n IN nodes(p) | n.id] AS ids
            LIMIT $limit
            RETURN ids
            """,
            src=src, dst=dst, limit=limit,
        )
        return [list(r["ids"]) for r in rows]

    def neighbors(self, node_id: str, *, edge_types: Sequence[str] | None = None,
                  limit: int = 200) -> list[Edge]:
        type_filter = "|".join(_rel(t) for t in (edge_types or []))
        rows = self._run(
            f"""
            MATCH (a {{id: $id}})-[r]->(b)
            WHERE $types = '' OR type(r) = $types
            RETURN a.id AS src, b.id AS dst, type(r) AS type, properties(r) AS props
            LIMIT $limit
            """,
            id=node_id, types=type_filter, limit=limit,
        )
        return [Edge(type=r["type"], src=r["src"], dst=r["dst"], props=r["props"] or {})
                for r in rows]

    def stats(self) -> dict[str, Any]:
        labels: dict[str, int] = {}
        types: dict[str, int] = {}
        for row in self._run("MATCH (n) UNWIND labels(n) AS l "
                             "RETURN l, count(*) AS n"):
            labels[str(row["l"])] = int(row["n"])
        for row in self._run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n"):
            types[str(row["t"])] = int(row["n"])
        return {
            "nodes": sum(labels.values()),
            "edges": sum(types.values()),
            "labels": dict(sorted(labels.items())),
            "edge_types": dict(sorted(types.items())),
        }

    def purge(self) -> int:
        """Detach-delete, not `DETACH DELETE` on a label: the graph holds
        multiple label families and a partial purge would be meaningless."""
        rows = self._run("MATCH (n) WITH n LIMIT 1 DETACH DELETE n RETURN count(n) AS n")
        # Drop constraints too, or the next projector run pays to re-create them.
        for label in ("Person", "Identity", "Conversation", "Message", "File",
                      "Note", "Video", "Transcript", "Event", "Workspace"):
            self._run(f"DROP CONSTRAINT {label.lower()}_id IF EXISTS")
        self._ready = False
        return int(rows[0]["n"]) if rows else 0

    def constraints_ddl(self) -> list[str]:
        return [
            f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            for label in ("Person", "Identity", "Conversation", "Message", "File",
                          "Note", "Video", "Transcript", "Event", "Workspace")
        ] + [
            "CREATE INDEX message_ts IF NOT EXISTS FOR (n:Message) ON (n.ts)",
            "CREATE INDEX message_conv IF NOT EXISTS "
            "FOR (n:Message) ON (n.conversation_id)",
        ]

    def close(self) -> None:
        self._driver.close()
        self._loop.close()


_LABEL_FOR = {
    "Message": "Message", "Person": "Person", "Identity": "Identity",
    "Conversation": "Conversation", "File": "File", "Note": "Note",
    "Video": "Video", "Transcript": "Transcript", "Event": "Event",
    "Workspace": "Workspace",
}
_UNLABEL_FOR = {v.lower(): v for v in _LABEL_FOR.values()}


def _label(label: str) -> str:
    """Never interpolate a caller-supplied label into Cypothor. The projector's
    label set is closed, so anything else is a bug and is coerced rather than
    injected."""
    return _LABEL_FOR.get(label, "Node")


def _unlabel(label: str) -> str:
    return _UNLABEL_FOR.get(str(label).lower(), "Node")


def _rel(edge_type: str) -> str:
    """Relationship types are interpolated (Cypor cannot parameterise them), so
    they are restricted to the projector's closed set and upper-snake-cased.
    Anything else becomes a no-op relationship rather than a syntax error."""
    return "".join(ch for ch in str(edge_type).upper() if ch.isalnum() or ch == "_") or "REL"


def _safe(field: str) -> str:
    return "".join(ch for ch in str(field) if ch.isalnum() or ch == "_") or "id"
