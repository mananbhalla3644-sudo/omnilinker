"""Storage interfaces.

The important design decision: the interfaces are expressed in **domain
operations**, not in Cypher or Mongo query syntax. That is what lets the
embedded backend (pure Python, no services) and the Neo4j/Mongo backend
implement the same contract, and it means a query bug cannot be a Cypher
injection - there is no string-built query anywhere in the request path.

Blueprint 5.1: Neo4j for relationship topology, MongoDB for documents, and the
rule that no source of truth lives in a derived store. Both are projections of
the canonical records, rebuildable from raw.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


#: Every operator the DSL supports. An operator outside this set is a bug in
#: the caller, and it is rejected rather than approximated: a silently
#: mis-evaluated filter returns wrong rows, which is worse than an error
#: because nothing downstream can tell.
FILTER_OPS = frozenset({
    "eq", "ne", "in", "nin", "gt", "gte", "lt", "lte",
    "contains", "startswith", "endswith", "exists",
})


class Filter:
    """A typed predicate. Field access uses dotted paths (`entities.person_ids`)
    so the request layer never builds a query string - there is no injection
    surface because there is no string to inject."""

    def __init__(self, field: str, op: str = "eq", value: Any = None) -> None:
        if op not in FILTER_OPS:
            raise ValueError(
                f"unknown filter operator {op!r}; supported: {sorted(FILTER_OPS)}"
            )
        self.field = field
        self.op = op
        self.value = value

    def __repr__(self) -> str:
        return f"Filter({self.field!r}, {self.op!r}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Filter) and (self.field, self.op, self.value)
                == (other.field, other.op, other.value))

    def __hash__(self) -> int:
        value = self.value if isinstance(self.value, (str, int, float, bool, type(None))) \
            else tuple(self.value or ())
        return hash((self.field, self.op, value))

    def to_dict(self) -> dict:
        return {"field": self.field, "op": self.op, "value": self.value}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Filter":
        return Filter(field=d["field"], op=d.get("op", "eq"), value=d.get("value"))


@dataclass
class Query:
    collection: str
    filters: list[Filter] = field(default_factory=list)
    sort: list[tuple[str, int]] = field(default_factory=list)  # (field, +1/-1)
    limit: int = 50
    skip: int = 0
    projection: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "filters": [f.to_dict() for f in self.filters],
            "sort": self.sort,
            "limit": self.limit,
            "skip": self.skip,
        }


class DocStore(ABC):
    """Document store: canonical records, lineage, search docs, audit log."""

    @abstractmethod
    def put(self, collection: str, doc_id: str, doc: Mapping[str, Any]) -> dict:
        """Idempotent upsert keyed by the natural id (blueprint 6.2.3)."""

    @abstractmethod
    def put_many(self, collection: str, docs: Sequence[Mapping[str, Any]], id_field: str) -> int:
        ...

    @abstractmethod
    def get(self, collection: str, doc_id: str) -> dict | None:
        ...

    @abstractmethod
    def get_many(self, collection: str, doc_ids: Sequence[str]) -> list[dict]:
        ...

    @abstractmethod
    def find(self, query: Query) -> list[dict]:
        ...

    @abstractmethod
    def count(self, collection: str, filters: Sequence[Filter] = ()) -> int:
        ...

    @abstractmethod
    def facet(
        self, collection: str, field: str, filters: Sequence[Filter] = (), limit: int = 25
    ) -> list[tuple[str, int]]:
        ...

    @abstractmethod
    def delete(self, collection: str, doc_id: str) -> bool:
        ...

    @abstractmethod
    def purge(self, collections: Iterable[str] | None = None) -> int:
        ...

    @abstractmethod
    def stats(self) -> dict[str, int]:
        ...


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    label: str
    props: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "props": self.props}


@dataclass
class Edge:
    type: str
    src: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.type, self.dst)

    def to_dict(self) -> dict:
        return {"type": self.type, "source": self.src, "target": self.dst, "props": self.props}


@dataclass
class Subgraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    truncated: bool = False
    total_nodes: int = 0

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "truncated": self.truncated,
            "total_nodes": self.total_nodes,
        }


class GraphStore(ABC):
    """Graph store: the relationship topology (blueprint 5.2)."""

    @abstractmethod
    def upsert_node(self, node: Node) -> None:
        ...

    @abstractmethod
    def upsert_edge(self, edge: Edge) -> None:
        ...

    @abstractmethod
    def delete_edge(self, src: str, edge_type: str, dst: str) -> bool:
        """Needed by the graph projector, which treats the graph as a pure
        function of the documents and therefore must be able to retract an edge
        it previously wrote when an identity merge changes its endpoint."""

    @abstractmethod
    def delete_edges_where(self, *, src: str | None = None, edge_type: str | None = None,
                           dst: str | None = None) -> int:
        ...

    @abstractmethod
    def upsert_batch(
        self, nodes: Sequence[Node] = (), edges: Sequence[Edge] = ()
    ) -> tuple[int, int]:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None:
        ...

    @abstractmethod
    def ego(
        self,
        focus: str,
        *,
        hops: int = 1,
        node_labels: Sequence[str] | None = None,
        edge_types: Sequence[str] | None = None,
        limit: int = 400,
    ) -> Subgraph:
        """Neighborhood around `focus`. Must be bounded - a Person can reach
        >1M messages, so no implementation may traverse unbounded (5.2.4)."""

    @abstractmethod
    def path(self, src: str, dst: str, *, max_hops: int = 4, limit: int = 5) -> list[list[str]]:
        """Up to `limit` shortest paths, each a list of node ids."""

    @abstractmethod
    def node(self, node_id: str) -> Node | None:
        """Fetch several nodes at once. Separate from `path()` because a path
        returns ids and the caller almost always needs the nodes to render
        them - a round trip per hop is the obvious way to build a graph view
        that stutters."""

    @abstractmethod
    def top_nodes(self, label: str, *, limit: int = 20,
                  by: str = "message_count") -> list[Node]:
        """Highest-degree / highest-weight nodes of one label.

        `by` names a node property, not a hardcoded metric, so the same call
        serves "most active people" and "largest files" without the caller
        needing to know how the backend ranks.
        """

    @abstractmethod
    def neighbors(
        self, node_id: str, *, edge_types: Sequence[str] | None = None, limit: int = 200
    ) -> list[Edge]:
        ...

    @abstractmethod
    def stats(self) -> dict[str, int]:
        ...

    @abstractmethod
    def purge(self) -> int:
        ...

    @abstractmethod
    def constraints_ddl(self) -> list[str]:
        """DDL a human can run against a real Neo4j to reproduce the model."""
        ...


# ---------------------------------------------------------------------------
# Factory (blueprint: services must not branch on deployment profile)
# ---------------------------------------------------------------------------


@dataclass
class Stores:
    docs: DocStore
    graph: GraphStore
    backend: str
    #: Where this profile keeps its state on disk, when it has a location.
    #: `None` for a remote profile, where the data is not ours to point at.
    data_root: Any = None

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "documents": self.docs.stats(),
            "graph": self.graph.stats(),
        }
