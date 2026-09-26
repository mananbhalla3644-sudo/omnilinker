"""Embedded graph store: in-memory adjacency + JSON persistence.

Blueprint 5.2 says Neo4j. This implements the same `GraphStore` contract with
bounded BFS so the ego/path semantics (and the 5.2.4 "never traverse
unbounded" guardrail) are enforced in the default profile too. The Neo4j
backend implements the identical contract with parameterized Cypher.

Why adjacency maps and not a real graph engine: at the scale this slice
handles (tens of thousands of nodes) plain dicts are faster than any network
hop, and the JSON file is diffable and inspectable, which matters a lot when
you are debugging a privacy tool.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from omnilinker.store.base import Edge, GraphStore, Node, Subgraph

#: Reverse relationship map (blueprint 5.2.2), so we can describe a node by
#: what it means rather than what it points at.
EDGE_LABELS: dict[str, str] = {
    "HAS_IDENTITY": "identity",
    "SENT_BY": "message",
    "AUTHORED": "message",
    "IN_CONVERSATION": "message",
    "MEMBER_OF": "conversation",
    "PARTICIPATED_IN": "conversation",
    "ATTACHED_TO": "attachment",
    "OWNS": "file",
    "SHARED_WITH": "file",
    "MENTIONS": "mention",
    "RESOLVES_TO": "resolved to",
    "PART_OF": "part of",
    "ABOUT": "about",
    "LINKED_TO": "linked to",
    "SIMILAR_TO": "similar to",
    "DUPLICATE_OF": "duplicate of",
    "VERSION_OF": "version of",
    "SENT_IN": "sent in",
    "CONTAINS": "contains",
    "OCCURRED_AT": "occurred at",
    "REPLY_TO": "reply to",
    "THREAD_ROOT": "thread root",
    "HAS_SUMMARY": "summary",
    "HAS_TRANSCRIPT": "transcript",
    "HAS_SEGMENT": "segment",
    "EDITED": "edited",
    "PARENT_OF": "parent of",
    "IN_SPACE": "in space",
    "PUBLISHED_BY": "published by",
    "MANUAL_MEMBERSHIP": "manual membership",
    "PREDICTED_DUE": "predicted due",
}


class EmbeddedGraphStore(GraphStore):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / "graph.json"
        self._lock = threading.RLock()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._out: dict[str, set[str]] = defaultdict(set)
        self._in: dict[str, set[str]] = defaultdict(set)
        self._dirty = False
        #: Public so callers (and tests) can locate the on-disk state
        #: without guessing at the file layout.
        self.data_root = root
        self._load()

    # -- persistence --------------------------------------------------
    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            payload = json.loads(self._file.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            self._file.replace(self._file.with_suffix(".json.corrupt"))
            return
        for raw in payload.get("nodes", []):
            node = Node(id=raw["id"], label=raw["label"], props=raw.get("props") or {})
            self._nodes[node.id] = node
        for raw in payload.get("edges", []):
            self._add_edge(Edge(raw["type"], raw["source"], raw["target"], raw.get("props") or {}))
        self._dirty = False

    def flush(self) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            payload = {
                "v": 1,
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges.values()],
            }
            tmp = self._file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "utf-8")
            tmp.replace(self._file)
            self._dirty = False
            return True

    def _add_edge(self, edge: Edge) -> None:
        if edge.key not in self._edges:
            self._edges[edge.key] = edge
            self._out[edge.src].add(edge.dst)
            self._in[edge.dst].add(edge.src)

    # -- GraphStore ---------------------------------------------------
    def upsert_node(self, node: Node) -> None:
        with self._lock:
            existing = self._nodes.get(node.id)
            if existing:
                existing.props.update(node.props)  # merge, like SET n += {...}
                existing.label = node.label
            else:
                self._nodes[node.id] = node
            self._dirty = True

    def upsert_edge(self, edge: Edge) -> None:
        with self._lock:
            existing = self._edges.get(edge.key)
            if existing:
                existing.props.update(edge.props)
            else:
                self._add_edge(edge)
            self._dirty = True

    def delete_edge(self, src: str, edge_type: str, dst: str) -> bool:
        with self._lock:
            key = (src, edge_type, dst)
            if key not in self._edges:
                return False
            del self._edges[key]
            self._out[src].discard(dst)
            self._in[dst].discard(src)
            self._dirty = True
            return True

    def delete_edges_where(self, *, src: str | None = None, edge_type: str | None = None,
                           dst: str | None = None) -> int:
        with self._lock:
            doomed = [
                key for key in self._edges
                if (src is None or key[0] == src)
                and (edge_type is None or key[1] == edge_type)
                and (dst is None or key[2] == dst)
            ]
            for key in doomed:
                self._edges.pop(key, None)
                self._out[key[0]].discard(key[2])
                self._in[key[2]].discard(key[0])
            if doomed:
                self._dirty = True
            return len(doomed)

    def upsert_batch(
        self, nodes: Sequence[Node] = (), edges: Sequence[Edge] = ()
    ) -> tuple[int, int]:
        with self._lock:
            for node in nodes:
                existing = self._nodes.get(node.id)
                if existing:
                    existing.props.update(node.props)
                    existing.label = node.label
                else:
                    self._nodes[node.id] = node
            for edge in edges:
                existing = self._edges.get(edge.key)
                if existing:
                    existing.props.update(edge.props)
                else:
                    self._add_edge(edge)
            if nodes or edges:
                self._dirty = True
            return len(nodes), len(edges)

    def get_node(self, node_id: str) -> Node | None:
        with self._lock:
            return self._nodes.get(node_id)

    def _incident(self, node_id: str) -> set[str]:
        return set(self._out.get(node_id, ())) | set(self._in.get(node_id, ()))

    def ego(
        self,
        focus: str,
        *,
        hops: int = 1,
        node_labels: Sequence[str] | None = None,
        edge_types: Sequence[str] | None = None,
        limit: int = 400,
    ) -> Subgraph:
        with self._lock:
            if focus not in self._nodes:
                return Subgraph(total_nodes=0)
            allowed_labels = set(node_labels) if node_labels else None
            allowed_edges = set(edge_types) if edge_types else None
            hops = max(1, min(int(hops), 3))  # hard cap: 5.2.4 guardrail

            seen: dict[str, int] = {focus: 0}
            order: list[str] = [focus]
            frontier: deque[str] = deque([focus])
            truncated = False

            while frontier:
                current = frontier.popleft()
                depth = seen[current]
                if depth >= hops:
                    continue
                for neighbour in self._incident(current):
                    if neighbour not in self._nodes:
                        continue
                    if allowed_labels and self._nodes[neighbour].label not in allowed_labels:
                        # Still traverse *through* a filtered node, just don't
                        # show it - otherwise "show me people" hides every path
                        # that routes through a Message.
                        if neighbour not in seen:
                            seen[neighbour] = depth + 1
                            frontier.append(neighbour)
                        continue
                    if neighbour not in seen:
                        seen[neighbour] = depth + 1
                        order.append(neighbour)
                        frontier.append(neighbour)
                    if len(order) > limit:
                        truncated = True
                        frontier.clear()
                        break

            nodes = [self._nodes[n] for n in order if n in self._nodes]
            keep = set(order)
            edges = [
                e
                for e in self._edges.values()
                if e.src in keep and e.dst in keep and (not allowed_edges or e.type in allowed_edges)
            ]
            total = len(seen)
            return Subgraph(nodes=nodes, edges=edges, truncated=truncated, total_nodes=total)

    def neighbors(
        self, node_id: str, *, edge_types: Sequence[str] | None = None, limit: int = 200
    ) -> list[Edge]:
        with self._lock:
            allowed = set(edge_types) if edge_types else None
            out: list[Edge] = []
            for key, edge in self._edges.items():
                if edge.src != node_id and edge.dst != node_id:
                    continue
                if allowed and edge.type not in allowed:
                    continue
                out.append(edge)
                if len(out) >= limit:
                    break
            return out

    def path(self, src: str, dst: str, *, max_hops: int = 4, limit: int = 5) -> list[list[str]]:
        """Bidirectional BFS. Returns up to `limit` distinct node-id paths."""
        with self._lock:
            if src not in self._nodes or dst not in self._nodes:
                return []
            if src == dst:
                return [[src]]
            max_hops = max(1, min(int(max_hops), 6))

            # forward distances
            fwd: dict[str, list[str]] = {src: [src]}
            q: deque[tuple[str, int]] = deque([(src, 0)])
            while q:
                node, depth = q.popleft()
                if depth >= max_hops:
                    continue
                for nb in self._incident(node):
                    if nb not in fwd:
                        fwd[nb] = fwd[node] + [nb]
                        q.append((nb, depth + 1))

            results: list[list[str]] = []
            if dst in fwd:
                results.append(fwd[dst])
            else:
                # backward search meeting in the middle
                bwd: dict[str, list[str]] = {dst: [dst]}
                q = deque([(dst, 0)])
                seen: set[str] = set()
                while q and len(results) < limit:
                    node, depth = q.popleft()
                    if depth >= max_hops:
                        continue
                    for nb in self._incident(node):
                        if nb in seen:
                            continue
                        seen.add(nb)
                        if nb in fwd:
                            results.append(fwd[nb] + bwd[node][::-1][1:])
                            if len(results) >= limit:
                                break
                        else:
                            bwd[nb] = bwd[node] + [nb]
                            q.append((nb, depth + 1))
            # shortest first, dedup
            uniq: list[list[str]] = []
            for path in sorted(results, key=len):
                if path not in uniq:
                    uniq.append(path)
            return uniq[:limit]

    def node(self, node_id: str) -> Node | None:
        """Singular alias of `get_node`, so `path()` and `node()` read as one
        API: fetch paths as ids, hydrate them with nodes."""
        return self.get_node(node_id)

    def top_nodes(self, label: str, *, limit: int = 20,
                  by: str = "message_count") -> list[Node]:
        with self._lock:
            candidates = [n for n in self._nodes.values() if n.label == label]
            # A missing property sorts last rather than throwing, so a graph
            # with heterogeneous node schemas still answers "most active".
            return sorted(
                candidates,
                key=lambda n: (-float(n.props.get(by) or 0.0), n.id),
            )[:limit]

    def stats(self) -> dict[str, int]:
        with self._lock:
            by_label: dict[str, int] = defaultdict(int)
            for node in self._nodes.values():
                by_label[node.label] += 1
            by_type: dict[str, int] = defaultdict(int)
            for edge in self._edges.values():
                by_type[edge.type] += 1
            return {
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "labels": dict(sorted(by_label.items())),
                "edge_types": dict(sorted(by_type.items())),
            }

    def purge(self) -> int:
        with self._lock:
            count = len(self._nodes)
            self._nodes.clear()
            self._edges.clear()
            self._out.clear()
            self._in.clear()
            self._dirty = True
            return count

    def constraints_ddl(self) -> list[str]:
        """The DDL a human runs against a real Neo4j to reproduce this model
        (blueprint 5.2.3). Exposed via the API so the two backends stay
        documented in one place."""
        labels = ["Workspace", "Person", "Identity", "Conversation", "Message", "File",
                  "Note", "EmailThread", "Event", "Video", "Project", "Topic", "Summary"]
        id_fields = {
            "Workspace": "workspace_id", "Person": "person_id", "Identity": "identity_id",
            "Conversation": "conversation_id", "Message": "message_id", "File": "file_id",
            "Note": "note_id", "EmailThread": "thread_id", "Event": "event_id",
            "Video": "video_id", "Project": "project_id", "Topic": "topic_id",
            "Summary": "summary_id",
        }
        out = []
        for label in labels:
            field = id_fields[label]
            out.append(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{field} IS UNIQUE;"
            )
        out.append("CREATE INDEX message_ts IF NOT EXISTS FOR (m:Message) ON (m.ts);")
        out.append("CREATE INDEX person_last IF NOT EXISTS FOR (p:Person) ON (p.last_seen);")
        out.append(
            "CREATE FULLTEXT INDEX message_text IF NOT EXISTS "
            "FOR (m:Message) ON EACH [m.body_text, m.provider];"
        )
        return out
