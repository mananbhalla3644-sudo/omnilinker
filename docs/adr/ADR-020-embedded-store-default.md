# ADR-020: the embedded store is the default

**Status:** accepted · **Area:** stores · **Reversible:** yes, configuration

## The blueprint specifies

Neo4j for the relationship graph, MongoDB for documents, dockerised.

## The default is

`EmbeddedDocStore` (one JSON file per collection) and `EmbeddedGraphStore` (an
in-memory adjacency map, flushed to one JSON file). The Mongo and Neo4j
backends are implemented and selected with `OMNI_MODE=docker`.

## Why

The product's first impression matters more than its asymptotics. A user who
clones this repo should get a working, populated, *legible* system in under a
minute, and should never have to install Docker to see what the thing does. A
`docker compose up` as the first step of a first run is a support burden that
disproportionately lands on exactly the users a personal-knowledge tool is for.

The interfaces make it a configuration change rather than a rewrite:

```python
class DocStore(ABC):
    def put(self, collection, doc_id, doc) -> dict: ...
    def find(self, query: Query) -> list[dict]: ...
    # facet / count / delete / purge / stats

class GraphStore(ABC):
    def upsert_node(self, node: Node) -> None: ...
    def ego(self, focus, *, hops, node_labels, edge_types, limit) -> Subgraph: ...
    # path / neighbors / delete_edge / top_nodes / constraints_ddl
```

The rule that makes the swap safe: **the store never sees a query string.**
Filters arrive as typed `Filter` objects and translation to a Mongo document
happens in exactly one function. There is no string to inject and no caller that
could build one — and the same holds for Cypher, where the projector emits node
and edge objects rather than statements.

## Residual risk

**The embedded backend is a single-process development profile, not a
deployment.** It is not tuned for millions of documents, holds the graph in
memory, and the search index is per-process — so `--workers > 1` would give
each worker a divergent view. The Dockerfile pins one worker and says why.

**Single-process means single-writer.** Concurrent requests serialise on the
store's lock. That is fine for one user; it is not a server.

**`OMNI_MODE=docker` is written but unexercised here.** Docker is not installed
on the build machine, so those backends are covered by contract tests against
the interface and by review, not by a running integration test. That gap is real
and is stated in the README rather than glossed.
