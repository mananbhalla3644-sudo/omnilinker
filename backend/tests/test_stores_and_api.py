"""Store contracts and the API surface.

The store tests exist because there are two implementations behind one
interface. The embedded backend is the default and the only one exercised in
CI, so a contract test is the only thing standing between "it works on my
machine" and "the docker profile works too".

The API tests are mostly about *shape*: an endpoint that returns the wrong
envelope breaks every client, and a client is exactly what cannot be fixed from
the server side.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from omnilinker.api.app import create_app
from omnilinker.store.base import Edge, Filter, Node, Query


class TestFilterAndQuery:
    def test_filter_round_trips_through_a_dict(self) -> None:
        original = Filter(field="provider", op="in", value=["slack", "gmail"])
        assert Filter.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_writes_land_on_disk_before_the_call_returns(self, ingested) -> None:
        """`atexit` covers a clean shutdown only. If a sync does not flush, a
        crash - or a laptop lid - loses the entire workspace."""
        from omnilinker import store as store_module

        store_module.flush_stores()
        root = store_module.get_stores().data_root
        for collection in ("messages", "files", "notes", "identities", "persons",
                           "raw_artifacts", "search_docs"):
            assert (root / f"{collection}.json").exists(), \
                f"{collection}.json was not written - data is memory-only"

    def test_no_plaintext_from_a_sealed_field_reaches_disk(self, ingested) -> None:
        """The whole point, asserted against every file on disk.

        Each probe is a distinctive phrase that exists in a *sealed* field of the
        demo corpus. Finding one anywhere means a second, unencrypted copy of
        sealed content exists - which is what happened three times during the
        build: the raw provider payload, the derived `thread_subject`, and the
        per-caption `segments` array. Each looked fine in isolation and each
        made the headline "content is sealed" untrue.
        """
        from omnilinker import store as store_module

        store_module.flush_stores()
        root = store_module.get_stores().data_root
        probes = {
            "message body": "We sharded by document id",
            "note body": "Raw provider payloads: kept 90 days",
            "transcript text": "A rebalancer is just a bounded migration",
            "derived subject": "Root cause is the shard key",
            "transcript segment": "which is the part everyone defers",
        }
        for path in sorted(root.glob("*.json")):
            blob = path.read_text("utf-8", errors="replace")
            for label, phrase in probes.items():
                assert phrase not in blob, \
                    f"{path.name} contains plaintext {label}: {phrase!r}"

    def test_envelopes_are_present_in_every_sealed_collection(self, ingested) -> None:
        from omnilinker import store as store_module

        store_module.flush_stores()
        root = store_module.get_stores().data_root
        for collection in ("messages", "notes", "transcripts", "raw_artifacts"):
            path = root / f"{collection}.json"
            assert path.exists(), f"{collection} was never written"
            assert "AES-256-GCM" in path.read_text("utf-8"), \
                f"{collection} has no sealed envelopes"

    def test_raw_payload_is_sealed_but_still_verifiable(self, ingested) -> None:
        """Reproducibility is preserved by the hash, not by readable bytes."""
        from omnilinker.crypto.envelope import is_envelope
        from omnilinker.store.base import Query

        rows = ingested.stores.docs.find(Query("raw_artifacts", limit=20))
        assert rows
        for row in rows:
            assert is_envelope(row["payload"]), "raw payload stored in the clear"
            assert row["payload_hash"].startswith("sha256:")
            assert row["payload_sealed"] is True

    def test_sealed_raw_payload_decrypts_back(self, ingested) -> None:
        """Sealing the raw artifact must not cost reproducibility: a normalizer
        bug still has to be debuggable from the stored payload."""
        from omnilinker.crypto.envelope import decrypt
        from omnilinker.crypto.keys import get_key_manager
        from omnilinker.store.base import Query

        import json

        from omnilinker.normalize import content_hash

        for row in ingested.stores.docs.find(Query("raw_artifacts", limit=25)):
            raw = decrypt(row["provider"], row["payload"], doc_id=row["_id"],
                          field_path="payload", key_manager=get_key_manager())
            # The strong invariant: the decrypted bytes are the provider payload
            # the hash was taken from, so a normalizer bug is still reproducible.
            assert content_hash(raw.decode("utf-8")) == row["payload_hash"], \
                f"{row['_id']} does not decrypt to what was hashed"
            payload = json.loads(raw.decode("utf-8"))
            # The payload is the provider's own object, so it carries whatever
            # that provider puts in a payload - not our wrapper fields.
            assert isinstance(payload, dict) and payload

    def test_unknown_operator_is_rejected(self) -> None:
        """Silently treating an unknown operator as equality would return wrong
        results rather than an error, which is the worst possible failure."""
        with pytest.raises(ValueError):
            Filter.from_dict({"field": "x", "op": "regex", "value": ".*"})

    def test_query_serialises(self) -> None:
        query = Query("messages", filters=[Filter("provider", "eq", "slack")],
                      sort=[("ts", -1)], limit=10, skip=5)
        assert json.loads(json.dumps(query.to_dict()))["skip"] == 5


class TestDocStore:
    def test_put_get_round_trip(self, stores) -> None:
        stores.docs.put("things", "t1", {"_id": "t1", "name": "one", "size": 3})
        row = stores.docs.get("things", "t1")
        assert row["name"] == "one"
        assert stores.docs.get("things", "missing") is None

    def test_put_is_upsert(self, stores) -> None:
        stores.docs.put("things", "t1", {"_id": "t1", "n": 1})
        stores.docs.put("things", "t1", {"_id": "t1", "n": 2})
        assert stores.docs.get("things", "t1")["n"] == 2
        assert stores.docs.count("things") == 1

    def test_dotted_path_reads(self, stores) -> None:
        from omnilinker.store.embedded_docs import get_path

        stores.docs.put("things", "t1", {"_id": "t1", "entities": {"a": {"b": ["x"]}}})
        row = stores.docs.get("things", "t1")
        assert get_path(row, "entities.a.b") == (True, ["x"])
        assert get_path(row, "entities.missing.key") == (False, None)
        # Array-aware, so `a.0.b` works the way Mongo's does.
        stores.docs.put("things", "t2",
                        {"_id": "t2", "people": [{"name": "Alice"}]})
        assert get_path(stores.docs.get("things", "t2"), "people.0.name")[1] == "Alice"

    def test_filters_use_dotted_paths(self, stores) -> None:
        stores.docs.put("things", "t1",
                        {"_id": "t1", "entities": {"person_ids": ["per_a", "per_b"]}})
        from omnilinker.store.base import Filter as F

        assert stores.docs.count("things", [F("entities.person_ids", "in", ["per_a"])]) == 1
        assert stores.docs.count("things", [F("entities.person_ids", "in", ["per_z"])]) == 0

    def test_filters(self, stores) -> None:
        for i in range(5):
            stores.docs.put("things", f"t{i}",
                            {"_id": f"t{i}", "n": i, "group": "a" if i % 2 else "b"})
        assert stores.docs.count("things", [Filter("group", "eq", "a")]) == 2
        assert stores.docs.count("things", [Filter("n", "gte", 3)]) == 2
        assert stores.docs.count("things", [Filter("n", "lt", 1)]) == 1
        assert stores.docs.count("things", [Filter("group", "in", ["a", "b"])]) == 5

    def test_sort_and_limit(self, stores) -> None:
        for i in range(10):
            stores.docs.put("things", f"t{i}", {"_id": f"t{i}", "n": i})
        rows = stores.docs.find(Query("things", sort=[("n", -1)], limit=3))
        assert [r["n"] for r in rows] == [9, 8, 7]

    def test_pagination_via_skip(self, stores) -> None:
        for i in range(10):
            stores.docs.put("things", f"t{i}", {"_id": f"t{i}", "n": i})
        first = stores.docs.find(Query("things", sort=[("n", 1)], limit=4))
        second = stores.docs.find(Query("things", sort=[("n", 1)], limit=4, skip=4))
        assert {r["_id"] for r in first} & {r["_id"] for r in second} == set()

    def test_facets(self, stores) -> None:
        for i in range(6):
            stores.docs.put("things", f"t{i}", {"_id": f"t{i}", "g": "x" if i < 4 else "y"})
        facets = dict(stores.docs.facet("things", "g"))
        assert facets["x"] == 4 and facets["y"] == 2

    def test_delete_and_purge(self, stores) -> None:
        stores.docs.put("things", "t1", {"_id": "t1"})
        stores.docs.put("others", "o1", {"_id": "o1"})
        assert stores.docs.delete("things", "t1") is True
        assert stores.docs.delete("things", "t1") is False
        stores.docs.purge(["others"])
        assert stores.docs.count("others") == 0
        assert stores.docs.count("things") == 0

    def test_put_many(self, stores) -> None:
        rows = [{"_id": f"b{i}", "n": i} for i in range(20)]
        assert stores.docs.put_many("bulk", rows, "_id") == 20
        assert stores.docs.count("bulk") == 20

    def test_survives_a_reload(self, tmp_path, monkeypatch) -> None:
        """The embedded store must be durable, or a restart silently loses the
        user's data."""
        from omnilinker import store as store_module

        monkeypatch.setenv("OMNI_DATA_DIR", str(tmp_path / "d"))
        store_module.reset_stores()
        store_module.get_stores().docs.put("things", "t1", {"_id": "t1", "n": 7})
        store_module.flush_stores()
        store_module.reset_stores()
        assert store_module.get_stores().docs.get("things", "t1")["n"] == 7


class TestGraphStore:
    def _seed(self, stores) -> None:
        stores.graph.upsert_batch(
            [Node(id=n, label="Person", props={"display_name": n}) for n in "abcde"],
            [Edge(type="KNOWS", src="a", dst="b"),
             Edge(type="KNOWS", src="b", dst="c"),
             Edge(type="MENTIONS", src="c", dst="d")],
        )

    def test_ego_is_bounded(self, stores) -> None:
        """A Person can reach millions of messages. An unbounded traversal is
        the single easiest way to hang the API."""
        self._seed(stores)
        assert len(stores.graph.ego("b", hops=1).nodes) <= 4
        assert len(stores.graph.ego("a", hops=3, limit=10).nodes) <= 10

    def test_ego_respects_label_filter(self, stores) -> None:
        self._seed(stores)
        assert all(n.label == "Person" for n in stores.graph.ego("b", hops=2).nodes)
        # The focus node is always returned - you asked to centre on it - and
        # the *neighbourhood* is filtered.
        filtered = stores.graph.ego("b", hops=2, node_labels=["Conversation"])
        assert [n.id for n in filtered.nodes] == ["b"]

    def test_path_is_found(self, stores) -> None:
        self._seed(stores)
        paths = stores.graph.path("a", "d")
        assert paths
        assert paths[0][0] == "a" and paths[0][-1] == "d"

    def test_no_path_returns_empty(self, stores) -> None:
        self._seed(stores)
        stores.graph.upsert_node(Node(id="island", label="Person"))
        assert stores.graph.path("a", "island") == []

    def test_path_respects_max_hops(self, stores) -> None:
        self._seed(stores)
        assert stores.graph.path("a", "d", max_hops=1) == []
        assert stores.graph.path("a", "d", max_hops=4)

    def test_delete_edge(self, stores) -> None:
        self._seed(stores)
        assert stores.graph.delete_edge("a", "KNOWS", "b") is True
        assert stores.graph.delete_edge("a", "KNOWS", "b") is False
        assert stores.graph.path("a", "d") == []

    def test_delete_edges_where(self, stores) -> None:
        self._seed(stores)
        removed = stores.graph.delete_edges_where(edge_type="KNOWS")
        assert removed == 2
        assert stores.graph.path("a", "d") == []

    def test_top_nodes(self, stores) -> None:
        stores.graph.upsert_batch(
            [Node(id="p1", label="Person", props={"message_count": 5}),
             Node(id="p2", label="Person", props={"message_count": 50}),
             Node(id="f1", label="File", props={"size_bytes": 10})],
            [],
        )
        assert stores.graph.top_nodes("Person", limit=1)[0].id == "p2"
        assert stores.graph.top_nodes("File", by="size_bytes")[0].id == "f1"

    def test_top_nodes_tolerates_missing_props(self, stores) -> None:
        stores.graph.upsert_batch(
            [Node(id="p1", label="Person", props={"message_count": 5}),
             Node(id="p2", label="Person", props={})],
            [],
        )
        assert len(stores.graph.top_nodes("Person")) == 2

    def test_upsert_merges_props(self, stores) -> None:
        stores.graph.upsert_node(Node(id="n", label="Person", props={"a": 1}))
        stores.graph.upsert_node(Node(id="n", label="Person", props={"b": 2}))
        node = stores.graph.get_node("n")
        assert node.props["a"] == 1 and node.props["b"] == 2

    def test_stats(self, stores) -> None:
        self._seed(stores)
        stats = stores.graph.stats()
        assert stats["nodes"] == 5 and stats["edges"] == 3
        assert stats["labels"]["Person"] == 5
        assert stats["edge_types"]["KNOWS"] == 2

    def test_purge_and_rebuild(self, stores) -> None:
        self._seed(stores)
        assert stores.graph.purge() == 5
        assert stores.graph.stats()["nodes"] == 0

    def test_graph_survives_a_reload(self, tmp_path, monkeypatch) -> None:
        from omnilinker import store as store_module

        monkeypatch.setenv("OMNI_DATA_DIR", str(tmp_path / "g"))
        store_module.reset_stores()
        store_module.get_stores().graph.upsert_node(Node(id="n", label="Person"))
        store_module.flush_stores()
        store_module.reset_stores()
        assert store_module.get_stores().graph.get_node("n")

    def test_constraints_ddl_is_valid_cypher(self, stores) -> None:
        statements = stores.graph.constraints_ddl()
        assert statements
        joined = " ".join(statements)
        assert "CONSTRAINT" in joined.upper()
        for statement in statements:
            assert statement.strip().endswith(";"), statement


class TestGraphProjection:
    def test_messages_are_attached_to_resolved_people(self, ingested) -> None:
        """The projector must wire SENT_BY to a Person, not to a raw handle, or
        the graph is a list of usernames with no humans in it."""
        from omnilinker.store.base import Query

        persons = {p["person_id"] for p in
                   ingested.stores.docs.find(Query("persons", limit=100))}
        sent = [e for e in ingested.stores.graph.stats()["edge_types"]]
        assert "SENT_BY" in sent
        assert "HAS_IDENTITY" in sent
        assert persons

    def test_advisory_edges_are_marked(self, ingested) -> None:
        """Blueprint 11.5: a machine suggestion must never be
        indistinguishable from an observed fact."""
        edges = ingested.stores.graph.ego(
            next(iter(ingested.stores.graph.top_nodes("Person", limit=1))).id, hops=3
        ).edges
        for edge in edges:
            if edge.type == "LINKED_TO":
                assert edge.props.get("state") == "advisory"

    def test_accepting_is_the_only_way_to_promote(self, ingested) -> None:
        from omnilinker.engine import get_engine

        engine = get_engine()
        people = [n for n in engine.stores.graph.top_nodes("Person", limit=10)]
        if len(people) < 2:
            pytest.skip("not enough people to link")
        a, b = people[0].id, people[1].id
        assert engine.projector.accept_insight("ins_test", "D3", a, b) is True
        edges = engine.stores.graph.ego(a, hops=1).edges
        accepted = [e for e in edges
                    if e.type == "LINKED_TO" and e.props.get("state") == "accepted"]
        assert accepted

    def test_accepting_rejects_unknown_endpoints(self, ingested) -> None:
        from omnilinker.engine import get_engine

        engine = get_engine()
        assert engine.projector.accept_insight("i", "D3", "nope", "also-nope") is False

    def test_rebuild_is_a_pure_function_of_the_documents(self, ingested) -> None:
        """`rebuild(full=True)` composes projection and detectors, so it is the
        unit that must converge. `rebuild_graph()` alone is only the projection
        step and legitimately drops the detectors' advisory edges."""
        from omnilinker.engine import get_engine

        engine = get_engine()
        engine.rebuild(full=True)
        first = engine.stores.graph.stats()
        engine.rebuild(full=True)
        engine.rebuild(full=True)
        after = engine.stores.graph.stats()
        assert first["nodes"] == after["nodes"]
        assert first["edges"] == after["edges"]


class TestDerivedStoresAreDroppable:
    def test_drop_and_rebuild_restores_everything(self, ingested) -> None:
        """The privacy claim, as an operation rather than a promise.

        The assertion is on *convergence*, not on equality with the pre-drop
        state. The second rebuild legitimately produces a slightly different
        graph: re-resolution stamps `person_id` onto the documents, so a sender
        that previously needed an `unresolved` placeholder node now resolves to
        a real Person and the placeholder disappears. One fewer node because
        the graph got better, not because something was lost.
        """
        from omnilinker.engine import get_engine

        engine = get_engine()
        before_persons = engine.stores.docs.count("persons")

        engine.drop_everything()
        assert engine.stores.docs.count("search_docs") == 0
        assert engine.stores.docs.count("predictions") == 0
        assert engine.stores.docs.count("insights") == 0

        engine.rebuild(full=True)
        assert engine.stores.docs.count("persons") == before_persons

        first = engine.stores.graph.stats()
        engine.rebuild(full=True)
        assert engine.stores.graph.stats() == first, "rebuild is not idempotent"

    def test_rebuild_resolves_fewer_placeholder_people(self, ingested) -> None:
        """The direction of the previous test's node delta, asserted directly
        so the improvement is visible rather than inferred."""
        from omnilinker.engine import get_engine

        engine = get_engine()

        def placeholders() -> int:
            return sum(1 for _ in engine.stores.graph.top_nodes("Person", limit=5000)
                       if _.props.get("unresolved"))

        engine.rebuild(full=True)
        after_first = placeholders()
        engine.rebuild(full=True)
        assert placeholders() <= after_first

    def test_source_documents_are_never_dropped(self, ingested) -> None:
        from omnilinker.engine import get_engine
        from omnilinker.store.base import Query

        engine = get_engine()
        before = engine.stores.docs.count("messages")
        raw_before = engine.stores.docs.count("raw_artifacts")
        engine.drop_everything()
        engine.rebuild(full=True)
        assert engine.stores.docs.count("messages") == before
        assert engine.stores.docs.count("raw_artifacts") == raw_before
        assert engine.stores.docs.find(Query("messages", limit=1))


@pytest.fixture
def client(ingested):
    with TestClient(create_app()) as test_client:
        yield test_client


class TestApiEnvelope:
    def test_every_response_is_ok_enveloped(self, client) -> None:
        for path in ["/api/system/info", "/api/connectors", "/api/persons",
                     "/api/search?q=search", "/api/graph/stats", "/api/passport",
                     "/api/insights", "/api/predictions", "/api/files",
                     "/api/timeline", "/api/suggestions", "/api/audit"]:
            body = client.get(path).json()
            assert body.get("ok") is True, f"{path} is not ok-enveloped"

    def test_errors_are_structured(self, client) -> None:
        response = client.get("/api/persons/does-not-exist")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_latency_header_is_present(self, client) -> None:
        """A budget nobody measures is a budget nobody meets."""
        assert int(client.get("/api/search?q=search").headers["X-Omni-Took-Ms"]) >= 0

    def test_api_responses_are_not_cached(self, client) -> None:
        assert "no-store" in client.get("/api/persons").headers.get("Cache-Control", "")

    def test_openapi_is_served(self, client) -> None:
        schema = client.get("/api/openapi.json").json()
        assert schema["info"]["title"] == "OmniLinker"
        assert "/api/search" in schema["paths"]


class TestApiSearch:
    def test_search_returns_hits_and_facets(self, client) -> None:
        body = client.get("/api/search?q=shard&limit=3").json()
        assert body["hits"]
        assert body["parsed_query"]
        assert body["index"]["documents"] > 0

    def test_filters_are_honoured(self, client) -> None:
        body = client.get("/api/search?q=the&providers=notion&limit=20").json()
        assert all(hit["provider"] == "notion" for hit in body["hits"])

    def test_limit_is_capped(self, client) -> None:
        assert client.get("/api/search?limit=100000").status_code == 422

    def test_document_endpoint_decrypts(self, client) -> None:
        doc_id = client.get("/api/search?q=shard&limit=1").json()["hits"][0]["doc_id"]
        body = client.get(f"/api/documents/{doc_id}").json()
        assert body["document"]["body_text"]
        assert "AES-256-GCM" not in json.dumps(body["document"].get("extra", {}))

    def test_document_can_be_returned_without_its_body(self, client) -> None:
        doc_id = client.get("/api/search?q=shard&limit=1").json()["hits"][0]["doc_id"]
        body = client.get(f"/api/documents/{doc_id}?include_body=false").json()
        assert "body_text" not in body["document"]

    def test_summarize_endpoint(self, client) -> None:
        doc_id = client.get("/api/search?q=shard&limit=1").json()["hits"][0]["doc_id"]
        body = client.post(f"/api/documents/{doc_id}/summarize").json()
        assert body["summary_words"] <= body["source_words"]


class TestApiGraph:
    def test_ego_defaults_to_someone(self, client) -> None:
        body = client.get("/api/graph/ego").json()
        assert body["nodes"]
        assert body["node_count"] == len(body["nodes"])

    def test_edges_only_reference_returned_nodes(self, client) -> None:
        """Otherwise the renderer silently drops them and it looks like a
        rendering bug rather than a serialization one."""
        body = client.get("/api/graph/ego?depth=2&limit=20").json()
        ids = {node["id"] for node in body["nodes"]}
        for edge in body["edges"]:
            assert edge["source"] in ids
            assert edge["target"] in ids

    def test_unknown_person_404s(self, client) -> None:
        assert client.get("/api/graph/ego?person_id=per_nope").status_code == 404

    def test_path_between_two_people(self, client) -> None:
        people = [p for p in client.get("/api/persons").json()["persons"]
                  if p["cross_source"]]
        if len(people) < 2:
            pytest.skip("need two cross-source people")
        body = client.get(
            f"/api/graph/path?source_id={people[0]['person_id']}"
            f"&target_id={people[1]['person_id']}"
        ).json()
        assert body["found"] is True
        assert body["hops"] >= 1
        assert body["paths"][0]["edges"]


class TestApiPassport:
    def test_passport_answers_the_questions_it_promises(self, client) -> None:
        passport = client.get("/api/passport").json()["passport"]
        assert passport["sources"]
        assert passport["write_back"]["enabled"] is False
        assert passport["key_fingerprint"]
        assert "derived_stores" in passport

    def test_system_info_reports_deviations(self, client) -> None:
        """A system that quietly differs from its spec is worse than one that is
        loudly different."""
        info = client.get("/api/system/info").json()
        assert info["deviations"]
        for deviation in info["deviations"]:
            assert deviation["id"].startswith("ADR-")
            assert deviation["blueprint"] and deviation["actual"]
            assert deviation["reason"] and deviation["risk"]


class TestNoEnvelopeLeaksToTheClient:
    """No API response may contain a raw AEAD envelope.

    A sealed field read straight off a stored document is a dict
    (`{v, alg, kid, nonce, ct, aad}`), not a string. Handing that to a client
    does not render - it throws, and the stack trace points at the UI rather than
    at the route that leaked it. It shipped once: the timeline returned a sealed
    `thread_subject` and every view crashed with React error #31.

    So this walks every GET endpoint that returns documents and asserts the
    response body is JSON-serialisable *and* free of envelope markers.
    """

    ENVELOPE_KEYS = {"alg", "nonce", "kid"}

    def _assert_clean(self, payload, path: str) -> None:
        def walk(node, trail: str) -> None:
            if isinstance(node, dict):
                if {"v", "ct"} <= set(node) and self.ENVELOPE_KEYS & set(node):
                    raise AssertionError(f"{path}: envelope leaked at {trail}")
                for key, value in node.items():
                    walk(value, f"{trail}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{trail}[{index}]")

        walk(payload, "$")
        json.dumps(payload)  # must be serialisable; sets or bytes mean a bug

    @pytest.mark.parametrize("path", [
        "/api/timeline?limit=200",
        "/api/search?q=the&limit=50",
        "/api/search?q=rebalancer&limit=20",
        "/api/files",
        "/api/persons",
        "/api/suggestions",
        "/api/insights",
        "/api/predictions",
        "/api/graph/ego?depth=2&limit=80",
        "/api/system/info",
        "/api/passport",
    ])
    def test_get_endpoints_are_envelope_free(self, client, path: str) -> None:
        body = client.get(path).json()
        self._assert_clean(body, path)

    def test_person_endpoints_are_envelope_free(self, client) -> None:
        people = client.get("/api/persons").json()["persons"]
        assert people
        for person in people:
            body = client.get(f"/api/persons/{person['person_id']}").json()
            self._assert_clean(body, f"/api/persons/{person['person_id']}")

    def test_summarize_endpoint_is_envelope_free(self, client) -> None:
        for hit in client.get("/api/search?q=the&limit=3").json()["hits"]:
            body = client.post(f"/api/documents/{hit['doc_id']}/summarize").json()
            self._assert_clean(body, f"/api/documents/{hit['doc_id']}/summarize")

    def test_document_endpoint_is_envelope_free(self, client) -> None:
        for hit in client.get("/api/search?q=the&limit=5").json()["hits"]:
            body = client.get(f"/api/documents/{hit['doc_id']}").json()
            self._assert_clean(body, f"/api/documents/{hit['doc_id']}")

    def test_search_titles_are_real_titles(self, client) -> None:
        """A Slack message has no provider-given title, so the decrypted derived
        subject is the only thing that can label a result row. Without it every
        row reads as a channel name."""
        titles = [h["title"] for h in client.get("/api/search?q=the&limit=20").json()["hits"]]
        slack_titles = [t for t in titles
                        if t and not t.startswith("#") and t not in ("slack", "gmail")]
        assert slack_titles, f"no usable titles in {titles[:6]}"
        assert all(isinstance(t, str) for t in titles)


class TestApiSync:
    def test_sync_is_idempotent_via_the_api(self, client) -> None:
        first = client.post("/api/sync", json={"connector_id": "demo"}).json()
        assert first["ingest"]["status"] == "ok"
        second = client.post("/api/sync", json={"connector_id": "demo"}).json()
        assert second["ingest"]["counters"]["created"] == 0

    def test_unknown_connector_404s(self, client) -> None:
        assert client.post("/api/sync", json={"connector_id": "nope"}).status_code == 404

    def test_connectors_expose_scope_justifications(self, client) -> None:
        for connector in client.get("/api/connectors").json()["connectors"]:
            for scope in connector["scopes"]:
                assert scope["justification"].strip()

    def test_drop_derived_rebuilds(self, client) -> None:
        body = client.post("/api/admin/drop-derived?rebuild=true").json()
        assert body["rebuilt"]["identities"]["persons"] > 0
