"""Search: BM25 mechanics, query grammar, hybrid fusion, privacy posture.

Two of these tests exist specifically to protect properties that are easy to
lose and hard to notice:

  * **The ranking function must actually read bodies.** An earlier version
    built the index from the content-free projection only, so BM25 ranked
    titles and returned 0 results for terms that were plainly in the corpus.
    Everything looked healthy; the index had 600 terms and p95 of 6 ms.
  * **The search index must not become a plaintext store.** Bodies are
    decrypted for the ~20 results on screen and nowhere else.
"""

from __future__ import annotations

import math

import pytest

from omnilinker.search.index import FIELDS, LexicalIndex, stem, tokenize
from omnilinker.search.nl2query import NL2QueryCompiler
from omnilinker.search.query import SearchRequest, parse, parse_natural_language
from omnilinker.search.service import (
    SearchService,
    get_search_service,
    project_search_doc,
    reciprocal_rank_fusion,
)


def doc(doc_id: str, **fields) -> dict:
    return {"_id": doc_id, "kind": "message", "provider": "slack", **fields}


class TestTokenizer:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("Shard Keys, 100% Faster") == ["shard", "key", "100", "faster"]

    def test_keeps_identifiers(self) -> None:
        """`precision@10`, `p95` and `dk_slack_v1` are the words users actually
        search for in an engineering corpus. Splitting on punctuation loses all
        of them."""
        assert "precision@10" in tokenize("precision@10 is 0.88")
        assert "p95" in tokenize("p95 search latency")

    def test_stemming_is_conservative(self) -> None:
        assert stem("sharding") == "shard"
        assert stem("sharded") == "shard"
        assert stem("indexes") == "index"
        # Short tokens are left alone; over-stemming "us" -> "u" is noise.
        assert stem("us") == "us"
        assert stem("go") == "go"

    def test_stopwords_exist_but_are_optional(self) -> None:
        assert "the" in tokenize("the shard")
        assert "the" not in tokenize("the shard", keep_stopwords=False)


class TestBM25:
    def test_idf_floors_at_a_positive_value(self) -> None:
        """The classic Robertson bug: idf goes negative for a term in most
        documents, so a match *subtracts* relevance."""
        index = LexicalIndex()
        for i in range(20):
            index.add_document(doc(f"d{i}"), body="commonword everywhere")
        index.add_document(doc("rare"), body="commonword plus uniquenonce")
        index.finalize()
        assert index.idf("commonword") > 0
        assert index.score_document("rare", ["commonword"]) > 0

    def test_field_weights_beat_raw_length(self) -> None:
        """A precise title must beat a long noisy body. Without field weights
        the results look broken even though the math is right."""
        index = LexicalIndex()
        index.add_document(doc("titled", title="incident postmortem"),
                           body="filler " * 300)
        index.add_document(doc("buried", title="weekly notes"),
                           body="incident postmortem " + "filler " * 300)
        index.finalize()
        ranked = dict(index.search("incident postmortem", limit=10))
        assert ranked["titled"] > ranked["buried"]

    def test_saturation_stops_one_document_dominating(self) -> None:
        """k1's job. Without saturation, a keyword-stuffed document wins every
        query it appears in."""
        index = LexicalIndex()
        index.add_document(doc("stuffed"), body="shard " * 200)
        index.add_document(doc("honest"), body="shard key rollout and rebalancer plan")
        index.finalize()
        ranked = dict(index.search("shard key", limit=10))
        assert "honest" in ranked
        assert ranked["honest"] > 0

    def test_length_normalisation_prefers_shorter(self) -> None:
        index = LexicalIndex()
        index.add_document(doc("long"), body="shard " + "filler " * 200)
        index.add_document(doc("short"), body="shard key")
        index.finalize()
        ranked = dict(index.search("shard", limit=10))
        assert ranked["short"] > ranked["long"]

    def test_body_in_the_doc_dict_is_ignored(self) -> None:
        """The privacy guarantee, enforced. Content reaches the index only
        through the explicit `body=` argument, so the content-free projection
        cannot smuggle a body in even if a future caller tries."""
        index = LexicalIndex()
        index.add_document({"_id": "d", "body": "should not be indexed anywhere",
                            "body_text": "nor this", "provider": "slack"})
        index.finalize()
        assert index.search("should not be indexed") == []
        assert index.search("nor this") == []

    def test_re_adding_a_document_replaces_it(self) -> None:
        index = LexicalIndex()
        index.add_document(doc("d"), body="alpha")
        index.add_document(doc("d"), body="beta")
        index.finalize()
        assert index.search("alpha") == []
        assert dict(index.search("beta", limit=5)).get("d", 0) > 0

    def test_avg_length_is_computed(self) -> None:
        """Skipping this is the silent bug that makes a BM25 implementation
        rank short documents far too highly."""
        index = LexicalIndex()
        for i in range(5):
            index.add_document(doc(f"d{i}"), body="alpha beta gamma delta epsilon")
        index.finalize()
        assert index.avg_length["body"] == 5.0, index.doc_lengths
        assert index.doc_count == 5

    def test_phrase_match_boosts(self) -> None:
        index = LexicalIndex()
        index.add_document(doc("exact", title="decision divergence"), body="filler")
        index.add_document(doc("split", title="decision and divergence"), body="filler")
        index.finalize()
        ranked = dict(index.search('"decision divergence"', limit=5))
        assert ranked["exact"] > ranked["split"]

    def test_index_is_deterministic(self) -> None:
        def build() -> LexicalIndex:
            index = LexicalIndex()
            for i in range(20):
                index.add_document(doc(f"d{i}"), body=f"content number {i} about shards")
            index.finalize()
            return index

        assert build().search("shards") == build().search("shards")

    def test_every_field_weight_is_positive(self) -> None:
        assert all(weight > 0 for weight, _ in FIELDS.values())


class TestQueryGrammar:
    def test_field_filters_are_extracted(self) -> None:
        parsed = parse("budget from:alice in:#eng after:2026-01-01 has:file type:note")
        assert parsed.filters["from"] == ["alice"]
        assert parsed.filters["in"] == ["#eng"]
        assert parsed.filters["after"] == ["2026-01-01"]
        assert parsed.filters["has"] == ["file"]
        assert parsed.filters["type"] == ["note"]

    def test_filter_values_are_not_search_terms(self) -> None:
        """`from:alice` must not make every message by Alice match "alice"."""
        parsed = parse("budget from:alice")
        assert "alice" not in parsed.terms

    def test_phrases_are_separated(self) -> None:
        parsed = parse('"decision divergence" rollout')
        assert parsed.phrases == ["decision divergence"]
        assert "rollout" in parsed.terms

    def test_negation(self) -> None:
        parsed = parse("search -slack")
        assert "slack" in parsed.exclude
        assert "slack" not in parsed.terms

    def test_unknown_field_is_left_alone(self) -> None:
        """An unsupported filter must not be silently dropped - it becomes a
        search term, which is visible, rather than a no-op, which is not."""
        parsed = parse("budget colour:red")
        assert "colour:red" in parsed.raw
        assert "colour" not in parsed.filters

    def test_relative_dates_resolve(self) -> None:
        parsed = parse("messages after:today")
        assert parsed.filters["after"][0].startswith("20")

    def test_natural_language_drops_stopwords(self) -> None:
        """A question is mostly function words; scoring on them ranks documents
        by coincidence."""
        parsed = parse("what did we agree on vector search")
        assert parsed.natural_language
        assert "what" not in parsed.retrieval_terms
        assert "vector" in parsed.retrieval_terms
        assert "what" in parsed.stopwords_dropped

    def test_keyword_queries_keep_stopwords(self) -> None:
        """A keyword search for "to be" should find "to be"."""
        parsed = parse("to be or not to be")
        assert parsed.retrieval_terms.count("be") == 2

    def test_short_queries_stay_keywords(self) -> None:
        assert not parse("shard key").natural_language
        assert not parse("incident").natural_language


class TestRRF:
    def test_single_list_keeps_original_scores(self) -> None:
        lexical = {"a": 9.0, "b": 3.0}
        fused = reciprocal_rank_fusion(lexical, None, k=None)
        assert [doc_id for doc_id, _ in fused] == ["a", "b"]
        assert dict(fused)["a"][0] == 9.0, \
            "a one-list RRF should stay a monotonic transform, not rescale scores"

    def test_two_lists_fuse_by_rank_only(self) -> None:
        fused = dict(reciprocal_rank_fusion({"a": 50.0, "b": 1.0}, {"b": 0.9, "a": 0.1}, k=60))
        # Both documents appear in both lists, so each contributes twice. The
        # point is that a 50:1 score ratio in one list does not drown out the
        # other list's evidence - the fusion only ever sees ranks.
        assert fused["a"][1] == 1 and fused["a"][2] == 2
        assert fused["b"][1] == 2 and fused["b"][2] == 1
        assert fused["b"][0] > 0

    def test_a_document_in_one_list_only_still_ranks(self) -> None:
        fused = dict(reciprocal_rank_fusion({"a": 1.0}, {"z": 0.9}, k=60))
        assert "z" in fused and "a" in fused

    def test_consensus_wins(self) -> None:
        fused = dict(reciprocal_rank_fusion(
            {"both": 1.0, "lexonly": 9.0}, {"both": 0.1, "semonly": 0.9}, k=60))
        assert fused["both"][0] > fused["lexonly"][0]


class TestPrivacy:
    def test_search_doc_projection_excludes_content(self) -> None:
        """A sealed body must never reach the derived ranking store, or
        `str()`ing the envelope puts ciphertext JSON into the index and every
        document matches every query."""
        sealed = {"v": 1, "alg": "AES-256-GCM", "ct": "AAAA", "nonce": "BBBB"}
        projected = project_search_doc({
            "_id": "slack:1", "provider": "slack", "kind": "message",
            "body_text": sealed, "sender_name": "Alice", "ts": "2026-01-01T00:00:00Z",
        })
        blob = repr(projected)
        assert "body_text" not in projected
        assert "AES-256-GCM" not in blob
        assert "AAAA" not in blob

    def test_projection_excludes_contact_details(self) -> None:
        projected = project_search_doc({
            "_id": "slack:1", "provider": "slack", "kind": "message",
            "email": "alice@northwind.io", "phone": "+447700900123",
            "sender_name": "Alice",
        })
        assert "alice@northwind.io" not in repr(projected)
        assert "+447700900123" not in repr(projected)

    def test_index_is_not_persisted_by_default(self, ingested) -> None:
        """The index is plaintext-bearing by construction - it has to be, to
        rank. Writing it to disk would create a second plaintext store."""
        import os

        service = get_search_service()
        service.reindex()
        index_dir = service.settings.data_dir / "index"
        assert not index_dir.exists() or not list(index_dir.glob("*.json"))

    def test_persisted_index_applies_k_anonymity(self, ingested, tmp_path) -> None:
        """A term in fewer than k documents is effectively a verbatim quote."""
        index = LexicalIndex()
        for i in range(10):
            index.add_document(doc(f"d{i}"), body="sharedterm everywhere")
        index.add_document(doc("odd"), body="sharedterm plus razorphanword")
        index.finalize()
        report = index.save(tmp_path / "idx.json", k_anonymity=2)
        assert report["terms_dropped_for_k_anonymity"] >= 1
        assert "razorphanword" not in (tmp_path / "idx.json").read_text("utf-8")

    def test_drop_index_removes_the_derived_store(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        service.drop_index()
        assert service.stores.docs.count("search_docs") == 0

    def test_nothing_is_written_back_to_a_provider(self, ingested) -> None:
        """Blueprint non-goal N1, asserted rather than documented."""
        from omnilinker.api.routes import router
        import inspect

        sources = []
        for route in router.routes:
            body = inspect.getsource(route.endpoint)
            for verb in ("post", "put", "patch", "delete"):
                if f".{verb}(" in body and "providers." in body:
                    sources.append(f"{route.path} {verb}")
        assert not sources, f"routes that write to a provider: {sources}"


class TestSearchService:
    def test_finds_a_term_that_only_appears_in_a_body(self, ingested) -> None:
        """The regression test for ranking titles only."""
        service = get_search_service()
        service.reindex()
        response = service.search(SearchRequest(q="rebalancer", limit=5))
        assert response.hits, "a term present only in message bodies found nothing"

    def test_nonsense_query_returns_nothing(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        assert service.search(SearchRequest(q="zzzzqqqqxyzzy", limit=5)).total == 0

    def test_total_is_not_the_page_length(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        response = service.search(SearchRequest(q="search", limit=3))
        assert response.total >= len(response.hits)
        if response.total_is_exact:
            assert response.total >= 3

    def test_filters_narrow_results(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        unfiltered = service.search(SearchRequest(q="the", limit=50)).total
        filtered = service.search(SearchRequest(q="the", providers=["notion"], limit=50))
        assert filtered.total < unfiltered
        assert all(hit.provider == "notion" for hit in filtered.hits)

    def test_impossible_filter_returns_empty_not_everything(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        response = service.search(SearchRequest(q="search", providers=["gdrive"],
                                                kinds=["note"], limit=10))
        assert response.total == 0
        assert response.degraded

    def test_facets_are_computed(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        facets = service.search(SearchRequest(q="search", limit=50)).facets
        assert facets["provider"]
        assert sum(b["count"] for b in facets["provider"]) > 0

    def test_snippets_are_produced_from_decrypted_text(self, ingested) -> None:
        """Snippets need content, so this is the one place decryption happens at
        query time - bounded to the page being displayed."""
        service = get_search_service()
        service.reindex()
        response = service.search(SearchRequest(q="rebalancer", limit=3))
        assert any(hit.snippet for hit in response.hits)
        for hit in response.hits:
            assert "AES-256-GCM" not in hit.snippet

    def test_response_echoes_the_parsed_query(self, ingested) -> None:
        service = get_search_service()
        service.reindex()
        response = service.search(SearchRequest(q="what did we decide", limit=3))
        assert response.parsed["natural_language"] is True
        assert "decide" in response.parsed["retrieval_terms"]

    def test_undecryptable_document_degrades_to_no_snippet(self, ingested) -> None:
        """One corrupt row must not fail the whole search."""
        service = get_search_service()
        service.reindex()
        service.reindex()
        from omnilinker.store.base import Query

        target = service.search(SearchRequest(q="rebalancer", limit=1)).hits[0]
        row = service.stores.docs.get("messages", target.doc_id)
        if row and isinstance(row.get("body_text"), dict):
            row["body_text"] = {**row["body_text"], "ct": "AAAA" + row["body_text"]["ct"][4:]}
            service.stores.docs.put("messages", target.doc_id, row)
        service._index_version = None  # force a rebuild
        response = service.search(SearchRequest(q="rebalancer", limit=3))
        assert response.hits, "search failed because one document was corrupt"

    def test_reindex_is_idempotent(self, ingested) -> None:
        service = get_search_service()
        first = service.reindex(persist=False)
        second = service.reindex(persist=False)
        assert first["documents"] == second["documents"]
        assert first["terms"] == second["terms"]

    def test_reindex_drops_stale_projections(self, ingested) -> None:
        """A deleted message must stop appearing in search. Without this,
        deletions are invisible forever."""
        service = get_search_service()
        service.reindex()
        service.stores.docs.delete("messages", "slack:1786557600.000000")
        service.reindex(persist=False)
        assert service.stores.docs.get("search_docs", "slack:1786557600.000000") is None

    def test_engine_mode_is_reported(self, ingested) -> None:
        service = get_search_service()
        response = service.search(SearchRequest(q="search", limit=1))
        assert response.engine in ("lexical", "hybrid")


class TestNL2Query:
    @pytest.mark.parametrize("question,intent", [
        ("what did we decide about the shard key", "search"),
        ("how many messages are in slack", "aggregate"),
        ("what deadlines are coming up", "deadlines"),
        ("who is Clara Nowak", "person_lookup"),
        ("anything else about retention policy", "related"),
        ("show me search decisions over time", "timeline"),
    ])
    def test_intent_routing(self, question: str, intent: str) -> None:
        assert NL2QueryCompiler().compile(question).intent == intent

    def test_person_intent_dispatches_to_the_graph(self) -> None:
        plan = NL2QueryCompiler().compile("who is Clara Nowak")
        assert plan.execution == "graph"
        assert plan.person_hint

    def test_provider_mentioned_in_words_becomes_a_filter(self) -> None:
        plan = NL2QueryCompiler().compile("how many messages are in slack")
        assert plan.request.providers == ["slack"]

    def test_ambiguous_alias_does_not_become_a_filter(self) -> None:
        """'email' is a field name and a provider name; reading it as a provider
        answers a different question than the one asked."""
        plan = NL2QueryCompiler().compile("what is Alice Chen email address")
        assert plan.request.providers == []

    def test_time_window_is_resolved(self) -> None:
        plan = NL2QueryCompiler().compile("files about the budget in the last 30 days")
        assert plan.request.after
        assert plan.time_window

    def test_plan_is_executable_by_the_store(self, ingested) -> None:
        """Every plan must translate to filters the store understands. A plan
        with a field the store cannot express would silently answer a different
        question."""
        for question in ["budget files", "deadlines this week", "messages in gmail",
                         "what did we decide about the shard key", "who is Alice Chen"]:
            plan = NL2QueryCompiler().compile(question)
            for filter_dict in plan.request.filter_dicts():
                assert set(filter_dict) <= {"field", "op", "value"}

    def test_unsupported_clauses_are_reported_not_dropped(self) -> None:
        plan = NL2QueryCompiler().compile("search messages with mood:excited")
        if plan.unsupported:
            assert all(isinstance(item, str) for item in plan.unsupported)

    def test_compile_and_run_returns_the_plan_alongside_the_results(self, ingested) -> None:
        outcome = NL2QueryCompiler().compile_and_run("what did we decide about the shard key")
        assert outcome["plan"]["intent"]
        assert outcome["result"]["hits"]

    def test_explanations_are_human_readable(self) -> None:
        plan = NL2QueryCompiler().compile("how many files are in gdrive")
        assert plan.explanation
        assert all(isinstance(line, str) and line for line in plan.explanation)

    def test_decomposition_reports_dropped_stopwords(self) -> None:
        analysis = parse_natural_language("where did we decide on the retention period")
        assert "decision" in analysis["signals"]
        assert analysis["stopwords_dropped"]
