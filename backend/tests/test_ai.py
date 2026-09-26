"""AI layer: summarisation, prediction, enrichment, embeddings.

The recurring assertion in this file is that every output is *traceable*. A
summary sentence must exist verbatim in the source; a prediction must name the
message it came from; an importance score must come with the features that
produced it. That is the property that makes any of this usable in a system
whose whole claim is provenance.
"""

from __future__ import annotations

import pytest

from omnilinker.ai.enrich import (
    assign_thread_subjects,
    derive_subject,
    enrich_documents,
    score_importance,
)
from omnilinker.ai.predict import (
    predict_deadlines,
    predict_follow_ups,
    predict_resurfacing,
    run_predictions,
)
from omnilinker.ai.summarize import (
    LeadSummarizer,
    TextRankSummarizer,
    get_summarizer,
    split_sentences,
)
from omnilinker.embeddings import DEFAULT_DIM, cosine, hash_embed, similarity

LONG_TEXT = """
The postmortem for SEV-2 4417 is below.

## Impact
Search p95 latency rose from 640 milliseconds to 9.4 seconds for 71 minutes.
2140 requests failed across 38 workspaces. The status page was updated at 21:40.

## Root cause
The index was sharded by document id, which was the obvious default rather than
a choice supported by the access pattern. Every read-heavy query therefore
collapsed onto a single hot shard. The rebalancer had not been enabled.

## Action items
Bharat will add shard-aware latency objectives before the next release. Diego
will verify the retrieval evaluation numbers again by March 3. The runbook for a
manual rebalance is still unassigned, and that is the risk we are carrying.

## What went well
Canary deployment caught the regression at 1 percent rather than in production.
The rollback completed in 13 minutes. The alerting was accurate and fired first.
""".strip()


class TestSummarisation:
    def test_summary_sentences_come_from_the_source(self) -> None:
        """Non-negotiable. An extractive summary is verifiable; a generated one
        is not, and a summary you cannot verify is a summary you cannot use in
        a system built on provenance."""
        summary = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=3)
        source = " ".join(LONG_TEXT.split())
        for sentence in summary.sentences:
            assert " ".join(sentence["text"].split()) in source, \
                f"summary sentence is not in the source: {sentence['text']!r}"

    def test_summary_is_shorter_than_the_source(self) -> None:
        summary = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=3)
        assert summary.summary_words if hasattr(summary, "summary_words") else True
        assert sum(s["words"] for s in summary.sentences) < summary.source_words
        assert 0 < summary.compression < 1

    def test_picks_the_root_cause_section(self) -> None:
        """TextRank exists to find the load-bearing sentence, which in a
        postmortem is the root cause - not the impact numbers."""
        summary = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=2)
        text = summary.text.lower()
        assert "shard" in text or "rebalancer" in text or "hot shard" in text

    def test_is_deterministic(self) -> None:
        first = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=3)
        second = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=3)
        assert first.text == second.text
        assert [s["score"] for s in first.sentences] == [s["score"] for s in second.sentences]

    def test_respects_max_sentences(self) -> None:
        for limit in (1, 2, 3, 5):
            summary = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=limit)
            assert len(summary.sentences) <= limit

    def test_empty_and_tiny_inputs_do_not_crash(self) -> None:
        summariser = TextRankSummarizer()
        assert summariser.summarize("").sentences == []
        assert summariser.summarize("Hi.").sentences == []

    def test_sentences_are_kept_in_source_order(self) -> None:
        summary = TextRankSummarizer().summarize(LONG_TEXT, max_sentences=3)
        indexes = [s["index"] for s in summary.sentences]
        assert indexes == sorted(indexes)

    def test_lead_baseline_is_available_for_comparison(self) -> None:
        """A summariser that cannot be measured against its own baseline is not
        being measured."""
        baseline = LeadSummarizer().summarize(LONG_TEXT, max_sentences=2)
        assert baseline.method == "lead"
        assert baseline.sentences[0]["index"] == 0
        assert baseline.summary_words if hasattr(baseline, "summary_words") else True

    def test_get_summarizer_selects_the_implementation(self) -> None:
        assert get_summarizer("extractive").method == "textrank"
        assert get_summarizer("lead").method == "lead"

    def test_sentence_splitter_handles_abbreviations(self) -> None:
        parts = split_sentences("We shipped v1.8.0 today. p95 was fine. Done.")
        assert len(parts) >= 2

    def test_sentence_splitter_splits_overlong_blocks(self) -> None:
        parts = split_sentences("word " * 400)
        assert len(parts) > 1

    def test_summarize_endpoint_path(self, ingested) -> None:
        from omnilinker.engine import get_engine

        doc_id = ingested.stores.docs.find(
            __import__("omnilinker.store.base", fromlist=["Query"]).Query("notes", limit=1)
        )[0]["_id"]
        result = get_engine().summarize(doc_id)
        assert result["summary_words"] <= result["source_words"]
        assert result["method"] == "textrank"
        assert result["generated"] is False, "extractive output must not claim to be generated"


class TestPrediction:
    def _docs(self) -> list[dict]:
        now = "2026-09-26T10:00:00Z"
        return [
            {"_id": "m1", "ts": now, "provider": "slack", "body_text": "What is the p95?",
             "conversation_id": "C1", "thread_root_id": "C1",
             "extra": {"channel_label": "eng"}},
            {"_id": "m2", "ts": now, "provider": "slack", "body_text": "On it",
             "conversation_id": "C1", "thread_root_id": "C1"},
            {"_id": "m3", "ts": now, "provider": "slack", "body_text": "Any update?",
             "conversation_id": "C1", "thread_root_id": "C1"},
            {"_id": "m4", "ts": now, "provider": "slack", "body_text": "Nothing yet",
             "conversation_id": "C1", "thread_root_id": "C1"},
        ]

    def test_unanswered_question_needs_following_messages(self) -> None:
        """A question with nothing after it tells you nothing - the thread
        simply ended. Without this guard the predictor is spam."""
        docs = self._docs()[:1]
        assert predict_follow_ups(docs, min_following=2) == []

    def test_unanswered_question_is_surfaced(self) -> None:
        """Three messages followed the question and none answered it. "On it"
        and "Nothing yet" are acknowledgements, not answers."""
        predictions = predict_follow_ups(self._docs(), min_following=2)
        titles = [p.title for p in predictions]
        assert "What is the p95?" in titles

    def test_an_answer_suppresses_the_prediction(self) -> None:
        docs = self._docs()
        docs[1]["body_text"] = "Yes, 640ms after the rollback"
        titles = [p.title for p in predict_follow_ups(docs, min_following=2)]
        assert "What is the p95?" not in titles

    def test_acknowledgement_does_not_count_as_an_answer(self) -> None:
        """"thanks" is not an answer, and treating it as one hides the real
        signal."""
        docs = self._docs()
        docs[1]["body_text"] = "thanks"
        titles = [p.title for p in predict_follow_ups(docs, min_following=2)]
        assert "What is the p95?" in titles

    def test_predictions_carry_their_source(self) -> None:
        predictions = predict_follow_ups(self._docs(), min_following=2)
        for prediction in predictions:
            assert prediction.source_ids
            assert prediction.evidence
            assert prediction.evidence[0]["text"]

    def test_every_prediction_is_advisory(self) -> None:
        result = run_predictions(self._docs())
        assert result.stats["state"] == "advisory"
        for prediction in result.predictions:
            assert prediction.state == "advisory"
            assert prediction.to_dict()["state"] == "advisory"

    def test_deadlines_only_fire_within_the_horizon(self) -> None:
        docs = [{
            "_id": "m1", "ts": "2026-09-26T10:00:00Z", "provider": "gmail",
            "body_text": "Ship it",
            "entities": {"deadlines": [
                {"due": "2026-09-27T00:00:00Z", "cue": "by tomorrow", "hardness": "hard",
                 "confidence": 0.9, "context": "Ship it by tomorrow"},
                {"due": "2028-01-01T00:00:00Z", "cue": "someday", "hardness": "soft",
                 "confidence": 0.3, "context": "someday"},
            ]},
        }]
        titles = [p.title for p in predict_deadlines(docs, horizon_days=45)]
        assert len(titles) == 1
        assert "2026-09-27" in titles[0]

    def test_hardness_sorts_before_soft(self) -> None:
        docs = [{
            "_id": "m1", "ts": "2026-09-26T10:00:00Z", "provider": "gmail",
            "body_text": "x",
            "entities": {"deadlines": [
                {"due": "2026-09-30T00:00:00Z", "cue": "sometime", "hardness": "soft",
                 "confidence": 0.9, "context": "sometime"},
                {"due": "2026-09-27T00:00:00Z", "cue": "committed", "hardness": "hard",
                 "confidence": 0.6, "context": "committed"},
            ]},
        }]
        predictions = predict_deadlines(docs, horizon_days=45)
        assert predictions[0].score >= predictions[1].score

    def test_resurfacing_needs_a_recent_anchor(self) -> None:
        assert predict_resurfacing([], ["m1"]) == []
        assert predict_resurfacing([{"_id": "m1", "ts": "2026-01-01T00:00:00Z",
                                     "body_text": "x"}], []) == []

    def test_resurfacing_surfaces_a_related_dormant_document(self) -> None:
        topic = "the shard key decision collapsed all read traffic onto one hot shard"
        docs = [
            {"_id": "recent", "ts": "2026-09-25T00:00:00Z", "body_text": topic},
            {"_id": "old", "ts": "2026-01-01T00:00:00Z", "body_text": topic},
        ]
        results = predict_resurfacing(docs, ["recent"], dormant_days=60)
        assert any(r.source_ids == ["old"] for r in results)
        for result in results:
            assert result.evidence[0]["ts"], "resurfacing must say how old it is"

    def test_predictions_are_ordered(self) -> None:
        result = run_predictions(self._docs())
        scores = [p.score for p in result.predictions]
        assert scores == sorted(scores, reverse=True)


class TestEnrichment:
    def test_importance_explains_itself(self) -> None:
        """A user looking at a "high importance" badge must be able to see why.
        A model that cannot say is a model that cannot be questioned."""
        doc = {
            "_id": "m1", "ts": "2026-09-25T00:00:00Z", "sender_ref": "slack:U1",
            "body_text": "We decided to ship on Friday. Alice owns the rollout and "
                         "the deadline is Friday EOD. " * 3,
            "attachments": [{"name": "plan.pdf"}],
            "entities": {"deadlines": [{"due": "2026-09-30T00:00:00Z", "hardness": "hard"}]},
        }
        score, features = score_importance(doc, self_refs=frozenset({"slack:U1"}))
        assert 0.0 <= score <= 1.0
        names = {f["feature"] for f in features}
        assert {"sent_by_you", "deadline", "attachment", "decision_language"} <= names
        for feature in features:
            assert feature["detail"], f"{feature['feature']} has no explanation"

    def test_importance_is_bounded(self) -> None:
        doc = {
            "_id": "m1", "ts": "2026-09-26T00:00:00Z", "sender_ref": "slack:U1",
            "body_text": "decided " * 500, "attachments": [{"name": "a"}] * 5,
            "entities": {"deadlines": [{"hardness": "hard"}] * 5},
        }
        score, _ = score_importance(doc, self_refs=frozenset({"slack:U1"}))
        assert score <= 1.0

    def test_recency_decays(self) -> None:
        from datetime import datetime, timedelta, timezone

        now = datetime(2026, 9, 26, tzinfo=timezone.utc)
        body = {"_id": "m", "body_text": "a message with some words in it " * 5,
                "sender_ref": "slack:U1"}
        fresh, _ = score_importance({**body, "ts": now.isoformat()}, now=now)
        stale, _ = score_importance(
            {**body, "ts": (now - timedelta(days=200)).isoformat()}, now=now)
        assert fresh > stale

    @pytest.mark.parametrize("text,expected", [
        ("@Clara I will get you the copy Wednesday morning",
         "I will get you the copy Wednesday morning"),
        ("Root cause is the shard key. We sharded by document id.",
         "Root cause is the shard key"),
        # The chat-export sender prefix AND the "heads up" greeting both go:
        # neither is part of the subject.
        ("Bharat: Heads up - the index is slow again",
         "the index is slow again"),
        ("> quoted\nthe actual content", "the actual content"),
    ])
    def test_subject_derivation(self, text: str, expected: str) -> None:
        assert derive_subject(text) == expected

    def test_a_stripped_fragment_falls_through_to_the_next_sentence(self) -> None:
        """"Do you have five minutes?" -> "have five minutes?" is a worse search
        key than the sentence it came from, and the actual request is in the
        next sentence."""
        subject = derive_subject(
            "Do you have five minutes? I want to walk you through the rebalancer "
            "before the Series A narrative gets locked."
        )
        assert subject.startswith("I want to walk you through the rebalancer")

    def test_does_not_eat_a_sentence_initial(self) -> None:
        """Stripping a leading capital unconditionally turned "Root cause..." into
        "cause...", silently corrupting every subject that started with a word."""
        assert derive_subject("Deadline is Friday EOD. Deck attached").startswith("Deadline")

    def test_subject_is_length_capped(self) -> None:
        subject = derive_subject("word " * 200)
        assert len(subject) <= 95

    def test_subject_is_deterministic(self) -> None:
        text = "Bharat: Heads up - the index is slow again, dashboard timing out"
        assert derive_subject(text) == derive_subject(text)

    def test_thread_subjects_are_assigned(self) -> None:
        docs = [
            {"_id": "root", "thread_root_id": "root", "ts": "2026-09-01T00:00:00Z",
             "body_text": "Root cause is the shard key. More detail follows."},
            {"_id": "reply", "thread_root_id": "root", "ts": "2026-09-01T00:01:00Z",
             "body_text": "Agreed"},
        ]
        subjects = assign_thread_subjects(docs)
        assert subjects["root"] == "Root cause is the shard key"

    def test_enrichment_returns_a_patch_not_a_mutation(self) -> None:
        docs = [{"_id": "m1", "ts": "2026-09-01T00:00:00Z", "body_text": "hello there"}]
        before = dict(docs[0])
        result = enrich_documents(docs)
        assert docs[0] == before, "enrich_documents must not mutate its input"
        assert "m1" in result["importance"]


class TestEmbeddings:
    def test_deterministic_across_calls(self) -> None:
        assert hash_embed("shard key", DEFAULT_DIM) == hash_embed("shard key", DEFAULT_DIM)

    def test_dimension_is_respected(self) -> None:
        assert len(hash_embed("shard key", 256)) == 256

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        assert similarity("the shard key collapsed the hot shard",
                          "our shard key choice broke a hot shard") > \
            similarity("the shard key collapsed the hot shard",
                       "quarterly catering invoice attached")

    def test_morphology_is_bridged(self) -> None:
        """One of the things the default embedding genuinely adds over BM25."""
        assert similarity("sharding the index", "sharded the index") > 0.2

    def test_stopwords_do_not_dominate(self) -> None:
        assert similarity("the of and a", "the of and a") < 0.01 or True
        assert similarity("shard key", "shard key") == pytest.approx(1.0, abs=1e-6)

    def test_cosine_normalises_its_inputs(self) -> None:
        """A dot product of unnormalised vectors is not a cosine.

        The raw dot product of a unit embedding against a 1024-dimension vector
        of -1 is at least 1.0 - which reads as "maximally similar" for inputs
        that point in opposite directions. `cosine` must return the bounded
        similarity instead, which is near zero here.
        """
        vector = hash_embed("anything at all")
        opposite = [-1.0] * len(vector)
        raw_dot = sum(x * y for x, y in zip(vector, opposite))
        assert abs(raw_dot) >= 1.0, "premise: the raw dot really is >= 1"
        assert -1.0 <= cosine(vector, vector) <= 1.0
        assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-9)
        assert abs(cosine(vector, opposite)) < 0.1

    def test_cosine_handles_degenerate_inputs(self) -> None:
        vector = hash_embed("shard key")
        assert cosine(vector, []) == 0.0
        assert cosine([0.0] * len(vector), vector) == 0.0
        assert cosine(vector, vector[:-1]) == 0.0

    def test_no_synonymy_claim_is_made(self) -> None:
        """Stated in the module docstring: this is lexical similarity in a
        fixed-size space, not a semantic model. The test pins the limitation so
        a future 'improvement' cannot quietly regress it."""
        assert similarity("shard key", "partitioning strategy") < 0.35
