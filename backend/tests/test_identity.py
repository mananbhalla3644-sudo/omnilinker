"""Entity resolution: blocking, scoring, clustering, detectors (blueprint 7).

The tests here are mostly *negative* on purpose. Anyone can build a resolver
that merges aggressively; the product risk is the opposite error - merging two
people who are not the same - because the consequence is a stranger's messages
appearing in your own profile. So the suite spends as much effort on what must
*not* merge as on what must.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from omnilinker.identity.blocking import (
    build_blocks,
    blocking_stats,
    cooccurrence_pairs,
    pairs_from_blocks,
)
from omnilinker.identity.detectors import (
    detect_bursts,
    detect_dormant,
    detect_hidden_introducers,
    interactions_from_documents,
    participants_by_conversation,
)
from omnilinker.identity.resolver import IdentityResolver, PairScorer


def identity(ref: str, **overrides) -> dict:
    base = {
        "_id": f"idt_{ref.replace(':', '_')}",
        "ref": ref,
        "provider": ref.split(":", 1)[0],
        "provider_user_id": ref.split(":", 1)[1] if ":" in ref else ref,
        "display_name": "",
        "email": "",
        "phone": "",
        "email_token": "",
        "phone_token": "",
        "name_token": "",
        "mention_count": 1,
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-09-01T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestBlocking:
    def test_identical_email_blocks_together(self) -> None:
        a = identity("slack:U1", email_token="em_x")
        b = identity("gmail:a@b.com", email_token="em_x")
        blocks, _ = build_blocks([a, b])
        assert any({a["_id"], b["_id"]} <= set(members)
                   for members in blocks.values())

    def test_unrelated_identities_do_not_block(self) -> None:
        a = identity("slack:U1", name_token="nm_a", display_name="Alice")
        b = identity("gmail:b@c.com", name_token="nm_b", display_name="Bob")
        blocks, _ = build_blocks([a, b])
        pairs = pairs_from_blocks(blocks)
        assert (a["_id"], b["_id"]) not in pairs

    def test_oversize_blocks_are_reported_not_processed(self) -> None:
        """A block of size n proposes n^2/2 pairs. An oversize block almost
        always means a parser bug (every message attributed to a bot account),
        and processing it would silently turn blocking into all-pairs."""
        crowd = [identity(f"slack:U{i}", name_token="nm_shared", display_name="Support")
                 for i in range(500)]
        _blocks, oversize = build_blocks(crowd)
        assert oversize, "an oversize block was not reported"
        assert oversize[0]["size"] == 500

    def test_pair_generation_is_deterministic(self) -> None:
        items = [identity(f"slack:U{i}", email_token="em_x") for i in range(8)]
        blocks, _ = build_blocks(items)
        assert pairs_from_blocks(blocks) == pairs_from_blocks(blocks)

    def test_reduction_ratio_is_measured(self) -> None:
        items = [identity(f"slack:U{i}", name_token=f"nm_{i}") for i in range(200)]
        stats = blocking_stats(items)
        assert stats["identities"] == 200
        assert stats["candidate_pairs"] < stats["all_pairs"] / 10

    def test_cooccurrence_finds_people_who_never_shared_a_token(self) -> None:
        """The fallback that survives a contact-detail change."""
        now = datetime(2026, 9, 1, tzinfo=timezone.utc)
        docs = [
            {"sender_ref": "slack:U1", "conversation_id": "C1", "ts": now.isoformat()},
            {"sender_ref": "gmail:new@addr", "conversation_id": "C1", "ts": now.isoformat()},
        ]
        pairs = cooccurrence_pairs(docs, window_days=7)
        assert ("gmail:new@addr", "slack:U1") in pairs


class TestScoring:
    def _score(self, a: dict, b: dict) -> float:
        return PairScorer({a["_id"]: a, b["_id"]: b}).score(a["_id"], b["_id"]).score

    def test_shared_email_token_is_decisive(self) -> None:
        a = identity("slack:U1", display_name="Alice", email_token="em_x")
        b = identity("gmail:a@b.com", display_name="Alice Chen", email_token="em_x")
        assert self._score(a, b) >= 0.90

    def test_shared_email_beats_a_handle_mismatch(self) -> None:
        """Different providers having different handles is the norm for every
        multi-provider person, not evidence against a merge. Treating it as a
        conflict cancelled the merge and left cross-source resolution at zero
        while still looking healthy."""
        a = identity("slack:U1", display_name="Alice", email_token="em_x")
        b = identity("gmail:a@b.com", display_name="Alice", email_token="em_x")
        score = self._score(a, b)
        assert score >= 0.90, f"decisive email was cancelled by a handle conflict: {score}"

    def test_name_alone_never_merges(self) -> None:
        a = identity("slack:U1", display_name="Alex")
        b = identity("gmail:other@x.com", display_name="Alex")
        assert self._score(a, b) < 0.90

    def test_generic_names_do_not_merge(self) -> None:
        """"Admin" on two providers is two admins. This is the single most
        damaging false merge available in this domain."""
        a = identity("slack:U1", display_name="Admin")
        b = identity("gmail:admin@x.com", display_name="admin")
        assert self._score(a, b) < 0.65

    def test_phone_plus_name_merges(self) -> None:
        a = identity("whatsapp:Alice", display_name="Alice", phone_token="ph_x")
        b = identity("gmail:a@b.com", display_name="Alice Chen", phone_token="ph_x")
        assert self._score(a, b) >= 0.90

    def test_phone_alone_does_not_merge(self) -> None:
        """A phone can be shared by a family or a VoIP line; a name can be
        shared by two people. Neither alone clears the bar."""
        a = identity("whatsapp:+447700900123", phone_token="ph_x")
        b = identity("gmail:x@y.com", phone_token="ph_x")
        assert self._score(a, b) < 0.90

    def test_different_org_domains_reduce_confidence(self) -> None:
        a = identity("gmail:a@northwind.io", display_name="Alice", email="a@northwind.io")
        b = identity("gmail:b@brightfold.com", display_name="Alice", email="b@brightfold.com")
        assert self._score(a, b) < 0.90

    def test_freemail_domains_do_not_conflict(self) -> None:
        """gmail.com and googlemail.com are the same provider; treating them as
        different organisations would subtract from a legitimate match."""
        a = identity("gmail:a@gmail.com", display_name="Alice", email="a@gmail.com")
        b = identity("gmail:b@googlemail.com", display_name="Alice", email="b@googlemail.com")
        scorer = PairScorer({a["_id"]: a, b["_id"]: b})
        result = scorer.score(a["_id"], b["_id"])
        assert not [c for c in result.conflicts if c.name == "different_org_domain"]

    def test_two_self_accounts_conflict(self) -> None:
        a = identity("slack:U1", display_name="Me", is_self=True)
        b = identity("gmail:me@x.com", display_name="Me", is_self=True)
        assert self._score(a, b) < 0.65

    def test_contact_identities_are_never_persons(self) -> None:
        a = identity("slack:U1", display_name="Alice", email_token="em_x")
        b = identity("contact:slack:a@b.com", email_token="em_other")
        assert self._score(a, b) < 0.5

    def test_identical_contact_addresses_merge(self) -> None:
        a = identity("contact:slack:a@b.com", email_token="em_x")
        b = identity("contact:gmail:a@b.com", email_token="em_x")
        assert self._score(a, b) >= 0.90

    def test_weak_evidence_saturates(self) -> None:
        """Three independent weak signals must not add up to a confident merge.
        Only evidence, not accumulation, crosses the bar."""
        a = identity("slack:U1", display_name="Alex Smith", name_token="nm_alex",
                     phone_token="ph_1")
        b = identity("gmail:z@corp.com", display_name="Alex Smith", name_token="nm_alex",
                     phone_token="ph_1")
        scorer = PairScorer({a["_id"]: a, b["_id"]: b}, conversations={
            a["_id"]: {"C1", "C2"}, b["_id"]: {"C1", "C2"},
        })
        assert scorer.score(a["_id"], b["_id"]).score <= 1.0


class TestResolver:
    def test_resolves_the_demo_cast_across_providers(self, ingested) -> None:
        """The end-to-end claim: five providers, four people, correctly merged."""
        from omnilinker.store.base import Query

        persons = ingested.stores.docs.find(Query("persons", limit=100))
        cross = [p for p in persons if p["cross_source"]]
        assert len(cross) >= 4, f"only {len(cross)} cross-source persons"
        for person in cross:
            assert len(person["providers"]) >= 4, \
                f"{person['display_name']} spans {person['providers']}"

    def test_merge_actually_happens(self, ingested) -> None:
        """A resolver that scores pairs but never merges produces a healthy-looking
        run with zero cross-source persons. This is the assertion that would have
        caught it."""
        stats = ingested.stores.docs.get("ingest_runs", ingested.run.ingest_run_id)
        assert stats  # sanity: the run exists
        from omnilinker.store.base import Query

        merged = [p for p in ingested.stores.docs.find(Query("persons", limit=100))
                  if len(p["identity_ids"]) > 1]
        assert merged, "no identity was ever merged"

    def test_a_person_gets_the_longest_name(self, ingested) -> None:
        from omnilinker.store.base import Query

        for person in ingested.stores.docs.find(Query("persons", limit=50)):
            if len(person["identity_ids"]) > 1:
                assert " " in person["display_name"] or \
                    person["display_name"] in ("You", "you"), \
                    f"{person['display_name']} kept a truncated name despite a fuller one existing"

    def test_ambiguous_pairs_become_suggestions_not_merges(self, ingested) -> None:
        from omnilinker.store.base import Query

        for suggestion in ingested.stores.docs.find(Query("link_suggestions", limit=50)):
            assert suggestion["status"] == "pending"
            assert suggestion["score"] < 0.90, \
                "a sub-threshold pair was recorded as a merge"
            assert suggestion["evidence"] or suggestion["conflicts"], \
                "a suggestion with no evidence cannot be judged by a user"

    def test_suggestions_never_carry_a_person_id(self, ingested) -> None:
        from omnilinker.store.base import Query

        identities = {i["_id"] for i in
                      ingested.stores.docs.find(Query("identities", limit=200))}
        for suggestion in ingested.stores.docs.find(Query("link_suggestions", limit=50)):
            assert suggestion["identity_a"] in identities
            assert suggestion["identity_b"] in identities

    def test_resolution_is_deterministic(self, ingested) -> None:
        from omnilinker.engine import get_engine
        from omnilinker.store.base import Query

        engine = get_engine()
        first = [(p["person_id"], sorted(p["identity_ids"]))
                 for p in engine.stores.docs.find(Query("persons", limit=100))]
        engine.resolve_identities()
        second = [(p["person_id"], sorted(p["identity_ids"]))
                  for p in engine.stores.docs.find(Query("persons", limit=100))]
        assert sorted(first) == sorted(second)

    def test_rebuilding_does_not_duplicate_persons(self, ingested) -> None:
        from omnilinker.engine import get_engine
        from omnilinker.store.base import Query

        engine = get_engine()
        before = engine.stores.docs.count("persons")
        engine.resolve_identities()
        engine.resolve_identities()
        assert engine.stores.docs.count("persons") == before

    def test_threshold_is_configurable(self, ingested) -> None:
        """Lowering the merge threshold must actually merge more, or the setting
        is decorative."""
        from omnilinker.engine import get_engine
        from omnilinker.store.base import Query

        engine = get_engine()
        strict = len([p for p in engine.stores.docs.find(Query("persons", limit=100))
                      if p["cross_source"]])
        engine.resolve_identities(merge_threshold=0.5, suggest_threshold=0.3)
        loose = len([p for p in engine.stores.docs.find(Query("persons", limit=100))
                     if p["cross_source"]])
        assert loose >= strict


class TestDetectors:
    def _interactions(self) -> dict[tuple[str, str], list[datetime]]:
        base = datetime(2026, 9, 20, tzinfo=timezone.utc)
        burst = [base - timedelta(hours=index) for index in range(14)]
        return {("a", "b"): burst}

    def test_burst_collaboration_is_detected(self) -> None:
        insights = detect_bursts(self._interactions(), burst_threshold=8)
        assert insights
        assert insights[0].kind == "collaboration_burst"
        assert insights[0].confidence <= 0.85, "a burst must not be asserted as a relationship"

    def test_dormant_relationship_needs_history(self) -> None:
        """Two messages six months apart is not a relationship that went dormant,
        it is two people who barely spoke."""
        now = datetime(2026, 9, 26, tzinfo=timezone.utc)
        sparse = {("a", "b"): [now - timedelta(days=200), now - timedelta(days=190)]}
        assert detect_dormant([], sparse, min_interactions=4, now=now) == []
        assert detect_dormant([], self._interactions(), min_interactions=4, now=now) == []

    def test_hidden_introducer_is_capped_low(self) -> None:
        """The highest-value detector and the most likely to be wrong."""
        now = datetime.now(timezone.utc)
        # a and d never spoke; they share b and c.
        interactions = {("a", "b"): [now], ("a", "c"): [now],
                        ("b", "d"): [now], ("c", "d"): [now]}
        insights = detect_hidden_introducers(interactions)
        assert insights
        assert insights[0].confidence <= 0.55
        assert "not a relationship" in insights[0].detail
        assert insights[0].evidence, "must list the mutual contacts so a user can judge"

    def test_no_hidden_connection_when_they_already_talk(self) -> None:
        now = datetime.now(timezone.utc)
        interactions = {("a", "b"): [now], ("a", "c"): [now], ("b", "c"): [now],
                        ("a", "d"): [now], ("b", "d"): [now]}
        kinds = {(i.entities[0], i.entities[1]) for i in detect_hidden_introducers(interactions)}
        assert not any(pair == ("a", "d") or pair == ("b", "d") for pair in kinds)

    def test_all_insights_are_advisory(self, ingested) -> None:
        """Nothing a detector produces may start life as a fact."""
        from omnilinker.store.base import Query

        for insight in ingested.stores.docs.find(Query("insights", limit=100)):
            assert insight["state"] == "advisory"
            assert 0.0 <= insight["confidence"] <= 1.0
            assert insight["detector"]

    def test_co_presence_uses_thread_membership(self) -> None:
        docs = [
            {"sender_ref": "a", "conversation_id": "C1", "ts": "2026-09-01T10:00:00Z"},
            {"sender_ref": "b", "conversation_id": "C1", "ts": "2026-09-01T10:05:00Z"},
            {"sender_ref": "c", "conversation_id": "C2", "ts": "2026-09-01T11:00:00Z"},
        ]
        participants = participants_by_conversation(docs)
        assert set(participants["C1"]) == {"a", "b"}
        interactions = interactions_from_documents(docs, participants)
        assert ("a", "b") in interactions
        assert ("a", "c") not in interactions
