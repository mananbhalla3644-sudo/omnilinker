"""Ingestion pipeline: idempotency, cursors, sealing, lineage (blueprint 6.6).

The two properties that matter most here are both about what happens the
*second* time:

  * **Idempotency.** Providers offer at-least-once delivery and overlapping
    windows. Re-ingesting the same data must overwrite in place and create
    nothing, or the document counts drift upward forever and every derived store
    inherits the duplicates.
  * **Cursors advance only on full success.** A partial failure must leave the
    cursor alone, so the next run re-reads the window. Re-reading is free
    because ingestion is idempotent; skipping data is not.
"""

from __future__ import annotations

from omnilinker.crypto.envelope import is_envelope
from omnilinker.pipeline import COLLECTION_FOR_KIND, NATURAL_KEY_FOR_KIND, IngestPipeline
from omnilinker.store.base import Query


class TestIngest:
    def test_ingests_the_demo_dataset(self, pipeline: IngestPipeline) -> None:
        run = pipeline.sync_connector("demo")
        assert run.status == "ok"
        assert run.counters.failed == 0
        assert run.counters.records > 50
        assert run.counters.created > 0

    def test_second_run_creates_nothing(self, pipeline: IngestPipeline) -> None:
        """The property that makes at-least-once delivery safe."""
        pipeline.sync_connector("demo")
        before = pipeline.stores.stats()["documents"]
        second = pipeline.sync_connector("demo")
        after = pipeline.stores.stats()["documents"]
        assert second.counters.created == 0
        assert second.counters.records == 0, "cursors should have nothing left to fetch"
        for name, count in before.items():
            if name in ("ingest_runs", "audit", "cursors"):
                continue
            assert after.get(name) == count, f"{name} grew on re-ingest: {count} -> {after.get(name)}"

    def test_reingest_over_an_explicit_window_updates_in_place(
        self, pipeline: IngestPipeline
    ) -> None:
        """Force a replay by clearing cursors, and check the counts hold."""
        pipeline.sync_connector("demo")
        before = pipeline.stores.stats()["documents"]
        for row in pipeline.stores.docs.find(Query("cursors", limit=100)):
            pipeline.stores.docs.delete("cursors", row["_id"])
        replay = pipeline.sync_connector("demo")
        after = pipeline.stores.stats()["documents"]
        assert replay.counters.created == 0
        assert replay.counters.updated > 0
        for name, count in before.items():
            if name in ("ingest_runs", "audit", "cursors"):
                continue
            assert after.get(name) == count

    def test_raw_artifacts_are_stored_before_normalization(
        self, pipeline: IngestPipeline
    ) -> None:
        """Evidence first. If a normalizer crashes on an unseen payload shape,
        the raw payload must already exist or the bug is unreproducible."""
        pipeline.sync_connector("demo")
        raw = pipeline.stores.docs.find(Query("raw_artifacts", limit=500))
        assert len(raw) >= 50
        for row in raw[:20]:
            assert row["payload"], "empty raw payload"
            assert row["payload_hash"].startswith("sha256:")
            assert row["ingest_run_id"], "raw artifact not linked to its ingest run"

    def test_every_document_has_lineage(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        for collection in ("messages", "files", "notes", "videos", "transcripts"):
            for row in pipeline.stores.docs.find(Query(collection, limit=300)):
                assert row.get("lineage"), f"{collection}/{row['_id']} has no lineage"
                assert row["lineage"]["source_artifact_id"]
                assert row["lineage"]["ingest_run_id"]

    def test_lineage_points_at_a_real_raw_artifact(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        row = pipeline.stores.docs.find(Query("messages", limit=1))[0]
        ref = row["lineage"]["raw_ref"]
        collection, provider, stream, artifact = ref.split(":", 3)
        assert collection == "raw_artifacts"
        assert pipeline.stores.docs.get(collection, f"{provider}:{stream}:{artifact}")

    def test_cursors_advance_per_stream(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        cursors = pipeline.stores.docs.find(Query("cursors", limit=100))
        assert len(cursors) >= 10
        for row in cursors:
            assert row["cursor"], "cursor not set"
            assert row["stream"]
            assert row["connector_id"] == "demo"

    def test_stream_filter_limits_the_pull(self, pipeline: IngestPipeline) -> None:
        run = pipeline.sync_connector("demo", stream_filter="gmail:me@example.com")
        assert len(run.streams) == 1
        assert run.streams[0]["stream"] == "gmail:me@example.com"
        assert pipeline.stores.docs.count("files") == 0, \
            "a filtered sync should not have pulled the drive stream"

    def test_unknown_connector_raises(self, pipeline: IngestPipeline) -> None:
        import pytest

        with pytest.raises(KeyError):
            pipeline.sync_connector("nope")

    def test_run_is_recorded_and_audited(self, pipeline: IngestPipeline) -> None:
        run = pipeline.sync_connector("demo")
        stored = pipeline.stores.docs.get("ingest_runs", run.ingest_run_id)
        assert stored and stored["status"] == "ok"
        assert stored["counters"]["created"] == run.counters.created
        audit = pipeline.stores.docs.find(Query("audit", limit=20))
        assert any(a.get("action") == "sync" for a in audit)


class TestSealing:
    def test_bodies_are_sealed_at_rest(self, pipeline: IngestPipeline) -> None:
        """The stored document must not contain the readable body.

        Empty bodies are skipped by `seal_fields` and that is correct: a
        WhatsApp media-only message has no text, and sealing "" would store a
        ciphertext with nothing in it. The assertion is therefore on every
        *non-empty* body, plus a coverage floor so that "everything is empty and
        therefore vacuously fine" cannot pass.
        """
        pipeline.sync_connector("demo")
        sealed = 0
        total = 0
        for collection, limit in (("messages", 60), ("notes", 20), ("transcripts", 10)):
            for row in pipeline.stores.docs.find(Query(collection, limit=limit)):
                body = row.get("body_text") or row.get("text") or ""
                if not body:
                    continue
                total += 1
                assert is_envelope(body), f"{collection}/{row['_id']} body is not sealed"
                sealed += 1
        assert total >= 40, f"only {total} non-empty bodies found"
        assert sealed == total

    def test_metadata_stays_queryable(self, pipeline: IngestPipeline) -> None:
        """Field-level sealing, not whole-record encryption: filters and facets
        need plaintext metadata, and that is the documented trade."""
        pipeline.sync_connector("demo")
        for row in pipeline.stores.docs.find(Query("messages", limit=40)):
            assert row["provider"]
            assert row["ts"]
            assert row["sender_ref"]

    def test_subjects_are_readable(self, pipeline: IngestPipeline) -> None:
        """Titles are the retrieval key; sealing them would make documents
        unfindable by the words the user already knows."""
        pipeline.sync_connector("demo")
        for row in pipeline.stores.docs.find(Query("messages", limit=40)):
            subject = (row.get("extra") or {}).get("subject")
            if subject:
                assert not is_envelope(subject), f"{row['_id']} subject was sealed"

    def test_decrypt_round_trips(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        row = pipeline.stores.docs.find(Query("messages", limit=1))[0]
        plain = pipeline.decrypt_record(row)
        assert plain["body_text"]
        assert not is_envelope(plain["body_text"])
        assert plain["provider"] == row["provider"]

    def test_tampered_record_fails_loudly(self, pipeline: IngestPipeline) -> None:
        """A single corrupt row must not silently become an empty document."""
        import pytest

        from omnilinker.crypto.envelope import DecryptionError

        pipeline.sync_connector("demo")
        row = pipeline.stores.docs.find(Query("messages", limit=1))[0]
        row["body_text"] = dict(row["body_text"])
        row["body_text"]["ct"] = "AAAA" + row["body_text"]["ct"][4:]
        with pytest.raises(DecryptionError):
            pipeline.decrypt_record(row)


class TestNaturalKeys:
    def test_every_kind_has_a_collection_and_a_key(self) -> None:
        assert set(COLLECTION_FOR_KIND) == set(NATURAL_KEY_FOR_KIND)

    def test_transcripts_do_not_reuse_the_video_key(self) -> None:
        """Sharing a key made transcripts and videos collide as graph nodes,
        and the video silently won."""
        assert NATURAL_KEY_FOR_KIND["transcript"] != NATURAL_KEY_FOR_KIND["video"]
        assert NATURAL_KEY_FOR_KIND["transcript"] != "video_id"

    def test_document_ids_are_provider_qualified(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        for collection in ("messages", "files", "notes", "videos", "transcripts"):
            for row in pipeline.stores.docs.find(Query(collection, limit=200)):
                assert ":" in row["_id"], f"{collection}/{row['_id']} is not provider-qualified"


class TestConversations:
    def test_conversations_are_created_and_counted(self, pipeline: IngestPipeline) -> None:
        pipeline.sync_connector("demo")
        conversations = pipeline.stores.docs.find(Query("conversations", limit=100))
        assert len(conversations) >= 10
        for conv in conversations:
            assert conv["conversation_key"]
            assert conv["message_count"] >= 1
            assert conv["first_ts"] <= conv["last_ts"], "first_ts is after last_ts"
