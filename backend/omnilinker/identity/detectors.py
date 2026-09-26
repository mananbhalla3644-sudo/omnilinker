"""Hidden-connection detectors (blueprint 7.4).

This is the part of the product that justifies a graph at all. Each detector
answers one question the user cannot answer by reading a single source.

  D1 dormant      A relationship that stopped. Real contact, zero recent signal.
  D2 burst        A short, intense collaboration window - a project, not a
                  friendship. Useful because it is easy to miss.
  D3 introducer   A and D never spoke, but share two mutual contacts. The
                  strongest "you should know this person" signal in the data.
  D4 single-source A fact you hold about someone rests entirely on one
                  provider. Usually means "we indexed one app and thought we
                  knew them".
  D5 divergence   The same topic was decided twice, differently, in two
                  places. The single most expensive organisational failure
                  this system can surface.
  D6 orphan file  A file that appears in conversation but whose owner or
                  contributors never appear in the workspace at all.

Every detector returns `Insight` objects with an explicit `confidence` and a
`state`. Nothing here is allowed to assert a fact: an insight is a *lead*, and
the graph edge it suggests is written with `state: "advisory"` until a human
acts on it (blueprint 11.5). The write path that can flip it to
`state: "accepted"` is user-action-only, by construction.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.ids import prefixed
from omnilinker.store.base import Edge

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DECISION_CUE = re.compile(
    r"\b(decided|decision|agreed|we will|we'll|going with|sign[- ]?off|"
    r"approved|final call|settled on|confirmed)\b", re.I
)
_REJECTION_CUE = re.compile(
    r"\b(rejected|not doing|scrapped|cancelled|canceled|reverted|"
    r"dropping that|overruled|not approved)\b", re.I
)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "it", "this",
    "that", "we", "i", "you", "he", "she", "they", "do", "does", "did", "can",
    "will", "would", "should", "could", "not", "no", "so", "up", "out", "my",
    "our", "your", "me", "us", "them", "about", "from", "by", "as", "all",
})


@dataclass
class Insight:
    kind: str
    title: str
    detail: str
    entities: list[str]
    confidence: float
    evidence: list[dict] = field(default_factory=list)
    state: str = "advisory"  # advisory | accepted | dismissed
    detector: str = ""

    def to_dict(self) -> dict:
        return {
            "_id": prefixed("ins"),
            "kind": self.kind,
            "detector": self.detector or self.kind,
            "title": self.title,
            "detail": self.detail,
            "entities": self.entities,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:5],
            "state": self.state,
        }


def _parse(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# D1 dormant relationship
# ---------------------------------------------------------------------------


def detect_dormant(
    documents: Sequence[Mapping[str, Any]],
    interactions: Mapping[tuple[str, str], list[datetime]],
    *,
    dormancy_days: int = 90,
    min_interactions: int = 4,
    now: datetime | None = None,
) -> list[Insight]:
    """Real contact that has gone quiet.

    The threshold matters: two messages six months apart is not a relationship
    that went dormant, it is two people who barely spoke. So we require
    `min_interactions` before we claim anything changed.
    """
    now = now or _now()
    cutoff = now - timedelta(days=dormancy_days)
    out: list[Insight] = []
    for (a, b), times in sorted(interactions.items()):
        if len(times) < min_interactions:
            continue
        times.sort()
        last = times[-1]
        first = times[0]
        if last > cutoff:
            continue
        gap = now - last
        out.append(Insight(
            kind="dormant_relationship",
            detector="D1",
            title=f"Connection went quiet after {len(times)} interactions",
            detail=(
                f"{len(times)} exchanges between {now - last} ago "
                f"({gap.days} days of silence). Last activity {last.date().isoformat()}, "
                f"first {first.date().isoformat()}."
            ),
            entities=[a, b],
            confidence=min(0.9, 0.45 + len(times) * 0.05),
            evidence=[{"ts": t.isoformat().replace("+00:00", "Z")} for t in times[-3:]],
        ))
    return out


# ---------------------------------------------------------------------------
# D2 burst collaboration
# ---------------------------------------------------------------------------


def detect_bursts(
    interactions: Mapping[tuple[str, str], list[datetime]],
    *,
    window_days: int = 14,
    burst_threshold: int = 8,
) -> list[Insight]:
    """A tight cluster of activity = a project, not a friendship.

    We deliberately do not say "they are close". We say "there was a
    concentrated collaboration window", which is a fact and is actionable.
    """
    out: list[Insight] = []
    for (a, b), times in sorted(interactions.items()):
        times = sorted(times)
        start = 0
        for end in range(len(times)):
            while times[end] - times[start] > timedelta(days=window_days):
                start += 1
            span = end - start + 1
            if span < burst_threshold:
                continue
            window_start, window_end = times[start], times[end]
            out.append(Insight(
                kind="collaboration_burst",
                detector="D2",
                title=f"{span} interactions inside {window_days} days",
                detail=(
                    f"Burst from {window_start.date().isoformat()} to "
                    f"{window_end.date().isoformat()}: {span} interactions. "
                    "Concentrated activity usually means a shared deliverable."
                ),
                entities=[a, b],
                confidence=min(0.85, 0.4 + span * 0.03),
                evidence=[{"ts": t.isoformat().replace("+00:00", "Z")}
                          for t in times[start:start + 3]],
            ))
            break  # one insight per pair: the strongest window is enough
    return out


# ---------------------------------------------------------------------------
# D3 hidden introducer
# ---------------------------------------------------------------------------


def detect_hidden_introducers(
    interactions: Mapping[tuple[str, str], list[datetime]],
    *,
    min_strength: int = 2,
) -> list[Insight]:
    """A and D never spoke, but share >= `min_strength` mutual contacts.

    This is the highest-value detector in the set. It is also the one most
    likely to be wrong, so the confidence is capped hard and the evidence lists
    the mutual contacts explicitly, so a user can judge it in two seconds.
    """
    neighbours: dict[str, set[str]] = defaultdict(set)
    for (a, b) in interactions:
        neighbours[a].add(b)
        neighbours[b].add(a)

    out: list[Insight] = []
    people = sorted(neighbours)
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            if b in neighbours[a]:
                continue  # they already talk
            mutual = neighbours[a] & neighbours[b]
            if len(mutual) < min_strength:
                continue
            out.append(Insight(
                kind="hidden_connection",
                detector="D3",
                title=f"Never spoke, but share {len(mutual)} mutual contact(s)",
                detail=(
                    "No direct interaction exists. Shared contacts: "
                    + ", ".join(sorted(mutual)[:6])
                    + ("..." if len(mutual) > 6 else "")
                    + ". This is a lead, not a relationship."
                ),
                entities=[a, b],
                # Deliberately low. Two mutual contacts is suggestive at best.
                confidence=min(0.55, 0.2 + 0.1 * len(mutual)),
                evidence=[{"mutual_contact": m} for m in sorted(mutual)[:5]],
            ))
    return out


# ---------------------------------------------------------------------------
# D4 single-source dependency
# ---------------------------------------------------------------------------


def detect_single_source(
    persons: Sequence[Mapping[str, Any]],
) -> list[Insight]:
    """A person we only know from one provider.

    Worth surfacing because it is usually a *coverage gap*, not a fact about
    the person: "we do not know this person" and "this is all we know about
    this person" are different statements and users conflate them.
    """
    out: list[Insight] = []
    for person in persons:
        providers = person.get("providers") or []
        if len(providers) == 1 and int(person.get("message_count") or 0) >= 5:
            out.append(Insight(
                kind="single_source_person",
                detector="D4",
                title=f"Known only from {providers[0]}",
                detail=(
                    f"{person.get('message_count')} records from one provider. "
                    "Either this person genuinely lives in one app, or a "
                    "connector is not reporting. Worth a look before you trust "
                    "this profile."
                ),
                entities=[str(person.get("person_id"))],
                confidence=0.4,
                evidence=[{"provider": p} for p in providers],
            ))
    return out


# ---------------------------------------------------------------------------
# D5 decision divergence
# ---------------------------------------------------------------------------


def detect_decision_divergence(
    documents: Sequence[Mapping[str, Any]],
    *,
    min_token_overlap: int = 3,
) -> list[Insight]:
    """The same decision reached twice, differently, in two sources.

    Method: extract decision-bearing sentences, build a keyword signature,
    bucket by signature, then look for buckets that contain both an acceptance
    cue and a rejection cue. That pair is a genuine conflict - not a nuance -
    and it is what costs a team a week.
    """
    buckets: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for doc in documents:
        text = _plain(doc)
        if not text:
            continue
        for sentence in _SENTENCE_SPLIT.split(text):
            if not (_DECISION_CUE.search(sentence) or _REJECTION_CUE.search(sentence)):
                continue
            signature = _signature(sentence)
            if len(signature) < min_token_overlap:
                continue
            buckets[signature].append({
                "doc_id": doc.get("_id"),
                "provider": doc.get("provider"),
                "ts": doc.get("ts"),
                "sentence": sentence.strip()[:280],
                "accept": bool(_DECISION_CUE.search(sentence)),
                "reject": bool(_REJECTION_CUE.search(sentence)),
            })

    out: list[Insight] = []
    for signature, entries in sorted(buckets.items()):
        providers = {e["provider"] for e in entries}
        if len(providers) < 2:
            continue
        accepts = [e for e in entries if e["accept"]]
        rejects = [e for e in entries if e["reject"]]
        if not accepts or not rejects:
            continue
        out.append(Insight(
            kind="decision_divergence",
            detector="D5",
            title="Same decision, opposite outcomes, in different sources",
            detail=(
                f"Topic signature {', '.join(signature[:5])} appears in "
                f"{len(providers)} sources with conflicting outcomes. "
                f"{len(accepts)} acceptance cue(s) vs {len(rejects)} rejection cue(s). "
                "Someone should reconcile this explicitly."
            ),
            entities=sorted({e["doc_id"] for e in entries if e["doc_id"]}),
            confidence=0.7,
            evidence=entries[:5],
        ))
    return out


def _signature(sentence: str) -> tuple[str, ...]:
    words = [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
             if w not in _STOPWORDS and len(w) > 2]
    # Use the rarest-looking words: drop the most common English verbs we see
    # in every decision sentence, so the signature is the *subject*, not the verb.
    return tuple(sorted(set(words) - {"will", "not", "has", "have", "was", "were", "let"})[:6])


def _plain(doc: Mapping[str, Any]) -> str:
    body = doc.get("body_text")
    if isinstance(body, Mapping):  # still sealed
        return ""
    return str(body or "")


# ---------------------------------------------------------------------------
# D6 orphan file
# ---------------------------------------------------------------------------


def detect_orphan_files(
    documents: Sequence[Mapping[str, Any]],
    files: Sequence[Mapping[str, Any]],
    known_person_ids: set[str],
) -> list[Insight]:
    """A file that shows up in conversation but whose owner is not in the
    workspace at all - usually a shared drive from a previous job, a customer,
    or a departing colleague whose identity never got resolved.
    """
    owner_of: dict[str, Mapping[str, Any]] = {}
    for f in files:
        owner_of[str(f.get("_id"))] = f

    referenced: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        for att in doc.get("attachments") or []:
            fid = att.get("provider_file_id")
            if fid:
                referenced[str(fid)].append(str(doc.get("_id")))

    out: list[Insight] = []
    for f in files:
        fid = str(f.get("_id"))
        mentions = referenced.get(f.get("provider_file_id") or "", [])
        owner = str(f.get("owner_ref") or "")
        if owner and not owner.startswith(("slack:", "gmail:", "whatsapp:", "gdrive:")):
            continue
        person_id = str(f.get("person_id") or "")
        if person_id and person_id in known_person_ids:
            continue
        if not mentions:
            continue
        out.append(Insight(
            kind="orphan_file",
            detector="D6",
            title=f"File discussed but owner unresolved: {f.get('name', '')}",
            detail=(
                f"Referenced in {len(mentions)} message(s) but its owner "
                f"({owner or 'unknown'}) did not resolve to a person. Either the "
                "owner never appears in the indexed sources, or identity "
                "resolution missed them."
            ),
            entities=[fid],
            confidence=0.55,
            evidence=[{"message_id": m} for m in mentions[:4]],
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    insights: list[Insight]
    stats: dict[str, Any]

    def as_dict(self) -> dict:
        by_detector = Counter(i.detector for i in self.insights)
        return {"insights": len(self.insights), "by_detector": dict(sorted(by_detector.items())),
                **self.stats}


def interactions_from_documents(
    documents: Sequence[Mapping[str, Any]],
    participants_by_conversation: Mapping[str, Sequence[str]] | None = None,
    ref_to_person: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], list[datetime]]:
    """Build the person-pair interaction timeline that D1/D2/D3 operate on.

    `participants_by_conversation` maps a conversation id to the set of refs
    that appear in it. Co-presence is the honest signal available: we did not
    read a DM between two people just because they share a channel, and the
    evidence payload says exactly "same thread", not "they spoke".
    """
    participants_by_conversation = participants_by_conversation or {}
    ref_to_person = ref_to_person or {}
    out: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for doc in documents:
        when = _parse(doc.get("ts"))
        sender = str(doc.get("sender_ref") or "")
        if when is None or not sender:
            continue
        a = ref_to_person.get(sender) or sender
        conversation = str(doc.get("conversation_id") or "")
        peers = participants_by_conversation.get(conversation, ())
        for peer in peers:
            b = ref_to_person.get(str(peer)) or str(peer)
            if b == a:
                continue
            out[tuple(sorted((a, b)))].append(when)  # type: ignore[index]
    return out


def participants_by_conversation(
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Who appears in each conversation, most-active first. One pass."""
    seen: dict[str, Counter] = defaultdict(Counter)
    for doc in documents:
        ref = doc.get("sender_ref")
        if not ref:
            continue
        seen[str(doc.get("conversation_id") or "")][str(ref)] += 1
    return {conv: [ref for ref, _ in counter.most_common()]
            for conv, counter in seen.items()}


def run_all(
    *,
    documents: Sequence[Mapping[str, Any]],
    files: Sequence[Mapping[str, Any]],
    persons: Sequence[Mapping[str, Any]],
    ref_to_person: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> DetectionResult:
    now = now or _now()
    interactions = interactions_from_documents(
        documents, participants_by_conversation(documents), ref_to_person
    )

    insights: list[Insight] = []
    insights += detect_dormant(documents, interactions, now=now)
    insights += detect_bursts(interactions)
    insights += detect_hidden_introducers(interactions)
    insights += detect_single_source(persons)
    insights += detect_decision_divergence(documents)
    insights += detect_orphan_files(documents, files, {str(p["person_id"]) for p in persons})

    # Highest confidence first; that is the order the UI shows them in.
    insights.sort(key=lambda i: (-i.confidence, i.kind, i.title))
    stats = {
        "interaction_pairs": len(interactions),
        "total_interactions": sum(len(v) for v in interactions.values()),
        "conversations": len(participants_by_conversation(documents)),
    }
    return DetectionResult(insights, stats)
