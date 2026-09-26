"""Prediction and proactive surfacing (blueprint 9.3).

Everything here is a *proposal* with a score, never a fact. The three
predictors share one design property: they are computed from artifacts the user
can open, and they carry the evidence that produced them. A prediction you
cannot check is a prediction you cannot act on.

  P1 deadline     a date mentioned in a message, with its hardness and the
                  sentence it came from. Extracted, not inferred - see
                  `normalize_dates` for why "Friday" alone is not a deadline.
  P2 follow-up    a question that never got an answer. High value, cheap to
                  compute, and the most obviously useful thing in this file.
  P3 resurfacing  something relevant to recent activity that has not been
                  touched in a long time.

The honesty rule, applied throughout: **an unanswered question is a prediction
only if the thread plausibly continued.** Silence after a question is normal
conversation, so a suggestion requires that (a) the question looks like a real
request, and (b) enough messages followed it for the absence of an answer to be
informative. Without (b) this predictor is spam.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.ids import prefixed

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: Questions that request information, not yes/no confirmations. "can we ship
#: Friday" is a scheduling statement wearing a question mark; "do you have the
#: number" is a request.
REQUEST_QUESTION = re.compile(
    r"^\s*(what|which|where|when|who|how|why|do you (?:have|know|see|have you)|"
    r"did you|have you|can you (?:send|share|confirm|check|find)|"
    r"could you (?:send|share|confirm|check)|"
    r"is there|are there|any (?:idea|thoughts|chance)|"
    r"thoughts\?|opinions\?|feedback\?)\b",
    re.I,
)

#: Cues that a following message is the answer. Deliberately excludes "on it",
#: "will do", "got it", "done", "ok" and "sounds good": those are
#: acknowledgements, and treating an acknowledgement as an answer is exactly
#: the case the follow-up predictor exists to surface.
ANSWER_CUE = re.compile(
    r"^\s*(yes|no|yep|nope|sure|here(?:'s| is)|found it|answer|confirmed|"
    r"approved|signed|re:|regarding)\b|https?://",
    re.I,
)

#: Politeness / acknowledgement only. Carries no information.
ACKNOWLEDGEMENT = re.compile(
    r"^\s*(thanks?|thank you|thx|ty|ack|roger|gotcha|cheers|ok|okay|cool|nice|"
    r"great|perfect|sounds good|agreed|yes|yep|yup|no problem|np|👍|🙏|:\+1:|"
    r"will do|on it|done|lol|haha|lmao|👍{1,3})\s*[.!]?\s*$",
    re.I,
)

_TOKEN = re.compile(r"[A-Za-z0-9_@'.+-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower().strip(".'-") for t in _TOKEN.findall(text or "")
            if len(t.strip(".'-")) > 2]


def _parse(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# P1 deadlines
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    kind: str
    title: str
    detail: str
    due: str = ""
    confidence: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    state: str = "advisory"
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "_id": prefixed("pred"),
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "due": self.due,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 5),
            "source_ids": self.source_ids,
            "evidence": self.evidence[:5],
            "state": self.state,
        }


def predict_deadlines(
    documents: Sequence[Mapping[str, Any]],
    *,
    horizon_days: int = 45,
    now: datetime | None = None,
) -> list[Prediction]:
    """Surface upcoming extracted deadlines, hardest first.

    `hardness` comes from `normalize_dates` and encodes a real distinction: a
    date that arrived with a commitment cue ("I'll have it Friday") is a promise
    and sorts above a date that merely appeared next to a date-ish word
    ("Friday works for the offsite"). Treating them equally is how a deadline
    list becomes something users learn to ignore.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(days=horizon_days)
    out: list[Prediction] = []
    for doc in documents:
        for deadline in (doc.get("entities") or {}).get("deadlines") or []:
            due = _parse(deadline.get("due"))
            if due is None or not (now <= due <= horizon):
                continue
            hardness = str(deadline.get("hardness", "medium"))
            weight = {"hard": 1.0, "medium": 0.7, "soft": 0.35}.get(hardness, 0.5)
            days = (due - now).days
            # Urgency decays: a promise due in 3 days outranks one due in 40,
            # but a hard commitment next month still outranks a soft "sometime"
            # next week.
            urgency = 1.0 / (1.0 + days / 10.0)
            out.append(Prediction(
                kind="deadline",
                title=f"{deadline.get('cue', 'Deadline')}: {due.date().isoformat()}",
                detail=(
                    f"{hardness} commitment, due in {days} day(s). "
                    f"Cue: {deadline.get('cue', 'unspecified')!r}. "
                    "Extracted from the sentence, not inferred."
                ),
                due=due.isoformat().replace("+00:00", "Z"),
                confidence=float(deadline.get("confidence") or weight),
                source_ids=[str(doc.get("_id"))],
                evidence=[{"text": str(deadline.get("context", ""))[:200],
                           "ts": doc.get("ts", ""),
                           "provider": doc.get("provider", "")}],
                score=weight * urgency * float(deadline.get("confidence") or 0.6),
            ))
    out.sort(key=lambda p: -p.score)
    return out


# ---------------------------------------------------------------------------
# P2 unanswered questions
# ---------------------------------------------------------------------------


def predict_follow_ups(
    documents: Sequence[Mapping[str, Any]],
    *,
    min_following: int = 2,
    now: datetime | None = None,
) -> list[Prediction]:
    """Questions that were never answered in their thread.

    The `min_following` guard is the whole design. A question with nothing after
    it tells you nothing - the thread simply ended. Requiring several later
    messages makes silence informative, and the resulting suggestion is one the
    user can act on rather than one they have to evaluate.
    """
    now = now or datetime.now(timezone.utc)
    by_thread: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for doc in documents:
        if doc.get("body_text") is None:
            continue
        thread = str(doc.get("thread_root_id") or doc.get("conversation_id")
                     or doc.get("_id"))
        by_thread[thread].append(doc)

    out: list[Prediction] = []
    for thread, docs in by_thread.items():
        ordered = sorted(docs, key=lambda d: (str(d.get("ts") or ""),
                                               str(d.get("_id") or "")))
        for position, doc in enumerate(ordered):
            body = doc.get("body_text")
            if not isinstance(body, str) or not REQUEST_QUESTION.match(body):
                continue
            following = ordered[position + 1: position + 1 + min_following + 1]
            if len(following) < min_following:
                continue
            if any(ANSWER_CUE.match(str(f.get("body_text") or ""))
                   for f in following if isinstance(f.get("body_text"), str)):
                continue  # somebody answered, or acknowledged
            asked_at = _parse(doc.get("ts")) or now
            days = max(0, (now - asked_at).days)
            out.append(Prediction(
                kind="unanswered_question",
                title=(body.strip().rstrip("?")[:80] + "?"),
                detail=(
                    f"Asked in #{doc.get('extra', {}).get('channel_label', 'thread')}"
                    f" and followed by {min_following} message(s) with no answer. "
                    f"Silent for {days} day(s)."
                ),
                confidence=0.65,
                source_ids=[str(doc.get("_id"))] + [str(f.get("_id")) for f in following],
                evidence=[{"text": body.strip()[:200], "ts": doc.get("ts", ""),
                           "provider": doc.get("provider", "")}],
                # An old unanswered question is more likely genuinely forgotten
                # than actively being discussed right now.
                score=0.5 + min(0.35, days / 200.0),
            ))
    out.sort(key=lambda p: -p.score)
    return out


# ---------------------------------------------------------------------------
# P3 resurfacing
# ---------------------------------------------------------------------------


def predict_resurfacing(
    documents: Sequence[Mapping[str, Any]],
    recent_ids: Sequence[str],
    *,
    dormant_days: int = 90,
    now: datetime | None = None,
) -> list[Prediction]:
    """Old documents that look relevant to what the user is doing now.

    Relevance is embedding similarity to the recent window, so this is
    "you were looking at search sharding again" rather than "you have not opened
    this in 100 days". Pure recency is not a prediction, it is a reminder.
    """
    now = now or datetime.now(timezone.utc)
    recent = {str(i) for i in recent_ids}
    if not recent:
        return []
    from omnilinker.embeddings import DEFAULT_DIM, cosine, hash_embed

    anchors: list[list[float]] = []
    dormant: list[tuple[Mapping[str, Any], datetime]] = []
    for doc in documents:
        when = _parse(doc.get("ts"))
        if when is None:
            continue
        if str(doc.get("_id")) in recent:
            body = doc.get("body_text")
            if isinstance(body, str):
                anchors.append(hash_embed(body, DEFAULT_DIM))
        elif now - when > timedelta(days=dormant_days):
            body = doc.get("body_text")
            if isinstance(body, str) and len(body) > 40:
                dormant.append((doc, when))
    if not anchors or not dormant:
        return []

    out: list[Prediction] = []
    for doc, when in dormant:
        body = doc.get("body_text")
        vector = hash_embed(str(body), DEFAULT_DIM)
        score = max(cosine(vector, a) for a in anchors)
        if score < 0.30:
            continue
        days = (now - when).days
        out.append(Prediction(
            kind="resurfacing",
            title=str(doc.get("extra", {}).get("subject")
                      or doc.get("title") or (str(body)[:70] + "...")),
            detail=(
                f"Untouched for {days} days, but closely related to what you just "
                f"read. Similarity {score:.2f}."
            ),
            due=when.isoformat().replace("+00:00", "Z"),
            confidence=min(0.85, score),
            source_ids=[str(doc.get("_id"))],
            evidence=[{"provider": doc.get("provider", ""), "ts": doc.get("ts", ""),
                       "text": str(body)[:180]}],
            score=score * (1.0 + min(0.5, days / 400.0)),
        ))
    out.sort(key=lambda p: -p.score)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    predictions: list[Prediction]
    stats: dict[str, Any]

    def to_dict(self) -> dict:
        by_kind: dict[str, int] = defaultdict(int)
        for prediction in self.predictions:
            by_kind[prediction.kind] += 1
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "count": len(self.predictions),
            "by_kind": dict(sorted(by_kind.items())),
            **self.stats,
        }


def run_predictions(
    documents: Sequence[Mapping[str, Any]],
    *,
    recent_ids: Sequence[str] = (),
    now: datetime | None = None,
) -> PredictionResult:
    now = now or datetime.now(timezone.utc)
    out: list[Prediction] = []
    out += predict_deadlines(documents, now=now)
    out += predict_follow_ups(documents, now=now)
    if recent_ids:
        out += predict_resurfacing(documents, recent_ids, now=now)
    out.sort(key=lambda p: (-p.score, p.kind))
    return PredictionResult(out, {
        "documents_scanned": len(documents),
        "state": "advisory",
        "note": "All predictions are advisory. Nothing here is written back to a "
                "provider, and accepting one is a user action.",
    })
