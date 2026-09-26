"""Document enrichment: importance scoring and synthetic subjects.

Two small features that carry disproportionate product weight.

**Importance** (`score_importance`). The dashboard needs a "what matters" rail,
and recency alone is a bad proxy - a 4-word "yes" from today outranks the
quarterly budget thread from last month under any pure-recency rule. The score
is a transparent weighted sum over features a user can name: it has an owner, it
is long enough to contain substance, it carries a commitment, it is a direct
mention of you, it has attachments. No learned model, because a user who asks
"why is this ranked high" has to get an answer, and a linear model over named
features is the only kind that can produce one.

**Synthetic subjects** (`assign_thread_subjects`). The blueprint's own eval notes
identify the single biggest retrieval weakness: Slack messages have no subject
line, and a thread whose only text is "@Alice can you look at this" is
unfindable by the words the user remembers. So a thread gets a subject derived
from its root message - the first sentence, mentions and URLs stripped, length
capped - and that subject is indexed at triple weight. The same evaluation that
found the problem is what justifies the fix, which is the point of having an
evaluation set.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.normalize_dates import extract_deadlines

# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------

#: feature -> weight. Linear on purpose (see module docstring).
W_OWNER_KNOWN = 1.6
W_DEADLINE = 2.0
W_ATTACHMENT = 1.2
W_MENTION_OF_SELF = 2.4
W_DECISION_CUE = 1.8
W_LENGTH = 1.0
W_RECENCY = 1.4
W_INBOUND = 0.6

#: Words that indicate a decision, an owner, or a commitment. Weighted higher
#: than generic business vocabulary because "who is doing this" and "when does
#: this land" are the two things a person opening a search box actually wants.
DECISION_CUE = re.compile(
    r"\b(decid\w*|agreed|sign(?:ed)?[\s-]?off|approv\w*|final\w*|ship\w*|"
    r"launch\w*|deadline|due\s+(?:by|date)|by\s+(?:eod|end\s+of)|"
    r"action item|owner|assign\w*|confirm\w*|blocker|outage|incident|"
    r"escalat\w*|roll\s*back|post[\s-]?mortem|root cause)\b",
    re.I,
)

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w[\w.\-]*")
_WS = re.compile(r"\s+")

#: 40 characters is roughly one search-result title. Longer subjects stop being
#: titles and start being sentences, which harms the UI without helping recall.
MAX_SUBJECT_CHARS = 90


def score_importance(
    doc: Mapping[str, Any],
    *,
    now: datetime | None = None,
    self_refs: frozenset[str] = frozenset(),
) -> tuple[float, list[dict]]:
    """Return (score in 0..1, feature breakdown).

    The breakdown is not debug output; it is rendered in the UI. A user looking
    at a "high importance" badge should be able to see it is high *because*
    someone was named and a deadline was extracted, not because a model said so.
    """
    now = now or datetime.now(timezone.utc)
    features: list[dict] = []

    def add(name: str, weight: float, detail: str) -> None:
        if weight:
            features.append({"feature": name, "weight": round(weight, 2), "detail": detail})

    raw_total = 0.0
    max_total = (W_OWNER_KNOWN + W_DEADLINE + W_ATTACHMENT + W_MENTION_OF_SELF
                 + W_DECISION_CUE + W_LENGTH + W_RECENCY + W_INBOUND)

    sender = str(doc.get("sender_ref") or "")
    if sender and sender in self_refs:
        add("sent_by_you", W_OWNER_KNOWN, sender)
        raw_total += W_OWNER_KNOWN

    deadlines = (doc.get("entities") or {}).get("deadlines") or []
    if deadlines:
        hard = sum(1 for d in deadlines if d.get("hardness") == "hard")
        add("deadline", W_DEADLINE, f"{len(deadlines)} extracted deadline(s), {hard} hard")
        raw_total += W_DEADLINE

    attachments = doc.get("attachments") or []
    if attachments:
        add("attachment", W_ATTACHMENT, ", ".join(
            str(a.get("name", "")) for a in attachments[:3]))
        raw_total += W_ATTACHMENT

    body = doc.get("body_text")
    if isinstance(body, str) and self_refs:
        mentioned = [m for m in _MENTION.findall(body)
                     if any(m.lstrip("@") == ref.split(":", 1)[-1]
                            for ref in self_refs if ":" in ref)]
        if mentioned:
            add("mentions_you", W_MENTION_OF_SELF, f"{len(mentioned)} mention(s) of you")
            raw_total += W_MENTION_OF_SELF

    if isinstance(body, str):
        match = DECISION_CUE.search(body)
        if match:
            # The matched cue is a fragment of a sealed body, so it is not
            # quoted into a plaintext field. The feature name and its weight are
            # what explain the score; the fragment is not needed for that, and
            # the user can open the document to see the sentence.
            add("decision_language", W_DECISION_CUE, "commitment or decision verb")
            raw_total += W_DECISION_CUE
        # Log length, not linear length: the difference between 20 and 200 words
        # matters, the difference between 200 and 400 does not.
        words = len(body.split())
        length_score = W_LENGTH * min(1.0, (max(0, words - 20) / 180.0) ** 0.5)
        if length_score:
            add("substance", round(length_score, 2), f"{words} words")
            raw_total += length_score

    ts = _parse(doc.get("ts"))
    if ts:
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        # Half-life of 21 days. A 90-day-old incident report should not
        # outrank yesterday's decision, but it should not vanish either.
        recency = W_RECENCY * (0.5 ** (age_days / 21.0))
        add("recency", round(recency, 2), f"{age_days:.0f} days old")
        raw_total += recency

    if doc.get("is_inbound"):
        add("inbound", W_INBOUND, "received, not sent")
        raw_total += W_INBOUND

    return min(1.0, raw_total / max_total), features


def _parse(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Synthetic subjects
# ---------------------------------------------------------------------------

#: A message that carries no subject has to borrow one. These prefixes produce
#: a useless subject if left in ("can you:" for "can you send me the deck?"),
#: so they are stripped - the request itself is the informative part.
_SUBJECT_PREFIX = re.compile(
    r"^(hey|hi|hello|ok|okay|quick one|quick|fyi|heads up|ps|re|"
    r"can you|could you|would you|do you|did you|is there|are there|"
    r"i think|i wanted|just|any update|update|status)\b[\s,:.\-]*",
    re.I,
)

_FILLER_TAIL = re.compile(
    r"\s*\b(?:thanks?|thank you|thx|ty|please|pls|cheers|appreciate it|"
    r"let me know|lmk|no rush|asap|if possible)\b[\s.!?]*$",
    re.I,
)


def derive_subject(text: str, *, max_chars: int = MAX_SUBJECT_CHARS) -> str:
    """First meaningful clause of a message, as a title.

    Deterministic and inspectable. The same message always produces the same
    subject, which matters because the subject is part of the search index and
    a subject that drifts between runs makes search results non-reproducible.
    """
    if not text:
        return ""
    body = text.strip()
    # Strip a leading block quote - it is someone else's words.
    body = re.sub(r"^>.*(?:\n|$)", "", body).strip()
    first = re.split(r"(?<=[.!?])\s|\n", body, maxsplit=1)[0]
    # A leading @mention is an addressing prefix, and the name that follows it
    # is part of that prefix rather than the subject: "@Alice can you send the
    # deck" should title as "send the deck", not "Alice can you send the deck".
    # Detection has to happen *before* the mention is stripped, and it is the
    # mention's presence that licenses the name removal - stripping a leading
    # capitalised word unconditionally ate the first word of every sentence
    # that happened to start with one ("Root cause is the shard key" became
    # "cause is the shard key"), which quietly corrupted every subject.
    addressed = bool(_MENTION.match(first))
    first = _MENTION.sub("", first)
    first = _URL.sub("", first)
    first = _WS.sub(" ", first).strip(" .,:;-")
    if addressed:
        first = re.sub(r"^[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?\s+", "", first, count=1)
    # A chat export prefixes every message with the sender ("Bharat: Heads up").
    # A capitalised word immediately followed by a colon and a space, at the very
    # start of the text, is that prefix - and a colon in the middle of a
    # sentence is not, which is why the anchor is `^` rather than a global
    # substitution.
    first = re.sub(r"^[A-Z][\w'’-]{0,24}:\s+", "", first, count=1)
    first = _SUBJECT_PREFIX.sub("", first)
    first = _FILLER_TAIL.sub("", first).strip(" .,:;-")

    # Stripping a courtesy prefix can leave a fragment rather than a subject:
    # "Do you have five minutes?" becomes "have five minutes?", which is a
    # worse search key than the sentence it came from. When what is left is too
    # short to describe anything, fall through to the next sentence - which is
    # where the actual request usually is ("...I want to walk you through the
    # rebalancer before the Series A narrative gets locked").
    if len(first.split()) < 4:
        for candidate in re.split(r"(?<=[.!?])\s|\n", body):
            candidate = _MENTION.sub("", candidate)
            candidate = _URL.sub("", candidate)
            candidate = _WS.sub(" ", candidate).strip(" .,:;-")
            candidate = _SUBJECT_PREFIX.sub("", candidate)
            candidate = _FILLER_TAIL.sub("", candidate).strip(" .,:;-")
            if len(candidate.split()) >= 4:
                first = candidate
                break
    if len(first) > max_chars:
        cut = first[:max_chars].rsplit(" ", 1)[0]
        first = (cut or first[:max_chars]) + "..."
    return first


def assign_thread_subjects(
    documents: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = True,
) -> dict[str, str]:
    """Give every thread root a subject, and return the updates to apply.

    A *thread* is keyed by `thread_root_id` when the provider has one, else by
    `conversation_id` - a channel with no threading is one long thread, and
    using its first message as the subject is exactly as bad as the original
    problem. For those, the subject comes from the busiest participant's first
    message instead, which is a better representative of what the channel is
    about.
    """
    by_root: dict[str, list[Mapping[str, Any]]] = {}
    for doc in documents:
        root = str(doc.get("thread_root_id") or doc.get("conversation_id")
                   or doc.get("_id"))
        by_root.setdefault(root, []).append(doc)

    updates: dict[str, str] = {}
    for root, docs in by_root.items():
        existing = next((str(d.get("thread_subject") or "") for d in docs
                         if d.get("thread_subject")), "")
        if existing and not overwrite:
            continue
        anchored = [d for d in docs if str(d.get("_id")) == root]
        if anchored:
            source_doc = anchored[0]
        else:
            # No explicit root: prefer the earliest message, not the longest.
            # The first message in a channel is a greeting; the longest is
            # usually the one that happens to be a pasted log.
            source_doc = min(docs, key=lambda d: (str(d.get("ts") or "9999"),
                                                  str(d.get("_id") or "")))
        body = source_doc.get("body_text")
        if not isinstance(body, str):
            continue
        subject = derive_subject(body)
        if subject:
            updates[str(source_doc.get("_id"))] = subject
    return updates


# ---------------------------------------------------------------------------
# Enrichment pass
# ---------------------------------------------------------------------------


def enrich_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    self_refs: Iterable[str] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score importance across a document set and compute thread subjects.

    Returns a patch set rather than mutating: the caller decides whether to
    write it, which keeps this function pure and trivially testable.
    """
    self_refs = frozenset(self_refs)
    now = now or datetime.now(timezone.utc)

    importance: dict[str, dict] = {}
    for doc in documents:
        score, features = score_importance(doc, now=now, self_refs=self_refs)
        importance[str(doc.get("_id"))] = {
            "importance": round(score, 4),
            "importance_features": features,
        }

    subjects = assign_thread_subjects(documents)
    return {"importance": importance, "thread_subjects": subjects,
            "documents": len(documents)}
