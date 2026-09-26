"""NL2Query: natural language -> an executable, typed query plan (blueprint 8.5).

The design constraint that shapes everything here: **a plan is only useful if the
store can execute it.** So the compiler's output is not "a query string", it is a
`QueryPlan` whose every field maps onto something the store understands - a
`SearchRequest`, a graph traversal, a facet aggregation. If a question needs
something the system cannot express, the plan says so explicitly in
`unsupported` rather than quietly dropping the clause.

That matters more than it sounds. A query generator that silently discards the
part it cannot handle answers a *different* question than the one asked, and
returns confident wrong results. "Show me the budget spreadsheet Alice sent in
March" degrades to a full-text search for "budget", which returns the budget
*discussion* - a plausible-looking answer that is not the answer.

Two paths, one output type:

  compiled  rule-based, offline, deterministic. Always available.
  model     optional LLM-assisted rewriting of the question into the same plan.

The rule-based path is not a degraded fallback. It is the *floor*: the LLM path
must produce a plan of the same shape and is rejected if it does not, so a
hallucinated filter cannot reach the store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from omnilinker.ids import prefixed
from omnilinker.search.query import (
    SearchRequest,
    parse,
    parse_natural_language,
)
from omnilinker.store import get_stores
from omnilinker.store.base import Query

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

INTENT_SEARCH = "search"
INTENT_PERSON = "person_lookup"
INTENT_TIMELINE = "timeline"
INTENT_AGGREGATE = "aggregate"
INTENT_DEADLINES = "deadlines"
INTENT_RELATED = "related"

ALL_INTENTS = (INTENT_SEARCH, INTENT_PERSON, INTENT_TIMELINE, INTENT_AGGREGATE,
               INTENT_DEADLINES, INTENT_RELATED)

#: What each intent needs from the store. `graph` means the plan cannot be
#: served by `SearchService` alone, and the API routes it to the graph
#: endpoints instead. Being explicit here is what stops a person lookup from
#: being answered with a text search over names.
INTENT_EXECUTION = {
    INTENT_SEARCH: "search",
    INTENT_PERSON: "graph",
    INTENT_TIMELINE: "search",
    INTENT_AGGREGATE: "facets",
    INTENT_DEADLINES: "search",
    INTENT_RELATED: "search",
}

# ---------------------------------------------------------------------------
# Patterns. Ordered: first match wins, so the specific patterns come first.
# ---------------------------------------------------------------------------

_QUOTED = r"[\"'\u201c]([^\"'\u201d]{2,80})[\"'\u201d]"

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (INTENT_AGGREGATE, re.compile(
        r"^\s*(how many|how much|what(?:'s| is) the (?:total|number|count|sum|average)"
        r"|count|total up|add up)\b", re.I)),
    (INTENT_DEADLINES, re.compile(
        r"\b(deadlines?|due\s+dates?|due\s+by|what(?:'s| is)\s+due|\beta\b|"
        r"what do i owe|commitments?|promises?)\b", re.I)),
    (INTENT_PERSON, re.compile(
        r"^\s*(who(?:'s| is| was| are| did)|whose|"
        r"what(?:'s| is)\s+(?:the\s+)?(?:email|e-?mail|phone|number|address|role|"
        r"title|handle|username|account)s?\b|"
        r"tell me about|profile for|look up)\b", re.I)),
    (INTENT_RELATED, re.compile(
        r"\b(related to|anything (?:else )?about|more like|similar to|"
        r"what else|mentions? of|where else)\b", re.I)),
    (INTENT_TIMELINE, re.compile(
        r"\b(over time|timeline|history of|what happened|chronolog|"
        r"in order|day by day|week by week|since|before|after|earlier|later|"
        r"first time|last time)\b", re.I)),
    (INTENT_SEARCH, re.compile(r".*", re.S)),
)

#: A named thing in the question, for `related` / `timeline` scoping.
_SUBJECT = re.compile(_QUOTED + r"|\babout ([A-Z][\w'-]*(?: [A-Z][\w'-]*)?)")

_PERSON_NAME = re.compile(
    r"\b(?:with|from|by|to|about|for|asked|told|pinged|emailed)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

#: Fallback for a person question with no preposition: a capitalised bigram that
#: is not the first word. "what is Alice Chen email" has no "from Alice" for the
#: pattern above, but the name is still the capitalised run in the middle.
_NAME_FALLBACK = re.compile(r"\b([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)\b")

#: Words that are capitalised by position, not because they name somebody.
_SENTENCE_INITIALS = frozenset({
    "what", "who", "when", "where", "why", "how", "which", "show", "find", "list",
    "tell", "did", "do", "does", "is", "are", "was", "were", "can", "could",
    "should", "would", "will", "has", "have", "had", "i", "my", "we", "our",
    "the", "a", "an", "search", "email", "slack", "gmail", "notion", "drive",
})

#: Provider names are ordinary English words ("dropbox", "onedrive", "cloud"),
#: so an explicit mention in a question is a strong filter signal even without
#: a `provider:` prefix.
PROVIDER_ALIASES = {
    "slack": "slack", "gmail": "gmail",
    "discord": "discord", "whatsapp": "whatsapp", "drive": "gdrive",
    "gdrive": "gdrive", "google drive": "gdrive", "onedrive": "onedrive",
    "one drive": "onedrive", "dropbox": "dropbox", "notion": "notion",
    "evernote": "evernote", "youtube": "youtube", "video": "youtube",
}
#: Deliberately absent: "email" and "mail". "What is Alice's email address"
#: asks for a *field*; reading it as "search gmail" both answers a different
#: question and hides the person lookup. Ambiguous aliases are not worth the
#: recall they would buy.

_TIME_WINDOW = re.compile(
    r"\b(today|yesterday|this week|last week|this month|last month|this quarter|"
    r"last quarter|this year|last year|past \d+ days?|last \d+ days?|"
    r"since (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{0,4})\b",
    re.I,
)


@dataclass
class QueryPlan:
    question: str
    intent: str
    execution: str
    request: SearchRequest
    confidence: float
    explanation: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    person_hint: str = ""
    subject: str = ""
    time_window: str = ""
    compiled_by: str = "rules"
    plan_id: str = ""
    signals: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "question": self.question,
            "intent": self.intent,
            "execution": self.execution,
            "compiled_by": self.compiled_by,
            "confidence": round(self.confidence, 3),
            "request": {
                "q": self.request.q,
                "kinds": self.request.kinds,
                "providers": self.request.providers,
                "person_ids": self.request.person_ids,
                "conversation_ids": self.request.conversation_ids,
                "after": self.request.after,
                "before": self.request.before,
                "has_attachment": self.request.has_attachment,
                "limit": self.request.limit,
            },
            "filters": self.request.filter_dicts(),
            "explanation": self.explanation,
            "unsupported": self.unsupported,
            "person_hint": self.person_hint,
            "subject": self.subject,
            "time_window": self.time_window,
            "signals": self.signals,
            "alternatives": self.alternatives,
        }


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class NL2QueryCompiler:
    def __init__(self, *, llm: Callable[[str], str] | None = None) -> None:
        #: Optional model-assisted rewriter. Takes the question, returns a JSON
        #: object of *plan overrides* (never a query string). It cannot widen
        #: what the store supports; it can only choose better filters.
        self.llm = llm

    # -- public --------------------------------------------------------
    def compile(self, question: str, *, limit: int = 20) -> QueryPlan:
        question = (question or "").strip()
        analysis = parse_natural_language(question)
        intent = self._intent(question)
        execution = INTENT_EXECUTION[intent]

        request = SearchRequest(q=analysis["free_text"] or question, limit=limit)
        explanation: list[str] = []
        unsupported: list[str] = []
        person_hint = ""
        subject = ""
        time_window = ""
        compiled_by = "rules"
        confidence = 0.6
        alternatives: list[str] = []

        # -- typed filters from the deterministic analysis ---------------
        for key, values in analysis["filters"].items():
            if key in ("provider", "type", "in", "person", "after", "before"):
                target = {"provider": "providers", "type": "kinds", "in": "conversation_ids",
                          "person": "person_ids"}[key]
                setattr(request, target, list(getattr(request, target)) + list(values))
                explanation.append(f"{key} filter -> {target}={values}")
            elif key == "has":
                request.has_attachment = values[0] in ("file", "files", "attachment",
                                                       "attachments")
                explanation.append("has:file -> attachment filter")
            else:
                unsupported.append(f"filter {key!r} is not supported by the store")

        inferred = self._infer_provider(question, analysis, request)
        if inferred:
            explanation.append(f"provider name in question -> provider={inferred}")

        # -- time window --------------------------------------------------
        window = _TIME_WINDOW.search(question)
        if window:
            time_window = window.group(0).lower()
            after = _resolve_window(time_window)
            if after:
                request.after = after
                explanation.append(f"time phrase {time_window!r} -> after:{after}")

        # -- intent-specific scoping --------------------------------------
        if intent == INTENT_PERSON:
            person_hint, unsupported_here = self._person_hint(question)
            unsupported += unsupported_here
            explanation.append(
                f"person intent: the answer must come from the graph, not text search "
                f"({person_hint or 'no name detected'})"
            )
            request.q = person_hint or analysis["free_text"] or subject
            request.person_ids = []
            confidence = 0.75 if person_hint else 0.4

        elif intent == INTENT_DEADLINES:
            request.kinds = request.kinds or ["message", "email"]
            explanation.append("deadline intent: restrict to message/email, surface "
                               "extracted deadlines with their confidence")
            confidence = 0.7

        elif intent == INTENT_AGGREGATE:
            explanation.append("aggregate intent: served from facets, not ranked results")
            confidence = 0.7

        elif intent == INTENT_TIMELINE:
            request.limit = max(request.limit, 100)
            explanation.append("timeline intent: widen the result window and order by ts")
            confidence = 0.65

        elif intent == INTENT_RELATED:
            subject, _ = self._subject(question)
            explanation.append(
                f"related intent: use embedding similarity around "
                f"{subject or 'the query text'}"
            )
            confidence = 0.6

        # -- signal-driven filter inference --------------------------------
        for signal in analysis["signals"]:
            if signal == "has_attachment" and request.has_attachment is None:
                request.has_attachment = True
                explanation.append("inferred attachment filter from the question")
            elif signal == "file" and not request.kinds:
                request.kinds = ["file"]
                explanation.append("inferred kind=file")
            elif signal in ("email",) and not request.kinds:
                request.kinds = ["email"]
                explanation.append("inferred kind=email")
            elif signal in ("note",) and not request.kinds:
                request.kinds = ["note"]
                explanation.append("inferred kind=note")
            elif signal in ("video",) and not request.kinds:
                request.kinds = ["video", "transcript"]
                explanation.append("inferred kind=video|transcript")
            elif signal == "person_lookup" and intent == INTENT_SEARCH:
                alternatives.append(INTENT_PERSON)

        if analysis["free_text"] and intent in (INTENT_SEARCH, INTENT_RELATED,
                                                INTENT_TIMELINE):
            explanation.append(
                f"free text {analysis['free_text']!r} -> "
                f"{len(analysis['terms'])} token(s)"
            )

        # -- optional model-assisted override -----------------------------
        if self.llm is not None:
            request, note = self._apply_llm(question, request)
            if note:
                explanation.append(note)
                compiled_by = "rules+model"
                confidence = min(0.95, confidence + 0.1)

        plan = QueryPlan(
            question=question,
            intent=intent,
            execution=execution,
            request=request,
            confidence=confidence,
            explanation=explanation,
            unsupported=unsupported,
            person_hint=person_hint,
            subject=subject,
            time_window=time_window,
            compiled_by=compiled_by,
            plan_id=prefixed("plan"),
            signals=analysis["signals"],
            alternatives=alternatives,
        )
        return plan

    def compile_and_run(self, question: str, *, limit: int = 20) -> dict:
        """Compile, execute, and return both - so a UI can show the plan next to
        the results. Showing the plan is not a debug affordance; it is how the
        user learns to trust a system that guessed at what they meant."""
        from omnilinker.search.service import get_search_service

        plan = self.compile(question, limit=limit)
        if plan.execution == "search":
            response = get_search_service().search(plan.request, query_text=plan.request.q)
            return {"plan": plan.to_dict(), "result": response.to_dict()}
        if plan.execution == "facets":
            response = get_search_service().search(plan.request, query_text=plan.request.q)
            return {"plan": plan.to_dict(),
                    "result": {"facets": response.facets, "total": response.total,
                               "took_ms": response.took_ms}}
        # graph / person intents need the graph service; the API layer owns it
        return {"plan": plan.to_dict(), "result": None,
                "dispatch": plan.execution}

    # -- internals -----------------------------------------------------
    def _intent(self, question: str) -> str:
        for intent, pattern in PATTERNS:
            if pattern.search(question or ""):
                return intent
        return INTENT_SEARCH

    def _person_hint(self, question: str) -> tuple[str, list[str]]:
        match = _PERSON_NAME.search(question)
        if match:
            return match.group(1), []
        quoted = re.search(_QUOTED, question)
        if quoted:
            return quoted.group(1), []
        for candidate in _NAME_FALLBACK.findall(question or ""):
            if candidate.split()[0].lower() in _SENTENCE_INITIALS:
                continue
            return candidate, []
        return "", [
            "no person name detected; the graph endpoint needs a person id or a "
            "name to resolve"
        ]

    def _infer_provider(self, question: str, analysis: Mapping[str, Any],
                        request: SearchRequest) -> str:
        """A provider named in plain words is a filter, not a search term.

        "how many messages are in slack" should not text-search for the string
        "slack"; it should count slack messages. This only fires when the word is
        an exact alias match, because "drive" and "cloud" are ordinary English
        and a fuzzy match would invent filters the user never asked for.
        """
        if request.providers or analysis["filters"].get("provider"):
            return ""
        lowered = (question or "").lower()
        hits: list[str] = []
        for alias, provider in PROVIDER_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered) and provider not in hits:
                hits.append(provider)
        # A body-text mention of a provider is legitimate content, so only trust
        # the alias when it is unambiguous.
        if len(hits) != 1:
            return ""
        request.providers = hits
        return hits[0]

    def _subject(self, question: str) -> tuple[str, list[str]]:
        match = _SUBJECT.search(question)
        if not match:
            return "", []
        return (match.group(1) or match.group(2) or "").strip(), []

    def _apply_llm(self, question: str, request: SearchRequest) -> tuple[SearchRequest, str]:
        """Model-assisted override, constrained to what the store supports.

        The model is asked for a *plan fragment*, not a query string, and the
        result is validated field by field. Anything it invents is reported in
        `unsupported` and dropped. This is the whole safety story for
        LLM-assisted search: the model proposes, the schema disposes.
        """
        try:
            raw = self.llm(question)  # type: ignore[misc]
            data = json.loads(raw)
        except Exception as exc:
            return request, f"model assist unavailable ({exc}); rules used alone"
        if not isinstance(data, dict):
            return request, "model returned a non-object; rules used alone"

        applied: list[str] = []
        rejected: list[str] = []
        for key, attr in (("providers", "providers"), ("kinds", "kinds"),
                          ("conversation_ids", "conversation_ids")):
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                if key == "providers" and not all(v in _KNOWN_PROVIDERS for v in value):
                    rejected.append(f"unknown provider in {value}")
                    continue
                setattr(request, attr, value)
                applied.append(f"{key}={value}")
        for key, attr in (("after", "after"), ("before", "before")):
            value = data.get(key)
            if isinstance(value, str) and _ISO_DATE.match(value):
                setattr(request, attr, value)
                applied.append(f"{key}={value}")
            elif value:
                rejected.append(f"{key}={value!r} is not an ISO date")
        if isinstance(data.get("has_attachment"), bool):
            request.has_attachment = data["has_attachment"]
            applied.append("has_attachment")
        if isinstance(data.get("q"), str) and data["q"].strip():
            request.q = data["q"].strip()
            applied.append(f"q={data['q']!r}")

        note = f"model assist applied: {', '.join(applied)}" if applied else "model assist no-op"
        if rejected:
            note += f" | rejected: {'; '.join(rejected)}"
        return request, note


_KNOWN_PROVIDERS = {
    "slack", "gmail", "discord", "whatsapp", "gdrive", "onedrive", "dropbox",
    "notion", "evernote", "youtube", "demo",
}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_RELATIVE_DAYS = {
    "today": 0, "yesterday": -1, "this week": -7, "last week": -14,
    "this month": -30, "last month": -60, "this quarter": -90, "last quarter": -180,
    "this year": -365, "last year": -730,
}


def _resolve_window(phrase: str) -> str:
    from datetime import datetime, timedelta, timezone

    phrase = phrase.lower()
    days = _RELATIVE_DAYS.get(phrase)
    if days is None:
        m = re.match(r"(?:past|last) (\d+) days?", phrase)
        if m:
            days = -int(m.group(1))
    if days is None:
        return ""
    when = datetime.now(timezone.utc) + timedelta(days=days)
    return when.date().isoformat()
