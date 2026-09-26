"""Query language: one parser shared by search, NL2Query and the API.

Grammar (blueprint 8.5). Small on purpose - every construct here has a UI
equivalent, and anything without one is not worth supporting:

    plain words            free text
    "exact phrase"         phrase match
    -word                  exclude
    from:clara             field filter
    in:#engineering       conversation/channel filter
    has:file               documents an attachment
    type:note|file|...     kind filter
    after:2026-01-01      date range (also before:)
    person:per_x           resolved person id
    link:accepted          graph link state filter (needs the graph join)

The parser returns a `ParsedQuery`. It never touches the store, so it is
trivially unit-testable and can be reused by the NL2Query compiler as a
fallback when the language model is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from omnilinker.search.index import STOPWORDS, tokenize

_FIELD = re.compile(r"\b(\w+):(\"[^\"]*\"|\S+)")
_WORD = re.compile(r"\"([^\"]+)\"|(\S+)")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{4}/\d{2}/\d{2}$|^\d{2}/\d{2}/\d{4}$")

VALID_FIELDS = {
    "from", "in", "has", "type", "after", "before", "person", "link", "provider",
    "is", "tag", "sent", "mentions",
}


#: Words that mark the start of a question. A query beginning with one of these
#: is natural language even if it is short ("why?").
QUESTION_OPENERS = frozenset({
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
    "did", "do", "does", "is", "are", "was", "were", "can", "could", "should",
    "would", "will", "has", "have", "had", "tell", "show", "find", "list",
})

#: A query is treated as natural language when this fraction of its tokens are
#: stopwords, and it is long enough for the fraction to mean something.
NL_STOPWORD_RATIO = 0.4
NL_MIN_TOKENS = 3


@dataclass
class ParsedQuery:
    text: str = ""
    phrases: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    raw: str = ""
    #: Tokens that survived stopword removal. For a question these carry all
    #: the retrieval signal; for a keyword query the stopwords themselves may be
    #: what the user meant, so `terms` is used instead.
    content_terms: list[str] = field(default_factory=list)
    natural_language: bool = False
    stopwords_dropped: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.terms and not self.phrases

    @property
    def retrieval_terms(self) -> list[str]:
        """The terms the ranker should actually use for this query."""
        if self.natural_language and self.content_terms:
            return self.content_terms
        return self.terms

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "terms": self.terms,
            "retrieval_terms": self.retrieval_terms,
            "phrases": self.phrases,
            "exclude": self.exclude,
            "filters": self.filters,
            "natural_language": self.natural_language,
            "stopwords_dropped": self.stopwords_dropped,
        }


def parse(query: str) -> ParsedQuery:
    raw = (query or "").strip()
    out = ParsedQuery(raw=raw)
    if not raw:
        return out

    # Pull field filters out first so their values never become search terms.
    def strip_fields(text: str) -> str:
        def take(match: re.Match) -> str:
            key, value = match.group(1).lower(), match.group(2).strip('"')
            if key not in VALID_FIELDS:
                return match.group(0)
            out.filters.setdefault(key, []).append(value)
            return " "

        return _FIELD.sub(take, text)

    remainder = strip_fields(raw)
    out.text = remainder.strip()

    for quoted, bare in _WORD.findall(remainder):
        token = quoted or bare
        if token.startswith("-") and len(token) > 1:
            out.exclude.extend(tokenize(token[1:]))
        elif quoted:
            out.phrases.append(quoted)
        else:
            out.terms.extend(tokenize(token))

    # Content terms vs. stopwords. `terms` keeps everything (a keyword search
    # for "to be" should find "to be"), but a question is mostly function
    # words, and scoring on them ranks documents by coincidence. So detect the
    # two cases and use the right term set for each.
    out.content_terms = [t for t in out.terms if t not in STOPWORDS]
    out.stopwords_dropped = [t for t in out.terms if t in STOPWORDS]
    out.natural_language = _looks_natural(out)
    out.filters = {k: normalise_filter(k, v) for k, v in out.filters.items()}
    return out


def _looks_natural(parsed: ParsedQuery) -> bool:
    """Heuristic, and a deliberately cheap one.

    Two independent signals, either sufficient:
      * it opens with a question word, or
      * at least `NL_MIN_TOKENS` tokens of which `NL_STOPWORD_RATIO` or more
        are stopwords - which is what a typed question looks like and a typed
        keyword never does.

    The failure mode of guessing wrong is small and in opposite directions:
    treating a question as keywords adds noise to the ranking, treating keywords
    as a question drops terms the user typed on purpose. When in doubt the
    keyword reading is kept, because it never silently loses a term the user
    chose.
    """
    words = (parsed.text or "").lower().split()
    if not words:
        return False
    if words[0] in QUESTION_OPENERS:
        return True
    tokens = [t for _, t in _WORD.findall(parsed.text or "") if t]
    if len(tokens) < NL_MIN_TOKENS:
        return False
    stop_count = sum(1 for t in tokens if t in STOPWORDS)
    return stop_count / len(tokens) >= NL_STOPWORD_RATIO


_RELATIVE = {
    "today": 0, "yesterday": -1, "week": -7, "month": -30, "year": -365,
    "7d": -7, "30d": -30, "90d": -90, "365d": -365, "24h": -1,
}


def normalise_filter(key: str, values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        v = value.strip()
        if key in ("after", "before", "sent") and v.lower() in _RELATIVE:
            v = _resolve_relative(v.lower())
        if key in ("type", "has", "link", "is"):
            v = v.lower()
        if key == "from":
            v = v.lstrip("@").lower()
        out.append(v)
    return out


def _resolve_relative(token: str) -> str:
    days = _RELATIVE[token]
    when = datetime.now(timezone.utc) + timedelta(days=days)
    return when.date().isoformat()


# ---------------------------------------------------------------------------
# Structured search request (what the API accepts)
# ---------------------------------------------------------------------------


@dataclass
class SearchRequest:
    q: str = ""
    kinds: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    person_ids: list[str] = field(default_factory=list)
    conversation_ids: list[str] = field(default_factory=list)
    after: str = ""
    before: str = ""
    has_attachment: bool | None = None
    limit: int = 20
    offset: int = 0
    facets: list[str] = field(default_factory=lambda: ["provider", "kind"])
    highlight: bool = True

    @classmethod
    def from_parsed(cls, parsed: ParsedQuery, *, limit: int = 20) -> "SearchRequest":
        req = cls(q=parsed.text or parsed.raw, limit=limit)
        f = parsed.filters
        req.kinds = f.get("type", [])
        req.providers = f.get("provider", []) or ([f["from"][0]] if "from" in f and
                                                  f["from"][0] in _KNOWN_PROVIDERS else [])
        req.person_ids = f.get("person", [])
        req.conversation_ids = f.get("in", [])
        req.after = f.get("after", [""])[0]
        req.before = f.get("before", [""])[0]
        if "has" in f:
            has = f["has"][0]
            req.has_attachment = has in ("file", "files", "attachment", "attachments")
        return req

    def filter_dicts(self) -> list[dict[str, Any]]:
        """Translate the request into typed store filters. This is the only
        place request -> filter translation happens, so the store never sees a
        string it has to parse."""
        out: list[dict[str, Any]] = []
        if self.kinds:
            out.append({"field": "kind", "op": "in", "value": self.kinds})
        if self.providers:
            out.append({"field": "provider", "op": "in", "value": self.providers})
        if self.person_ids:
            out.append({"field": "entities.person_ids", "op": "in", "value": self.person_ids})
        if self.conversation_ids:
            out.append({"field": "conversation_id", "op": "in",
                        "value": self.conversation_ids})
        if self.after:
            out.append({"field": "ts", "op": "gte", "value": self.after})
        if self.before:
            out.append({"field": "ts", "op": "lte", "value": self.before})
        if self.has_attachment is not None:
            out.append({"field": "attachments", "op": "exists", "value": self.has_attachment})
        return out


_KNOWN_PROVIDERS = {
    "slack", "gmail", "discord", "whatsapp", "gdrive", "onedrive", "dropbox",
    "notion", "evernote", "youtube", "demo",
}


def parse_natural_language(question: str) -> dict[str, Any]:
    """Deterministic question decomposition.

    This is not a fallback bolted on for when the LLM is down - it is the
    *safety net* for NL2Query. The compiler in `nl2query.py` uses it to build
    the typed plan, and every plan it produces must be expressible in exactly
    this grammar, because that grammar is all the store can execute. If the
    LLM hallucinates a filter we do not support, the plan is rejected at
    compile time rather than silently ignored at run time.
    """
    text = (question or "").strip()
    lowered = text.lower()
    filters: dict[str, list[str]] = {}
    parsed = parse(text)

    signals: list[str] = []
    if re.search(r"\b(file|document|deck|pdf|spreadsheet|attachment)s?\b", lowered):
        signals.append("file")
    if re.search(r"\b(email|mail|inbox|gmail)\b", lowered):
        signals.append("email")
    if re.search(r"\b(note|notion|page|spec|doc)\b", lowered):
        signals.append("note")
    if re.search(r"\b(video|watch|transcript|playlist)\b", lowered):
        signals.append("video")
    if re.search(r"\b(who|with whom|from whom|about whom)\b", lowered):
        signals.append("person_lookup")
    if re.search(r"\b(when|deadline|due|by when|eta)\b", lowered):
        signals.append("deadline")
    if re.search(r"\b(attach|attached|attachment)\b", lowered):
        signals.append("has_attachment")
    if re.search(r"\b(decide|decided|decision|agreed|outcome)\b", lowered):
        signals.append("decision")
    if re.search(r"\b(today|yesterday|this week|last week|this month|last month)\b", lowered):
        signals.append("time_relative")

    for key, values in parsed.filters.items():
        filters.setdefault(key, []).extend(values)
    if "has" in filters and "has_attachment" not in signals:
        signals.append("has_attachment")

    return {
        "question": text,
        "free_text": parsed.text,
        "terms": parsed.terms,
        "phrases": parsed.phrases,
        "filters": filters,
        "signals": signals,
        # The words that carry no retrieval value. Reported so the UI can
        # explain why a result set looks the way it does.
        "stopwords_dropped": [w for w in lowered.split() if w in STOPWORDS],
    }
