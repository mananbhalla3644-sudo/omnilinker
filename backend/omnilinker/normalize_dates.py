"""Temporal expression parsing (blueprint 11.4 P1).

The deadline pipeline's most important half is deterministic and rule-based:
"by Friday", "EOD tomorrow", "next week", "Mar 3", "end of month" have to be
resolved against the *message timestamp* in the author's timezone, with
discourse anchoring ("next Friday" said on a Thursday means the coming Friday).

No model, no network, fully unit-testable. The LLM layer in `ai/predict.py`
only *classifies* the sentence; the date arithmetic happens here.
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Any

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
    "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# "by friday", "due mar 3", "deadline: 2026-10-02", "eod tomorrow"
DEADLINE_CUE = re.compile(
    r"\b(by|before|due|deadline|needs? (?:to|be)|no later than|"
    r"eod|end of (?:day|week|month)|make sure (?:to|it)|must be|"
    r"cut[- ]?off|ship(?:ping)?|submit|turn in|hand ?off)\b",
    re.I,
)
HARD = re.compile(r"\b(must|required|mandatory|hard deadline|no later than|critical)\b", re.I)
SOFT = re.compile(r"\b(aim|target|hopefully|maybe|try to|ideally|asap|soon|if possible)\b", re.I)

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_MON_DAY = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I
)
_DAY_MON = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MONTHS) + r")\b", re.I
)
_WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.I)
_REL = re.compile(
    r"\b(today|tonight|tomorrow|tmrw|yesterday|"
    r"next (?:week|month|quarter|year|mon|tue|wed|thu|fri|sat|sun)|"
    r"this (?:week|month|quarter|year|mon|tue|wed|thu|fri|sat|sun)|"
    r"in (\d+) (?:days?|weeks?|months?|hours?)|"
    r"(\d+) (?:days?|weeks?) (?:from now|later)|"
    r"end of (?:day|week|month|quarter|year)|"
    r"eod|eow|next sprint)\b",
    re.I,
)


def _anchor(anchor: str | datetime | None) -> datetime:
    if isinstance(anchor, datetime):
        return anchor if anchor.tzinfo else anchor.replace(tzinfo=timezone.utc)
    if isinstance(anchor, str) and anchor:
        try:
            return datetime.fromisoformat(anchor.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _at(day: datetime, *, hour: int = 18, end: bool = False) -> datetime:
    """Deadline default: 18:00 local on the resolved day (EOD-ish)."""
    return day.replace(hour=hour if not end else 23, minute=0 if not end else 59, second=0, microsecond=0)


def _next_weekday(base: datetime, target: int, *, allow_today: bool) -> datetime:
    delta = (target - base.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return base + timedelta(days=delta)


def extract_dates(text: str, anchor: str | datetime | None = None, *, limit: int = 8) -> list[dict]:
    """Every date-ish reference in the text, with a resolved absolute value."""
    if not text:
        return []
    base = _anchor(anchor)
    out: list[dict] = []

    def add(raw: str, when: datetime, kind: str, precision: str = "day") -> None:
        out.append({
            "text": raw,
            "value": when.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "local_date": when.date().isoformat(),
            "kind": kind,
            "precision": precision,
        })

    for m in _ISO.finditer(text):
        try:
            d = datetime.fromisoformat(m.group(0)).replace(tzinfo=base.tzinfo or timezone.utc)
        except ValueError:
            continue
        add(m.group(0), d, "absolute", "day")

    for m in _SLASH.finditer(text):
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= a <= 31 and 1 <= b <= 12):
            continue
        year = base.year
        if y:
            year = int(y) + (2000 if len(y) == 2 else 0)
        try:
            d = datetime(year, b, a, tzinfo=base.tzinfo or timezone.utc)
        except ValueError:
            continue
        add(m.group(0), d, "absolute", "day")

    for rx, order in ((_MON_DAY, "md"), (_DAY_MON, "dm")):
        for m in rx.finditer(text):
            mon = MONTHS[m.group(1 if order == "md" else 2).lower()]
            day = int(m.group(2 if order == "md" else 1))
            year = base.year
            try:
                d = datetime(year, mon, day, tzinfo=base.tzinfo or timezone.utc)
            except ValueError:
                continue
            if d < base - timedelta(days=1):  # bare "Mar 3" said in Nov means next year
                d = d.replace(year=year + 1)
            add(m.group(0), d, "absolute", "day")

    for m in _WEEKDAY.finditer(text):
        wd = WEEKDAYS[m.group(1).lower()]
        ctx = text[max(0, m.start() - 12): m.end() + 12].lower()
        allow_today = "this" in ctx
        d = _next_weekday(base, wd, allow_today=allow_today)
        add(m.group(0), d, "weekday", "day")

    for m in _REL.finditer(text):
        phrase = m.group(0).lower()
        low = phrase.split()[0]
        if low == "today":
            add(m.group(0), _at(base), "relative", "day")
        elif low in {"tonight"}:
            add(m.group(0), _at(base, end=True), "relative", "day")
        elif low == "tomorrow":
            add(m.group(0), _at(base + timedelta(days=1)), "relative", "day")
        elif low == "tmrw":
            add(m.group(0), _at(base + timedelta(days=1)), "relative", "day")
        elif low == "yesterday":
            add(m.group(0), _at(base - timedelta(days=1)), "relative", "day")
        elif low in {"eod", "end"}:
            if phrase.endswith("day"):
                add(m.group(0), _at(base, end=True), "relative", "day")
            elif phrase.endswith("week"):
                add(m.group(0), _at(base + timedelta(days=(6 - base.weekday())), end=True), "relative", "week")
            elif phrase.endswith("month"):
                last = calendar.monthrange(base.year, base.month)[1]
                add(m.group(0), _at(base.replace(day=last), end=True), "relative", "month")
        elif low == "next":
            unit = phrase.split()[1]
            if unit in WEEKDAYS:
                add(m.group(0), _at(_next_weekday(base, WEEKDAYS[unit], allow_today=False)),
                    "relative", "day")
            elif unit == "week":
                add(m.group(0), _at(base + timedelta(days=7)), "relative", "week")
            elif unit == "month":
                add(m.group(0), _at(_add_months(base, 1)), "relative", "month")
            elif unit == "quarter":
                add(m.group(0), _at(_add_months(base, 3)), "relative", "quarter")
            elif unit == "year":
                add(m.group(0), _at(base.replace(year=base.year + 1)), "relative", "year")
        elif low == "this":
            unit = phrase.split()[1]
            if unit in WEEKDAYS:
                add(m.group(0), _at(_next_weekday(base, WEEKDAYS[unit], allow_today=True)),
                    "relative", "day")
            elif unit == "week":
                add(m.group(0), _at(base + timedelta(days=(6 - base.weekday())), end=True),
                    "relative", "week")
            elif unit == "month":
                last = calendar.monthrange(base.year, base.month)[1]
                add(m.group(0), _at(base.replace(day=last), end=True), "relative", "month")
        elif low == "in" and m.group(2):
            n, unit = int(m.group(2)), m.group(3)
            add(m.group(0), _at(base + timedelta(**{unit.rstrip("s"): n})), "relative", unit)
        elif m.group(4):
            n, unit = int(m.group(4)), m.group(5)
            add(m.group(0), _at(base + timedelta(**{unit: n})), "relative", unit)
        elif phrase == "asap":
            add(m.group(0), _at(base + timedelta(days=1)), "relative", "day")

    seen: set[tuple[str, str]] = set()
    uniq: list[dict] = []
    for item in out:
        key = (item["text"].lower(), item["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq[:limit]


def _add_months(base: datetime, months: int) -> datetime:
    month = base.month - 1 + months
    year = base.year + month // 12
    month = month % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return base.replace(year=year, month=month, day=day)


def extract_deadlines(text: str, anchor: str | datetime | None = None, *, limit: int = 5) -> list[dict]:
    """Deadline candidates only: a cue word AND a resolvable date in the same
    clause. Requiring both is what keeps the 'upcoming' widget free of noise
    (blueprint 11.4: precision over recall, and every card is dismissible)."""
    if not text:
        return []
    results: list[dict] = []
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        if not DEADLINE_CUE.search(sentence):
            continue
        dates = extract_dates(sentence, anchor, limit=3)
        if not dates:
            continue
        hardness = "hard" if HARD.search(sentence) else ("soft" if SOFT.search(sentence) else "medium")
        cue = DEADLINE_CUE.search(sentence)
        results.append({
            "sentence": sentence.strip()[:280],
            "cue": cue.group(0).lower() if cue else "",
            "due": dates[0]["value"],
            "due_date": dates[0]["local_date"],
            "hardness": hardness,
            "resolved_from": dates[0]["text"],
            "confidence": {"hard": 0.85, "medium": 0.6, "soft": 0.4}[hardness],
        })
        if len(results) >= limit:
            break
    return results
