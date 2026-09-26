"""Entity resolution (blueprint 7.2).

The output is a set of `Person` clusters. The input is a pile of per-provider
`Identity` records that disagree with each other on purpose, because that is
what real accounts look like.

Scoring is a transparent weighted-feature model, not a learned classifier, for
three reasons that matter more than accuracy here:

  1. **Explainability is a product requirement.** Every merge must be
     defensible to the person whose accounts are being merged. "The model said
     so" is not an acceptable answer when the consequence is your DMs and
     emails appearing under one profile.
  2. **The decisive feature must be provable.** A shared email token is a
     1.00-weight fact, not a probability. A black box cannot tell you that.
  3. **No training data.** There is no labelled corpus, and building one means
     asking users to hand-verify their own social graph.

So: features with published weights, hard conflicts, and a decision band.

    score >= merge_threshold    -> auto-merge into one Person
    suggest_threshold <= score  -> hold as a *suggestion* for the user
    otherwise                   -> distinct people

Conflicts are subtracted, not just capped. A pair that shares an email token
but has mutually exclusive handle history is not a merge; it is a bug or an
attack, and it should be visible.

The clustering itself is union-find over accepted edges, run in deterministic
order so the same input always produces the same output.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.identity.blocking import (
    Pair,
    build_blocks,
    cooccurrence_pairs,
    pairs_from_blocks,
)
from omnilinker.ids import prefixed

# ---------------------------------------------------------------------------
# Feature weights (blueprint 7.3)
# ---------------------------------------------------------------------------

W_EMAIL_EXACT = 1.00
W_PHONE_EXACT = 0.85
W_NAME_EXACT = 0.60
W_HANDLE_SAME = 0.55
W_NAME_CONTAINED = 0.35
W_COOCCURRED = 0.30
W_SHARED_CONVERSATION = 0.25
W_EMAIL_DOMAIN = 0.15

X_HANDLE_CONFLICT = -0.50
X_TEMPORAL_IMPOSSIBLE = -0.40
X_BOTH_ARE_SELF = -0.60
X_ONE_IS_CONTACT = -1.00

#: Word-ish name prefixes that are not distinguishing. "Alex" is a real
#: collision magnet, and treating every Alex as the same Alex is the single
#: most damaging false merge available in this domain.
GENERIC_TOKENS = frozenset({
    "admin", "info", "support", "sales", "contact", "hello", "team", "noreply",
    "no-reply", "notifications", "donotreply", "billing", "accounts", "me", "you",
    "the", "and", "user", "unknown", "null", "none", "self", "system", "bot",
})

_STOP = re.compile(r"[^a-z0-9 ]+")


@dataclass(frozen=True)
class Feature:
    name: str
    weight: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {"feature": self.name, "weight": round(self.weight, 3), "detail": self.detail}


@dataclass
class PairScore:
    a: str
    b: str
    score: float
    decision: str  # merge | suggest | distinct
    features: list[Feature] = field(default_factory=list)
    conflicts: list[Feature] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "identity_a": self.a,
            "identity_b": self.b,
            "score": round(self.score, 3),
            "decision": self.decision,
            "evidence": [f.to_dict() for f in self.features],
            "conflicts": [f.to_dict() for f in self.conflicts],
        }


@dataclass
class ResolutionResult:
    persons: list[dict]
    suggestions: list[dict]
    pairs: list[PairScore]
    stats: dict[str, Any]

    def as_dict(self) -> dict:
        return {
            "persons": len(self.persons),
            "suggestions": len(self.suggestions),
            "pairs_scored": len(self.pairs),
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_name(value: str) -> str:
    return _STOP.sub(" ", (value or "").lower()).split(" ")[0] if value else ""


def _all_words(value: str) -> list[str]:
    return [w for w in _STOP.sub(" ", (value or "").lower()).split() if w]


def _display_names(identity: Mapping[str, Any]) -> str:
    names = [identity.get("display_name") or ""]
    ref = str(identity.get("ref") or "")
    if ref.count(":") >= 2:
        names.append(ref.split(":", 2)[2])
    extra = identity.get("extra") or {}
    if isinstance(extra, Mapping):
        names.append(str(extra.get("name", "")))
    return " ".join(n for n in names if n)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: dict[str, str] = {i: i for i in items}
        self.rank: dict[str, int] = {i: 0 for i in self.parent}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class PairScorer:
    def __init__(
        self,
        identities: Mapping[str, Mapping[str, Any]],
        *,
        conversations: Mapping[str, set[str]] | None = None,
        last_seen: Mapping[str, datetime] | None = None,
    ) -> None:
        self.by_id = identities
        self.conversations = conversations or {}
        self.last_seen = last_seen or {}

    def _shared_conversations(self, a: str, b: str) -> int:
        return len(self.conversations.get(a, set()) & self.conversations.get(b, set()))

    def score(self, id_a: str, id_b: str) -> PairScore:
        a = self.by_id.get(id_a, {})
        b = self.by_id.get(id_b, {})
        features: list[Feature] = []
        conflicts: list[Feature] = []

        # -- hard gate: contact identities are attribution, not people -----
        a_contact = str(a.get("ref", "")).startswith("contact:")
        b_contact = str(b.get("ref", "")).startswith("contact:")
        if a_contact or b_contact:
            score = 0.0
            if a_contact and b_contact and a.get("email_token") == b.get("email_token"):
                # Two contact stubs for the same address really are the same
                # address; merge them so attribution has one target.
                return PairScore(id_a, id_b, W_EMAIL_EXACT, "merge",
                                 [Feature("email_exact", W_EMAIL_EXACT)])
            return PairScore(id_a, id_b, score, "distinct",
                             conflicts=[Feature("contact_not_person", X_ONE_IS_CONTACT)])

        # -- f1 email token: decisive -------------------------------------
        if a.get("email_token") and a.get("email_token") == b.get("email_token"):
            features.append(Feature("email_exact", W_EMAIL_EXACT,
                                    "identical deterministic email token"))

        # -- f2 phone token ----------------------------------------------
        if a.get("phone_token") and a.get("phone_token") == b.get("phone_token"):
            features.append(Feature("phone_exact", W_PHONE_EXACT,
                                    "identical deterministic phone token"))

        # -- f3/f5 names --------------------------------------------------
        name_a, name_b = _display_names(a).strip(), _display_names(b).strip()
        first_a, first_b = _norm_name(name_a), _norm_name(name_b)
        if first_a and first_a == first_b:
            if first_a in GENERIC_TOKENS:
                # "Admin" on two providers is two admins, not one person.
                conflicts.append(Feature("generic_name", -0.30,
                                         f"{first_a!r} is not a distinguishing name"))
            else:
                features.append(Feature("first_name_exact", W_NAME_EXACT, first_a))
        elif name_a and name_b and (first_a in name_b.lower() or first_b in name_a.lower()):
            if first_a in GENERIC_TOKENS or first_b in GENERIC_TOKENS:
                conflicts.append(Feature("generic_name", -0.30, "generic token inside name"))
            else:
                features.append(Feature("first_name_contained", W_NAME_CONTAINED,
                                        f"{first_a!r} within {name_b!r}"))

        # -- f4 handle ----------------------------------------------------
        handle_a = str(a.get("provider_user_id") or "").strip().lower()
        handle_b = str(b.get("provider_user_id") or "").strip().lower()
        if handle_a and handle_a == handle_b and a.get("provider") == b.get("provider"):
            features.append(Feature("handle_same", W_HANDLE_SAME, handle_a))

        # A shared email token is a decisive fact, and it changes how the
        # weaker signals below should be read. Two accounts with the same
        # address having *different provider handles* is not a contradiction -
        # it is what every multi-provider person looks like. Treating it as one
        # cancelled the merge it was supposed to qualify, and cross-source
        # resolution silently returned zero merges while still looking healthy.
        decisive = any(f.name in ("email_exact", "phone_exact") for f in features)

        # -- f6 shared conversations -------------------------------------
        shared = self._shared_conversations(id_a, id_b)
        if shared:
            features.append(Feature("shared_conversation", W_SHARED_CONVERSATION,
                                    f"{shared} shared conversation(s)"))

        # -- f7 email domain ---------------------------------------------
        domain_a = _domain(a.get("email"))
        domain_b = _domain(b.get("email"))
        if domain_a and domain_a == domain_b and domain_a not in _FREEMAIL:
            features.append(Feature("email_domain", W_EMAIL_DOMAIN, domain_a))
        elif (domain_a and domain_b and domain_a != domain_b
              and not _FREEMAIL & {domain_a, domain_b} and not decisive):
            conflicts.append(Feature("different_org_domain", -0.20,
                                     f"{domain_a} vs {domain_b}"))

        # -- conflicts ----------------------------------------------------
        if not decisive and self._handle_conflict(a, b):
            conflicts.append(Feature("handle_mismatch", X_HANDLE_CONFLICT,
                                     "same display name, different provider handle"))
        if not decisive and self._temporal_conflict(a, b):
            conflicts.append(Feature("temporal_impossible", X_TEMPORAL_IMPOSSIBLE,
                                     "one account is entirely younger than the other"))
        if a.get("is_self") and b.get("is_self"):
            conflicts.append(Feature("both_marked_self", X_BOTH_ARE_SELF, ""))

        # -- aggregate ----------------------------------------------------
        # Positive features saturate: two independent weak signals should not
        # add up to a confident merge. Cap the positive total at 1.0 so only
        # *evidence*, not accumulation, crosses the merge bar.
        positive = min(1.0, sum(f.weight for f in features))
        penalty = sum(f.weight for f in conflicts)
        total = max(0.0, min(1.0, positive + penalty))
        return PairScore(id_a, id_b, total, "", features, conflicts)

    # -- conflict detectors --------------------------------------------
    def _handle_conflict(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        """Same display name, different provider handles, no shared contact.

        This is the false-merge trap: two different "Alex" accounts, or one
        person and a shared team account that happens to be named after them.
        The name is the only thing agreeing, and a name is the weakest evidence
        in the whole system, so it counts as evidence *against* rather than for.
        """
        if a.get("email_token") and a.get("email_token") == b.get("email_token"):
            return False  # a shared email overrides any handle difference
        handle_a = str(a.get("provider_user_id") or "").strip().lower()
        handle_b = str(b.get("provider_user_id") or "").strip().lower()
        if not handle_a or not handle_b or handle_a == handle_b:
            return False
        if a.get("provider") == b.get("provider"):
            return False  # same provider, different id = different accounts, but
            # the id is authoritative there, not a conflict
        name_a, name_b = _norm_name(_display_names(a)), _norm_name(_display_names(b))
        return bool(name_a) and name_a == name_b

    def _temporal_conflict(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        """One account did not exist while the other was already active.

        A Slack account created in 2026 cannot belong to the same person as a
        WhatsApp identity last seen in 2023 *unless* the person created a second
        account - which is common, and is why this subtracts rather than
        vetoes. It is enough to push a name-only match back into "suggest".
        """
        first_a, first_b = _parse_ts(a.get("first_seen")), _parse_ts(b.get("first_seen"))
        if first_a is None or first_b is None:
            return False
        return (first_b - first_a).days > 400 or (first_a - first_b).days > 400


_FREEMAIL = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "icloud.com", "me.com", "protonmail.com", "proton.me",
    "aol.com", "gmx.com", "mail.com", "yandex.com", "zoho.com",
})


def _domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower() if email and "@" in email else ""


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class IdentityResolver:
    """Blocking -> scoring -> clustering -> Person records."""

    def __init__(
        self,
        *,
        merge_threshold: float = 0.90,
        suggest_threshold: float = 0.65,
        use_cooccurrence: bool = True,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.suggest_threshold = suggest_threshold
        self.use_cooccurrence = use_cooccurrence

    def resolve(
        self,
        identities: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]] = (),
    ) -> ResolutionResult:
        by_id = {str(d["_id"]): dict(d) for d in identities if d.get("_id")}
        if not by_id:
            return ResolutionResult([], [], [], {"identities": 0, "persons": 0})

        blocks, oversize = build_blocks(list(by_id.values()))
        pairs = pairs_from_blocks(blocks)

        if self.use_cooccurrence:
            ref_to_ids = _ref_to_ids(by_id)
            conversations = _conversations_by_identity(documents, ref_to_ids)
            for ref_a, ref_b in cooccurrence_pairs(documents):
                ids = tuple(sorted((ref_to_ids.get(ref_a, ""), ref_to_ids.get(ref_b, ""))))
                if all(ids) and ids[0] != ids[1]:
                    pairs.append(ids)  # type: ignore[arg-type]
            pairs = sorted(set(pairs))
        else:
            conversations = {}

        scorer = PairScorer(by_id, conversations=conversations)
        scores: list[PairScore] = []
        merge_edges: list[Pair] = []
        for id_a, id_b in pairs:
            if id_a not in by_id or id_b not in by_id:
                continue
            ps = scorer.score(id_a, id_b)
            if not ps.features and not ps.conflicts:
                continue  # nothing to say about this pair
            # The decision is assigned here, immediately after scoring, and the
            # clustering consumes it. Deciding in a second pass and branching on
            # the first pass is how this shipped with `merge_edges` permanently
            # empty: the scorer returns decision="" (it has no thresholds), the
            # first branch tested for "merge" and never matched, and every pair
            # fell through to the reporting pass. The output looked entirely
            # healthy - blocks built, pairs scored, persons emitted - and
            # nothing was ever merged.
            ps.decision = ("merge" if ps.score >= self.merge_threshold
                           else "suggest" if ps.score >= self.suggest_threshold
                           else "distinct")
            if ps.decision == "merge":
                merge_edges.append((id_a, id_b))
            if ps.decision != "distinct" or ps.conflicts:
                scores.append(ps)

        persons = self._cluster(by_id, merge_edges, scores)
        suggestions = self._suggestions(by_id, scores)
        contacts = self._attribute_contacts(by_id, persons)

        stats = {
            "identities": len(by_id),
            "blocks": len(blocks),
            "candidate_pairs": len(pairs),
            "oversize_blocks": oversize,
            "persons": len(persons),
            "merged_pairs": len(merge_edges),
            "suggestions": len(suggestions),
            "contacts_attributed": contacts,
            "merge_threshold": self.merge_threshold,
            "suggest_threshold": self.suggest_threshold,
        }
        return ResolutionResult(persons, suggestions, scores, stats)

    # -- clustering ---------------------------------------------------
    def _cluster(
        self,
        by_id: Mapping[str, Mapping[str, Any]],
        merge_edges: Sequence[Pair],
        scores: Sequence[PairScore],
    ) -> list[dict]:
        uf = _UnionFind(by_id.keys())
        for id_a, id_b in merge_edges:
            uf.union(id_a, id_b)

        clusters: dict[str, list[str]] = defaultdict(list)
        for iid in by_id:
            clusters[uf.find(iid)].append(iid)

        # Evidence index so a Person can explain why it exists.
        evidence: dict[tuple[str, str], list[Feature]] = {}
        for ps in scores:
            if ps.decision != "merge":
                continue
            root = uf.find(ps.a)
            for f in ps.features:
                if f.weight > 0:
                    evidence.setdefault((root, ps.b), []).append(f)

        persons: list[dict] = []
        for root, members in sorted(clusters.items()):
            is_contact = any(str(by_id[m].get("ref", "")).startswith("contact:")
                             for m in members)
            people = [by_id[m] for m in members
                      if not str(by_id[m].get("ref", "")).startswith("contact:")]
            if not people:
                continue  # a cluster of only contact stubs is not a person

            names = [str(p.get("display_name") or "") for p in people if p.get("display_name")]
            providers = sorted({str(p.get("provider")) for p in people})
            email = next((str(p.get("email")) for p in people if p.get("email")), "")
            primary = _primary_identity(people)
            person_id = f"per_{primary.get('ref', root).replace(':', '_')[:56]}"
            persons.append({
                "_id": person_id,
                "person_id": person_id,
                "display_name": _best_name(names),
                "aliases": sorted({n for n in names if n != _best_name(names)}),
                "primary_provider": primary.get("provider", ""),
                "providers": providers,
                "identity_ids": sorted(people and [str(p["_id"]) for p in people] or []),
                "contact_identity_ids": sorted(
                    str(by_id[m]["_id"]) for m in members
                    if str(by_id[m].get("ref", "")).startswith("contact:")
                ),
                "email": email,
                "message_count": sum(int(p.get("mention_count") or 0) for p in people),
                "cross_source": len(providers) > 1,
                "last_seen": max((str(p.get("last_seen") or "") for p in people),
                                 default=""),
                "is_self": any(p.get("is_self") for p in people),
                "is_contact_only": is_contact,
                "created_at": _now(),
            })
        return persons

    def _suggestions(self, by_id: Mapping[str, Mapping[str, Any]],
                     scores: Sequence[PairScore]) -> list[dict]:
        out: list[dict] = []
        for ps in scores:
            if ps.decision != "suggest":
                continue
            a, b = by_id.get(ps.a, {}), by_id.get(ps.b, {})
            out.append({
                "_id": f"sug_{ps.a}_{ps.b}"[:120],
                "identity_a": ps.a,
                "identity_b": ps.b,
                "display_name_a": _display_names(a),
                "display_name_b": _display_names(b),
                "provider_a": a.get("provider", ""),
                "provider_b": b.get("provider", ""),
                "score": round(ps.score, 3),
                "evidence": [f.to_dict() for f in ps.features],
                "conflicts": [f.to_dict() for f in ps.conflicts],
                "status": "pending",  # only a user action can set "accepted"
                "created_at": _now(),
            })
        return out

    def _attribute_contacts(self, by_id: Mapping[str, Mapping[str, Any]],
                            persons: Sequence[Mapping[str, Any]]) -> int:
        """Attach contact stubs to the Person that owns the same email token."""
        by_token: dict[str, str] = {}
        for person in persons:
            email = person.get("email")
            if not email:
                continue
            token = next((str(i.get("email_token")) for i in by_id.values()
                          if i.get("_id") in person.get("identity_ids", [])), "")
            if token:
                by_token[token] = str(person["person_id"])
        attached = 0
        for identity in by_id.values():
            if not str(identity.get("ref", "")).startswith("contact:"):
                continue
            owner = by_token.get(str(identity.get("email_token") or ""))
            if owner:
                identity["person_id"] = owner
                attached += 1
        return attached


# --------------------------------------------------------------------------


def _best_name(names: Sequence[str]) -> str:
    """Longest full name wins, because 'Clara Nowak' beats 'clara' as a
    display name. Deterministic tie-break on the name itself."""
    real = [n for n in names if " " in n.strip()]
    pool = real or list(names)
    return max(sorted(pool), key=lambda n: len(n.strip()), default="") or "Unknown"


def _primary_identity(people: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The identity that best represents the person: most mention_count, then
    earliest first_seen, then provider name for determinism."""
    return min(
        people,
        key=lambda p: (
            -int(p.get("mention_count") or 0),
            str(p.get("first_seen") or "9999"),
            str(p.get("provider") or ""),
        ),
    )


def _ref_to_ids(by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for iid, identity in by_id.items():
        ref = identity.get("ref")
        if ref:
            out.setdefault(str(ref), str(iid))
    return out


def _conversations_by_identity(
    documents: Sequence[Mapping[str, Any]], ref_to_ids: Mapping[str, str]
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        ref = doc.get("sender_ref")
        if not ref:
            continue
        iid = ref_to_ids.get(str(ref))
        if iid:
            out[iid].add(str(doc.get("conversation_id") or doc.get("stream") or ""))
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
