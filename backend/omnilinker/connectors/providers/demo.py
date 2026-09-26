"""Demo source: a deterministic, self-contained multi-source dataset.

Two jobs, and both matter:

1. **Seed data.** A fresh install must show a populated dashboard, a graph with
   real structure, and search results - otherwise the product looks broken
   before the user has connected anything.
2. **Golden corpus.** The blueprint (15.1) requires contract tests for every
   connector's `normalize()`. The synthetic payloads here are stored as
   fixtures and the normalizers are asserted against them, so the pipeline is
   covered without touching a real provider.

It is deterministic: a fixed seed produces byte-identical output, so search
scores, clustering and graph shape are all reproducible in tests.

The people are deliberately *cross-source linked but inconsistent* - Alice is
`U01ABC` on Slack, `alice.chen@northwind.io` on Gmail and "Alice" in a
WhatsApp export. That inconsistency is the whole point: it is what the identity
resolver (section 7) has to solve, and it is why the demo graph has structure
worth looking at.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from omnilinker.connectors.contract import (
    CanonicalRecord,
    Connector,
    ConnectorDescriptor,
    PullPage,
    RawEnvelope,
    Scope,
    StreamDescriptor,
    TokenGrant,
)
from omnilinker.ids import prefixed
from omnilinker.normalize import clean_text, content_hash, iso
from omnilinker.normalize_dates import extract_dates, extract_deadlines

SEED = 20260926
NOW = datetime(2026, 9, 26, 18, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------

PEOPLE: list[dict[str, Any]] = [
    {
        "key": "alice",
        "name": "Alice Chen",
        "slack_id": "U01ALICE",
        "gmail": "alice.chen@northwind.io",
        "wa": "Alice",
        "discord_id": "2211445566778899001",
        "role": "Head of Product",
        "phone": "+447700900123",
    },
    {
        "key": "bharat",
        "name": "Bharat Iyer",
        "slack_id": "U02BHARAT",
        "gmail": "b.iyer@northwind.io",
        "wa": "Bharat",
        "discord_id": "2211445566778899002",
        "role": "Staff Engineer",
        "phone": "+919876543210",
    },
    {
        "key": "clara",
        "name": "Clara Nowak",
        "slack_id": "U03CLARA",
        "gmail": "clara.nowak@brightfold.com",
        "wa": "Clara",
        "discord_id": "2211445566778899003",
        "role": "Design Lead",
        "phone": "+48123456789",
    },
    {
        "key": "diego",
        "name": "Diego Fuentes",
        "slack_id": "U04DIEGO",
        "gmail": "diego.fuentes@brightfold.com",
        "wa": "Diego",
        "discord_id": "2211445566778899004",
        "role": "Data Scientist",
        "phone": "+34600111222",
    },
    {
        "key": "self",
        "name": "You",
        "slack_id": "U00SELF",
        "gmail": "me@example.com",
        "wa": "You",
        "discord_id": "2211445566778899005",
        "role": "Founder",
        "phone": "",
        "is_self": True,
    },
]

#: Slack and WhatsApp do not agree on Clara's surname. Identity resolution has to
#: notice that "Clara" on Slack and "Clara Nowak" on Gmail are the same human.
CLARA_SLACK_NAME = "clara"

CHANNELS = [
    {"id": "C0PRODUCT", "name": "product", "type": "channel", "topic": "Roadmap, specs, decisions"},
    {"id": "C0ENG", "name": "engineering", "type": "channel", "topic": "Ship logs, incidents, reviews"},
    {"id": "C0RANDOM", "name": "random", "type": "channel", "topic": "Non-work chatter"},
    {"id": "C0INCIDENT", "name": "incident-4417", "type": "channel", "topic": "SEV-2 latency spike"},
    {"id": "D0ALICE", "name": "DM Alice", "type": "im", "peer": "alice"},
    {"id": "D0BHARAT", "name": "DM Bharat", "type": "im", "peer": "bharat"},
    {"id": "D0CLARA", "name": "DM Clara", "type": "im", "peer": "clara"},
]

FILES = [
    ("F1QPRDSK", "Q3-Product-Review.pdf", "application/pdf", 482_133,
     "slides/2026-09-18-product-review.pdf"),
    ("F1BUDGET", "FY27-Budget-v4.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 91_002,
     "finance/fy27-budget-v4.xlsx"),
    ("F1ONBOARD", "Onboarding-Checklist.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 24_881,
     "people/onboarding-checklist.docx"),
    ("F1INCIDENT", "SEV2-postmortem-4417.md", "text/markdown", 12_004,
     "engineering/postmortems/sev2-4417.md"),
    ("F1CONTRACT", "Brightfold-MSA-signed.pdf", "application/pdf", 671_299,
     "legal/brightfold-msa-signed.pdf"),
    ("F1DECK", "Series-A-narrative-v7.key", "application/vnd.apple.keynote", 8_442_910,
     "fundraising/series-a-narrative-v7.key"),
    ("F1RETENTION", "Retention-cohorts-Q3.csv", "text/csv", 1_204_887,
     "analytics/retention-cohorts-q3.csv"),
]

NOTES = [
    {
        "page_id": "notion-product-spec",
        "title": "Unified Search - Product Spec",
        "created": "2026-07-02T09:15:00Z",
        "edited": "2026-09-24T16:40:00Z",
        "parent": "notion-root",
        "body": """# Unified Search - Product Spec

## Problem
People cannot answer "what did we decide about X" because the decision lives in
Slack for one team, in email for another, and in a Notion page that nobody
linked to either.

## Requirements
- Cross-source lexical search with sub-2s p95 over 5M documents
- Graph traversal from any entity without unbounded fan-out
- Every result carries a citation to the source artifact

## Non-goals
- Writing back to any source (blueprint N1)
- Full field-level E2EE (rejected for v1, see 10.2)

## Open questions
1. Do we ship vector search in v1 or behind a flag?
2. What is the k for k-anonymized token projections?
""",
    },
    {
        "page_id": "notion-incident-4417",
        "title": "SEV-2 4417 postmortem draft",
        "created": "2026-09-20T22:05:00Z",
        "edited": "2026-09-22T11:00:00Z",
        "parent": "notion-eng",
        "body": """# SEV-2 4417 postmortem draft

## Impact
Search p95 went from 640ms to 9.4s for 71 minutes. 2,140 requests failed.

## Timeline
- 21:04 UTC deploy of index sharding (v1.8.0)
- 21:11 UTC p95 alert fires
- 21:22 UTC rollback initiated
- 22:15 UTC fully recovered

## Root cause
The shard key was the document id, but every query in the incident window hit
one hot shard because the rebalancer had not run.

## Action items
- [x] Add shard-aware latency SLOs
- [x] Canary the rebalancer at 1%
- [ ] Write the runbook for manual rebalance

Owner: Bharat. Target: end of month.
""",
    },
    {
        "page_id": "notion-hiring",
        "title": "Staff engineer loop - notes",
        "created": "2026-08-11T13:30:00Z",
        "edited": "2026-08-11T13:30:00Z",
        "parent": "notion-people",
        "body": """# Staff engineer loop - notes

Interviews on 14 Aug. Panel: Alice, Bharat, Clara.

Feedback form reminders go out by EOD Friday. Two of five candidates asked
about on-call expectations, so we need a written answer before the debrief.

## Scores
- system design: strong
- cross-team influence: mixed
- writing: strong
""",
    },
    {
        "page_id": "notion-privacy",
        "title": "Data retention policy",
        "created": "2026-06-01T10:00:00Z",
        "edited": "2026-09-19T09:12:00Z",
        "parent": "notion-root",
        "body": """# Data retention policy

- Raw provider payloads: kept 90 days, then hard-deleted
- Derived stores: rebuildable at any time from raw
- Audit log: kept 12 months, append-only
- Keystore: never leaves the trust boundary

Contact for red-team questions: alice.chen@northwind.io
""",
    },
]

VIDEOS = [
    {
        "video_id": "yt-scaling-search",
        "title": "Scaling inverted indexes to a billion documents",
        "channel_title": "Systems Weekly",
        "published": "2026-05-14T12:00:00Z",
        "duration_s": 1847,
        "tags": ["search", "indexing", "sharding"],
        "captions": [
            (0, 41, "Let's talk about sharded inverted indexes and why most teams get the shard key wrong."),
            (41, 96, "The default instinct is to shard by document id. That works right up until your access pattern is not uniform."),
            (96, 168, "In this incident we sharded by document id and every hot query collapsed onto a single shard for seventy one minutes."),
            (168, 240, "The fix was not a bigger cluster. The fix was choosing a shard key that matches the dominant query predicate."),
            (240, 331, "Now let us look at the rebalancer, which is the part everyone defers and then regrets."),
            (331, 410, "A rebalancer is just a bounded migration with a rate limit and a kill switch. It should ship in week one."),
        ],
    },
    {
        "video_id": "yt-negotiation",
        "title": "Negotiating enterprise contracts without a lawyer",
        "channel_title": "Founder Sessions",
        "published": "2026-06-30T15:00:00Z",
        "duration_s": 1420,
        "tags": ["sales", "contracts"],
        "captions": [
            (0, 52, "Enterprise deals stall on three things: liability, data residency, and termination for convenience."),
            (52, 140, "Liability first. Cap it at twelve months of fees and offer a supercap only above a threshold."),
            (140, 226, "Data residency is where procurement discovers your architecture. Answer it before they ask."),
            (226, 305, "Termination for convenience is the one you can trade. Give them 60 days notice and you get the price."),
            (305, 388, "Never let legal invent the commercial terms. They will trade your margin for a clause nobody reads."),
        ],
    },
]

# ---------------------------------------------------------------------------
# Message scripts
# ---------------------------------------------------------------------------

SLACK_LINES: list[tuple[str, str, str, list[str]]] = [
    # (channel, sender key, body, attachments)
    ("C0PRODUCT", "bharat", "Spec for unified search is in Notion: Unified Search - Product Spec. Comments welcome by EOD Friday.", []),
    ("C0PRODUCT", "alice", "Agree on the non-goals. Biggest open question is whether we ship vector search in v1 or behind a flag. My vote: behind a flag, we do not have the eval set yet.", ["F1QPRDSK"]),
    ("C0PRODUCT", "clara", "Design review is Thursday at 15:00. I need the final copy by Wednesday 12:00 or I am designing against stale text again.", []),
    ("C0PRODUCT", "alice", "@Clara I will get you the copy Wednesday morning. Add Clara Nowak as the reviewer on the spec so the comment threads resolve to a person, not a handle.", ["F1QPRDSK"]),
    ("C0ENG", "bharat", "Index sharding v1.8.0 is on staging. Canary at 1% then full deploy once p95 holds under 800ms.", []),
    ("C0ENG", "diego", "Retrieval eval harness is ready. 412 labelled queries. precision@10 is 0.86 on the lexical baseline, 0.88 with the hybrid rerank.", []),
    ("C0ENG", "bharat", "0.88 is above the 0.85 target from the blueprint. Nice. Can you send me the per-source breakdown, I expect Slack to be the weak one.", []),
    ("C0ENG", "diego", "Per-source: gmail 0.93, slack 0.71, whatsapp 0.84, notion 0.95. Slack loses on short messages with lots of @mentions and no subject line.", []),
    ("C0ENG", "bharat", "That is fixable with a title field. Take the first sentence of a thread root as a synthetic subject. Deadline: Mar 3 for the eval rerun.", []),
    ("C0INCIDENT", "bharat", "SEV-2 open. p95 search went 640ms to 9.4s at 21:04 UTC after the sharding deploy. Rolling back now.", []),
    ("C0INCIDENT", "diego", "Confirmed from the dashboard. One shard is taking 71% of the read traffic.", []),
    ("C0INCIDENT", "alice", "How many customers were affected and can we post a status page entry before 22:00?", []),
    ("C0INCIDENT", "bharat", "2,140 failed requests, 38 workspaces. Status page draft is in the postmortem doc. Target resolution: 22:15 UTC.", ["F1INCIDENT"]),
    ("C0INCIDENT", "diego", "Root cause is the shard key. We sharded by document id, every hot query hit one shard. Same mistake as the talk I watched on scaling search.", []),
    ("C0RANDOM", "clara", "The office coffee machine is broken again. Third time this month. I have started a sign-up sheet, add your name.", []),
    ("C0RANDOM", "diego", "Add me. Also whoever is in charge of the fridge: there is a container of unlabeled soup and I am not gambling.", []),
    ("D0ALICE", "bharat", "Do you have five minutes? I want to walk you through the rebalancer before the Series A narrative gets locked.", []),
    ("D0ALICE", "alice", "Sure. Send me the deck, I will read it tonight. Can you also drop the numbers in writing so the diligence room has a source?", ["F1DECK"]),
    ("D0ALICE", "bharat", "Deck attached. The burn multiple number is the one to be careful about, I normalised it two different ways in the appendix.", ["F1DECK"]),
    ("D0CLARA", "clara", "One more thing on the design review deck: the graph view needs a legend that explains derived versus observed edges. Users will misread it otherwise.", []),
    ("D0CLARA", "alice", "Good catch. Derived edges get a dashed style and a hover tooltip explaining the inference.", []),
    ("D0CLARA", "clara", "Perfect. I will have the updated mockups by Friday.", ["F1QPRDSK"]),
    ("D0BHARAT", "bharat", "I need your read on the retention policy before the privacy review. alice.chen@northwind.io is the escalation contact.", []),
    ("D0BHARAT", "alice", "Send it over. Raw payloads at 90 days is right, but the audit log at 12 months needs a legal opinion, I am not signing off on that alone.", []),
]

GMAIL_MESSAGES: list[dict[str, Any]] = [
    {
        "from": "alice", "subject": "Series A narrative - need your numbers by Friday",
        "body": """Bharat,

I am locking the Series A narrative v7 tonight and I need the infrastructure
numbers reconciled. Specifically:

- p95 search latency, before and after sharding
- ingest success rate over the last 30 days
- cost per 1M documents indexed

The diligence room will ask for sources, so please send the query or the
screenshot, not just the number. Deadline is Friday EOD.

Deck: Series-A-narrative-v7.key

Thanks,
Alice Chen
Head of Product, northwind.io
+44 7700 900123
""",
        "hours_ago": 30,
        "label": ["INBOX", "IMPORTANT"],
        "attachments": [FILES[5]],
        "thread": "thr-series-a",
    },
    {
        "from": "bharat", "subject": "Re: Series A narrative - need your numbers by Friday",
        "body": """Here is the reconciled set.

p95 latency: 640ms pre-sharding, 640ms post-rollback. The 9.4s figure was the
incident window only and we should not present it as steady state.
Ingest success: 99.2% over 30 days (failures are almost all 403 from revoked
Google Drive grants).
Cost: 4.10 USD per 1M documents at current embedding settings.

If vector search stays behind a flag, say so explicitly. Investors have started
asking whether we are a search company, and "we are building a data index" is a
weaker answer than "we ship lexical first".

Bharat Iyer
Staff Engineer
+91 98765 43210
""",
        "hours_ago": 22,
        "label": ["SENT"],
        "attachments": [],
        "thread": "thr-series-a",
    },
    {
        "from": "diego", "subject": "Retrieval eval: per-source breakdown (precision@10 = 0.88)",
        "body": """Hi all,

Ran the 412-query eval set against the hybrid baseline. Headline precision@10
is 0.88, above the 0.85 target.

Per source:
  gmail    0.93   (subject lines give a huge free boost)
  notion   0.95
  whatsapp 0.84
  slack    0.71   <- the problem

Slack is weak because messages have no subject and are mostly @mentions and
short replies. I propose a synthetic subject from the thread root's first
sentence. I can have the rerun done by Mar 3.

Raw results: Retention-cohorts-Q3.csv is unrelated, ignore that, the real file
is coming. Reach me at diego.fuentes@brightfold.com or +34600111222.

Diego
""",
        "hours_ago": 50,
        "label": ["INBOX"],
        "attachments": [],
        "thread": "thr-eval",
    },
    {
        "from": "clara", "subject": "Design review Thursday - need final copy Wednesday 12:00",
        "body": """Hi,

Design review moved to Thursday 15:00. I need final copy by Wednesday 12:00.

Two things I need decisions on:
1. Does the graph view get a legend distinguishing observed from derived edges?
2. Are we showing file contents inline, or metadata only with a deep link?

If we show inline we own redaction for every provider, which is a week of work
we do not have before the review. My recommendation: metadata plus deep link.

Clara Nowak
Brightfold
+48123456789
""",
        "hours_ago": 74,
        "label": ["INBOX"],
        "attachments": [FILES[0]],
        "thread": "thr-design",
    },
    {
        "from": "clara", "subject": "Re: Design review Thursday - need final copy Wednesday 12:00",
        "body": """Also attaching the last two decks we showed investors, they will ask why the
dashboard changed. Q3-Product-Review.pdf is the one they saw.

Clara
""",
        "hours_ago": 70,
        "label": ["INBOX"],
        "attachments": [FILES[0]],
        "thread": "thr-design",
    },
    {
        "from": "bharat", "subject": "Postmortem draft - SEV-2 4417",
        "body": """Draft attached, comments inline please.

The part I want scrutiny on is the shard key decision. We sharded by document
id because it was the obvious default, not because the access pattern supported
it. Every read-heavy query collapses onto one shard.

Action items with owners are at the bottom. I have not assigned the runbook
task yet.

Bharat
""",
        "hours_ago": 100,
        "label": ["SENT"],
        "attachments": [FILES[3]],
        "thread": "thr-pm",
    },
    {
        "from": "alice", "subject": "Brightfold MSA countersigned",
        "body": """Both signatures are in. Brightfold-MSA-signed.pdf is the final version.

Liability capped at 12 months of fees, data residency in eu-west-2 confirmed,
termination for convenience at 60 days notice in their favour.

Net: 40,000 EUR ARR starting next quarter. Not the number we needed but it
closes the diligence gap on enterprise readiness.

Alice
""",
        "hours_ago": 200,
        "label": ["INBOX"],
        "attachments": [FILES[4]],
        "thread": "thr-msa",
    },
    {
        "from": "alice", "subject": "FY27 budget - v4",
        "body": """v4 attached. Changes from v3: infra line up 18% (search is not free at
scale), contractor line down 30%.

I need sign-off by end of month. Reply with objections, silence is not consent.

Alice
""",
        "hours_ago": 320,
        "label": ["INBOX"],
        "attachments": [FILES[1]],
        "thread": "thr-budget",
    },
]

WHATSAPP_CHAT = """[02/09/2026, 08:14] Alice: Are we still doing the offsite on the 18th? Clara says the venue wants a headcount today
[02/09/2026, 08:19] Clara: Yes, 18th. I need the headcount by EOD today or we lose the room.
[02/09/2026, 08:21] Alice: Put me down for 2. Bharat is remote that week so 1 for him.
[02/09/2026, 08:24] Clara: <attached: 0001-PHOTO-2026-09-02.jpg>
[02/09/2026, 08:31] Bharat: Can I get the agenda before Thursday? I fly Friday morning.
[04/09/2026, 19:02] Clara: Agenda attached. 09:30 start, dinner booked for 19:30.
[04/09/2026, 19:40] Alice: Perfect. Add Clara Nowak to the invite, my calendar only has her personal number.
[11/09/2026, 22:17] Bharat: Heads up - the search index is slow again, dashboard is timing out on every query. Is this the same shard problem?
[11/09/2026, 22:19] Alice: Different issue. Bharat is looking at it.
[11/09/2026, 22:20] Bharat: Same shard problem. Rolling back, should be fine by 23:00.
[11/09/2026, 23:04] Bharat: Recovered. p95 is back under a second. Postmortem incoming.
[11/09/2026, 23:11] Alice: <attached: 0002-SEV2-postmortem-4417.pdf>
[18/09/2026, 12:30] Clara: Offsite photos are up. It went well, thank you both.
[18/09/2026, 12:34] Bharat: Agreed. Best offsite we have had.
[20/09/2026, 20:15] Alice: One more thing before the fundraise - I need everyone to look at the data retention policy before Friday. Legal will ask.
[24/09/2026, 09:22] Clara: Read it. Two questions, both in the doc. Also the escalation contact should be a role not a personal email.
"""

SYSTEM_LINES = [
    "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
    "Alice created this group",
    "You added Bharat",
]


# ---------------------------------------------------------------------------
# Golden fixtures for the remaining connector families
# ---------------------------------------------------------------------------
#
# Blueprint 15.1 requires a contract test for *every* connector, and a skipped
# test is not a contract test. These payloads are the real API shapes for
# Discord, OneDrive, Dropbox and Evernote, so the same normalizers and the same
# assertions cover all eleven connectors without credentials.
#
# They are deliberately small. Their job is to exercise the shape of each
# provider's schema - the discriminator, the nested container, the id format -
# not to add narrative to the workspace.

DISCORD_LINES: list[tuple[str, str, str, str]] = [
    # (author_key, content, channel_label, guild)
    ("bharat", "Standup notes are in the doc. Blocking issue: the rebalancer runbook "
               "is still unassigned.", "general", "Northwind"),
    ("clara", "Reviewed the deck. The graph legend needs a dashed edge style for "
              "machine-suggested links.", "design", "Northwind"),
    ("alice", "Confirmed: we ship the search spec Thursday. Deadline is Mar 3 for "
              "the eval rerun.", "general", "Northwind"),
]

ONEDRIVE_FILES: list[tuple[str, str, str, int]] = [
    ("onedrive:clara", "Q4-Okr-Proposal.docx",
     "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 88_410),
    ("onedrive:bharat", "Cost-per-Document.csv", "text/csv", 15_902),
    ("onedrive:self", "Meeting-Notes-2026-09-24.md", "text/markdown", 6_120),
]

DROPBOX_FILES: list[tuple[str, str, int]] = [
    ("dropbox:me", "/eng/runbooks/Manual-Rebalance.md", 11_240),
    ("dropbox:me", "/legal/Brightfold-DPA-v3.pdf", 402_117),
    ("dropbox:clara", "/design/Series-A-Narrative.key", 8_442_910),
]

EVERNOTE_NOTES: list[tuple[str, str, str]] = [
    ("evernote:alice", "Retention policy follow-ups",
     "Legal wants an answer on the 12-month audit log window.\n\n"
     "- confirm with outside counsel\n"
     "- decide whether derived stores count as a copy\n"
     "- write the escalation path as a role, not a person"),
    ("evernote:bharat", "Shard key postmortem notes",
     "We sharded by document id because it was the obvious default.\n\n"
     "Every read-heavy query collapsed onto one shard for 71 minutes. "
     "The rebalancer had never been enabled."),
]


def _person(key: str) -> dict[str, Any]:
    return next(p for p in PEOPLE if p["key"] == key)


def _slack_file(file_id: str) -> dict[str, Any]:
    """Slack-shaped attachment for one of the FILES entries."""
    fid, name, mime, size, _path = next(f for f in FILES if f[0] == file_id)
    return {
        "id": fid,
        "name": name,
        "mimetype": mime,
        "size": size,
        "url_private": f"https://files.slack.local/{fid}/{name}",
        "permalink": f"https://northwind.slack.com/files/{fid}",
    }


def _slack_ts(dt: datetime) -> str:
    return f"{dt.timestamp():.6f}"


class DemoConnector(Connector):
    """Synthetic multi-source provider. No network, no credentials, no side effects."""

    descriptor = ConnectorDescriptor(
        id="demo",
        display_name="Demo workspace (synthetic)",
        version="1.0.0",
        auth_flow="none",
        base_url="mem://",
        docs_url="internal://demo",
        scopes=(
            Scope("synthetic", "Generate a deterministic local dataset for demos and tests",
                  "normal"),
        ),
        realtime_webhooks=False,
        incremental_cursor=True,
        cursor_field="index",
        supports_reactions=True,
        supports_edits=False,
        content_kinds=("message", "email", "file", "note", "video", "transcript", "identity"),
        rate_limit_per_sec=1000.0,
        rate_limit_burst=1000,
        backfill_strategy="batch",
        notes="Deterministic from SEED. Used as the seed data set and as the golden "
              "corpus for connector contract tests.",
    )

    def __init__(self, *, seed: int = SEED, now: datetime = NOW) -> None:
        self.seed = seed
        self.now = now
        self.rng = random.Random(seed)

    # -- auth ---------------------------------------------------------
    def authorize(self, req: Any) -> Any:  # pragma: no cover - not applicable
        from omnilinker.connectors.contract import AuthorizationResult

        return AuthorizationResult(authorization_url="mem://demo/authorized", state=req.state)

    def exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> TokenGrant:
        return TokenGrant(access_token="demo-token", expires_at=None)

    # -- streams ------------------------------------------------------
    def discover_streams(self, grant: TokenGrant | None = None,
                         user_ref: str | None = None) -> list[StreamDescriptor]:
        streams: list[StreamDescriptor] = []
        for ch in CHANNELS:
            streams.append(StreamDescriptor(key=ch["id"], kind="slack_conversation",
                                            label=f"#{ch['name']}",
                                            meta={"channel": ch}))
        streams.append(StreamDescriptor(key="gmail:me@example.com", kind="mailbox",
                                        label="me@example.com", meta={}))
        streams.append(StreamDescriptor(key="wa:Offsite Planning", kind="whatsapp_chat",
                                        label="Offsite Planning", meta={}))
        streams.append(StreamDescriptor(key="notion:workspace", kind="notion_workspace",
                                        label="Notion", meta={}))
        streams.append(StreamDescriptor(key="gdrive:root", kind="drive", label="My Drive",
                                        meta={}))
        streams.append(StreamDescriptor(key="youtube:watchlist", kind="playlist",
                                        label="Watch later", meta={}))
        streams.append(StreamDescriptor(key="discord:guild", kind="discord_guild",
                                        label="Northwind", meta={}))
        streams.append(StreamDescriptor(key="onedrive:root", kind="drive",
                                        label="OneDrive", meta={}))
        streams.append(StreamDescriptor(key="dropbox:root", kind="drive",
                                        label="Dropbox", meta={}))
        streams.append(StreamDescriptor(key="evernote:notebook", kind="notebook",
                                        label="Inbox", meta={}))
        return streams

    # -- pull ---------------------------------------------------------
    def pull(self, grant: TokenGrant | None, stream: StreamDescriptor, cursor: str | None,
             page: int = 0) -> PullPage:
        start = int(cursor) if cursor and cursor.isdigit() else 0
        envelopes = list(self._stream_envelopes(stream))
        batch = envelopes[start: start + 500]
        return PullPage(records=batch, next_cursor=str(start + len(batch)),
                        has_more=start + len(batch) < len(envelopes))

    def _stream_envelopes(self, stream: StreamDescriptor) -> Iterator[RawEnvelope]:
        kind = stream.kind
        if kind == "slack_conversation":
            yield from self._slack_envelopes(stream)
        elif kind == "mailbox":
            yield from self._gmail_envelopes()
        elif kind == "whatsapp_chat":
            yield from self._whatsapp_envelopes()
        elif kind == "notion_workspace":
            yield from self._notion_envelopes()
        elif kind == "playlist":
            yield from self._youtube_envelopes()
        elif kind == "discord_guild":
            yield from self._discord_envelopes()
        elif kind == "notebook":
            yield from self._evernote_envelopes()
        elif kind == "drive":
            # Three providers share the "drive" stream kind and their payload
            # shapes are nothing alike, so route on the key prefix explicitly.
            # A single `else` fallback here silently fed Google Drive envelopes
            # to the OneDrive and Dropbox normalizers - the kind of routing bug
            # that produces a plausible-looking workspace with two empty
            # sources in it.
            prefix = stream.key.split(":", 1)[0]
            if prefix == "onedrive":
                yield from self._onedrive_envelopes()
            elif prefix == "dropbox":
                yield from self._dropbox_envelopes()
            else:
                yield from self._drive_envelopes()

    # -- per-source envelopes ----------------------------------------
    def _slack_envelopes(self, stream: StreamDescriptor) -> Iterator[RawEnvelope]:
        """Messages for one channel/DM.

        Each Slack message belongs to exactly one conversation, so a stream
        returns only its own messages - matching the real API, where
        `conversations.history` is per channel.
        """
        channel_id = stream.key
        for i, (chan_id, sender_key, body, atts) in enumerate(SLACK_LINES):
            if chan_id != channel_id:
                continue
            chan = next(c for c in CHANNELS if c["id"] == chan_id)
            # Timestamp is derived from the line index, not a running
            # accumulator, so it is globally unique and order-independent.
            ts = self.now - timedelta(days=45) + timedelta(hours=7 * i)
            person = _person(sender_key)
            display = CLARA_SLACK_NAME if sender_key == "clara" else person["name"].split()[0].lower()
            payload = {
                "ts": _slack_ts(ts),
                "user": person["slack_id"],
                "text": body,
                "thread_ts": _slack_ts(ts - timedelta(hours=1)) if "Re:" in body else _slack_ts(ts),
                "_stream_label": chan["name"],
                "_channel_type": chan["type"],
                "_display_name": display,
                # Slack's `users.info` returns the workspace email unless the
                # user has restricted it. Modelling that faithfully is what
                # makes the demo a real identity-resolution test: without it,
                # Slack and Gmail accounts for the same human share nothing but
                # a first name, which is not evidence.
                "_user_profile": {
                    "id": person["slack_id"],
                    "name": display,
                    "real_name": person["name"],
                    "email": person["gmail"],
                    "phone": person.get("phone", ""),
                },
                "reactions": ([{"name": "eyes", "count": 2, "users": ["U02BHARAT", "U04DIEGO"]}]
                              if i % 5 == 0 else []),
                "files": [_slack_file(fid) for fid in atts],
                "_is_self": person.get("is_self", False),
            }
            yield RawEnvelope(
                source_artifact_id=payload["ts"],
                provider="slack",
                stream=chan_id,
                kind="message",
                payload=payload,
                provider_meta={"ts": payload["ts"]},
                fetched_at=iso(),
                workspace_id="",
            )

    def _gmail_envelopes(self) -> Iterator[RawEnvelope]:
        for i, m in enumerate(GMAIL_MESSAGES):
            sender = _person(m["from"])
            ts = self.now - timedelta(hours=m["hours_ago"])
            payload = {
                "id": f"gmail-msg-{i:03d}",
                "threadId": m["thread"],
                "historyId": f"hist-{i:03d}",
                "labelIds": m["label"],
                "internalDate": str(int(ts.timestamp() * 1000)),
                "snippet": m["body"].strip().splitlines()[0][:120],
                "payload": {
                    "mimeType": "multipart/mixed",
                    "headers": [
                        {"name": "Subject", "value": m["subject"]},
                        {"name": "From", "value": f"{sender['name']} <{sender['gmail']}>"},
                        {"name": "To", "value": "me@example.com"},
                        {"name": "Cc", "value": ", ".join(p["gmail"] for p in PEOPLE
                                                          if p["key"] != m["from"])},
                        {"name": "Message-Id", "value": f"<{m['thread']}@northwind.io>"},
                        {"name": "Date", "value": iso(ts)},
                    ],
                    "parts": [
                        {"mimeType": "text/plain", "body": {
                            "data": _b64(m["body"]), "size": len(m["body"])}},
                    ] + [
                        {
                            "mimeType": mime,
                            "filename": name,
                            "body": {"attachmentId": f"att-{fid}", "size": size},
                        }
                        for (fid, name, mime, size, _p) in m["attachments"]
                    ],
                },
            }
            yield RawEnvelope(
                source_artifact_id=payload["id"],
                provider="gmail",
                stream="me@example.com",
                kind="email",
                payload=payload,
                provider_meta={"threadId": m["thread"], "labelIds": m["label"],
                               "historyId": f"hist-{i:03d}"},
                fetched_at=iso(),
                workspace_id="",
            )

    def _whatsapp_envelopes(self) -> Iterator[RawEnvelope]:
        from omnilinker.connectors.providers.whatsapp import parse_chat_text

        messages = parse_chat_text(WHATSAPP_CHAT, "Offsite Planning")
        for m in messages:
            yield RawEnvelope(
                source_artifact_id=f"wa:Offsite Planning:{m['index']}",
                provider="whatsapp",
                stream="wa:Offsite Planning",
                kind="message",
                payload={**m, "_chat_label": "Offsite Planning", "_media_dir": ""},
                provider_meta={"index": m["index"]},
                fetched_at=iso(),
                workspace_id="",
            )

    def _notion_envelopes(self) -> Iterator[RawEnvelope]:
        for note in NOTES:
            blocks = _markdown_to_blocks(note["body"])
            yield RawEnvelope(
                source_artifact_id=note["page_id"],
                provider="notion",
                stream="notion:workspace",
                kind="note",
                payload={
                    "id": note["page_id"],
                    "object": "page",
                    "created_time": note["created"],
                    "last_edited_time": note["edited"],
                    "url": f"https://notion.so/{note['page_id'].replace('-', '')}",
                    "parent": {"page_id": note["parent"]},
                    "icon": {"emoji": _icon_for(note["page_id"])},
                    "properties": {
                        "title": {"type": "title",
                                  "title": [{"plain_text": note["title"]}]}
                    },
                    "_blocks": blocks,
                },
                provider_meta={"last_edited_time": note["edited"]},
                fetched_at=iso(),
                workspace_id="",
            )

    def _drive_envelopes(self) -> Iterator[RawEnvelope]:
        owners = ["alice", "bharat", "clara", "diego", "self"]
        for i, (fid, name, mime, size, path) in enumerate(FILES):
            owner = _person(owners[i % len(owners)])
            ts = self.now - timedelta(days=10 + i * 7)
            yield RawEnvelope(
                source_artifact_id=fid,
                provider="gdrive",
                stream="root",
                kind="file",
                payload={
                    "id": fid,
                    "name": name,
                    "mimeType": mime,
                    "size": str(size),
                    "modifiedTime": iso(ts),
                    "createdTime": iso(ts - timedelta(days=3)),
                    "parents": [f"FOLDER{i}"],
                    "owners": [{"emailAddress": owner["gmail"],
                                "displayName": owner["name"]}],
                    "shared": i % 2 == 0,
                    "md5Checksum": content_hash(fid)[7:39],
                    "webViewLink": f"https://drive.google.com/file/d/{fid}/view",
                    "trashed": False,
                },
                provider_meta={"parents": [f"FOLDER{i}"], "shared": i % 2 == 0},
                fetched_at=iso(),
                workspace_id="",
            )

    def _youtube_envelopes(self) -> Iterator[RawEnvelope]:
        for v in VIDEOS:
            srt = _captions_to_srt(v["captions"])
            yield RawEnvelope(
                source_artifact_id=v["video_id"],
                provider="youtube",
                stream="youtube:watchlist",
                kind="video",
                payload={
                    "id": {"videoId": v["video_id"]},
                    "snippet": {
                        "title": v["title"],
                        "description": f"{v['title']} - {v['channel_title']}",
                        "channelId": v["channel_title"].replace(" ", "-").lower(),
                        "channelTitle": v["channel_title"],
                        "publishedAt": v["published"],
                        "duration": f"PT{int(v['duration_s'] // 60)}M{v['duration_s'] % 60}S",
                        "tags": v["tags"],
                        "thumbnails": {"default": {}, "high": {}},
                    },
                    "_playlist_label": "Watch later",
                    "_caption_srt": srt,
                    "_caption_lang": "en",
                },
                fetched_at=iso(),
                workspace_id="",
            )

    def _discord_envelopes(self) -> Iterator[RawEnvelope]:
        for i, (author_key, content, channel, guild) in enumerate(DISCORD_LINES):
            person = _person(author_key)
            when = self.now - timedelta(days=30 - i * 3)
            # A Discord snowflake is `(ms since the Discord epoch) << 22`, and
            # `snowflake_ts` adds the epoch back. Storing a raw offset here
            # dates every message to 1970 - which looks fine in a fixture and
            # silently wrecks the timeline.
            offset_ms = int((when - datetime(2015, 1, 1, tzinfo=timezone.utc))
                            .total_seconds() * 1000)
            snowflake = str((offset_ms << 22) + i + 1)
            yield RawEnvelope(
                source_artifact_id=snowflake,
                provider="discord",
                stream="discord:guild",
                kind="message",
                payload={
                    "id": snowflake,
                    "type": 0,
                    "content": content,
                    "author": {"id": person["discord_id"],
                               "username": person["name"].split()[0].lower(),
                               "global_name": person["name"]},
                    "timestamp": iso(when),
                    "_stream_label": channel,
                    "_guild_name": guild,
                    "message_reference": {"message_id": snowflake} if i else None,
                    "reactions": ([{"emoji": {"name": "eyes"}, "count": 1,
                                    "user_id": person["discord_id"]}] if i == 1 else []),
                },
                provider_meta={},
                fetched_at=iso(),
                workspace_id="",
            )

    def _onedrive_envelopes(self) -> Iterator[RawEnvelope]:
        for i, (owner_ref, name, mime, size) in enumerate(ONEDRIVE_FILES):
            when = self.now - timedelta(days=20 + i * 4)
            yield RawEnvelope(
                source_artifact_id=f"od-{i:03d}",
                provider="onedrive",
                stream="onedrive:root",
                kind="file",
                payload={
                    "id": f"od-{i:03d}",
                    "name": name,
                    "size": size,
                    "eTag": f'"{{{i:08x}}}"',
                    "lastModifiedDateTime": iso(when),
                    "createdDateTime": iso(when - timedelta(days=2)),
                    "parentReference": {"id": f"root-{i}", "driveId": "drive-1"},
                    "file": {"mimeType": mime},
                    "folder": None,
                    "@microsoft.graph.downloadUrl":
                        f"https://graph.microsoft.com/v1.0/drives/1/items/od-{i:03d}/content",
                    "shared": {"scope": "users" if i == 0 else None},
                },
                provider_meta={},
                fetched_at=iso(),
                workspace_id="",
            )

    def _dropbox_envelopes(self) -> Iterator[RawEnvelope]:
        for i, (owner_ref, path, size) in enumerate(DROPBOX_FILES):
            when = self.now - timedelta(days=14 + i * 5)
            yield RawEnvelope(
                source_artifact_id=f"id:dbx{i:03d}",
                provider="dropbox",
                stream="dropbox:root",
                kind="file",
                payload={
                    ".tag": "file",
                    "id": f"id:dbx{i:03d}",
                    "name": path.rsplit("/", 1)[-1],
                    "path_display": path,
                    "path_lower": path.lower(),
                    "size": size,
                    "client_modified": iso(when),
                    "server_modified": iso(when + timedelta(minutes=2)),
                    "rev": f"{i:016x}",
                    "is_downloadable": True,
                    "sharing_info": {"read_only": False},
                },
                provider_meta={},
                fetched_at=iso(),
                workspace_id="",
            )

    def _evernote_envelopes(self) -> Iterator[RawEnvelope]:
        for i, (author_ref, title, body) in enumerate(EVERNOTE_NOTES):
            when = self.now - timedelta(days=25 + i * 7)
            yield RawEnvelope(
                source_artifact_id=f"en-{i:03d}",
                provider="evernote",
                stream="evernote:notebook",
                kind="note",
                payload={
                    "id": f"en-{i:03d}",
                    "title": title,
                    # Evernote stores ENML, not markdown. The extra newline
                    # escapes below are what the live API actually returns.
                    "content": "<en-note><div><br></div>"
                               + body.replace("\n", "<br>\n")
                               + "</en-note>",
                    "created": int((when - timedelta(days=1)).timestamp() * 1000),
                    "updated": int(when.timestamp() * 1000),
                    "noteAttributes": {"created": int(when.timestamp() * 1000)},
                    "tags": ["follow-up", "policy"] if i == 0 else ["postmortem"],
                    "notebookGuid": "nb-1",
                },
                provider_meta={},
                fetched_at=iso(),
                workspace_id="",
            )

    # -- normalize ----------------------------------------------------
    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        """Dispatch to the real provider normalizer.

        This is deliberate: the demo produces *provider-shaped* payloads and
        runs them through the *actual* Slack/Gmail/Notion/YouTube normalizers.
        So demo data exercises production code paths, and the normalizers are
        covered by the same tests. The only exception is WhatsApp, which needs a
        file, so it is fed the pre-parsed message dicts.
        """
        if raw.provider == "whatsapp":
            from omnilinker.connectors.providers.whatsapp import WhatsAppConnector

            return WhatsAppConnector().normalize(raw)

        # The envelope already carries the provider-shaped payload, including
        # Slack's user profile, so there is nothing to patch here. An earlier
        # version synthesised the profile at normalize() time, which meant the
        # demo could never exercise a profile the *provider* supplied - and the
        # workspace email it carries is precisely what identity resolution needs.
        return _DEMO_NORMALIZERS[raw.provider]().normalize(raw)


# ---------------------------------------------------------------------------
# Normalizer registry used by the demo path
# ---------------------------------------------------------------------------

def _slack_normalizer():
    from omnilinker.connectors.providers.slack import SlackConnector

    return SlackConnector()


def _gmail_normalizer():
    from omnilinker.connectors.providers.gmail import GmailConnector

    return GmailConnector()


def _notion_normalizer():
    from omnilinker.connectors.providers.notes import NotionConnector

    return NotionConnector()


def _gdrive_normalizer():
    from omnilinker.connectors.providers.cloud import GoogleDriveConnector

    return GoogleDriveConnector()


def _youtube_normalizer():
    from omnilinker.connectors.providers.youtube import YouTubeConnector

    return YouTubeConnector()


def _discord_normalizer():
    from omnilinker.connectors.providers.discord import DiscordConnector

    return DiscordConnector()


def _onedrive_normalizer():
    from omnilinker.connectors.providers.cloud import OneDriveConnector

    return OneDriveConnector()


def _dropbox_normalizer():
    from omnilinker.connectors.providers.cloud import DropboxConnector

    return DropboxConnector()


def _evernote_normalizer():
    from omnilinker.connectors.providers.notes import EvernoteConnector

    return EvernoteConnector()


_DEMO_NORMALIZERS = {
    "slack": _slack_normalizer,
    "gmail": _gmail_normalizer,
    "discord": _discord_normalizer,
    "whatsapp": None,  # handled separately: it needs a pre-parsed export
    "gdrive": _gdrive_normalizer,
    "onedrive": _onedrive_normalizer,
    "dropbox": _dropbox_normalizer,
    "notion": _notion_normalizer,
    "evernote": _evernote_normalizer,
    "youtube": _youtube_normalizer,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    import base64

    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _markdown_to_blocks(md: str) -> list[dict]:
    """Markdown -> Notion block objects.

    Note the `type` discriminator alongside the body key: that is the shape
    `GET /blocks/{id}/children` actually returns, and the normalizer keys off
    it. Emitting `{"paragraph": {...}}` without it silently produces an empty
    document - a bug this demo earned the hard way.
    """
    blocks: list[dict] = []
    for i, line in enumerate(md.splitlines()):
        if not line.strip():
            continue
        checked = False
        if line.startswith("### "):
            btype, text = "heading_3", line[4:]
        elif line.startswith("## "):
            btype, text = "heading_2", line[3:]
        elif line.startswith("# "):
            btype, text = "heading_1", line[2:]
        elif line.startswith("- [x] "):
            btype, text, checked = "to_do", line[6:], True
        elif line.startswith("- [ ] "):
            btype, text = "to_do", line[6:]
        elif line.startswith("- "):
            btype, text = "bulleted_list_item", line[2:]
        elif line[0].isdigit() and ". " in line[:4]:
            btype, text = "numbered_list_item", line.split(". ", 1)[1]
        else:
            btype, text = "paragraph", line

        body: dict[str, Any] = {
            "rich_text": [{"type": "text", "plain_text": text,
                           "annotations": {"bold": False, "italic": False}}],
            "color": "default",
        }
        if btype == "to_do":
            body["checked"] = checked
        if btype.startswith("heading_"):
            body["is_toggleable"] = False
        blocks.append({
            "object": "block",
            "id": f"blk-{i:03d}",
            "type": btype,
            "has_children": False,
            btype: body,
        })
    return blocks


def _captions_to_srt(captions: list[tuple[int, int, str]]) -> str:
    def ts(seconds: int) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d},000"

    out: list[str] = []
    for i, (start, end, text) in enumerate(captions, start=1):
        out.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(out)


def _icon_for(page_id: str) -> str:
    return {
        "notion-product-spec": "🔎",
        "notion-incident-4417": "🚨",
        "notion-hiring": "🧑‍💼",
        "notion-privacy": "🔒",
    }.get(page_id, "📄")
