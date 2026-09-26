"""Connector contract (blueprint 6.1).

This is the single most important abstraction in the system: the core never
learns a provider's schema. Every provider-specific quirk dies inside a
connector package; everything downstream sees `CanonicalRecord`.

Four capabilities, exactly as the blueprint specifies:
    descriptor()          -> ConnectorDescriptor
    authorize(req)        -> AuthorizationResult
    refresh_tokens(grant)  -> TokenGrant
    discover_streams(u)   -> [StreamDescriptor]
    pull(u, stream, cursor, page) -> PullPage
    normalize(raw)        -> [CanonicalRecord]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """A requested permission with a written justification.

    The connector registry (6.2.1 REGISTER gate) refuses any scope whose
    justification is empty - that is a code-review-enforced privacy control,
    not a convention.
    """

    name: str
    justification: str
    sensitivity: str = "normal"  # normal | sensitive | restricted

    def __post_init__(self) -> None:
        if not self.justification.strip():
            raise ValueError(
                f"scope {self.name!r} has no justification; the registry will reject this "
                "connector (blueprint 6.2.1 REGISTER static checks)"
            )


@dataclass(frozen=True)
class ConnectorDescriptor:
    id: str
    display_name: str
    version: str
    auth_flow: str  # oauth2_code | oauth2_device | export_file | api_key
    authorize_url: str | None = None
    token_url: str | None = None
    refresh_url: str | None = None
    base_url: str = ""
    scopes: tuple[Scope, ...] = ()
    pkce_required: bool = True
    docs_url: str = ""
    # capabilities
    realtime_webhooks: bool = False
    incremental_cursor: bool = True
    cursor_field: str | None = None
    supports_tombstones: bool = False
    supports_edits: bool = True
    supports_reactions: bool = False
    supports_threads: bool = True
    supports_attachments: bool = True
    content_kinds: tuple[str, ...] = ("message",)
    rate_limit_per_sec: float = 5.0
    rate_limit_burst: int = 10
    max_page_size: int = 100
    backfill_strategy: str = "incremental"  # incremental | batch | reconcile
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scopes"] = [asdict(s) for s in self.scopes]
        d["scopes"] = [{"name": s.name, "justification": s.justification,
                        "sensitivity": s.sensitivity} for s in self.scopes]
        return d


@dataclass
class TokenGrant:
    """Sealed tokens. `access_token` is never logged, never returned by the API."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    scopes: tuple[str, ...] = ()
    token_type: str = "Bearer"
    extra: dict[str, Any] = field(default_factory=dict)

    def needs_refresh(self, skew: float = 120.0) -> bool:
        return self.expires_at is not None and self.expires_at - skew < _now()


@dataclass
class AuthorizationRequest:
    workspace: str
    redirect_uri: str
    state: str
    code_verifier: str = ""
    scopes: Sequence[str] | None = None
    login_hint: str | None = None


@dataclass
class AuthorizationResult:
    authorization_url: str
    state: str
    expires_in: int = 600


@dataclass
class StreamDescriptor:
    """A logical sub-stream. One connector exposes many (per channel, per
    notebook, per folder). Cursors are per (workspace, source, stream)."""

    key: str
    kind: str
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawEnvelope:
    """Immutable provider payload plus fetch metadata (blueprint 6.1)."""

    source_artifact_id: str
    provider: str
    stream: str
    kind: str
    payload: Any
    provider_meta: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""
    #: Tenant boundary. Carried on every envelope so the pipeline never has to
    #: thread it separately, and so a multi-tenant connector cannot leak.
    workspace_id: str = ""


@dataclass
class PullPage:
    records: list[RawEnvelope] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False
    quota: dict[str, Any] = field(default_factory=dict)
    partial_failure: dict[str, Any] | None = None


@dataclass
class CanonicalRecord:
    """Provider-independent. `kind` selects the target collection and the
    graph writer's node type."""

    kind: str  # message | email | file | note | event | video | identity | membership
    data: dict[str, Any]
    source_artifact_id: str
    provider: str
    stream: str = ""


def _now() -> float:
    import time

    return time.time()


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class Connector(ABC):
    """Every connector implements this. Nothing else in the codebase knows
    which provider it is talking to.

    Only three methods are abstract - `discover_streams`, `pull`, `normalize` -
    because those are the ones that exist for every auth flow including
    file-based ones. The OAuth methods have defaults that raise, because a
    WhatsApp export has no authorization URL and forcing every file connector
    to write a stub that says "not applicable" is noise.
    """

    descriptor: ConnectorDescriptor

    # -- required -----------------------------------------------------
    @abstractmethod
    def discover_streams(self, grant: TokenGrant | None = None,
                         user_ref: str | None = None) -> list[StreamDescriptor]:
        """What can be synced, and what identifies each sub-stream."""

    @abstractmethod
    def pull(
        self, grant: TokenGrant | None, stream: StreamDescriptor, cursor: str | None, page: int = 0
    ) -> PullPage:
        """Fetch one page. Must be idempotent and resumable from `cursor`."""

    @abstractmethod
    def normalize(self, raw: RawEnvelope) -> list[CanonicalRecord]:
        """Provider payload -> canonical records. Pure function, no I/O.

        Contract test requirement: this is the only method that needs a golden
        payload corpus (blueprint 15.1), because it is the only place provider
        schema knowledge exists.
        """

    # -- OAuth family (default: not applicable) ----------------------
    def authorize(self, req: AuthorizationRequest) -> AuthorizationResult:
        raise NotImplementedError(
            f"{self.descriptor.id}: auth_flow={self.descriptor.auth_flow} has no "
            "authorization step (file/API-key connectors do not use one)"
        )

    def exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> TokenGrant:
        raise NotImplementedError(
            f"{self.descriptor.id}: auth_flow={self.descriptor.auth_flow} does not "
            "exchange an authorization code"
        )

    def refresh_tokens(self, grant: TokenGrant) -> TokenGrant:
        return grant

    def revoke(self, grant: TokenGrant) -> None:
        """Best-effort remote revocation. Default: nothing to do."""

    # -- introspection ------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        d = self.descriptor
        return {
            "realtime": d.realtime_webhooks,
            "incremental": d.incremental_cursor,
            "cursor_field": d.cursor_field,
            "tombstones": d.supports_tombstones,
            "edits": d.supports_edits,
            "reactions": d.supports_reactions,
            "threads": d.supports_threads,
            "attachments": d.supports_attachments,
            "content_kinds": list(d.content_kinds),
            "backfill": d.backfill_strategy,
        }

    def health(self) -> dict[str, Any]:
        return {"connector": self.descriptor.id, "status": "unknown"}


def iter_records(pages: Iterable[PullPage]) -> Iterable[RawEnvelope]:
    for page in pages:
        yield from page.records
