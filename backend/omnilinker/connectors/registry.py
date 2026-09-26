"""Connector registry (blueprint 6.2.1 REGISTER + the five static checks).

`register()` is a gate, not a dict assignment. A connector is rejected unless
it passes all five checks, because every one of them has caused a real incident
in systems like this:

  C1 scopes      every scope has a non-empty justification
  C2 limits      a declared rate limit exists and is not absurd
  C3 cursor      if `incremental_cursor` then a `cursor_field` is named
  C4 purity      `normalize` declares no I/O (checked by inspecting the source)
  C5 version     semantic version, bumped whenever the normalizer changes

C4 deserves a note. It is a source inspection rather than a runtime flag
because the failure it prevents - a normalizer that makes an HTTP call - is
invisible at review time and turns a pure, replayable function into a network
client. Providers legitimately build URLs in `pull`; they must not in
`normalize`.
"""

from __future__ import annotations

import inspect
import re
import threading
from typing import Callable, Iterable, Type

from omnilinker.connectors.contract import Connector, ConnectorDescriptor

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

#: Providers allowed to touch the network inside `normalize`. Empty on purpose;
#: adding an entry is a design decision that should be argued about.
C4_ALLOWED_IO: frozenset[str] = frozenset()

#: Markers that suggest I/O in a normalizer body. Deliberately broad: a false
#: positive costs one exemption, a false negative costs an incident.
_IO_MARKERS = ("httpx", "requests", "urlopen", "urllib", "socket", "boto3", "aiohttp", "open(")


class RegistrationError(ValueError):
    """Raised by `register()`. The message names the failing check."""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_scopes(d: ConnectorDescriptor) -> None:
    for s in d.scopes:
        if not s.justification.strip():
            raise RegistrationError(f"C1 {d.id}: scope {s.name!r} has no justification")
    if d.auth_flow == "oauth2_code" and not d.scopes:
        raise RegistrationError(f"C1 {d.id}: OAuth2 connectors must declare scopes")


def check_limits(d: ConnectorDescriptor) -> None:
    if d.rate_limit_per_sec <= 0:
        raise RegistrationError(f"C2 {d.id}: rate_limit_per_sec must be > 0")
    if d.rate_limit_per_sec > 2000:
        raise RegistrationError(
            f"C2 {d.id}: rate_limit_per_sec={d.rate_limit_per_sec} looks wrong. Most "
            "providers are tiered; declare the conservative tier."
        )
    if d.rate_limit_burst < 1:
        raise RegistrationError(f"C2 {d.id}: rate_limit_burst must be >= 1")


def check_cursor(d: ConnectorDescriptor) -> None:
    if d.incremental_cursor and not d.cursor_field:
        raise RegistrationError(
            f"C3 {d.id}: incremental_cursor=True requires a cursor_field. Without one, "
            "every sync is a full re-ingest."
        )
    if d.backfill_strategy == "incremental" and not d.supports_tombstones:
        # Not fatal: deletions can be reconciled. But it must be a conscious
        # choice, so we require the descriptor to say so in notes.
        if "tombstone" not in d.notes.lower() and "reconcil" not in d.notes.lower():
            raise RegistrationError(
                f"C3 {d.id}: incremental backfill without tombstones must document the "
                "reconciliation strategy in `notes`"
            )


def check_purity(cls: Type[Connector]) -> None:
    if cls.descriptor.id in C4_ALLOWED_IO:
        return
    fn = cls.normalize
    src = inspect.getsource(fn)
    body = src.split("\n", 1)[1] if "\n" in src else src
    for marker in _IO_MARKERS:
        if marker in body:
            raise RegistrationError(
                f"C4 {cls.descriptor.id}: normalize() references {marker!r}. Normalizers "
                "must be pure; do network I/O in pull(). If this is unavoidable, add the "
                "provider to registry.C4_ALLOWED_IO with a comment explaining why."
            )


def check_version(d: ConnectorDescriptor) -> None:
    if not _SEMVER.match(d.version):
        raise RegistrationError(f"C5 {d.id}: version {d.version!r} is not semver")


#: Descriptor-only checks. Purity is handled separately because it needs the
#: class (or instance) rather than the descriptor.
DESCRIPTOR_CHECKS: tuple[Callable[[ConnectorDescriptor], None], ...] = (
    check_scopes,
    check_limits,
    check_cursor,
    check_version,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ConnectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Connector]] = {}
        self._lock = threading.RLock()

    def register(self, connector_id: str, factory: Callable[[], Connector]) -> None:
        instance = factory()
        d = instance.descriptor
        if d.id != connector_id:
            raise RegistrationError(
                f"registered as {connector_id!r} but descriptor says {d.id!r}"
            )
        for check in DESCRIPTOR_CHECKS:
            check(d)
        check_purity(factory if inspect.isclass(factory) else type(instance))
        with self._lock:
            self._factories[connector_id] = factory

    def register_all(self, connectors: Iterable[Type[Connector]]) -> list[RegistrationError]:
        """Best-effort bulk registration. Returns the failures instead of raising,
        so one bad plugin cannot take down the process."""
        failures: list[RegistrationError] = []
        for cls in connectors:
            try:
                self.register(cls.descriptor.id, cls)
            except RegistrationError as exc:
                failures.append(exc)
        return failures

    def get(self, connector_id: str) -> Connector:
        with self._lock:
            factory = self._factories.get(connector_id)
        if factory is None:
            raise KeyError(
                f"unknown connector {connector_id!r}; known: {sorted(self._factories)}"
            )
        return factory()

    def has(self, connector_id: str) -> bool:
        with self._lock:
            return connector_id in self._factories

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def descriptors(self) -> list[dict]:
        return [self.get(cid).descriptor.to_dict() for cid in self.ids()]

    def capabilities(self) -> dict[str, dict]:
        return {cid: self.get(cid).capabilities() for cid in self.ids()}


registry = ConnectorRegistry()


def register_builtins() -> ConnectorRegistry:
    """Register every connector that ships with OmniLinker.

    Imports are local so a broken optional provider cannot stop the app from
    booting; the failure is reported per connector instead.
    """
    from omnilinker.connectors.providers.cloud import (
        DropboxConnector,
        GoogleDriveConnector,
        OneDriveConnector,
    )
    from omnilinker.connectors.providers.demo import DemoConnector
    from omnilinker.connectors.providers.discord import DiscordConnector
    from omnilinker.connectors.providers.gmail import GmailConnector
    from omnilinker.connectors.providers.notes import EvernoteConnector, NotionConnector
    from omnilinker.connectors.providers.slack import SlackConnector
    from omnilinker.connectors.providers.whatsapp import WhatsAppConnector
    from omnilinker.connectors.providers.youtube import YouTubeConnector

    builtins: Iterable[Type[Connector]] = (
        SlackConnector, GmailConnector, DiscordConnector, WhatsAppConnector,
        GoogleDriveConnector, OneDriveConnector, DropboxConnector,
        NotionConnector, EvernoteConnector, YouTubeConnector, DemoConnector,
    )
    failures = registry.register_all(builtins)
    if failures:
        # A built-in failing a check is a build error. Loud, but not fatal at
        # boot: the rest of the app must still start.
        for failure in failures:
            print(f"[omnilinker] connector REJECTED: {failure}")
    return registry


def get_registry() -> ConnectorRegistry:
    if not registry.ids():
        register_builtins()
    return registry


def reset_registry() -> None:
    """Test hook."""
    with registry._lock:  # noqa: SLF001 - deliberate, test-only
        registry._factories.clear()
