"""Key hierarchy: UMK -> WKEK -> per-source DEK, derived with HKDF-SHA256.

Blueprint section 10.2.

    User Master Key (UMK)          32 random bytes, generated once, stored in a
                                  0600 keystore file (or injected by a KMS in
                                  cloud mode). Never used directly on data.
      |
      +-- Workspace KEK (WKEK)     = HKDF(UMK, info="omnilinker/wkek/v1|ws")
      |     wrap-only
      |
      +-- per-source DEK           = HKDF(WKEK, info="omnilinker/dek/v1|<source>")
      |     encrypts that source's content: raw payloads, canonical bodies,
      |     attachments, transcripts, extracted document text
      |
      +-- per-field token          = HMAC-SHA256(DEK_source, normalized_value)
            deterministic, so "same email" matches across sources WITHOUT ever
            storing or decrypting the email itself.

Design notes:
  * Every derivation is bound to a versioned `info` string. Changing the scheme
    later means a new info string, so old keys keep working and a migration is
    a re-wrap, not a re-encrypt-everything.
  * The `pepper` for deterministic tokens is separate from the DEK so that a
    leaked token index cannot be used as an encryption key and vice versa.
  * Keys are cached in memory per (workspace, source). A `KeyManager` is cheap;
    construct one per process and reuse it.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_LEN = 32
SCHEME_VERSION = 1


def _hkdf(ikm: bytes, info: str, salt: bytes = b"", length: int = KEY_LEN) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt or None,
        info=info.encode("utf-8"),
    )
    return hkdf.derive(ikm)


@dataclass(frozen=True)
class DerivedKeys:
    """The three key roles for one (workspace, source) pair."""

    source: str
    dek: bytes
    pepper: bytes
    kid: str


class KeyManager:
    """Owns the UMK and derives everything else. Thread-safe."""

    def __init__(
        self,
        workspace: str,
        master_key: bytes,
        *,
        keystore_path: Path | None = None,
    ) -> None:
        if len(master_key) != KEY_LEN:
            raise ValueError(f"master key must be {KEY_LEN} bytes, got {len(master_key)}")
        self.workspace = workspace
        self._umk = master_key
        self.keystore_path = keystore_path
        self._cache: dict[str, DerivedKeys] = {}
        self._lock = threading.Lock()

    # -- construction -------------------------------------------------
    @classmethod
    def load(
        cls,
        workspace: str,
        *,
        data_dir: Path,
        master_key_b64: str | None = None,
        keystore_path: Path | None = None,
    ) -> "KeyManager":
        """Load from an explicit key, else from a keystore file, else create one.

        Creating a key on first run is a deliberate, loud action: it prints a
        warning. Losing this file means losing the data (blueprint 10.3).
        """
        if master_key_b64:
            return cls(workspace, base64.b64decode(master_key_b64), keystore_path=keystore_path)

        path = keystore_path or (data_dir / "vault" / "keystore.json")
        if path.exists():
            blob = json.loads(path.read_text("utf-8"))
            if blob.get("workspace") not in (None, workspace):
                raise RuntimeError(
                    f"keystore at {path} belongs to workspace {blob.get('workspace')!r}, "
                    f"not {workspace!r}. Refusing to decrypt with the wrong key."
                )
            return cls(workspace, base64.b64decode(blob["master_key_b64"]), keystore_path=path)

        master = os.urandom(KEY_LEN)
        km = cls(workspace, master, keystore_path=path)
        km._persist(master)
        return km

    def _persist(self, master: bytes) -> None:
        assert self.keystore_path is not None
        self.keystore_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "v": SCHEME_VERSION,
            "workspace": self.workspace,
            "kdf": "raw-32-byte (local profile). Use a KMS + passphrase KDF for cloud.",
            "master_key_b64": base64.b64encode(master).decode(),
        }
        # Write then chmod: on Windows chmod is advisory, on POSIX it is the
        # difference between "private" and "world readable".
        self.keystore_path.write_text(json.dumps(payload, indent=2), "utf-8")
        try:
            os.chmod(self.keystore_path, 0o600)
        except OSError:
            pass

    # -- derivation ---------------------------------------------------
    @property
    def wkek(self) -> bytes:
        return _hkdf(self._umk, f"omnilinker/wkek/v{SCHEME_VERSION}|{self.workspace}")

    def dek(self, source: str) -> DerivedKeys:
        source = source.strip().lower()
        with self._lock:
            hit = self._cache.get(source)
            if hit is not None:
                return hit
        wkek = self.wkek
        dek = _hkdf(wkek, f"omnilinker/dek/v{SCHEME_VERSION}|{source}")
        pepper = _hkdf(wkek, f"omnilinker/pepper/v{SCHEME_VERSION}|{source}")
        derived = DerivedKeys(source=source, dek=dek, pepper=pepper, kid=f"dek_{source}_v{SCHEME_VERSION}")
        with self._lock:
            self._cache[source] = derived
        return derived

    @property
    def identity_pepper(self) -> bytes:
        """Workspace-wide pepper for **deterministic identity tokens**.

        This is deliberately *not* `dek(source).pepper`, and getting that wrong
        is invisible until cross-source resolution silently returns nothing:

            token_for_email(dek("slack").pepper, "a@x.com")
              != token_for_email(dek("gmail").pepper, "a@x.com")

        because the two peppers differ by construction. Every token then matches
        only within its own provider, which feels like working resolution and
        produces zero cross-source merges. The system looks healthy - blocks are
        built, pairs are scored, persons are emitted - and the one thing it
        exists to do never happens.

        The two secrets have different jobs, so they are different keys:

        * per-source **DEK** encrypts content. Sharing it across sources would
          mean one compromised provider exposes every source.
        * workspace **identity pepper** derives comparable tokens. It must be
          shared, because comparison across sources is the entire purpose, and
          it never encrypts anything.

        Losing the identity pepper costs only the ability to re-derive tokens;
        it does not affect any document.
        """
        return _hkdf(self.wkek, f"omnilinker/identity-pepper/v{SCHEME_VERSION}|{self.workspace}")

    # -- introspection (for the /api/system/info + data-passport views)
    def fingerprint(self) -> str:
        """Non-reversible id of the UMK - lets a user confirm two installs
        share a key without ever revealing it."""
        return _hkdf(self._umk, "omnilinker/fingerprint")[:8].hex()

    def describe(self) -> dict:
        return {
            "scheme": f"HKDF-SHA256 v{SCHEME_VERSION}",
            "workspace": self.workspace,
            "kek_fingerprint": self.fingerprint(),
            "keystore": str(self.keystore_path) if self.keystore_path else "<memory>",
            "kdf": "AEAD envelope, AAD-bound to document id + field path",
        }


_manager: KeyManager | None = None
_manager_lock = threading.Lock()


def get_key_manager() -> KeyManager:
    """Process-wide singleton bound to the configured workspace."""
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is None:
            from omnilinker.config import get_settings

            s = get_settings()
            _manager = KeyManager.load(
                s.default_workspace,
                data_dir=s.data_dir,
                master_key_b64=s.master_key_b64,
                keystore_path=s.keystore_path,
            )
    return _manager


def reset_key_manager() -> None:
    """Test hook - forces a fresh keystore load (or creation)."""
    global _manager
    with _manager_lock:
        _manager = None
