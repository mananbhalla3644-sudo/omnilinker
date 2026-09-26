"""Crypto package: key hierarchy, envelope encryption, deterministic tokens.

See blueprint section 10.2. Threat modelled: an attacker with full read
access to the databases and object store. Content must be unreadable to them.
"""

from omnilinker.crypto.envelope import decrypt, encrypt, is_envelope
from omnilinker.crypto.keys import KeyManager, get_key_manager
from omnilinker.crypto.tokens import (
    normalize_email,
    normalize_phone,
    token_for_email,
    token_for_phone,
)

__all__ = [
    "KeyManager",
    "get_key_manager",
    "encrypt",
    "decrypt",
    "is_envelope",
    "token_for_email",
    "token_for_phone",
    "normalize_email",
    "normalize_phone",
]
