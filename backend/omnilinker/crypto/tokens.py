"""Deterministic, non-reversible tokens for equality-matching secrets.

Blueprint 10.2: "per-field Deterministic Token = HMAC-SHA256(DEK_source,
normalized_value) ... lets us match 'same email' across sources WITHOUT
storing or decrypting the email."

This is what makes cross-source identity resolution (blueprint 7.2 feature f1,
the decisive 1.00-weight feature) possible on encrypted data: two Gmail and
Slack records produce the same token, so the resolver sees equality without
ever seeing plaintext.

Threat note: a deterministic token is only as strong as the HMAC key, but the
*values* live in a small domain (emails, phone numbers). An attacker holding
both the token index and the pepper could brute-force them. Mitigations:
  - the pepper is separate from the DEK and never leaves the trust boundary;
  - the token index is a *derived store* that can be dropped and rebuilt;
  - the UI offers a "redact contact details" preset that stops producing
    contact tokens for new ingests entirely.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NON_DIGIT = re.compile(r"\D+")


def normalize_email(email: str) -> str:
    """Lowercase, strip whitespace. Gmail-style dot/plus folding is applied
    only for providers known to treat them as equivalent, via `fold_google`."""
    if not email:
        return ""
    e = unicodedata.normalize("NFKC", email).strip().lower()
    return e


def fold_google(email: str) -> str:
    """Gmail ignores dots in the local part and everything after '+'."""
    e = normalize_email(email)
    if "@" not in e:
        return e
    local, _, domain = e.partition("@")
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.split("+", 1)[0].replace(".", "")
    return f"{local}@{domain}"


#: Shortest run of digits that can be a telephone number. Below this it is a
#: quantity - a year, a headcount, an incident number, a price. Normalising
#: "2026" or "4417" as a phone number is not a cosmetic problem: every
#: document that happens to contain the same figure then produces an identical
#: phone token, and the resolver merges two unrelated people on it.
MIN_PHONE_DIGITS = 7


def normalize_phone(phone: str) -> str:
    """E.164-ish: strip everything but digits, keep a leading +.

    WhatsApp exports give local formats with no country code, so this is
    best-effort by design - the resolver treats phone equality as strong (0.85)
    but not decisive, precisely because of this ambiguity.

    Returns "" for anything shorter than `MIN_PHONE_DIGITS` digits, which is the
    whole point of the guard: see that constant.
    """
    if not phone:
        return ""
    p = unicodedata.normalize("NFKC", phone).strip()
    plus = p.startswith("+")
    digits = _NON_DIGIT.sub("", p)
    if len(digits) < MIN_PHONE_DIGITS:
        return ""
    return f"+{digits}" if plus else digits


def is_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(value)))


def _token(key: bytes, value: str, domain: str) -> str:
    mac = hmac.new(key, f"{domain}|{value}".encode("utf-8"), hashlib.sha256)
    return f"{domain[:2]}_{mac.hexdigest()[:32]}"


def token_for_email(pepper: bytes, email: str, *, fold_google_addresses: bool = True) -> str:
    value = fold_google(email) if fold_google_addresses else normalize_email(email)
    if not value:
        return ""
    return _token(pepper, value, "em")


def token_for_phone(pepper: bytes, phone: str) -> str:
    value = normalize_phone(phone)
    if not value:
        return ""
    return _token(pepper, value, "ph")


def token_for_name(pepper: bytes, name: str) -> str:
    """Name normalization for blocking keys (7.2 stage 1, k_name_norm)."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    n = re.sub(r"\s+", " ", n).strip()
    if not n:
        return ""
    return _token(pepper, n, "nm")
