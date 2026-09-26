"""Rate limiting and retry helpers (blueprint 6.5).

The failure mode everyone forgets: OAuth apps have an *aggregate* quota. Ten
users polling aggressively throttles the app for everyone, and the provider
starts returning 429 with `Retry-After` for hours. So there are two limiters
here, not one:

  * per-(source, user, route) token bucket  - fairness between users
  * per-source app-level limiter            - protects the OAuth app quota
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Classic token bucket. `acquire()` blocks; `try_acquire()` does not."""

    rate_per_sec: float
    capacity: float
    _tokens: float = field(init=False)
    _ts: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._ts = time.monotonic()

    def try_acquire(self, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, seconds_to_wait)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._ts
            self._ts = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            if self._tokens >= cost:
                self._tokens -= cost
                return True, 0.0
            deficit = cost - self._tokens
            return False, deficit / max(self.rate_per_sec, 1e-6)

    def acquire(self, cost: float = 1.0, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            ok, wait = self.try_acquire(cost)
            if ok:
                return True
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.25))


class RateLimiter:
    """Per-key buckets plus one shared app-level bucket."""

    def __init__(self, rate_per_sec: float, burst: int, app_rate_per_sec: float | None = None) -> None:
        self.app = TokenBucket(app_rate_per_sec or rate_per_sec, max(burst * 4, burst))
        self._default = (rate_per_sec, burst)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def bucket(self, key: str = "default", rate_per_sec: float | None = None,
               burst: int | None = None) -> TokenBucket:
        with self._lock:
            hit = self._buckets.get(key)
            if hit is None:
                base_rate, base_burst = self._default
                hit = TokenBucket(rate_per_sec or base_rate, burst or base_burst)
                self._buckets[key] = hit
            return hit

    def acquire(self, key: str = "default", cost: float = 1.0, timeout: float = 30.0) -> bool:
        # App-level first: if the app quota is spent, there is no point burning
        # the user's own budget on a request that will 429 anyway.
        if not self.app.acquire(cost, timeout=timeout):
            return False
        return self.bucket(key).acquire(cost, timeout=timeout)


class AdaptiveThrottle:
    """Multiplicative decrease on 429, slow additive recovery on success.

    Blueprint 6.5: "On 429: halve effective rate, set a global cool-off, retry
    with full jitter, honor Retry-After."
    """

    def __init__(self, rate_per_sec: float, minimum: float = 0.1) -> None:
        self.base = rate_per_sec
        self.current = rate_per_sec
        self.minimum = minimum
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    def penalize(self, retry_after: float | None = None) -> float:
        with self._lock:
            self.current = max(self.minimum, self.current / 2)
            cool = retry_after if retry_after is not None else min(60.0, 5.0 * (self.base / self.current))
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + cool)
            return self.current

    def reward(self) -> None:
        with self._lock:
            if time.monotonic() < self._cooldown_until:
                return
            self.current = min(self.base, self.current * 1.1)

    def wait_time(self) -> float:
        with self._lock:
            return max(0.0, self._cooldown_until - time.monotonic())


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.4
    max_delay: float = 30.0
    jitter: str = "full"  # full | equal | none

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        # Exponential backoff. Attempt is 1-based.
        raw = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        match self.jitter:
            case "full":
                return random.uniform(0, raw)
            case "equal":
                return raw / 2 + random.uniform(0, raw / 2)
            case _:
                return raw

    def should_retry(self, attempt: int, status: int | None, retryable_exc: bool = False) -> bool:
        if attempt >= self.max_attempts:
            return False
        if retryable_exc:
            return True
        if status is None:
            return False
        return status == 429 or 500 <= status < 600
