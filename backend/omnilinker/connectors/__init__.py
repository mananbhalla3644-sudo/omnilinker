"""Connector framework: one contract, one registry, one SDK."""

from omnilinker.connectors.base_rest import ApiError, BaseRestConnector, new_pkce_pair, new_state
from omnilinker.connectors.contract import (
    AuthorizationRequest,
    AuthorizationResult,
    CanonicalRecord,
    Connector,
    ConnectorDescriptor,
    PullPage,
    RawEnvelope,
    Scope,
    StreamDescriptor,
    TokenGrant,
)
from omnilinker.connectors.ratelimit import AdaptiveThrottle, RateLimiter, RetryPolicy, TokenBucket
from omnilinker.connectors.registry import (
    ConnectorRegistry,
    RegistrationError,
    get_registry,
    register_builtins,
    registry,
)

__all__ = [
    "Connector",
    "ConnectorDescriptor",
    "Scope",
    "TokenGrant",
    "AuthorizationRequest",
    "AuthorizationResult",
    "StreamDescriptor",
    "RawEnvelope",
    "PullPage",
    "CanonicalRecord",
    "BaseRestConnector",
    "ApiError",
    "new_pkce_pair",
    "new_state",
    "RateLimiter",
    "TokenBucket",
    "RetryPolicy",
    "AdaptiveThrottle",
    "ConnectorRegistry",
    "RegistrationError",
    "registry",
    "get_registry",
    "register_builtins",
]
