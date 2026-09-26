"""Entity resolution: blocking, scoring, clustering, and hidden connections."""

from omnilinker.identity.blocking import (
    blocking_stats,
    build_blocks,
    cooccurrence_pairs,
    pairs_from_blocks,
)
from omnilinker.identity.detectors import (
    DetectionResult,
    Insight,
    interactions_from_documents,
    participants_by_conversation,
    run_all,
)
from omnilinker.identity.resolver import (
    IdentityResolver,
    PairScore,
    ResolutionResult,
)

__all__ = [
    "IdentityResolver",
    "PairScore",
    "ResolutionResult",
    "Insight",
    "DetectionResult",
    "run_all",
    "interactions_from_documents",
    "participants_by_conversation",
    "build_blocks",
    "pairs_from_blocks",
    "cooccurrence_pairs",
    "blocking_stats",
]
