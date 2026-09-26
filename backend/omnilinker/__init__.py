"""OmniLinker - unified cross-source data intelligence platform.

Blueprint: OmniLinker_Blueprint.txt v1.0.0
This package is a working vertical slice of the architecture described there.

Layer map (mirrors blueprint section 3.3):
    config      -> environment / deployment profile
    ids         -> ULID entity identifiers (ADR-013)
    crypto      -> envelope encryption, key hierarchy, deterministic tokens
    store       -> DocStore + GraphStore interfaces w/ embedded and docker backends
    connectors  -> Connector contract, SDK helpers, per-provider plugins
    normalize   -> canonical schema + provider-independent mapping
    pipeline    -> ingestion orchestrator (raw -> canonical -> stores)
    identity    -> entity resolution, clustering, hidden-connection detectors
    search      -> inverted index, hybrid retrieval + RRF, NL2Query compiler
    ai          -> extractive summarization, deadline/recurrence prediction
    api         -> FastAPI surface
"""

__version__ = "0.1.0"
BLUEPRINT_VERSION = "1.0.0"
