"""Configuration and deployment profiles (blueprint section 3.5).

Three profiles:
    embedded  - zero external services. DocStore/GraphStore/Index are local
                files + memory. Everything else identical. This is the profile
                that runs on a laptop with no Docker installed.
    docker    - real MongoDB + Neo4j + Qdrant + MinIO via docker-compose.
    cloud     - docker profile + managed services + KEDA-scale workers.

`OMNI_MODE` selects the profile. The rest of the codebase never branches on
the profile directly - it asks the factory in `omnilinker.store.get_stores()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- profile -------------------------------------------------------
    mode: str = field(default_factory=lambda: os.environ.get("OMNI_MODE", "embedded"))

    # --- storage -------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OMNI_DATA_DIR", "./omni-data")).resolve()
    )
    mongo_uri: str = field(
        default_factory=lambda: os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    )
    mongo_db: str = field(default_factory=lambda: os.environ.get("MONGO_DB", "omnilinker"))
    neo4j_uri: str = field(
        default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "omnilinker"))

    # --- crypto --------------------------------------------------------
    # In embedded/local mode the master key lives in a 0600 keystore file.
    # In cloud mode it is injected by the KMS/External Secrets operator.
    master_key_b64: str | None = field(default_factory=lambda: os.environ.get("OMNI_MASTER_KEY_B64"))
    keystore_path: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["OMNI_KEYSTORE_PATH"]).resolve()
            if os.environ.get("OMNI_KEYSTORE_PATH")
            else None
        )
    )

    # --- workspace -----------------------------------------------------
    default_workspace: str = field(
        default_factory=lambda: os.environ.get("OMNI_WORKSPACE", "ws_local")
    )

    # --- ingestion -----------------------------------------------------
    request_timeout: float = field(default_factory=lambda: float(os.environ.get("OMNI_HTTP_TIMEOUT", "20")))
    rate_limit_per_sec: float = field(
        default_factory=lambda: float(os.environ.get("OMNI_RATE_LIMIT", "5"))
    )
    enable_encryption: bool = field(default_factory=lambda: _env_bool("OMNI_ENCRYPT", True))

    # --- search --------------------------------------------------------
    enable_vector: bool = field(default_factory=lambda: _env_bool("OMNI_VECTOR", False))
    embed_model: str = field(
        default_factory=lambda: os.environ.get("OMNI_EMBED_MODEL", "hash-1024")
    )
    rrf_k: int = field(default_factory=lambda: _env_int("OMNI_RRF_K", 60))
    result_limit: int = field(default_factory=lambda: _env_int("OMNI_RESULT_LIMIT", 20))

    # --- identity ------------------------------------------------------
    merge_threshold: float = field(default_factory=lambda: float(os.environ.get("OMNI_MERGE_T", "0.90")))
    suggest_threshold: float = field(default_factory=lambda: float(os.environ.get("OMNI_SUGGEST_T", "0.65")))

    # --- ai ------------------------------------------------------------
    # "extractive" needs no model and no network. "local" / "cloud" route to a
    # real model gateway; see omnilinker.ai for the router.
    ai_mode: str = field(default_factory=lambda: os.environ.get("AI_MODE", "extractive"))

    # --- derived stores -------------------------------------------------
    #: Write the inverted index to disk. Off by default: the index is
    #: plaintext-bearing by construction (it has to be, to rank), so the
    #: default posture is "rebuild in memory on boot" rather than "keep a
    #: second copy of the corpus's vocabulary on the filesystem".
    persist_index: bool = field(default_factory=lambda: _env_bool("OMNI_PERSIST_INDEX", False))
    #: k-anonymity floor applied when the index *is* persisted. A term in fewer
    #: than k documents is effectively a verbatim quote from one document.
    index_k_anonymity: int = field(default_factory=lambda: _env_int("OMNI_INDEX_K", 2))

    # --- web -----------------------------------------------------------
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.environ.get("OMNI_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        )
    )
    #: The origin the provider redirects back to. Must match the value
    #: registered in the provider's app console, and it is the single value that
    #: has to change when the app moves - so it is config rather than something
    #: assembled per connector.
    public_url: str = field(
        default_factory=lambda: os.environ.get("OMNI_PUBLIC_URL", "http://127.0.0.1:8900")
    )
    serve_frontend: bool = field(default_factory=lambda: _env_bool("OMNI_SERVE_FRONTEND", True))
    frontend_dist: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["OMNI_FRONTEND_DIST"]).resolve()
            if os.environ.get("OMNI_FRONTEND_DIST")
            else None
        )
    )

    # ------------------------------------------------------------------
    @property
    def uses_embedded_store(self) -> bool:
        return self.mode == "embedded"

    def ensure_dirs(self) -> None:
        for sub in ("", "vault", "blobs", "index", "exports"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    def as_public_dict(self) -> dict:
        """Safe-to-serve config for the /api/system/info endpoint (no secrets)."""
        return {
            "mode": self.mode,
            "workspace": self.default_workspace,
            "data_dir": str(self.data_dir),
            "encryption": "envelope-aes-256-gcm" if self.enable_encryption else "disabled",
            "ai_mode": self.ai_mode,
            "vector_search": self.enable_vector,
            "merge_threshold": self.merge_threshold,
            "rrf_k": self.rrf_k,
        }


# Keys this module injected, so `reset_env_file` can remove exactly those and
# nothing else.
_env_file_keys: set[str] = set()


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Read a `.env` file into `os.environ`, without overwriting real env vars.

    `python-dotenv` is a dependency for fifteen lines, and the semantics matter
    more than the code: a value already present in the environment wins. That is
    what makes `OMNI_PORT=9000 ./run.sh` override a checked-in `.env`, and it is
    the opposite of what naive loading does.

    This was a real gap - `.env.example` shipped with a documented list of
    settings and nothing ever read the file, so a user could fill it in
    perfectly and see no effect.
    """
    if os.environ.get("OMNI_SKIP_ENV_FILE") == "1":
        # Set by the test suite. Without it, a developer's real `.env` - which
        # holds their actual Slack client secret - leaks into the test process
        # and the suite's results start depending on a file that is not in the
        # repository. A test that passes on one machine and fails on another
        # because of untracked local state is not a test.
        return {}

    target = Path(path) if path else Path.cwd() / ".env"
    if not target.exists():
        # Also look one level up, so `backend/` and `frontend/` both find a
        # repo-root .env.
        parent = target.parent.parent / ".env"
        if not parent.exists():
            return {}
        target = parent
    loaded: dict[str, str] = {}
    try:
        text = target.read_text("utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        _env_file_keys.add(key)
        loaded[key] = value
    return loaded


_loaded_env_file: bool = False


def ensure_env_loaded() -> dict[str, str]:
    """Idempotent. Called before the first `get_settings()`."""
    global _loaded_env_file
    if _loaded_env_file:
        return {}
    _loaded_env_file = True
    explicit = os.environ.get("OMNI_ENV_FILE")
    return load_env_file(explicit) if explicit else load_env_file()


def reset_env_file() -> None:
    """Test hook - forget that a `.env` was read, *and* undo it.

    `load_env_file` writes into `os.environ`, so the values outlive any single
    `Settings` object. Resetting only `_settings` left the developer's real
    credentials sitting in the process environment where every later test could
    see them, which is why this also removes exactly the keys it added. Keys
    that were already set before loading are left alone - those belong to the
    caller, not to us.
    """
    global _loaded_env_file
    for key in _env_file_keys:
        os.environ.pop(key, None)
    _env_file_keys.clear()
    _loaded_env_file = False


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        ensure_env_loaded()
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Test hook - forces re-read of the environment."""
    global _settings
    _settings = None
