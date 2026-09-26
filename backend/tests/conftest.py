"""Shared fixtures.

Every test runs against a *fresh* embedded store in a temp directory, and the
singleton caches (`get_stores`, `get_key_manager`, `get_search_service`,
`get_engine`) are reset between tests. Without the reset the first test to
touch a store would own it for the whole session, and the suite would pass
while depending on execution order - which is the failure mode that makes a
test suite worse than no test suite, because it looks green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Set before anything imports `omnilinker.config`, because the app builds itself
# at import time and calls `get_settings()`, which loads `.env`. Without this,
# the developer's real `.env` - including their actual Slack client secret -
# lands in the test process, and the suite's results depend on untracked local
# state. A test whose verdict changes because of a file in someone's home
# directory is not a test, so the suite opts out and sets what it needs itself.
os.environ["OMNI_SKIP_ENV_FILE"] = "1"
os.environ.pop("OMNI_OAUTH_SLACK_CLIENT_ID", None)
os.environ.pop("OMNI_OAUTH_SLACK_CLIENT_SECRET", None)


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every singleton at a throwaway data directory."""
    data_dir = tmp_path / "omni-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OMNI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OMNI_WORKSPACE", "ws_test")
    monkeypatch.setenv("OMNI_ENCRYPT", "1")
    monkeypatch.setenv("OMNI_VECTOR", "0")

    from omnilinker import config, crypto, engine, search, store

    # Settings is a memoized singleton, and *something* reads it at import time
    # (`omnilinker.api.app` builds the app at module scope, which calls
    # `get_settings()`). By the time this fixture runs, the cache is already
    # populated with the real data directory - so without this reset every
    # "isolated" test actually shared `backend/omni-data`, and results depended
    # on execution order. The env var above is necessary but not sufficient.
    config.reset_settings()
    # `load_env_file` writes into `os.environ` and remembers the keys it wrote in
    # a module-level set. Left alone, that set accumulates across tests, so a
    # later `reset_env_file()` would remove variables that a previous test
    # happened to set - a leak that shows up as a baffling `KeyError` in a test
    # that never touched the environment. Both halves are reset together.
    config.reset_env_file()

    crypto.keys.reset_key_manager()
    store.reset_stores()
    search.reset_search_service()
    engine.reset_engine()
    yield data_dir
    config.reset_env_file()
    config.reset_settings()
    crypto.keys.reset_key_manager()
    store.reset_stores()
    search.reset_search_service()
    engine.reset_engine()


@pytest.fixture
def pipeline():
    from omnilinker.pipeline import IngestPipeline

    return IngestPipeline()


@pytest.fixture
def ingested(pipeline):
    """A workspace with the demo dataset fully ingested and every derived
    store built. Returns the pipeline, with the ingest run on `.run`."""
    run = pipeline.sync_connector("demo")
    assert run.status == "ok", run.errors
    from omnilinker.engine import get_engine

    get_engine().rebuild()
    pipeline.run = run  # type: ignore[attr-defined]
    return pipeline


@pytest.fixture
def stores():
    from omnilinker.store import get_stores

    return get_stores()
