"""FastAPI application factory.

One process serves the API *and* the built frontend, so the whole product is
`uvicorn omnilinker.api.app:app` with no reverse proxy and no CORS in the happy
path. CORS is still configured because the dev workflow runs Vite on :5173
against the API on :8000, and a dev-only origin is a much smaller thing to
justify than a permissive production policy.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omnilinker import __version__
from omnilinker.api.routes import router
from omnilinker.config import get_settings
from omnilinker.connectors.registry import get_registry
from omnilinker.engine import get_engine
from omnilinker.ids import prefixed
from omnilinker.normalize import iso


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Registration failures are reported, not raised: a broken optional provider
    # must not stop the app from booting with the other ten.
    registry = get_registry()
    engine = get_engine()
    print(f"[omnilinker] v{__version__} mode={settings.mode} "
          f"workspace={settings.default_workspace}")
    print(f"[omnilinker] data_dir={settings.data_dir}")
    print(f"[omnilinker] connectors={registry.ids()}")
    print(f"[omnilinker] encryption={'on' if settings.enable_encryption else 'off'} "
          f"kek={engine.pipeline.km.fingerprint()}")
    print(f"[omnilinker] search={'hybrid' if settings.enable_vector else 'lexical'} "
          f"ai={settings.ai_mode}")
    try:
        yield
    finally:
        get_engine().pipeline.audit({"action": "shutdown"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OmniLinker",
        version=__version__,
        description=(
            "Unified search, relationship graph and entity resolution across "
            "every source you own, with field-level envelope encryption on the "
            "content and no write-back to any provider."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def timing(request: Request, call_next: Any) -> Any:
        """Server-side latency on every response.

        The blueprint commits to a p95 search budget, and a budget nobody
        measures is a budget nobody meets. The header is also what the frontend
        displays, so a slow response is visible in the UI rather than only in a
        log nobody reads.
        """
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = int((time.perf_counter() - started) * 1000)
        response.headers["X-Omni-Took-Ms"] = str(elapsed)
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """One error shape, everywhere.

        The client is the only consumer, so a consistent envelope with the
        exception type and a request id is worth more than a stack trace. The
        id goes into the audit log, which is how a user-reported bug becomes a
        findable record instead of a screenshot.
        """
        request_id = prefixed("req")
        try:
            get_engine().pipeline.audit({
                "action": "unhandled_error",
                "request_id": request_id,
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
        except Exception:
            pass  # never let the error reporter itself raise
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": {"type": type(exc).__name__,
                                             "message": str(exc)[:500],
                                             "request_id": request_id}},
        )

    app.include_router(router)
    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Any) -> None:
    """Serve the built SPA from `/`, with an API-only fallback.

    When the frontend has not been built yet, `/` returns a page that says so
    and lists the API, rather than a 404. A bare 404 on the root of a running
    service looks like a broken install; this reads as an unfinished step, which
    is what it is.
    """
    dist = settings.frontend_dist
    if not dist or not dist.exists():
        @app.get("/", include_in_schema=False)
        def api_only() -> Any:
            from fastapi.responses import HTMLResponse

            return HTMLResponse(_API_ONLY_HTML, status_code=200)
        return

    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    index = dist / "index.html"

    @app.get("/", include_in_schema=False)
    def spa_root() -> Any:
        return FileResponse(str(index))

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Any:
        """Client-side routing needs the SPA shell for unknown paths, but a
        mistyped API route must still 404 as an API route."""
        if full_path.startswith("api/"):
            return JSONResponse({"ok": False, "error": {"type": "NotFound",
                                                        "message": f"no route /{full_path}"}},
                                status_code=404)
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index))


_API_ONLY_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OmniLinker API</title>
<style>
  body{font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
       max-width:44rem;margin:4rem auto;padding:0 1.5rem;color:#1a1a1a;background:#fbfbfa}
  code{background:#eee;padding:.15em .4em;border-radius:3px;font-size:.9em}
  h1{font-size:1.4rem} li{margin:.3rem 0}
  .box{background:#fff;border:1px solid #e5e5e2;border-radius:8px;padding:1.2rem 1.5rem}
  .warn{border-left:3px solid #d98324;padding-left:.9rem;margin:1.2rem 0}
</style></head>
<body>
<div class="box">
<h1>OmniLinker API is running</h1>
<p>The frontend has not been built. Two commands fix it:</p>
<pre><code>cd frontend
npm install
npm run build</code></pre>
<div class="warn"><p>The API is fully usable in the meantime:</p>
<ul>
  <li><a href="/api/docs">/api/docs</a> - interactive OpenAPI docs</li>
  <li><a href="/api/system/info">/api/system/info</a> - capabilities and deviations</li>
  <li><code>POST /api/sync</code> - seed the demo workspace</li>
  <li><code>GET /api/search?q=shard+key</code> - search</li>
  <li><code>GET /api/graph/ego</code> - relationship graph</li>
</ul></div>
</div>
</body></html>"""


app = create_app()
