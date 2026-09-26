"""HTTP layer. `omnilinker.api.app:app` is the ASGI entry point."""

from omnilinker.api.app import app, create_app
from omnilinker.api.routes import router

__all__ = ["app", "create_app", "router"]
