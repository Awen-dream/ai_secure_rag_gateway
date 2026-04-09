from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import register_middleware


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A secure enterprise knowledge access gateway with domain-driven modules "
            "for ingestion, retrieval, conversation, risk control, and audit."
        ),
    )
    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    admin_console_path = Path(__file__).resolve().parent / "static" / "admin_console.html"

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "service": "secure-enterprise-rag-gateway"}

    @app.get("/admin-console", response_class=HTMLResponse)
    def admin_console() -> str:
        """Return the built-in admin console page for dashboard, evaluation and governance workflows."""

        return admin_console_path.read_text(encoding="utf-8")

    return app


app = create_app()
