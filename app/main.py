from fastapi import FastAPI

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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "service": "secure-enterprise-rag-gateway"}

    return app


app = create_app()
