from fastapi import APIRouter

from app.api.v1 import admin, auth, chat, docs, metrics


api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(docs.router, prefix="/docs", tags=["docs"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
