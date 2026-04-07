import os

from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Secure Enterprise RAG Gateway")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    sqlite_path: str = os.getenv("APP_SQLITE_PATH", ".data/secure_rag_gateway.db")


settings = AppSettings()
