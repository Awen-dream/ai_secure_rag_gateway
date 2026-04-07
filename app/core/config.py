import os
from typing import Optional

from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Secure Enterprise RAG Gateway")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    sqlite_path: str = os.getenv("APP_SQLITE_PATH", ".data/secure_rag_gateway.db")
    chunk_tokenizer_model: str = os.getenv("CHUNK_TOKENIZER_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    chunk_tokenizer_encoding: Optional[str] = os.getenv("CHUNK_TOKENIZER_ENCODING")


settings = AppSettings()
