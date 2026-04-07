import os
from typing import Optional

from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Secure Enterprise RAG Gateway")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    sqlite_path: str = os.getenv("APP_SQLITE_PATH", ".data/secure_rag_gateway.db")
    elasticsearch_index: str = os.getenv("APP_ELASTICSEARCH_INDEX", "knowledge_chunks")
    pgvector_table: str = os.getenv("APP_PGVECTOR_TABLE", "document_embeddings")
    chunk_tokenizer_model: str = os.getenv("CHUNK_TOKENIZER_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    chunk_tokenizer_encoding: Optional[str] = os.getenv("CHUNK_TOKENIZER_ENCODING")
    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "400"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))


settings = AppSettings()
