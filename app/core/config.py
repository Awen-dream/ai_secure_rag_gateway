import os
from typing import Optional

from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Secure Enterprise RAG Gateway")
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    repository_backend: str = os.getenv("APP_REPOSITORY_BACKEND", "sqlite")
    sqlite_path: str = os.getenv("APP_SQLITE_PATH", ".data/secure_rag_gateway.db")
    postgres_dsn: Optional[str] = os.getenv("APP_POSTGRES_DSN")
    postgres_auto_init_schema: bool = os.getenv("APP_POSTGRES_AUTO_INIT_SCHEMA", "true").lower() == "true"
    elasticsearch_index: str = os.getenv("APP_ELASTICSEARCH_INDEX", "knowledge_chunks")
    elasticsearch_mode: str = os.getenv("APP_ELASTICSEARCH_MODE", "local-fallback")
    elasticsearch_endpoint: Optional[str] = os.getenv("APP_ELASTICSEARCH_ENDPOINT")
    elasticsearch_auto_init_index: bool = os.getenv("APP_ELASTICSEARCH_AUTO_INIT_INDEX", "false").lower() == "true"
    pgvector_table: str = os.getenv("APP_PGVECTOR_TABLE", "document_embeddings")
    pgvector_mode: str = os.getenv("APP_PGVECTOR_MODE", "local-fallback")
    pgvector_dsn: Optional[str] = os.getenv("APP_PGVECTOR_DSN")
    pgvector_auto_init_schema: bool = os.getenv("APP_PGVECTOR_AUTO_INIT_SCHEMA", "false").lower() == "true"
    embedding_dimension: int = int(os.getenv("APP_EMBEDDING_DIMENSION", "1536"))
    chunk_tokenizer_model: str = os.getenv("CHUNK_TOKENIZER_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    chunk_tokenizer_encoding: Optional[str] = os.getenv("CHUNK_TOKENIZER_ENCODING")
    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "400"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    openai_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))


settings = AppSettings()
