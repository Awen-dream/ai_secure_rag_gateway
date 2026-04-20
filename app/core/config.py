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
    redis_mode: str = os.getenv("APP_REDIS_MODE", "local-fallback")
    redis_url: Optional[str] = os.getenv("APP_REDIS_URL")
    session_cache_ttl_seconds: int = int(os.getenv("APP_SESSION_CACHE_TTL_SECONDS", "3600"))
    retrieval_cache_ttl_seconds: int = int(os.getenv("APP_RETRIEVAL_CACHE_TTL_SECONDS", "300"))
    eval_dataset_path: str = os.getenv("APP_EVAL_DATASET_PATH", ".data/evaluation/dataset.jsonl")
    eval_runs_dir: str = os.getenv("APP_EVAL_RUNS_DIR", ".data/evaluation/runs")
    eval_baseline_path: str = os.getenv("APP_EVAL_BASELINE_PATH", ".data/evaluation/baseline.json")
    rate_limit_window_seconds: int = int(os.getenv("APP_RATE_LIMIT_WINDOW_SECONDS", "60"))
    rate_limit_max_requests: int = int(os.getenv("APP_RATE_LIMIT_MAX_REQUESTS", "30"))
    document_ingestion_queue_name: str = os.getenv("APP_DOCUMENT_INGESTION_QUEUE_NAME", "queue:document_ingestion")
    document_ingestion_worker_poll_seconds: float = float(os.getenv("APP_DOCUMENT_INGESTION_WORKER_POLL_SECONDS", "1.0"))
    source_sync_scheduler_poll_seconds: float = float(os.getenv("APP_SOURCE_SYNC_SCHEDULER_POLL_SECONDS", "60.0"))
    ingestion_engine: str = os.getenv("APP_INGESTION_ENGINE", "native")
    evaluation_engine: str = os.getenv("APP_EVAL_ENGINE", "native")
    source_sync_engine: str = os.getenv("APP_SOURCE_SYNC_ENGINE", "native")
    feishu_base_url: str = os.getenv("APP_FEISHU_BASE_URL", "https://open.feishu.cn/open-apis")
    feishu_app_id: Optional[str] = os.getenv("APP_FEISHU_APP_ID")
    feishu_app_secret: Optional[str] = os.getenv("APP_FEISHU_APP_SECRET")
    llm_default_provider: str = os.getenv("APP_LLM_DEFAULT_PROVIDER", "openai")
    llm_runtime: str = os.getenv("APP_LLM_RUNTIME", "native")
    llm_generation_provider: str = os.getenv("APP_LLM_GENERATION_PROVIDER", "")
    llm_query_understanding_provider: str = os.getenv("APP_LLM_QUERY_UNDERSTANDING_PROVIDER", "")
    llm_reranker_provider: str = os.getenv("APP_LLM_RERANKER_PROVIDER", "")
    embedding_provider: str = os.getenv("APP_EMBEDDING_PROVIDER", "local-fallback")
    embedding_runtime: str = os.getenv("APP_EMBEDDING_RUNTIME", "native")
    embedding_api_key: Optional[str] = os.getenv("APP_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY"))
    embedding_base_url: str = os.getenv("APP_EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    embedding_model: str = os.getenv("APP_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_timeout_seconds: float = float(os.getenv("APP_EMBEDDING_TIMEOUT_SECONDS", "30"))
    embedding_dimensions: Optional[int] = (
        int(os.getenv("APP_EMBEDDING_DIMENSIONS")) if os.getenv("APP_EMBEDDING_DIMENSIONS") else None
    )
    reranker_mode: str = os.getenv("APP_RERANKER_MODE", "cross-encoder-fallback")
    reranker_top_n: int = int(os.getenv("APP_RERANKER_TOP_N", "8"))
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
    document_staging_dir: str = os.getenv("APP_DOCUMENT_STAGING_DIR", ".data/staging/documents")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
    openai_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    qwen_api_key: Optional[str] = os.getenv("APP_QWEN_API_KEY", os.getenv("QWEN_API_KEY"))
    qwen_model: str = os.getenv("APP_QWEN_MODEL", os.getenv("QWEN_MODEL", "qwen-plus"))
    qwen_base_url: str = os.getenv(
        "APP_QWEN_BASE_URL",
        os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    qwen_timeout_seconds: float = float(os.getenv("APP_QWEN_TIMEOUT_SECONDS", "30"))
    qwen_max_output_tokens: int = int(os.getenv("APP_QWEN_MAX_OUTPUT_TOKENS", "900"))
    qwen_temperature: float = float(os.getenv("APP_QWEN_TEMPERATURE", "0.1"))
    deepseek_api_key: Optional[str] = os.getenv("APP_DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
    deepseek_model: str = os.getenv("APP_DEEPSEEK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_base_url: str = os.getenv(
        "APP_DEEPSEEK_BASE_URL",
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    deepseek_timeout_seconds: float = float(os.getenv("APP_DEEPSEEK_TIMEOUT_SECONDS", "30"))
    deepseek_max_output_tokens: int = int(os.getenv("APP_DEEPSEEK_MAX_OUTPUT_TOKENS", "900"))
    deepseek_temperature: float = float(os.getenv("APP_DEEPSEEK_TEMPERATURE", "0.1"))


settings = AppSettings()
