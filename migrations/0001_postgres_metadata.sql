CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT,
    source_connector TEXT,
    source_document_id TEXT,
    source_document_version TEXT,
    owner_id TEXT NOT NULL,
    department_scope JSONB NOT NULL,
    visibility_scope JSONB NOT NULL,
    security_level INTEGER NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    last_error TEXT,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    tags JSONB NOT NULL,
    current BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_content_hash
ON documents(tenant_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_source_ref
ON documents(tenant_id, source_connector, source_document_id, current);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_no INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    security_level INTEGER NOT NULL,
    department_scope JSONB NOT NULL,
    role_scope JSONB NOT NULL,
    metadata_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scene TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    active_topic TEXT NOT NULL DEFAULT '',
    permission_signature TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json JSONB NOT NULL,
    token_usage INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id TEXT PRIMARY KEY,
    scene TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    output_schema JSONB NOT NULL,
    enabled BOOLEAN NOT NULL,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    high_risk_terms JSONB NOT NULL,
    restricted_departments JSONB NOT NULL,
    enabled BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    query TEXT NOT NULL,
    rewritten_query TEXT NOT NULL DEFAULT '',
    scene TEXT NOT NULL DEFAULT '',
    retrieval_docs_json JSONB NOT NULL,
    prompt_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    risk_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    conversation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_summary TEXT NOT NULL,
    action TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS source_sync_runs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    mode TEXT NOT NULL,
    continue_on_error BOOLEAN NOT NULL,
    request_json JSONB NOT NULL,
    result_items_json JSONB NOT NULL,
    total INTEGER NOT NULL,
    succeeded INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    imported_new INTEGER NOT NULL,
    reused_current INTEGER NOT NULL,
    created_new_version INTEGER NOT NULL,
    queued INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
