# Secure Enterprise RAG Gateway

一个按领域分层组织的企业知识安全问答平台骨架，当前重点落在：

- 统一接入与鉴权
- 权限前置检索
- 风险策略与拒答
- 会话、引用与审计闭环

## 运行

```bash
uvicorn app.main:app --reload
```

## Chunking 默认值

当前文档切分会优先按标题、列表、表格、代码块等结构切分，再在 section 内按真实 tokenizer token budget 聚合。

- 默认 tokenizer: `text-embedding-3-small`
- 默认 chunk 大小: `400` tokens
- 默认 overlap: `60` tokens

可以通过这些环境变量覆盖：

- `CHUNK_TOKENIZER_MODEL`
- `CHUNK_TOKENIZER_ENCODING`
- `CHUNK_MAX_TOKENS`
- `CHUNK_OVERLAP_TOKENS`

快速查看某段文本在当前 tokenizer 下的 token 数和 chunk 分布：

```bash
printf '你的文本内容' | python3 scripts/inspect_chunk_tokens.py
python3 scripts/inspect_chunk_tokens.py docs/architecture.md
```

## 当前目录

```text
app/
├── api/
├── application/
├── core/
├── domain/
├── infrastructure/
└── tests/
docs/
scripts/
```

## 关键接口

- `GET /healthz`
- `GET /api/v1/auth/me`
- `POST /api/v1/docs/upload`
- `GET /api/v1/docs`
- `POST /api/v1/chat/query`
- `POST /api/v1/chat/stream`
- `GET /api/v1/admin/prompts`
- `GET /api/v1/admin/policies`
- `GET /api/v1/admin/audit`
- `GET /api/v1/metrics/retrieval`

## 鉴权请求头

- `X-User-Id`
- `X-Tenant-Id`
- `X-Department-Id`
- `X-Role`
- `X-Clearance-Level`

## 说明

当前版本已经切到本地 SQLite 持久化与可插拔 hybrid retrieval 骨架，文档、会话、模板、策略、审计会落盘到 `APP_SQLITE_PATH`。

当前检索层已包含：

- 关键词召回
- 轻量向量召回
- 按意图动态调权
- 融合打分与弱相关截断
- 显式的 Elasticsearch / PGVector 适配器边界

后续接入 PostgreSQL、Redis、Milvus、PGVector、Elasticsearch、LlamaIndex、LangChain 时，可以直接替换基础设施层实现而保留现有领域与 API 边界。

接口与核心方法备注说明规范见 [docs/development-standards.md](docs/development-standards.md)。
