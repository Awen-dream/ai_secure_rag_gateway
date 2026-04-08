# Secure Enterprise RAG Gateway

一个按领域分层组织的企业知识安全问答平台骨架，当前重点落在：

- 统一接入与鉴权
- 权限前置检索
- 风险策略与拒答
- 会话、引用与审计闭环
- OpenAI Responses API 驱动的可回退生成层

## 运行

```bash
uvicorn app.main:app --reload
make run-ingestion-worker
```

## PGVector + Elasticsearch 联调

先安装项目依赖，然后运行联调脚本：

```bash
python -m pip install -r requirements.txt
bash scripts/run_pg_es_integration.sh
```

脚本会：

- 用 `docker-compose.integration.yml` 启动本地 `pgvector` 和 `elasticsearch`
- 等待两个容器健康
- 优先使用项目 `.venv`，否则沿用当前 `python` 跑真实集成测试

也可以直接用 `Makefile`：

```bash
make install
make test-unit
make test-integration
make test-all
make run-ingestion-worker
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

## OpenAI 生成层

当前问答主链路已支持 OpenAI Responses API：

- 配置了 `OPENAI_API_KEY` 时，`/api/v1/chat/query` 会调用真实 OpenAI 生成层
- 未配置时，会安全回退到本地规则回答，方便本地开发和离线测试

相关环境变量：

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_OUTPUT_TOKENS`
- `OPENAI_TEMPERATURE`

## Redis 缓存与限流

当前平台已支持 Redis 用于：

- 会话摘要缓存
- 检索结果缓存
- 用户问答限流

相关环境变量：

- `APP_REDIS_MODE`
- `APP_REDIS_URL`
- `APP_SESSION_CACHE_TTL_SECONDS`
- `APP_RETRIEVAL_CACHE_TTL_SECONDS`
- `APP_RATE_LIMIT_WINDOW_SECONDS`
- `APP_RATE_LIMIT_MAX_REQUESTS`

## 文档异步任务流

当前文档上传已支持真实异步任务流：

- API 负责写入文档元数据、暂存原始文件并把 `doc_id` 入队
- ingestion worker 独立消费队列，执行解析、切分、embedding、索引

相关环境变量：

- `APP_DOCUMENT_STAGING_DIR`
- `APP_DOCUMENT_INGESTION_QUEUE_NAME`
- `APP_DOCUMENT_INGESTION_WORKER_POLL_SECONDS`

## 飞书数据源

当前已支持从 Feishu 导入 `docx` 文档和 `wiki` 节点背后的 `docx` 文档。

相关环境变量：

- `APP_FEISHU_BASE_URL`
- `APP_FEISHU_APP_ID`
- `APP_FEISHU_APP_SECRET`

导入接口：

- `POST /api/v1/admin/sources/feishu/import`

健康检查接口：

- `GET /api/v1/admin/sources/feishu/health`

## Embedding 与 Rerank

当前 PGVector 已支持切换到真实 embedding provider，并在混合检索后执行可插拔 reranker。

相关环境变量：

- `APP_EMBEDDING_PROVIDER`
- `APP_EMBEDDING_API_KEY`
- `APP_EMBEDDING_BASE_URL`
- `APP_EMBEDDING_MODEL`
- `APP_EMBEDDING_TIMEOUT_SECONDS`
- `APP_EMBEDDING_DIMENSIONS`
- `APP_RERANKER_MODE`
- `APP_RERANKER_TOP_N`

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
- `POST /api/v1/docs/upload-file`
- `GET /api/v1/docs/{doc_id}`
- `GET /api/v1/docs`
- `POST /api/v1/docs/{doc_id}/retry`
- `POST /api/v1/chat/query`
- `POST /api/v1/chat/stream`
- `GET /api/v1/admin/prompts`
- `GET /api/v1/admin/policies`
- `GET /api/v1/admin/audit`
- `GET /api/v1/admin/retrieval/backends`
- `GET /api/v1/admin/retrieval/backends/{backend}/health`
- `GET /api/v1/admin/cache/health`
- `GET /api/v1/admin/queue/document-ingestion/health`
- `GET /api/v1/admin/sources/feishu/health`
- `POST /api/v1/admin/sources/feishu/import`
- `POST /api/v1/admin/retrieval/explain`
- `GET /api/v1/admin/retrieval/backends/{backend}/plan`
- `POST /api/v1/admin/retrieval/backends/elasticsearch/init-index`
- `POST /api/v1/admin/retrieval/backends/pgvector/init-schema`
- `GET /api/v1/metrics/retrieval`

## 鉴权请求头

- `X-User-Id`
- `X-Tenant-Id`
- `X-Department-Id`
- `X-Role`
- `X-Clearance-Level`

## 说明

当前版本已经切到本地 SQLite 持久化与可插拔 hybrid retrieval 骨架，文档、会话、模板、策略、审计会落盘到 `APP_SQLITE_PATH`。

元数据主库现在支持两种模式：

- `APP_REPOSITORY_BACKEND=sqlite`：本地开发默认模式
- `APP_REPOSITORY_BACKEND=postgres`：企业部署主库模式，需要配置 `APP_POSTGRES_DSN`

当前检索层已包含：

- 关键词召回
- 轻量向量召回
- 按意图动态调权
- 启发式 reranker
- 融合打分与弱相关截断
- 显式的 Elasticsearch / PGVector 适配器边界
- 管理侧可查看后端配置与检索解释
- PGVector DDL / upsert / search SQL 预览与 schema 初始化入口
- ES mapping / bulk / search body 预览与索引初始化入口
- OpenAI-compatible embedding provider 接口，可替换本地 deterministic embedding

当前生成层已包含：

- Prompt 模板读取与渲染
- 授权证据与引用绑定后再调用 LLM
- OpenAI Responses API 适配器
- 无 key / 调用失败时的安全回退

当前元数据层已包含：

- SQLite 仓储
- PostgreSQL 仓储
- 统一仓储契约，服务层无感切换
- PostgreSQL 初始化 SQL，见 `migrations/0001_postgres_metadata.sql`
- `documents.last_error`，用于追踪失败原因与重试

当前缓存层已包含：

- Redis client 与本地 fallback
- session summary cache
- retrieval cache
- chat 接口固定窗口限流

当前权限层已包含：

- 统一 `AccessFilter` 构造
- 文档访问与 chunk 访问共用同一套权限判断
- Elasticsearch / PGVector 查询计划共用同一套 access filter 语义

当前输出安全层已包含：

- 常见 PII 脱敏：邮箱、手机号、身份证、银行卡号
- 密钥/口令/令牌类内容拒答
- 高敏部门输出降级为仅返回引用
- 输出侧风险动作会回写到问答响应和审计日志

当前文档接入层已包含：

- JSON 文本上传
- multipart 文件上传
- `PDF / DOCX / Markdown / HTML / Text` 解析
- HTML 标签剥离与块级换行保留
- DOCX `word/document.xml` 文本抽取
- 本地 staging source store，支持把原始上传文件与文本暂存到 `APP_DOCUMENT_STAGING_DIR`
- 文档状态机：`pending -> parsing -> chunking -> embedding -> indexing -> success/failed`
- `async_mode` 时只负责入队，真实解析/切分/索引由独立 ingestion worker 消费 `APP_DOCUMENT_INGESTION_QUEUE_NAME`
- 可通过 `retry` 入口重试失败文档
- 飞书外部数据源导入：支持 `docx` 链接和 `wiki` 节点链接导入

当前外部数据源层已包含：

- Feishu tenant token 交换
- Feishu `docx raw_content` 抓取
- Feishu `wiki -> docx` 节点解析
- 管理侧导入接口，导入后直接复用文档异步 ingest 队列

后续接入 PostgreSQL、Redis、Milvus、PGVector、Elasticsearch、LlamaIndex、LangChain 时，可以直接替换基础设施层实现而保留现有领域与 API 边界。

接口与核心方法备注说明规范见 [docs/development-standards.md](docs/development-standards.md)。
