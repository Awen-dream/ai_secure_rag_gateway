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

内置后台页面：

```bash
open http://127.0.0.1:8000/admin-console
```

页面本身用于调试和运营展示，真正访问后台数据时仍然通过管理员请求头调用 API。

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

## 评测平台、后台管理与运营看板

当前版本已经内置一套轻量可用的评测与管理控制台能力：

- `/admin-console`
  - 可直接查看运营看板、评测数据集、评测运行历史、文档清单和审计日志
- `/api/v1/admin/dashboard/summary`
  - 聚合文档、流量、质量、风险、评测、缓存、队列和飞书连接器状态
- `/api/v1/admin/documents`
  - 管理侧文档清单，支持状态、来源类型和关键词过滤
- `/api/v1/admin/documents/{doc_id}/retire`
  - 退役文档并同步清理索引
- `/api/v1/admin/evaluation/dataset/replace`
  - 替换离线评测数据集
- `/api/v1/admin/evaluation/dataset/bootstrap`
  - 基于当前成功文档自动生成 starter 评测集
- `/api/v1/admin/evaluation/dataset/upsert`
  - 单条新增或更新评测样本
- `/api/v1/admin/evaluation/dataset/import`
  - 通过 `replace` 或 `upsert` 语义批量导入样本
- `/api/v1/admin/evaluation/dataset/export`
  - 导出当前评测集，返回 JSON 和 JSONL 文本
- `/api/v1/admin/evaluation/dataset/template`
  - 返回样本模板与批量示例
- `/api/v1/admin/evaluation/dataset/bulk-annotate`
  - 批量补 labels / reviewed / reviewed_by / notes
- `/api/v1/admin/evaluation/baseline`
  - 查看或更新质量基线
- `/api/v1/admin/evaluation/release-gate`
  - 返回显式的发布门禁 checklist 与 pass/fail 结论

当前运营看板聚合的核心指标包括：

- 文档总量、当前生效数、失败数、来源分布
- 生命周期分布：active、deprecated、retired、stale
- 总问答量、24h 请求量、活跃会话、独立用户
- 引用覆盖率、拒答率、改写率、平均延迟
- 最新 offline/shadow evaluation 与 release readiness 决策
- top questions、top documents、缓存/队列/飞书连接健康

当前评测平台已经支持：

- 评测样本 CRUD
- 样本模板与批量导入导出
- 批量标注与 review coverage 统计
- 质量基线持久化
- 基线驱动的 quality gate
- release readiness gate
- 本地/CI 统一的 release gate fixture 预置与阻断式发布门禁

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
- `GET /admin-console`
- `GET /api/v1/admin/prompts`
- `GET /api/v1/admin/policies`
- `GET /api/v1/admin/audit`
- `GET /api/v1/admin/dashboard/summary`
- `GET /api/v1/admin/documents`
- `POST /api/v1/admin/documents/{doc_id}/retire`
- `POST /api/v1/admin/documents/{doc_id}/deprecate`
- `POST /api/v1/admin/documents/{doc_id}/replace`
- `POST /api/v1/admin/documents/{doc_id}/restore`
- `GET /api/v1/admin/documents/stale`
- `GET /api/v1/admin/evaluation/dataset`
- `GET /api/v1/admin/evaluation/dataset/stats`
- `GET /api/v1/admin/evaluation/dataset/overview`
- `GET /api/v1/admin/evaluation/dataset/template`
- `GET /api/v1/admin/evaluation/dataset/export`
- `GET /api/v1/admin/evaluation/dataset/{sample_id}`
- `POST /api/v1/admin/evaluation/dataset/upsert`
- `POST /api/v1/admin/evaluation/dataset/import`
- `DELETE /api/v1/admin/evaluation/dataset/{sample_id}`
- `POST /api/v1/admin/evaluation/dataset/replace`
- `POST /api/v1/admin/evaluation/dataset/bulk-annotate`
- `POST /api/v1/admin/evaluation/dataset/bootstrap`
- `GET /api/v1/admin/evaluation/baseline`
- `POST /api/v1/admin/evaluation/baseline`
- `POST /api/v1/admin/evaluation/run`
- `POST /api/v1/admin/evaluation/run-shadow`
- `GET /api/v1/admin/evaluation/runs`
- `GET /api/v1/admin/evaluation/release-readiness`
- `GET /api/v1/admin/evaluation/release-gate`
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

当前评测与治理层已包含：

- 离线评测数据集存储、替换和 starter dataset bootstrap
- 评测样本单条 upsert / delete / 批量标注
- 样本模板、JSON/JSONL 导出、replace/upsert 导入
- offline evaluation / shadow evaluation
- 质量基线持久化与基线驱动 quality gate
- trend summary / regression alerts
- release readiness report 与显式 release gate checklist
- `scripts/prepare_release_gate_fixture.py` 可生成最小文档、评测样本、offline run 与 shadow run
- `make prepare-release-gate` / `make release-gate` 可在本地复现 CI 的发布门禁流程
- 文档生命周期治理：`active / deprecated / retired`、`replaced_by_doc_id`、`source_last_seen_at`
- 管理台可直接做 `deprecate / replace / restore / stale` 治理动作
- 结构化审计检索与后台看板汇总

本地启动 ingestion worker：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_document_ingestion_worker.py
```

或直接使用：

```bash
make run-ingestion-worker
```

当前外部数据源层已包含：

- Feishu tenant token 交换
- Feishu `docx raw_content` 抓取
- Feishu `wiki -> docx` 节点解析
- 管理侧导入接口，导入后直接复用文档异步 ingest 队列

后续接入 PostgreSQL、Redis、Milvus、PGVector、Elasticsearch、LlamaIndex、LangChain 时，可以直接替换基础设施层实现而保留现有领域与 API 边界。

接口与核心方法备注说明规范见 [docs/development-standards.md](docs/development-standards.md)。
