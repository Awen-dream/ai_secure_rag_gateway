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
- `GET /api/v1/admin/retrieval/backends`
- `GET /api/v1/admin/retrieval/backends/{backend}/health`
- `GET /api/v1/admin/cache/health`
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
- 融合打分与弱相关截断
- 显式的 Elasticsearch / PGVector 适配器边界
- 管理侧可查看后端配置与检索解释
- PGVector DDL / upsert / search SQL 预览与 schema 初始化入口
- ES mapping / bulk / search body 预览与索引初始化入口

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

后续接入 PostgreSQL、Redis、Milvus、PGVector、Elasticsearch、LlamaIndex、LangChain 时，可以直接替换基础设施层实现而保留现有领域与 API 边界。

接口与核心方法备注说明规范见 [docs/development-standards.md](docs/development-standards.md)。
