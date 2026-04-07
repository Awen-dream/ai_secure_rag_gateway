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

当前版本仍然使用内存存储与规则检索，但代码边界已经按企业级落地方向拆开，后续接入 PostgreSQL、Redis、Milvus、PGVector、Elasticsearch、LlamaIndex、LangChain 时可以直接替换基础设施层实现。
