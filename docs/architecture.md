# Architecture

本项目按 `api / application / domain / infrastructure` 分层，在线问答主链按步骤拆成清晰的编排层和能力层。

## Layering

### `api`
- 负责 HTTP 路由、鉴权依赖注入、请求响应模型绑定。
- 典型入口：
  - `app/api/v1/chat.py`
  - `app/api/v1/admin.py`
  - `app/api/deps.py`

### `application`
- 负责跨领域的流程编排，是在线问答链路的主承载层。
- 当前在线问答相关模块：
  - `app/application/access`
    - access filter、access signature、权限边界收敛
  - `app/application/session`
    - session context、follow-up、history summary、session cache
  - `app/application/query`
    - query understanding、rewrite、query planning
  - `app/application/retrieval`
    - recall planning、rerank layer
  - `app/application/context`
    - context assembly
  - `app/application/prompting`
    - prompt build
  - `app/application/generation`
    - generation、fallback、guard、validation
  - `app/application/chat`
    - 端到端 chat orchestration

### `domain`
- 负责业务实体、领域模型、领域规则和可复用的领域服务。
- 典型内容：
  - `app/domain/chat/models.py`, `schemas.py`
  - `app/domain/prompts/models.py`
  - `app/domain/retrieval/models.py`, `rerankers.py`
  - `app/domain/risk/*`
  - `app/domain/documents/*`

### `infrastructure`
- 负责外部系统接入和底层实现。
- 典型内容：
  - `app/infrastructure/llm`
  - `app/infrastructure/search`
  - `app/infrastructure/vectorstore`
  - `app/infrastructure/db`
  - `app/infrastructure/external_sources`
  - `app/infrastructure/cache`

## Online QA Flow

当前在线问答链路对应关系如下：

1. `Access / Session`
   - `app/application/access/service.py`
   - `app/application/session/service.py`
   - `app/application/session/cache.py`
   - `app/application/session/summarizer.py`
2. `Query Planning`
   - `app/application/query/planning.py`
   - `app/application/query/understanding.py`
   - `app/application/query/rewrite.py`
3. `Recall Planning`
   - `app/application/retrieval/planning.py`
4. `Retrieval Execution`
   - `app/domain/retrieval/services.py`
   - `app/infrastructure/search/elasticsearch.py`
   - `app/infrastructure/vectorstore/pgvector.py`
5. `Rerank`
   - `app/application/retrieval/rerank.py`
   - `app/application/retrieval/llm_reranker.py`
   - `app/domain/retrieval/rerankers.py`
6. `Context Assembly`
   - `app/application/context/builder.py`
7. `Prompt Build`
   - `app/application/prompting/builder.py`
8. `Generation & Guard`
   - `app/application/generation/service.py`
9. `Chat Orchestration`
   - `app/application/chat/orchestrator.py`

## Naming Conventions

- `*Orchestrator`
  - 用于端到端流程编排，例如 `ChatOrchestrator`
- `*PlanningService`
  - 用于“决定下一步怎么做”的计划层，例如 `QueryPlanningService`、`RecallPlanningService`
- `*ContextService`
  - 用于承接请求级或会话级上下文构建，例如 `SessionContextService`
- `*BuilderService`
  - 用于把上游结果组装成下游可消费对象，例如 `ContextBuilderService`、`PromptBuilderService`
- `*RerankService`
  - 用于多路候选结果的融合、打分和重排编排
- `*Reranker`
  - 用于具体重排引擎实现，例如 `HeuristicReranker`、`LLMReranker`
- `*TemplateService`
  - 用于模板生命周期和模板级校验

## Retrieval Notes

- Elasticsearch keyword plan 现在会下沉 `tag/year/exact phrase` 等过滤和 boost 信号，而不是只在内存候选集上处理。
- Rerank 层支持可插拔引擎，当前包含本地 heuristic/cross-encoder-fallback 路径，以及基于 OpenAI 的 `LLMReranker`。

## Current Canonical Paths

- Chat online-QA orchestration:
  - `app/application/chat/orchestrator.py`
- Prompt template lifecycle and validation:
  - `app/domain/prompts/template_service.py`

新代码应直接依赖这些 canonical 路径。
