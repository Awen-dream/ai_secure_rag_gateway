# API

对外接口分为用户侧与管理侧，两者都通过统一的 FastAPI 网关暴露。

## Admin Evaluation

- `GET /api/v1/admin/evaluation/dataset`
  - 查看当前已加载的离线评测集样本。
- `GET /api/v1/admin/evaluation/dataset/stats`
  - 返回评测样本总量、review coverage、scene 分布、label 分布和 status 分布。
- `GET /api/v1/admin/evaluation/dataset/overview`
  - 返回 dataset stats + 当前质量基线快照，适合后台工作台首页使用。
- `GET /api/v1/admin/evaluation/dataset/template`
  - 返回单条样本模板和批量示例。
- `GET /api/v1/admin/evaluation/dataset/export`
  - 导出当前评测集，返回 JSON 样本和 JSONL 文本。
- `GET /api/v1/admin/evaluation/dataset/{sample_id}`
  - 查看单个评测样本。
- `POST /api/v1/admin/evaluation/dataset/upsert`
  - 新增或更新单个评测样本。
- `POST /api/v1/admin/evaluation/dataset/import`
  - 按 `replace` 或 `upsert` 语义批量导入评测样本。
- `DELETE /api/v1/admin/evaluation/dataset/{sample_id}`
  - 删除单个评测样本。
- `POST /api/v1/admin/evaluation/dataset/replace`
  - 替换当前离线评测集。
- `POST /api/v1/admin/evaluation/dataset/bulk-annotate`
  - 对一批样本统一写入 labels、reviewed、reviewed_by、notes 或 status。
- `POST /api/v1/admin/evaluation/dataset/bootstrap`
  - 基于当前成功文档自动生成 starter 评测集。
- `GET /api/v1/admin/evaluation/baseline`
  - 查看当前质量基线配置。
- `POST /api/v1/admin/evaluation/baseline`
  - 更新质量基线，用于 quality gate、regression alert 和 release readiness。
- `POST /api/v1/admin/evaluation/run`
  - 运行一轮离线评估，可选 `limit` 参数限制样本数，并返回 `quality_gate` 门禁结论。
- `POST /api/v1/admin/evaluation/run-shadow`
  - 运行一轮 shadow evaluation，对比当前主链路与启发式 shadow 基线，并返回 `winner`/`winner_reasons`。
- `GET /api/v1/admin/evaluation/runs`
  - 查看已持久化的 offline / shadow 评测历史。
- `GET /api/v1/admin/evaluation/runs/{run_id}`
  - 查看某一次评测运行的完整明细。
- `GET /api/v1/admin/evaluation/trend`
  - 基于最近的 offline run 生成趋势摘要、回归告警和 `quality_gate` 门禁结论。
- `GET /api/v1/admin/evaluation/shadow-report`
  - 返回最近一轮 shadow evaluation 的 winner、recommendation 和变化样本统计。
- `GET /api/v1/admin/evaluation/release-readiness`
  - 基于最近的 offline/shadow run 生成发布前检查结论。
- `GET /api/v1/admin/evaluation/release-gate`
  - 返回显式的 release gate checklist，并附带 pass/fail 结果，适合脚本和 CI 调用。
- `GET /api/v1/metrics/evaluation`
  - 返回基于当前评测集即时计算的评估摘要指标，并附带 trend、alerts 与 release_readiness。

## Admin Operations

- `GET /admin-console`
  - 返回内置后台页面，用于直接查看运营看板、评测集、文档清单和审计日志。
- `GET /api/v1/admin/dashboard/summary`
  - 返回聚合后的文档、流量、质量、风险、评测、缓存、队列和飞书连接器状态。
- `GET /api/v1/admin/documents`
  - 返回管理侧文档清单，支持 `tenant_id`、`status`、`source_type`、`search`、`current_only`、`limit` 过滤。
- `POST /api/v1/admin/documents/{doc_id}/retire`
  - 退役一个文档版本，并同步清理对应索引。
- `GET /api/v1/admin/audit`
  - 支持通过 `limit`、`risk_level`、`action`、`scene`、`query` 过滤结构化审计记录。
