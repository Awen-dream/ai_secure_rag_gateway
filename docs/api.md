# API

对外接口分为用户侧与管理侧，两者都通过统一的 FastAPI 网关暴露。

## Admin Evaluation

- `GET /api/v1/admin/evaluation/dataset`
  - 查看当前已加载的离线评测集样本。
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
- `GET /api/v1/metrics/evaluation`
  - 返回基于当前评测集即时计算的评估摘要指标，并附带 trend、alerts 与 release_readiness。
