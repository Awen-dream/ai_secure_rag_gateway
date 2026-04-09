# API

对外接口分为用户侧与管理侧，两者都通过统一的 FastAPI 网关暴露。

## Admin Evaluation

- `GET /api/v1/admin/evaluation/dataset`
  - 查看当前已加载的离线评测集样本。
- `POST /api/v1/admin/evaluation/run`
  - 运行一轮离线评估，可选 `limit` 参数限制样本数。
- `POST /api/v1/admin/evaluation/run-shadow`
  - 运行一轮 shadow evaluation，对比当前主链路与启发式 shadow 基线。
- `GET /api/v1/admin/evaluation/runs`
  - 查看已持久化的 offline / shadow 评测历史。
- `GET /api/v1/admin/evaluation/runs/{run_id}`
  - 查看某一次评测运行的完整明细。
- `GET /api/v1/admin/evaluation/trend`
  - 基于最近的 offline run 生成趋势摘要与回归告警。
- `GET /api/v1/metrics/evaluation`
  - 返回基于当前评测集即时计算的评估摘要指标，并附带 trend 与 alerts。
