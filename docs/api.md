# API

对外接口分为用户侧与管理侧，两者都通过统一的 FastAPI 网关暴露。

## Admin Evaluation

- `GET /api/v1/admin/evaluation/dataset`
  - 查看当前已加载的离线评测集样本。
- `POST /api/v1/admin/evaluation/run`
  - 运行一轮离线评估，可选 `limit` 参数限制样本数。
- `GET /api/v1/metrics/evaluation`
  - 返回基于当前评测集即时计算的评估摘要指标。
