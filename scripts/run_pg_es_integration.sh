#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  docker compose -f docker-compose.integration.yml down
}

trap cleanup EXIT

docker compose -f docker-compose.integration.yml up -d

echo "Waiting for redis, pgvector and elasticsearch to become healthy..."
for _ in $(seq 1 60); do
  REDIS_STATUS="$(docker inspect -f '{{.State.Health.Status}}' secure-rag-redis 2>/dev/null || true)"
  PG_STATUS="$(docker inspect -f '{{.State.Health.Status}}' secure-rag-pgvector 2>/dev/null || true)"
  ES_STATUS="$(docker inspect -f '{{.State.Health.Status}}' secure-rag-elasticsearch 2>/dev/null || true)"
  if [[ "$REDIS_STATUS" == "healthy" && "$PG_STATUS" == "healthy" && "$ES_STATUS" == "healthy" ]]; then
    break
  fi
  sleep 2
done

if [[ -f .venv/bin/activate ]]; then
  # Prefer the project virtualenv locally when it exists.
  source .venv/bin/activate
fi

python -m unittest \
  app.tests.integration.test_pg_es_retrieval_integration \
  app.tests.integration.test_postgres_metadata_repository_integration \
  app.tests.integration.test_redis_integration
