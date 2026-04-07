PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
UVICORN ?= .venv/bin/uvicorn
COMPOSE_FILE ?= docker-compose.integration.yml
UNIT_TESTS = \
	app.tests.unit.test_ingestion_pipelines \
	app.tests.unit.test_import \
	app.tests.unit.test_hybrid_retrieval \
	app.tests.unit.test_openai_client \
	app.tests.unit.test_chat_llm_integration \
	app.tests.unit.test_retrieval_backends \
	app.tests.unit.test_admin_retrieval_endpoints
INTEGRATION_TESTS = \
	app.tests.integration.test_pg_es_retrieval_integration \
	app.tests.integration.test_postgres_metadata_repository_integration

.PHONY: install run test-unit test-integration test-all integration-up integration-down compile

install:
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app.main:app --reload

compile:
	PYTHONPYCACHEPREFIX=/tmp/pycache $(PYTHON) -m py_compile $$(find app -name '*.py' -print) main.py

test-unit:
	$(PYTHON) -m unittest $(UNIT_TESTS)

integration-up:
	docker compose -f $(COMPOSE_FILE) up -d

integration-down:
	docker compose -f $(COMPOSE_FILE) down

test-integration:
	bash scripts/run_pg_es_integration.sh

test-all: test-unit test-integration
