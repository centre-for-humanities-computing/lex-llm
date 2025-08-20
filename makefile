# Makefile for the Lex LLM project

.PHONY: install run static-type-check lint lint-check test pr help

# Default target
default: help

install:
	@echo "--- 🚀 Installing project ---"
	make generate-api
	uv venv
	uv pip install -e build/lex_db_api  # Installs lex_db_api as a package
	uv sync

install-dev:
	@echo "--- 🚀 Installing development dependencies ---"
	make generate-api
	uv venv
	uv pip install -e build/lex_db_api  # Installs lex_db_api as a package
	uv sync --dev

generate-api:
	@echo "--- 🔧 Generating API client (docker) ---"
	@mkdir -p build
	docker run --rm \
		-v ${PWD}:/local \
		openapitools/openapi-generator-cli generate \
		-i /local/openapi/lex-db.yaml \
		-g python \
		-o /local/build/lex_db_api \
		--additional-properties=packageName=lex_db_api,pyproject=true

clean-api:
	@echo "--- 🧹 Cleaning generated client ---"
	rm -rf build/

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	uv run mypy src

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						            # running ruff formatting
	uv run ruff check . --fix								# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	uv run ruff format . --check						    # running ruff formatting
	uv run ruff check . 							        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest src/tests/

pr:
	@echo "--- 🚀 Running PR checks ---"
	make install
	make lint
	make static-type-check
	make test
	make generate-openapi-schema
	@echo "--- ✅ All checks passed ---"
	@echo "--- 🚀 Ready to make a PR ---"

run:
	@echo "--- ▶️ Running the application ---"
	make generate-openapi-schema
	uv run main.py

run-dev: install-dev
	@echo "--- ▶️ Running the application in dev mode (hot reload) ---"
	make generate-openapi-schema
	uvicorn main:app --reload --host 0.0.0.0 --port 10000

generate-openapi-schema:
	@echo "--- 📜 Generating OpenAPI schema ---"
	uv run generate_openapi.py main:app --out openapi/openapi.yaml
	@echo "OpenAPI schema generated successfully."

help:
	@echo "Makefile for the Lex LLM project"
	@echo ""
	@echo "Available commands:"
	@echo "  install            Install project dependencies"
	@echo "  run                Run the application"
	@echo "  static-type-check  Run static type checks"
	@echo "  lint               Run linters"
	@echo "  lint-check         Check if the project is linted"
	@echo "  test               Run tests"
	@echo "  pr                 Run all checks for a pull request"
	@echo "  help               Show this help message"
