include .env
export $(shell sed 's/=.*//' .env)

install:
	@echo "--- 🚀 Installing project ---"
	uv sync

install-dev:
	@echo "--- 🚀 Installing development dependencies ---"
	uv sync --dev

generate-api: # Can be run without installing openapi-generator-cli
	@echo "--- 🔧 Generating API client (docker) ---"
	@mkdir -p build
	docker run --rm \
		-v ${PWD}:/local \
		openapitools/openapi-generator-cli generate \
		-i /local/openapi/lex-db.yaml \
		-g python \
		-o /local/build/lex_db_api \
		--additional-properties=packageName=lex_db_api

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
	make generate-api
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

