install:
	@echo "--- 🚀 Installing project ---"
	uv sync

generate-api-docker: # Can be run without installing openapi-generator-cli
	@echo "--- 🔧 Generating API client (docker) ---"
	@mkdir -p build
	docker run --rm \
		-v ${PWD}:/local \
		openapitools/openapi-generator-cli generate \
		-i /local/openapi/lex-db.yaml \
		-g python \
		-o /local/build/lex_db_api \
		--additional-properties=packageName=lex_db_api

generate-api:
	@echo "--- 🔧 Generating API client (local) ---"
	@mkdir -p build
	uv run openapi-generator generate \
		-i openapi/lex-db.yaml \
		-g python \
		-o build/lex_db_api \
		--additional-properties=packageName=lex_db_api

clean-api:
	@echo "--- 🧹 Cleaning generated client ---"
	rm -rf build/

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	mypy src/ 

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
	@echo "Ready to make a PR"
