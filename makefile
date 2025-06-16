install:
	@echo "--- 🚀 Installing project ---"
	uv sync

generate-api:
	@echo "--- 🔧 Generating API client ---"
	@mkdir -p build
	docker run --rm \
		-v ${PWD}:/local \
		openapitools/openapi-generator-cli generate \
		-i /local/openapi/lex-db.yaml \
		-g python \
		-o /local/build/lex-db-client \
		--additional-properties=packageName=lex_db_client,projectName=lex-db-client

# Alternative without Docker (requires openapi-generator to be installed)
generate-api-local:
	@echo "--- 🔧 Generating API client (local) ---"
	@mkdir -p build
	openapi-generator generate \
		-i openapi/lex-db.yaml \
		-g python \
		-o build/lex-db-client \
		--additional-properties=packageName=lex_db_client,projectName=lex-db-client

clean-api:
	@echo "--- 🧹 Cleaning generated client ---"
	rm -rf build/

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	mypy src/ --config-file mypy.ini

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
	make lint
	make static-type-check
	make test
	@echo "Ready to make a PR"
