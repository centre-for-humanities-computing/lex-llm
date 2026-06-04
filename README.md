# Lex LLM
The orchestrator for the Lex LLM project, enabling intelligent workflow execution and AI-driven responses.

## Repositories
This project is part of the broader Lex ecosystem, which includes several interconnected repositories:

| Name      | Description                 |
|---------|-----------------------------|
| [lex-llm] | Core Python orchestration logic and API |
| [lex-ui]  | Frontend interface for Lex users |
| [lex-db]  | Database backend with document storage and search |

[lex-llm]: https://github.com/centre-for-humanities-computing/lex-llm  
[lex-ui]: https://github.com/centre-for-humanities-computing/lex-llm-ui  
[lex-db]: https://github.com/centre-for-humanities-computing/lex-db

---

## API Structure
The main API is defined in [`src/lex_llm/api/routes.py`](src/lex_llm/api/routes.py). The available endpoints are:

- **POST `/workflows/{workflow_id}/run`**  
  Executes a specific workflow by ID. Accepts a JSON payload containing user input, conversation history, and conversation ID. Returns a streaming response in NDJSON format.

- **GET `/workflows/metadata`**  
  Retrieves metadata for all available workflows. Useful for discovering what workflows are supported by the system.

- **GET `/workflows/{workflow_id}/metadata`**  
  Retrieves metadata for a specific workflow by ID. Returns 404 if the workflow does not exist, along with a list of available workflows.

- **GET `/health`**  
  Simple health check endpoint. Returns `{"status": "healthy"}` when the service is running.

- **Lifespan Events**  
  On startup and shutdown, logs are printed to indicate the state of the AI Orchestration Service.

### Example: Calling the Workflow API
Use `curl` to stream results from a running workflow:

```bash
curl -N -X POST "http://0.0.0.0:10000/workflows/test_workflow/run" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Tell me about artificial intelligence.",
    "conversation_history": [
      {"role": "user", "content": "Hi!"},
      {"role": "assistant", "content": "Hello, how can I help you?"}
    ],
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

> 💡 Note: The `-N` flag disables buffering to ensure streaming works properly.

---

## Running the API
This project uses a `Makefile` to simplify common development and deployment tasks.

### Key Commands
- **Run the production API:**  
  ```bash
  make run
  ```
  Generates the OpenAPI schema and starts the application server.

- **Run in development mode (with hot reload):**  
  ```bash
  make run-dev
  ```
  Installs dev dependencies, generates the schema, and starts Uvicorn with auto-reload enabled on port `10000`.

- **Other Useful Commands:**
  | Command | Description |
  |--------|-------------|
  | `make install` | Install project and API client dependencies |
  | `make install-dev` | Install development dependencies |
  | `make lint` | Format and fix code using `ruff` |
  | `make lint-check` | Check formatting and linting without fixing |
  | `make static-type-check` | Run `mypy` for type checking |
  | `make test` | Run unit tests with `pytest` |
  | `make pr` | Run all checks required for a pull request |
  | `make generate-api` | Generate OpenAPI client from `lex-db.yaml` |
  | `make generate-openapi-schema` | Generate OpenAPI schema (`openapi/openapi.yaml`) |

> 📌 Tip: Always run `make pr` before pushing changes to ensure everything is consistent.

---

## Communication with LexDB
This project integrates with **LexDB** via an auto-generated OpenAPI client, enabling robust interaction with the database.

### For Developers
- The OpenAPI client is generated using `make generate-api`, which uses Docker and OpenAPI Generator.
- Generated client is located at `build/lex_db_api`.
- Supported operations include:
  - Listing available tables
  - Full-text search across articles
  - Vector-based semantic search using embeddings
- Example usage can be found in `src/examples/lex_db_search_example.py`.

#### Example Usage
```python
from lex_db_api.configuration import Configuration
from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.models.vector_search_request import VectorSearchRequest
from lex_db_api.api_client import ApiClient
import os

api_host = os.getenv("DB_HOST", "http://0.0.0.0:8000")
api_client = ApiClient(configuration=Configuration(host=api_host))
api = LexDbApi(api_client=api_client)

# Get available tables
tables = api.get_tables()
print("Tables:", tables)

# Full-text search
results_fts = api.get_articles(query="Rundetårn", limit=2)
print("Full-text results:", results_fts)

# Vector search
req_vector = VectorSearchRequest(
    query_text="Hvad er Rundetårn?",
    top_k=3,
)
results_vector = api.vector_search("openai_large_3_sections", req_vector)
print("Vector search results:", results_vector)

# Fetch full articles from result IDs
if results_vector.results:
    article_ids = {int(result["source_article_id"]) for result in results_vector.results}
    full_articles = api.get_articles(ids=str(list(article_ids)))
    for article in full_articles:
        print(article)
```

### For Users
The LexDB integration enables:
- Fast full-text search across all documents
- Semantic search using vector embeddings
- Access to structured metadata and article content

All database interactions are handled automatically by the application, so no manual setup is required for end users.

---

## Observability

Each workflow run streams a structured NDJSON event sequence.  The observability
extensions add per-request timing, per-step backend routing information, and a
persistent JSONL log for offline analysis.

### Events added / extended

| Event | When | Data |
|-------|------|------|
| `workflow_step` ("completed") | After each step | `output.duration_ms` + `output.llm_calls[*]` per LLM call in the step |
| `workflow_step` ("failed") | If a step raises | `output.duration_ms` + `error` |
| `workflow_metrics` | Immediately before `stream_end` | `workflow_id`, `e2e_ms`, `ttft_any_ms`, `ttft_answer_ms`, `backend_summary`, `step_count`, `outcome` |

### TTFT semantics

- `ttft_any_ms` — monotonic time from `execute()` entry to the first chunk-like
  event (`text_chunk`, `lead_paragraph`, `answer_body`, `interpretation`).
- `ttft_answer_ms` — time to the first answer-body event (`text_chunk` or
  `answer_body`).  Always ≤ `ttft_any_ms` because any chunk is counted as
  "any".

### Routing telemetry

Each `LLMProvider` that supports routing (`RoutingLLMProvider`) exposes an
`observe()` context manager.  Steps that wrap their LLM call in
`async with llm_provider.observe(callback):` capture a `RouteDecision` with:

| Field | Values |
|-------|--------|
| `backend` | `"primary"` or `"fallback"` |
| `trigger` | `"ok"`, `"probe_overload"`, `"probe_scrape_error"`, `"primary_pre_first_token_error"` |
| `reason` | Human-readable explanation (queue depth, exception text, etc.) |
| `model` | Model name on the selected backend |

These decisions appear as `output.llm_calls[*]` in the `workflow_step`
"completed" event.

### JSONL recorder

A bounded async queue writes one JSON line per request to
`LEX_LLM_TELEMETRY_DIR` (default `./telemetry/`), daily-rotated.  Each row
includes:

```json
{
  "ts": "2026-06-03T12:00:00+0000",
  "conversation_id": "...",
  "run_id": "...",
  "workflow_id": "beta_workflow_v4_local",
  "outcome": "ok",
  "user_input_len": 42,
  "e2e_ms": 4520.12,
  "ttft_any_ms": 312.45,
  "ttft_answer_ms": 312.45,
  "step_count": 4
}
```

The recorder is lossless under normal load and drops rows (with a log warning)
if the internal queue exceeds 10 000 pending items, so it never back-pressures
the request path.

### Trace propagation to the inference server

`DGXProvider` sends the orchestrator's `run_id` as the `X-Lex-Run-Id` HTTP
header on every request to the DGX Spark.  Configure nginx to log it with:

```
log_format lex '$remote_addr - $remote_user [$time_local] '
               '"$request" $status $body_bytes_sent '
               '"$http_x_lex_run_id"';
```

This lets you join nginx access logs with the JSONL rows by `run_id`.
