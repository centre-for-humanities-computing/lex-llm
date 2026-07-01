# Lex LLM

Orchestrator for the Lex project — runs AI workflows that search an encyclopedia
knowledge base and generate cited, editorial-quality answers in Danish.

## Repositories

| Name      | Description                 |
|-----------|-----------------------------|
| [lex-llm] | Core Python orchestration logic and API |
| [lex-db]  | Database backend with document storage and search |

[lex-llm]: https://github.com/centre-for-humanities-computing/lex-llm  
[lex-db]: https://github.com/centre-for-humanities-computing/lex-db

---

## Quick Start

```bash
# 1. Clone and set up environment variables
cp .env.example .env
# Edit .env — add your API keys and set DB_HOST

# 2. Install dependencies (uses uv)
make install

# 3. Run the server
make run        # production (port 8001)
make run-dev    # development with hot reload (port 10000)
```

The project uses **[uv](https://docs.astral.sh/uv/)** for dependency management.
All commands go through the `Makefile` — run `make help` to see everything.

---

## Architecture

```
main.py                          FastAPI entry point
└── lex_llm/api/routes.py        REST endpoints + lifespan
    ├── orchestrator.py          Step execution engine (sequential + parallel)
    ├── event_emitter.py         NDJSON streaming to clients
    ├── workflow_utils.py        Dynamic workflow module loader
    └── connectors/              LLM provider abstraction
        ├── llm_provider.py          Base interface
        ├── routing_llm_provider.py  Primary/fallback with load probing
        ├── dgx_provider.py          Self-hosted inference (DGX Spark)
        ├── scaleway_provider.py     Scaleway Generative APIs
        ├── openai_provider.py       OpenAI
        ├── openrouter_provider.py   OpenRouter
        └── cortecs_provider.py      Cortecs

lex_llm/workflows/               One file per workflow variant
lex_llm/tools/                   Reusable step functions (search, generation, etc.)
lex_llm/utils/                   Helpers (RRF fusion, retrieval utilities)
lex_llm/prompts_search_synthesis.py   Danish prompt templates for the search pipeline
```

### Workflow plugin system

Workflows are discovered automatically. Drop a `.py` file in
`src/lex_llm/workflows/` that exposes two functions, and it becomes callable
via the API with no registration needed:

```python
# my_workflow.py
def get_workflow(request: WorkflowRunRequest) -> Orchestrator:
    ...

def get_metadata() -> dict:
    return {"id": "my_workflow", "name": "...", "description": "..."}
```

The API reads the filesystem at startup — `POST /workflows/my_workflow/run`
just works. See existing workflows in `src/lex_llm/workflows/` for examples.

### LLM providers

All LLM calls go through a common `LLMProvider` interface. The
`RoutingLLMProvider` adds automatic primary/fallback routing: it probes the
local inference server's `/metrics` endpoint and falls back to a cloud provider
if the local backend is overloaded or unreachable.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workflows/{workflow_id}/run` | Run a workflow. Returns NDJSON stream. |
| `GET`  | `/workflows/metadata` | List all available workflows and their metadata. |
| `GET`  | `/workflows/{workflow_id}/metadata` | Metadata for a single workflow. |
| `GET`  | `/health` | Health check. |

### Example request

```bash
curl -N -X POST "http://0.0.0.0:8001/workflows/test_workflow/run" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Hvad er Rundetårn?",
    "conversation_history": [
      {"role": "user", "content": "Hej!"},
      {"role": "assistant", "content": "Hej, hvordan kan jeg hjælpe?"}
    ],
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

> Use port `10000` for `make run-dev`. The `-N` flag disables buffering so
> streaming works correctly.

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies + generate LexDB API client |
| `make install-dev` | Install with dev dependencies |
| `make run` | Start production server (port 8001) |
| `make run-dev` | Start with hot reload (port 10000) |
| `make lint` | Format and fix with ruff |
| `make lint-check` | Check formatting without fixing |
| `make static-type-check` | Run mypy |
| `make test` | Run pytest |
| `make pr` | Full PR checklist (install, lint, type-check, test, schema) |
| `make generate-api` | Regenerate LexDB client from `openapi/lex-db.yaml` |
| `make generate-openapi-schema` | Write OpenAPI schema to `openapi/openapi.yaml` |

Always run `make pr` before pushing.

---

## LexDB Integration

The project talks to [LexDB](https://github.com/centre-for-humanities-computing/lex-db)
through an auto-generated OpenAPI client at `build/lex_db_api/`. The client is
regenerated with `make generate-api` (requires Docker).

The high-level connector in `src/lex_llm/api/connectors/lex_db_connector.py`
wraps the generated client with typed models (`LexChunk`, `LexArticle`) and
helper functions (chunk grouping, RRF fusion). Workflow tools use this
connector — you rarely need the raw generated client directly.

For a standalone usage example, see
[`src/examples/lex_db_search_example.py`](src/examples/lex_db_search_example.py).

---

## Observability

Each workflow run produces a structured NDJSON event stream. The observability
layer adds per-request timing, per-step LLM routing decisions, and a persistent
JSONL log for offline analysis. See the source in
`src/lex_llm/api/observability/`.

### Events

| Event | When | Key fields |
|-------|------|------------|
| `workflow_step` (completed) | After each step | `output.duration_ms`, `output.llm_calls[*]` |
| `workflow_step` (failed) | If a step raises | `output.duration_ms`, `error` |
| `workflow_metrics` | Before `stream_end` | `e2e_ms`, `ttft_any_ms`, `ttft_answer_ms`, `backend_summary`, `step_count`, `outcome` |

### TTFT semantics

- **`ttft_any_ms`** — time from workflow start to the first chunk-like event
  (`text_chunk`, `lead_paragraph`, `answer_body`, `interpretation`).
- **`ttft_answer_ms`** — time to the first answer-body event. Always >= `ttft_any_ms`.

### Routing telemetry

When a step uses `RoutingLLMProvider`, each LLM call records a route decision:

| Field | Values |
|-------|--------|
| `backend` | `"primary"` or `"fallback"` |
| `trigger` | `"ok"`, `"probe_overload"`, `"probe_scrape_error"`, `"primary_pre_first_token_error"` |
| `reason` | Human-readable explanation |
| `model` | Model name on the selected backend |

### JSONL recorder

Writes one JSON line per request to `LEX_LLM_TELEMETRY_DIR` (default
`./telemetry/`), daily-rotated. Non-blocking — drops rows with a warning if the
internal queue exceeds 10 000 items.

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

### Trace propagation

`DGXProvider` sends the orchestrator's `run_id` as the `X-Lex-Run-Id` HTTP
header. Configure nginx to log it:

```
log_format lex '$remote_addr - $remote_user [$time_local] '
               '"$request" $status $body_bytes_sent '
               '"$http_x_lex_run_id"';
```

Join nginx access logs with JSONL rows by `run_id`.
