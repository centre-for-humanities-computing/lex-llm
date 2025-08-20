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
> 🚀 **Tip for Contributors:** Run `make pr` before submitting changes to ensure linting, typing, and tests pass.
