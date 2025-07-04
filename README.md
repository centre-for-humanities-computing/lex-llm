# Lex LLM
The orchestrator for the Lex LLM

## Repositories

This project is a part of the Lex project, and consist of multiple repositories.

| Name      | Description                 |
| --------- | --------------------------- |
| [lex-llm] | The python code for Lex LLM |
| [lex-ui]  | The front end for Lex       |
| [lex-db]  | The database for Lex LLM    |

[lex-llm]: https://github.com/centre-for-humanities-computing/lex-llm
[lex-ui]: https://github.com/centre-for-humanities-computing/lex-llm-ui
[lex-db]: https://github.com/centre-for-humanities-computing/lex-db

---

## API Structure

The main API is defined in [`src/lex_llm/api/routes.py`](src/lex_llm/api/routes.py):

- **POST `/workflows/{workflow_id}/run`**  
  Runs a workflow with the given ID. Accepts a JSON body with user input, conversation history, and conversation ID. Returns a streaming NDJSON response.
- **GET `/health`**  
  Health check endpoint, returns a simple status.
- **Lifespan events**  
  Prints startup and shutdown messages for the orchestration service.

### Example: Calling the Workflow API

You can call the workflow endpoint using `curl` as follows:

```bash
curl -N -X POST "http://0.0.0.0:8000/workflows/test-workflow/run" \
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

---

## Running the API

The project uses a `makefile` to simplify common tasks. Key commands:

- **Run the API:**  
  ```bash
  make run
  ```
  This generates the OpenAPI schema and starts the application.

- **Run in development mode (hot reload):**  
  ```bash
  make run-dev
  ```
  This also generates the OpenAPI schema and starts the server with hot reload enabled.

- **Other useful commands:**  
  - `make lint` – Format and lint the code.
  - `make static-type-check` – Run static type checks.
  - `make test` – Run tests.
  - `make generate-api` – Generate the LexDB OpenAPI client.
  - `make generate-openapi-schema` – Generate the OpenAPI schema for this API.

---

## Communication with LexDB

This project communicates with LexDB through an auto-generated [OpenAPI](https://learn.openapis.org) client, providing a robust interface for database operations.

### For Developers
- The OpenAPI client can be generated with the `make generate-api` command (which is also called for `make pr`)
- When generated, the OpenAPI client is located in `build/lex_db_api`
- Main operations supported:
  - Retrieving available tables
  - Full-text search across documents
  - Vector-based semantic search using embeddings
- Example usage can be found in `src/examples/lex_db_search_example.py`

Example configuration and usage:
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
print("Results:", results_fts)

# Vector search
req_vector = VectorSearchRequest(
    query_text="Hvad er Rundetårn?",
    top_k=3,
)
results_vector = api.vector_search("small_003", req_vector)
print("VectorSearchResults:", results_vector)
if results_vector.results:
    article_ids = {
        int(result["source_article_id"]) for result in results_vector.results
    }
    print(article_ids)
    print(type(article_ids))
    full_articles = api.get_articles(ids=str(list(article_ids)))
    for article in full_articles:
        print(article)
```

### For Users
The LexDB integration enables:
- Fast full-text search across all stored documents
- Semantic search using vector embeddings
- Access to structured document metadata and content

The database connection is automatically handled by the application, requiring no additional setup from end users.


