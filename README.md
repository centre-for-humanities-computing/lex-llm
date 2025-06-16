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

## Communication with LexDB

This project communicates with LexDB through an auto-generated OpenAPI client, providing a robust interface for database operations.

### For Developers
- The OpenAPI client is located in `build/lex-db-client`
- Main operations supported:
  - Retrieving available tables
  - Full-text search across documents
  - Vector-based semantic search using embeddings
- Example usage can be found in `src/scripts/lex_db_search_example.py`

Example configuration:
```python
from lex_db_client.configuration import Configuration
from lex_db_client.api.lex_db_api import LexDbApi
from lex_db_client.api_client import ApiClient

api_host = "http://0.0.0.0:8000"  # Configure your LexDB host
api_client = ApiClient(configuration=Configuration(host=api_host))
api = LexDbApi(api_client=api_client)
```

### For Users
The LexDB integration enables:
- Fast full-text search across all stored documents
- Semantic search using vector embeddings
- Access to structured document metadata and content

The database connection is automatically handled by the application, requiring no additional setup from end users.


