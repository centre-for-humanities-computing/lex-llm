import sys
from pathlib import Path


def test_client_path_exists() -> None:
    """Test that the build directory with the client exists."""
    repo_root = Path(__file__).parent.parent.parent
    client_path = repo_root / "build" / "lex_db_api"
    assert client_path.exists(), (
        "Client directory not found. Did you run 'make generate-client'?"
    )


def test_can_import_client() -> None:
    """Test that we can import the client modules."""
    # Add client to path
    repo_root = Path(__file__).parent.parent.parent
    client_path = repo_root / "build" / "lex_db_api"
    sys.path.insert(0, str(client_path))

    # Try importing - will raise ImportError if something is wrong
    from lex_db_api.configuration import Configuration
    from lex_db_api.api.lex_db_api import LexDbApi
    from lex_db_api.api_client import ApiClient

    # Test basic object creation
    config = Configuration(host="http://dummy")
    client = ApiClient(configuration=config)
    api = LexDbApi(api_client=client)

    assert api is not None, "Failed to create API client"
