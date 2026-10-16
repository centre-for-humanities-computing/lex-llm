"""Errors reach the caller with their type and cause, and clients time out."""

import importlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, AsyncGenerator

import httpx
import openai
import pytest
import urllib3
from fastapi import FastAPI
from fastapi.testclient import TestClient
from lex_db_api.api.lex_db_api import LexDbApi
from lex_db_api.api_client import ApiClient
from lex_db_api.configuration import Configuration
from lex_db_api.exceptions import ApiException
from lex_db_api.models.text_type import TextType

from lex_llm.api import routes
from lex_llm.api.connectors import cortecs_provider, lex_db_connector
from lex_llm.api.event_emitter import EventEmitter
from lex_llm.api.event_models import ConversationMessage, WorkflowRunRequest
from lex_llm.api.orchestrator import Orchestrator
from lex_llm.utils.errors import describe_exception


# --------------------------------------------------------------------------- #
# Local servers                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def hanging_server() -> Iterator[str]:
    """Accepts connections and reads the request, but never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    held: list[socket.socket] = []
    stop = threading.Event()

    def serve() -> None:
        sock.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
                held.append(conn)
            except OSError:
                continue

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    stop.set()
    thread.join()
    for conn in held:
        conn.close()
    sock.close()


@pytest.fixture
def error_server() -> Iterator[str]:
    """Answers every POST like lex-db does when a search fails."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(
                {"detail": "Search error: PoolTimeout: couldn't get a connection"}
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


# --------------------------------------------------------------------------- #
# describe_exception                                                          #
# --------------------------------------------------------------------------- #


def test_describe_exception_includes_type() -> None:
    assert describe_exception(ValueError("bad value")) == "ValueError: bad value"


def test_describe_exception_empty_message_gives_type_only() -> None:
    assert describe_exception(TimeoutError()) == "TimeoutError"


def test_describe_exception_flattens_newlines() -> None:
    assert describe_exception(RuntimeError("a\nb\n  c")) == "RuntimeError: a b c"


def test_describe_exception_follows_explicit_cause() -> None:
    request = httpx.Request("POST", "https://api.cortecs.ai/v1/chat/completions")
    try:
        try:
            raise httpx.ConnectError("[Errno 111] Connection refused")
        except httpx.ConnectError as inner:
            raise openai.APIConnectionError(request=request) from inner
    except openai.APIConnectionError as exc:
        described = describe_exception(exc)
    assert described == (
        "APIConnectionError: Connection error. | caused by: "
        "ConnectError: [Errno 111] Connection refused"
    )


def test_describe_exception_follows_implicit_context() -> None:
    try:
        try:
            raise KeyError("x")
        except KeyError:
            raise RuntimeError("while handling")
    except RuntimeError as exc:
        described = describe_exception(exc)
    assert described == "RuntimeError: while handling | caused by: KeyError: 'x'"


def test_describe_exception_respects_from_none() -> None:
    try:
        try:
            raise KeyError("x")
        except KeyError:
            raise RuntimeError("clean") from None
    except RuntimeError as exc:
        described = describe_exception(exc)
    assert described == "RuntimeError: clean"


def test_describe_exception_skips_causes_already_in_the_message() -> None:
    try:
        try:
            try:
                raise TimeoutError()  # httpcore.ReadTimeout
            except TimeoutError as a:
                raise TimeoutError() from a  # httpx.ReadTimeout, same name
        except TimeoutError as b:
            raise RuntimeError(f"Max retries exceeded (Caused by {b!r})") from b
    except RuntimeError as exc:
        described = describe_exception(exc)
    assert described == "RuntimeError: Max retries exceeded (Caused by TimeoutError())"


def test_describe_exception_survives_cycles() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert describe_exception(a) == "ValueError: a | caused by: ValueError: b"


# --------------------------------------------------------------------------- #
# Orchestrator error events                                                   #
# --------------------------------------------------------------------------- #


def _request() -> WorkflowRunRequest:
    return WorkflowRunRequest(
        conversation_id="conv-err",
        user_input="Hvad er en fregat?",
        conversation_history=[ConversationMessage(role="user", content="Hej")],
    )


async def _run(orch: Orchestrator) -> list[dict[str, Any]]:
    return [json.loads(e) async for e in orch.execute()]


def _error_message(events: list[dict[str, Any]]) -> str:
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    message: str = errors[0]["data"]["message"]
    return message


@pytest.mark.asyncio
async def test_error_event_carries_type_and_cause() -> None:
    async def _generate(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        request = httpx.Request("POST", "https://api.cortecs.ai/v1/chat/completions")
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as inner:
            raise openai.APITimeoutError(request=request) from inner
        yield None

    events = await _run(Orchestrator(_request(), [(_generate, "")]))

    expected = "APITimeoutError: Request timed out. | caused by: ReadTimeout: timed out"
    assert _error_message(events) == (
        f"Workflow failed at step '_generate': {expected}"
    )
    failed = [
        e["data"]
        for e in events
        if e["event"] == "workflow_step" and e["data"]["status"] == "failed"
    ]
    assert failed[0]["error"] == expected


@pytest.mark.asyncio
async def test_error_event_with_empty_exception_message() -> None:
    async def _search(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        raise TimeoutError()
        yield None

    events = await _run(Orchestrator(_request(), [(_search, "")]))
    assert _error_message(events) == "Workflow failed at step '_search': TimeoutError"


def _error_data(events: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    data: dict[str, Any] = errors[0]["data"]
    return data


def _failing_step(exc: Exception) -> Any:
    async def _step(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        raise exc
        yield None

    return _step


@pytest.mark.asyncio
async def test_error_event_code_is_cortecs_status() -> None:
    request = httpx.Request("POST", "https://api.cortecs.ai/v1/chat/completions")
    exc = openai.RateLimitError(
        "Error code: 429", response=httpx.Response(429, request=request), body=None
    )

    events = await _run(Orchestrator(_request(), [(_failing_step(exc), "")]))

    assert _error_data(events)["code"] == "429"


@pytest.mark.asyncio
async def test_error_event_code_is_lexdb_status(
    monkeypatch: pytest.MonkeyPatch, error_server: str
) -> None:
    _point_lexdb_at(monkeypatch, error_server)

    async def _search(
        context: dict[str, Any], emitter: EventEmitter
    ) -> AsyncGenerator[str | None, None]:
        await lex_db_connector.LexDBConnector().batch_fulltext_search(["fregat"])
        yield None

    events = await _run(Orchestrator(_request(), [(_search, "")]))

    assert _error_data(events)["code"] == "500"


@pytest.mark.asyncio
async def test_error_event_code_empty_without_status() -> None:
    events = await _run(Orchestrator(_request(), [(_failing_step(TimeoutError()), "")]))

    # code is None and the emitter drops None fields, as before this change.
    assert "code" not in _error_data(events)


# --------------------------------------------------------------------------- #
# LexDB connector: errors propagate, calls time out                           #
# --------------------------------------------------------------------------- #


def _point_lexdb_at(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    client = ApiClient(configuration=Configuration(host=host))
    monkeypatch.setattr(lex_db_connector, "lexdb_api", LexDbApi(api_client=client))


@pytest.mark.asyncio
async def test_lexdb_http_error_propagates_with_status_and_body(
    monkeypatch: pytest.MonkeyPatch, error_server: str
) -> None:
    _point_lexdb_at(monkeypatch, error_server)
    connector = lex_db_connector.LexDBConnector()

    with pytest.raises(ApiException) as info:
        await connector.batch_fulltext_search(["fregat"])
    assert info.value.status == 500
    assert "PoolTimeout: couldn't get a connection" in describe_exception(info.value)

    with pytest.raises(ApiException):
        await connector.batch_vector_search([("fregat", TextType.QUERY)])


@pytest.mark.asyncio
async def test_lexdb_unreachable_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as s:  # grab a free port, then close it
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    _point_lexdb_at(monkeypatch, f"http://127.0.0.1:{port}")

    with pytest.raises(urllib3.exceptions.MaxRetryError) as info:
        await lex_db_connector.LexDBConnector().batch_vector_search(
            [("fregat", TextType.QUERY)]
        )
    assert "NewConnectionError" in describe_exception(info.value)


@pytest.mark.asyncio
async def test_lexdb_call_times_out(
    monkeypatch: pytest.MonkeyPatch, hanging_server: str
) -> None:
    _point_lexdb_at(monkeypatch, hanging_server)
    monkeypatch.setattr(lex_db_connector, "_LEXDB_TIMEOUT", (1.0, 0.5))

    t0 = time.perf_counter()
    with pytest.raises(urllib3.exceptions.ReadTimeoutError):
        await lex_db_connector.LexDBConnector().batch_fulltext_search(["fregat"])
    assert time.perf_counter() - t0 < 5


def test_lexdb_default_timeout() -> None:
    assert lex_db_connector._LEXDB_TIMEOUT == (5.0, 35.0)


# --------------------------------------------------------------------------- #
# Cortecs provider timeouts                                                   #
# --------------------------------------------------------------------------- #


def test_cortecs_client_has_explicit_timeout_and_retries() -> None:
    client = cortecs_provider.CortecsProvider()._client
    assert client.timeout == httpx.Timeout(60.0, connect=5.0)
    assert client.max_retries == 1


@pytest.fixture
def cortecs_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Reload the module so env overrides are read; restore it afterwards."""
    yield lambda: importlib.reload(cortecs_provider)
    monkeypatch.undo()
    importlib.reload(cortecs_provider)


@pytest.mark.asyncio
async def test_cortecs_times_out_waiting_for_first_token(
    monkeypatch: pytest.MonkeyPatch, hanging_server: str, cortecs_module: Any
) -> None:
    monkeypatch.setenv("CORTECS_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CORTECS_MAX_RETRIES", "0")
    monkeypatch.setenv("CORTECS_BASE_URL", hanging_server)
    monkeypatch.setenv("CORTECS_API_KEY", "test")
    module = cortecs_module()

    provider = module.CortecsProvider()
    assert provider._client.max_retries == 0

    t0 = time.perf_counter()
    with pytest.raises(openai.APITimeoutError) as info:
        await provider.generate([ConversationMessage(role="user", content="Hej")])
    assert time.perf_counter() - t0 < 5
    assert describe_exception(info.value).startswith(
        "APITimeoutError: Request timed out. | caused by: ReadTimeout"
    )


@pytest.fixture
def stalling_stream_server() -> Iterator[str]:
    """Starts an SSE completion, sends one token, then goes silent."""
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk = {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "Hej"}}],
            }
            data = f"data: {json.dumps(chunk)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            release.wait(10)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    release.set()
    server.shutdown()
    server.server_close()


@pytest.mark.asyncio
async def test_cortecs_times_out_between_tokens(
    monkeypatch: pytest.MonkeyPatch, stalling_stream_server: str, cortecs_module: Any
) -> None:
    monkeypatch.setenv("CORTECS_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CORTECS_BASE_URL", stalling_stream_server)
    monkeypatch.setenv("CORTECS_API_KEY", "test")
    provider = cortecs_module().CortecsProvider()

    received: list[str] = []
    t0 = time.perf_counter()
    with pytest.raises(httpx.ReadTimeout):
        async for token in provider.generate_stream(
            [ConversationMessage(role="user", content="Hej")]
        ):
            received.append(token)
    assert received == ["Hej"]
    assert time.perf_counter() - t0 < 5


# --------------------------------------------------------------------------- #
# Routes: unknown workflow ids                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture
def api_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


@pytest.mark.parametrize(
    "method, url",
    [("post", "/workflows/nope/run"), ("get", "/workflows/nope/metadata")],
)
def test_unknown_workflow_is_json_404(
    api_client: TestClient, method: str, url: str
) -> None:
    body = {"conversation_id": "c", "user_input": "hej", "conversation_history": []}
    response = api_client.request(method, url, json=body)

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["msg"] == "Workflow 'nope' not found."
    assert "editorial_workflow_v2_gemma4_26b" in detail["available_workflows"]


def test_known_workflow_metadata_still_served(api_client: TestClient) -> None:
    response = api_client.get("/workflows/editorial_workflow_v2_gemma4_26b/metadata")

    assert response.status_code == 200
    assert response.json()["workflow_id"] == "editorial_workflow_v2_gemma4_26b"
