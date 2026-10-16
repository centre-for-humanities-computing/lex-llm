"""Readable one-line descriptions of exceptions."""

_MAX_CHAIN = 5


def describe_exception(exc: BaseException) -> str:
    """Render an exception and its causes as a single line.

    ``str(exc)`` alone is often not enough to tell what went wrong. The openai
    SDK raises ``APIConnectionError("Connection error.")`` with the real httpx
    error only in ``__cause__``, and exceptions such as ``TimeoutError()`` have
    an empty message. This includes the exception type and follows the cause
    chain, e.g.::

        APIConnectionError: Connection error. | caused by: ConnectError: [Errno 111] Connection refused

    Causes that would only repeat what is already there are skipped: one whose
    repr is embedded in the text (urllib3's MaxRetryError says "Caused by
    NewConnectionError(...)") or that renders the same as the previous part
    (httpx re-raises httpcore's identically named errors).
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < _MAX_CHAIN:
        seen.add(id(current))
        message = " ".join(str(current).split())
        name = type(current).__name__
        part = f"{name}: {message}" if message else name
        if not parts or (part != parts[-1] and repr(current) not in " ".join(parts)):
            parts.append(part)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return " | caused by: ".join(parts)
