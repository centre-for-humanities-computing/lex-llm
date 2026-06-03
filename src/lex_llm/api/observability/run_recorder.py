"""Bounded, non-blocking JSONL recorder for workflow telemetry.

Writes one JSON line per completed workflow run to a daily-rotated file.
Uses an asyncio.Queue to avoid blocking the hot path. Drops rows with a
warning when the queue is full so the recorder never back-pressures
request handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date
from typing import TextIO

_LOGGER = logging.getLogger(__name__)


class RunRecorder:
    """Async JSONL recorder with daily rotation and bounded queue.

    Start the recorder with ``await recorder.start()`` and stop it
    with ``await recorder.stop()`` (which drains the queue).  Submit
    rows with ``await recorder.submit(row_dict)`` — guaranteed
    non-blocking (discards if queue is full).
    """

    def __init__(
        self,
        directory: str = "",
        max_queue: int = 10_000,
    ) -> None:
        self._directory = directory or os.environ.get(
            "LEX_LLM_TELEMETRY_DIR", "./telemetry"
        )
        self._max_queue = max_queue
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self._file_handle: TextIO | None = None
        self._current_date: date | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background writer task."""
        os.makedirs(self._directory, exist_ok=True)
        self._task = asyncio.create_task(self._writer_loop())
        _LOGGER.info("RunRecorder started — %s", self._directory)

    async def stop(self) -> None:
        """Signal the writer to drain and stop."""
        if self._task is None:
            return
        # Sentinel to stop the loop
        await self._queue.put(None)
        await self._task
        self._close_file()
        _LOGGER.info("RunRecorder stopped")

    # ── submit ───────────────────────────────────────────────────────

    async def submit(self, row: dict) -> None:
        """Enqueue a row for writing.  Non-blocking — drops if full."""
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            _LOGGER.warning("RunRecorder queue full — dropping telemetry row")

    # ── internals ────────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        while True:
            row = await self._queue.get()
            if row is None:
                break
            try:
                self._write_row(row)
            except Exception:
                _LOGGER.exception("RunRecorder write error")
            finally:
                self._queue.task_done()

    def _write_row(self, row: dict) -> None:
        today = date.today()
        if today != self._current_date:
            self._rotate(today)
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        if self._file_handle:
            self._file_handle.write(line)
            self._file_handle.flush()

    def _rotate(self, today: date) -> None:
        self._close_file()
        path = os.path.join(self._directory, f"lex-llm-{today.isoformat()}.jsonl")
        self._file_handle = open(path, "a", encoding="utf-8")
        self._current_date = today
        _LOGGER.info("RunRecorder rotated to %s", path)

    def _close_file(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


# Module-level singleton — start/stop managed by routes.py lifespan
_recorder: RunRecorder | None = None


def get_recorder() -> RunRecorder:
    global _recorder
    if _recorder is None:
        _recorder = RunRecorder()
    return _recorder