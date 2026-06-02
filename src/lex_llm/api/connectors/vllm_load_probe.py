"""Lightweight on-demand probe of a vLLM /metrics endpoint."""

import re
import time
import logging
import httpx

_LOGGER = logging.getLogger(__name__)

# ── vLLM metric-line parser ──────────────────────────────────────────
# Example line:
#   vllm:num_requests_waiting{model_name="gemma-4-26B-A4B-it"} 3.0
_METRIC_RE = re.compile(r"^(vllm:\w+)\{.*\bmodel_name=\"([^\"]+)\".*\}\s+([\d.eE+\-]+)")


class VLLMLoadProbe:
    """Scrape vLLM prometheus metrics on demand and decide if the
    backend should be treated as overloaded.

    *Fail-closed*: if the scrape itself fails (timeout, connection
    refused, etc.), ``is_overloaded()`` returns ``(True, reason)`` so
    the router falls back to the cloud provider rather than risking a
    user-visible failure.

    Parameters
    ----------
    metrics_url:
        Full URL to the vLLM metrics endpoint, e.g.
        ``http://dgx:80/metrics/gemma-large``.
    model_name:
        vLLM served-model-name label (e.g. ``"gemma-4-26B-A4B-it"``).
        Only metrics tagged with this model are evaluated.
    ttl:
        Minimum seconds between actual scrapes.  Back-to-back calls
        within this window return the previous decision.
    max_waiting:
        If ``num_requests_waiting >= max_waiting``, the backend is
        considered overloaded.
    max_tpot_seconds:
        If the mean time-per-output-token exceeds this threshold,
        the backend is considered overloaded (tokens are being
        produced too slowly).
    timeout:
        HTTP timeout for the metrics scrape.
    """

    def __init__(
        self,
        metrics_url: str,
        model_name: str,
        *,
        ttl: float = 1.0,
        max_waiting: int = 4,
        max_tpot_seconds: float = 0.15,
        timeout: float = 1.5,
    ) -> None:
        self._metrics_url = metrics_url
        self._model_name = model_name
        self._ttl = ttl
        self._max_waiting = max_waiting
        self._max_tpot_seconds = max_tpot_seconds
        self._timeout = timeout

        # Cached decision
        self._last_check: float = 0.0
        self._cached_overloaded: bool = False
        self._cached_reason: str = "not yet probed"

    # ── public API ───────────────────────────────────────────────────

    async def is_overloaded(self) -> tuple[bool, str]:
        """Return ``(overloaded: bool, reason: str)``.

        Reasons include ``"ok"`` when healthy, or a human-readable
        description of the overload condition (or scrape error).
        """
        now = time.monotonic()
        if now - self._last_check < self._ttl:
            return self._cached_overloaded, self._cached_reason

        try:
            overloaded, reason = await self._scrape()
        except Exception as exc:
            _LOGGER.warning("Metrics scrape failed: %s", exc)
            overloaded, reason = True, f"scrape error: {exc}"

        self._last_check = now
        self._cached_overloaded = overloaded
        self._cached_reason = reason
        return overloaded, reason

    # ── internal ─────────────────────────────────────────────────────

    async def _scrape(self) -> tuple[bool, str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._metrics_url)
            resp.raise_for_status()
            text = resp.text

        running = waiting = 0
        tpot_sum = tpot_count = 0.0

        for line in text.splitlines():
            m = _METRIC_RE.match(line.strip())
            if m is None:
                continue
            metric_name, model_label, value_str = m.groups()
            if model_label != self._model_name:
                continue
            value = float(value_str)

            if metric_name == "vllm:num_requests_running":
                running = int(value)
            elif metric_name == "vllm:num_requests_waiting":
                waiting = int(value)
            elif metric_name == "vllm:request_time_per_output_token_seconds_sum":
                tpot_sum = value
            elif metric_name == "vllm:request_time_per_output_token_seconds_count":
                tpot_count = value

        tpot_mean = (tpot_sum / tpot_count) if tpot_count > 0 else None

        _LOGGER.debug(
            "vLLM metrics: running=%d waiting=%d tpot_mean=%s",
            running,
            waiting,
            f"{tpot_mean:.3f}" if tpot_mean is not None else "N/A",
        )

        if waiting >= self._max_waiting:
            return True, f"queue depth {waiting} >= {self._max_waiting}"
        if tpot_mean is not None and tpot_mean > self._max_tpot_seconds:
            return True, f"tpot {tpot_mean:.3f}s > {self._max_tpot_seconds}s"
        return False, "ok"
