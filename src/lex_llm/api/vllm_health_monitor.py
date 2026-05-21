# llm/health_monitor.py
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx
from prometheus_client.parser import text_string_to_metric_families

from lex_db.utils import get_logger

logger = get_logger()


@dataclass
class BackendHealth:
    """Snapshot of a vLLM backend's load state."""
    healthy: bool = True
    num_running: int = 0
    num_waiting: int = 0
    gpu_cache_usage: float = 0.0
    # Rolling p50 of recent TTFT in seconds (from prometheus histogram)
    ttft_p50: Optional[float] = None
    tpot_p50: Optional[float] = None  # time per output token
    last_updated: float = field(default_factory=time.time)
    consecutive_failures: int = 0


class VLLMHealthMonitor:
    """
    Background poller for a vLLM backend's /metrics endpoint.
    Provides a thread-safe overload decision used by the router.
    """

    def __init__(
        self,
        metrics_url: str,
        model_name: str,
        *,
        poll_interval: float = 1.5,
        # Overload thresholds — tune these to your benchmarks
        max_waiting: int = 4,            # queue depth threshold
        max_running: int = 10,           # your benchmarked concurrency limit
        max_tpot_seconds: float = 0.15,  # 0.15s/token == ~6.6 tok/s floor
        max_cache_usage: float = 0.90,
        failure_threshold: int = 3,      # consecutive scrape failures = unhealthy
    ):
        self.metrics_url = metrics_url
        self.model_name = model_name
        self.poll_interval = poll_interval
        self.max_waiting = max_waiting
        self.max_running = max_running
        self.max_tpot_seconds = max_tpot_seconds
        self.max_cache_usage = max_cache_usage
        self.failure_threshold = failure_threshold

        self._state = BackendHealth()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=2.0)

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
        await self._client.aclose()

    async def _poll_loop(self):
        while True:
            try:
                await self._scrape_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health monitor scrape failed: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _scrape_once(self):
        try:
            r = await self._client.get(self.metrics_url)
            r.raise_for_status()
        except Exception as e:
            async with self._lock:
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.failure_threshold:
                    self._state.healthy = False
                logger.warning(
                    f"Metrics scrape failed ({self._state.consecutive_failures}): {e}"
                )
            return

        running = waiting = cache = 0.0
        tpot_sum = tpot_count = ttft_sum = ttft_count = 0.0

        for family in text_string_to_metric_families(r.text):
            for sample in family.samples:
                # vLLM tags metrics with the served model name — filter to ours
                if sample.labels.get("model_name", self.model_name) != self.model_name:
                    continue
                name = sample.name
                if name == "vllm:num_requests_running":
                    running = sample.value
                elif name == "vllm:num_requests_waiting":
                    waiting = sample.value
                elif name == "vllm:gpu_cache_usage_perc":
                    cache = sample.value
                elif name == "vllm:time_per_output_token_seconds_sum":
                    tpot_sum = sample.value
                elif name == "vllm:time_per_output_token_seconds_count":
                    tpot_count = sample.value
                elif name == "vllm:time_to_first_token_seconds_sum":
                    ttft_sum = sample.value
                elif name == "vllm:time_to_first_token_seconds_count":
                    ttft_count = sample.value

        # Compute running averages from the cumulative histograms.
        # For a more accurate p50 you'd diff against the previous scrape, but
        # the cumulative mean is good enough for an overload signal.
        tpot_mean = (tpot_sum / tpot_count) if tpot_count > 0 else None
        ttft_mean = (ttft_sum / ttft_count) if ttft_count > 0 else None

        async with self._lock:
            self._state = BackendHealth(
                healthy=True,
                num_running=int(running),
                num_waiting=int(waiting),
                gpu_cache_usage=cache,
                tpot_p50=tpot_mean,
                ttft_p50=ttft_mean,
                last_updated=time.time(),
                consecutive_failures=0,
            )

    async def snapshot(self) -> BackendHealth:
        async with self._lock:
            # Return a copy so callers can't mutate internal state
            return BackendHealth(**self._state.__dict__)

    async def is_overloaded(self) -> tuple[bool, str]:
        """Returns (overloaded, reason). The reason is useful for logging/metrics."""
        s = await self.snapshot()
        if not s.healthy:
            return True, f"backend unhealthy ({s.consecutive_failures} scrape failures)"
        if s.num_waiting >= self.max_waiting:
            return True, f"queue depth {s.num_waiting} >= {self.max_waiting}"
        if s.num_running >= self.max_running:
            return True, f"running {s.num_running} >= {self.max_running}"
        if s.tpot_p50 is not None and s.tpot_p50 > self.max_tpot_seconds:
            return True, f"tpot {s.tpot_p50:.3f}s > {self.max_tpot_seconds}s"
        if s.gpu_cache_usage > self.max_cache_usage:
            return True, f"kv cache {s.gpu_cache_usage:.0%} > {self.max_cache_usage:.0%}"
        return False, "ok"