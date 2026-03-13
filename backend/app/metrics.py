"""
PitchPal v2 - Request Metrics Collector

Tracks latency percentiles, cache performance, agent behaviour,
and error rates. Exposed via GET /metrics.

All operations are thread-safe (Lock-protected).
Resets on server restart — in-memory only (no persistence needed here).

Example /metrics response:
{
  "uptime_seconds": 3612,
  "evaluations": {
    "total": 24,
    "fresh": 9,
    "from_cache": 15,
    "cache_hit_rate": "62.5%"
  },
  "latency": {
    "p50_s": 28.4,
    "p75_s": 34.1,
    "p95_s": 41.2,
    "avg_s": 30.6
  },
  "agent": {
    "avg_tool_calls_per_eval": 5.3,
    "fact_checks_fired": 7,
    "fact_check_rate": "77.8%"
  },
  "errors": {
    "total_errors": 1,
    "rate_limits_hit": 3,
    "error_rate": "4.0%"
  }
}
"""

import math
import time
from threading import Lock
from typing import List


class MetricsCollector:
    """
    Singleton in-memory metrics store.
    Import the `metrics` instance at the bottom of this file.
    """

    def __init__(self):
        self._lock = Lock()
        self._latencies: List[float] = []   # seconds — fresh evals only
        self._evaluations: int = 0          # total successful completions
        self._cache_hits: int = 0           # eval-level cache hits
        self._cache_misses: int = 0         # fresh agent runs
        self._errors: int = 0               # evaluation errors
        self._tool_calls_total: int = 0     # total Tavily calls across all evals
        self._fact_checks_fired: int = 0    # evals that produced ≥1 contradiction
        self._rate_limits_hit: int = 0      # how often rate limiter blocked a request
        self._start_time: float = time.time()

    # ── Recording ─────────────────────────────────────────────

    def record_evaluation(
        self,
        latency_s: float,
        from_cache: bool,
        tool_calls: int = 0,
        contradictions: int = 0,
    ) -> None:
        """Call after every successful evaluation (cached or fresh)."""
        with self._lock:
            self._evaluations += 1
            if from_cache:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
                self._latencies.append(latency_s)
                self._tool_calls_total += tool_calls
                if contradictions > 0:
                    self._fact_checks_fired += 1

    def record_error(self) -> None:
        """Call when an evaluation fails with an exception."""
        with self._lock:
            self._errors += 1

    def record_rate_limit(self) -> None:
        """Call when a request is blocked by the rate limiter."""
        with self._lock:
            self._rate_limits_hit += 1

    # ── Stats snapshot ────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return a full metrics snapshot. Thread-safe read."""
        with self._lock:
            total = self._evaluations
            uptime_s = round(time.time() - self._start_time)
            misses = self._cache_misses
            cache_total = self._cache_hits + misses

            cache_hit_rate = (
                f"{self._cache_hits / cache_total * 100:.1f}%"
                if cache_total > 0 else "n/a"
            )
            avg_tools = (
                round(self._tool_calls_total / misses, 1)
                if misses > 0 else 0.0
            )
            fact_check_rate = (
                f"{self._fact_checks_fired / misses * 100:.1f}%"
                if misses > 0 else "n/a"
            )
            error_rate = (
                f"{self._errors / (total + self._errors) * 100:.1f}%"
                if (total + self._errors) > 0 else "0.0%"
            )

            return {
                "uptime_seconds": uptime_s,
                "evaluations": {
                    "total": total,
                    "fresh": misses,
                    "from_cache": self._cache_hits,
                    "cache_hit_rate": cache_hit_rate,
                },
                "latency": {
                    "p50_s": self._percentile(self._latencies, 50),
                    "p75_s": self._percentile(self._latencies, 75),
                    "p95_s": self._percentile(self._latencies, 95),
                    "avg_s": (
                        round(sum(self._latencies) / len(self._latencies), 2)
                        if self._latencies else 0.0
                    ),
                    "samples": len(self._latencies),
                },
                "agent": {
                    "avg_tool_calls_per_eval": avg_tools,
                    "fact_checks_fired": self._fact_checks_fired,
                    "fact_check_rate": fact_check_rate,
                },
                "errors": {
                    "total_errors": self._errors,
                    "rate_limits_hit": self._rate_limits_hit,
                    "error_rate": error_rate,
                },
            }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _percentile(data: List[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = math.ceil(pct / 100 * len(sorted_data)) - 1
        return round(sorted_data[max(0, idx)], 2)


# ── Singleton ─────────────────────────────────────────────────
metrics = MetricsCollector()
