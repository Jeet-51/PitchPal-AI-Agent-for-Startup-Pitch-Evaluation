"""
PitchPal v2 - IP-Based Rate Limiter

Limits the number of FRESH evaluations per IP per time window.
Cache hits are NOT counted (zero API cost, so no need to limit).

Config (via .env):
  RATE_LIMIT_STARTUP_MAX=3       # max fresh evals for startup role per window
  RATE_LIMIT_INVESTOR_MAX=5      # max fresh evals for investor role per window
  RATE_LIMIT_WINDOW_HOURS=24     # reset window in hours
"""

import time
import threading
import logging
from typing import Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a client IP has exceeded their evaluation limit."""

    def __init__(self, limit: int, window_hours: int, retry_after_seconds: int):
        self.limit = limit
        self.window_hours = window_hours
        self.retry_after_seconds = max(0, retry_after_seconds)
        hours = retry_after_seconds // 3600
        minutes = (retry_after_seconds % 3600) // 60
        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"
        super().__init__(
            f"You've used all {limit} free evaluations for this {window_hours}-hour window. "
            f"Resets in {time_str}."
        )


class RateLimiter:
    """
    Thread-safe in-memory IP rate limiter.

    Structure: { ip: { role: { count: int, window_start: float } } }

    Cache hits bypass this entirely — only fresh LLM+Tavily calls are counted.
    """

    def __init__(self):
        # { ip_str: { role_str: { "count": int, "window_start": float } } }
        self._data: Dict[str, Dict[str, dict]] = {}
        self._lock = threading.Lock()

    # ── Helpers ──────────────────────────────────────────────

    def _window_seconds(self) -> int:
        return settings.RATE_LIMIT_WINDOW_HOURS * 3600

    def _get_limit(self, role: str) -> int:
        if role == "investor":
            return settings.RATE_LIMIT_INVESTOR_MAX
        return settings.RATE_LIMIT_STARTUP_MAX

    # ── Public API ───────────────────────────────────────────

    def check_and_increment(self, ip: str, role: str) -> tuple[int, int]:
        """
        Check the rate limit for this IP + role and increment the counter.

        Returns (current_count, limit).
        Raises RateLimitExceeded if the limit has been hit.

        IMPORTANT: Only call this for FRESH evaluations (not cache hits).
        """
        limit = self._get_limit(role)
        window = self._window_seconds()
        now = time.time()

        with self._lock:
            # Initialise tracking structures if needed
            if ip not in self._data:
                self._data[ip] = {}
            if role not in self._data[ip]:
                self._data[ip][role] = {"count": 0, "window_start": now}

            entry = self._data[ip][role]

            # Reset if the window has expired
            if now - entry["window_start"] > window:
                entry["count"] = 0
                entry["window_start"] = now
                logger.info(f"Rate limit window reset for ip={ip}, role={role}")

            # Check if over limit
            if entry["count"] >= limit:
                retry_after = int(entry["window_start"] + window - now)
                logger.warning(
                    f"Rate limit EXCEEDED: ip={ip}, role={role}, "
                    f"count={entry['count']}/{limit}, retry_after={retry_after}s"
                )
                raise RateLimitExceeded(
                    limit=limit,
                    window_hours=settings.RATE_LIMIT_WINDOW_HOURS,
                    retry_after_seconds=retry_after,
                )

            # Increment and return
            entry["count"] += 1
            current = entry["count"]
            logger.info(
                f"Rate limit OK: ip={ip}, role={role}, count={current}/{limit}"
            )
            return current, limit

    def get_status(self, ip: str, role: str) -> dict:
        """
        Return current rate limit status for an IP + role (no side effects).
        Safe to call on every page load for the frontend counter.
        """
        limit = self._get_limit(role)
        window = self._window_seconds()
        now = time.time()

        with self._lock:
            entry = self._data.get(ip, {}).get(role)

            if not entry or now - entry["window_start"] > window:
                return {
                    "count": 0,
                    "limit": limit,
                    "remaining": limit,
                    "window_hours": settings.RATE_LIMIT_WINDOW_HOURS,
                    "retry_after_seconds": 0,
                    "reset_at": None,
                }

            remaining = max(0, limit - entry["count"])
            retry_after = max(0, int(entry["window_start"] + window - now)) if remaining == 0 else 0

            return {
                "count": entry["count"],
                "limit": limit,
                "remaining": remaining,
                "window_hours": settings.RATE_LIMIT_WINDOW_HOURS,
                "retry_after_seconds": retry_after,
                "reset_at": int(entry["window_start"] + window),
            }

    def get_global_stats(self) -> dict:
        """Return aggregate stats for all tracked IPs (for /stats endpoint)."""
        with self._lock:
            return {
                "tracked_ips": len(self._data),
                "limits": {
                    "startup_max": settings.RATE_LIMIT_STARTUP_MAX,
                    "investor_max": settings.RATE_LIMIT_INVESTOR_MAX,
                    "window_hours": settings.RATE_LIMIT_WINDOW_HOURS,
                },
            }


# ── Singleton ─────────────────────────────────────────────────
rate_limiter = RateLimiter()
