"""
Unit tests for RateLimiter (app.agent.rate_limiter).
Each test creates a fresh RateLimiter instance for isolation.
"""

import time
import pytest
from unittest.mock import patch

from app.agent.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiter:

    def _make_limiter(self):
        return RateLimiter()

    def test_rate_limiter_allows_up_to_limit(self):
        limiter = self._make_limiter()
        # Default startup limit is 3
        for i in range(3):
            count, limit = limiter.check_and_increment("1.2.3.4", "startup")
            assert count == i + 1
            assert limit == 3

    def test_rate_limiter_blocks_after_limit(self):
        limiter = self._make_limiter()
        for _ in range(3):
            limiter.check_and_increment("1.2.3.4", "startup")

        with pytest.raises(RateLimitExceeded):
            limiter.check_and_increment("1.2.3.4", "startup")

    def test_rate_limiter_investor_has_higher_limit(self):
        limiter = self._make_limiter()
        for i in range(5):
            count, limit = limiter.check_and_increment("1.2.3.4", "investor")
            assert count == i + 1
            assert limit == 5

        with pytest.raises(RateLimitExceeded):
            limiter.check_and_increment("1.2.3.4", "investor")

    def test_rate_limiter_different_ips_independent(self):
        limiter = self._make_limiter()
        # Use up limit for IP A
        for _ in range(3):
            limiter.check_and_increment("10.0.0.1", "startup")
        with pytest.raises(RateLimitExceeded):
            limiter.check_and_increment("10.0.0.1", "startup")

        # IP B should still have full quota
        count, limit = limiter.check_and_increment("10.0.0.2", "startup")
        assert count == 1
        assert limit == 3

    def test_rate_limiter_window_reset(self):
        limiter = self._make_limiter()
        # Use up the limit
        for _ in range(3):
            limiter.check_and_increment("1.2.3.4", "startup")

        with pytest.raises(RateLimitExceeded):
            limiter.check_and_increment("1.2.3.4", "startup")

        # Fast-forward time past the 24h window
        future_time = time.time() + (25 * 3600)
        with patch("app.agent.rate_limiter.time.time", return_value=future_time):
            count, limit = limiter.check_and_increment("1.2.3.4", "startup")
            assert count == 1  # Counter should have reset

    def test_rate_limiter_get_status_no_side_effects(self):
        limiter = self._make_limiter()
        # Call get_status multiple times
        for _ in range(10):
            status = limiter.get_status("5.5.5.5", "startup")
        assert status["count"] == 0
        assert status["remaining"] == 3
        # Should still be able to use full quota
        count, _ = limiter.check_and_increment("5.5.5.5", "startup")
        assert count == 1

    def test_rate_limiter_global_stats(self):
        limiter = self._make_limiter()
        limiter.check_and_increment("10.0.0.1", "startup")
        limiter.check_and_increment("10.0.0.2", "startup")
        stats = limiter.get_global_stats()
        assert stats["tracked_ips"] == 2
        assert "limits" in stats
