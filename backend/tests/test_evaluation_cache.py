"""
Unit tests for EvaluationCache (app.agent.evaluation_cache).
Each test creates a fresh EvaluationCache instance for isolation.
"""

import time
import pytest
from unittest.mock import patch

from app.agent.evaluation_cache import EvaluationCache


class TestEvaluationCache:

    def _make_cache(self):
        """Create a fresh cache instance that skips disk loading."""
        cache = EvaluationCache.__new__(EvaluationCache)
        cache._cache = {}
        cache._embeddings = {}
        import threading
        cache._lock = threading.Lock()
        cache._hits = 0
        cache._misses = 0
        return cache

    def _sample_eval_data(self):
        return {
            "overall_score": 7.5,
            "dimensions": [],
            "key_strengths": ["good team"],
            "main_concerns": [],
            "next_steps": [],
        }

    def test_cache_miss_on_empty(self):
        cache = self._make_cache()
        result = cache.get("some pitch text here", "startup")
        assert result is None

    def test_cache_set_and_get(self):
        cache = self._make_cache()
        pitch = "This is a valid pitch about machine learning and healthcare."
        cache.set(
            pitch_text=pitch,
            role="startup",
            startup_name="TestCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=1.5,
            llm_provider="mock",
        )
        result = cache.get(pitch, "startup")
        assert result is not None
        assert result["startup_name"] == "TestCo"
        assert result["processing_time"] == 1.5

    def test_cache_different_roles_separate(self):
        cache = self._make_cache()
        pitch = "A pitch about fintech innovation in emerging markets."
        cache.set(
            pitch_text=pitch,
            role="startup",
            startup_name="FinTechCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=1.0,
            llm_provider="mock",
        )
        # Same pitch, different role should be a miss
        result = cache.get(pitch, "investor")
        assert result is None

        # Same pitch, same role should be a hit
        result = cache.get(pitch, "startup")
        assert result is not None

    def test_cache_ttl_expiry(self):
        cache = self._make_cache()
        pitch = "An AI startup focused on natural language processing."
        cache.set(
            pitch_text=pitch,
            role="startup",
            startup_name="NLPCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=2.0,
            llm_provider="mock",
        )
        # Verify it's cached
        assert cache.get(pitch, "startup") is not None

        # Fast-forward past TTL (24h + 1h)
        future = time.time() + (25 * 3600)
        with patch("app.agent.evaluation_cache.time.time", return_value=future):
            result = cache.get(pitch, "startup")
            assert result is None

    def test_cache_delete_entry(self):
        cache = self._make_cache()
        pitch = "A blockchain startup for supply chain management."
        cache.set(
            pitch_text=pitch,
            role="startup",
            startup_name="ChainCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=1.0,
            llm_provider="mock",
        )
        assert cache.get(pitch, "startup") is not None

        deleted = cache.delete_entry(pitch, "startup")
        assert deleted is True

        assert cache.get(pitch, "startup") is None

    def test_cache_clear(self):
        cache = self._make_cache()
        for i in range(5):
            cache.set(
                pitch_text=f"Pitch number {i} about various technology innovations and market opportunities.",
                role="startup",
                startup_name=f"Startup{i}",
                evaluation_data=self._sample_eval_data(),
                steps_data=[],
                processing_time=1.0,
                llm_provider="mock",
            )
        assert len(cache._cache) == 5

        cache.clear()
        assert len(cache._cache) == 0

    def test_cache_stats(self):
        cache = self._make_cache()
        pitch = "A robotics startup automating warehouse operations."
        cache.set(
            pitch_text=pitch,
            role="startup",
            startup_name="RoboCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=1.0,
            llm_provider="mock",
        )
        # One miss
        cache.get("nonexistent pitch that is not in the cache", "startup")
        # One hit
        cache.get(pitch, "startup")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_cache_normalization(self):
        cache = self._make_cache()
        pitch1 = "Hello World and some additional text to make this long enough for testing purposes."
        pitch2 = "  hello   world  and  some  additional  text  to  make  this  long  enough  for  testing  purposes. "
        cache.set(
            pitch_text=pitch1,
            role="startup",
            startup_name="NormCo",
            evaluation_data=self._sample_eval_data(),
            steps_data=[],
            processing_time=1.0,
            llm_provider="mock",
        )
        # pitch2 should produce the same cache key after normalization
        result = cache.get(pitch2, "startup")
        assert result is not None
        assert result["startup_name"] == "NormCo"
