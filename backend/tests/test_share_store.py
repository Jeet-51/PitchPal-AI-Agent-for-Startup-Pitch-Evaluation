"""
Unit tests for ShareStore (app.agent.share_store).
Each test creates a fresh ShareStore instance for isolation.
"""

import time
import pytest
from unittest.mock import patch

from app.agent.share_store import ShareStore, SHARE_TTL, MAX_SHARES


class TestShareStore:

    def _make_store(self):
        """Create a fresh ShareStore that skips disk loading."""
        store = ShareStore.__new__(ShareStore)
        store._store = {}
        return store

    def _sample_evaluation(self):
        return {"overall_score": 8.0, "dimensions": []}

    def test_create_and_get_share(self):
        store = self._make_store()
        share_id = store.create(
            evaluation=self._sample_evaluation(),
            startup_name="TestStartup",
            role="startup",
            processing_time=2.0,
            llm_provider="mock",
        )
        assert share_id is not None
        assert len(share_id) > 0

        entry = store.get(share_id)
        assert entry is not None
        assert entry["startup_name"] == "TestStartup"
        assert entry["evaluation"]["overall_score"] == 8.0

    def test_share_view_counter(self):
        store = self._make_store()
        share_id = store.create(
            evaluation=self._sample_evaluation(),
            startup_name="ViewsTest",
            role="startup",
        )
        # Initial views should be 0, first get increments to 1
        entry = store.get(share_id)
        assert entry["views"] == 1

        entry = store.get(share_id)
        assert entry["views"] == 2

        entry = store.get(share_id)
        assert entry["views"] == 3

    def test_share_expired(self):
        store = self._make_store()
        share_id = store.create(
            evaluation=self._sample_evaluation(),
            startup_name="ExpiredStartup",
            role="startup",
        )
        # Fast-forward past the 7-day TTL
        future = time.time() + SHARE_TTL + 3600
        with patch("app.agent.share_store.time.time", return_value=future):
            entry = store.get(share_id)
            assert entry is None

    def test_share_not_found(self):
        store = self._make_store()
        entry = store.get("nonexistent_share_id_12345")
        assert entry is None

    def test_share_fifo_eviction(self):
        store = self._make_store()
        # Create MAX_SHARES entries
        first_id = None
        for i in range(MAX_SHARES):
            sid = store.create(
                evaluation=self._sample_evaluation(),
                startup_name=f"Startup{i}",
                role="startup",
            )
            if i == 0:
                first_id = sid

        # The store should have MAX_SHARES entries (some may have been evicted)
        assert len(store._store) <= MAX_SHARES

        # Adding one more should trigger FIFO eviction of the oldest
        store.create(
            evaluation=self._sample_evaluation(),
            startup_name="NewStartup",
            role="startup",
        )
        assert len(store._store) <= MAX_SHARES

        # The oldest entry should have been evicted
        entry = store.get(first_id)
        assert entry is None

    def test_share_stats(self):
        store = self._make_store()
        store.create(
            evaluation=self._sample_evaluation(),
            startup_name="StatStartup1",
            role="startup",
        )
        sid2 = store.create(
            evaluation=self._sample_evaluation(),
            startup_name="StatStartup2",
            role="startup",
        )
        # View one share twice
        store.get(sid2)
        store.get(sid2)

        stats = store.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active_shares"] == 2
        assert stats["total_views"] == 2
        assert stats["ttl_days"] == 7
