import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch ReActAgent before importing app so the health endpoints don't need real LLM keys
_mock_agent_cls = MagicMock()
_mock_agent_instance = MagicMock()
_mock_agent_instance.llm.get_provider_name.return_value = "mock"
_mock_agent_cls.return_value = _mock_agent_instance

with patch.dict("sys.modules", {}):
    pass

import sys
from unittest.mock import patch as _patch

# We need to patch ReActAgent at import time so the app module can load
# without requiring real API keys.
_patcher = _patch("app.agent.react_agent.ReActAgent", _mock_agent_cls)
_patcher.start()

from app.main import app
from app.agent.evaluation_cache import evaluation_cache
from app.agent.semantic_cache import semantic_cache
from app.agent.share_store import share_store
from app.agent.rate_limiter import rate_limiter


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset all caches and rate limiter between tests."""
    evaluation_cache._cache.clear()
    evaluation_cache._hits = 0
    evaluation_cache._misses = 0
    semantic_cache.clear()
    share_store._store.clear()
    rate_limiter._data.clear()
    yield
    evaluation_cache._cache.clear()
    evaluation_cache._hits = 0
    evaluation_cache._misses = 0
    semantic_cache.clear()
    share_store._store.clear()
    rate_limiter._data.clear()


VALID_PITCH = (
    "HealthAI uses machine learning to analyze medical images and detect "
    "early signs of diseases. Our proprietary AI model achieved 94% accuracy "
    "in detecting pneumonia from chest X-rays, outperforming average radiologist "
    "readings. We target rural hospitals and clinics in underserved areas."
)
