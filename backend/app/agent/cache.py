"""
PitchPal v2 - Evaluation Cache (File-Backed)
Persistent cache that survives server restarts.

How it works:
- Generates a hash from (startup_name + pitch_text)
- If the same pitch was evaluated within the TTL window, returns cached result
- Cache is stored both in-memory (fast reads) AND on disk (persistence)
- On startup, loads existing cache from disk
- Cache entries auto-expire after CACHE_TTL seconds (default: 6 hours)
"""

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from app.models.schemas import PitchEvaluation, AgentStep

logger = logging.getLogger(__name__)

# Cache TTL in seconds (6 hours — generous since pitch data doesn't change fast)
CACHE_TTL = 21600

# Cache file location
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "evaluations.json"


class EvaluationCache:
    """
    File-backed cache for pitch evaluations.

    In-memory dict for fast reads + JSON file on disk for persistence.
    Loads from disk on startup, writes to disk on every new cache entry.
    """

    def __init__(self):
        self._cache: dict = {}
        self._hits = 0
        self._misses = 0
        self._load_from_disk()

    def _generate_key(self, startup_name: str, pitch_text: str) -> str:
        """Generate a unique cache key from pitch content."""
        normalized = f"{startup_name.lower().strip()}::{pitch_text.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _load_from_disk(self):
        """Load cached evaluations from disk on startup."""
        try:
            if CACHE_FILE.exists():
                raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                now = time.time()
                loaded = 0
                expired = 0

                for key, entry in raw.items():
                    # Skip expired entries
                    if now - entry["timestamp"] > CACHE_TTL:
                        expired += 1
                        continue

                    # Reconstruct Pydantic models from dicts
                    self._cache[key] = {
                        "evaluation": PitchEvaluation(**entry["evaluation"]),
                        "steps": [AgentStep(**s) for s in entry["steps"]],
                        "timestamp": entry["timestamp"],
                        "startup_name": entry.get("startup_name", "Unknown"),
                    }
                    loaded += 1

                logger.info(f"Cache loaded from disk: {loaded} entries ({expired} expired, skipped)")
            else:
                logger.info("No cache file found — starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e} — starting fresh")
            self._cache = {}

    def _save_to_disk(self):
        """Persist current cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Convert Pydantic models to dicts for JSON serialization
            serializable = {}
            for key, entry in self._cache.items():
                serializable[key] = {
                    "evaluation": entry["evaluation"].model_dump(),
                    "steps": [s.model_dump() for s in entry["steps"]],
                    "timestamp": entry["timestamp"],
                    "startup_name": entry.get("startup_name", "Unknown"),
                }

            CACHE_FILE.write_text(
                json.dumps(serializable, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(f"Cache saved to disk: {len(serializable)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def get(self, startup_name: str, pitch_text: str) -> Optional[Tuple[PitchEvaluation, List[AgentStep]]]:
        """
        Look up a cached evaluation.

        Returns:
            Tuple of (PitchEvaluation, List[AgentStep]) if cache hit, None if miss.
        """
        key = self._generate_key(startup_name, pitch_text)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            logger.info(f"Cache MISS for '{startup_name}' (key={key})")
            return None

        # Check TTL
        age = time.time() - entry["timestamp"]
        if age > CACHE_TTL:
            del self._cache[key]
            self._save_to_disk()
            self._misses += 1
            logger.info(f"Cache EXPIRED for '{startup_name}' (age={age:.0f}s, key={key})")
            return None

        self._hits += 1
        logger.info(f"Cache HIT for '{startup_name}' (age={age:.0f}s, key={key})")
        return entry["evaluation"], entry["steps"]

    def set(self, startup_name: str, pitch_text: str, evaluation: PitchEvaluation, steps: List[AgentStep]):
        """Store an evaluation result in the cache and persist to disk."""
        key = self._generate_key(startup_name, pitch_text)
        self._cache[key] = {
            "evaluation": evaluation,
            "steps": steps,
            "timestamp": time.time(),
            "startup_name": startup_name,
        }
        logger.info(f"Cache SET for '{startup_name}' (key={key}, total={len(self._cache)})")
        self._save_to_disk()

    def clear(self):
        """Clear all cached entries (memory + disk)."""
        self._cache.clear()
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
        except Exception:
            pass
        logger.info("Cache CLEARED (memory + disk)")

    def get_stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v["timestamp"] > CACHE_TTL]
        for k in expired_keys:
            del self._cache[k]

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": CACHE_TTL,
            "persistent": True,
            "cache_file": str(CACHE_FILE),
        }


# Singleton instance — shared across the app
evaluation_cache = EvaluationCache()
