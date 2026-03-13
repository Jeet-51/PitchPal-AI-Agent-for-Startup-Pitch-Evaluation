"""
PitchPal v2 - Evaluation-Level Cache

Caches FULL evaluation results so that the exact same pitch + role
always returns the exact same scores. Eliminates non-determinism.

How it works:
1. Normalize the pitch text (lowercase, strip extra whitespace)
2. Hash (normalized_pitch_text + role) using SHA-256
3. If hash exists in cache and not expired → return cached evaluation instantly
4. If miss → run agent pipeline, cache the result for future
5. Cache persists to disk (survives server restarts)

TTL: 24 hours (research data stays reasonably fresh)
"""

import hashlib
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────
EVAL_CACHE_TTL = 86400  # 24 hours

# Cache file location
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
EVAL_CACHE_FILE = CACHE_DIR / "evaluation_cache.json"


def _normalize_pitch(pitch_text: str) -> str:
    """
    Normalize pitch text for consistent hashing.
    Lowercases, strips, collapses whitespace.
    """
    text = pitch_text.lower().strip()
    # Collapse all whitespace (spaces, tabs, newlines) into single spaces
    text = " ".join(text.split())
    return text


def _make_key(pitch_text: str, role: str) -> str:
    """
    Create a deterministic cache key from pitch text + role.
    Uses SHA-256 hash of normalized text.
    """
    normalized = _normalize_pitch(pitch_text)
    raw = f"{role}::{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EvaluationCache:
    """
    Cache for full evaluation results.
    Same pitch + same role → same scores, every time.
    """

    def __init__(self):
        self._cache: dict = {}  # {hash_key: {evaluation, steps, timestamp, startup_name, ...}}
        # Similarity index: {hash_key: {"embedding": list, "startup_name": str, "role": str}}
        self._embeddings: dict = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._load_from_disk()

    def _load_from_disk(self):
        """Load cached evaluations from disk on startup."""
        try:
            if EVAL_CACHE_FILE.exists():
                raw = json.loads(EVAL_CACHE_FILE.read_text(encoding="utf-8"))
                now = time.time()
                loaded = 0
                expired = 0

                for key, entry in raw.items():
                    if now - entry.get("timestamp", 0) > EVAL_CACHE_TTL:
                        expired += 1
                        continue
                    self._cache[key] = entry
                    loaded += 1

                logger.info(f"Evaluation cache loaded: {loaded} entries ({expired} expired)")
            else:
                logger.info("No evaluation cache file -- starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load evaluation cache: {e}")
            self._cache = {}

    def _save_to_disk(self):
        """Persist cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            EVAL_CACHE_FILE.write_text(
                json.dumps(self._cache, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to save evaluation cache: {e}")

    def get(self, pitch_text: str, role: str) -> Optional[dict]:
        """
        Look up a cached evaluation.
        Returns dict with 'evaluation', 'steps', 'processing_time', etc. or None.
        """
        key = _make_key(pitch_text, role)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                logger.info(f"Evaluation cache MISS (key={key[:12]}...)")
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.get("timestamp", 0) > EVAL_CACHE_TTL:
                del self._cache[key]
                self._misses += 1
                logger.info(f"Evaluation cache EXPIRED (key={key[:12]}...)")
                return None

            self._hits += 1
            logger.info(
                f"Evaluation cache HIT: \"{entry.get('startup_name', '?')}\" "
                f"(role={role}, key={key[:12]}...)"
            )
            return entry

    def set(
        self,
        pitch_text: str,
        role: str,
        startup_name: str,
        evaluation_data: dict,
        steps_data: list,
        processing_time: float,
        llm_provider: str,
        cache_hits: int = 0,
        embedding: Optional[list] = None,
    ):
        """Store a completed evaluation in the cache."""
        key = _make_key(pitch_text, role)

        with self._lock:
            self._cache[key] = {
                "startup_name": startup_name,
                "role": role,
                "evaluation": evaluation_data,
                "steps": steps_data,
                "processing_time": processing_time,
                "llm_provider": llm_provider,
                "cache_hits": cache_hits,
                "timestamp": time.time(),
            }

            # Store embedding for similarity search (if provided)
            if embedding:
                self._embeddings[key] = {
                    "embedding": embedding,
                    "startup_name": startup_name,
                    "role": role,
                }

            logger.info(
                f"Evaluation cache SET: \"{startup_name}\" "
                f"(role={role}, key={key[:12]}..., total={len(self._cache)})"
            )
            self._save_to_disk()

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            # Clean expired
            expired_keys = [k for k, v in self._cache.items()
                            if now - v.get("timestamp", 0) > EVAL_CACHE_TTL]
            for k in expired_keys:
                del self._cache[k]

            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_hours": EVAL_CACHE_TTL // 3600,
            }

    def delete_entry(self, pitch_text: str, role: str) -> bool:
        """
        Delete a specific entry from the cache by pitch_text + role.
        Returns True if an entry was found and deleted, False otherwise.
        """
        key = _make_key(pitch_text, role)
        with self._lock:
            if key in self._cache:
                startup_name = self._cache[key].get("startup_name", "?")
                del self._cache[key]
                self._save_to_disk()
                logger.info(f"Evaluation cache ENTRY DELETED: \"{startup_name}\" (role={role}, key={key[:12]}...)")
                return True
            logger.info(f"Evaluation cache DELETE: entry not found (key={key[:12]}...)")
            return False

    def clear(self):
        """Clear all cached evaluations."""
        with self._lock:
            self._cache.clear()
            try:
                if EVAL_CACHE_FILE.exists():
                    EVAL_CACHE_FILE.unlink()
            except Exception:
                pass
            logger.info("Evaluation cache CLEARED")


    def find_similar(self, embedding: list, role: str, threshold: float = 0.87) -> Optional[dict]:
        """
        Find the most similar previously-evaluated pitch using cosine similarity.
        Only compares against entries in the same role.
        Returns {startup_name, similarity, role} if found, else None.
        """
        if not embedding or not self._embeddings:
            return None

        best_sim = 0.0
        best_match = None

        with self._lock:
            for key, meta in self._embeddings.items():
                if meta.get("role") != role:
                    continue
                sim = _cosine_similarity(embedding, meta["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_match = meta | {"similarity": round(sim, 3)}

        if best_match and best_sim >= threshold:
            logger.info(
                f"Similar pitch found: '{best_match['startup_name']}' "
                f"similarity={best_sim:.3f}"
            )
            return best_match
        return None


# ── Helpers ───────────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity — mirrors what semantic_cache uses."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Singleton instance
evaluation_cache = EvaluationCache()
