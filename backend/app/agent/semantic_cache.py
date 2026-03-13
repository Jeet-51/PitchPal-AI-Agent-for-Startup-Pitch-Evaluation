"""
PitchPal v2 - Semantic Tool Result Cache (Gemini Embeddings)

Replaces the old Jaccard + domain-keyword approach with real vector
similarity using Google's text-embedding-004 model.

How it works:
1. On get()  → embed the incoming query → cosine-compare against all
               cached entries for the same tool type → return if ≥ threshold
2. On set()  → embed the query → store (embedding, result) on disk
3. Embeddings are generated via the same GEMINI_API_KEY already in use —
   zero extra cost, zero new dependencies, ~50–100 ms per call.

Why cosine similarity > Jaccard for this task:
  "pharmaceutical Africa last-mile delivery"
  "pharma distribution sub-Saharan supply chain"
  → Jaccard ≈ 0%   (no shared tokens)
  → Cosine  ≈ 0.87 (same semantic concept)

Threshold: 0.82  (empirically: same topic = 0.85+, different topic = 0.50-)
TTL: 24 hours (same as evaluation cache)
"""

import json
import math
import threading
import time
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.72   # calibrated: similar topics ~0.74+, unrelated ~0.49-
CACHE_TTL            = 86400  # 24 hours in seconds
EMBEDDING_MODEL      = "models/gemini-embedding-001"   # 3072-dim, confirmed available

# Cache file (same directory as before)
CACHE_DIR  = Path(__file__).parent.parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "tool_results.json"


# ── Embedding ────────────────────────────────────────────────

def _get_embedding(text: str) -> Optional[list]:
    """
    Call Gemini text-embedding-004 and return a 768-dim float list.
    Returns None on any failure so the caller can gracefully skip caching.
    """
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        return response["embedding"]          # list[float], len=768
    except Exception as exc:
        logger.warning(f"Gemini embedding failed: {exc}")
        return None


# ── Cosine Similarity ────────────────────────────────────────

def _cosine(vec_a: list, vec_b: list) -> float:
    """
    Pure-Python cosine similarity — no numpy required.
    Returns 0.0 → 1.0  (1.0 = identical direction).
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Cache Class ──────────────────────────────────────────────

class SemanticToolCache:
    """
    Semantic cache for Tavily tool results.
    Stores (tool_name, query embedding, raw result) on disk.
    On lookup: embed new query → cosine-compare → return if similar enough.
    """

    def __init__(self):
        self._cache: list = []   # list of dicts (in-memory)
        self._lock = threading.Lock()
        self._hits   = 0
        self._misses = 0
        self._tokens_saved = 0
        self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────

    def _load_from_disk(self):
        """
        Load cache from disk, silently skipping:
          - expired entries  (older than CACHE_TTL)
          - legacy entries   (old Jaccard format that has 'keywords' not 'embedding')
        """
        try:
            if not CACHE_FILE.exists():
                logger.info("Semantic cache: no file found — starting fresh")
                return

            raw  = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            now  = time.time()
            loaded = expired = skipped = 0

            for entry in raw:
                # Skip expired
                if now - entry.get("timestamp", 0) > CACHE_TTL:
                    expired += 1
                    continue
                # Skip legacy Jaccard entries (they have 'keywords', not 'embedding')
                if "embedding" not in entry:
                    skipped += 1
                    continue
                self._cache.append(entry)
                loaded += 1

            logger.info(
                f"Semantic cache loaded: {loaded} entries "
                f"({expired} expired, {skipped} legacy-format skipped)"
            )
        except Exception as exc:
            logger.warning(f"Semantic cache load error: {exc}")
            self._cache = []

    def _save_to_disk(self):
        """Persist current in-memory cache to JSON."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(
                json.dumps(self._cache, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"Semantic cache save error: {exc}")

    # ── Public API ───────────────────────────────────────────

    def get(self, tool_name: str, query: str) -> Optional[str]:
        """
        Look up a semantically similar cached result.
        Returns the cached string result, or None on miss / embedding failure.
        """
        query_vec = _get_embedding(query)
        if query_vec is None:
            # Embedding call failed — treat as miss, still call Tavily
            self._misses += 1
            return None

        with self._lock:
            # Evict expired entries
            now = time.time()
            self._cache = [e for e in self._cache if now - e["timestamp"] <= CACHE_TTL]

            best_entry = None
            best_score = 0.0

            for entry in self._cache:
                if entry["tool_name"] != tool_name:
                    continue
                score = _cosine(query_vec, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry and best_score >= SIMILARITY_THRESHOLD:
                self._hits        += 1
                self._tokens_saved += 500
                logger.info(
                    f"Semantic cache HIT  [{tool_name}] "
                    f"cosine={best_score:.3f}  "
                    f"query=\"{query[:60]}\""
                )
                return best_entry["result"]

            self._misses += 1
            logger.info(
                f"Semantic cache MISS [{tool_name}] "
                f"best_cosine={best_score:.3f}  "
                f"query=\"{query[:60]}\""
            )
            return None

    def set(self, tool_name: str, query: str, result: str):
        """
        Store a Tavily result with its query embedding.
        Silently skips storage if the embedding call fails.
        """
        embedding = _get_embedding(query)
        if embedding is None:
            logger.warning(
                f"Skipping cache set for [{tool_name}] — embedding failed"
            )
            return

        with self._lock:
            self._cache.append({
                "tool_name": tool_name,
                "query":     query,
                "embedding": embedding,          # 768 floats
                "result":    result,
                "timestamp": time.time(),
            })
            logger.info(
                f"Semantic cache SET  [{tool_name}] "
                f"total_entries={len(self._cache)}"
            )
            self._save_to_disk()

    def get_stats(self) -> dict:
        """Return cache performance statistics."""
        with self._lock:
            now = time.time()
            self._cache = [e for e in self._cache if now - e["timestamp"] <= CACHE_TTL]

            total    = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0

            tool_counts: dict = {}
            for entry in self._cache:
                t = entry["tool_name"]
                tool_counts[t] = tool_counts.get(t, 0) + 1

            return {
                "entries":          len(self._cache),
                "hits":             self._hits,
                "misses":           self._misses,
                "hit_rate":         f"{hit_rate:.1f}%",
                "est_tokens_saved": self._tokens_saved,
                "threshold":        f"{SIMILARITY_THRESHOLD:.0%}",
                "embedding_model":  EMBEDDING_MODEL,
                "entries_by_tool":  tool_counts,
            }

    def clear(self):
        """Clear all in-memory and on-disk cache entries."""
        with self._lock:
            self._cache.clear()
            try:
                if CACHE_FILE.exists():
                    CACHE_FILE.unlink()
            except Exception:
                pass
            logger.info("Semantic cache CLEARED")


# ── Singleton ────────────────────────────────────────────────
semantic_cache = SemanticToolCache()
