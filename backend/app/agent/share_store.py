"""
PitchPal v2 - Shareable Evaluation Link Store

Allows users to generate a public /eval/{uuid} link
for any evaluation they run. Links expire after 7 days.

Storage: in-memory + JSON file (same pattern as evaluation cache)
Max entries: 1000 (FIFO eviction)
TTL: 7 days
"""

import json
import logging
import secrets
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
SHARE_TTL = 7 * 24 * 3600       # 7 days in seconds
MAX_SHARES = 1000                 # Max entries before FIFO eviction

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
SHARE_FILE = CACHE_DIR / "share_store.json"


class ShareStore:
    """
    Stores shared evaluation results, keyed by a random UUID.

    Structure: { share_id: { evaluation, startup_name, role, created_at, expires_at, views } }
    """

    def __init__(self):
        self._store: dict = {}
        self._load_from_disk()

    # ── Persistence ───────────────────────────────────────────

    def _load_from_disk(self):
        try:
            if SHARE_FILE.exists():
                raw = json.loads(SHARE_FILE.read_text(encoding="utf-8"))
                now = time.time()
                loaded, expired = 0, 0
                for sid, entry in raw.items():
                    if entry.get("expires_at", 0) < now:
                        expired += 1
                        continue
                    self._store[sid] = entry
                    loaded += 1
                logger.info(f"Share store loaded: {loaded} valid, {expired} expired")
            else:
                logger.info("Share store: no file found — starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load share store: {e}")
            self._store = {}

    def _save_to_disk(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            SHARE_FILE.write_text(
                json.dumps(self._store, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to save share store: {e}")

    # ── FIFO eviction ─────────────────────────────────────────

    def _evict_if_needed(self):
        """Remove expired entries and FIFO-evict if over MAX_SHARES."""
        now = time.time()
        # Remove expired
        expired = [k for k, v in self._store.items() if v.get("expires_at", 0) < now]
        for k in expired:
            del self._store[k]
        # FIFO eviction if still over limit
        if len(self._store) >= MAX_SHARES:
            oldest = sorted(self._store.items(), key=lambda x: x[1].get("created_at", 0))
            for sid, _ in oldest[:len(self._store) - MAX_SHARES + 1]:
                del self._store[sid]

    # ── Public API ────────────────────────────────────────────

    def create(
        self,
        evaluation: dict,
        startup_name: str,
        role: str,
        processing_time: float = 0.0,
        llm_provider: str = "unknown",
    ) -> str:
        """
        Store an evaluation and return a share_id (UUID).
        The share link is /eval/{share_id}.
        """
        self._evict_if_needed()

        share_id = secrets.token_urlsafe(24)
        now = time.time()

        self._store[share_id] = {
            "startup_name": startup_name,
            "role": role,
            "evaluation": evaluation,
            "processing_time": processing_time,
            "llm_provider": llm_provider,
            "created_at": now,
            "expires_at": now + SHARE_TTL,
            "views": 0,
        }

        self._save_to_disk()
        logger.info(f"Share created: id={share_id[:8]}... startup={startup_name} role={role}")
        return share_id

    def get(self, share_id: str) -> Optional[dict]:
        """
        Retrieve a shared evaluation by share_id.
        Increments view counter. Returns None if not found or expired.
        """
        entry = self._store.get(share_id)
        if not entry:
            return None

        # Check expiry
        if entry.get("expires_at", 0) < time.time():
            del self._store[share_id]
            self._save_to_disk()
            logger.info(f"Share expired: id={share_id[:8]}...")
            return None

        # Increment views
        entry["views"] = entry.get("views", 0) + 1
        self._save_to_disk()

        logger.info(f"Share retrieved: id={share_id[:8]}... views={entry['views']}")
        return entry

    def get_stats(self) -> dict:
        now = time.time()
        active = sum(1 for v in self._store.values() if v.get("expires_at", 0) >= now)
        total_views = sum(v.get("views", 0) for v in self._store.values())
        return {
            "active_shares": active,
            "total_entries": len(self._store),
            "total_views": total_views,
            "ttl_days": SHARE_TTL // 86400,
        }


# ── Singleton ─────────────────────────────────────────────────
share_store = ShareStore()
