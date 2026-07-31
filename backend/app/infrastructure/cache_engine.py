import json
import hashlib
import time
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger, logger_timer

# Optional Redis Client initialization
redis_client = None
HAS_REDIS = False

try:
    import redis
    if settings.REDIS_URL:
        redis_client = redis.Redis.from_url(
            settings.REDIS_URL, decode_responses=True, socket_timeout=2.0
        )
        # Test ping connection
        redis_client.ping()
        HAS_REDIS = True
        logger.info(f"[Cache Engine] Successfully connected to Redis Store at '{settings.REDIS_URL}'")
except Exception as _e:
    redis_client = None
    HAS_REDIS = False
    logger.info("[Cache Engine] Redis unavailable or offline. Operating in ultra-fast Python RAM In-Memory mode.")


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute Cosine Similarity between two high-dimensional float vector embeddings."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


class RetrievalCache:
    """Tier 1 Cache: Exact query & session hash lookup to skip redundant Qdrant/BM25 retrieval."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}

    def _hash_key(self, query: str, session_id: str, search_scope: str) -> str:
        raw_str = f"{query.strip().lower()}:{session_id}:{search_scope}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(
        self, query: str, session_id: str, search_scope: str
    ) -> Optional[Dict[str, Any]]:
        key = self._hash_key(query, session_id, search_scope)

        # 1. Try Redis Store if active
        if HAS_REDIS and redis_client:
            try:
                cached_val = redis_client.get(f"retrieval:{key}")
                if cached_val:
                    logger.info(f"[RetrievalCache] REDIS HIT for query: '{query}'")
                    return json.loads(cached_val)
            except Exception as e:
                logger.warning(f"[RetrievalCache] Redis read error: {e}")

        # 2. Fallback to RAM In-Memory store
        entry = self._store.get(key)
        if not entry:
            return None

        # Check TTL expiration
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            logger.info(f"[RetrievalCache] Expired cache key for query: '{query}'")
            del self._store[key]
            return None

        logger.info(f"[RetrievalCache] RAM HIT for query: '{query}' (Skipping Qdrant/BM25)")
        return entry["data"]

    def set(
        self, query: str, session_id: str, search_scope: str, data: Dict[str, Any]
    ) -> None:
        key = self._hash_key(query, session_id, search_scope)
        
        # 1. Store in Redis if active
        if HAS_REDIS and redis_client:
            try:
                redis_client.setex(
                    f"retrieval:{key}", self.ttl_seconds, json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"[RetrievalCache] Redis write error: {e}")

        # 2. Store in RAM In-Memory store
        self._store[key] = {"timestamp": time.time(), "data": data}
        logger.info(f"[RetrievalCache] SET cache key for query: '{query}'")


class SemanticCosineCache:
    """Tier 2 Cache: Vector Cosine Similarity (>0.95) to return instant 10ms responses for conceptual queries."""

    def __init__(self, similarity_threshold: float = 0.95, ttl_seconds: int = 7200):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._entries: List[Dict[str, Any]] = []

    def get(
        self, query_vector: List[float], trace_id: str = "N/A"
    ) -> Optional[Dict[str, Any]]:
        if not query_vector:
            return None

        with logger_timer("SemanticCosineCache: Similarity Scan", trace_id=trace_id) as log:
            now = time.time()
            best_score = 0.0
            best_entry = None

            for entry in self._entries:
                if now - entry["timestamp"] > self.ttl_seconds:
                    continue

                score = cosine_similarity(query_vector, entry["vector"])
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry and best_score >= self.similarity_threshold:
                log.info(
                    f"[SemanticCache] HIT! Similarity Score={best_score:.4f} >= {self.similarity_threshold} (Instant 10ms Return)"
                )
                cached_copy = dict(best_entry["payload"])
                cached_copy["semantic_cache_hit"] = True
                cached_copy["semantic_similarity_score"] = round(best_score, 4)
                return cached_copy

            log.info(f"[SemanticCache] MISS. Best similarity score was {best_score:.4f}")
            return None

    def set(
        self, query_vector: List[float], payload: Dict[str, Any]
    ) -> None:
        if not query_vector:
            return
        self._entries.append(
            {
                "timestamp": time.time(),
                "vector": query_vector,
                "payload": payload,
            }
        )
        logger.info(f"[SemanticCache] STORED conceptual query vector into Semantic Cache")


# Singleton Cache Instances
retrieval_cache = RetrievalCache(ttl_seconds=3600)
semantic_cache = SemanticCosineCache(similarity_threshold=0.95, ttl_seconds=7200)
