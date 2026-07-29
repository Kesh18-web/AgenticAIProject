from typing import Any, Dict, List
from backend.app.core.logging import logger, logger_timer
from backend.app.db.bm25 import BM25Indexer
from backend.app.db.qdrant import QdrantVectorStoreManager


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion (RRF) algorithm to merge dense vector search and BM25 search rankings."""
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Helper to calculate key
    def get_chunk_key(c: Dict[str, Any]) -> str:
        return str(c.get("doc_id", "")) + "_" + str(c.get("text", "")[:50])

    # Process Dense Results with dense_weight multiplier
    for rank, chunk in enumerate(dense_results):
        key = get_chunk_key(chunk)
        chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (dense_weight * (1.0 / (k + rank + 1)))

    # Process BM25 Results with bm25_weight multiplier
    for rank, chunk in enumerate(bm25_results):
        key = get_chunk_key(chunk)
        if key not in chunk_map:
            chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (bm25_weight * (1.0 / (k + rank + 1)))

    # Sort chunks by final RRF score descending
    fused_chunks = []
    for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = dict(chunk_map[key])
        chunk["score"] = float(score)
        chunk["retrieval_method"] = "weighted_hybrid_rrf"
        fused_chunks.append(chunk)

    return fused_chunks


class HybridRetriever:
    """Hybrid Retriever combining Qdrant Dense Vector Search & BM25 Keyword Search via Weighted RRF."""

    def __init__(
        self,
        qdrant_mgr: QdrantVectorStoreManager,
        bm25_mgr: BM25Indexer,
    ):
        self.qdrant_mgr = qdrant_mgr
        self.bm25_mgr = bm25_mgr

    def retrieve_hybrid(
        self,
        collection_name: str,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        trace_id: str = "N/A",
    ) -> List[Dict[str, Any]]:
        """Retrieve and rank candidates using hybrid dense vector search + BM25 keyword search + Weighted RRF."""
        with logger_timer("HybridRetriever: Weighted Search & Fusion", trace_id=trace_id) as log:
            log.info(
                f"Executing weighted hybrid retrieval for query: '{query}' | dense_weight={dense_weight:.2f} | bm25_weight={bm25_weight:.2f}"
            )

            # 1. Execute Dense Vector Search
            dense_hits = []
            if query_embedding:
                dense_hits = self.qdrant_mgr.search_dense(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    limit=top_k * 2,
                )

            # 2. Execute BM25 Keyword Search
            bm25_hits = self.bm25_mgr.search_bm25(query=query, top_k=top_k * 2)

            # 3. Fuse via Weighted Reciprocal Rank Fusion (Weighted RRF)
            fused_candidates = reciprocal_rank_fusion(
                dense_results=dense_hits,
                bm25_results=bm25_hits,
                k=60,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )

            final_candidates = fused_candidates[:top_k]
            log.info(
                f"Retrieved {len(dense_hits)} dense + {len(bm25_hits)} BM25 -> Fused into {len(final_candidates)} top candidates (Weighted RRF)"
            )
            return final_candidates
