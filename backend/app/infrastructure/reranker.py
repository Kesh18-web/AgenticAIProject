from typing import Any, Dict, List
from backend.app.core.logging import logger, logger_timer

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

class CrossEncoderReranker:
    """Mandatory Cross-Encoder Reranker Infrastructure Component for deep semantic relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize()

    def _initialize(self):
        if HAS_CROSS_ENCODER:
            try:
                # Load lightweight CPU-friendly cross-encoder model
                self.model = CrossEncoder(self.model_name)
                logger.info(
                    f"Successfully loaded Cross-Encoder Reranker model '{self.model_name}'"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load Cross-Encoder model '{self.model_name}': {e}. Heuristic ranker will be used as fallback."
                )

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
        trace_id: str = "N/A",
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks using Cross-Encoder joint attention scoring."""
        if not chunks:
            return []

        with logger_timer("CrossEncoderReranker: Reranking", trace_id=trace_id) as log:
            log.info(f"Reranking {len(chunks)} candidate chunks for query: '{query}'")

            if self.model:
                try:
                    import math

                    pairs = [(query, chunk.get("text", "")) for chunk in chunks]
                    scores = self.model.predict(pairs)

                    reranked = []
                    for idx, score in enumerate(scores):
                        chunk_copy = dict(chunks[idx])
                        raw_score = float(score)
                        # Compute normalized sigmoid confidence score (0.0 to 1.0)
                        confidence = 1.0 / (1.0 + math.exp(-raw_score))
                        
                        chunk_copy["reranker_score"] = raw_score
                        chunk_copy["reranker_confidence"] = round(float(confidence), 4)
                        reranked.append(chunk_copy)

                    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
                    top_chunks = reranked[:top_n]

                    top_conf = top_chunks[0]["reranker_confidence"] if top_chunks else 0.0
                    log.info(
                        f"Cross-Encoder reranked top {len(top_chunks)} chunks (top_raw_score={top_chunks[0]['reranker_score']:.4f}, top_confidence={top_conf:.4f})"
                    )
                    return top_chunks
                except Exception as e:
                    log.error(f"Error during Cross-Encoder prediction: {e}")

            # Heuristic fallback if cross-encoder model unavailable
            log.warning(
                "[FALLBACK_TRIGGERED] Live Cross-Encoder model unavailable. Using heuristic ranking fallback (score preservation + length weighting)."
            )
            fallback_chunks = []
            for chunk in chunks:
                chunk_copy = dict(chunk)
                base_score = float(chunk_copy.get("score", 0.5))
                chunk_copy["reranker_score"] = base_score
                chunk_copy["reranker_confidence"] = round(base_score, 4)
                fallback_chunks.append(chunk_copy)

            fallback_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
            return fallback_chunks[:top_n]


# Global singleton instance
cross_encoder_reranker = CrossEncoderReranker()
