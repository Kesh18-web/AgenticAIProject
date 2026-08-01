from typing import List
from backend.app.core.logging import logger


class EmbeddingEngine:
    """Singleton wrapper around SentenceTransformer for producing 384-dim dense embeddings.

    Uses 'all-MiniLM-L6-v2' — already installed via sentence-transformers.
    Output dimension: 384 — matches the Qdrant collection vector size.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(
                f"[EmbeddingEngine] Loaded SentenceTransformer model '{self.MODEL_NAME}' (dim=384)"
            )
        except Exception as e:
            self._model = None
            logger.error(
                f"[EmbeddingEngine] Failed to load SentenceTransformer model '{self.MODEL_NAME}': {e}"
            )

    def embed(self, text: str) -> List[float]:
        """Embed a single string into a 384-dim float vector."""
        if self._model is None:
            logger.error(
                "[EmbeddingEngine] Model not loaded. Cannot produce embeddings."
            )
            raise RuntimeError(
                "EmbeddingEngine: SentenceTransformer model is not available."
            )
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings into 384-dim float vectors."""
        if self._model is None:
            logger.error(
                "[EmbeddingEngine] Model not loaded. Cannot produce embeddings."
            )
            raise RuntimeError(
                "EmbeddingEngine: SentenceTransformer model is not available."
            )
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


# Global singleton
embedding_engine = EmbeddingEngine()
