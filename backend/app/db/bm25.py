import re
from typing import Any, Dict, List, Optional
from backend.app.core.logging import logger

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25_LIB = True
except ImportError:
    HAS_BM25_LIB = False


def tokenize_text(text: str) -> List[str]:
    """Basic alphanumeric lowercased tokenizer for BM25 keyword matching."""
    text_clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in text_clean.split() if len(token) > 1]


class BM25Indexer:
    """In-memory BM25 Keyword Indexer for precise keyword & technical term retrieval."""

    def __init__(self):
        self.bm25_index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Index a list of text chunks using Rank-BM25 Okapi algorithm."""
        if not HAS_BM25_LIB:
            logger.warning("rank_bm25 library not available. BM25 indexing skipped.")
            return False

        if not chunks:
            logger.warning("No chunks provided to BM25 indexer.")
            return False

        self.chunks = chunks
        self.corpus_tokens = [
            tokenize_text(chunk.get("text", "")) for chunk in chunks
        ]

        try:
            self.bm25_index = BM25Okapi(self.corpus_tokens)
            logger.info(
                f"Successfully indexed {len(chunks)} chunks into BM25 Keyword Store."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return False

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 keyword index and return top_k candidates with BM25 scores."""
        if not self.bm25_index or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        try:
            scores = self.bm25_index.get_scores(query_tokens)

            # Zip chunks with scores and sort descending
            scored_results = []
            for idx, score in enumerate(scores):
                if score > 0.0:  # Only return chunks with non-zero keyword match
                    chunk_copy = dict(self.chunks[idx])
                    chunk_copy["score"] = float(score)
                    chunk_copy["retrieval_method"] = "bm25"
                    scored_results.append(chunk_copy)

            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:top_k]
        except Exception as e:
            logger.error(f"Error during BM25 search for query '{query}': {e}")
            return []


# Global singleton instance
bm25_mgr = BM25Indexer()

