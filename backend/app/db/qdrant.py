from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    HAS_QDRANT_LIB = True
except ImportError:
    HAS_QDRANT_LIB = False


class QdrantVectorStoreManager:
    """Manages Qdrant vector database operations supporting memory mode and remote cloud/host."""

    def __init__(self):
        self.client: Optional[Any] = None
        self._initialize()

    def _initialize(self):
        if not HAS_QDRANT_LIB:
            logger.warning(
                "qdrant-client library not imported yet. Qdrant operations will be deferred."
            )
            return

        try:
            if settings.QDRANT_MODE == "memory":
                self.client = QdrantClient(":memory:")
                logger.info(
                    "Initialized Qdrant Vector Client in IN-MEMORY mode for ultra-fast local dev."
                )
            else:
                self.client = QdrantClient(
                    url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
                )
                logger.info(f"Connected to Qdrant server at {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(
                f"Failed to initialize Qdrant client: {e}. Falling back to :memory: mode."
            )
            self.client = QdrantClient(":memory:")

    def ensure_collection(
        self, collection_name: str, vector_size: int = 384
    ) -> bool:
        """Create Qdrant collection if it does not exist, and ensure payload index on session_id."""
        if not self.client:
            return False

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    ),
                )
                logger.info(
                    f"Created Qdrant collection '{collection_name}' (dim={vector_size})"
                )

            # Always ensure the session_id payload index exists (idempotent)
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="session_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                logger.info(
                    f"Ensured payload index on 'session_id' for collection '{collection_name}'"
                )
            except Exception as idx_err:
                # Index may already exist — not an error
                logger.debug(
                    f"Payload index on 'session_id' already exists or could not be created: {idx_err}"
                )

            return True
        except Exception as e:
            logger.error(
                f"Error creating Qdrant collection '{collection_name}': {e}"
            )
            return False

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> bool:
        """Upsert text chunks and their dense embeddings into Qdrant."""
        if not self.client or not chunks:
            return False

        try:
            vector_size = len(embeddings[0])
            self.ensure_collection(collection_name, vector_size=vector_size)

            points = []
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                doc_id = chunk.get("doc_id", "doc_unknown")
                point_id = abs(hash(f"{doc_id}_{idx}")) & 0x7FFFFFFF

                payload = {
                    "text": chunk.get("text", ""),
                    "doc_id": doc_id,
                    "source_name": chunk.get("source_name", "unknown"),
                    "page_number": chunk.get("page_number", 1),
                    "chunk_index": chunk.get("chunk_index", idx),
                    "session_id": chunk.get("session_id", "default_session"),
                }

                points.append(
                    qmodels.PointStruct(id=point_id, vector=emb, payload=payload)
                )

            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(
                f"Successfully upserted {len(points)} vector chunks into '{collection_name}'"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error upserting vectors to Qdrant collection '{collection_name}': {e}"
            )
            return False

    def search_dense(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search in Qdrant with optional session_id / session_ids filtering."""
        if not self.client:
            return []

        try:
            # Build Qdrant payload filter if session_id / session_ids provided
            query_filter = None
            if HAS_QDRANT_LIB:
                if session_ids:
                    query_filter = qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(
                                key="session_id",
                                match=qmodels.MatchValue(value=s),
                            )
                            for s in session_ids
                        ]
                    )
                elif session_id:
                    query_filter = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="session_id",
                                match=qmodels.MatchValue(value=session_id),
                            )
                        ]
                    )

            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                )
                results = getattr(response, "points", response)
            elif hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                )
            else:
                results = []

            retrieved = []
            for res in results:
                payload = res.payload or {}
                retrieved.append(
                    {
                        "chunk_id": res.id,
                        "score": float(res.score),
                        "text": payload.get("text", ""),
                        "doc_id": payload.get("doc_id", ""),
                        "source_name": payload.get("source_name", ""),
                        "page_number": payload.get("page_number", 1),
                        "session_id": payload.get("session_id", ""),
                        "retrieval_method": "dense",
                    }
                )
            return retrieved
        except Exception as e:
            logger.error(
                f"Error searching Qdrant collection '{collection_name}': {e}"
            )
            return []


# Global singleton instance
qdrant_store = QdrantVectorStoreManager()
