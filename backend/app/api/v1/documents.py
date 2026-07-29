import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.core.logging import logger
from backend.app.db.bm25 import bm25_mgr
from backend.app.db.qdrant import qdrant_store

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    source_name: str = "Uploaded Document"


@router.post("/index")
async def index_document(req: DocumentUploadRequest):
    """Index document text content into BM25 Keyword store and Qdrant Dense Vector store."""
    try:
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"

        # Basic paragraph chunking heuristic
        raw_paragraphs = [p.strip() for p in req.content.split("\n\n") if p.strip()]

        chunks: List[Dict[str, Any]] = []
        dummy_embeddings: List[List[float]] = []

        for idx, para in enumerate(raw_paragraphs):
            chunk = {
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "source_name": req.source_name,
                "page_number": 1,
                "chunk_index": idx,
                "text": para,
            }
            chunks.append(chunk)

            # Generate vector embedding (using deterministic test representation or model)
            emb = [0.01 * ((i + idx) % 100 + 1) for i in range(384)]
            dummy_embeddings.append(emb)

        # 1. Index into BM25
        bm25_mgr.index_chunks(chunks)

        # 2. Index into Qdrant
        qdrant_store.upsert_chunks(
            collection_name="enterprise_documents",
            chunks=chunks,
            embeddings=dummy_embeddings,
        )

        logger.info(
            f"Successfully indexed document '{req.title}' ({len(chunks)} chunks) into BM25 & Qdrant"
        )
        return {
            "status": "success",
            "doc_id": doc_id,
            "title": req.title,
            "chunks_indexed": len(chunks),
        }
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
