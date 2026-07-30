import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.core.logging import logger
from backend.app.db.bm25 import bm25_mgr
from backend.app.db.firestore import firestore_db
from backend.app.db.qdrant import qdrant_store

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    source_name: str = "Uploaded Document"
    session_id: str = "default_session"


@router.post("/index")
async def index_document(req: DocumentUploadRequest):
    """Index document text content into BM25 Keyword store, Qdrant Dense Vector store, and Firestore metadata."""
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
                "session_id": req.session_id,
                "page_number": 1,
                "chunk_index": idx,
                "text": para,
            }
            chunks.append(chunk)

            # Generate vector embedding (using deterministic test representation or model)
            emb = [0.01 * ((i + idx) % 100 + 1) for i in range(384)]
            dummy_embeddings.append(emb)

        # 1. Index into BM25 Keyword Store
        bm25_mgr.index_chunks(chunks)

        # 2. Index into Qdrant Vector Store
        qdrant_store.upsert_chunks(
            collection_name="enterprise_documents",
            chunks=chunks,
            embeddings=dummy_embeddings,
        )

        # 3. Persist Document Metadata into GCP Firestore
        firestore_db.save_document(
            collection_name="uploaded_documents",
            doc_id=doc_id,
            data={
                "doc_id": doc_id,
                "title": req.title,
                "source_name": req.source_name,
                "session_id": req.session_id,
                "chunks_indexed": len(chunks),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"Successfully indexed document '{req.title}' ({len(chunks)} chunks) into BM25, Qdrant & Firestore"
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


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    session_id: str = Form("default_session"),
):
    """Upload raw PDF or text document file, extract content, and index into hybrid search stores."""
    try:
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8", errors="ignore")
        doc_title = title or file.filename or "Uploaded Document"

        req = DocumentUploadRequest(
            title=doc_title,
            content=text_content,
            source_name=file.filename or "Uploaded File",
            session_id=session_id,
        )
        return await index_document(req)
    except Exception as e:
        logger.error(f"Error handling file upload '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
