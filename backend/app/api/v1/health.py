from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.db.bm25 import bm25_mgr
from backend.app.db.firestore import firestore_db
from backend.app.db.qdrant import qdrant_store

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def get_health_status():
    """Return backend service status, component initialization, and active environment."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "qdrant_mode": settings.QDRANT_MODE,
        "firestore_is_mock": firestore_db.is_mock,
        "bm25_chunks_count": len(bm25_mgr.chunks),
        "llm_providers": {
            "openai": bool(settings.OPENAI_API_KEY),
            "anthropic": bool(settings.ANTHROPIC_API_KEY),
            "gemini": bool(settings.GEMINI_API_KEY),
            "groq": bool(settings.GROQ_API_KEY),
        },
    }
