from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.analyze import router as analyze_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.health import router as health_router
from backend.app.core.config import settings
from backend.app.core.logging import logger

app = FastAPI(
    title="Enterprise AI Analyst API",
    description="Production Modular AI Agent Runtime & Hybrid RAG Engine powered by LangGraph, Qdrant, and Firestore",
    version="1.0.0",
)

# Enable CORS for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info(
        f"Started Enterprise AI Analyst FastAPI Server on port {settings.PORT} (env={settings.ENVIRONMENT})"
    )
