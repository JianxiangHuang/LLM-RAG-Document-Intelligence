from fastapi import FastAPI

from app.api.routes_health import router as health_router
from app.api.routes_upload_documents import router as upload_router
from app.api.query_router import router as query_router

app = FastAPI(
    title="RAG Document Intelligence API",
    description="A FastAPI backend for document ingestion, semantic search, and RAG answering.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)
