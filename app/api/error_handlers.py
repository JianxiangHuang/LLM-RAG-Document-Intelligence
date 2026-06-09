from fastapi import HTTPException

from services.exceptions import (
    DatabaseAccessError,
    EmbeddingServiceError,
    LLMServiceError,
    RAGServiceError,
    RetrievalServiceError,
    SystemConfigurationError,
)


def map_exception_to_http(error: Exception) -> HTTPException:
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))

    if isinstance(error, SystemConfigurationError):
        return HTTPException(status_code=500, detail="Server configuration error.")

    if isinstance(error, EmbeddingServiceError):
        return HTTPException(status_code=502, detail="Embedding service failed.")

    if isinstance(error, LLMServiceError):
        return HTTPException(status_code=502, detail="LLM service failed.")

    if isinstance(error, RetrievalServiceError):
        return HTTPException(status_code=500, detail="Vector retrieval failed.")

    if isinstance(error, DatabaseAccessError):
        return HTTPException(status_code=500, detail="Database access failed.")

    if isinstance(error, RAGServiceError):
        return HTTPException(status_code=500, detail="RAG service failed.")

    if isinstance(error, RuntimeError):
        return HTTPException(status_code=500, detail="Internal server error.")

    return HTTPException(status_code=500, detail="Internal server error.")