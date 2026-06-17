from services.exceptions import (
    DatabaseAccessError,
    EmbeddingServiceError,
    LLMServiceError,
    RAGServiceError,
    RetrievalServiceError,
    SystemConfigurationError,
)


def get_processing_error_message(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "File not found."

    if isinstance(error, UnicodeDecodeError):
        return "Could not decode document as UTF-8 text."

    if isinstance(error, ValueError):
        return "Invalid value."

    if isinstance(error, SystemConfigurationError):
        return "Server configuration error."

    if isinstance(error, EmbeddingServiceError):
        return "Embedding service failed."

    if isinstance(error, LLMServiceError):
        return "LLM service failed."

    if isinstance(error, RetrievalServiceError):
        return "Vector retrieval failed."

    if isinstance(error, DatabaseAccessError):
        return "Database access failed."

    if isinstance(error, RAGServiceError):
        return "RAG service failed."

    if isinstance(error, RuntimeError):
        return "Internal processing error."

    return "Unexpected processing error."
