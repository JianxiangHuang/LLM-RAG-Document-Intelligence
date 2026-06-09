class RAGServiceError(Exception):
    """Base class for RAG service errors."""


class EmbeddingServiceError(RAGServiceError):
    """Raised when embedding generation fails."""


class DatabaseAccessError(Exception):
    """Raised when database access fails."""


class RetrievalServiceError(DatabaseAccessError):
    """Raised when vector retrieval fails."""


class LLMServiceError(RAGServiceError):
    """Raised when answer generation fails."""


class SystemConfigurationError(Exception):
    """Raised when system configuration is invalid."""
