from openai import OpenAI
from services.exceptions import EmbeddingServiceError

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("Texts cannot be empty.")

    for text in texts:
        if not text.strip():
            raise ValueError("Text chunks cannot be empty.")

    try:
        client = OpenAI()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except Exception as e:
        raise EmbeddingServiceError("Failed to generate embeddings.") from e

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(texts):
        raise EmbeddingServiceError("Embedding count does not match input text count.")

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise EmbeddingServiceError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}."
            )

    return embeddings


def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    return embed_texts([text])[0]
