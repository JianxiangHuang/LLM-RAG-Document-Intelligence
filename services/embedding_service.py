from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("Texts cannot be empty.")

    for text in texts:
        if not text:
            raise ValueError("Text chunks cannot be empty.")

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(texts):
        raise ValueError("Embedding count does not match input text count.")

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}."
            )

    return embeddings


def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    return embed_texts([text])[0]
