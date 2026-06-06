def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    if not text:
        raise ValueError("Text is empty")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    if overlap < 0:
        raise ValueError("Overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    step = chunk_size - overlap

    for chunk_index, start_char in enumerate(range(0, len(text), step)):
        end_char = min(start_char + chunk_size, len(text))
        chunk_content = text[start_char:end_char]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_content,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return chunks
