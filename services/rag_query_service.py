import os

from dotenv import load_dotenv
from openai import OpenAI

from services.document_repository import search_similar_chunks
from services.embedding_service import embed_text
from services.exceptions import (
    DatabaseAccessError,
    LLMServiceError,
    RetrievalServiceError,
    SystemConfigurationError,
)


DEFAULT_TOP_K = 5
LLM_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_CONTEXT_CHARACTERS = 80000

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def semantic_search(question: str) -> dict:
    question = _normalize_question(question)
    chunks = _retrieve_relevant_chunks(question)

    return {
        "question": question,
        "result_count": len(chunks),
        "sources": chunks,
    }


def answer_question(question: str) -> dict:
    question = _normalize_question(question)
    _ensure_llm_api_key()

    chunks = _retrieve_relevant_chunks(question)

    context = _format_chunks_as_context(chunks) if chunks else ""
    try:
        client = _create_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=_build_messages(question, context),
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
    except Exception as e:
        raise LLMServiceError("Failed to generate an answer using the LLM.") from e
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "answer_mode": "rag" if chunks else "llm_only",
        "source_count": len(chunks),
        "sources": chunks,
    }


def _format_chunks_as_context(chunks: list[dict]) -> str:
    context_parts = []
    total_characters = 0

    for chunk in chunks:
        chunk_text = chunk["text"]
        source_text = (
            f"[Source {chunk['source_id']}]\n"
            f"document_id: {chunk['document_id']}\n"
            f"filename: {chunk['filename']}\n"
            f"chunk_index: {chunk['chunk_index']}\n"
            f"distance: {chunk['distance']:.4f}\n"
            f"text:\n{chunk_text}\n"
        )

        if total_characters + len(source_text) > MAX_CONTEXT_CHARACTERS:
            break

        context_parts.append(source_text)
        total_characters += len(source_text)

    return "\n".join(context_parts)


def _retrieve_relevant_chunks(question: str) -> list[dict]:
    try:
        query_embedding = embed_text(question)
        result = search_similar_chunks(query_embedding, DEFAULT_TOP_K)
    except DatabaseAccessError as e:
        raise RetrievalServiceError("Failed to retrieve relevant chunks from the database.") from e
    return result


def _build_messages(question: str, context: str) -> list[dict]:
    if context:
        system_content = (
            "You are a helpful assistant for a RAG document intelligence system. "
            "Answer the user's question using only the provided context. "
            "If the context does not contain enough information, say that the answer "
            "is not available in the provided documents. "
            "Cite the source numbers you used, such as [Source 1]."
        )
        user_content = f"Question:\n{question}\n\nContext:\n{context}"
    else:
        system_content = (
            "You are a helpful assistant. No relevant document context was found. "
            "Answer using general knowledge, and clearly state that the answer is not "
            "based on uploaded documents."
        )
        user_content = f"Question:\n{question}"

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _normalize_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question cannot be empty.")

    return normalized_question


def _ensure_llm_api_key() -> None:
    if DEEPSEEK_API_KEY is None:
        raise SystemConfigurationError("DEEPSEEK_API_KEY environment variable is not set.")


def _create_llm_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
