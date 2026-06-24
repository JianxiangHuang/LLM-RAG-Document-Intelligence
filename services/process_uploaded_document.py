from pathlib import Path

from services import embedding_service
from services.document_chunker import simple_chunking
from services.document_parser import parse_document
from repositories.document_repository import (
    change_document_status,
    get_document_info,
    save_document_chunks,
    save_uploaded_document_info,
    update_chunks_embeddings,
    update_document_charcount_and_chunkcount,
)
from services.process_errors import get_processing_error_message


def process_uploaded_document(document_id: int) -> None:
    try:
        document_info = get_document_info(document_id)
        path = Path(document_info["saved_path"])

        text = parse_document(path)
        chunks = simple_chunking(text)

        update_document_charcount_and_chunkcount(document_id, len(text), len(chunks))
        save_document_chunks(document_id, chunks)

        embedded_chunks = embedding_service.embed_texts(
            [chunk["text"] for chunk in chunks]
        )
        update_chunks_embeddings(document_id, embedded_chunks)
    except Exception as error:
        error_message = get_processing_error_message(error)
        change_document_status(document_id, error_message)


def save_uploaded_document_to_db(file_info: dict) -> int:
    file_info["char_count"] = 0
    file_info["chunk_count"] = 0
    return save_uploaded_document_info(file_info)
