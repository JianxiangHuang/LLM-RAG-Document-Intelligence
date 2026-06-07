from pathlib import Path

from services.document_chunker import chunk_text
from services.document_parser import parse_document
from services.document_repository import save_document_with_chunks
TEXT_PREVIEW_LENGTH = 500
CHUNK_PREVIEW_COUNT = 1


def process_uploaded_document(document_path: str|Path,file_info: dict)->dict:
    path = Path(document_path)
    text=parse_document(path)
    chunks=chunk_text(text)
    file_info['char_count']=len(text)
    file_info['chunk_count']=len(chunks)
    document_id=save_document_with_chunks(file_info,'chunked',chunks)
    return {
        'document_id': document_id,
        'status': 'chunked',
        'filename': path.name,
        'char_count': len(text),
        'chunk_count': len(chunks),
        'text_preview': text[:TEXT_PREVIEW_LENGTH],
        'chunks_preview': chunks[:CHUNK_PREVIEW_COUNT],
    }


