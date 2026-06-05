from pathlib import Path

from services.document_parser import parse_document

TEXT_PREVIEW_LENGTH = 500


def process_uploaded_document(document_path: str|Path)->dict:
    path = Path(document_path)
    text=parse_document(path)
    return {
        'status': 'parsed',
        'filename': path.name,
        'text': text[:TEXT_PREVIEW_LENGTH],
    }


