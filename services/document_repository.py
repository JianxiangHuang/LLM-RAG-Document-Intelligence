from models.database import SessionLocal
from models.document_models import Document, DocumentChunk

def save_document_with_chunks(file_info:dict,status:str,chunks:list[dict])->int:

    if chunks is None:
        raise ValueError("Chunks cannot be None")

    if not chunks:
        raise ValueError("Document must have at least one chunk.")

    try:
        db = SessionLocal()
        db_document=Document(**file_info,status=status)
        db.add(db_document)
        db.flush()
        document_id = db_document.id

        chunk_rows = []
        for chunk in chunks:
            db_document_chunk=DocumentChunk(document_id=document_id,
                                            chunk_index=chunk['chunk_index'],
                                            text=chunk['text'],
                                            start_char=chunk['start_char'],
                                            end_char=chunk['end_char'],)
            chunk_rows.append(db_document_chunk)
        db.add_all(chunk_rows)
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()

    return document_id



