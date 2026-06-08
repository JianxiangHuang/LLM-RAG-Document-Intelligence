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


def update_chunks_embeddings(document_id: int,embeddings: list[list[float]],)-> dict:
    if not embeddings:
        raise ValueError("Embeddings cannot be empty.")

    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        db_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        if not db_chunks:
            raise ValueError(f"No chunks found for document: {document_id}")

        if len(db_chunks) != len(embeddings):
            raise ValueError("Embedding count does not match chunk count.")

        for db_chunk, embedding in zip(db_chunks, embeddings):
            db_chunk.embedding = embedding

        document.status = "embedded"

        db.commit()

        return {
            "document_id": document_id,
            "status": document.status,
            "embedded_chunk_count": len(db_chunks),
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()