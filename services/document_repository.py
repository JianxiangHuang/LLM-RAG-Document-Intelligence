from sqlalchemy.exc import SQLAlchemyError

from models.database import SessionLocal
from models.document_models import Document, DocumentChunk
from services.exceptions import DatabaseAccessError


def save_uploaded_document_info(file_info: dict) -> int:
    if file_info is None:
        raise ValueError("File info cannot be None")

    db = SessionLocal()

    try:
        db_document = Document(**file_info, status="uploaded")
        db.add(db_document)
        db.commit()
        document_id = db_document.id
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to save document info.") from e
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return document_id


def get_document_info(document_id: int) -> dict:
    if document_id is None:
        raise ValueError("Document info cannot be None")

    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        return document.to_dict()
    except SQLAlchemyError as e:
        raise DatabaseAccessError("Failed to get document info.") from e
    except Exception:
        raise
    finally:
        db.close()


def update_document_charcount_and_chunkcount(
    document_id: int,
    char_count: int,
    chunk_count: int,
) -> int:
    if document_id is None or char_count is None or chunk_count is None:
        raise ValueError("Document info cannot be None")

    if char_count < 0 or chunk_count < 0:
        raise ValueError("Document counts cannot be negative")

    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        document.char_count = char_count
        document.chunk_count = chunk_count
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to update document info.") from e
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return document_id


def save_document_chunks(document_id: int, chunks: list[dict]) -> int:
    if chunks is None:
        raise ValueError("Chunks cannot be None")

    if not chunks:
        raise ValueError("Document must have at least one chunk.")

    if document_id is None:
        raise ValueError("Document info cannot be None")

    db = SessionLocal()

    try:
        db_document = db.query(Document).filter(Document.id == document_id).first()
        if db_document is None:
            raise ValueError(f"Document not found: {document_id}")

        chunk_rows = []
        for chunk in chunks:
            db_document_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                start_char=chunk["start_char"],
                end_char=chunk["end_char"],
            )
            chunk_rows.append(db_document_chunk)
        db.add_all(chunk_rows)

        db_document.status = "chunked"
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to save document chunks.") from e
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return document_id


def change_document_status(document_id: int, status: str) -> None:
    if document_id is None:
        raise ValueError("Document info cannot be None")

    if status is None:
        raise ValueError("Document status cannot be None")

    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        document.status = status
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to change document status.") from e
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def save_document_with_chunks(file_info:dict,status:str,chunks:list[dict])->int:

    if chunks is None:
        raise ValueError("Chunks cannot be None")

    if not chunks:
        raise ValueError("Document must have at least one chunk.")

    db = SessionLocal()
    try:
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
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to save document and chunks.") from e
    except Exception:
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
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseAccessError("Failed to update chunk embeddings.") from e

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def search_similar_chunks(embed_question: list[float], top_k: int = 5) -> list[dict]:
    if not embed_question:
        raise ValueError("Query embedding cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if top_k > 20:
        raise ValueError("top_k must be less than or equal to 20.")

    db = SessionLocal()

    try:
        distance = DocumentChunk.embedding.cosine_distance(embed_question).label("distance")

        rows = (
            db.query(DocumentChunk, Document, distance)
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(DocumentChunk.embedding.isnot(None))
            .order_by(distance)
            .limit(top_k)
            .all()
        )

        results = []

        for source_id, (chunk, document, distance_value) in enumerate(rows, start=1):
            results.append(
                {
                    "source_id": source_id,
                    "distance": float(distance_value),
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "filename": document.filename,
                    "chunk_count": document.chunk_count,
                    "created_at": document.created_at.isoformat(),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )

        return results
    except SQLAlchemyError as e:
        raise DatabaseAccessError("Failed to search similar chunks.") from e
    finally:
        db.close()
